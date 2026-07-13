from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXPECTED_SOURCES = ("batch1", "batch2", "third_batch")
EXPECTED_SUBTYPES = ("A", "AB", "B1", "B2", "B3", "TC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered H4 quality-randomization gates."
    )
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--h3-oof", required=True)
    parser.add_argument("--h3-lodo", required=True)
    parser.add_argument("--h4-oof", required=True)
    parser.add_argument("--h4-lodo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str})
    required = {
        "case_id",
        "label_idx",
        "source_dataset",
        "task_l6_label",
        "prob_high",
        "pred_idx",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["label_idx"] = frame["label_idx"].astype(int)
    frame["pred_idx"] = frame["pred_idx"].astype(int)
    frame["prob_high"] = frame["prob_high"].astype(float)
    if len(frame) != 591 or frame["case_id"].nunique() != 591:
        raise ValueError(f"Expected 591 unique cases in {path}, found {len(frame)}")
    if set(frame["source_dataset"].unique()) != set(EXPECTED_SOURCES):
        raise ValueError(f"Unexpected source set in {path}")
    if set(frame["task_l6_label"].unique()) != set(EXPECTED_SUBTYPES):
        raise ValueError(f"Unexpected subtype set in {path}")
    if not np.isfinite(frame["prob_high"].to_numpy()).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    fixed_predictions = (frame["prob_high"].to_numpy() >= 0.5).astype(int)
    if not np.array_equal(fixed_predictions, frame["pred_idx"].to_numpy()):
        raise ValueError(f"Predictions in {path} are not thresholded at 0.5")
    return frame.sort_values("case_id").reset_index(drop=True)


def align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    reference = frames[0]
    for candidate in frames[1:]:
        if not reference["case_id"].equals(candidate["case_id"]):
            raise ValueError("Prediction case IDs do not align")
        for column in ("label_idx", "source_dataset", "task_l6_label"):
            if not reference[column].equals(candidate[column]):
                raise ValueError(f"Predictions differ in {column}")
    return frames


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["label_idx"].to_numpy(dtype=int)
    pred = frame["pred_idx"].to_numpy(dtype=int)
    prob = frame["prob_high"].to_numpy(dtype=float)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "n": int(len(frame)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "auc": float(roc_auc_score(y, prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def source_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    return {
        str(source): metrics(group.reset_index(drop=True))
        for source, group in frame.groupby("source_dataset", sort=True)
    }


def subtype_accuracy(frame: pd.DataFrame) -> dict[str, float]:
    return {
        str(subtype): float((group["pred_idx"] == group["label_idx"]).mean())
        for subtype, group in frame.groupby("task_l6_label", sort=True)
    }


def paired_transitions(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    reference, candidate = align(reference, candidate)
    ref_correct = reference["pred_idx"].eq(reference["label_idx"])
    cand_correct = candidate["pred_idx"].eq(candidate["label_idx"])
    state = np.select(
        [
            ref_correct & cand_correct,
            ~ref_correct & cand_correct,
            ref_correct & ~cand_correct,
        ],
        ["stable_correct", "candidate_rescue", "candidate_harm"],
        default="persistent_error",
    )
    audit = candidate[["source_dataset", "task_l6_label"]].copy()
    audit["transition"] = state

    def counts(frame: pd.DataFrame) -> dict[str, int]:
        observed = frame["transition"].value_counts()
        return {
            name: int(observed.get(name, 0))
            for name in (
                "stable_correct",
                "candidate_rescue",
                "candidate_harm",
                "persistent_error",
            )
        }

    return {
        "overall": counts(audit),
        "by_source": {
            str(name): counts(group)
            for name, group in audit.groupby("source_dataset", sort=True)
        },
        "by_subtype": {
            str(name): counts(group)
            for name, group in audit.groupby("task_l6_label", sort=True)
        },
    }


def paired_source_risk_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    reference, candidate = align(reference, candidate)
    rng = np.random.default_rng(seed)
    y = reference["label_idx"].to_numpy(dtype=np.int8)
    ref_pred = reference["pred_idx"].to_numpy(dtype=np.int8)
    cand_pred = candidate["pred_idx"].to_numpy(dtype=np.int8)
    strata = reference.groupby(["source_dataset", "label_idx"], sort=True).indices
    draws = np.concatenate(
        [
            rng.choice(np.asarray(indices), size=(replicates, len(indices)), replace=True)
            for indices in strata.values()
        ],
        axis=1,
    )
    sampled_y = y[draws]

    def sampled_metrics(prediction: np.ndarray) -> dict[str, np.ndarray]:
        sampled_prediction = prediction[draws]
        sensitivity = ((sampled_prediction == 1) & (sampled_y == 1)).sum(axis=1) / (
            sampled_y == 1
        ).sum(axis=1)
        specificity = ((sampled_prediction == 0) & (sampled_y == 0)).sum(axis=1) / (
            sampled_y == 0
        ).sum(axis=1)
        return {
            "balanced_accuracy": (sensitivity + specificity) / 2.0,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    reference_draws = sampled_metrics(ref_pred)
    candidate_draws = sampled_metrics(cand_pred)
    reference_point = metrics(reference)
    candidate_point = metrics(candidate)
    result: dict[str, Any] = {"replicates": replicates, "seed": seed}
    for name in ("balanced_accuracy", "sensitivity", "specificity"):
        delta = candidate_draws[name] - reference_draws[name]
        result[name] = {
            "point_delta": float(candidate_point[name]) - float(reference_point[name]),
            "mean_delta": float(delta.mean()),
            "ci95_lower": float(np.quantile(delta, 0.025)),
            "ci95_upper": float(np.quantile(delta, 0.975)),
            "probability_delta_gt_zero": float((delta > 0).mean()),
        }
    return result


def gate_row(number: int, label: str, passed: bool, status: str | None = None) -> dict[str, Any]:
    return {
        "gate_number": number,
        "gate": label,
        "passed": bool(passed),
        "status": status or ("pass" if passed else "fail"),
    }


def markdown(result: dict[str, Any]) -> str:
    h4_oof = result["metrics"]["h4_quality"]["fivefold"]
    h4_lodo = result["metrics"]["h4_quality"]["source_lodo"]
    lines = [
        "# H4 quality-domain randomization locked gate summary",
        "",
        "All main metrics use clean-image inference, threshold 0.5, and 100% coverage. Source-LODO is an internal batch-transfer proxy, not independent multicenter external validation.",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Model | Five-fold BAcc | Five-fold AUC | Five-fold Sens | Five-fold Spec | LODO BAcc | LODO AUC | LODO Sens | LODO Spec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("c2", "h3_pe", "h4_quality"):
        fivefold = result["metrics"][name]["fivefold"]
        lodo = result["metrics"][name]["source_lodo"]
        lines.append(
            f"| {name} | {fivefold['balanced_accuracy']:.4f} | {fivefold['auc']:.4f} | "
            f"{fivefold['sensitivity']:.4f} | {fivefold['specificity']:.4f} | "
            f"{lodo['balanced_accuracy']:.4f} | {lodo['auc']:.4f} | "
            f"{lodo['sensitivity']:.4f} | {lodo['specificity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gates",
            "",
            "| # | Gate | Status |",
            "| ---: | --- | --- |",
        ]
    )
    for row in result["gates"]:
        lines.append(f"| {row['gate_number']} | {row['gate']} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Held-source and boundary results",
            "",
            "| Source | C2 BAcc | H3 PE BAcc | H4 BAcc | H4 - C2 | H4 - H3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source in EXPECTED_SOURCES:
        source_row = result["by_source"][source]
        lines.append(
            f"| {source} | {source_row['c2']['balanced_accuracy']:.4f} | "
            f"{source_row['h3_pe']['balanced_accuracy']:.4f} | "
            f"{source_row['h4_quality']['balanced_accuracy']:.4f} | "
            f"{source_row['h4_minus_c2_bacc']:+.4f} | "
            f"{source_row['h4_minus_h3_bacc']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "| Subtype | C2 accuracy | H3 PE accuracy | H4 accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for subtype in EXPECTED_SUBTYPES:
        lines.append(
            f"| {subtype} | {result['by_subtype']['c2'][subtype]:.4f} | "
            f"{result['by_subtype']['h3_pe'][subtype]:.4f} | "
            f"{result['by_subtype']['h4_quality'][subtype]:.4f} |"
        )
    for reference_name in ("c2", "h3_pe"):
        bootstrap = result["paired_bootstrap"][reference_name]["balanced_accuracy"]
        lines.append(
            "\nH4 minus " + reference_name + " LODO BAcc: "
            f"{bootstrap['point_delta']:+.4f}, 95% CI "
            f"[{bootstrap['ci95_lower']:+.4f}, {bootstrap['ci95_upper']:+.4f}]."
        )
    lines.extend(
        [
            "",
            "H4 clean five-fold operating point: "
            f"BAcc {h4_oof['balanced_accuracy']:.4f}, sensitivity {h4_oof['sensitivity']:.4f}, specificity {h4_oof['specificity']:.4f}.",
            "H4 clean source-LODO operating point: "
            f"BAcc {h4_lodo['balanced_accuracy']:.4f}, sensitivity {h4_lodo['sensitivity']:.4f}, specificity {h4_lodo['specificity']:.4f}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1000 bootstrap replicates")

    c2_oof, h3_oof, h4_oof = align(
        load_predictions(Path(args.c2_oof)),
        load_predictions(Path(args.h3_oof)),
        load_predictions(Path(args.h4_oof)),
    )
    c2_lodo, h3_lodo, h4_lodo = align(
        load_predictions(Path(args.c2_lodo)),
        load_predictions(Path(args.h3_lodo)),
        load_predictions(Path(args.h4_lodo)),
    )

    model_frames = {
        "c2": (c2_oof, c2_lodo),
        "h3_pe": (h3_oof, h3_lodo),
        "h4_quality": (h4_oof, h4_lodo),
    }
    result: dict[str, Any] = {
        "candidate": "pe_spatial_quality_dr_consistency_v1",
        "threshold": 0.5,
        "coverage": 1.0,
        "metrics": {
            name: {"fivefold": metrics(frames[0]), "source_lodo": metrics(frames[1])}
            for name, frames in model_frames.items()
        },
    }
    by_source = {name: source_metrics(frames[1]) for name, frames in model_frames.items()}
    result["by_source"] = {
        source: {
            "c2": by_source["c2"][source],
            "h3_pe": by_source["h3_pe"][source],
            "h4_quality": by_source["h4_quality"][source],
            "h4_minus_c2_bacc": float(by_source["h4_quality"][source]["balanced_accuracy"])
            - float(by_source["c2"][source]["balanced_accuracy"]),
            "h4_minus_h3_bacc": float(by_source["h4_quality"][source]["balanced_accuracy"])
            - float(by_source["h3_pe"][source]["balanced_accuracy"]),
        }
        for source in EXPECTED_SOURCES
    }
    result["by_subtype"] = {
        name: subtype_accuracy(frames[1]) for name, frames in model_frames.items()
    }
    result["paired_bootstrap"] = {
        "c2": paired_source_risk_bootstrap(
            c2_lodo, h4_lodo, args.bootstrap_replicates, args.seed
        ),
        "h3_pe": paired_source_risk_bootstrap(
            h3_lodo, h4_lodo, args.bootstrap_replicates, args.seed + 1
        ),
    }
    result["error_transitions"] = {
        "versus_c2": paired_transitions(c2_lodo, h4_lodo),
        "versus_h3_pe": paired_transitions(h3_lodo, h4_lodo),
    }

    h4_oof_metrics = result["metrics"]["h4_quality"]["fivefold"]
    h4_lodo_metrics = result["metrics"]["h4_quality"]["source_lodo"]
    h3_lodo_metrics = result["metrics"]["h3_pe"]["source_lodo"]
    h4_subtypes = result["by_subtype"]["h4_quality"]
    c2_source_deltas = [
        result["by_source"][source]["h4_minus_c2_bacc"] for source in EXPECTED_SOURCES
    ]
    primary_gates = [
        gate_row(1, "Five-fold OOF BAcc >= 0.7903", h4_oof_metrics["balanced_accuracy"] >= 0.7903),
        gate_row(
            2,
            "Five-fold sensitivity >= 0.7772 and specificity >= 0.7635",
            h4_oof_metrics["sensitivity"] >= 0.7772
            and h4_oof_metrics["specificity"] >= 0.7635,
        ),
        gate_row(3, "Source-LODO BAcc >= 0.7641", h4_lodo_metrics["balanced_accuracy"] >= 0.7641),
        gate_row(4, "Source-LODO sensitivity >= 0.7354", h4_lodo_metrics["sensitivity"] >= 0.7354),
        gate_row(5, "Source-LODO specificity >= 0.7527", h4_lodo_metrics["specificity"] >= 0.7527),
        gate_row(6, "Source-LODO B1 accuracy >= 0.6000", h4_subtypes["B1"] >= 0.6000),
        gate_row(7, "Source-LODO B2 accuracy >= 0.6629", h4_subtypes["B2"] >= 0.6629),
        gate_row(8, "At least two held-out sources improve versus C2", sum(delta > 0 for delta in c2_source_deltas) >= 2),
        gate_row(9, "No held-out source declines by more than 0.02 versus C2", min(c2_source_deltas) >= -0.02),
        gate_row(
            10,
            "Source-LODO BAcc and sensitivity both exceed H3 PE",
            h4_lodo_metrics["balanced_accuracy"] > h3_lodo_metrics["balanced_accuracy"]
            and h4_lodo_metrics["sensitivity"] > h3_lodo_metrics["sensitivity"],
        ),
        gate_row(
            11,
            "Paired LODO BAcc delta versus C2 has CI95 lower bound > 0",
            result["paired_bootstrap"]["c2"]["balanced_accuracy"]["ci95_lower"] > 0,
        ),
    ]
    primary_passed = all(row["passed"] for row in primary_gates)
    confirmation_gate = gate_row(
        12,
        "Confirmation seed remains directionally positive",
        False,
        "required_not_run" if primary_passed else "not_evaluated_primary_failed",
    )
    coverage_gate = gate_row(
        13,
        "Threshold 0.5 and coverage 100%",
        len(h4_oof) == 591 and len(h4_lodo) == 591,
    )
    result["gates"] = primary_gates + [confirmation_gate, coverage_gate]
    result["decision"] = (
        "PRIMARY_PASS_CONFIRMATION_REQUIRED" if primary_passed else "NO_GO"
    )
    result["bootstrap_replicates"] = args.bootstrap_replicates
    result["seed"] = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "h4_gate_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(result["gates"]).to_csv(output_dir / "h4_gate_table.csv", index=False)
    (output_dir / "h4_gate_summary.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
