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
REFERENCE = {
    "oof_bacc": 0.7514,
    "lodo_bacc": 0.7441,
    "lodo_sensitivity": 0.7354,
    "lodo_specificity": 0.7527,
    "b1_accuracy": 0.5000,
    "b2_accuracy": 0.6629,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize locked H3B gates against the C2 prediction baseline."
    )
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--candidate", action="append", nargs=3, required=True,
                        metavar=("NAME", "OOF_CSV", "LODO_CSV"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
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
    if tuple(sorted(frame["source_dataset"].unique())) != tuple(sorted(EXPECTED_SOURCES)):
        raise ValueError(f"Unexpected sources in {path}")
    if tuple(sorted(frame["task_l6_label"].unique())) != tuple(sorted(EXPECTED_SUBTYPES)):
        raise ValueError(f"Unexpected subtypes in {path}")
    if not np.isfinite(frame["prob_high"].to_numpy()).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    expected_pred = (frame["prob_high"].to_numpy() >= 0.5).astype(int)
    if not np.array_equal(expected_pred, frame["pred_idx"].to_numpy()):
        raise ValueError(f"Predictions in {path} are not fixed-threshold 0.5")
    return frame.sort_values("case_id").reset_index(drop=True)


def align(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not reference["case_id"].equals(candidate["case_id"]):
        raise ValueError("Reference and candidate case IDs do not align")
    for column in ("label_idx", "source_dataset", "task_l6_label"):
        if not reference[column].equals(candidate[column]):
            raise ValueError(f"Reference and candidate differ in {column}")
    return reference, candidate


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


def grouped_metrics(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    return {
        str(name): metrics(group.reset_index(drop=True))
        for name, group in frame.groupby(column, sort=True)
    }


def subtype_accuracy(frame: pd.DataFrame) -> dict[str, float]:
    return {
        str(name): float((group["pred_idx"] == group["label_idx"]).mean())
        for name, group in frame.groupby("task_l6_label", sort=True)
    }


def paired_stratified_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
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

    def bacc(prediction: np.ndarray) -> np.ndarray:
        sampled_pred = prediction[draws]
        sensitivity = ((sampled_pred == 1) & (sampled_y == 1)).sum(axis=1) / (
            sampled_y == 1
        ).sum(axis=1)
        specificity = ((sampled_pred == 0) & (sampled_y == 0)).sum(axis=1) / (
            sampled_y == 0
        ).sum(axis=1)
        return (sensitivity + specificity) / 2.0

    delta = bacc(cand_pred) - bacc(ref_pred)
    point = (
        float(metrics(candidate)["balanced_accuracy"])
        - float(metrics(reference)["balanced_accuracy"])
    )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "point_delta_bacc": point,
        "mean_delta_bacc": float(delta.mean()),
        "ci95_lower": float(np.quantile(delta, 0.025)),
        "ci95_upper": float(np.quantile(delta, 0.975)),
        "probability_delta_gt_zero": float((delta > 0).mean()),
    }


def evaluate_candidate(
    name: str,
    c2_oof: pd.DataFrame,
    c2_lodo: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    candidate_lodo: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    c2_oof, candidate_oof = align(c2_oof, candidate_oof)
    c2_lodo, candidate_lodo = align(c2_lodo, candidate_lodo)
    oof = metrics(candidate_oof)
    lodo = metrics(candidate_lodo)
    c2_lodo_sources = grouped_metrics(c2_lodo, "source_dataset")
    candidate_sources = grouped_metrics(candidate_lodo, "source_dataset")
    source_deltas = {
        source: float(candidate_sources[source]["balanced_accuracy"])
        - float(c2_lodo_sources[source]["balanced_accuracy"])
        for source in EXPECTED_SOURCES
    }
    c2_subtypes = subtype_accuracy(c2_lodo)
    candidate_subtypes = subtype_accuracy(candidate_lodo)
    b1_delta = candidate_subtypes["B1"] - c2_subtypes["B1"]
    b2_delta = candidate_subtypes["B2"] - c2_subtypes["B2"]
    mean_boundary_delta = (
        (candidate_subtypes["B1"] + candidate_subtypes["B2"])
        - (c2_subtypes["B1"] + c2_subtypes["B2"])
    ) / 2.0
    bootstrap = paired_stratified_bootstrap(
        c2_lodo, candidate_lodo, replicates=replicates, seed=seed
    )
    gates = [
        ("OOF BAcc >= 0.7664", float(oof["balanced_accuracy"]) >= 0.7664),
        ("LODO BAcc >= 0.7641", float(lodo["balanced_accuracy"]) >= 0.7641),
        ("LODO sensitivity >= 0.7354", float(lodo["sensitivity"]) >= 0.7354),
        ("LODO specificity >= 0.7527", float(lodo["specificity"]) >= 0.7527),
        ("At least two held-out sources improve", sum(v > 0 for v in source_deltas.values()) >= 2),
        ("No held-out source declines by more than 0.02", min(source_deltas.values()) >= -0.02),
        ("B1 accuracy >= 0.5000", candidate_subtypes["B1"] >= 0.5000),
        ("B2 accuracy >= 0.6629", candidate_subtypes["B2"] >= 0.6629),
        (
            "Mean B1/B2 improves >= 0.03 and neither declines",
            mean_boundary_delta >= 0.03 and b1_delta >= 0 and b2_delta >= 0,
        ),
        ("Bootstrap LODO delta BAcc CI lower > 0", float(bootstrap["ci95_lower"]) > 0),
        ("Confirmation seed directionally positive", False),
        ("Threshold 0.5 and coverage 100%", len(candidate_lodo) == 591),
    ]
    gate_rows = [
        {
            "gate_number": index,
            "gate": label,
            "passed": bool(passed),
            "status": (
                "not_evaluated_primary_failed"
                if index == 11
                else ("pass" if passed else "fail")
            ),
        }
        for index, (label, passed) in enumerate(gates, start=1)
    ]
    return {
        "candidate": name,
        "decision": "GO" if all(row["passed"] for row in gate_rows) else "NO_GO",
        "oof": oof,
        "source_lodo": lodo,
        "source_lodo_by_source": candidate_sources,
        "c2_source_lodo_by_source": c2_lodo_sources,
        "source_lodo_bacc_deltas": source_deltas,
        "source_lodo_subtype_accuracy": candidate_subtypes,
        "c2_source_lodo_subtype_accuracy": c2_subtypes,
        "b1_delta": float(b1_delta),
        "b2_delta": float(b2_delta),
        "mean_b1_b2_delta": float(mean_boundary_delta),
        "paired_source_risk_stratified_bootstrap": bootstrap,
        "gates": gate_rows,
    }


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "# H3B locked gate summary",
        "",
        "All metrics use threshold 0.5 and 100% coverage. Source-LODO is an internal batch-robustness proxy, not multicenter external validation.",
        "",
        "| Candidate | Decision | OOF BAcc | LODO BAcc | LODO AUC | LODO Sens | LODO Spec | B1 | B2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in results["candidates"]:
        lodo = candidate["source_lodo"]
        subtypes = candidate["source_lodo_subtype_accuracy"]
        lines.append(
            f"| {candidate['candidate']} | {candidate['decision']} | "
            f"{candidate['oof']['balanced_accuracy']:.4f} | "
            f"{lodo['balanced_accuracy']:.4f} | {lodo['auc']:.4f} | "
            f"{lodo['sensitivity']:.4f} | {lodo['specificity']:.4f} | "
            f"{subtypes['B1']:.4f} | {subtypes['B2']:.4f} |"
        )
    for candidate in results["candidates"]:
        lines.extend(
            [
                "",
                f"## {candidate['candidate']}",
                "",
                "| # | Gate | Status |",
                "| ---: | --- | --- |",
            ]
        )
        for row in candidate["gates"]:
            lines.append(f"| {row['gate_number']} | {row['gate']} | {row['status']} |")
        bootstrap = candidate["paired_source_risk_stratified_bootstrap"]
        lines.extend(
            [
                "",
                f"Paired LODO delta BAcc: {bootstrap['point_delta_bacc']:+.4f}, "
                f"95% CI [{bootstrap['ci95_lower']:+.4f}, {bootstrap['ci95_upper']:+.4f}].",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1000 bootstrap replicates")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c2_oof = load_predictions(Path(args.c2_oof))
    c2_lodo = load_predictions(Path(args.c2_lodo))
    candidates = []
    for offset, (name, oof_path, lodo_path) in enumerate(args.candidate):
        candidates.append(
            evaluate_candidate(
                name,
                c2_oof,
                c2_lodo,
                load_predictions(Path(oof_path)),
                load_predictions(Path(lodo_path)),
                replicates=args.bootstrap_replicates,
                seed=args.seed + offset,
            )
        )
    results = {
        "reference_locked_metrics": REFERENCE,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "candidates": candidates,
    }
    (output_dir / "h3b_gate_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    gate_rows = []
    for candidate in candidates:
        for row in candidate["gates"]:
            gate_rows.append({"candidate": candidate["candidate"], **row})
    pd.DataFrame(gate_rows).to_csv(output_dir / "h3b_gate_table.csv", index=False)
    (output_dir / "h3b_gate_summary.md").write_text(markdown(results), encoding="utf-8")
    print(
        json.dumps(
            {
                item["candidate"]: {
                    "decision": item["decision"],
                    "oof_bacc": item["oof"]["balanced_accuracy"],
                    "lodo_bacc": item["source_lodo"]["balanced_accuracy"],
                    "bootstrap": item["paired_source_risk_stratified_bootstrap"],
                }
                for item in candidates
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
