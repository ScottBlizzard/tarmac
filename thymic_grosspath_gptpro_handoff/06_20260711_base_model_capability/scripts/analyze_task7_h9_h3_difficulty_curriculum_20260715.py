from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


EXPERIMENT = "H9_H3_DIFFICULTY_BALANCED_CURRICULUM_20260715"
STATIC = "SOURCE_RISK_SUBTYPE_TEMPERED"
CURRICULUM = "SOURCE_RISK_SUBTYPE_CURRICULUM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze locked H9 source-LODO gates.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--control-predictions", required=True)
    parser.add_argument("--c1-predictions", required=True)
    parser.add_argument("--c2-predictions", required=True)
    parser.add_argument("--h3-predictions", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def canonical_source(value: Any) -> str:
    text = str(value)
    if text.startswith("third_batch"):
        return "third_batch"
    if text in {"batch1", "batch2"}:
        return text
    raise ValueError(f"Unexpected source: {value!r}")


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {"case_id", "label_idx", "prob_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prediction columns in {path}")
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError(f"Expected 591 unique cases in {path}")
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["prob_high"] = pd.to_numeric(frame["prob_high"], errors="raise").astype(float)
    if not np.isfinite(frame["prob_high"]).all():
        raise ValueError(f"Nonfinite probabilities in {path}")
    if "source_dataset" in frame:
        frame["source_dataset"] = frame["source_dataset"].map(canonical_source)
    return frame


def align_probability(base: pd.DataFrame, path: Path, name: str) -> np.ndarray:
    reference = load_predictions(path)[["case_id", "label_idx", "prob_high"]].rename(
        columns={"label_idx": f"{name}_label", "prob_high": f"{name}_prob"}
    )
    aligned = base[["case_id", "label_idx"]].merge(
        reference, on="case_id", how="left", validate="one_to_one"
    )
    if aligned[f"{name}_prob"].isna().any():
        raise ValueError(f"Reference {name} is missing cases")
    if not np.array_equal(
        aligned["label_idx"].to_numpy(int), aligned[f"{name}_label"].to_numpy(int)
    ):
        raise ValueError(f"Reference {name} label mismatch")
    return aligned[f"{name}_prob"].to_numpy(float)


def metric_record(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "n": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "auc": float(roc_auc_score(labels, probability)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bacc(labels: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    prediction = np.asarray(probability) >= 0.5
    low = labels == 0
    high = labels == 1
    return float((np.mean(~prediction[low]) + np.mean(prediction[high])) / 2)


def comparison_seed(base_seed: int, name: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    return int((base_seed + suffix) % (2**32 - 1))


def paired_bootstrap(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    reference: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = frame["label_idx"].to_numpy(int)
    work = frame.reset_index(drop=True)
    strata = [
        group.index.to_numpy(int)
        for _, group in work.groupby(["source_dataset", "label_idx"], sort=True)
    ]
    if len(strata) != 6 or any(len(group) == 0 for group in strata):
        raise ValueError("H9 bootstrap requires six source-risk strata")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in strata]
        )
        deltas[index] = bacc(labels[sampled], candidate[sampled]) - bacc(
            labels[sampled], reference[sampled]
        )
    return {
        "point_delta_bacc": float(bacc(labels, candidate) - bacc(labels, reference)),
        "bootstrap_mean_delta_bacc": float(deltas.mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def aggregate_tables(
    frame: pd.DataFrame, methods: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = frame["label_idx"].to_numpy(int)
    overall_rows = []
    source_rows = []
    subtype_rows = []
    for method, probability in methods.items():
        overall_rows.append({"method": method, **metric_record(labels, probability)})
        for source, group in frame.groupby("source_dataset", sort=True):
            rows = group.index.to_numpy(int)
            source_rows.append(
                {
                    "method": method,
                    "source_dataset": source,
                    **metric_record(labels[rows], probability[rows]),
                }
            )
        prediction = probability >= 0.5
        for subtype, group in frame.groupby("task_l6_label", sort=False):
            rows = group.index.to_numpy(int)
            correct = prediction[rows] == labels[rows]
            subtype_rows.append(
                {
                    "method": method,
                    "subtype": subtype,
                    "n": int(len(rows)),
                    "correct_n": int(correct.sum()),
                    "accuracy": float(correct.mean()),
                    "mean_prob_high": float(probability[rows].mean()),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(source_rows), pd.DataFrame(subtype_rows)


def diagnostic_difficulty(
    frame: pd.DataFrame,
    c1: np.ndarray,
    c2: np.ndarray,
    h3: np.ndarray,
    methods: dict[str, np.ndarray],
) -> pd.DataFrame:
    labels = frame["label_idx"].to_numpy(int)
    correct_count = sum((probability >= 0.5).astype(int) == labels for probability in (c1, c2, h3))
    names = np.full(len(frame), "persistent_hard", dtype="<U15")
    names[correct_count == 1] = "salvage"
    names[correct_count == 2] = "medium"
    names[correct_count == 3] = "easy"
    frame["diagnostic_difficulty"] = names
    rows = []
    for difficulty, group in frame.groupby("diagnostic_difficulty", sort=True):
        indices = group.index.to_numpy(int)
        for method, probability in methods.items():
            correct = (probability[indices] >= 0.5).astype(int) == labels[indices]
            rows.append(
                {
                    "difficulty": difficulty,
                    "method": method,
                    "n": int(len(indices)),
                    "correct_n": int(correct.sum()),
                    "accuracy": float(correct.mean()),
                }
            )
    return pd.DataFrame(rows)


def rescue_record(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    reference: np.ndarray,
    subset_name: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    labels = frame["label_idx"].to_numpy(int)
    candidate_correct = (candidate >= 0.5).astype(int) == labels
    reference_correct = (reference >= 0.5).astype(int) == labels
    rescue = (~reference_correct) & candidate_correct & mask
    harm = reference_correct & (~candidate_correct) & mask
    return {
        "subset": subset_name,
        "n": int(mask.sum()),
        "rescue_n": int(rescue.sum()),
        "harm_n": int(harm.sum()),
        "net_rescue": int(rescue.sum() - harm.sum()),
    }


def lookup_subtype(table: pd.DataFrame, method: str, subtype: str) -> int:
    row = table[(table["method"] == method) & (table["subtype"] == subtype)]
    if len(row) != 1:
        raise ValueError(f"Missing subtype row: {method} {subtype}")
    return int(row.iloc[0]["correct_n"])


def markdown_report(
    decision: str,
    gates: list[dict[str, Any]],
    overall: pd.DataFrame,
    subtype: pd.DataFrame,
    source: pd.DataFrame,
    bootstrap: pd.DataFrame,
    rescue: pd.DataFrame,
    control_error: float,
) -> str:
    curriculum = overall[overall["method"] == CURRICULUM].iloc[0]
    lines = [
        "# H9 H3 Difficulty-Balanced Curriculum Results",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "## Integrity",
        "",
        f"- Control maximum probability reproduction error: {control_error:.3e}",
        "- Coverage: 591/591 at threshold 0.5",
        "- Trainable parameters per candidate: 151,107",
        "",
        "## Curriculum overall",
        "",
        f"- BAcc: {curriculum['balanced_accuracy']:.4f}",
        f"- accuracy: {curriculum['accuracy']:.4f}",
        f"- AUC: {curriculum['auc']:.4f}",
        f"- sensitivity: {curriculum['sensitivity']:.4f} ({int(curriculum['tp'])}/223)",
        f"- specificity: {curriculum['specificity']:.4f} ({int(curriculum['tn'])}/368)",
        "",
        "## Overall comparison",
        "",
        "| Method | BAcc | Accuracy | AUC | Sensitivity | Specificity | TP | TN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['method']} | {row['balanced_accuracy']:.4f} | {row['accuracy']:.4f} | "
            f"{row['auc']:.4f} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{int(row['tp'])} | {int(row['tn'])} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Result | Requirement |",
            "| --- | --- | --- |",
        ]
    )
    for gate in gates:
        lines.append(
            f"| {gate['gate']} | {'PASS' if gate['passed'] else 'FAIL'} | {gate['requirement']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary subtypes",
            "",
            "| Method | B1 | B2 |",
            "| --- | ---: | ---: |",
        ]
    )
    for method in overall["method"]:
        lines.append(
            f"| {method} | {lookup_subtype(subtype, method, 'B1')}/62 | "
            f"{lookup_subtype(subtype, method, 'B2')}/89 |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "| Comparison | Delta BAcc | 95% CI | P(delta>0) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in bootstrap.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['point_delta_bacc']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['probability_delta_gt_zero']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Rescue and harm versus locked H3",
            "",
            "| Subset | N | Rescued | Harmed | Net |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in rescue.iterrows():
        lines.append(
            f"| {row['subset']} | {int(row['n'])} | {int(row['rescue_n'])} | "
            f"{int(row['harm_n'])} | {int(row['net_rescue'])} |"
        )
    lines.extend(
        [
            "",
            "All images, paths, identifiers, dense features, checkpoints, per-case predictions,",
            "and dynamic difficulty assignments remain server-only.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> bool:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    curriculum_frame = load_predictions(run_dir / CURRICULUM / "oof_predictions.csv")
    curriculum_frame = curriculum_frame.sort_values("feature_row").reset_index(drop=True)
    required_metadata = {"source_dataset", "task_l6_label", "feature_row"}
    if not required_metadata.issubset(curriculum_frame.columns):
        raise ValueError("H9 curriculum predictions lack locked metadata")

    labels = curriculum_frame["label_idx"].to_numpy(int)
    curriculum = curriculum_frame["prob_high"].to_numpy(float)
    static = align_probability(
        curriculum_frame, run_dir / STATIC / "oof_predictions.csv", "static"
    )
    control = align_probability(
        curriculum_frame, Path(args.control_predictions), "control"
    )
    c1 = align_probability(curriculum_frame, Path(args.c1_predictions), "c1")
    c2 = align_probability(curriculum_frame, Path(args.c2_predictions), "c2")
    h3 = align_probability(curriculum_frame, Path(args.h3_predictions), "h3")
    control_error = float(np.max(np.abs(control - h3)))

    methods = {
        "SOURCE_RISK_CONTROL": control,
        STATIC: static,
        CURRICULUM: curriculum,
        "LOCKED_C1": c1,
        "LOCKED_C2": c2,
        "LOCKED_H3": h3,
    }
    overall, source, subtype = aggregate_tables(curriculum_frame, methods)
    difficulty = diagnostic_difficulty(curriculum_frame, c1, c2, h3, methods)

    bootstrap_rows = []
    comparisons = {
        "CURRICULUM_MINUS_STATIC": (curriculum, static),
        "CURRICULUM_MINUS_LOCKED_H3": (curriculum, h3),
        "STATIC_MINUS_LOCKED_H3": (static, h3),
        "CURRICULUM_MINUS_LOCKED_C2": (curriculum, c2),
    }
    bootstrap_lookup: dict[str, dict[str, Any]] = {}
    for name, (candidate, reference) in comparisons.items():
        result = paired_bootstrap(
            curriculum_frame,
            candidate,
            reference,
            args.bootstrap_replicates,
            comparison_seed(args.bootstrap_seed, name),
        )
        bootstrap_lookup[name] = result
        bootstrap_rows.append({"comparison": name, **result})
    bootstrap = pd.DataFrame(bootstrap_rows)

    boundary_mask = curriculum_frame["task_l6_label"].isin(["B1", "B2"]).to_numpy()
    persistent_mask = (
        curriculum_frame["diagnostic_difficulty"].astype(str) == "persistent_hard"
    ).to_numpy()
    rescue = pd.DataFrame(
        [
            rescue_record(curriculum_frame, curriculum, h3, "all", np.ones(591, dtype=bool)),
            rescue_record(curriculum_frame, curriculum, h3, "B1+B2", boundary_mask),
            rescue_record(
                curriculum_frame, curriculum, h3, "diagnostic_persistent_hard", persistent_mask
            ),
        ]
    )

    curriculum_metrics = metric_record(labels, curriculum)
    static_metrics = metric_record(labels, static)
    source_curriculum = source[source["method"] == CURRICULUM].set_index("source_dataset")
    source_h3 = source[source["method"] == "LOCKED_H3"].set_index("source_dataset")
    source_delta = (
        source_curriculum["balanced_accuracy"] - source_h3["balanced_accuracy"]
    ).to_dict()
    b1_correct = lookup_subtype(subtype, CURRICULUM, "B1")
    b2_correct = lookup_subtype(subtype, CURRICULUM, "B2")
    third_b2 = curriculum_frame[
        (curriculum_frame["source_dataset"] == "third_batch")
        & (curriculum_frame["task_l6_label"] == "B2")
    ]
    third_b2_correct = int(
        ((third_b2["prob_high"].to_numpy(float) >= 0.5).astype(int) == third_b2["label_idx"].to_numpy(int)).sum()
    )
    boundary_net = int(rescue.loc[rescue["subset"] == "B1+B2", "net_rescue"].iloc[0])
    persistent_net = int(
        rescue.loc[rescue["subset"] == "diagnostic_persistent_hard", "net_rescue"].iloc[0]
    )
    curriculum_static = bootstrap_lookup["CURRICULUM_MINUS_STATIC"]
    curriculum_h3 = bootstrap_lookup["CURRICULUM_MINUS_LOCKED_H3"]

    gates = [
        {
            "gate": "G0",
            "requirement": "control reproduces locked H3 within 1e-5",
            "value": control_error,
            "passed": control_error <= 1e-5,
        },
        {
            "gate": "P1",
            "requirement": "curriculum BAcc >= 0.7739",
            "value": curriculum_metrics["balanced_accuracy"],
            "passed": curriculum_metrics["balanced_accuracy"] >= 0.7739,
        },
        {
            "gate": "P2",
            "requirement": "TP >= 164",
            "value": curriculum_metrics["tp"],
            "passed": curriculum_metrics["tp"] >= 164,
        },
        {
            "gate": "P3",
            "requirement": "TN >= 299",
            "value": curriculum_metrics["tn"],
            "passed": curriculum_metrics["tn"] >= 299,
        },
        {
            "gate": "P4",
            "requirement": "B1 correct >= 40/62",
            "value": b1_correct,
            "passed": b1_correct >= 40,
        },
        {
            "gate": "P5",
            "requirement": "B2 correct >= 59/89",
            "value": b2_correct,
            "passed": b2_correct >= 59,
        },
        {
            "gate": "P6",
            "requirement": "third-batch B2 correct >= 18/29",
            "value": third_b2_correct,
            "passed": third_b2_correct >= 18,
        },
        {
            "gate": "P7",
            "requirement": "curriculum-static delta >= 0.0100 and CI low > 0",
            "value": curriculum_static,
            "passed": curriculum_static["point_delta_bacc"] >= 0.0100
            and curriculum_static["ci95_low"] > 0,
        },
        {
            "gate": "P8",
            "requirement": "curriculum-H3 delta >= 0.0200 and CI low > 0",
            "value": curriculum_h3,
            "passed": curriculum_h3["point_delta_bacc"] >= 0.0200
            and curriculum_h3["ci95_low"] > 0,
        },
        {
            "gate": "P9",
            "requirement": "positive H3 delta >=2/3 sources, none <-0.02, min BAcc >=0.7381",
            "value": {
                "source_delta": source_delta,
                "minimum_bacc": float(source_curriculum["balanced_accuracy"].min()),
            },
            "passed": sum(value > 0 for value in source_delta.values()) >= 2
            and min(source_delta.values()) >= -0.0200
            and float(source_curriculum["balanced_accuracy"].min()) >= 0.7381,
        },
        {
            "gate": "P10",
            "requirement": "B1+B2 net correct gain vs H3 >= 7",
            "value": boundary_net,
            "passed": boundary_net >= 7,
        },
        {
            "gate": "P11",
            "requirement": "diagnostic persistent-hard net gain vs H3 >= 5",
            "value": persistent_net,
            "passed": persistent_net >= 5,
        },
    ]
    integrity = bool(gates[0]["passed"])
    primary_pass = integrity and all(bool(gate["passed"]) for gate in gates[1:])
    static_h3_delta = bootstrap_lookup["STATIC_MINUS_LOCKED_H3"]
    static_boundary_net = int(
        rescue_record(curriculum_frame, static, h3, "B1+B2", boundary_mask)["net_rescue"]
    )
    if not integrity:
        decision = "INVALID_CONTROL_REPRODUCTION"
    elif primary_pass:
        decision = "CURRICULUM_CAPABILITY_PASS"
    elif static_h3_delta["point_delta_bacc"] >= 0.0100 and static_boundary_net >= 0:
        decision = "BALANCING_SIGNAL_ONLY"
    elif curriculum_h3["point_delta_bacc"] >= 0.0100:
        decision = "CURRICULUM_SIGNAL_INSUFFICIENT"
    else:
        decision = "NO_GO_CURRENT_CURRICULUM"

    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    source.to_csv(output_dir / "source_metrics.csv", index=False)
    subtype.to_csv(output_dir / "subtype_metrics.csv", index=False)
    difficulty.to_csv(output_dir / "diagnostic_difficulty_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    rescue.to_csv(output_dir / "rescue_harm.csv", index=False)
    decision_payload = {
        "experiment": EXPERIMENT,
        "stage": "source_lodo",
        "decision": decision,
        "primary_pass": primary_pass,
        "control_max_abs_error": control_error,
        "curriculum_metrics": curriculum_metrics,
        "static_metrics": static_metrics,
        "gates": gates,
    }
    atomic_write_text(
        output_dir / "gate_decision.json",
        json.dumps(decision_payload, ensure_ascii=False, indent=2, default=json_default),
    )
    atomic_write_text(output_dir / "FINAL_DECISION.txt", decision + "\n")
    atomic_write_text(
        output_dir / "H9_H3_DIFFICULTY_BALANCED_CURRICULUM_RESULTS_20260715.md",
        markdown_report(
            decision, gates, overall, subtype, source, bootstrap, rescue, control_error
        ),
    )
    print(json.dumps(decision_payload, ensure_ascii=False, indent=2, default=json_default))
    return primary_pass


def main() -> None:
    args = parse_args()
    passed = run(args)
    if args.enforce_gates and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
