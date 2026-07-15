from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from run_task7_spatial_relational_20260713 import metric_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the locked H12 source-LODO or fivefold GO/NO-GO gates."
    )
    parser.add_argument("--candidate-oof", required=True)
    parser.add_argument("--reference-oof", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-mode", choices=("source_lodo", "fivefold"), required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def canonical_source(value: object) -> str:
    text = str(value)
    return "third_batch" if text.startswith("third_batch") else text


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def load_predictions(path: str | Path, role: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {
        "case_id",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        "prob_high",
        "pred_idx",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{role} predictions lack columns: {missing}")
    if frame["case_id"].duplicated().any():
        raise ValueError(f"{role} predictions contain duplicate cases")
    frame = frame.copy()
    frame["source_dataset"] = frame["source_dataset"].map(canonical_source)
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["prob_high"] = pd.to_numeric(frame["prob_high"], errors="raise").astype(float)
    frame["pred_idx"] = pd.to_numeric(frame["pred_idx"], errors="raise").astype(int)
    if not np.isfinite(frame["prob_high"]).all():
        raise ValueError(f"{role} predictions contain non-finite probabilities")
    if not frame["prob_high"].between(0.0, 1.0).all():
        raise ValueError(f"{role} probabilities fall outside [0, 1]")
    fixed_predictions = (frame["prob_high"].to_numpy() >= 0.5).astype(int)
    if not np.array_equal(fixed_predictions, frame["pred_idx"].to_numpy()):
        raise ValueError(f"{role} predictions do not use the locked 0.5 threshold")
    return frame.sort_values("case_id").reset_index(drop=True)


def align_predictions(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if len(candidate) != 591 or len(reference) != 591:
        raise ValueError(
            f"Locked H12 analysis requires 591 candidate and reference rows: "
            f"{len(candidate)}, {len(reference)}"
        )
    if not candidate["case_id"].equals(reference["case_id"]):
        candidate_only = sorted(set(candidate["case_id"]) - set(reference["case_id"]))
        reference_only = sorted(set(reference["case_id"]) - set(candidate["case_id"]))
        raise ValueError(
            f"Candidate/reference case sets differ: candidate_only={candidate_only[:5]}, "
            f"reference_only={reference_only[:5]}"
        )
    for column in ("source_dataset", "task_l6_label", "label_idx"):
        if not candidate[column].equals(reference[column]):
            raise ValueError(f"Candidate/reference {column} values differ")
    paired = candidate[
        [
            "case_id",
            "source_dataset",
            "task_l6_label",
            "label_idx",
            "prob_high",
            "pred_idx",
        ]
    ].copy()
    paired = paired.rename(
        columns={"prob_high": "candidate_prob_high", "pred_idx": "candidate_pred_idx"}
    )
    paired["reference_prob_high"] = reference["prob_high"].to_numpy()
    paired["reference_pred_idx"] = reference["pred_idx"].to_numpy()
    if "fold_id" in candidate:
        paired["fold_id"] = pd.to_numeric(candidate["fold_id"], errors="raise").astype(int)
    paired["candidate_correct"] = (
        paired["candidate_pred_idx"] == paired["label_idx"]
    )
    paired["reference_correct"] = (
        paired["reference_pred_idx"] == paired["label_idx"]
    )
    return paired


def paired_stratified_bootstrap(
    paired: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if repetitions != 10000:
        raise ValueError("Locked H12 analysis requires exactly 10,000 bootstrap replicates")
    labels = paired["label_idx"].to_numpy(dtype=int)
    candidate = paired["candidate_prob_high"].to_numpy(dtype=float) >= 0.5
    reference = paired["reference_prob_high"].to_numpy(dtype=float) >= 0.5
    strata = [
        group.index.to_numpy(dtype=int)
        for _, group in paired.groupby(["source_dataset", "label_idx"], sort=True)
    ]
    if len(strata) != 6 or any(len(indices) == 0 for indices in strata):
        raise ValueError("Expected six non-empty source-by-risk bootstrap strata")
    rng = np.random.default_rng(seed)
    candidate_positive_correct = np.zeros(repetitions, dtype=np.int32)
    candidate_negative_correct = np.zeros(repetitions, dtype=np.int32)
    reference_positive_correct = np.zeros(repetitions, dtype=np.int32)
    reference_negative_correct = np.zeros(repetitions, dtype=np.int32)
    for indices in strata:
        sampled = rng.choice(
            indices, size=(repetitions, len(indices)), replace=True
        )
        label = int(labels[indices[0]])
        if label == 1:
            candidate_positive_correct += candidate[sampled].sum(axis=1)
            reference_positive_correct += reference[sampled].sum(axis=1)
        else:
            candidate_negative_correct += (~candidate[sampled]).sum(axis=1)
            reference_negative_correct += (~reference[sampled]).sum(axis=1)
    positive_n = int((labels == 1).sum())
    negative_n = int((labels == 0).sum())
    candidate_bacc = 0.5 * (
        candidate_positive_correct / positive_n
        + candidate_negative_correct / negative_n
    )
    reference_bacc = 0.5 * (
        reference_positive_correct / positive_n
        + reference_negative_correct / negative_n
    )
    deltas = candidate_bacc - reference_bacc
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean_delta_bacc": float(np.mean(deltas)),
        "median_delta_bacc": float(np.median(deltas)),
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
    }


def group_comparison(paired: pd.DataFrame, group_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_name, group in paired.groupby(group_column, sort=True):
        candidate = metric_record(group["label_idx"], group["candidate_prob_high"])
        reference = metric_record(group["label_idx"], group["reference_prob_high"])
        records.append(
            {
                group_column: group_name,
                "n": int(len(group)),
                "candidate_bacc": candidate["balanced_accuracy"],
                "reference_bacc": reference["balanced_accuracy"],
                "delta_bacc": candidate["balanced_accuracy"]
                - reference["balanced_accuracy"],
                "candidate_accuracy": candidate["accuracy"],
                "reference_accuracy": reference["accuracy"],
                "candidate_correct": int(group["candidate_correct"].sum()),
                "reference_correct": int(group["reference_correct"].sum()),
            }
        )
    return pd.DataFrame(records)


def subtype_counts(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for subtype, group in paired.groupby("task_l6_label", sort=True):
        records.append(
            {
                "task_l6_label": subtype,
                "n": int(len(group)),
                "candidate_correct": int(group["candidate_correct"].sum()),
                "reference_correct": int(group["reference_correct"].sum()),
                "correct_delta": int(
                    group["candidate_correct"].sum() - group["reference_correct"].sum()
                ),
            }
        )
    return pd.DataFrame(records)


def source_lodo_gates(
    paired: pd.DataFrame,
    candidate_metrics: dict[str, Any],
    source_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
    bootstrap: dict[str, float | int],
) -> dict[str, Any]:
    subtype = subtype_metrics.set_index("task_l6_label")
    b1_correct = int(subtype.loc["B1", "candidate_correct"])
    b2_correct = int(subtype.loc["B2", "candidate_correct"])
    third_b2 = paired[
        (paired["source_dataset"] == "third_batch")
        & (paired["task_l6_label"] == "B2")
    ]
    third_b2_correct = int(third_b2["candidate_correct"].sum())
    source_improvement_count = int((source_metrics["delta_bacc"] > 0.0).sum())
    minimum_source_bacc = float(source_metrics["candidate_bacc"].min())
    minimum_source_delta = float(source_metrics["delta_bacc"].min())
    checks = {
        "coverage_591_unique_threshold_0_5": len(paired) == 591
        and not paired["case_id"].duplicated().any(),
        "pooled_bacc_at_least_0_7639": candidate_metrics["balanced_accuracy"] >= 0.7639,
        "high_risk_correct_at_least_157": int(candidate_metrics["tp"]) >= 157,
        "low_risk_correct_at_least_299": int(candidate_metrics["tn"]) >= 299,
        "b1_correct_at_least_40": b1_correct >= 40,
        "b2_correct_at_least_56": b2_correct >= 56,
        "third_b2_correct_at_least_17": third_b2_correct >= 17,
        "at_least_two_sources_improve": source_improvement_count >= 2,
        "minimum_source_bacc_at_least_0_7381": minimum_source_bacc >= 0.7381,
        "no_source_harm_exceeds_0_015": minimum_source_delta >= -0.015,
        "bootstrap_mean_delta_positive": bootstrap["mean_delta_bacc"] > 0.0,
        "bootstrap_probability_positive_at_least_0_80": bootstrap[
            "probability_delta_positive"
        ]
        >= 0.80,
        "bootstrap_lower_ci_above_minus_0_010": bootstrap["ci_2_5"] > -0.010,
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "pooled_bacc": candidate_metrics["balanced_accuracy"],
            "high_risk_correct": int(candidate_metrics["tp"]),
            "low_risk_correct": int(candidate_metrics["tn"]),
            "b1_correct": b1_correct,
            "b2_correct": b2_correct,
            "third_b2_n": int(len(third_b2)),
            "third_b2_correct": third_b2_correct,
            "source_improvement_count": source_improvement_count,
            "minimum_source_bacc": minimum_source_bacc,
            "minimum_source_delta": minimum_source_delta,
        },
    }


def fivefold_gates(
    candidate_metrics: dict[str, Any],
    fold_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
    bootstrap: dict[str, float | int],
    point_delta: float,
) -> dict[str, Any]:
    subtype = subtype_metrics.set_index("task_l6_label")
    b1_correct = int(subtype.loc["B1", "candidate_correct"])
    b2_correct = int(subtype.loc["B2", "candidate_correct"])
    minimum_fold_bacc = float(fold_metrics["candidate_bacc"].min())
    checks = {
        "pooled_bacc_at_least_0_8053": candidate_metrics["balanced_accuracy"] >= 0.8053,
        "sensitivity_at_least_0_7900": candidate_metrics["sensitivity"] >= 0.7900,
        "specificity_at_least_0_7900": candidate_metrics["specificity"] >= 0.7900,
        "b1_correct_at_least_42": b1_correct >= 42,
        "b2_correct_at_least_62": b2_correct >= 62,
        "all_fold_bacc_at_least_0_70": minimum_fold_bacc >= 0.70,
        "paired_point_delta_positive": point_delta > 0.0,
        "bootstrap_lower_ci_above_minus_0_010": bootstrap["ci_2_5"] > -0.010,
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "pooled_bacc": candidate_metrics["balanced_accuracy"],
            "sensitivity": candidate_metrics["sensitivity"],
            "specificity": candidate_metrics["specificity"],
            "b1_correct": b1_correct,
            "b2_correct": b2_correct,
            "minimum_fold_bacc": minimum_fold_bacc,
            "paired_point_delta_bacc": point_delta,
        },
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(format_value(value) for value in values) + " |")
    return "\n".join(rows)


def render_report(
    split_mode: str,
    decision: dict[str, Any],
    candidate_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    bootstrap: dict[str, float | int],
    source_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame | None,
) -> str:
    lines = [
        "# H12 Locked Gate Analysis",
        "",
        f"- Split mode: `{split_mode}`",
        f"- Decision: **{decision['decision']}**",
        f"- Candidate BAcc: `{candidate_metrics['balanced_accuracy']:.6f}`",
        f"- H3 reference BAcc: `{reference_metrics['balanced_accuracy']:.6f}`",
        f"- Point delta BAcc: `{candidate_metrics['balanced_accuracy'] - reference_metrics['balanced_accuracy']:+.6f}`",
        f"- Bootstrap mean delta: `{bootstrap['mean_delta_bacc']:+.6f}`",
        f"- Bootstrap P(delta > 0): `{bootstrap['probability_delta_positive']:.4f}`",
        f"- Bootstrap 95% CI: `[{bootstrap['ci_2_5']:+.6f}, {bootstrap['ci_97_5']:+.6f}]`",
        "",
        "## Locked checks",
        "",
    ]
    for name, passed in decision["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(["", "## Source comparison", "", markdown_table(source_metrics)])
    lines.extend(["", "## Subtype correct counts", "", markdown_table(subtype_metrics)])
    if fold_metrics is not None:
        lines.extend(["", "## Fold comparison", "", markdown_table(fold_metrics)])
    lines.extend(
        [
            "",
            "The decision is mechanical and uses only the thresholds locked before training.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.seed not in (20260715, 20260716):
        raise ValueError("Only the locked H12 primary or confirmation seed is allowed")
    candidate = load_predictions(args.candidate_oof, "candidate")
    reference = load_predictions(args.reference_oof, "reference")
    paired = align_predictions(candidate, reference)
    candidate_metrics = metric_record(paired["label_idx"], paired["candidate_prob_high"])
    reference_metrics = metric_record(paired["label_idx"], paired["reference_prob_high"])
    point_delta = (
        candidate_metrics["balanced_accuracy"] - reference_metrics["balanced_accuracy"]
    )
    bootstrap = paired_stratified_bootstrap(
        paired, repetitions=args.bootstrap_reps, seed=args.seed
    )
    source_metrics = group_comparison(paired, "source_dataset")
    subtype_metrics = subtype_counts(paired)
    fold_metrics = None
    if args.split_mode == "source_lodo":
        decision = source_lodo_gates(
            paired,
            candidate_metrics,
            source_metrics,
            subtype_metrics,
            bootstrap,
        )
    else:
        if "fold_id" not in paired:
            raise ValueError("Fivefold candidate predictions lack fold_id")
        fold_metrics = group_comparison(paired, "fold_id")
        decision = fivefold_gates(
            candidate_metrics,
            fold_metrics,
            subtype_metrics,
            bootstrap,
            point_delta,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_predictions.csv", paired)
    write_csv(output_dir / "source_comparison.csv", source_metrics)
    write_csv(output_dir / "subtype_comparison.csv", subtype_metrics)
    if fold_metrics is not None:
        write_csv(output_dir / "fold_comparison.csv", fold_metrics)
    payload = {
        "split_mode": args.split_mode,
        "candidate_metrics": candidate_metrics,
        "reference_metrics": reference_metrics,
        "point_delta_bacc": point_delta,
        "bootstrap": bootstrap,
        "gate": decision,
    }
    write_json(output_dir / "h12_locked_gate.json", payload)
    (output_dir / "H12_LOCKED_GATE_REPORT.md").write_text(
        render_report(
            args.split_mode,
            decision,
            candidate_metrics,
            reference_metrics,
            bootstrap,
            source_metrics,
            subtype_metrics,
            fold_metrics,
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
