from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)


DEFAULT_LOCKED_C1_ROOT = Path(
    "/workspace/thymic_project/experiments/base_model_capability_20260711/"
    "phase2_siglipl512_local_pyramid_screen"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired predeclared analysis for the Task7 A1 pilot.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--control-run-dir",
        default=None,
        help="Optional run containing c1_hier_mil predictions when native variants were rerun separately.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--locked-c1-fivefold",
        default=str(
            DEFAULT_LOCKED_C1_ROOT
            / "347_siglipl512_localpyramid6_gated_fivefold_cw_20260711"
            / "oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--locked-c1-lodo",
        default=str(
            DEFAULT_LOCKED_C1_ROOT
            / "348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711"
            / "oof_predictions.csv"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_summary(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probability))
        balanced_accuracy = float(balanced_accuracy_score(y_true, prediction))
    else:
        auc = float("nan")
        balanced_accuracy = float(accuracy_score(y_true, prediction))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "sensitivity": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_prediction(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig")
    required = ["case_id", "label_idx", "prob_high", "source_dataset", "task_l6_label"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[required].copy()
    frame = frame.rename(columns={"prob_high": f"prob_{name}"})
    if frame["case_id"].duplicated().any():
        raise ValueError(f"Prediction file has duplicate cases: {path}")
    return frame


def align_predictions(reference: pd.DataFrame, candidate: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    probability_column = f"prob_{candidate_name}"
    aligned = reference.merge(
        candidate[["case_id", "label_idx", probability_column]],
        on="case_id",
        how="inner",
        suffixes=("", "_candidate"),
        validate="one_to_one",
    )
    if len(aligned) != len(reference) or len(aligned) != len(candidate):
        raise ValueError(
            f"Case mismatch for {candidate_name}: reference={len(reference)} candidate={len(candidate)} aligned={len(aligned)}"
        )
    if not np.array_equal(aligned["label_idx"], aligned["label_idx_candidate"]):
        raise ValueError(f"Label mismatch for {candidate_name}")
    return aligned.drop(columns=["label_idx_candidate"])


def stratified_bootstrap_difference(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy(dtype=int) for _, group in frame.groupby(["source_dataset", "label_idx"])]
    baseline = metric_summary(frame["label_idx"], frame[baseline_column])
    candidate = metric_summary(frame["label_idx"], frame[candidate_column])
    bacc_differences = np.empty(iterations, dtype=float)
    auc_differences = np.empty(iterations, dtype=float)
    labels = frame["label_idx"].to_numpy(dtype=int)
    baseline_probability = frame[baseline_column].to_numpy(dtype=float)
    candidate_probability = frame[candidate_column].to_numpy(dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        sampled_labels = labels[sampled]
        bacc_differences[iteration] = balanced_accuracy_score(
            sampled_labels, (candidate_probability[sampled] >= 0.5).astype(int)
        ) - balanced_accuracy_score(sampled_labels, (baseline_probability[sampled] >= 0.5).astype(int))
        auc_differences[iteration] = roc_auc_score(
            sampled_labels, candidate_probability[sampled]
        ) - roc_auc_score(sampled_labels, baseline_probability[sampled])
    baseline_prediction = baseline_probability >= 0.5
    candidate_prediction = candidate_probability >= 0.5
    correct_baseline = baseline_prediction == labels
    correct_candidate = candidate_prediction == labels
    return {
        "n": len(frame),
        "baseline_bacc": baseline["balanced_accuracy"],
        "candidate_bacc": candidate["balanced_accuracy"],
        "delta_bacc": float(candidate["balanced_accuracy"] - baseline["balanced_accuracy"]),
        "delta_bacc_ci_low": float(np.quantile(bacc_differences, 0.025)),
        "delta_bacc_ci_high": float(np.quantile(bacc_differences, 0.975)),
        "baseline_auc": baseline["auc"],
        "candidate_auc": candidate["auc"],
        "delta_auc": float(candidate["auc"] - baseline["auc"]),
        "delta_auc_ci_low": float(np.quantile(auc_differences, 0.025)),
        "delta_auc_ci_high": float(np.quantile(auc_differences, 0.975)),
        "rescued": int((~correct_baseline & correct_candidate).sum()),
        "harmed": int((correct_baseline & ~correct_candidate).sum()),
        "net_rescue": int((~correct_baseline & correct_candidate).sum() - (correct_baseline & ~correct_candidate).sum()),
        "bootstrap_iterations": iterations,
    }


def subgroup_rows(frame: pd.DataFrame, model_name: str, probability_column: str, split_mode: str) -> list[dict[str, Any]]:
    rows = []
    for group_type, column in [
        ("overall", None),
        ("source_dataset", "source_dataset"),
        ("task_l6_label", "task_l6_label"),
    ]:
        groups = [("all", frame)] if column is None else frame.groupby(column, dropna=False)
        for group_name, group in groups:
            rows.append(
                {
                    "model": model_name,
                    "split_mode": split_mode,
                    "group_type": group_type,
                    "group": str(group_name),
                    **metric_summary(group["label_idx"], group[probability_column]),
                }
            )
    return rows


def source_positive_count(frame: pd.DataFrame, baseline_column: str, candidate_column: str) -> int:
    positives = 0
    for _, group in frame.groupby("source_dataset"):
        baseline = metric_summary(group["label_idx"], group[baseline_column])["balanced_accuracy"]
        candidate = metric_summary(group["label_idx"], group[candidate_column])["balanced_accuracy"]
        positives += int(candidate > baseline)
    return positives


def advancement_decision(
    oof_comparison: dict[str, Any],
    lodo_comparison: dict[str, Any],
    lodo_frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
) -> dict[str, Any]:
    positive_sources = source_positive_count(lodo_frame, baseline_column, candidate_column)
    checks = {
        "oof_delta_at_least_0_03": oof_comparison["delta_bacc"] >= 0.03,
        "oof_ci_lower_above_zero": oof_comparison["delta_bacc_ci_low"] > 0.0,
        "lodo_delta_at_least_0_02": lodo_comparison["delta_bacc"] >= 0.02,
        "positive_lodo_sources_at_least_2": positive_sources >= 2,
    }
    return {
        **checks,
        "positive_lodo_sources": positive_sources,
        "passes_fixed_family_pilot_gate": bool(all(checks.values())),
        "note": "A pass advances to nested confirmation; it is not itself a locked selected-pipeline estimate.",
    }


def format_float(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = ["c1_hier_mil", "native_hier_mil", "native_cross_attention"]
    split_modes = ["fivefold", "source_lodo"]
    control_run_dir = Path(args.control_run_dir) if args.control_run_dir else run_dir
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for split_mode in split_modes:
        for variant in variants:
            variant_run_dir = control_run_dir if variant == "c1_hier_mil" else run_dir
            predictions[(variant, split_mode)] = load_prediction(
                variant_run_dir / variant / split_mode / "oof_predictions.csv", variant
            )
    locked_paths = {
        "fivefold": Path(args.locked_c1_fivefold),
        "source_lodo": Path(args.locked_c1_lodo),
    }
    for split_mode, path in locked_paths.items():
        predictions[("locked_c1", split_mode)] = load_prediction(path, "locked_c1")

    metric_rows = []
    comparison_rows = []
    comparison_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    aligned_lookup: dict[tuple[str, str, str], pd.DataFrame] = {}
    comparisons = [
        ("c1_hier_mil", "native_hier_mil"),
        ("c1_hier_mil", "native_cross_attention"),
        ("native_hier_mil", "native_cross_attention"),
        ("locked_c1", "native_hier_mil"),
        ("locked_c1", "native_cross_attention"),
    ]
    for split_mode in split_modes:
        for model_name in ["locked_c1", *variants]:
            frame = predictions[(model_name, split_mode)]
            metric_rows.extend(subgroup_rows(frame, model_name, f"prob_{model_name}", split_mode))
        for comparison_index, (baseline_name, candidate_name) in enumerate(comparisons):
            aligned = align_predictions(
                predictions[(baseline_name, split_mode)],
                predictions[(candidate_name, split_mode)],
                candidate_name,
            )
            result = stratified_bootstrap_difference(
                aligned,
                f"prob_{baseline_name}",
                f"prob_{candidate_name}",
                args.bootstrap_iterations,
                args.seed + comparison_index + (100 if split_mode == "source_lodo" else 0),
            )
            result.update(
                {
                    "split_mode": split_mode,
                    "baseline": baseline_name,
                    "candidate": candidate_name,
                }
            )
            comparison_rows.append(result)
            comparison_lookup[(split_mode, baseline_name, candidate_name)] = result
            aligned_lookup[(split_mode, baseline_name, candidate_name)] = aligned

    metrics = pd.DataFrame(metric_rows)
    comparisons_frame = pd.DataFrame(comparison_rows)
    metrics.to_csv(output_dir / "a1_model_and_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    comparisons_frame.to_csv(output_dir / "a1_paired_bootstrap_comparisons.csv", index=False, encoding="utf-8-sig")

    decisions = {}
    for candidate_name in ["native_hier_mil", "native_cross_attention"]:
        baseline_name = "locked_c1"
        oof_key = ("fivefold", baseline_name, candidate_name)
        lodo_key = ("source_lodo", baseline_name, candidate_name)
        decisions[candidate_name] = advancement_decision(
            comparison_lookup[oof_key],
            comparison_lookup[lodo_key],
            aligned_lookup[lodo_key],
            f"prob_{baseline_name}",
            f"prob_{candidate_name}",
        )
    write_json(output_dir / "a1_advancement_decisions.json", decisions)

    report_lines = [
        "# Task7 A1 Native-Detail Pilot Results",
        "",
        "All values use 100% coverage and threshold 0.5. Fixed families are reported separately; no pooled-OOF winner replacement was performed.",
        "",
        "## Overall metrics",
        "",
        "| Model | Split | BAcc | AUC | Sensitivity | Specificity |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    overall = metrics[(metrics["group_type"] == "overall") & (metrics["group"] == "all")]
    for _, row in overall.iterrows():
        report_lines.append(
            f"| {row['model']} | {row['split_mode']} | {format_float(row['balanced_accuracy'])} | "
            f"{format_float(row['auc'])} | {format_float(row['sensitivity'])} | {format_float(row['specificity'])} |"
        )
    report_lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Split | Baseline | Candidate | Delta BAcc [95% CI] | Delta AUC [95% CI] | Rescue/Harm |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparisons_frame.iterrows():
        report_lines.append(
            f"| {row['split_mode']} | {row['baseline']} | {row['candidate']} | "
            f"{format_float(row['delta_bacc'])} [{format_float(row['delta_bacc_ci_low'])}, {format_float(row['delta_bacc_ci_high'])}] | "
            f"{format_float(row['delta_auc'])} [{format_float(row['delta_auc_ci_low'])}, {format_float(row['delta_auc_ci_high'])}] | "
            f"{int(row['rescued'])}/{int(row['harmed'])} |"
        )
    report_lines.extend(["", "## Predeclared advancement decisions", ""])
    for candidate_name, decision in decisions.items():
        report_lines.append(
            f"- `{candidate_name}`: **{'PASS' if decision['passes_fixed_family_pilot_gate'] else 'NO-GO'}**; "
            f"positive LODO sources {decision['positive_lodo_sources']}/3."
        )
    report_lines.extend(
        [
            "",
            "A PASS only authorizes fully nested confirmation and resolution stress testing. A NO-GO closes this fixed native-tile family after the two prespecified architectures.",
        ]
    )
    (output_dir / "A1_NATIVE_DETAIL_PILOT_RESULTS_20260712.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(report_lines), flush=True)


if __name__ == "__main__":
    main()
