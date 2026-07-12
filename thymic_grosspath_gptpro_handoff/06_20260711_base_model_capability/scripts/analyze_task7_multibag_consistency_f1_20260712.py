from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_task7_native_detail_a1_20260712 as common


DEFAULT_LOCKED_C1_ROOT = Path(
    "/workspace/thymic_project/experiments/base_model_capability_20260711/"
    "phase2_siglipl512_local_pyramid_screen"
)
DEFAULT_E1_ROOT = Path(
    "/workspace/thymic_project/experiments/base_model_capability_20260711/"
    "e1_anatomy_roi_20260712"
)
F1_VARIANTS = ("multibag_mean", "multibag_consistency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Task7 F1 multi-bag analysis.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--e1-run-dir", default=str(DEFAULT_E1_ROOT))
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


def fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def group_deltas(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    group_column: str,
) -> dict[str, float]:
    rows = {}
    for name, group in frame.groupby(group_column, dropna=False):
        baseline = common.metric_summary(group["label_idx"], group[baseline_column])[
            "balanced_accuracy"
        ]
        candidate = common.metric_summary(group["label_idx"], group[candidate_column])[
            "balanced_accuracy"
        ]
        rows[str(name)] = float(candidate - baseline)
    return rows


def plan_quality(run_dir: Path, e1_run_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(run_dir / "f1_random_bag_plan.csv", encoding="utf-8-sig")
    centers = frame.sort_values("scale").drop_duplicates(
        ["feature_row", "bag_index", "role"], keep="first"
    )
    target_distance = np.hypot(
        centers["center_x"] - centers["target_center_x"],
        centers["center_y"] - centers["target_center_y"],
    )
    within_bag_distances = []
    for _, group in centers.groupby(["feature_row", "bag_index"]):
        coordinates = group[["center_x", "center_y"]].to_numpy(dtype=float)
        if len(coordinates) == 2:
            within_bag_distances.append(
                float(np.linalg.norm(coordinates[0] - coordinates[1]))
            )

    e1_frame = pd.read_csv(e1_run_dir / "anatomy_roi_plan.csv", encoding="utf-8-sig")
    e1_first = (
        e1_frame[e1_frame["route"] == "random"]
        .sort_values("scale")
        .drop_duplicates(["case_id", "role"], keep="first")
    )
    f1_first = centers[centers["bag_index"] == 0]
    aligned = f1_first.merge(
        e1_first[["case_id", "role", "center_x", "center_y"]],
        on=["case_id", "role"],
        suffixes=("_f1", "_e1"),
        validate="one_to_one",
    )
    coordinate_difference = np.maximum(
        np.abs(aligned["center_x_f1"] - aligned["center_x_e1"]),
        np.abs(aligned["center_y_f1"] - aligned["center_y_e1"]),
    )
    coverage_gap = centers["coverage_gap"].to_numpy(dtype=float)
    return {
        "cases": int(centers["feature_row"].nunique()),
        "bag_count": int(centers["bag_index"].nunique()),
        "random_center_count": int(len(centers)),
        "random_fallback_rate": float(centers["random_fallback"].astype(bool).mean()),
        "mean_coverage_gap": float(np.mean(coverage_gap)),
        "p95_coverage_gap": float(np.quantile(coverage_gap, 0.95)),
        "max_coverage_gap": float(np.max(coverage_gap)),
        "minimum_target_center_distance": float(np.min(target_distance)),
        "minimum_within_bag_center_distance": float(np.min(within_bag_distances)),
        "bag0_max_coordinate_difference_from_e1": float(
            np.max(coordinate_difference)
        ),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    e1_run_dir = Path(args.e1_run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    locked_paths = {
        "fivefold": Path(args.locked_c1_fivefold),
        "source_lodo": Path(args.locked_c1_lodo),
    }
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for split_mode, locked_path in locked_paths.items():
        predictions[("locked_c1", split_mode)] = common.load_prediction(
            locked_path, "locked_c1"
        )
        predictions[("e1_base", split_mode)] = common.load_prediction(
            e1_run_dir / split_mode / "base_only" / "oof_predictions.csv",
            "e1_base",
        )
        predictions[("single_random", split_mode)] = common.load_prediction(
            e1_run_dir
            / split_mode
            / "matched_random_roi"
            / "oof_predictions.csv",
            "single_random",
        )
        for variant in F1_VARIANTS:
            predictions[(variant, split_mode)] = common.load_prediction(
                run_dir / split_mode / variant / "oof_predictions.csv", variant
            )

    model_names = ("locked_c1", "e1_base", "single_random", *F1_VARIANTS)
    comparisons_to_run = [
        ("e1_base", "multibag_mean"),
        ("e1_base", "multibag_consistency"),
        ("single_random", "multibag_mean"),
        ("single_random", "multibag_consistency"),
        ("multibag_mean", "multibag_consistency"),
        ("locked_c1", "multibag_consistency"),
    ]
    metric_rows = []
    comparison_rows = []
    aligned_lookup: dict[tuple[str, str, str], pd.DataFrame] = {}
    comparison_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for split_mode in ("fivefold", "source_lodo"):
        for model_name in model_names:
            frame = predictions[(model_name, split_mode)]
            metric_rows.extend(
                common.subgroup_rows(frame, model_name, f"prob_{model_name}", split_mode)
            )
        for offset, (baseline, candidate) in enumerate(comparisons_to_run):
            aligned = common.align_predictions(
                predictions[(baseline, split_mode)],
                predictions[(candidate, split_mode)],
                candidate,
            )
            result = common.stratified_bootstrap_difference(
                aligned,
                f"prob_{baseline}",
                f"prob_{candidate}",
                args.bootstrap_iterations,
                args.seed + offset + (100 if split_mode == "source_lodo" else 0),
            )
            result.update(
                {"split_mode": split_mode, "baseline": baseline, "candidate": candidate}
            )
            comparison_rows.append(result)
            comparison_lookup[(split_mode, baseline, candidate)] = result
            aligned_lookup[(split_mode, baseline, candidate)] = aligned

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(
        output_dir / "f1_model_and_subgroup_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        output_dir / "f1_paired_bootstrap_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary = "multibag_consistency"
    oof = comparison_lookup[("fivefold", "e1_base", primary)]
    lodo = comparison_lookup[("source_lodo", "e1_base", primary)]
    single_oof = comparison_lookup[("fivefold", "single_random", primary)]
    single_lodo = comparison_lookup[("source_lodo", "single_random", primary)]
    lodo_frame = aligned_lookup[("source_lodo", "e1_base", primary)]
    source_deltas = group_deltas(
        lodo_frame, "prob_e1_base", f"prob_{primary}", "source_dataset"
    )
    subtype_deltas = group_deltas(
        lodo_frame, "prob_e1_base", f"prob_{primary}", "task_l6_label"
    )
    quality = plan_quality(run_dir, e1_run_dir)
    checks = {
        "oof_delta_at_least_0_02": oof["delta_bacc"] >= 0.02,
        "lodo_delta_at_least_0_015": lodo["delta_bacc"] >= 0.015,
        "positive_lodo_sources_at_least_2": sum(
            value > 0 for value in source_deltas.values()
        )
        >= 2,
        "no_source_drop_below_minus_0_02": min(source_deltas.values()) >= -0.02,
        "consistency_beats_single_random_oof": single_oof["delta_bacc"] > 0,
        "consistency_beats_single_random_lodo": single_lodo["delta_bacc"] > 0,
        "B1_not_below_minus_0_05": subtype_deltas.get("B1", 0.0) >= -0.05,
        "B2_not_below_minus_0_05": subtype_deltas.get("B2", 0.0) >= -0.05,
        "bag_count_is_3": quality["bag_count"] == 3,
        "random_fallback_rate_at_most_0_05": quality["random_fallback_rate"]
        <= 0.05,
        "mean_coverage_gap_at_most_0_02": quality["mean_coverage_gap"] <= 0.02,
        "p95_coverage_gap_at_most_0_05": quality["p95_coverage_gap"] <= 0.05,
        "minimum_target_distance_at_least_0_15": quality[
            "minimum_target_center_distance"
        ]
        >= 0.15,
        "minimum_within_bag_distance_at_least_0_15": quality[
            "minimum_within_bag_center_distance"
        ]
        >= 0.15,
        "bag0_exactly_reproduces_e1": quality[
            "bag0_max_coordinate_difference_from_e1"
        ]
        <= 1e-12,
    }
    decision = {
        **checks,
        "source_deltas": source_deltas,
        "subtype_deltas": subtype_deltas,
        "plan_quality": quality,
        "passes_f1_gate": bool(all(checks.values())),
    }
    write_json(output_dir / "f1_advancement_decision.json", decision)

    report = [
        "# Task7 F1 Multi-Random-Bag Consistency Results",
        "",
        "All models are direct image classifiers at 100% coverage and threshold 0.5. F1 uses three independently sampled, label-free, tissue-matched ROI bags with one shared visual head. No behavior, source, confidence, or error input is used.",
        "",
        "## Overall metrics",
        "",
        "| Model | Split | BAcc | AUC | Sensitivity | Specificity |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    overall = metrics[(metrics["group_type"] == "overall") & (metrics["group"] == "all")]
    for _, row in overall.iterrows():
        report.append(
            f"| {row['model']} | {row['split_mode']} | {fmt(row['balanced_accuracy'])} | "
            f"{fmt(row['auc'])} | {fmt(row['sensitivity'])} | {fmt(row['specificity'])} |"
        )
    report.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Baseline | Candidate | Split | Delta BAcc [95% CI] | Delta AUC [95% CI] | Rescue/Harm |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparisons.iterrows():
        report.append(
            f"| {row['baseline']} | {row['candidate']} | {row['split_mode']} | "
            f"{fmt(row['delta_bacc'])} [{fmt(row['delta_bacc_ci_low'])}, {fmt(row['delta_bacc_ci_high'])}] | "
            f"{fmt(row['delta_auc'])} [{fmt(row['delta_auc_ci_low'])}, {fmt(row['delta_auc_ci_high'])}] | "
            f"{int(row['rescued'])}/{int(row['harmed'])} |"
        )
    report.extend(
        [
            "",
            "## Plan quality",
            "",
            f"- Cases/bags/centers: {quality['cases']}/{quality['bag_count']}/{quality['random_center_count']}.",
            f"- Fallback rate: {fmt(quality['random_fallback_rate'])}.",
            f"- Mean/P95/max tissue-coverage gap: {fmt(quality['mean_coverage_gap'])}/{fmt(quality['p95_coverage_gap'])}/{fmt(quality['max_coverage_gap'])}.",
            f"- Minimum target/within-bag distance: {fmt(quality['minimum_target_center_distance'])}/{fmt(quality['minimum_within_bag_center_distance'])}.",
            f"- Bag-0 max coordinate difference from E1: {quality['bag0_max_coordinate_difference_from_e1']:.3e}.",
            "",
            "## Decision",
            "",
            f"- F1 primary (`multibag_consistency`): **{'PASS' if decision['passes_f1_gate'] else 'NO-GO'}**.",
            f"- LODO source deltas versus E1 base: `{source_deltas}`.",
            f"- LODO subtype deltas versus E1 base: `{subtype_deltas}`.",
            "",
            "A PASS means location-invariant local rereading adds transferable image evidence. A NO-GO closes this random-bag consistency follow-up; the mean-only ablation cannot replace the preregistered primary after results are seen.",
        ]
    )
    report_path = output_dir / "F1_MULTIBAG_RESULTS_20260712.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
