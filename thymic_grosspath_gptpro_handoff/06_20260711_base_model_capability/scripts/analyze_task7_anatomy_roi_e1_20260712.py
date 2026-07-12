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
VARIANTS = ("base_only", "anatomy_roi", "matched_random_roi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Task7 E1 anatomy-ROI analysis.")
    parser.add_argument("--run-dir", required=True)
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


def fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def roi_plan_quality(run_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(run_dir / "anatomy_roi_plan.csv", encoding="utf-8-sig")
    centers = frame.sort_values("scale").drop_duplicates(
        ["feature_row", "route", "role"], keep="first"
    )
    anatomy = centers[centers["route"] == "anatomy"].copy()
    random = centers[centers["route"] == "random"].copy()
    paired = anatomy.merge(
        random,
        on=["feature_row", "role"],
        suffixes=("_anatomy", "_random"),
        validate="one_to_one",
    )
    coverage_gap = np.abs(
        paired["detail_tissue_coverage_anatomy"]
        - paired["detail_tissue_coverage_random"]
    )
    anatomy_random_distance = np.hypot(
        paired["center_x_anatomy"] - paired["center_x_random"],
        paired["center_y_anatomy"] - paired["center_y_random"],
    )
    random_pair_distances = []
    for _, group in random.groupby("feature_row"):
        coordinates = group[["center_x", "center_y"]].to_numpy(dtype=float)
        if len(coordinates) == 2:
            random_pair_distances.append(float(np.linalg.norm(coordinates[0] - coordinates[1])))
    return {
        "cases": int(frame["feature_row"].nunique()),
        "anatomy_fallback_centers": int(anatomy["fallback"].astype(bool).sum()),
        "random_fallback_centers": int(random["fallback"].astype(bool).sum()),
        "random_fallback_rate": float(random["fallback"].astype(bool).mean()),
        "mean_absolute_detail_tissue_coverage_gap": float(
            np.mean(coverage_gap)
        ),
        "p95_absolute_detail_tissue_coverage_gap": float(np.quantile(coverage_gap, 0.95)),
        "max_absolute_detail_tissue_coverage_gap": float(
            np.max(coverage_gap)
        ),
        "minimum_corresponding_anatomy_random_center_distance": float(
            np.min(anatomy_random_distance)
        ),
        "minimum_random_pair_center_distance": float(np.min(random_pair_distances)),
        "mean_anatomy_detail_tissue_coverage": {
            str(role): float(group["detail_tissue_coverage"].mean())
            for role, group in anatomy.groupby("role")
        },
        "mean_random_detail_tissue_coverage": {
            str(role): float(group["detail_tissue_coverage"].mean())
            for role, group in random.groupby("role")
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    locked_paths = {
        "fivefold": Path(args.locked_c1_fivefold),
        "source_lodo": Path(args.locked_c1_lodo),
    }
    for split_mode, path in locked_paths.items():
        predictions[("locked_c1", split_mode)] = common.load_prediction(path, "locked_c1")
        for variant in VARIANTS:
            predictions[(variant, split_mode)] = common.load_prediction(
                run_dir / split_mode / variant / "oof_predictions.csv", variant
            )

    comparisons_to_run = [
        ("base_only", "anatomy_roi"),
        ("base_only", "matched_random_roi"),
        ("matched_random_roi", "anatomy_roi"),
        ("locked_c1", "anatomy_roi"),
    ]
    metric_rows = []
    comparison_rows = []
    aligned_lookup: dict[tuple[str, str, str], pd.DataFrame] = {}
    comparison_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for split_mode in ("fivefold", "source_lodo"):
        for model_name in ("locked_c1", *VARIANTS):
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
        output_dir / "e1_model_and_subgroup_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        output_dir / "e1_paired_bootstrap_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof = comparison_lookup[("fivefold", "base_only", "anatomy_roi")]
    lodo = comparison_lookup[("source_lodo", "base_only", "anatomy_roi")]
    random_oof = comparison_lookup[("fivefold", "matched_random_roi", "anatomy_roi")]
    random_lodo = comparison_lookup[("source_lodo", "matched_random_roi", "anatomy_roi")]
    lodo_frame = aligned_lookup[("source_lodo", "base_only", "anatomy_roi")]
    source_deltas = group_deltas(
        lodo_frame, "prob_base_only", "prob_anatomy_roi", "source_dataset"
    )
    subtype_deltas = group_deltas(
        lodo_frame, "prob_base_only", "prob_anatomy_roi", "task_l6_label"
    )
    checks = {
        "oof_delta_at_least_0_02": oof["delta_bacc"] >= 0.02,
        "lodo_delta_at_least_0_015": lodo["delta_bacc"] >= 0.015,
        "positive_lodo_sources_at_least_2": sum(
            value > 0 for value in source_deltas.values()
        )
        >= 2,
        "no_source_drop_below_minus_0_02": min(source_deltas.values()) >= -0.02,
        "anatomy_beats_random_oof": random_oof["delta_bacc"] > 0,
        "anatomy_beats_random_lodo": random_lodo["delta_bacc"] > 0,
        "B1_not_below_minus_0_05": subtype_deltas.get("B1", 0.0) >= -0.05,
        "B2_not_below_minus_0_05": subtype_deltas.get("B2", 0.0) >= -0.05,
    }
    plan_quality = roi_plan_quality(run_dir)
    checks.update(
        {
            "random_fallback_rate_at_most_0_05": plan_quality["random_fallback_rate"]
            <= 0.05,
            "mean_tissue_coverage_gap_at_most_0_02": plan_quality[
                "mean_absolute_detail_tissue_coverage_gap"
            ]
            <= 0.02,
            "p95_tissue_coverage_gap_at_most_0_05": plan_quality[
                "p95_absolute_detail_tissue_coverage_gap"
            ]
            <= 0.05,
            "corresponding_center_distance_at_least_0_15": plan_quality[
                "minimum_corresponding_anatomy_random_center_distance"
            ]
            >= 0.15,
            "random_pair_distance_at_least_0_15": plan_quality[
                "minimum_random_pair_center_distance"
            ]
            >= 0.15,
        }
    )
    decision = {
        **checks,
        "source_deltas": source_deltas,
        "subtype_deltas": subtype_deltas,
        "roi_plan_quality": plan_quality,
        "passes_e1_gate": bool(all(checks.values())),
    }
    write_json(output_dir / "e1_advancement_decision.json", decision)

    report = [
        "# Task7 E1 Label-Free Anatomy-ROI Results",
        "",
        "All models are direct image classifiers at 100% coverage and threshold 0.5. The ROI locator is fixed, deterministic, label-free, and source-blind. Anatomy and random ROI views use identical outside-specimen neutralization.",
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
            "| Baseline | Candidate | Split | Delta BAcc [95% CI] | Rescue/Harm |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for _, row in comparisons.iterrows():
        report.append(
            f"| {row['baseline']} | {row['candidate']} | {row['split_mode']} | "
            f"{fmt(row['delta_bacc'])} [{fmt(row['delta_bacc_ci_low'])}, {fmt(row['delta_bacc_ci_high'])}] | "
            f"{int(row['rescued'])}/{int(row['harmed'])} |"
        )
    report.extend(
        [
            "",
            "## ROI-plan quality",
            "",
            f"- Cases: {plan_quality['cases']}.",
            f"- Anatomy/random fallback centers: {plan_quality['anatomy_fallback_centers']}/{plan_quality['random_fallback_centers']}.",
            f"- Random fallback rate: {fmt(plan_quality['random_fallback_rate'])}.",
            f"- Mean/P95/max absolute detail tissue-coverage gap: {fmt(plan_quality['mean_absolute_detail_tissue_coverage_gap'])}/{fmt(plan_quality['p95_absolute_detail_tissue_coverage_gap'])}/{fmt(plan_quality['max_absolute_detail_tissue_coverage_gap'])}.",
            f"- Minimum corresponding/random-pair center distances: {fmt(plan_quality['minimum_corresponding_anatomy_random_center_distance'])}/{fmt(plan_quality['minimum_random_pair_center_distance'])}.",
            "",
            "## Decision",
            "",
            f"- E1: **{'PASS' if decision['passes_e1_gate'] else 'NO-GO'}**.",
            f"- LODO source deltas: `{source_deltas}`.",
            f"- LODO subtype deltas: `{subtype_deltas}`.",
            "",
            "A PASS advances the fixed image-grounded coarse-to-fine route to confirmation. A NO-GO means these automatic anatomy proxies do not provide transferable fine evidence and must not be represented as a successful multistage model.",
        ]
    )
    report_path = output_dir / "E1_ANATOMY_ROI_RESULTS_20260712.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
