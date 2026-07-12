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
VARIANTS = ("base_only", "attention_roi", "matched_random_roi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Task7 D1 attention-ROI analysis.")
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
        ("base_only", "attention_roi"),
        ("base_only", "matched_random_roi"),
        ("matched_random_roi", "attention_roi"),
        ("locked_c1", "attention_roi"),
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
                predictions[(baseline, split_mode)], predictions[(candidate, split_mode)], candidate
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
    metrics.to_csv(output_dir / "d1_model_and_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(
        output_dir / "d1_paired_bootstrap_comparisons.csv", index=False, encoding="utf-8-sig"
    )

    oof = comparison_lookup[("fivefold", "base_only", "attention_roi")]
    lodo = comparison_lookup[("source_lodo", "base_only", "attention_roi")]
    random_oof = comparison_lookup[("fivefold", "matched_random_roi", "attention_roi")]
    random_lodo = comparison_lookup[("source_lodo", "matched_random_roi", "attention_roi")]
    lodo_frame = aligned_lookup[("source_lodo", "base_only", "attention_roi")]
    source_deltas = group_deltas(
        lodo_frame, "prob_base_only", "prob_attention_roi", "source_dataset"
    )
    subtype_deltas = group_deltas(
        lodo_frame, "prob_base_only", "prob_attention_roi", "task_l6_label"
    )
    checks = {
        "oof_delta_at_least_0_02": oof["delta_bacc"] >= 0.02,
        "lodo_delta_at_least_0_015": lodo["delta_bacc"] >= 0.015,
        "positive_lodo_sources_at_least_2": sum(value > 0 for value in source_deltas.values()) >= 2,
        "no_source_drop_below_minus_0_02": min(source_deltas.values()) >= -0.02,
        "attention_beats_random_oof": random_oof["delta_bacc"] > 0,
        "attention_beats_random_lodo": random_lodo["delta_bacc"] > 0,
        "B1_not_below_minus_0_05": subtype_deltas.get("B1", 0.0) >= -0.05,
        "B2_not_below_minus_0_05": subtype_deltas.get("B2", 0.0) >= -0.05,
    }
    decision = {
        **checks,
        "source_deltas": source_deltas,
        "subtype_deltas": subtype_deltas,
        "passes_d1_gate": bool(all(checks.values())),
    }
    write_json(output_dir / "d1_advancement_decision.json", decision)

    report = [
        "# Task7 D1 Attention-ROI Results",
        "",
        "All models are direct image classifiers at 100% coverage and threshold 0.5. The ROI selector is refit inside every outer split and receives no behavior or source features.",
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
            "## Decision",
            "",
            f"- D1: **{'PASS' if decision['passes_d1_gate'] else 'NO-GO'}**.",
            f"- LODO source deltas: `{source_deltas}`.",
            f"- LODO subtype deltas: `{subtype_deltas}`.",
            "",
            "A PASS still requires comparison with the two-reader manual-ROI oracle. A NO-GO means label-trained visual attention did not provide transferable localization and must not be presented as a coarse-to-fine gain.",
        ]
    )
    report_path = output_dir / "D1_ATTENTION_ROI_RESULTS_20260712.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
