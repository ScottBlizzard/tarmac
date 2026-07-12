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

CANDIDATES = {
    "swinv2l_384": {
        "fivefold": "370_swinv2l_sixview_gated_oof_20260712",
        "source_lodo": "371_swinv2l_sixview_gated_lodo_20260712",
    },
    "convnextv2l_384": {
        "fivefold": "372_convnextv2l_sixview_gated_oof_20260712",
        "source_lodo": "373_convnextv2l_sixview_gated_lodo_20260712",
    },
    "siglip_so400m_512": {
        "fivefold": "374_siglipso400m_sixview_gated_oof_20260712",
        "source_lodo": "375_siglipso400m_sixview_gated_lodo_20260712",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predeclared Wave C backbone analysis.")
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


def grouped_delta(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    group_column: str,
) -> dict[str, float]:
    result = {}
    for group_name, group in frame.groupby(group_column, dropna=False):
        baseline = common.metric_summary(group["label_idx"], group[baseline_column])[
            "balanced_accuracy"
        ]
        candidate = common.metric_summary(group["label_idx"], group[candidate_column])[
            "balanced_accuracy"
        ]
        result[str(group_name)] = float(candidate - baseline)
    return result


def advancement_decision(
    oof_comparison: dict[str, Any],
    lodo_comparison: dict[str, Any],
    lodo_frame: pd.DataFrame,
    candidate_column: str,
) -> dict[str, Any]:
    source_deltas = grouped_delta(
        lodo_frame, "prob_locked_c1", candidate_column, "source_dataset"
    )
    subtype_deltas = grouped_delta(
        lodo_frame, "prob_locked_c1", candidate_column, "task_l6_label"
    )
    candidate_metrics = common.metric_summary(lodo_frame["label_idx"], lodo_frame[candidate_column])
    positive_sources = sum(value > 0 for value in source_deltas.values())
    boundary_deltas = [subtype_deltas.get(name, float("nan")) for name in ("B1", "B2")]
    checks = {
        "oof_delta_at_least_0_015": oof_comparison["delta_bacc"] >= 0.015,
        "lodo_delta_at_least_0_015": lodo_comparison["delta_bacc"] >= 0.015,
        "positive_sources_at_least_2": positive_sources >= 2,
        "no_source_drop_below_minus_0_02": min(source_deltas.values()) >= -0.02,
        "lodo_sensitivity_at_least_0_70": candidate_metrics["sensitivity"] >= 0.70,
        "lodo_specificity_at_least_0_70": candidate_metrics["specificity"] >= 0.70,
        "no_boundary_subtype_drop_below_minus_0_05": all(
            np.isnan(value) or value >= -0.05 for value in boundary_deltas
        ),
    }
    return {
        **checks,
        "source_deltas": source_deltas,
        "subtype_deltas": subtype_deltas,
        "positive_sources": positive_sources,
        "passes_wave_c_gate": bool(all(checks.values())),
    }


def fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    locked_paths = {
        "fivefold": Path(args.locked_c1_fivefold),
        "source_lodo": Path(args.locked_c1_lodo),
    }
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for split_mode, path in locked_paths.items():
        predictions[("locked_c1", split_mode)] = common.load_prediction(path, "locked_c1")
    for candidate_name, paths in CANDIDATES.items():
        for split_mode, relative in paths.items():
            predictions[(candidate_name, split_mode)] = common.load_prediction(
                run_dir / relative / "oof_predictions.csv", candidate_name
            )

    metric_rows = []
    comparison_rows = []
    aligned: dict[tuple[str, str], pd.DataFrame] = {}
    comparison_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for split_mode in ("fivefold", "source_lodo"):
        for model_name in ("locked_c1", *CANDIDATES):
            frame = predictions[(model_name, split_mode)]
            metric_rows.extend(
                common.subgroup_rows(frame, model_name, f"prob_{model_name}", split_mode)
            )
        for offset, candidate_name in enumerate(CANDIDATES):
            frame = common.align_predictions(
                predictions[("locked_c1", split_mode)],
                predictions[(candidate_name, split_mode)],
                candidate_name,
            )
            result = common.stratified_bootstrap_difference(
                frame,
                "prob_locked_c1",
                f"prob_{candidate_name}",
                args.bootstrap_iterations,
                args.seed + offset + (100 if split_mode == "source_lodo" else 0),
            )
            result.update(
                {"split_mode": split_mode, "baseline": "locked_c1", "candidate": candidate_name}
            )
            comparison_rows.append(result)
            comparison_lookup[(split_mode, candidate_name)] = result
            aligned[(split_mode, candidate_name)] = frame

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(output_dir / "wave_c_model_and_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(
        output_dir / "wave_c_paired_bootstrap_comparisons.csv", index=False, encoding="utf-8-sig"
    )
    decisions = {
        candidate_name: advancement_decision(
            comparison_lookup[("fivefold", candidate_name)],
            comparison_lookup[("source_lodo", candidate_name)],
            aligned[("source_lodo", candidate_name)],
            f"prob_{candidate_name}",
        )
        for candidate_name in CANDIDATES
    }
    write_json(output_dir / "wave_c_advancement_decisions.json", decisions)

    report = [
        "# Task7 Wave C Six-View Backbone Results",
        "",
        "All models use fixed threshold 0.5, 100% coverage, one prespecified gated head, and no external-set selection.",
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
            "## Paired against locked C1",
            "",
            "| Candidate | Split | Delta BAcc [95% CI] | Delta AUC [95% CI] | Rescue/Harm |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparisons.iterrows():
        report.append(
            f"| {row['candidate']} | {row['split_mode']} | {fmt(row['delta_bacc'])} "
            f"[{fmt(row['delta_bacc_ci_low'])}, {fmt(row['delta_bacc_ci_high'])}] | "
            f"{fmt(row['delta_auc'])} [{fmt(row['delta_auc_ci_low'])}, {fmt(row['delta_auc_ci_high'])}] | "
            f"{int(row['rescued'])}/{int(row['harmed'])} |"
        )
    report.extend(["", "## Decisions", ""])
    for candidate_name, decision in decisions.items():
        report.append(
            f"- `{candidate_name}`: **{'PASS' if decision['passes_wave_c_gate'] else 'NO-GO'}**; "
            f"positive LODO sources {decision['positive_sources']}/3; source deltas {decision['source_deltas']}."
        )
    report.extend(
        [
            "",
            "A PASS authorizes a second fixed seed and prespecified equal-weight complementarity test. A NO-GO closes that backbone branch without pooling, loss, or threshold search.",
        ]
    )
    report_path = output_dir / "WAVE_C_SIXVIEW_BACKBONE_RESULTS_20260712.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
