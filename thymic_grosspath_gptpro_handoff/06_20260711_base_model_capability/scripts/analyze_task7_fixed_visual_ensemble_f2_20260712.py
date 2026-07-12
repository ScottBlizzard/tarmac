from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import analyze_task7_native_detail_a1_20260712 as common


ROOT = Path("/workspace/thymic_project/experiments/base_model_capability_20260711")
DEFAULT_C1_ROOT = ROOT / "phase2_siglipl512_local_pyramid_screen"
DEFAULT_C2_ROOT = (
    ROOT / "phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion"
)
DEFAULT_F1_ROOT = ROOT / "f1_multibag_consistency_20260712"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 F2 fixed equal visual ensemble analysis."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--c1-fivefold",
        default=str(
            DEFAULT_C1_ROOT
            / "347_siglipl512_localpyramid6_gated_fivefold_cw_20260711"
            / "oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--c1-lodo",
        default=str(
            DEFAULT_C1_ROOT
            / "348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711"
            / "oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--c2-fivefold", default=str(DEFAULT_C2_ROOT / "oof_predictions.csv")
    )
    parser.add_argument(
        "--c2-lodo", default=str(DEFAULT_C2_ROOT / "lodo_predictions.csv")
    )
    parser.add_argument(
        "--f1-fivefold",
        default=str(
            DEFAULT_F1_ROOT
            / "fivefold"
            / "multibag_consistency"
            / "oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--f1-lodo",
        default=str(
            DEFAULT_F1_ROOT
            / "source_lodo"
            / "multibag_consistency"
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


def fixed_fusion(c1: pd.DataFrame, f1: pd.DataFrame) -> pd.DataFrame:
    aligned = common.align_predictions(c1, f1, "f1")
    aligned["prob_f2"] = 0.5 * aligned["prob_c1"] + 0.5 * aligned["prob_f1"]
    return aligned[
        [
            "case_id",
            "label_idx",
            "source_dataset",
            "task_l6_label",
            "prob_f2",
        ]
    ].copy()


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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        ("c1", "fivefold"): Path(args.c1_fivefold),
        ("c1", "source_lodo"): Path(args.c1_lodo),
        ("c2", "fivefold"): Path(args.c2_fivefold),
        ("c2", "source_lodo"): Path(args.c2_lodo),
        ("f1", "fivefold"): Path(args.f1_fivefold),
        ("f1", "source_lodo"): Path(args.f1_lodo),
    }
    predictions: dict[tuple[str, str], pd.DataFrame] = {
        key: common.load_prediction(path, key[0]) for key, path in paths.items()
    }
    for split_mode in ("fivefold", "source_lodo"):
        predictions[("f2", split_mode)] = fixed_fusion(
            predictions[("c1", split_mode)], predictions[("f1", split_mode)]
        )
        frame = predictions[("f2", split_mode)].rename(
            columns={"prob_f2": "prob_high"}
        )
        frame["pred_idx"] = (frame["prob_high"] >= 0.5).astype(int)
        frame.to_csv(
            output_dir / f"f2_{split_mode}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    comparisons_to_run = [
        ("c1", "f2"),
        ("f1", "f2"),
        ("c2", "f2"),
        ("c1", "c2"),
    ]
    metric_rows = []
    comparison_rows = []
    comparison_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    aligned_lookup: dict[tuple[str, str, str], pd.DataFrame] = {}
    for split_mode in ("fivefold", "source_lodo"):
        for model_name in ("c1", "c2", "f1", "f2"):
            frame = predictions[(model_name, split_mode)]
            metric_rows.extend(
                common.subgroup_rows(
                    frame, model_name, f"prob_{model_name}", split_mode
                )
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
        output_dir / "f2_model_and_subgroup_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        output_dir / "f2_paired_bootstrap_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_c1 = comparison_lookup[("fivefold", "c1", "f2")]
    lodo_c1 = comparison_lookup[("source_lodo", "c1", "f2")]
    oof_f1 = comparison_lookup[("fivefold", "f1", "f2")]
    lodo_f1 = comparison_lookup[("source_lodo", "f1", "f2")]
    oof_c2 = comparison_lookup[("fivefold", "c2", "f2")]
    lodo_c2 = comparison_lookup[("source_lodo", "c2", "f2")]
    lodo_frame = aligned_lookup[("source_lodo", "c1", "f2")]
    source_deltas = group_deltas(lodo_frame, "prob_c1", "prob_f2", "source_dataset")
    subtype_deltas = group_deltas(lodo_frame, "prob_c1", "prob_f2", "task_l6_label")
    checks = {
        "oof_delta_vs_c1_at_least_0_01": oof_c1["delta_bacc"] >= 0.01,
        "lodo_delta_vs_c1_at_least_0_01": lodo_c1["delta_bacc"] >= 0.01,
        "f2_beats_f1_oof": oof_f1["delta_bacc"] > 0,
        "f2_beats_f1_lodo": lodo_f1["delta_bacc"] > 0,
        "positive_lodo_sources_at_least_2": sum(
            value > 0 for value in source_deltas.values()
        )
        >= 2,
        "no_source_drop_below_minus_0_02": min(source_deltas.values()) >= -0.02,
        "B1_not_below_c1_minus_0_02": subtype_deltas.get("B1", 0.0) >= -0.02,
        "B2_not_below_c1_minus_0_02": subtype_deltas.get("B2", 0.0) >= -0.02,
        "oof_not_below_c2_minus_0_005": oof_c2["delta_bacc"] >= -0.005,
        "lodo_beats_c2": lodo_c2["delta_bacc"] > 0,
    }
    decision = {
        **checks,
        "source_deltas_vs_c1": source_deltas,
        "subtype_deltas_vs_c1": subtype_deltas,
        "passes_f2_gate": bool(all(checks.values())),
        "fusion_rule": "0.5 * C1 probability + 0.5 * F1 multibag-consistency probability",
        "threshold": 0.5,
    }
    write_json(output_dir / "f2_advancement_decision.json", decision)

    report = [
        "# Task7 F2 Fixed Visual Ensemble Results",
        "",
        "F2 is a fixed 1:1 probability average of locked C1 and F1 multi-bag consistency. Both members are direct image readers. No routing, confidence feature, learned fusion weight, source input, rejection, or threshold search is used.",
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
            "## Decision",
            "",
            f"- F2: **{'PASS' if decision['passes_f2_gate'] else 'NO-GO'}**.",
            f"- LODO source deltas versus C1: `{source_deltas}`.",
            f"- LODO subtype deltas versus C1: `{subtype_deltas}`.",
            "",
            "A PASS permits F2 to remain an internal image-grounded candidate for fresh blind external testing. It does not prove external generalization. A NO-GO closes this fixed C1+F1 ensemble; no weight or threshold scan follows.",
        ]
    )
    (output_dir / "F2_FIXED_VISUAL_ENSEMBLE_RESULTS_20260712.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
