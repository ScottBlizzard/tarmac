from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import analyze_task7_native_detail_a1_20260712 as common


ROOT = Path("/workspace/thymic_project/experiments/base_model_capability_20260711")
DEFAULT_C1_ROOT = ROOT / "phase2_siglipl512_local_pyramid_screen"
MODELS = ("locked_c1", "base_only", "physician_roi", "matched_random_roi")
RUN_VARIANTS = ("base_only", "physician_roi", "matched_random_roi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Task7 G1 physician ROI oracle.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--locked-annotations", required=True)
    parser.add_argument("--secure-key", required=True)
    parser.add_argument("--output-dir", default=None)
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
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def box_from_annotation(row: pd.Series) -> tuple[float, float, float, float] | None:
    columns = ["roi1_x1_norm", "roi1_y1_norm", "roi1_x2_norm", "roi1_y2_norm"]
    values = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        return None
    return tuple(values.tolist())


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def reader_audit(annotations: pd.DataFrame, secure: pd.DataFrame) -> dict[str, Any]:
    merged = annotations.merge(
        secure[["oracle_id", "label_idx"]], on="oracle_id", validate="many_to_one"
    )
    reader_metrics = {}
    for reader_id, group in merged.groupby("reader_id"):
        mapped = group["physician_risk_judgment"].astype(str).str.lower().map(
            {"low": 0, "high": 1}
        )
        valid = mapped.notna()
        metrics = (
            common.metric_summary(group.loc[valid, "label_idx"], mapped[valid].astype(float))
            if valid.any()
            else {}
        )
        reader_metrics[str(reader_id)] = {
            "classifiable_n": int(valid.sum()),
            "indeterminate_n": int((~valid).sum()),
            "image_sufficient_yes": int(
                (group["image_sufficient_for_low_high_judgment"].astype(str).str.lower() == "yes").sum()
            ),
            "mean_confidence": float(
                pd.to_numeric(group["physician_confidence_1_to_5"], errors="coerce").mean()
            ),
            **metrics,
        }

    pivot = annotations.pivot(index="oracle_id", columns="reader_id", values="physician_risk_judgment")
    comparable = pivot.isin(["low", "high"]).all(axis=1)
    mapping = {"low": 0, "high": 1}
    risk_kappa = (
        float(
            cohen_kappa_score(
                pivot.loc[comparable, "reader_1"].map(mapping),
                pivot.loc[comparable, "reader_2"].map(mapping),
            )
        )
        if comparable.sum() >= 2
        else float("nan")
    )

    roi_rows = []
    for oracle_id, group in annotations.groupby("oracle_id"):
        lookup = {str(row["reader_id"]): row for _, row in group.iterrows()}
        first = box_from_annotation(lookup["reader_1"])
        second = box_from_annotation(lookup["reader_2"])
        both = first is not None and second is not None
        iou = box_iou(first, second) if both else float("nan")
        roi_rows.append(
            {
                "oracle_id": oracle_id,
                "both_reader_roi": both,
                "roi1_iou": iou,
                "top_roi_hit": bool(both and iou >= 0.25),
            }
        )
    roi_frame = pd.DataFrame(roi_rows)
    both_frame = roi_frame[roi_frame["both_reader_roi"]]
    return {
        "reader_metrics": reader_metrics,
        "comparable_risk_cases": int(comparable.sum()),
        "risk_judgment_kappa": risk_kappa,
        "both_reader_roi_cases": int(len(both_frame)),
        "mean_roi1_iou": float(both_frame["roi1_iou"].mean()) if len(both_frame) else float("nan"),
        "top_roi_hit_rate": float(both_frame["top_roi_hit"].mean()) if len(both_frame) else 0.0,
        "roi_case_table": roi_frame,
    }


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
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(
        args.locked_annotations,
        dtype={"oracle_id": str, "reader_id": str},
        encoding="utf-8-sig",
    )
    secure = pd.read_csv(
        args.secure_key,
        dtype={"oracle_id": str, "case_id": str},
        encoding="utf-8-sig",
    )
    audit = reader_audit(annotations, secure)
    roi_case_table = audit.pop("roi_case_table")
    secure_roi = secure[["oracle_id", "case_id"]].merge(
        roi_case_table, on="oracle_id", validate="one_to_one"
    )
    routed_ids = set(secure_roi.loc[secure_roi["both_reader_roi"], "case_id"].astype(str))
    secure_roi.to_csv(
        output_dir / "g1_reader_roi_agreement.csv", index=False, encoding="utf-8-sig"
    )

    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    c1_paths = {
        "fivefold": Path(args.c1_fivefold),
        "source_lodo": Path(args.c1_lodo),
    }
    for split_mode, c1_path in c1_paths.items():
        c1 = common.load_prediction(c1_path, "locked_c1")
        c1 = c1[c1["case_id"].isin(set(secure["case_id"].astype(str)))].copy()
        if len(c1) != len(secure):
            raise ValueError(f"C1 {split_mode} did not align to all oracle cases")
        predictions[("locked_c1", split_mode)] = c1
        for variant in RUN_VARIANTS[1:]:
            predictions[(variant, split_mode)] = common.load_prediction(
                run_dir / split_mode / variant / "oof_predictions.csv", variant
            )
        predictions[("base_only", split_mode)] = common.load_prediction(
            run_dir / split_mode / "base_only" / "oof_predictions.csv", "base_only"
        )

    metric_rows = []
    comparison_rows = []
    comparison_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    aligned_lookup: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    comparisons_to_run = [
        ("locked_c1", "physician_roi"),
        ("base_only", "physician_roi"),
        ("matched_random_roi", "physician_roi"),
    ]
    for split_mode in ("fivefold", "source_lodo"):
        for model_name in MODELS:
            frame = predictions[(model_name, split_mode)]
            metric_rows.extend(
                common.subgroup_rows(frame, model_name, f"prob_{model_name}", split_mode)
            )
        for subset_name, subset_ids in (("all", None), ("both_reader_roi", routed_ids)):
            for offset, (baseline, candidate) in enumerate(comparisons_to_run):
                reference = predictions[(baseline, split_mode)]
                target = predictions[(candidate, split_mode)]
                if subset_ids is not None:
                    reference = reference[reference["case_id"].isin(subset_ids)].copy()
                    target = target[target["case_id"].isin(subset_ids)].copy()
                aligned = common.align_predictions(reference, target, candidate)
                result = common.stratified_bootstrap_difference(
                    aligned,
                    f"prob_{baseline}",
                    f"prob_{candidate}",
                    args.bootstrap_iterations,
                    args.seed
                    + offset
                    + (100 if split_mode == "source_lodo" else 0)
                    + (1000 if subset_name == "both_reader_roi" else 0),
                )
                result.update(
                    {
                        "split_mode": split_mode,
                        "subset": subset_name,
                        "baseline": baseline,
                        "candidate": candidate,
                    }
                )
                comparison_rows.append(result)
                key = (split_mode, subset_name, baseline, candidate)
                comparison_lookup[key] = result
                aligned_lookup[key] = aligned

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(output_dir / "g1_model_metrics.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(
        output_dir / "g1_paired_comparisons.csv", index=False, encoding="utf-8-sig"
    )
    write_json(output_dir / "g1_reader_audit.json", audit)

    primary_oof = comparison_lookup[
        ("fivefold", "both_reader_roi", "locked_c1", "physician_roi")
    ]
    primary_lodo = comparison_lookup[
        ("source_lodo", "both_reader_roi", "locked_c1", "physician_roi")
    ]
    random_oof = comparison_lookup[
        ("fivefold", "both_reader_roi", "matched_random_roi", "physician_roi")
    ]
    random_lodo = comparison_lookup[
        ("source_lodo", "both_reader_roi", "matched_random_roi", "physician_roi")
    ]
    lodo_frame = aligned_lookup[
        ("source_lodo", "both_reader_roi", "locked_c1", "physician_roi")
    ]
    source_deltas = group_deltas(
        lodo_frame, "prob_locked_c1", "prob_physician_roi", "source_dataset"
    )
    subtype_deltas = group_deltas(
        lodo_frame, "prob_locked_c1", "prob_physician_roi", "task_l6_label"
    )
    checks = {
        "both_reader_roi_cases_at_least_60": audit["both_reader_roi_cases"] >= 60,
        "top_roi_hit_rate_at_least_0_70": audit["top_roi_hit_rate"] >= 0.70,
        "routed_oof_delta_vs_c1_at_least_0_08": primary_oof["delta_bacc"] >= 0.08,
        "routed_lodo_delta_vs_c1_positive": primary_lodo["delta_bacc"] > 0,
        "nonnegative_lodo_sources_at_least_2": sum(
            value >= 0 for value in source_deltas.values()
        )
        >= 2,
        "B1_net_rescue_nonnegative": subtype_deltas.get("B1", 0.0) >= 0,
        "B2_net_rescue_nonnegative": subtype_deltas.get("B2", 0.0) >= 0,
        "physician_beats_random_oof": random_oof["delta_bacc"] > 0,
        "physician_beats_random_lodo": random_lodo["delta_bacc"] > 0,
    }
    decision = {
        **checks,
        "source_deltas_vs_c1": source_deltas,
        "subtype_deltas_vs_c1": subtype_deltas,
        "passes_g1_gate": bool(all(checks.values())),
    }
    write_json(output_dir / "g1_advancement_decision.json", decision)

    report = [
        "# Task7 G1 Two-Reader Physician ROI Oracle Results",
        "",
        "This is a human-assisted upper-bound study, not an automatic model result. All diagnostic heads are direct image classifiers at 100% coverage and threshold 0.5.",
        "",
        "## Reader audit",
        "",
        f"- Both-reader ROI cases: {audit['both_reader_roi_cases']}/120.",
        f"- Mean ROI1 IoU / top-hit rate (IoU>=0.25): {fmt(audit['mean_roi1_iou'])}/{fmt(audit['top_roi_hit_rate'])}.",
        f"- Blinded risk-judgment kappa: {fmt(audit['risk_judgment_kappa'])}.",
        "",
        "## Overall model metrics",
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
            "| Subset | Baseline | Candidate | Split | Delta BAcc [95% CI] | Rescue/Harm |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for _, row in comparisons.iterrows():
        report.append(
            f"| {row['subset']} | {row['baseline']} | {row['candidate']} | {row['split_mode']} | "
            f"{fmt(row['delta_bacc'])} [{fmt(row['delta_bacc_ci_low'])}, {fmt(row['delta_bacc_ci_high'])}] | "
            f"{int(row['rescued'])}/{int(row['harmed'])} |"
        )
    report.extend(
        [
            "",
            "## Decision",
            "",
            f"- G1 manual ROI oracle: **{'PASS' if decision['passes_g1_gate'] else 'NO-GO'}**.",
            f"- Routed LODO source deltas versus C1: `{source_deltas}`.",
            f"- Routed LODO subtype deltas versus C1: `{subtype_deltas}`.",
            "",
            "PASS permits a fully nested anatomical ROI detector experiment. NO-GO stops automatic ROI development on the same single photographs.",
        ]
    )
    (output_dir / "G1_PHYSICIAN_ROI_ORACLE_RESULTS_20260712.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
