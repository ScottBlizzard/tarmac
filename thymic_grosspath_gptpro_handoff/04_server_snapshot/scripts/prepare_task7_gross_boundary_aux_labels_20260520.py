from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_task7_gross_feature_probe_20260520 import extract_gross_features
from run_task7_hardcore_gross_calibrator_20260520 import add_manual_gross_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task7 gross-finding boundary auxiliary labels.")
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--delta-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/03_hardcore_gross_calibrator/main__img_stack_gross_core__train_allhard_apply_hardcore__balanced_accuracy/routed_case_delta.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/gross_boundary_aux_20260520",
    )
    return parser.parse_args()


def yesno(value: float, threshold: float = 0.5) -> str:
    return "yes" if float(value) >= threshold else "no"


def hemonec_label(row: pd.Series) -> str:
    hemo = int(row.get("kw_hemorrhage", 0) > 0)
    nec = int(row.get("kw_necrosis", 0) > 0)
    cyst = int(row.get("kw_cystic", 0) > 0)
    score = hemo + nec + cyst
    if score >= 2:
        return "marked"
    if score == 1:
        return "mild"
    return "none"


def irregularity_label(row: pd.Series) -> str:
    high_cues = [
        "kw_boundary_unclear",
        "kw_capsule_absent",
        "kw_capsule_involved",
        "kw_lung_attached",
        "kw_pericardium_attached",
        "kw_pleura_attached",
        "kw_necrosis",
        "kw_lobulated",
    ]
    high_score = sum(int(row.get(col, 0) > 0) for col in high_cues)
    protective_score = int(row.get("kw_boundary_clear", 0) > 0) + int(row.get("kw_capsule_complete", 0) > 0)
    if high_score >= 1 and high_score >= protective_score:
        return "high"
    return "low"


def confound_target(row: pd.Series, status: str) -> str:
    label = safe_int(row.get("label_idx", -1), -1)
    base_pred = safe_int(row.get("base_pred_idx", -1), -1)
    score = float(row.get("manual_gross_highrisk_score", 0.0))
    invasive = any(int(row.get(col, 0) > 0) for col in ["kw_lung_attached", "kw_pericardium_attached", "kw_capsule_involved"])
    if label == 0 and base_pred == 1:
        return "TC" if invasive or score >= 1.5 else "B2"
    if label == 1 and base_pred == 0:
        return "A_AB" if score <= 0.5 else "B1"
    if status == "hurt":
        return "B2" if label == 0 else "A_AB"
    if score >= 1.5:
        return "TC" if invasive else "B2"
    if score <= -1.0:
        return "A_AB"
    return "none"


def safe_int(value: object, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def status_from_delta(row: pd.Series) -> str:
    base_correct = str(row.get("base_correct", "")).lower() == "true"
    new_correct = str(row.get("new_correct", "")).lower() == "true"
    if (not base_correct) and new_correct:
        return "rescued_by_gross"
    if base_correct and (not new_correct):
        return "hurt_by_gross"
    if (not base_correct) and (not new_correct):
        return "still_wrong_after_gross"
    return "stable_correct_after_gross"


def bucket(row: pd.Series, status: str) -> str:
    label = int(row.get("label_idx", -1))
    score = float(row.get("manual_gross_highrisk_score", 0.0))
    if status == "hurt_by_gross":
        return "gross_trap_do_not_overtrust"
    if label == 0 and score >= 0.5:
        return "low_truth_with_highrisk_like_gross"
    if label == 1 and score <= 0.0:
        return "high_truth_with_bland_gross"
    if label == 0:
        return "low_with_protective_gross"
    return "high_with_invasive_or_complex_gross"


def sample_weight(status: str, row: pd.Series) -> float:
    if status == "rescued_by_gross":
        return 1.45
    if status == "still_wrong_after_gross":
        return 1.25
    if status == "hurt_by_gross":
        return 0.70
    if str(row.get("difficulty_fine", "")) == "hard_salvage_teacher":
        return 1.15
    return 1.00


def cue_summary(row: pd.Series) -> str:
    cues: list[str] = []
    if row.get("kw_boundary_unclear", 0) > 0:
        cues.append("界不清")
    if row.get("kw_boundary_clear", 0) > 0:
        cues.append("界清")
    if row.get("kw_capsule_complete", 0) > 0:
        cues.append("包膜完整/可见包膜")
    if row.get("kw_capsule_absent", 0) > 0:
        cues.append("包膜不明确/未见包膜")
    if row.get("kw_capsule_involved", 0) > 0:
        cues.append("包膜受侵")
    if row.get("kw_lung_attached", 0) > 0:
        cues.append("肺组织相关")
    if row.get("kw_pericardium_attached", 0) > 0:
        cues.append("心包相关")
    if row.get("kw_hemorrhage", 0) > 0:
        cues.append("出血")
    if row.get("kw_necrosis", 0) > 0:
        cues.append("坏死")
    if row.get("kw_cystic", 0) > 0:
        cues.append("囊性/囊变")
    if row.get("kw_lobulated", 0) > 0:
        cues.append("结节/分叶")
    max_dim = float(row.get("tumor_max_dim_mm", 0.0) or 0.0)
    if max_dim > 0:
        cues.append(f"肿瘤最大径约{max_dim:.0f}mm")
    return "；".join(cues) if cues else "肉眼所见未提取到明确结构线索"


def to_aux_row(row: pd.Series, status: str) -> dict[str, object]:
    pale_uniform = "yes" if row.get("kw_boundary_clear", 0) > 0 and row.get("kw_capsule_complete", 0) > 0 and float(row.get("manual_gross_highrisk_score", 0.0)) <= 0 else "no"
    round_smooth = "yes" if row.get("kw_boundary_clear", 0) > 0 or row.get("kw_capsule_complete", 0) > 0 else "no"
    microcystic = yesno(row.get("kw_cystic", 0))
    multinodular = "yes" if row.get("kw_lobulated", 0) > 0 or row.get("kw_septum", 0) > 0 else "no"
    irregularity = irregularity_label(row)
    return {
        "case_id": row["case_id"],
        "image_name": Path(str(row["training_image_path"])).name,
        "antishortcut_source": "gross_boundary_doctor_table",
        "antishortcut_bucket": bucket(row, status),
        "main_sample_weight": sample_weight(status, row),
        "original_case_id": row.get("original_case_id", ""),
        "task_l6_label": row.get("task_l6_label", ""),
        "task_l7_label": row.get("task_l7_label", ""),
        "difficulty": row.get("difficulty", ""),
        "difficulty_fine": row.get("difficulty_fine", ""),
        "gross_delta_status": status,
        "base_pred_idx": row.get("base_pred_idx", ""),
        "base_prob_high_risk_group": row.get("base_prob_high_risk_group", ""),
        "gross_calibrated_pred_idx": row.get("pred_idx", ""),
        "gross_calibrated_prob_high_risk_group": row.get("prob_high_risk_group", ""),
        "gross_score": row.get("manual_gross_highrisk_score", 0.0),
        "gross_cue_summary": cue_summary(row),
        "exp_manual_pale_uniform": pale_uniform,
        "exp_manual_round_smooth": round_smooth,
        "exp_manual_microcystic": microcystic,
        "exp_manual_multinodular": multinodular,
        "exp_manual_hemonec": hemonec_label(row),
        "exp_manual_irregularity": irregularity,
        "exp_manual_confound_target": confound_target(row, status),
        "exp_manual_view_limit": "yes" if int(row.get("gross_has_text", 0) <= 0) else "no",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(args.registry_csv, dtype={"case_id": str, "original_case_id": str}).fillna("")
    curriculum = pd.read_csv(args.curriculum_csv, dtype={"case_id": str}).fillna("")
    delta = pd.read_csv(args.delta_csv, dtype={"case_id": str}).fillna("")

    gross = add_manual_gross_scores(extract_gross_features(registry))
    feature_cols = [col for col in gross.columns if col not in registry.columns]
    base = pd.concat([registry.reset_index(drop=True), gross[feature_cols].reset_index(drop=True)], axis=1)
    base = base.merge(curriculum[["case_id", "difficulty", "difficulty_fine", "label_idx"]], on="case_id", how="left")

    delta_cols = [
        "case_id",
        "pred_idx",
        "prob_high_risk_group",
        "base_pred_idx",
        "base_prob_high_risk_group",
        "base_correct",
        "new_correct",
    ]
    merged = base.merge(delta[delta_cols], on="case_id", how="left")
    merged["gross_delta_status"] = "not_routed"
    routed_mask = merged["base_correct"].astype(str).str.len() > 0
    merged.loc[routed_mask, "gross_delta_status"] = merged.loc[routed_mask].apply(status_from_delta, axis=1)

    all_rows = [to_aux_row(row, str(row["gross_delta_status"])) for _, row in merged.iterrows()]
    all_labels = pd.DataFrame(all_rows).sort_values(["case_id"]).reset_index(drop=True)

    hard_mask = all_labels["gross_delta_status"].isin(
        ["rescued_by_gross", "hurt_by_gross", "still_wrong_after_gross", "stable_correct_after_gross"]
    )
    hard_labels = all_labels[hard_mask].copy().reset_index(drop=True)
    hard_labels.to_csv(output_dir / "task7_gross_boundary_aux_labels_hardcore_20260520.csv", index=False, encoding="utf-8-sig")
    all_labels.to_csv(output_dir / "task7_gross_boundary_aux_labels_allcases_20260520.csv", index=False, encoding="utf-8-sig")

    review_cols = [
        "case_id",
        "original_case_id",
        "task_l6_label",
        "task_l7_label",
        "difficulty_fine",
        "gross_delta_status",
        "base_pred_idx",
        "base_prob_high_risk_group",
        "gross_calibrated_pred_idx",
        "gross_calibrated_prob_high_risk_group",
        "gross_score",
        "gross_cue_summary",
        "肉眼所见",
    ]
    review = all_labels.merge(registry[["case_id", "肉眼所见"]], on="case_id", how="left")
    review = review.loc[hard_mask, review_cols].sort_values(["gross_delta_status", "task_l6_label", "original_case_id"])
    review.to_csv(output_dir / "task7_gross_boundary_hardcore_delta_review_20260520.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for col in ["gross_delta_status", "antishortcut_bucket", "exp_manual_confound_target", "exp_manual_irregularity", "exp_manual_hemonec"]:
        counts = hard_labels[col].value_counts(dropna=False)
        for value, count in counts.items():
            summary_rows.append({"field": col, "value": value, "count": int(count)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "task7_gross_boundary_aux_label_summary_20260520.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {output_dir}")
    print("hard labels:", len(hard_labels), "all labels:", len(all_labels))
    print(hard_labels["gross_delta_status"].value_counts().to_string())
    print("\nconfound target:")
    print(hard_labels["exp_manual_confound_target"].value_counts().to_string())
    print("\nirregularity:")
    print(hard_labels["exp_manual_irregularity"].value_counts().to_string())


if __name__ == "__main__":
    main()
