from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_local_review_path(row: pd.Series, local_image_root: Path) -> str:
    if row["source_dataset"] != "batch1":
        return ""
    return str(local_image_root / row["selected_image_relpath"])


def make_template(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["view_type_round1"] = ""
    out["view_type_confidence"] = ""
    out["cut_surface_degree"] = ""
    out["outer_surface_degree"] = ""
    out["mixed_context"] = ""
    out["tumor_visible_degree"] = ""
    out["fat_context_degree"] = ""
    out["scale_visible"] = ""
    out["is_preferred_main_view"] = ""
    out["alternate_view_needed"] = ""
    out["review_notes"] = ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-registry", required=True)
    parser.add_argument("--selected-manifest", required=True)
    parser.add_argument("--task7-oof", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-image-root", required=True)
    parser.add_argument("--priority-correct-per-group", type=int, default=20)
    args = parser.parse_args()

    case_registry = pd.read_csv(args.case_registry)
    manifest = pd.read_csv(args.selected_manifest)
    task7_oof = pd.read_csv(args.task7_oof)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_image_root = Path(args.local_image_root)

    case_registry["case_id"] = case_registry["case_id"].astype(str)
    manifest["case_id"] = manifest["case_id"].astype(str)
    manifest["training_case_id"] = manifest["training_case_id"].astype(str)
    task7_oof["case_id"] = task7_oof["case_id"].astype(str)

    merged = manifest.merge(
        case_registry[
            [
                "case_id",
                "source_dataset",
                "source_case_folder",
                "who_type_raw",
                "image_count",
                "image_filenames",
                "low_high_risk_group",
            ]
        ],
        on=["case_id", "source_dataset", "source_case_folder", "who_type_raw"],
        how="left",
        suffixes=("", "_case"),
    )
    merged = merged.merge(
        task7_oof[
            [
                "case_id",
                "label_idx",
                "pred_idx",
                "n_images",
                "prob_low_risk_group",
                "prob_high_risk_group",
                "fold_id",
            ]
        ],
        left_on="training_case_id",
        right_on="case_id",
        how="left",
        suffixes=("", "_oof"),
    )

    merged["local_review_path"] = merged.apply(
        lambda r: build_local_review_path(r, local_image_root), axis=1
    )
    merged["local_review_exists"] = merged["local_review_path"].apply(
        lambda p: Path(p).exists() if p else False
    )
    merged["is_high_risk_fn"] = (
        (merged["label_idx"] == 1) & (merged["pred_idx"] == 0)
    )
    merged["is_low_risk_fp"] = (
        (merged["label_idx"] == 0) & (merged["pred_idx"] == 1)
    )
    merged["is_correct"] = merged["label_idx"] == merged["pred_idx"]

    merged["priority_bucket"] = ""
    merged["priority_reason"] = ""

    multiview_mask = merged["selection_rule"] == "multi_image_use_second"
    merged.loc[multiview_mask, "priority_bucket"] = "P1_multiview"
    merged.loc[multiview_mask, "priority_reason"] = "原始多图病例，需优先判断切面优先是否合理"

    high_risk_fn_mask = merged["is_high_risk_fn"]
    merged.loc[high_risk_fn_mask & ~multiview_mask, "priority_bucket"] = "P2_high_risk_fn"
    merged.loc[
        high_risk_fn_mask & ~multiview_mask, "priority_reason"
    ] = "Task7当前主模型把高危判成低危，优先看是否与视图类型有关"

    high_risk_correct = merged[
        (merged["label_idx"] == 1)
        & (merged["pred_idx"] == 1)
        & (merged["priority_bucket"] == "")
    ].sort_values("prob_high_risk_group", ascending=False)
    low_risk_correct = merged[
        (merged["label_idx"] == 0)
        & (merged["pred_idx"] == 0)
        & (merged["priority_bucket"] == "")
    ].sort_values("prob_low_risk_group", ascending=False)

    high_take = high_risk_correct.head(args.priority_correct_per_group).index
    low_take = low_risk_correct.head(args.priority_correct_per_group).index
    merged.loc[high_take, "priority_bucket"] = "P3_high_risk_control"
    merged.loc[high_take, "priority_reason"] = "高危正确对照，帮助比较视图类型是否本身更稳定"
    merged.loc[low_take, "priority_bucket"] = "P4_low_risk_control"
    merged.loc[low_take, "priority_reason"] = "低危正确对照，帮助比较视图类型是否本身更稳定"

    full_cols = [
        "training_case_id",
        "case_id",
        "source_dataset",
        "source_case_folder",
        "who_type_raw",
        "low_high_risk_group",
        "image_count",
        "image_filenames",
        "selection_rule",
        "selected_image_name",
        "selected_image_relpath",
        "selected_image_path",
        "training_image_name",
        "training_image_path",
        "local_review_path",
        "local_review_exists",
        "fold_id",
        "label_idx",
        "pred_idx",
        "prob_low_risk_group",
        "prob_high_risk_group",
        "is_high_risk_fn",
        "is_low_risk_fp",
        "priority_bucket",
        "priority_reason",
    ]
    full = make_template(merged[full_cols])
    full.to_csv(output_dir / "task7_viewtype_full_template.csv", index=False, encoding="utf-8-sig")

    priority = full[full["priority_bucket"] != ""].copy()
    bucket_order = {
        "P1_multiview": 1,
        "P2_high_risk_fn": 2,
        "P3_high_risk_control": 3,
        "P4_low_risk_control": 4,
    }
    priority["_bucket_order"] = priority["priority_bucket"].map(bucket_order)
    priority = priority.sort_values(
        ["_bucket_order", "source_dataset", "source_case_folder", "case_id"]
    ).drop(columns="_bucket_order")
    priority.to_csv(
        output_dir / "task7_viewtype_priority_template.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_lines = [
        "# Task7视图类型标注模板说明",
        "",
        f"- 当前主图总数：{len(full)}",
        f"- 优先标注清单总数：{len(priority)}",
        f"- P1 原始多图病例：{(priority['priority_bucket'] == 'P1_multiview').sum()}",
        f"- P2 Task7高危漏诊病例：{(priority['priority_bucket'] == 'P2_high_risk_fn').sum()}",
        f"- P3 高危正确对照：{(priority['priority_bucket'] == 'P3_high_risk_control').sum()}",
        f"- P4 低危正确对照：{(priority['priority_bucket'] == 'P4_low_risk_control').sum()}",
        "",
        "建议视图标签字段：",
        "- view_type_round1: cut_surface / outer_surface / mixed / unclear",
        "- view_type_confidence: high / medium / low",
        "- cut_surface_degree: none / weak / moderate / strong",
        "- outer_surface_degree: none / weak / moderate / strong",
        "- mixed_context: yes / no",
        "- tumor_visible_degree: low / medium / high",
        "- fat_context_degree: low / medium / high",
        "- scale_visible: yes / no",
        "- is_preferred_main_view: yes / no",
        "- alternate_view_needed: yes / no",
        "",
        "优先顺序：先标多图病例，再标Task7高危漏诊病例，最后补正确对照组。",
    ]
    (output_dir / "task7_viewtype_labeling_plan.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
