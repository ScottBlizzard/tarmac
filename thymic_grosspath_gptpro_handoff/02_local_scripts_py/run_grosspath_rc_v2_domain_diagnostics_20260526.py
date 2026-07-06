from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs" / "grosspath_rc_v1_20260526"
OUT = ROOT / "outputs" / "grosspath_rc_v2_20260526"
REPORT_DIR = ROOT / "汇报"


QUALITY = ROOT / "artifacts" / "third_batch_shift_20260521" / "old_vs_third_image_quality_stats.csv"
VIEW_OLD = ROOT / "artifacts" / "third_batch_shift_20260521" / "old_viewtype_full_template.csv"
VIEW_WORK_DIR = ROOT / "汇报" / "Task7视图标签工作包_2026-05-15"
REGISTRY = ROOT / "artifacts" / "server_task567" / "combined_case_registry.csv"
THIRD_FRAME = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "44_old_third_unified_feature_cv_20260523"
    / "third_frame_with_folds.csv"
)


CORE_CONCEPTS = [
    "boundary_clear",
    "boundary_unclear",
    "capsule_any",
    "capsule_complete",
    "capsule_absent",
    "capsule_involved",
    "hemorrhage",
    "necrosis",
    "cystic_change",
    "nodular_lobulated",
    "texture_soft",
    "texture_medium",
    "texture_tough",
    "cut_surface_mentioned",
    "surface_mentioned",
    "gross_highrisk_score",
    "gross_conflict_score",
    "tumor_max_dim_mm",
]


def norm_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return "".join(ch for ch in s if ch.isdigit())


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(V1 / "v1_dev_scores.csv")
    ext = pd.read_csv(V1 / "v1_external_scores.csv")
    for df in (dev, ext):
        df["original_case_id_norm"] = df["original_case_id"].map(norm_id)
        df["main_correct"] = df["main_pred"].astype(int).eq(df["label_idx"].astype(int)).astype(int)
        df["main_wrong"] = 1 - df["main_correct"]
        df["error_type"] = np.select(
            [
                df["label_idx"].astype(int).eq(1) & df["main_pred"].astype(int).eq(0),
                df["label_idx"].astype(int).eq(0) & df["main_pred"].astype(int).eq(1),
            ],
            ["FN_high_to_low", "FP_low_to_high"],
            default="correct",
        )
    return dev, ext


def merge_development_metadata(dev: pd.DataFrame) -> pd.DataFrame:
    dev = dev.copy()

    q = pd.read_csv(QUALITY)
    q["original_case_id_norm"] = q["original_case_id"].map(norm_id)
    q = q.sort_values(["dataset", "original_case_id_norm"]).drop_duplicates("original_case_id_norm")
    q_cols = [
        "original_case_id_norm",
        "width",
        "height",
        "megapixels",
        "aspect",
        "file_kb",
        "brightness_mean",
        "contrast_std",
        "saturation_mean",
        "border_brightness",
        "border_saturation",
        "border_blue_ratio",
        "border_pale_ratio",
        "subject_area_proxy",
        "red_tissue_ratio",
        "path",
    ]
    dev = dev.merge(q[q_cols], on="original_case_id_norm", how="left")

    view = pd.read_csv(VIEW_OLD)
    view["original_case_id_norm"] = view["case_id"].map(norm_id)
    view_cols = [
        "original_case_id_norm",
        "view_type_round1",
        "view_type_confidence",
        "cut_surface_degree",
        "outer_surface_degree",
        "mixed_context",
        "tumor_visible_degree",
        "fat_context_degree",
        "scale_visible",
        "is_preferred_main_view",
        "alternate_view_needed",
    ]
    view = view[view_cols].drop_duplicates("original_case_id_norm")
    dev = dev.merge(view, on="original_case_id_norm", how="left")

    view_seed = build_partial_view_labels()
    if not view_seed.empty:
        dev = dev.merge(view_seed, on="case_id", how="left")

    reg = pd.read_csv(REGISTRY)
    reg["original_case_id_norm"] = reg["original_case_id"].map(norm_id)
    reg_cols = [
        "original_case_id_norm",
        "image_count",
        "original_image_count",
        "selection_rule",
        "selected_original_image_name",
        "selected_original_image_relpath",
        "selected_original_image_path",
        "source_case_folder",
        "who_type_raw",
    ]
    reg = reg[reg_cols].drop_duplicates("original_case_id_norm")
    dev = dev.merge(reg, on="original_case_id_norm", how="left", suffixes=("", "_registry"))

    third = pd.read_csv(THIRD_FRAME)
    third["original_case_id_norm"] = third["original_case_id"].map(norm_id)
    third = third[["original_case_id_norm", "image_path", "image_name", "source_folder"]].drop_duplicates(
        "original_case_id_norm"
    )
    dev = dev.merge(third, on="original_case_id_norm", how="left", suffixes=("", "_third"))
    dev["analysis_image_path"] = dev["path"].fillna(dev["image_path"])
    dev["analysis_source_folder"] = dev["source_case_folder"].fillna(dev["source_folder"])
    dev["view_type_final"] = dev["view_type_round1"]
    if "view_type_seed" in dev.columns:
        dev["view_type_final"] = dev["view_type_final"].fillna(dev["view_type_seed"])
    fallback_view = pd.Series(
        np.where(dev["domain"].eq("old"), "unlabeled_old", "unlabeled_third"),
        index=dev.index,
    )
    dev["view_type_final"] = dev["view_type_final"].fillna(fallback_view)
    if "view_label_source" not in dev.columns:
        dev["view_label_source"] = np.nan
    fallback_source = pd.Series(
        np.where(dev["view_type_final"].astype(str).str.startswith("unlabeled"), "missing", "legacy_template"),
        index=dev.index,
    )
    dev["view_label_source"] = dev["view_label_source"].fillna(fallback_source)

    add_bins(dev)
    return dev


def build_partial_view_labels() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    p1 = VIEW_WORK_DIR / "task7_viewtype_p1_multiview_round1.csv"
    if p1.exists():
        df = pd.read_csv(p1)
        part = df[
            [
                "training_case_id",
                "current_selected_view_type",
                "current_selected_confidence",
                "cut_surface_degree",
                "outer_surface_degree",
                "mixed_context",
                "tumor_visible_degree",
                "fat_context_degree",
            ]
        ].rename(
            columns={
                "training_case_id": "case_id",
                "current_selected_view_type": "view_type_seed",
                "current_selected_confidence": "view_type_confidence_seed",
            }
        )
        part["view_label_source"] = "p1_multiview"
        part["priority"] = 2
        parts.append(part)
    p2 = VIEW_WORK_DIR / "task7_viewtype_p2_high_risk_fn_round1.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        part = df[
            [
                "training_case_id",
                "view_type_round1",
                "view_type_confidence",
                "cut_surface_degree",
                "outer_surface_degree",
                "mixed_context",
            ]
        ].rename(
            columns={
                "training_case_id": "case_id",
                "view_type_round1": "view_type_seed",
                "view_type_confidence": "view_type_confidence_seed",
            }
        )
        part["view_label_source"] = "p2_highrisk_fn"
        part["priority"] = 3
        parts.append(part)
    seed = VIEW_WORK_DIR / "task7_viewtype_seed_labels.csv"
    if seed.exists():
        df = pd.read_csv(seed)
        part = df.rename(columns={"training_case_id": "case_id", "view_type_seed": "view_type_seed"})
        part["view_type_confidence_seed"] = np.nan
        part["cut_surface_degree"] = np.nan
        part["outer_surface_degree"] = np.nan
        part["mixed_context"] = np.nan
        part["view_label_source"] = part.get("view_label_source", "seed")
        part["priority"] = 1
        parts.append(
            part[
                [
                    "case_id",
                    "view_type_seed",
                    "view_type_confidence_seed",
                    "cut_surface_degree",
                    "outer_surface_degree",
                    "mixed_context",
                    "view_label_source",
                    "priority",
                ]
            ]
        )
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out[out["case_id"].notna() & out["view_type_seed"].notna()].copy()
    out = out.sort_values(["case_id", "priority"], ascending=[True, False]).drop_duplicates("case_id")
    keep = [
        "case_id",
        "view_type_seed",
        "view_type_confidence_seed",
        "cut_surface_degree",
        "outer_surface_degree",
        "mixed_context",
        "view_label_source",
    ]
    return out[[c for c in keep if c in out.columns]]


def add_bins(df: pd.DataFrame) -> None:
    numeric_cols = [
        "megapixels",
        "file_kb",
        "brightness_mean",
        "contrast_std",
        "saturation_mean",
        "border_blue_ratio",
        "subject_area_proxy",
        "red_tissue_ratio",
        "bbox_area_ratio",
        "fg_ratio",
        "lap_var",
        "glare_ratio",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s
        if s.notna().sum() < 8 or s.nunique(dropna=True) < 3:
            continue
        try:
            df[f"{col}_quartile"] = pd.qcut(s, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        except ValueError:
            pass
    if "manual_quality_status_v1" in df.columns:
        df["quality_group"] = df["manual_quality_status_v1"].fillna("unknown")
    else:
        df["quality_group"] = "not_labeled"
    if "image_count" in df.columns:
        df["multi_image_group"] = np.where(pd.to_numeric(df["image_count"], errors="coerce").fillna(1) > 1, "multi", "single")


def metric(df: pd.DataFrame) -> dict[str, object]:
    if len(df) == 0:
        return {
            "n": 0,
            "acc": np.nan,
            "bacc": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "fn_rate_high": np.nan,
            "fp_rate_low": np.nan,
        }
    y = df["label_idx"].astype(int).to_numpy()
    p = df["main_pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "n": int(len(df)),
        "acc": float(accuracy_score(y, p)),
        "bacc": float(balanced_accuracy_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fn_rate_high": float(fn / (fn + tp)) if fn + tp else np.nan,
        "fp_rate_low": float(fp / (fp + tn)) if fp + tn else np.nan,
    }


def subgroup_metrics(df: pd.DataFrame, group_cols: list[str], name: str, min_n: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    overall_wrong = float(df["main_wrong"].mean()) if len(df) else np.nan
    for col in group_cols:
        if col not in df.columns:
            continue
        tmp = df.copy()
        tmp[col] = tmp[col].astype("object").where(tmp[col].notna(), "missing")
        for val, sub in tmp.groupby(col, dropna=False):
            if len(sub) < min_n:
                continue
            m = metric(sub)
            wrong_rate = float(sub["main_wrong"].mean())
            rows.append(
                {
                    "analysis": name,
                    "feature": col,
                    "value": str(val),
                    "wrong_rate": wrong_rate,
                    "relative_wrong_rate": wrong_rate / overall_wrong if overall_wrong and not np.isnan(overall_wrong) else np.nan,
                    **m,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["relative_wrong_rate", "n"], ascending=[False, False])
    return out


def concept_error_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for c in CORE_CONCEPTS:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if c in ["gross_highrisk_score", "tumor_max_dim_mm"]:
            if s.notna().sum() < 20:
                continue
            try:
                groups = pd.qcut(s, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
            except ValueError:
                continue
        else:
            groups = np.where(s.fillna(0).astype(float) > 0, "present", "absent")
        tmp = df.copy()
        tmp["_concept_group"] = groups
        subm = subgroup_metrics(tmp, ["_concept_group"], c, min_n=5)
        if not subm.empty:
            subm["concept"] = c
            rows.append(subm)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["concept", "value", "n", "acc", "bacc", "wrong_rate", "relative_wrong_rate", "fn", "fp", "fn_rate_high", "fp_rate_low"]]


def prepare_training_lists(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Only use development data. These are candidate lists for future training, not external-set tuning.
    safe = dev[
        dev["main_correct"].eq(1)
        & dev["core_agree_all"].astype(int).eq(1)
        & dev["margin162"].astype(float).ge(dev["margin162"].quantile(0.55))
    ].copy()
    hard = dev[dev["main_wrong"].eq(1)].copy()
    focus_mask = dev["domain"].eq("third") | dev["view_type_final"].astype(str).str.contains(
        "outer|mixed", case=False, na=False
    )
    if "subject_area_proxy_quartile" in dev.columns:
        focus_mask = focus_mask | dev["subject_area_proxy_quartile"].astype(str).isin(["Q1_low", "Q4_high"])
    domain_focus = dev[focus_mask].copy()

    cols = [
        "case_id",
        "original_case_id",
        "domain",
        "third_split",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "main_prob",
        "main_pred",
        "main_correct",
        "error_type",
        "core_agree_all",
        "core_agree_count",
        "margin162",
        "view_type_final",
        "megapixels",
        "brightness_mean",
        "contrast_std",
        "border_blue_ratio",
        "subject_area_proxy",
        "red_tissue_ratio",
        "analysis_image_path",
    ]
    cols = [c for c in cols if c in dev.columns]
    return safe[cols], hard[cols], domain_focus[cols]


def make_plots(dev_subgroups: pd.DataFrame, concept_subgroups: pd.DataFrame) -> None:
    if not dev_subgroups.empty:
        top = dev_subgroups[dev_subgroups["n"].ge(10)].head(14).copy()
        if not top.empty:
            top["label"] = top["feature"] + "=" + top["value"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top["label"][::-1], top["wrong_rate"][::-1], color="#b65f4a")
            ax.set_xlabel("Wrong rate")
            ax.set_title("GrossPath-RC v2: highest-risk development subgroups")
            ax.set_xlim(0, max(0.5, float(top["wrong_rate"].max()) * 1.15))
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(OUT / "v2_dev_high_risk_subgroups.png", dpi=220)
            plt.close(fig)
    if not concept_subgroups.empty:
        top = concept_subgroups[concept_subgroups["n"].ge(10)].sort_values("relative_wrong_rate", ascending=False).head(14)
        if not top.empty:
            top = top.copy()
            top["label"] = top["concept"] + "=" + top["value"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top["label"][::-1], top["wrong_rate"][::-1], color="#647aa3")
            ax.set_xlabel("Wrong rate")
            ax.set_title("GrossPath-RC v2: concept-linked error enrichment")
            ax.set_xlim(0, max(0.5, float(top["wrong_rate"].max()) * 1.15))
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(OUT / "v2_concept_error_enrichment.png", dpi=220)
            plt.close(fig)


def df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "暂无可用结果。"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    dev: pd.DataFrame,
    ext: pd.DataFrame,
    dev_subgroups: pd.DataFrame,
    ext_subgroups: pd.DataFrame,
    concept_subgroups: pd.DataFrame,
    safe: pd.DataFrame,
    hard: pd.DataFrame,
    domain_focus: pd.DataFrame,
) -> None:
    overview_rows = []
    for name, df in [
        ("dev_all_old_plus_third", dev),
        ("old", dev[dev["domain"].eq("old")]),
        ("third_all", dev[dev["domain"].eq("third")]),
        ("third_holdout234", dev[dev["third_split"].eq("holdout234")]),
        ("external_strict", ext[ext["strict_task7_eval"].astype(int).eq(1)]),
        ("external_readable_auto", ext[ext["manual_quality_status_v1"].eq("pass_readable")]),
    ]:
        overview_rows.append({"group": name, **metric(df)})
    overview = pd.DataFrame(overview_rows)
    view_coverage = (
        dev.groupby(["view_type_final", "view_label_source"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    report = f"""# GrossPath-RC v2 域泛化诊断与训练清单

日期：2026-05-26

## 本轮定位

v1 已经证明：单纯 stacking 和域内可靠性 guard 不能解决严格外部泛化。v2 先不调外部集分数，而是只用旧数据+第三批开发数据，把错误和图像质量、视角、尺度、核心概念联系起来，形成下一轮训练的可执行清单。

## 主模型表现总览

{df_to_md(overview, 30)}

## 视角标签覆盖情况

当前可用的逐病例视角标签主要来自 2026-05-15 的多图/高危漏诊回看工作包，覆盖 {int((~dev['view_type_final'].astype(str).str.startswith('unlabeled')).sum())}/{len(dev)} 例。它可以作为视角辅助头的种子标签，但还不能直接支持全量 cut/outer/mixed 分层结论。

{df_to_md(view_coverage, 20)}

## 开发集错误富集子群

下面表格按相对错误率排序。`relative_wrong_rate > 1` 表示这个子群比开发集平均更容易错。

{df_to_md(dev_subgroups[dev_subgroups['n'].ge(10)], 25)}

## 概念相关错误富集

这些结果只来自开发集已有肉眼所见/经验概念，不使用外部集真值调参。

{df_to_md(concept_subgroups[concept_subgroups['n'].ge(10)].sort_values('relative_wrong_rate', ascending=False), 25)}

## 外部集仅作诊断参考

外部表只用于描述分布和风险，不用于选择训练阈值。

{df_to_md(ext_subgroups[ext_subgroups['n'].ge(5)], 20)}

## 生成的训练清单

1. `v2_training_safe_core_cases.csv`：高一致性、主模型正确、边距较大的稳定样本，共 {len(safe)} 例。可作为稳定原型/teacher anchor。
2. `v2_training_hard_error_cases.csv`：开发集主模型错误样本，共 {len(hard)} 例。用于 hard mining、复核器训练、概念冲突回看。
3. `v2_training_domain_focus_cases.csv`：第三批、视角特殊、主体尺度异常等域泛化重点样本，共 {len(domain_focus)} 例。用于颜色/尺度/背景增强和域泛化训练。

## v2 后续训练建议

1. 先做不依赖外部集的增强实验：颜色温度、亮度、饱和度、背景蓝板扰动、主体面积随机缩放、whole+crop 多尺度一致性。
2. 加入 view/quality 辅助头：视角不一定要求医生级精确，但要让 backbone 显式知道 cut、outer、mixed、主体过小、背景异常这些域因素。
3. 核心概念只保留少数高价值项：边界、包膜、结节/分叶、囊变/坏死/出血、主体尺度。概念数量少但要能被图像预测，否则不要进入主模型。
4. 训练目标从单一 CE 改成多目标：主任务分类 + 视角/质量辅助 + 概念辅助 + 多增强一致性。这样比继续堆阈值更可能提升外部泛化。
5. 外部严格集只能在模型冻结后评估一次；如果要分析外部失败原因，只能作为事后报告，不能反向调策略。
"""
    (REPORT_DIR / "2026-05-26_GrossPath-RC_v2域泛化诊断与训练清单.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    dev, ext = read_inputs()
    dev = merge_development_metadata(dev)
    add_bins(ext)

    dev_group_cols = [
        "domain",
        "third_split",
        "task_l6_label",
        "task_l7_label",
        "view_type_final",
        "multi_image_group",
        "megapixels_quartile",
        "file_kb_quartile",
        "brightness_mean_quartile",
        "contrast_std_quartile",
        "saturation_mean_quartile",
        "border_blue_ratio_quartile",
        "subject_area_proxy_quartile",
        "red_tissue_ratio_quartile",
        "core_agree_all",
        "core_agree_count",
    ]
    ext_group_cols = [
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "manual_quality_status_v1",
        "fg_ratio_quartile",
        "lap_var_quartile",
        "glare_ratio_quartile",
        "bbox_area_ratio_quartile",
        "core_agree_all",
        "core_agree_count",
    ]
    dev_subgroups = subgroup_metrics(dev, dev_group_cols, "development", min_n=5)
    ext_subgroups = subgroup_metrics(ext, ext_group_cols, "external_diagnostic_only", min_n=5)
    concept_subgroups = concept_error_table(dev)
    safe, hard, domain_focus = prepare_training_lists(dev)

    dev.to_csv(OUT / "v2_development_diagnostic_table.csv", index=False, encoding="utf-8-sig")
    ext.to_csv(OUT / "v2_external_diagnostic_table.csv", index=False, encoding="utf-8-sig")
    dev_subgroups.to_csv(OUT / "v2_development_subgroup_error_metrics.csv", index=False, encoding="utf-8-sig")
    ext_subgroups.to_csv(OUT / "v2_external_subgroup_error_metrics_diagnostic_only.csv", index=False, encoding="utf-8-sig")
    concept_subgroups.to_csv(OUT / "v2_concept_error_enrichment_dev.csv", index=False, encoding="utf-8-sig")
    safe.to_csv(OUT / "v2_training_safe_core_cases.csv", index=False, encoding="utf-8-sig")
    hard.to_csv(OUT / "v2_training_hard_error_cases.csv", index=False, encoding="utf-8-sig")
    domain_focus.to_csv(OUT / "v2_training_domain_focus_cases.csv", index=False, encoding="utf-8-sig")
    make_plots(dev_subgroups, concept_subgroups)
    write_report(dev, ext, dev_subgroups, ext_subgroups, concept_subgroups, safe, hard, domain_focus)
    summary = {
        "dev_n": int(len(dev)),
        "external_n": int(len(ext)),
        "safe_core_n": int(len(safe)),
        "hard_error_n": int(len(hard)),
        "domain_focus_n": int(len(domain_focus)),
        "outputs": {
            "dev_diag": str(OUT / "v2_development_diagnostic_table.csv"),
            "dev_subgroups": str(OUT / "v2_development_subgroup_error_metrics.csv"),
            "concept_enrichment": str(OUT / "v2_concept_error_enrichment_dev.csv"),
            "safe_core": str(OUT / "v2_training_safe_core_cases.csv"),
            "hard_errors": str(OUT / "v2_training_hard_error_cases.csv"),
            "domain_focus": str(OUT / "v2_training_domain_focus_cases.csv"),
        },
    }
    (OUT / "v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
