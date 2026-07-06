from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "grosspath_rc_v0_20260526"
EQ = ROOT / "outputs" / "external_quality_gate_20260525"


CONCEPT_GROUPS: dict[str, list[str]] = {
    "boundary_clear": ["界清", "边界清", "界限清", "界尚清", "边界尚清"],
    "boundary_unclear": ["界不清", "边界不清", "界欠清", "边界欠清", "界限不清"],
    "capsule_any": ["包膜"],
    "capsule_complete": ["包膜完整", "有包膜", "似有包膜", "可见包膜", "包膜尚完整"],
    "capsule_absent": ["未见明显包膜", "无包膜", "未见包膜"],
    "capsule_involved": ["包膜侵犯", "侵犯包膜", "突破包膜", "累及包膜"],
    "invasion": ["侵袭", "侵及", "累及", "浸润", "侵犯"],
    "fat_involved_or_attached": ["脂肪"],
    "lung_attached": ["肺"],
    "pericardium_attached": ["心包"],
    "pleura_attached": ["胸膜"],
    "hemorrhage": ["出血", "暗红", "血性", "血肿", "凝血"],
    "necrosis": ["坏死", "坏死样"],
    "cystic_change": ["囊性", "囊变", "囊腔", "囊实"],
    "calcification": ["钙化"],
    "nodular_lobulated": ["结节", "分叶", "多结节", "结节状", "结节融合"],
    "septum": ["分隔", "纤维分隔"],
    "homogeneous": ["均质", "均匀"],
    "gray_white": ["灰白"],
    "gray_yellow": ["灰黄"],
    "gray_red": ["灰红", "红色"],
    "gray_brown": ["灰褐", "褐色"],
    "gray_black": ["灰黑", "黑色"],
    "texture_soft": ["质嫩", "质软"],
    "texture_medium": ["质中"],
    "texture_tough": ["质韧", "质硬"],
    "texture_fragile": ["质脆", "易碎"],
    "cut_surface_mentioned": ["切面"],
    "surface_mentioned": ["表面", "外观"],
}

HIGH_RISK_CONCEPTS = [
    "boundary_unclear",
    "capsule_absent",
    "capsule_involved",
    "invasion",
    "fat_involved_or_attached",
    "lung_attached",
    "pericardium_attached",
    "pleura_attached",
    "necrosis",
    "hemorrhage",
    "cystic_change",
    "nodular_lobulated",
    "texture_tough",
]

LOW_RISK_CONCEPTS = [
    "boundary_clear",
    "capsule_complete",
    "texture_soft",
    "homogeneous",
]


def norm_id(x: object) -> str:
    if pd.isna(x):
        return ""
    text = str(x).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"\D", "", text)


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    text = str(x)
    text = text.replace("\u3000", " ").replace("×", "*").replace("x", "*").replace("X", "*")
    return re.sub(r"\s+", "", text)


def has_any(text: str, words: list[str]) -> int:
    return int(any(w in text for w in words))


def count_any(text: str, words: list[str]) -> int:
    return int(sum(text.count(w) for w in words))


def parse_sizes(text: str) -> dict[str, float]:
    dims_all: list[list[float]] = []
    dims_tumor: list[list[float]] = []
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)"
        r"(?:\s*\*\s*(\d+(?:\.\d+)?))?\s*(cm|mm|CM|MM)?"
    )
    for m in pattern.finditer(text):
        vals = [float(m.group(1)), float(m.group(2))]
        if m.group(3):
            vals.append(float(m.group(3)))
        unit = (m.group(4) or "mm").lower()
        factor = 10.0 if unit == "cm" else 1.0
        vals = [v * factor for v in vals]
        dims_all.append(vals)
        win = text[max(0, m.start() - 24) : min(len(text), m.end() + 16)]
        if any(k in win for k in ["肿物", "肿块", "肿瘤", "结节", "病灶", "包块"]):
            if not any(k in win for k in ["脂肪", "心包", "部分肺", "胸腺组织", "组织大小"]):
                dims_tumor.append(vals)

    def summarize(prefix: str, rows: list[list[float]]) -> dict[str, float]:
        out = {
            f"{prefix}_n_size_mentions": float(len(rows)),
            f"{prefix}_max_dim_mm": 0.0,
            f"{prefix}_max_area_mm2": 0.0,
            f"{prefix}_max_volume_mm3": 0.0,
        }
        for vals in rows:
            if not vals:
                continue
            out[f"{prefix}_max_dim_mm"] = max(out[f"{prefix}_max_dim_mm"], max(vals))
            if len(vals) >= 2:
                out[f"{prefix}_max_area_mm2"] = max(out[f"{prefix}_max_area_mm2"], vals[0] * vals[1])
            if len(vals) >= 3:
                out[f"{prefix}_max_volume_mm3"] = max(
                    out[f"{prefix}_max_volume_mm3"], vals[0] * vals[1] * vals[2]
                )
        return out

    out = summarize("all", dims_all)
    out.update(summarize("tumor", dims_tumor))
    return out


def build_concepts() -> tuple[pd.DataFrame, list[str]]:
    src = ROOT / "gross_findings_parsed_20260520.csv"
    raw = pd.read_csv(src, dtype=str)
    raw["original_case_id"] = raw["病理号"].map(norm_id)
    raw["肉眼所见_norm"] = raw["肉眼所见"].map(norm_text)
    raw["病理诊断_norm"] = raw["病理诊断"].map(norm_text)

    def join_unique(vals: pd.Series) -> str:
        seen: list[str] = []
        for v in vals.dropna().astype(str):
            v = v.strip()
            if v and v not in seen:
                seen.append(v)
        return "；".join(seen)

    grouped = (
        raw.groupby("original_case_id", dropna=False)
        .agg(
            sex=("性别", "first"),
            age=("年龄", "first"),
            gross_text=("肉眼所见_norm", join_unique),
            diagnosis_text=("病理诊断_norm", join_unique),
            n_gross_rows=("肉眼所见_norm", "size"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["original_case_id"].astype(bool)].copy()
    grouped["concept_has_gross_text"] = (grouped["gross_text"].str.len() > 0).astype(int)
    grouped["gross_text_len"] = grouped["gross_text"].str.len()

    for name, words in CONCEPT_GROUPS.items():
        grouped[name] = grouped["gross_text"].map(lambda t, ws=words: has_any(t, ws)).astype(int)
        grouped[f"{name}_count"] = grouped["gross_text"].map(lambda t, ws=words: count_any(t, ws)).astype(int)

    size_df = pd.DataFrame([parse_sizes(t) for t in grouped["gross_text"]])
    grouped = pd.concat([grouped.reset_index(drop=True), size_df], axis=1)
    grouped["age_num"] = pd.to_numeric(grouped["age"], errors="coerce")
    grouped["sex_male"] = grouped["sex"].fillna("").astype(str).str.contains("男", regex=False).astype(int)
    grouped["sex_female"] = grouped["sex"].fillna("").astype(str).str.contains("女", regex=False).astype(int)

    size_z = np.log1p(grouped["tumor_max_dim_mm"].astype(float))
    if float(size_z.std()) > 1e-8:
        size_z = (size_z - size_z.mean()) / size_z.std()
    grouped["gross_highrisk_score"] = (
        grouped[HIGH_RISK_CONCEPTS].sum(axis=1).astype(float)
        - grouped[LOW_RISK_CONCEPTS].sum(axis=1).astype(float)
        + 0.25 * size_z.fillna(0.0)
    )
    grouped["gross_conflict_score"] = (
        (grouped[HIGH_RISK_CONCEPTS].sum(axis=1) > 0).astype(int)
        + (grouped[LOW_RISK_CONCEPTS].sum(axis=1) > 0).astype(int)
        - 1
    ).clip(lower=0)

    concept_cols = (
        list(CONCEPT_GROUPS.keys())
        + ["gross_highrisk_score", "gross_conflict_score", "tumor_max_dim_mm", "tumor_max_area_mm2"]
    )
    return grouped, concept_cols


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})


def build_dev_table(concepts: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dev104 = read_csv(EQ / "dev_104_selected_guard92_oof_predictions.csv")
    dev108 = read_csv(EQ / "dev_108_selected_guard92_oof_predictions.csv")
    dev103 = read_csv(EQ / "dev_103_vitl448_tta_oof_case_predictions.csv")
    dev107 = read_csv(EQ / "dev_107_qkvb_tta_oof_case_predictions.csv")

    base = dev104[
        [
            "case_id",
            "original_case_id",
            "domain",
            "third_split",
            "fold_id",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "oof_prob_high",
            "threshold",
            "oof_pred_idx",
            "oof_correct",
        ]
    ].rename(
        columns={
            "oof_prob_high": "prob104",
            "threshold": "thr104",
            "oof_pred_idx": "pred104",
            "oof_correct": "correct104",
        }
    )
    base = base.merge(
        dev108[["case_id", "oof_prob_high", "threshold", "oof_pred_idx", "oof_correct"]].rename(
            columns={
                "oof_prob_high": "prob108",
                "threshold": "thr108",
                "oof_pred_idx": "pred108",
                "oof_correct": "correct108",
            }
        ),
        on="case_id",
        how="left",
    )
    base = base.merge(
        dev103[["case_id", "prob_high_risk_group", "pred_idx"]].rename(
            columns={"prob_high_risk_group": "prob103_vitl", "pred_idx": "pred103_vitl"}
        ),
        on="case_id",
        how="left",
    )
    base = base.merge(
        dev107[["case_id", "prob_high_risk_group", "pred_idx"]].rename(
            columns={"prob_high_risk_group": "prob107_qkvb", "pred_idx": "pred107_qkvb"}
        ),
        on="case_id",
        how="left",
    )
    base["prob162_blend"] = 0.2 * base["prob104"].astype(float) + 0.8 * base["prob108"].astype(float)
    base["thr162_blend"] = 0.595
    base["pred162_blend"] = (base["prob162_blend"] >= base["thr162_blend"]).astype(int)
    base["prob_mean_core"] = base[["prob162_blend", "prob103_vitl", "prob107_qkvb"]].astype(float).mean(axis=1)

    base["original_case_id"] = base["original_case_id"].map(norm_id)
    merged = base.merge(concepts, on="original_case_id", how="left", suffixes=("", "_gross"))
    merged["concept_matched"] = merged["concept_has_gross_text"].fillna(0).astype(int)
    for c in concepts.columns:
        if c not in ["original_case_id", "sex", "age", "gross_text", "diagnosis_text"]:
            if c in merged.columns and pd.api.types.is_numeric_dtype(merged[c]):
                merged[c] = merged[c].fillna(0)
    return merged, ["prob162_blend", "prob103_vitl", "prob107_qkvb"]


def build_external_table(concepts: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ext = read_csv(EQ / "external_quality_predictions_merged_manual_v1.csv")
    run163 = read_csv(EQ / "run163_vitl448_tta_predictions.csv")
    run109 = read_csv(EQ / "run109_qkvb_tta_predictions.csv")
    run172 = read_csv(EQ / "external_run172_externalmimic_tta_predictions.csv")

    base = ext[
        [
            "case_id",
            "original_case_id",
            "source_folder",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "strict_task7_eval",
            "image_name",
            "locked162_blend_prob_high",
            "locked162_blend_threshold",
            "locked162_blend_pred_idx",
            "locked162_blend_correct",
            "quality_status",
            "quality_score",
            "manual_quality_status_v1",
            "fg_ratio",
            "lap_var",
            "glare_ratio",
            "bbox_area_ratio",
        ]
    ].rename(
        columns={
            "locked162_blend_prob_high": "prob162_blend",
            "locked162_blend_threshold": "thr162_blend",
            "locked162_blend_pred_idx": "pred162_blend",
            "locked162_blend_correct": "correct162_blend",
        }
    )
    base = base.merge(
        run163[["case_id", "prob_high_risk_group", "pred_idx"]].rename(
            columns={"prob_high_risk_group": "prob103_vitl", "pred_idx": "pred103_vitl"}
        ),
        on="case_id",
        how="left",
    )
    base = base.merge(
        run109[["case_id", "prob_high_risk_group", "pred_idx"]].rename(
            columns={"prob_high_risk_group": "prob107_qkvb", "pred_idx": "pred107_qkvb"}
        ),
        on="case_id",
        how="left",
    )
    base = base.merge(
        run172[["case_id", "prob_high_risk_group", "pred_idx"]].rename(
            columns={"prob_high_risk_group": "prob172_external_mimic", "pred_idx": "pred172_external_mimic"}
        ),
        on="case_id",
        how="left",
    )
    base["prob_mean_core"] = base[["prob162_blend", "prob103_vitl", "prob107_qkvb"]].astype(float).mean(axis=1)
    base["original_case_id"] = base["original_case_id"].map(norm_id)
    merged = base.merge(concepts, on="original_case_id", how="left", suffixes=("", "_gross"))
    merged["concept_matched"] = merged["concept_has_gross_text"].fillna(0).astype(int)
    for c in concepts.columns:
        if c not in ["original_case_id", "sex", "age", "gross_text", "diagnosis_text"]:
            if c in merged.columns and pd.api.types.is_numeric_dtype(merged[c]):
                merged[c] = merged[c].fillna(0)
    return merged, ["prob162_blend", "prob103_vitl", "prob107_qkvb"]


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def metric_dict(name: str, y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": safe_auc(y, prob),
        "sensitivity_high": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity_low": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def score_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.1, 0.9, 161):
        pred = (prob >= t).astype(int)
        s = balanced_accuracy_score(y, pred)
        if (s, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t = float(t)
            best_s = float(s)
    return best_t, best_s


def add_behavior_features(df: pd.DataFrame, prob_cols: list[str], mean_threshold: float) -> pd.DataFrame:
    out = df.copy()
    probs = out[prob_cols].astype(float).to_numpy()
    out["core_prob_std"] = probs.std(axis=1)
    out["core_prob_range"] = probs.max(axis=1) - probs.min(axis=1)
    out["margin162"] = (out["prob162_blend"].astype(float) - 0.595).abs()
    out["margin_mean_core"] = (out["prob_mean_core"].astype(float) - mean_threshold).abs()
    pred_cols = []
    for col in prob_cols:
        pcol = f"pred_for_{col}"
        threshold = 0.595 if col == "prob162_blend" else 0.5
        out[pcol] = (out[col].astype(float) >= threshold).astype(int)
        pred_cols.append(pcol)
    preds = out[pred_cols].to_numpy(dtype=int)
    out["core_agree_all"] = (preds.max(axis=1) == preds.min(axis=1)).astype(int)
    out["core_agree_count"] = np.maximum(preds.sum(axis=1), len(pred_cols) - preds.sum(axis=1))
    out["reliability_simple"] = out["margin_mean_core"] - out["core_prob_std"]
    out["base_pred_idx"] = out["pred162_blend"].astype(int)
    out["reliability_base_margin"] = out["margin162"]
    out["reliability_base_agreement"] = out["margin162"] + 0.15 * out["core_agree_all"] - out["core_prob_std"]
    out["concept_highrisk_z"] = out["gross_highrisk_score"].astype(float)
    if out["concept_highrisk_z"].std() > 1e-8:
        out["concept_highrisk_z"] = (out["concept_highrisk_z"] - out["concept_highrisk_z"].mean()) / out[
            "concept_highrisk_z"
        ].std()
    out["mean_pred_idx"] = (out["prob_mean_core"].astype(float) >= mean_threshold).astype(int)
    out["concept_direction"] = (out["concept_highrisk_z"] > 0).astype(int)
    out["concept_conflicts_mean_pred"] = (
        (out["concept_matched"].astype(int) == 1) & (out["concept_direction"] != out["mean_pred_idx"])
    ).astype(int)
    return out


def eval_by_groups(df: pd.DataFrame, prob_col: str, threshold: float, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, mask in groups.items():
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        rows.append(metric_dict(f"{prob_col}__{name}", sub["label_idx"].to_numpy(int), sub[prob_col].to_numpy(float), threshold))
    return pd.DataFrame(rows)


def concept_cv(dev: pd.DataFrame, concept_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = dev["label_idx"].to_numpy(int)
    folds = sorted(dev["fold_id"].dropna().astype(int).unique())
    feature_sets = {
        "concept_oracle": concept_cols,
        "model_probs": ["prob162_blend", "prob103_vitl", "prob107_qkvb"],
        "model_plus_concepts_oracle": ["prob162_blend", "prob103_vitl", "prob107_qkvb"] + concept_cols,
    }
    for name, cols in feature_sets.items():
        oof = np.zeros(len(dev), dtype=float)
        for fold in folds:
            train = dev["fold_id"].astype(int).to_numpy() != fold
            test = ~train
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            )
            clf.fit(dev.loc[train, cols].astype(float), y[train])
            oof[test] = clf.predict_proba(dev.loc[test, cols].astype(float))[:, 1]
        t, _ = score_threshold(y, oof)
        rows.append(metric_dict(name, y, oof, t))
    return pd.DataFrame(rows)


def error_router_cv(dev: pd.DataFrame, concept_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_label = dev["label_idx"].to_numpy(int)
    forced_pred = dev["base_pred_idx"].to_numpy(int)
    y_error = (forced_pred != y_label).astype(int)
    folds = sorted(dev["fold_id"].dropna().astype(int).unique())
    feature_sets = {
        "uncertainty_disagreement": [
            "prob162_blend",
            "prob103_vitl",
            "prob107_qkvb",
            "prob_mean_core",
            "core_prob_std",
            "core_prob_range",
            "margin162",
            "margin_mean_core",
            "core_agree_all",
            "core_agree_count",
        ],
        "uncertainty_disagreement_plus_concepts": [
            "prob162_blend",
            "prob103_vitl",
            "prob107_qkvb",
            "prob_mean_core",
            "core_prob_std",
            "core_prob_range",
            "margin162",
            "margin_mean_core",
            "core_agree_all",
            "core_agree_count",
        ]
        + concept_cols
        + ["concept_matched", "concept_conflicts_mean_pred"],
    }
    rows = []
    pred_table = dev[["case_id", "original_case_id", "label_idx", "fold_id", "base_pred_idx", "mean_pred_idx"]].copy()
    for name, cols in feature_sets.items():
        oof = np.zeros(len(dev), dtype=float)
        for fold in folds:
            train = dev["fold_id"].astype(int).to_numpy() != fold
            test = ~train
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            )
            clf.fit(dev.loc[train, cols].astype(float), y_error[train])
            oof[test] = clf.predict_proba(dev.loc[test, cols].astype(float))[:, 1]
        pred_table[f"error_risk_{name}"] = oof
        rows.append(
            {
                "router": name,
                "n": int(len(dev)),
                "error_rate": float(y_error.mean()),
                "error_auc": safe_auc(y_error, oof),
                "error_ap": float(average_precision_score(y_error, oof)) if len(np.unique(y_error)) > 1 else float("nan"),
                "top20pct_error_enrichment": float(y_error[np.argsort(-oof)[: max(1, int(round(0.2 * len(oof))))]].mean() / max(y_error.mean(), 1e-9)),
            }
        )
    return pd.DataFrame(rows), pred_table


def risk_coverage_curve(
    df: pd.DataFrame,
    pred_col: str,
    reliability_col: str,
    group_name: str,
    coverage_points: list[float] | None = None,
) -> pd.DataFrame:
    if coverage_points is None:
        coverage_points = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    rows = []
    y = df["label_idx"].to_numpy(int)
    pred = df[pred_col].to_numpy(int)
    order = np.argsort(-df[reliability_col].to_numpy(float))
    for cov in coverage_points:
        k = max(1, int(round(cov * len(df))))
        idx = order[:k]
        yy = y[idx]
        pp = pred[idx]
        tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
        review_idx = order[k:]
        review_error_rate = float((pred[review_idx] != y[review_idx]).mean()) if len(review_idx) else float("nan")
        rows.append(
            {
                "group": group_name,
                "policy": reliability_col,
                "coverage": float(k / len(df)),
                "auto_n": int(k),
                "review_n": int(len(df) - k),
                "auto_accuracy": float(accuracy_score(yy, pp)),
                "auto_bacc": float(balanced_accuracy_score(yy, pp)) if len(np.unique(yy)) == 2 else float("nan"),
                "auto_sensitivity_high": float(tp / (tp + fn)) if tp + fn else float("nan"),
                "auto_specificity_low": float(tn / (tn + fp)) if tn + fp else float("nan"),
                "auto_low_high_miss_rate": float(fn / (tn + fn)) if (tn + fn) else float("nan"),
                "review_error_rate": review_error_rate,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def consensus_policy_metrics(df: pd.DataFrame, group_masks: dict[str, pd.Series]) -> pd.DataFrame:
    policies = {
        "auto_if_162_103_agree": df["pred_for_prob162_blend"].eq(df["pred_for_prob103_vitl"]),
        "auto_if_162_107_agree": df["pred_for_prob162_blend"].eq(df["pred_for_prob107_qkvb"]),
        "auto_if_103_107_agree": df["pred_for_prob103_vitl"].eq(df["pred_for_prob107_qkvb"]),
        "auto_if_all3_agree": df[["pred_for_prob162_blend", "pred_for_prob103_vitl", "pred_for_prob107_qkvb"]]
        .nunique(axis=1)
        .eq(1),
    }
    rows = []
    for group, gmask in group_masks.items():
        gdf = df[gmask].copy()
        if gdf.empty:
            continue
        for policy, pmask in policies.items():
            sub = gdf[pmask.loc[gdf.index]].copy()
            review = gdf[~pmask.loc[gdf.index]].copy()
            if sub.empty:
                continue
            y = sub["label_idx"].to_numpy(int)
            pred = sub["base_pred_idx"].to_numpy(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            review_error_rate = (
                float((review["base_pred_idx"].to_numpy(int) != review["label_idx"].to_numpy(int)).mean())
                if len(review)
                else float("nan")
            )
            rows.append(
                {
                    "group": group,
                    "policy": policy,
                    "coverage": float(len(sub) / len(gdf)),
                    "auto_n": int(len(sub)),
                    "review_n": int(len(gdf) - len(sub)),
                    "auto_accuracy": float(accuracy_score(y, pred)),
                    "auto_bacc": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan"),
                    "auto_sensitivity_high": float(tp / (tp + fn)) if tp + fn else float("nan"),
                    "auto_specificity_low": float(tn / (tn + fp)) if tn + fp else float("nan"),
                    "review_error_rate": review_error_rate,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )
    return pd.DataFrame(rows)


def train_router_apply_external(
    dev: pd.DataFrame, external: pd.DataFrame, concept_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_error = (dev["base_pred_idx"].to_numpy(int) != dev["label_idx"].to_numpy(int)).astype(int)
    feature_sets = {
        "uncertainty_disagreement": [
            "prob162_blend",
            "prob103_vitl",
            "prob107_qkvb",
            "prob_mean_core",
            "core_prob_std",
            "core_prob_range",
            "margin162",
            "margin_mean_core",
            "core_agree_all",
            "core_agree_count",
        ],
        "uncertainty_disagreement_plus_concepts": [
            "prob162_blend",
            "prob103_vitl",
            "prob107_qkvb",
            "prob_mean_core",
            "core_prob_std",
            "core_prob_range",
            "margin162",
            "margin_mean_core",
            "core_agree_all",
            "core_agree_count",
        ]
        + concept_cols
        + ["concept_matched", "concept_conflicts_mean_pred"],
    }
    out = external.copy()
    rows = []
    for name, cols in feature_sets.items():
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        )
        clf.fit(dev[cols].astype(float), y_error)
        risk = clf.predict_proba(external[cols].astype(float))[:, 1]
        out[f"error_risk_{name}"] = risk
        y_ext_error = (external["base_pred_idx"].to_numpy(int) != external["label_idx"].to_numpy(int)).astype(int)
        rows.append(
            {
                "router": name,
                "external_error_rate": float(y_ext_error.mean()),
                "external_error_auc": safe_auc(y_ext_error, risk),
                "external_error_ap": float(average_precision_score(y_ext_error, risk))
                if len(np.unique(y_ext_error)) > 1
                else float("nan"),
            }
        )
    return pd.DataFrame(rows), out


def write_concept_dictionary(path: Path) -> None:
    lines = ["# Gross Concept Dictionary v1", ""]
    for name, words in CONCEPT_GROUPS.items():
        lines.append(f"## {name}")
        lines.append("关键词：" + "、".join(words))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_空表_"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    concepts, concept_cols = build_concepts()
    concepts.to_csv(OUT / "gross_concepts_v1.csv", index=False, encoding="utf-8-sig")
    write_concept_dictionary(OUT / "gross_concept_dictionary_v1.md")

    dev, prob_cols = build_dev_table(concepts)
    ext, ext_prob_cols = build_external_table(concepts)
    y_dev = dev["label_idx"].to_numpy(int)
    mean_thr, mean_dev_bacc = score_threshold(y_dev, dev["prob_mean_core"].to_numpy(float))

    dev = add_behavior_features(dev, prob_cols, mean_thr)
    ext = add_behavior_features(ext, ext_prob_cols, mean_thr)
    dev.to_csv(OUT / "dev_model_behavior_table.csv", index=False, encoding="utf-8-sig")
    ext.to_csv(OUT / "external_model_behavior_table.csv", index=False, encoding="utf-8-sig")

    dev_groups = {
        "dev_all_old_plus_third": pd.Series(True, index=dev.index),
        "old": dev["domain"].eq("old"),
        "third_all": dev["domain"].eq("third"),
        "third_holdout234": dev["third_split"].eq("holdout234"),
        "concept_matched": dev["concept_matched"].eq(1),
    }
    ext_groups = {
        "external_all": pd.Series(True, index=ext.index),
        "external_strict": ext["strict_task7_eval"].astype(int).eq(1),
        "external_readable_auto": ext["manual_quality_status_v1"].eq("pass_readable"),
        "concept_matched": ext["concept_matched"].eq(1),
    }

    perf_rows = []
    for col, thr in [
        ("prob162_blend", 0.595),
        ("prob103_vitl", 0.5),
        ("prob107_qkvb", 0.5),
        ("prob_mean_core", mean_thr),
    ]:
        perf_rows.append(eval_by_groups(dev, col, thr, dev_groups))
    dev_perf = pd.concat(perf_rows, ignore_index=True)
    dev_perf.to_csv(OUT / "dev_forced_classification_metrics.csv", index=False, encoding="utf-8-sig")

    ext_perf_rows = []
    for col, thr in [
        ("prob162_blend", 0.595),
        ("prob103_vitl", 0.5),
        ("prob107_qkvb", 0.5),
        ("prob_mean_core", mean_thr),
    ]:
        ext_perf_rows.append(eval_by_groups(ext, col, thr, ext_groups))
    ext_perf = pd.concat(ext_perf_rows, ignore_index=True)
    ext_perf.to_csv(OUT / "external_forced_classification_metrics.csv", index=False, encoding="utf-8-sig")

    concept_numeric_cols = [
        c
        for c in concept_cols
        if c in dev.columns and pd.api.types.is_numeric_dtype(dev[c])
    ]
    concept_metrics = concept_cv(dev, concept_numeric_cols)
    concept_metrics.to_csv(OUT / "concept_oracle_cv_metrics.csv", index=False, encoding="utf-8-sig")

    router_metrics, router_dev_pred = error_router_cv(dev, concept_numeric_cols)
    router_metrics.to_csv(OUT / "router_error_detection_metrics_dev.csv", index=False, encoding="utf-8-sig")
    router_dev_pred.to_csv(OUT / "router_error_risk_oof_dev.csv", index=False, encoding="utf-8-sig")

    external_router_metrics, ext_scored = train_router_apply_external(dev, ext, concept_numeric_cols)
    external_router_metrics.to_csv(OUT / "router_error_detection_metrics_external_stress.csv", index=False, encoding="utf-8-sig")
    ext_scored.to_csv(OUT / "external_model_behavior_with_router_scores.csv", index=False, encoding="utf-8-sig")

    curves = []
    for group_name, mask in dev_groups.items():
        sub = dev[mask].copy()
        if len(sub):
            curves.append(risk_coverage_curve(sub, "base_pred_idx", "reliability_base_margin", group_name))
            curves.append(risk_coverage_curve(sub, "base_pred_idx", "reliability_base_agreement", group_name))
            sub2 = sub.merge(router_dev_pred[["case_id", "error_risk_uncertainty_disagreement_plus_concepts"]], on="case_id")
            sub2["router_reliability_plus_concepts"] = -sub2["error_risk_uncertainty_disagreement_plus_concepts"]
            curves.append(risk_coverage_curve(sub2, "base_pred_idx", "router_reliability_plus_concepts", group_name))
    dev_curve = pd.concat(curves, ignore_index=True)
    dev_curve.to_csv(OUT / "risk_coverage_curve_dev.csv", index=False, encoding="utf-8-sig")

    curves = []
    for group_name, mask in ext_groups.items():
        sub = ext_scored[mask].copy()
        if len(sub):
            curves.append(risk_coverage_curve(sub, "base_pred_idx", "reliability_base_margin", group_name))
            curves.append(risk_coverage_curve(sub, "base_pred_idx", "reliability_base_agreement", group_name))
            sub["router_reliability_plus_concepts"] = -sub["error_risk_uncertainty_disagreement_plus_concepts"]
            curves.append(risk_coverage_curve(sub, "base_pred_idx", "router_reliability_plus_concepts", group_name))
    ext_curve = pd.concat(curves, ignore_index=True)
    ext_curve.to_csv(OUT / "risk_coverage_curve_external_stress.csv", index=False, encoding="utf-8-sig")

    dev_consensus = consensus_policy_metrics(dev, dev_groups)
    ext_consensus = consensus_policy_metrics(ext, ext_groups)
    dev_consensus.to_csv(OUT / "consensus_policy_metrics_dev.csv", index=False, encoding="utf-8-sig")
    ext_consensus.to_csv(OUT / "consensus_policy_metrics_external_stress.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(OUT),
        "n_gross_concept_cases": int(len(concepts)),
        "dev_rows": int(len(dev)),
        "external_rows": int(len(ext)),
        "dev_concept_match_rate": float(dev["concept_matched"].mean()),
        "external_concept_match_rate": float(ext["concept_matched"].mean()),
        "mean_core_threshold_selected_on_dev_bacc": mean_thr,
        "mean_core_dev_bacc": mean_dev_bacc,
        "best_dev_forced": dev_perf.sort_values("balanced_accuracy", ascending=False).head(5).to_dict("records"),
        "best_external_forced": ext_perf.sort_values("balanced_accuracy", ascending=False).head(5).to_dict("records"),
        "concept_oracle_metrics": concept_metrics.to_dict("records"),
        "router_dev_metrics": router_metrics.to_dict("records"),
        "router_external_stress_metrics": external_router_metrics.to_dict("records"),
        "best_dev_consensus": dev_consensus.sort_values("auto_bacc", ascending=False).head(8).to_dict("records"),
        "best_external_consensus": ext_consensus.sort_values("auto_bacc", ascending=False).head(8).to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# GrossPath-RC v0 实验小结",
        "",
        f"输出目录：`{OUT}`",
        "",
        "## 数据覆盖",
        "",
        f"- 结构化肉眼所见病例数：{len(concepts)}",
        f"- 开发集病例数：{len(dev)}，概念匹配率：{dev['concept_matched'].mean():.3f}",
        f"- 外部压力测试病例数：{len(ext)}，概念匹配率：{ext['concept_matched'].mean():.3f}",
        f"- `prob_mean_core` 阈值只在开发集选择：{mean_thr:.3f}，开发集 BAcc：{mean_dev_bacc:.4f}",
        "",
        "## 强制分类最佳结果",
        "",
        df_to_md(dev_perf.sort_values("balanced_accuracy", ascending=False), max_rows=8),
        "",
        "## 外部压力测试强制分类最佳结果",
        "",
        df_to_md(ext_perf.sort_values("balanced_accuracy", ascending=False), max_rows=8),
        "",
        "## 概念信号与模型融合",
        "",
        df_to_md(concept_metrics),
        "",
        "## 错误/复核识别",
        "",
        df_to_md(router_metrics),
        "",
        "## 外部压力测试上的错误风险识别",
        "",
        df_to_md(external_router_metrics),
        "",
        "## Consensus 自动放行策略",
        "",
        "开发集：",
        "",
        df_to_md(dev_consensus.sort_values("auto_bacc", ascending=False), max_rows=10),
        "",
        "外部压力测试：",
        "",
        df_to_md(ext_consensus.sort_values("auto_bacc", ascending=False), max_rows=10),
        "",
        "## 说明",
        "",
        "- `concept_oracle` 使用医生肉眼所见结构化概念，属于上限/机制验证，不是部署时直接输入。",
        "- 外部集结果只作为已暴露压力测试，不用于阈值回调。",
        "- v0 的重点是判断概念、分歧、不确定性是否能支撑风险控制主线。",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
