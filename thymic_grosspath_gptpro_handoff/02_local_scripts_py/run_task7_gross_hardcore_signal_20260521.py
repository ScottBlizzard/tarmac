# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether doctors' gross findings can identify hard-core cases and improve Task7 routing."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--split-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv",
    )
    parser.add_argument(
        "--curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--review-score-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--best41-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/63_gross_hardcore_signal_fixed_20260521",
    )
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "ap": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_recall_fscore_support(y, pred, zero_division=0)[0][1]),
        "recall": float(precision_recall_fscore_support(y, pred, zero_division=0)[1][1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u3000", " ").replace("×", "*").replace("x", "*").replace("X", "*").replace("脳", "*")
    return re.sub(r"\s+", "", text)


def has_any(text: str, words: list[str]) -> float:
    return float(any(word in text for word in words))


def count_any(text: str, words: list[str]) -> float:
    return float(sum(text.count(word) for word in words))


def parse_size(text: str) -> dict[str, float]:
    all_dims: list[list[float]] = []
    tumor_dims: list[list[float]] = []
    size_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)"
        r"(?:\s*\*\s*(\d+(?:\.\d+)?))?\s*(cm|mm|CM|MM)?"
    )
    for match in size_pattern.finditer(text):
        dims = [float(match.group(1)), float(match.group(2))]
        if match.group(3):
            dims.append(float(match.group(3)))
        unit = (match.group(4) or "mm").lower()
        factor = 10.0 if unit == "cm" else 1.0
        dims = [dim * factor for dim in dims]
        all_dims.append(dims)
        window = text[max(0, match.start() - 20) : min(len(text), match.end() + 16)]
        if any(key in window for key in ["肿物", "肿块", "肿瘤", "结节", "病灶", "包块"]):
            if not any(key in window for key in ["脂肪", "心包", "部分肺", "胸腺组织", "组织大小"]):
                tumor_dims.append(dims)

    def summarize(prefix: str, dims_list: list[list[float]]) -> dict[str, float]:
        out = {
            f"{prefix}_n_size_mentions": float(len(dims_list)),
            f"{prefix}_max_dim_mm": 0.0,
            f"{prefix}_max_area_mm2": 0.0,
            f"{prefix}_max_volume_mm3": 0.0,
        }
        for dims in dims_list:
            if not dims:
                continue
            out[f"{prefix}_max_dim_mm"] = max(out[f"{prefix}_max_dim_mm"], max(dims))
            if len(dims) >= 2:
                out[f"{prefix}_max_area_mm2"] = max(out[f"{prefix}_max_area_mm2"], dims[0] * dims[1])
            if len(dims) >= 3:
                out[f"{prefix}_max_volume_mm3"] = max(
                    out[f"{prefix}_max_volume_mm3"], dims[0] * dims[1] * dims[2]
                )
        return out

    result = summarize("all", all_dims)
    result.update(summarize("tumor", tumor_dims))
    return result


def find_col(df: pd.DataFrame, wanted: str, fallback_contains: list[str] | None = None) -> str:
    for col in df.columns:
        if str(col).strip() == wanted:
            return str(col)
    if fallback_contains:
        for col in df.columns:
            name = str(col)
            if all(part in name for part in fallback_contains):
                return str(col)
    raise KeyError(f"Missing column: {wanted}")


def build_hand_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    text_col = find_col(df, "肉眼所见", ["肉眼", "所见"])
    age_col = find_col(df, "年龄", ["龄"])
    sex_col = find_col(df, "性别", ["性", "别"])
    texts = df[text_col].map(norm_text)
    feat = pd.DataFrame(index=df.index)

    age = pd.to_numeric(df[age_col], errors="coerce")
    feat["age"] = age.fillna(age.median()).astype(float)
    sex = df[sex_col].fillna("").astype(str)
    feat["sex_male"] = sex.str.contains("男", regex=False).astype(float)
    feat["sex_female"] = sex.str.contains("女", regex=False).astype(float)
    feat["gross_text_len"] = texts.str.len().astype(float)
    feat["gross_has_text"] = (texts.str.len() > 0).astype(float)
    feat = pd.concat([feat, pd.DataFrame([parse_size(text) for text in texts], index=df.index)], axis=1)

    groups: dict[str, list[str]] = {
        "boundary_clear": ["界清", "界限清", "边界清", "界尚清", "边界尚清"],
        "boundary_unclear": ["界不清", "边界不清", "界欠清", "边界欠清", "界限不清"],
        "capsule_any": ["包膜"],
        "capsule_complete": ["包膜完整", "有包膜", "似有包膜", "可见包膜", "包膜尚完整"],
        "capsule_absent": ["未见明显包膜", "无包膜", "未见包膜"],
        "capsule_involved": ["包膜侵犯", "侵犯包膜", "突破包膜", "累及包膜"],
        "invasion": ["侵犯", "累及", "侵及", "浸润"],
        "adhesion": ["粘连", "紧贴", "相连"],
        "fat_attached": ["脂肪"],
        "lung_attached": ["肺"],
        "pericardium_attached": ["心包"],
        "pleura_attached": ["胸膜"],
        "hemorrhage": ["出血", "暗红", "血性", "血肿"],
        "necrosis": ["坏死"],
        "cystic": ["囊性", "囊变", "囊腔", "囊实"],
        "calcification": ["钙化"],
        "lobulated": ["分叶", "结节状", "多结节", "结节融合"],
        "septum": ["分隔", "纤维分隔"],
        "gray_white": ["灰白"],
        "gray_yellow": ["灰黄"],
        "gray_red": ["灰红", "红色"],
        "gray_brown": ["灰褐", "褐色"],
        "gray_black": ["灰黑", "黑色"],
        "texture_tender": ["质嫩", "质软"],
        "texture_medium": ["质中"],
        "texture_tough": ["质韧", "质硬"],
        "texture_fragile": ["质脆", "易碎"],
        "slice_mentioned": ["切面"],
        "surface_mentioned": ["表面", "外观"],
        "multi_nodule": ["多发", "多结节", "多个结节"],
    }
    for name, words in groups.items():
        feat[f"kw_{name}"] = texts.map(lambda text, ws=words: has_any(text, ws)).astype(float)
        feat[f"cnt_{name}"] = texts.map(lambda text, ws=words: count_any(text, ws)).astype(float)

    for col in list(feat.columns):
        if col.startswith(("all_", "tumor_", "cnt_")) or col == "gross_text_len":
            feat[f"log1p_{col}"] = np.log1p(feat[col].astype(float))

    log_size = np.log1p(feat["tumor_max_dim_mm"].astype(float))
    log_area = np.log1p(feat["tumor_max_area_mm2"].astype(float))
    size_z = (log_size - log_size.mean()) / max(float(log_size.std()), 1e-8)
    area_z = (log_area - log_area.mean()) / max(float(log_area.std()), 1e-8)
    feat["manual_highrisk_score"] = (
        1.5 * feat["kw_boundary_unclear"]
        + 1.4 * feat["kw_capsule_absent"]
        + 1.4 * feat["kw_capsule_involved"]
        + 1.2 * feat["kw_invasion"]
        + 0.8 * feat["kw_lung_attached"]
        + 0.8 * feat["kw_pericardium_attached"]
        + 0.5 * feat["kw_necrosis"]
        + 0.35 * feat["kw_hemorrhage"]
        + 0.25 * feat["kw_lobulated"]
        + 0.25 * size_z.fillna(0.0)
        + 0.15 * area_z.fillna(0.0)
        - 1.3 * feat["kw_boundary_clear"]
        - 1.0 * feat["kw_capsule_complete"]
        - 0.25 * feat["kw_texture_tender"]
    )
    feat["manual_lowrisk_score"] = (
        1.2 * feat["kw_boundary_clear"]
        + 1.0 * feat["kw_capsule_complete"]
        + 0.35 * feat["kw_texture_tender"]
        - 0.9 * feat["kw_boundary_unclear"]
        - 0.9 * feat["kw_capsule_absent"]
        - 0.9 * feat["kw_invasion"]
        - 0.4 * feat["kw_necrosis"]
    )
    feat["manual_conflict_score"] = np.abs(feat["manual_highrisk_score"] - feat["manual_lowrisk_score"])
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0), texts


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def load_frame(project_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(project_root / args.registry_csv, dtype={"case_id": str, "original_case_id": str})
    split = pd.read_csv(project_root / args.split_csv, dtype={"case_id": str})
    split = split[["case_id", "master_fold_id"]].rename(columns={"master_fold_id": "fold_id"})
    curriculum = pd.read_csv(project_root / args.curriculum_csv, dtype={"case_id": str})
    curriculum = curriculum[["case_id", "difficulty", "difficulty_fine", "correct_count", "mean_true_prob", "mean_margin"]]
    best41 = pd.read_csv(project_root / args.best41_csv, dtype={"case_id": str})
    best41 = best41[
        ["case_id", "final_prob_high", "final_pred", "pred_upper", "p_upper", "base_wrong", "final_wrong"]
    ].rename(
        columns={
            "final_prob_high": "p_best41",
            "final_pred": "pred_best41",
            "pred_upper": "pred_upper",
            "p_upper": "p_upper",
        }
    )
    review = pd.read_csv(project_root / args.review_score_csv, dtype={"case_id": str})
    keep_review = ["case_id"] + [
        col
        for col in review.columns
        if col.startswith("p_") or col.startswith("pred_") or col.startswith("review_score_")
    ]
    review = review[keep_review].copy()

    df = registry.merge(split, on="case_id", how="inner")
    df = df.merge(curriculum, on="case_id", how="left").merge(best41, on="case_id", how="left")
    df = df.merge(review, on="case_id", how="left", suffixes=("", "_review"))
    df = df[df["task_l7_label"].isin(["low_risk_group", "high_risk_group"])].reset_index(drop=True)
    df["fold_id"] = df["fold_id"].astype(int)
    df["label_idx"] = (df["task_l7_label"] == "high_risk_group").astype(int)
    df["hard_core"] = (df["difficulty_fine"] == "hard_core").astype(int)
    df["best41_correct"] = (df["pred_best41"].astype(int) == df["label_idx"].astype(int)).astype(int)
    df["best41_wrong"] = 1 - df["best41_correct"]
    df["best41_fn"] = ((df["label_idx"].astype(int) == 1) & (df["pred_best41"].astype(int) == 0)).astype(int)
    df["best41_fp"] = ((df["label_idx"].astype(int) == 0) & (df["pred_best41"].astype(int) == 1)).astype(int)
    df["best41_conf"] = np.maximum(df["p_best41"].astype(float), 1.0 - df["p_best41"].astype(float))
    hand, texts = build_hand_features(df)

    model_cols = [
        col
        for col in df.columns
        if (
            col.startswith("p_")
            or col.startswith("pred_")
            or col.startswith("review_score_")
            or col in ["image_count", "best41_conf"]
        )
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    model = df[model_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p = df["p_best41"].astype(float).to_numpy()
    model["logit_best41"] = logit(p)
    model["best41_margin"] = np.abs(p - 0.5)
    model["gross_high_minus_model_logit"] = hand["manual_highrisk_score"].to_numpy() - logit(p)
    model["gross_low_minus_model_lowlogit"] = hand["manual_lowrisk_score"].to_numpy() + logit(p)
    model = pd.concat([model, pd.get_dummies(df[["selection_rule", "difficulty"]].fillna(""), dtype=float)], axis=1)
    return df, texts, hand, model


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    use_text: bool
    use_hand: bool
    use_model: bool
    ngram_range: tuple[int, int] = (2, 4)
    min_df: int = 1


def build_fold_features(
    cfg: FeatureConfig,
    train_text: pd.Series,
    test_text: pd.Series,
    train_hand: pd.DataFrame,
    test_hand: pd.DataFrame,
    train_model: pd.DataFrame,
    test_model: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, object]]:
    train_parts: list[sparse.csr_matrix] = []
    test_parts: list[sparse.csr_matrix] = []
    detail: dict[str, object] = {}
    if cfg.use_text:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        xtr = vectorizer.fit_transform(train_text)
        xte = vectorizer.transform(test_text)
        train_parts.append(xtr)
        test_parts.append(xte)
        detail["n_text_features"] = int(xtr.shape[1])
    dense_train_parts = []
    dense_test_parts = []
    if cfg.use_hand:
        dense_train_parts.append(train_hand)
        dense_test_parts.append(test_hand)
    if cfg.use_model:
        dense_train_parts.append(train_model)
        dense_test_parts.append(test_model)
    if dense_train_parts:
        train_dense = pd.concat(dense_train_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_dense = pd.concat(dense_test_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaler = StandardScaler()
        train_parts.append(sparse.csr_matrix(scaler.fit_transform(train_dense)))
        test_parts.append(sparse.csr_matrix(scaler.transform(test_dense)))
        detail["n_numeric_features"] = int(train_dense.shape[1])
    if not train_parts:
        raise ValueError(f"No features selected for {cfg.name}")
    return sparse.hstack(train_parts).tocsr(), sparse.hstack(test_parts).tocsr(), detail


def make_classifier(model_name: str, seed: int, positive_ratio: float | None = None):
    if model_name == "logreg":
        return LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed)
    if model_name == "logreg_c1":
        return LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed)
    if model_name == "extra":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def oof_predict(
    df: pd.DataFrame,
    texts: pd.Series,
    hand: pd.DataFrame,
    model: pd.DataFrame,
    y: np.ndarray,
    cfg: FeatureConfig,
    model_name: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    folds = df["fold_id"].astype(int).to_numpy()
    prob = np.full(len(df), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for fold in sorted(set(folds)):
        tr = folds != fold
        te = folds == fold
        if len(np.unique(y[tr])) < 2:
            prob[te] = float(y[tr].mean())
            continue
        xtr, xte, detail = build_fold_features(
            cfg,
            texts[tr],
            texts[te],
            hand.loc[tr],
            hand.loc[te],
            model.loc[tr],
            model.loc[te],
        )
        clf = make_classifier(model_name, seed=20260521 + int(fold))
        clf.fit(xtr, y[tr])
        prob[te] = clf.predict_proba(xte)[:, 1]
        rows.append({"fold_id": int(fold), "train_n": int(tr.sum()), "test_n": int(te.sum()), **detail})
    return prob, rows


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        elif objective == "f1":
            score = f1_score(y, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_s = float(score)
            best_t = float(t)
    return best_t, best_s


def detector_metrics(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> dict[str, object]:
    threshold, _ = best_threshold(y, prob, objective)
    pred = (prob >= threshold).astype(int)
    row = metric_dict(y, pred, prob)
    row["threshold"] = threshold
    positives = int(y.sum())
    order = np.argsort(-prob)
    for k in [10, 20, 30, 38, positives, 80, 100]:
        kk = min(int(k), len(y))
        if kk <= 0:
            continue
        hit = int(y[order[:kk]].sum())
        row[f"top{kk}_precision"] = float(hit / kk)
        row[f"top{kk}_recall"] = float(hit / max(positives, 1))
    return row


def feature_association(df: pd.DataFrame, hand: pd.DataFrame, target_col: str) -> pd.DataFrame:
    y = df[target_col].astype(int).to_numpy()
    rows = []
    for col in hand.columns:
        values = hand[col].astype(float).to_numpy()
        if np.allclose(values, values[0]):
            continue
        binary = (values > 0).astype(int) if set(np.unique(values)).issubset({0, 1}) else (values >= np.median(values)).astype(int)
        a = int(((binary == 1) & (y == 1)).sum())
        b = int(((binary == 1) & (y == 0)).sum())
        c = int(((binary == 0) & (y == 1)).sum())
        d = int(((binary == 0) & (y == 0)).sum())
        odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        prev_pos = float(a / max(a + c, 1))
        prev_neg = float(b / max(b + d, 1))
        corr = float(np.corrcoef(values, y)[0, 1]) if np.std(values) > 1e-8 else 0.0
        rows.append(
            {
                "target": target_col,
                "feature": col,
                "odds_ratio_binary_median": odds,
                "positive_group_mean": float(values[y == 1].mean()) if (y == 1).any() else float("nan"),
                "negative_group_mean": float(values[y == 0].mean()) if (y == 0).any() else float("nan"),
                "binary_prevalence_positive": prev_pos,
                "binary_prevalence_negative": prev_neg,
                "abs_corr": abs(corr),
                "corr": corr,
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_corr", "odds_ratio_binary_median"], ascending=False)


def train_corrector_oof(
    df: pd.DataFrame,
    texts: pd.Series,
    hand: pd.DataFrame,
    model: pd.DataFrame,
    cfg: FeatureConfig,
    train_scope: str,
) -> pd.DataFrame:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    prob = np.full(len(df), np.nan, dtype=float)
    pred = np.full(len(df), -1, dtype=int)
    for fold in sorted(set(folds)):
        if train_scope == "all":
            tr = folds != fold
        elif train_scope == "hard_and_salvage":
            tr = (folds != fold) & df["difficulty_fine"].isin(["hard_core", "hard_salvage_teacher"]).to_numpy()
        elif train_scope == "hard_core":
            tr = (folds != fold) & (df["difficulty_fine"].to_numpy() == "hard_core")
        else:
            raise ValueError(train_scope)
        te = folds == fold
        if tr.sum() < 16 or len(np.unique(y[tr])) < 2:
            tr = folds != fold
        xtr, xte, _ = build_fold_features(cfg, texts[tr], texts[te], hand.loc[tr], hand.loc[te], model.loc[tr], model.loc[te])
        clf = make_classifier("logreg", seed=20260581 + int(fold))
        clf.fit(xtr, y[tr])
        fold_prob = clf.predict_proba(xte)[:, 1]
        prob[te] = fold_prob
        pred[te] = (fold_prob >= 0.5).astype(int)
    out = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
            "p_best41",
            "pred_best41",
        ]
    ].copy()
    out["corrector_prob"] = prob
    out["corrector_pred"] = pred
    out["base_correct"] = out["pred_best41"].astype(int) == out["label_idx"].astype(int)
    out["corrector_correct"] = out["corrector_pred"].astype(int) == out["label_idx"].astype(int)
    return out


def route_budget_curve(
    df: pd.DataFrame,
    router_prob: np.ndarray,
    corrector: pd.DataFrame,
    route_name: str,
    corrector_name: str,
) -> pd.DataFrame:
    y = df["label_idx"].astype(int).to_numpy()
    base_pred = df["pred_best41"].astype(int).to_numpy()
    base_prob = df["p_best41"].astype(float).to_numpy()
    corr_pred = corrector["corrector_pred"].astype(int).to_numpy()
    corr_prob = corrector["corrector_prob"].astype(float).to_numpy()
    order = np.argsort(-router_prob)
    rows = []
    for budget in [0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 45, 50, 60, 65, 70, 80, 100]:
        n_route = int(round(len(df) * budget / 100.0))
        routed = np.zeros(len(df), dtype=bool)
        if n_route > 0:
            routed[order[:n_route]] = True
        pred = base_pred.copy()
        prob = base_prob.copy()
        pred[routed] = corr_pred[routed]
        prob[routed] = corr_prob[routed]
        routed_acc = float((pred[routed] == y[routed]).mean()) if routed.any() else float("nan")
        pass_acc = float((pred[~routed] == y[~routed]).mean()) if (~routed).any() else float("nan")
        hard_routed = int(df.loc[routed, "hard_core"].sum()) if routed.any() else 0
        hard_total = int(df["hard_core"].sum())
        rescue = int(((base_pred != y) & (pred == y) & routed).sum())
        hurt = int(((base_pred == y) & (pred != y) & routed).sum())
        rows.append(
            {
                "route_name": route_name,
                "corrector_name": corrector_name,
                "budget_pct": budget,
                "routed_n": int(routed.sum()),
                "pass_n": int((~routed).sum()),
                "overall_acc": float(accuracy_score(y, pred)),
                "overall_bacc": float(balanced_accuracy_score(y, pred)),
                "overall_f1": float(f1_score(y, pred, zero_division=0)),
                "pass_acc": pass_acc,
                "routed_acc": routed_acc,
                "hard_core_routed": hard_routed,
                "hard_core_recall": float(hard_routed / max(hard_total, 1)),
                "rescue_n": rescue,
                "hurt_n": hurt,
                "net_rescue": rescue - hurt,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, texts, hand, model = load_frame(project_root, args)

    base_metrics = metric_dict(
        df["label_idx"].astype(int).to_numpy(),
        df["pred_best41"].astype(int).to_numpy(),
        df["p_best41"].astype(float).to_numpy(),
    )
    (output_dir / "base41_metrics.json").write_text(json.dumps(base_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    feature_configs = [
        FeatureConfig("gross_hand", use_text=False, use_hand=True, use_model=False),
        FeatureConfig("gross_text", use_text=True, use_hand=False, use_model=False),
        FeatureConfig("gross_text_hand", use_text=True, use_hand=True, use_model=False),
        FeatureConfig("model_visible", use_text=False, use_hand=False, use_model=True),
        FeatureConfig("gross_hand_model", use_text=False, use_hand=True, use_model=True),
        FeatureConfig("gross_text_hand_model", use_text=True, use_hand=True, use_model=True),
    ]
    targets = {
        "hard_core": df["hard_core"].astype(int).to_numpy(),
        "best41_wrong": df["best41_wrong"].astype(int).to_numpy(),
        "best41_fn": df["best41_fn"].astype(int).to_numpy(),
        "best41_fp": df["best41_fp"].astype(int).to_numpy(),
    }
    detector_rows = []
    detector_probs: dict[str, np.ndarray] = {}
    for target_name, y in targets.items():
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for cfg in feature_configs:
            for clf_name in ["logreg", "logreg_c1", "extra"]:
                prob, fold_rows = oof_predict(df, texts, hand, model, y, cfg, clf_name)
                row = detector_metrics(y, prob)
                run_name = f"{target_name}__{cfg.name}__{clf_name}"
                row.update(
                    {
                        "run_name": run_name,
                        "target": target_name,
                        "feature_set": cfg.name,
                        "classifier": clf_name,
                        "positives": int(y.sum()),
                    }
                )
                detector_rows.append(row)
                detector_probs[run_name] = prob
                pd.DataFrame(
                    {
                        "case_id": df["case_id"],
                        "original_case_id": df["original_case_id"],
                        "fold_id": df["fold_id"],
                        "label_idx": df["label_idx"],
                        "difficulty_fine": df["difficulty_fine"],
                        "target": y,
                        "prob": prob,
                    }
                ).to_csv(output_dir / f"{run_name}_oof.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(fold_rows).to_csv(output_dir / f"{run_name}_folds.csv", index=False, encoding="utf-8-sig")

    detector_summary = pd.DataFrame(detector_rows).sort_values(["target", "ap", "auc"], ascending=[True, False, False])
    detector_summary.to_csv(output_dir / "gross_hardcore_detector_summary.csv", index=False, encoding="utf-8-sig")

    assoc = pd.concat(
        [
            feature_association(df, hand, "hard_core").head(80),
            feature_association(df, hand, "best41_wrong").head(80),
            feature_association(df, hand, "best41_fn").head(80),
            feature_association(df, hand, "best41_fp").head(80),
        ],
        ignore_index=True,
    )
    assoc.to_csv(output_dir / "gross_feature_association_top.csv", index=False, encoding="utf-8-sig")

    corrector_configs = [
        ("corrector_all_gross_text_hand_model", FeatureConfig("gross_text_hand_model", True, True, True), "all"),
        ("corrector_hard_gross_text_hand_model", FeatureConfig("gross_text_hand_model", True, True, True), "hard_and_salvage"),
        ("corrector_hard_gross_text_hand", FeatureConfig("gross_text_hand", True, True, False), "hard_and_salvage"),
    ]
    corrector_tables = {}
    for name, cfg, scope in corrector_configs:
        table = train_corrector_oof(df, texts, hand, model, cfg, scope)
        corrector_tables[name] = table
        y = table["label_idx"].astype(int).to_numpy()
        pred = table["corrector_pred"].astype(int).to_numpy()
        prob = table["corrector_prob"].astype(float).to_numpy()
        metrics = metric_dict(y, pred, prob)
        table.to_csv(output_dir / f"{name}_oof.csv", index=False, encoding="utf-8-sig")
        (output_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Candidate route scores: best detector per target plus simple model/gross conflicts.
    route_scores: dict[str, np.ndarray] = {
        "simple_best41_uncertainty": 1.0 - df["best41_conf"].astype(float).to_numpy(),
        "simple_gross_model_conflict": np.abs(model["gross_high_minus_model_logit"].to_numpy())
        + np.abs(model["gross_low_minus_model_lowlogit"].to_numpy()),
    }
    for target_name in targets:
        sub = detector_summary[detector_summary["target"] == target_name]
        for _, row in sub.head(3).iterrows():
            route_scores[str(row["run_name"])] = detector_probs[str(row["run_name"])]

    budget_rows = []
    for route_name, scores in route_scores.items():
        for corr_name, corr_table in corrector_tables.items():
            budget_rows.append(route_budget_curve(df, scores, corr_table, route_name, corr_name))
    budget_summary = pd.concat(budget_rows, ignore_index=True)
    budget_summary = budget_summary.sort_values(["overall_acc", "overall_bacc", "net_rescue"], ascending=False)
    budget_summary.to_csv(output_dir / "deployable_route_corrector_budget_summary.csv", index=False, encoding="utf-8-sig")

    case_export = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "difficulty",
            "difficulty_fine",
            "hard_core",
            "p_best41",
            "pred_best41",
            "best41_wrong",
            "best41_fn",
            "best41_fp",
        ]
    ].copy()
    case_export["gross_text"] = texts
    for col in ["manual_highrisk_score", "manual_lowrisk_score", "manual_conflict_score"]:
        case_export[col] = hand[col]
    for name, scores in route_scores.items():
        case_export[f"route_score__{name}"] = scores
    case_export.to_csv(output_dir / "case_level_gross_signal_table.csv", index=False, encoding="utf-8-sig")

    report = {
        "n_cases": int(len(df)),
        "base41": base_metrics,
        "best_detectors": detector_summary.groupby("target").head(5).to_dict(orient="records"),
        "best_budget_rows": budget_summary.head(20).to_dict(orient="records"),
    }
    (output_dir / "gross_hardcore_signal_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Base41:", json.dumps(base_metrics, ensure_ascii=False))
    print("\nBest detectors:")
    print(detector_summary.groupby("target").head(5).to_string(index=False))
    print("\nBest route-corrector budgets:")
    print(budget_summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
