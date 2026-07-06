from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


TASK7_CLASSES = ("low_risk_group", "high_risk_group")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 gross-finding text stacking probe.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/54_gross_text_stacking_20260521",
    )
    parser.add_argument("--mode", default="fast", choices=("fast", "full"))
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out


def score(y: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y, pred))
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y, pred))
    if objective == "f1":
        return float(f1_score(y, pred, zero_division=0))
    raise ValueError(objective)


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.10, 0.90, 81):
        pred = (prob >= t).astype(int)
        s = score(y, pred, objective)
        key = (s, -abs(float(t) - 0.5))
        if key > (best_s, -abs(best_t - 0.5)):
            best_s = s
            best_t = float(t)
    return best_t, best_s


def get_col(df: pd.DataFrame, wanted: str) -> str:
    if wanted in df.columns:
        return wanted
    for col in df.columns:
        if str(col).strip() == wanted:
            return str(col)
    raise KeyError(f"Missing column: {wanted}")


def norm_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\u3000", " ").replace("×", "*").replace("x", "*").replace("X", "*")
    return re.sub(r"\s+", "", text)


def has_any(text: str, words: list[str]) -> float:
    return float(any(w in text for w in words))


def count_any(text: str, words: list[str]) -> float:
    return float(sum(text.count(w) for w in words))


def size_features(text: str) -> dict[str, float]:
    dims: list[list[float]] = []
    tumor_dims: list[list[float]] = []
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*[*]\s*(\d+(?:\.\d+)?)"
        r"(?:\s*[*]\s*(\d+(?:\.\d+)?))?\s*(cm|mm|CM|MM)?"
    )
    for m in pattern.finditer(text):
        values = [float(m.group(1)), float(m.group(2))]
        if m.group(3):
            values.append(float(m.group(3)))
        unit = (m.group(4) or "mm").lower()
        factor = 10.0 if unit == "cm" else 1.0
        values = [v * factor for v in values]
        dims.append(values)
        window = text[max(0, m.start() - 18) : min(len(text), m.end() + 12)]
        if any(k in window for k in ["肿物", "肿块", "肿瘤", "结节", "病灶", "包块"]):
            if not any(k in window for k in ["脂肪", "心包", "肺", "胸腺组织", "组织大小"]):
                tumor_dims.append(values)

    def summarize(prefix: str, values: list[list[float]]) -> dict[str, float]:
        out = {
            f"{prefix}_n_size_mentions": float(len(values)),
            f"{prefix}_max_dim_mm": 0.0,
            f"{prefix}_max_area_mm2": 0.0,
            f"{prefix}_max_volume_mm3": 0.0,
        }
        for ds in values:
            if not ds:
                continue
            out[f"{prefix}_max_dim_mm"] = max(out[f"{prefix}_max_dim_mm"], max(ds))
            if len(ds) >= 2:
                out[f"{prefix}_max_area_mm2"] = max(out[f"{prefix}_max_area_mm2"], ds[0] * ds[1])
            if len(ds) >= 3:
                out[f"{prefix}_max_volume_mm3"] = max(out[f"{prefix}_max_volume_mm3"], ds[0] * ds[1] * ds[2])
        return out

    out = summarize("all", dims)
    out.update(summarize("tumor", tumor_dims))
    return out


def hand_gross_features(df: pd.DataFrame) -> pd.DataFrame:
    text_col = get_col(df, "肉眼所见")
    age_col = get_col(df, "年龄")
    sex_col = get_col(df, "性别")
    texts = df[text_col].map(norm_text)
    feat = pd.DataFrame(index=df.index)
    age = pd.to_numeric(df[age_col], errors="coerce")
    feat["age"] = age.fillna(age.median()).astype(float)
    sex = df[sex_col].fillna("").astype(str)
    feat["sex_male"] = sex.str.contains("男", regex=False).astype(float)
    feat["sex_female"] = sex.str.contains("女", regex=False).astype(float)
    feat["gross_text_len"] = texts.str.len().astype(float)
    feat["gross_has_text"] = (texts.str.len() > 0).astype(float)
    feat = pd.concat([feat, pd.DataFrame([size_features(t) for t in texts], index=df.index)], axis=1)

    groups = {
        "boundary_clear": ["界清", "边界清", "界尚清", "边界尚清"],
        "boundary_unclear": ["界不清", "边界不清", "界欠清", "边界欠清"],
        "capsule_any": ["包膜"],
        "capsule_complete": ["包膜完整", "有包膜", "似有包膜", "可见包膜", "包膜尚完整"],
        "capsule_absent": ["未见明显包膜", "无包膜", "未见包膜"],
        "capsule_involved": ["包膜侵犯", "侵犯包膜", "突破包膜", "累及包膜"],
        "invasion": ["侵犯", "累及", "侵及", "浸润"],
        "fat_attached": ["脂肪"],
        "lung_attached": ["肺"],
        "pericardium_attached": ["心包"],
        "pleura_attached": ["胸膜"],
        "hemorrhage": ["出血", "暗红", "血性"],
        "necrosis": ["坏死"],
        "cystic": ["囊性", "囊变", "囊腔", "囊实"],
        "calcification": ["钙化"],
        "lobulated": ["分叶", "结节状", "多结节"],
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
    }
    for name, words in groups.items():
        feat[f"kw_{name}"] = texts.map(lambda t, ws=words: has_any(t, ws)).astype(float)
        feat[f"cnt_{name}"] = texts.map(lambda t, ws=words: count_any(t, ws)).astype(float)

    for col in list(feat.columns):
        if col.startswith(("all_", "tumor_", "cnt_")) or col == "gross_text_len":
            feat[f"log1p_{col}"] = np.log1p(feat[col].astype(float))

    size_z = np.log1p(feat["tumor_max_dim_mm"].astype(float))
    area_z = np.log1p(feat["tumor_max_area_mm2"].astype(float))
    if float(size_z.std()) > 1e-8:
        size_z = (size_z - size_z.mean()) / size_z.std()
    if float(area_z.std()) > 1e-8:
        area_z = (area_z - area_z.mean()) / area_z.std()
    feat["manual_gross_highrisk_score"] = (
        1.5 * feat["kw_boundary_unclear"]
        + 1.2 * feat["kw_capsule_absent"]
        + 1.4 * feat["kw_capsule_involved"]
        + 1.1 * feat["kw_invasion"]
        + 0.8 * feat["kw_lung_attached"]
        + 0.7 * feat["kw_pericardium_attached"]
        + 0.4 * feat["kw_pleura_attached"]
        + 0.4 * feat["kw_necrosis"]
        + 0.2 * feat["kw_hemorrhage"]
        + 0.25 * size_z.fillna(0.0)
        + 0.15 * area_z.fillna(0.0)
        - 1.3 * feat["kw_boundary_clear"]
        - 1.0 * feat["kw_capsule_complete"]
        - 0.25 * feat["kw_texture_tender"]
    )
    feat["manual_capsule_boundary_balance"] = (
        feat["kw_boundary_unclear"]
        + feat["kw_capsule_absent"]
        + feat["kw_capsule_involved"]
        - feat["kw_boundary_clear"]
        - feat["kw_capsule_complete"]
    )
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_data(project_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    registry = pd.read_csv(project_root / args.registry_csv, dtype={"case_id": str, "original_case_id": str})
    split = pd.read_csv(project_root / args.split_csv, dtype={"case_id": str})
    split = split[["case_id", "master_fold_id"]].rename(columns={"master_fold_id": "fold_id"})
    curriculum = pd.read_csv(project_root / args.curriculum_csv, dtype={"case_id": str})
    curriculum = curriculum[["case_id", "difficulty", "difficulty_fine"]]

    df = registry.merge(split, on="case_id", how="inner")
    df = df.merge(curriculum, on="case_id", how="left")
    df = df[df["task_l7_label"].isin(TASK7_CLASSES)].copy()
    df["fold_id"] = df["fold_id"].astype(int)
    df["label_idx"] = (df["task_l7_label"] == "high_risk_group").astype(int)

    review = pd.read_csv(project_root / args.review_score_csv, dtype={"case_id": str})
    prob_cols = [c for c in review.columns if c.startswith("p_")]
    score_cols = [c for c in review.columns if c.startswith("review_score_")]
    keep = ["case_id"] + prob_cols + score_cols
    review = review[keep].copy()

    best41 = pd.read_csv(project_root / args.best41_csv, dtype={"case_id": str})
    best41 = best41[["case_id", "final_prob_high", "final_pred", "pred_upper", "p_upper"]].copy()
    best41 = best41.rename(
        columns={
            "final_prob_high": "p_best41",
            "final_pred": "pred_best41",
            "p_upper": "p_best41_upper",
            "pred_upper": "pred_best41_upper",
        }
    )

    df = df.merge(review, on="case_id", how="left").merge(best41, on="case_id", how="left")
    text = df[get_col(df, "肉眼所见")].map(norm_text)
    hand = hand_gross_features(df)

    numeric_cols = [
        c
        for c in df.columns
        if (c.startswith("p_") or c.startswith("review_score_") or c.startswith("pred_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = df[numeric_cols].copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    numeric = pd.concat([numeric.reset_index(drop=True), hand.reset_index(drop=True)], axis=1)
    return df.reset_index(drop=True), text.reset_index(drop=True), numeric.reset_index(drop=True)


def build_sparse(
    train_text: pd.Series,
    test_text: pd.Series,
    train_num: pd.DataFrame | None,
    test_num: pd.DataFrame | None,
    cfg: dict[str, object],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, object]]:
    pieces_train = []
    pieces_test = []
    detail: dict[str, object] = {}

    if cfg["use_text"]:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"],
            max_df=cfg["max_df"],
            sublinear_tf=True,
            norm="l2",
        )
        xtr_text = vectorizer.fit_transform(train_text)
        xte_text = vectorizer.transform(test_text)
        pieces_train.append(xtr_text)
        pieces_test.append(xte_text)
        detail["n_text_features"] = int(xtr_text.shape[1])

    if cfg["use_numeric"] and train_num is not None and test_num is not None:
        scaler = StandardScaler()
        xtr_num = scaler.fit_transform(train_num)
        xte_num = scaler.transform(test_num)
        pieces_train.append(sparse.csr_matrix(xtr_num))
        pieces_test.append(sparse.csr_matrix(xte_num))
        detail["n_numeric_features"] = int(train_num.shape[1])

    return sparse.hstack(pieces_train).tocsr(), sparse.hstack(pieces_test).tocsr(), detail


def inner_select(
    df: pd.DataFrame,
    text: pd.Series,
    numeric: pd.DataFrame,
    train_mask: np.ndarray,
    cfg: dict[str, object],
    objective: str,
) -> tuple[float, float, float]:
    y = df["label_idx"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    c_grid = cfg["c_grid"]
    inner_prob_by_c: dict[float, np.ndarray] = {}
    best_c = float(c_grid[0])
    best_t = 0.5
    best_s = -1.0
    for c in c_grid:
        prob = np.full(len(df), np.nan, dtype=float)
        for inner_fold in sorted(set(folds[train_mask])):
            tr = train_mask & (folds != inner_fold)
            va = train_mask & (folds == inner_fold)
            if tr.sum() < 16 or va.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            xtr, xva, _ = build_sparse(
                text[tr],
                text[va],
                numeric.loc[tr] if cfg["use_numeric"] else None,
                numeric.loc[va] if cfg["use_numeric"] else None,
                cfg,
            )
            clf = LogisticRegression(
                C=float(c),
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=20260521,
            )
            clf.fit(xtr, y[tr])
            prob[va] = clf.predict_proba(xva)[:, 1]
        valid = train_mask & ~np.isnan(prob)
        if valid.sum() < 16 or len(np.unique(y[valid])) < 2:
            continue
        threshold, current = best_threshold(y[valid], prob[valid], objective)
        key = (current, -abs(threshold - 0.5), -abs(np.log10(float(c))))
        if key > (best_s, -abs(best_t - 0.5), -abs(np.log10(best_c))):
            best_s = current
            best_c = float(c)
            best_t = threshold
        inner_prob_by_c[float(c)] = prob
    return best_c, best_t, best_s


def run_config(
    df: pd.DataFrame,
    text: pd.Series,
    numeric: pd.DataFrame,
    cfg: dict[str, object],
    objective: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    y = df["label_idx"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    prob = np.full(len(df), np.nan, dtype=float)
    choices: list[dict[str, object]] = []
    for fold in sorted(set(folds)):
        train_mask = folds != fold
        test_mask = folds == fold
        c, threshold, inner_score = inner_select(df, text, numeric, train_mask, cfg, objective)
        xtr, xte, detail = build_sparse(
            text[train_mask],
            text[test_mask],
            numeric.loc[train_mask] if cfg["use_numeric"] else None,
            numeric.loc[test_mask] if cfg["use_numeric"] else None,
            cfg,
        )
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=20260521 + int(fold),
        )
        clf.fit(xtr, y[train_mask])
        prob[test_mask] = clf.predict_proba(xte)[:, 1]
        choices.append(
            {
                "fold_id": int(fold),
                "c": c,
                "threshold": threshold,
                "inner_score": inner_score,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                **detail,
            }
        )
    threshold_global = 0.5
    pred = np.zeros(len(df), dtype=int)
    for choice in choices:
        fold = choice["fold_id"]
        pred[folds == fold] = (prob[folds == fold] >= float(choice["threshold"])).astype(int)
    oof = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
        ]
    ].copy()
    oof["prob_high_risk_group"] = prob
    oof["prob_low_risk_group"] = 1.0 - prob
    oof["pred_idx"] = pred
    oof["global_pred_idx_050"] = (prob >= threshold_global).astype(int)
    return oof, choices


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, text, numeric = load_data(project_root, args)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    configs = []
    base_feature_sets = [
        (True, False, "gross_text_only"),
        (False, True, "numeric_only"),
        (True, True, "gross_text_plus_numeric"),
    ]
    if args.mode == "fast":
        ngram_ranges = [(2, 4)]
        min_dfs = [1]
        c_grid = [0.03, 0.1, 0.3, 1.0]
        objectives = ["balanced_accuracy", "accuracy"]
    else:
        ngram_ranges = [(2, 4), (2, 5), (3, 5)]
        min_dfs = [1, 2]
        c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        objectives = ["balanced_accuracy", "accuracy", "f1"]
    for use_text, use_numeric, name in base_feature_sets:
        for ngram_range in ngram_ranges if use_text else [(1, 1)]:
            for min_df in min_dfs if use_text else [1]:
                configs.append(
                    {
                        "name": f"{name}_ng{ngram_range[0]}{ngram_range[1]}_mindf{min_df}",
                        "use_text": use_text,
                        "use_numeric": use_numeric,
                        "ngram_range": ngram_range,
                        "min_df": min_df,
                        "max_df": 0.95,
                        "c_grid": c_grid,
                    }
                )

    rows = []
    for cfg in configs:
        for objective in objectives:
            run_name = f"{cfg['name']}__{objective}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            oof, choices = run_config(df, text, numeric, cfg, objective)
            metrics = metric_dict(
                oof["label_idx"].to_numpy(dtype=int),
                oof["pred_idx"].to_numpy(dtype=int),
                oof["prob_high_risk_group"].to_numpy(dtype=float),
            )
            row = {
                **metrics,
                "config": run_name,
                "use_text": bool(cfg["use_text"]),
                "use_numeric": bool(cfg["use_numeric"]),
                "ngram_range": str(cfg["ngram_range"]),
                "min_df": int(cfg["min_df"]),
                "objective": objective,
            }
            rows.append(row)
            oof["correct"] = oof["pred_idx"].astype(int) == oof["label_idx"].astype(int)
            oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(choices).to_csv(run_dir / "fold_choices.csv", index=False)
            (run_dir / "overall_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False).to_csv(
                output_dir / "gross_text_stacking_summary.partial.csv", index=False, encoding="utf-8-sig"
            )

    summary = pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary.to_csv(output_dir / "gross_text_stacking_summary.csv", index=False, encoding="utf-8-sig")
    best = summary.iloc[0].to_dict()
    (output_dir / "best_summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
