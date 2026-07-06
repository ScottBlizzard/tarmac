from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v25_auto_quality_gate_20260527"

MAIN_THRESHOLD = 0.595
ROBUST_THRESHOLD = 0.57


def metrics_binary(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def diag_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return metrics_binary(y, pred)


def prep_external() -> pd.DataFrame:
    df = pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv")
    df = df.copy()
    df["quality_review_target"] = df["manual_quality_status_v1"].astype(str).ne("pass_readable").astype(int)
    df["main_pred"] = (df["prob_base162"].astype(float) >= MAIN_THRESHOLD).astype(int)
    df["robust_prob"] = df[["prob_base162", "prob103_vitl", "prob_mean_core"]].mean(axis=1).astype(float)
    df["robust_pred"] = (df["robust_prob"] >= ROBUST_THRESHOLD).astype(int)
    df["safety_trigger"] = ((df["main_pred"] == 0) & (df["robust_pred"] == 1)).astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        "width",
        "height",
        "megapixels",
        "file_mb",
        "fg_ratio",
        "bbox_area_ratio",
        "touch_edges",
        "lap_var",
        "tenengrad",
        "brightness_mean",
        "contrast",
        "dark_ratio",
        "bright_ratio",
        "glare_ratio",
        "saturation_mean",
        "bg_r",
        "bg_g",
        "bg_b",
        "dist_p90",
        "sat_mean",
        "quality_score",
    ]
    categorical = ["quality_status"]
    return [c for c in numeric if c in df.columns], [c for c in categorical if c in df.columns]


def model_specs(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    tree_preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return {
        "quality_logistic": Pipeline(
            [
                ("prep", preprocess),
                ("clf", LogisticRegression(C=0.4, class_weight="balanced", max_iter=2000, random_state=20260527)),
            ]
        ),
        "quality_rf": Pipeline(
            [
                ("prep", tree_preprocess),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=3,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=20260527,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "quality_extra_trees": Pipeline(
            [
                ("prep", tree_preprocess),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=3,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=20260527,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def oof_quality_scores(df: pd.DataFrame, features: list[str], model: Pipeline) -> np.ndarray:
    y = df["quality_review_target"].to_numpy(dtype=int)
    scores = np.zeros(len(df), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260527)
    for train_idx, test_idx in cv.split(df[features], y):
        model.fit(df.iloc[train_idx][features], y[train_idx])
        scores[test_idx] = model.predict_proba(df.iloc[test_idx][features])[:, 1]
    return scores


def choose_quality_threshold(y: np.ndarray, score: np.ndarray, objective: str) -> float:
    best_key: tuple[float, ...] | None = None
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.951, 0.005):
        pred = (score >= threshold).astype(int)
        m = metrics_binary(y, pred)
        if objective == "f1":
            key = (float(m["f1"]), float(m["sensitivity"]), -abs(float(threshold) - 0.5))
        elif objective == "recall90":
            penalty = 0.0 if float(m["sensitivity"]) >= 0.90 else -1.0
            key = (penalty, float(m["specificity"]), float(m["f1"]), -abs(float(threshold) - 0.5))
        elif objective == "balanced":
            key = (float(m["balanced_accuracy"]), float(m["f1"]), -abs(float(threshold) - 0.5))
        else:
            raise ValueError(objective)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def evaluate_flow(df: pd.DataFrame, quality_review: np.ndarray, policy: str) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    main = df["main_pred"].to_numpy(dtype=int)
    robust = df["robust_pred"].to_numpy(dtype=int)
    safety = df["safety_trigger"].to_numpy(dtype=bool)
    quality_review = np.asarray(quality_review, dtype=bool)
    if policy == "quality_review_else_main":
        review = quality_review
        pred = np.where(review, y, main)
    elif policy == "quality_review_else_safety_switch":
        review = quality_review
        pred = np.where(review, y, np.where(safety, robust, main))
    elif policy == "quality_or_safety_review_else_main":
        review = quality_review | safety
        pred = np.where(review, y, main)
    elif policy == "quality_review_and_safety_switch_no_human_safety":
        review = quality_review
        pred = np.where(review, y, np.where(safety, robust, main))
    else:
        raise ValueError(policy)
    out = diag_metrics(y, pred)
    out["review_n"] = int(review.sum())
    out["review_rate"] = float(review.mean())
    out["quality_review_n"] = int(quality_review.sum())
    out["safety_trigger_n"] = int(safety.sum())
    out["union_quality_safety_n"] = int((quality_review | safety).sum())
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prep_external()
    numeric, categorical = feature_columns(df)
    features = numeric + categorical
    yq = df["quality_review_target"].to_numpy(dtype=int)

    quality_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    pred_df = df[["case_id", "original_case_id", "manual_quality_status_v1", "quality_review_target", "quality_status", "quality_score", "label_idx", "main_pred", "robust_pred", "safety_trigger"]].copy()

    # Heuristic quality gate from existing image-quality score. This is fully automatic.
    heuristic_preds = {
        "heuristic_nonpass_status": df["quality_status"].astype(str).ne("pass").to_numpy(),
        "heuristic_score_lt_100": (pd.to_numeric(df["quality_score"], errors="coerce").fillna(0) < 100).to_numpy(),
        "heuristic_score_lt_92": (pd.to_numeric(df["quality_score"], errors="coerce").fillna(0) < 92).to_numpy(),
    }
    for name, pred_bool in heuristic_preds.items():
        m = metrics_binary(yq, pred_bool.astype(int))
        m.update({"model": name, "threshold": np.nan, "roc_auc": np.nan, "average_precision": np.nan})
        quality_rows.append(m)
        pred_df[f"{name}_review"] = pred_bool.astype(int)
        for policy in ["quality_review_else_main", "quality_review_else_safety_switch", "quality_or_safety_review_else_main"]:
            row = {"quality_model": name, "quality_objective": "heuristic", "policy": policy}
            row.update(evaluate_flow(df, pred_bool, policy))
            flow_rows.append(row)

    for name, model in model_specs(numeric, categorical).items():
        score = oof_quality_scores(df, features, model)
        pred_df[f"{name}_risk_oof"] = score
        for objective in ["f1", "balanced", "recall90"]:
            threshold = choose_quality_threshold(yq, score, objective)
            pred_bool = score >= threshold
            m = metrics_binary(yq, pred_bool.astype(int))
            m.update(
                {
                    "model": name,
                    "threshold": threshold,
                    "objective": objective,
                    "roc_auc": roc_auc_score(yq, score),
                    "average_precision": average_precision_score(yq, score),
                }
            )
            quality_rows.append(m)
            pred_df[f"{name}_{objective}_review"] = pred_bool.astype(int)
            for policy in ["quality_review_else_main", "quality_review_else_safety_switch", "quality_or_safety_review_else_main"]:
                row = {"quality_model": name, "quality_objective": objective, "policy": policy}
                row.update(evaluate_flow(df, pred_bool, policy))
                flow_rows.append(row)

    pd.DataFrame(quality_rows).to_csv(OUT_DIR / "v25_quality_gate_cv_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow_rows).to_csv(OUT_DIR / "v25_quality_safety_flow_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT_DIR / "v25_quality_gate_case_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(OUT_DIR / "v25_quality_gate_features.csv", index=False, encoding="utf-8-sig")

    print(f"[done] {OUT_DIR}")
    print("\nQuality gate metrics:")
    print(pd.DataFrame(quality_rows).sort_values(["f1", "balanced_accuracy"], ascending=False).to_string(index=False))
    print("\nFlow metrics sorted by balanced accuracy:")
    print(pd.DataFrame(flow_rows).sort_values(["balanced_accuracy", "accuracy"], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
