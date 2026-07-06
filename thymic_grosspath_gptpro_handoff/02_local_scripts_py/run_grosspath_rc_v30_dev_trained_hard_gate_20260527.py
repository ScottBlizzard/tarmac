from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
EXT_QUALITY_DIR = ROOT / "outputs" / "external_quality_gate_20260525"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v30_dev_trained_hard_gate_20260527"

MAIN_THRESHOLD = 0.595
ROBUST_THRESHOLD = 0.57
BUDGETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]
USE_SLOW_DEV_IMAGE_EXTRACTION = False


PROB_FEATURES = [
    "prob_base162",
    "prob103_vitl",
    "prob107_qkvb",
    "prob_mean_core",
    "core_prob_std",
    "core_prob_range",
    "abs_162_103",
    "abs_162_107",
    "abs_103_107",
    "margin162",
    "margin_mean_core",
    "core_agree_count",
    "score_margin_agree",
    "score_v0_simple",
    "prob_stack_plain",
    "prob_stack_balanced",
    "main_prob",
    "main_margin_abs",
    "robust_prob",
    "robust_margin_abs",
    "main_robust_abs_diff",
]

IMAGE_FEATURES = [
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
]

BIN_FEATURES = [
    "main_pred",
    "robust_pred",
    "p2_pred",
    "safety_trigger",
    "main_robust_disagree",
    "low_main_high_robust",
    "high_main_low_robust",
]

CAT_FEATURES = ["view_type_final", "multi_image_group"]


def norm_case_id(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace(".0", "").strip()


def image_stats(path_value: object) -> dict[str, float]:
    if pd.isna(path_value):
        return {k: np.nan for k in IMAGE_FEATURES}
    path = Path(str(path_value))
    if not path.exists():
        return {k: np.nan for k in IMAGE_FEATURES}

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {k: np.nan for k in IMAGE_FEATURES}

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # A loose foreground estimate; exact segmentation is not required for gate features.
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = float((mask > 0).mean())
    if fg_ratio > 0.8:
        mask = 255 - mask
        fg_ratio = float((mask > 0).mean())
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        bbox_area_ratio = np.nan
        touch_edges = np.nan
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bbox_area_ratio = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / (w * h))
        touch_edges = int(x0 <= 3) + int(y0 <= 3) + int(x1 >= w - 4) + int(y1 >= h - 4)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    dark_ratio = float((gray < 30).mean())
    bright_ratio = float((gray > 235).mean())
    glare_ratio = float(((gray > 235) & (hsv[:, :, 1] < 35)).mean())
    saturation = hsv[:, :, 1].astype(float)

    border = np.concatenate([rgb[:20].reshape(-1, 3), rgb[-20:].reshape(-1, 3), rgb[:, :20].reshape(-1, 3), rgb[:, -20:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb.astype(float) - bg.reshape(1, 1, 3), axis=2)

    return {
        "width": float(w),
        "height": float(h),
        "megapixels": float(w * h / 1_000_000),
        "file_mb": float(path.stat().st_size / 1024 / 1024),
        "fg_ratio": fg_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "touch_edges": float(touch_edges),
        "lap_var": lap_var,
        "tenengrad": tenengrad,
        "brightness_mean": brightness,
        "contrast": contrast,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "glare_ratio": glare_ratio,
        "saturation_mean": float(saturation.mean()),
        "bg_r": float(bg[0]),
        "bg_g": float(bg[1]),
        "bg_b": float(bg[2]),
        "dist_p90": float(np.percentile(dist, 90)),
        "sat_mean": float(saturation.mean()),
    }


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["main_prob"] = out["prob_base162"].astype(float)
    out["main_pred"] = (out["main_prob"] >= MAIN_THRESHOLD).astype(int)
    out["main_margin_abs"] = (out["main_prob"] - 0.5).abs()
    out["robust_prob"] = out[["prob_base162", "prob103_vitl", "prob_mean_core"]].mean(axis=1).astype(float)
    out["robust_pred"] = (out["robust_prob"] >= ROBUST_THRESHOLD).astype(int)
    out["robust_margin_abs"] = (out["robust_prob"] - 0.5).abs()
    out["main_robust_abs_diff"] = (out["main_prob"] - out["robust_prob"]).abs()
    out["main_robust_disagree"] = out["main_pred"].ne(out["robust_pred"]).astype(int)
    out["low_main_high_robust"] = ((out["main_pred"] == 0) & (out["robust_pred"] == 1)).astype(int)
    out["high_main_low_robust"] = ((out["main_pred"] == 1) & (out["robust_pred"] == 0)).astype(int)
    out["safety_trigger"] = out["low_main_high_robust"]
    out["p2_pred"] = np.where(out["safety_trigger"].eq(1), out["robust_pred"], out["main_pred"]).astype(int)
    out["p2_wrong"] = out["p2_pred"].ne(out["label_idx"].astype(int)).astype(int)
    return out


def load_development() -> pd.DataFrame:
    df = pd.read_csv(V2_DIR / "v2_development_diagnostic_table.csv")
    df = add_model_features(df)
    if not USE_SLOW_DEV_IMAGE_EXTRACTION:
        return df
    cache = OUT_DIR / "v30_dev_image_features.csv"
    if cache.exists():
        feat = pd.read_csv(cache)
    else:
        rows = []
        for _, row in df.iterrows():
            stats = image_stats(row.get("analysis_image_path", row.get("path")))
            stats["case_id"] = row["case_id"]
            rows.append(stats)
        feat = pd.DataFrame(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        feat.to_csv(cache, index=False, encoding="utf-8-sig")
    existing = [c for c in IMAGE_FEATURES if c in df.columns]
    df = df.drop(columns=existing, errors="ignore").merge(feat, on="case_id", how="left")
    return df


def load_external() -> pd.DataFrame:
    df = pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv")
    df = add_model_features(df)
    q = pd.read_csv(EXT_QUALITY_DIR / "external_quality_features_v0.csv")
    merge_cols = ["image_name"] + [c for c in IMAGE_FEATURES if c in q.columns]
    q = q[merge_cols].drop_duplicates("image_name")
    df = df.drop(columns=[c for c in IMAGE_FEATURES if c in df.columns], errors="ignore").merge(q, on="image_name", how="left")
    return df


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


def make_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    linear_pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    tree_pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return {
        "hard_logistic": Pipeline(
            [
                ("prep", linear_pre),
                ("clf", LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, random_state=20260527)),
            ]
        ),
        "hard_rf": Pipeline(
            [
                ("prep", tree_pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=20260527,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hard_extra_trees": Pipeline(
            [
                ("prep", tree_pre),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        max_depth=4,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=20260527,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hard_gbdt": Pipeline(
            [
                ("prep", tree_pre),
                ("clf", GradientBoostingClassifier(n_estimators=120, learning_rate=0.04, max_depth=2, random_state=20260527)),
            ]
        ),
    }


def oof_and_external_scores(dev: pd.DataFrame, ext: pd.DataFrame, features: list[str], model: Pipeline) -> tuple[np.ndarray, np.ndarray]:
    y = dev["p2_wrong"].to_numpy(dtype=int)
    oof = np.zeros(len(dev), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260527)
    for tr, te in cv.split(dev[features], y):
        model.fit(dev.iloc[tr][features], y[tr])
        oof[te] = model.predict_proba(dev.iloc[te][features])[:, 1]
    model.fit(dev[features], y)
    ext_score = model.predict_proba(ext[features])[:, 1]
    return oof, ext_score


def evaluate_review_policy(df: pd.DataFrame, review_flag: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2_pred = df["p2_pred"].to_numpy(dtype=int)
    p2_wrong = df["p2_wrong"].to_numpy(dtype=bool)
    review = np.asarray(review_flag, dtype=bool)
    final_pred = p2_pred.copy()
    final_pred[review] = y[review]
    m = metrics_binary(y, final_pred)
    m.update(
        {
            "n": int(len(df)),
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_p2_wrong_n": int((review & p2_wrong).sum()),
            "captured_p2_wrong_rate": float((review & p2_wrong).sum() / p2_wrong.sum()) if p2_wrong.sum() else 0.0,
            "review_on_p2_correct_n": int((review & ~p2_wrong).sum()),
            "review_precision_vs_p2_error": float((review & p2_wrong).sum() / review.sum()) if review.sum() else 0.0,
            "missed_p2_wrong_n": int((~review & p2_wrong).sum()),
        }
    )
    return m


def top_budget(scores: np.ndarray, budget: float) -> np.ndarray:
    n_review = int(round(len(scores) * budget))
    if n_review <= 0:
        return np.zeros(len(scores), dtype=bool)
    order = np.argsort(-scores)
    flag = np.zeros(len(scores), dtype=bool)
    flag[order[:n_review]] = True
    return flag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_development()
    ext = load_external()

    numeric = [c for c in PROB_FEATURES + IMAGE_FEATURES + BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical

    rows = []
    score_frames = []
    for model_name, model in make_models(numeric, categorical).items():
        oof, ext_score = oof_and_external_scores(dev, ext, features, model)
        try:
            dev_auc = roc_auc_score(dev["p2_wrong"], oof)
            ext_auc = roc_auc_score(ext["p2_wrong"], ext_score)
        except ValueError:
            dev_auc = float("nan")
            ext_auc = float("nan")

        score_frames.append(
            pd.DataFrame(
                {
                    "case_id": ext["case_id"],
                    "original_case_id": ext["original_case_id"],
                    "task_l6_label": ext["task_l6_label"],
                    "label_idx": ext["label_idx"],
                    "p2_pred": ext["p2_pred"],
                    "p2_wrong": ext["p2_wrong"],
                    "model": model_name,
                    "hard_risk_score": ext_score,
                }
            )
        )

        for budget in BUDGETS:
            dev_review = top_budget(oof, budget)
            ext_review = top_budget(ext_score, budget)
            dev_m = evaluate_review_policy(dev, dev_review)
            ext_m = evaluate_review_policy(ext, ext_review)
            rows.append(
                {
                    "model": model_name,
                    "selection": "fixed_review_budget_no_external_labels",
                    "budget": budget,
                    "dev_auc": dev_auc,
                    "external_auc": ext_auc,
                    **{f"dev_{k}": v for k, v in dev_m.items()},
                    **{f"external_{k}": v for k, v in ext_m.items()},
                }
            )

    metrics = pd.DataFrame(rows)
    scores = pd.concat(score_frames, ignore_index=True)
    metrics.to_csv(OUT_DIR / "v30_dev_trained_hard_gate_metrics.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUT_DIR / "v30_external_hard_gate_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features, "type": ["categorical" if f in categorical else "numeric" for f in features]}).to_csv(
        OUT_DIR / "v30_hard_gate_features.csv", index=False, encoding="utf-8-sig"
    )

    selected_scores = scores.loc[scores["model"].eq("hard_logistic")].copy()
    selected_cases = []
    ext_case_cols = [
        "case_id",
        "original_case_id",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "image_name",
        "quality_status",
        "quality_score",
        "manual_quality_status_v1",
        "main_prob",
        "main_pred",
        "robust_prob",
        "robust_pred",
        "prob_mean_core",
        "p2_pred",
        "p2_wrong",
    ]
    ext_lookup = ext[[c for c in ext_case_cols if c in ext.columns]].copy()
    selected_scores = selected_scores[["case_id", "hard_risk_score"]].merge(ext_lookup, on="case_id", how="left")
    for budget in [0.50, 0.60, 0.75]:
        flag = top_budget(selected_scores["hard_risk_score"].to_numpy(), budget)
        tmp = selected_scores.copy()
        tmp.insert(1, "budget", budget)
        tmp["review_flag"] = flag.astype(int)
        tmp["bucket"] = np.select(
            [
                tmp["review_flag"].eq(1) & tmp["p2_wrong"].eq(1),
                tmp["review_flag"].eq(1) & tmp["p2_wrong"].eq(0),
                tmp["review_flag"].eq(0) & tmp["p2_wrong"].eq(1),
            ],
            ["captured_p2_error", "review_on_p2_correct", "missed_p2_error"],
            default="auto_correct",
        )
        selected_cases.append(tmp)
    pd.concat(selected_cases, ignore_index=True).sort_values(["budget", "review_flag", "hard_risk_score"], ascending=[True, False, False]).to_csv(
        OUT_DIR / "v30_hard_logistic_selected_case_routes.csv", index=False, encoding="utf-8-sig"
    )

    view_cols = [
        "model",
        "budget",
        "dev_auc",
        "external_auc",
        "dev_balanced_accuracy",
        "dev_captured_p2_wrong_rate",
        "external_balanced_accuracy",
        "external_review_rate",
        "external_captured_p2_wrong_rate",
        "external_review_precision_vs_p2_error",
        "external_missed_p2_wrong_n",
        "external_fn",
        "external_fp",
    ]
    print(metrics[view_cols].sort_values(["external_balanced_accuracy", "external_review_rate"], ascending=[False, True]).head(20).to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
