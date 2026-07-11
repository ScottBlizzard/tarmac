from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_image_resized(path: str, max_dim: int) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        im.draft("RGB", (max_dim, max_dim))
        im.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
        img = np.asarray(im.convert("RGB"))
    return img, orig_w, orig_h


def resize_for_stats(img: np.ndarray, max_dim: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return labels == idx


def safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "std", "p10", "p25", "p50", "p75", "p90"]}
    flat = values.reshape(-1).astype(np.float32)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_p10": float(np.percentile(flat, 10)),
        f"{prefix}_p25": float(np.percentile(flat, 25)),
        f"{prefix}_p50": float(np.percentile(flat, 50)),
        f"{prefix}_p75": float(np.percentile(flat, 75)),
        f"{prefix}_p90": float(np.percentile(flat, 90)),
    }


def extract_features(row: pd.Series, max_dim: int) -> dict[str, Any]:
    image_path = str(row["image_path"])
    img, w0, h0 = read_image_resized(image_path, max_dim=max_dim)
    h, w = img.shape[:2]
    rgb = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch = hsv[:, :, 0] / 179.0
    s_ch = hsv[:, :, 1] / 255.0
    v_ch = hsv[:, :, 2] / 255.0
    gray_f = gray.astype(np.float32) / 255.0

    raw_mask = ((s_ch > 0.10) & (v_ch < 0.985)) | ((gray_f < 0.93) & (s_ch > 0.035))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = largest_component(mask.astype(bool))
    if mask.mean() < 0.005:
        mask = np.ones((h, w), dtype=bool)

    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    fg_area = float(mask.mean())
    bbox_area = float((bbox_w * bbox_h) / max(h * w, 1))
    fg_pixels = rgb[mask]
    bg_pixels = rgb[~mask]
    fg_gray = gray_f[mask]
    bg_gray = gray_f[~mask]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    contour_area = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, closed=True))
        contour_area = float(cv2.contourArea(cnt))
    circularity = float(4.0 * np.pi * contour_area / (perimeter * perimeter + 1e-6))
    extent = float(contour_area / max(bbox_w * bbox_h, 1))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 50, 150)
    bbox_gray = gray[y1:y2, x1:x2]
    bbox_lap = cv2.Laplacian(bbox_gray, cv2.CV_32F)

    out: dict[str, Any] = {
        "case_id": row["case_id"],
        "domain": row["domain"],
        "label_idx": int(row["label_idx"]),
        "task_l6_label": row.get("task_l6_label", ""),
        "orig_width": int(w0),
        "orig_height": int(h0),
        "orig_aspect": float(w0 / max(h0, 1)),
        "work_width": int(w),
        "work_height": int(h),
        "fg_area_ratio": fg_area,
        "fg_bbox_area_ratio": bbox_area,
        "fg_bbox_x1_ratio": float(x1 / max(w, 1)),
        "fg_bbox_y1_ratio": float(y1 / max(h, 1)),
        "fg_bbox_w_ratio": float(bbox_w / max(w, 1)),
        "fg_bbox_h_ratio": float(bbox_h / max(h, 1)),
        "fg_bbox_aspect": float(bbox_w / max(bbox_h, 1)),
        "fg_centroid_x_ratio": float(xs.mean() / max(w, 1)),
        "fg_centroid_y_ratio": float(ys.mean() / max(h, 1)),
        "fg_circularity": circularity,
        "fg_extent": extent,
        "laplacian_var_whole": float(np.var(lap)),
        "laplacian_var_bbox": float(np.var(bbox_lap)),
        "edge_density_whole": float((edges > 0).mean()),
        "edge_density_fg": float((edges[mask] > 0).mean()) if mask.any() else np.nan,
        "bg_area_ratio": float((~mask).mean()),
    }
    for i, name in enumerate(["r", "g", "b"]):
        out.update(safe_stats(rgb[:, :, i], f"whole_{name}"))
        out.update(safe_stats(fg_pixels[:, i], f"fg_{name}"))
        out.update(safe_stats(bg_pixels[:, i], f"bg_{name}"))
    for arr, name in [(h_ch, "h"), (s_ch, "s"), (v_ch, "v"), (gray_f, "gray")]:
        out.update(safe_stats(arr, f"whole_{name}"))
        out.update(safe_stats(arr[mask], f"fg_{name}"))
        out.update(safe_stats(arr[~mask], f"bg_{name}"))
    out.update(safe_stats(fg_gray - np.nanmean(bg_gray) if bg_gray.size else fg_gray, "fg_gray_minus_bg_mean"))
    return out


def metric_dict(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_recall": float(tp / max(tp + fn, 1)),
        "low_specificity": float(tn / max(tn + fp, 1)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["auc"] = float("nan")
    return out


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best = -1.0
    for t in thresholds:
        score = balanced_accuracy_score(y, (prob >= t).astype(int))
        if score > best:
            best = float(score)
            best_t = float(t)
    return best_t, best


def model_specs(seed: int) -> dict[str, Any]:
    return {
        "logreg_l2_balanced": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.3,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_depth=4,
                        min_samples_leaf=6,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        max_depth=4,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry_csv)

    feature_rows = []
    failures = []
    for idx, row in registry.iterrows():
        try:
            feature_rows.append(extract_features(row, args.max_dim))
        except Exception as exc:
            failures.append({"case_id": row.get("case_id", ""), "image_path": row.get("image_path", ""), "error": str(exc)})
        if (idx + 1) % 100 == 0:
            print(f"[features] {idx + 1}/{len(registry)}", flush=True)
    features = pd.DataFrame(feature_rows)
    features.to_csv(out_dir / "roi_stats_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(out_dir / "roi_stats_failures.csv", index=False, encoding="utf-8-sig")
    if failures:
        raise RuntimeError(f"Feature extraction failed for {len(failures)} images; see roi_stats_failures.csv")

    meta_cols = {"case_id", "domain", "label_idx", "task_l6_label"}
    feature_cols = [c for c in features.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(features[c])]
    domains = sorted(features["domain"].unique().tolist())
    train_sets = {
        "train_old_only": ["old_data"],
        "train_old_plus_third": ["old_data", "third_batch"],
    }
    rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    models = model_specs(args.seed)
    for train_name, train_domains in train_sets.items():
        train_mask = features["domain"].isin(train_domains)
        x_train = features.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
        y_train = features.loc[train_mask, "label_idx"].astype(int).to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for model_name, estimator in models.items():
            print(f"[probe] {train_name} {model_name}", flush=True)
            oof_prob = cross_val_predict(clone(estimator), x_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            threshold, oof_score = choose_threshold(y_train, oof_prob)
            oof_metrics = metric_dict(y_train, oof_prob, threshold)
            rows.append(
                {
                    "train_config": train_name,
                    "model": model_name,
                    "eval_domain": "train_oof",
                    "train_domains": ",".join(train_domains),
                    "threshold_source": "train_5fold_oof_balanced_accuracy",
                    "oof_selection_score": oof_score,
                    **oof_metrics,
                }
            )
            fitted = clone(estimator).fit(x_train, y_train)
            joblib.dump(
                {"model": fitted, "feature_cols": feature_cols, "threshold": threshold, "train_domains": train_domains},
                out_dir / f"{train_name}_{model_name}.joblib",
            )
            for domain in domains:
                mask = features["domain"] == domain
                x_eval = features.loc[mask, feature_cols].to_numpy(dtype=np.float32)
                y_eval = features.loc[mask, "label_idx"].astype(int).to_numpy()
                prob = fitted.predict_proba(x_eval)[:, 1]
                metrics = metric_dict(y_eval, prob, threshold)
                rows.append(
                    {
                        "train_config": train_name,
                        "model": model_name,
                        "eval_domain": domain,
                        "train_domains": ",".join(train_domains),
                        "threshold_source": "train_5fold_oof_balanced_accuracy",
                        "oof_selection_score": oof_score,
                        **metrics,
                    }
                )
                pred = features.loc[mask, ["case_id", "domain", "label_idx", "task_l6_label"]].copy()
                pred["train_config"] = train_name
                pred["model"] = model_name
                pred["prob_high"] = prob
                pred["threshold"] = threshold
                pred["pred_idx"] = (prob >= threshold).astype(int)
                pred_rows.append(pred)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "roi_stats_probe_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(out_dir / "roi_stats_probe_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "registry_csv": args.registry_csv,
        "n_cases": int(len(features)),
        "n_features": int(len(feature_cols)),
        "domains": domains,
        "train_sets": train_sets,
        "boundary": "Strict/new external labels are used only for evaluation; thresholds are selected from training-domain 5-fold OOF predictions.",
    }
    (out_dir / "roi_stats_probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.sort_values(["train_config", "model", "eval_domain"]).to_string(index=False), flush=True)
    print(f"[ok] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
