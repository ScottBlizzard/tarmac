#!/usr/bin/env python3
"""Train locked Task7 meta-fusion models from old+third OOF predictions.

This experiment deliberately uses only old_data + third_batch labels for model
selection. strict_external and new_external_160 are evaluated once after the
meta-model and threshold are locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIRS = {
    "105": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/105_dinov3_vitl16_qkvb_task7_whole352_token_lastblock_lowlr_5fold_20260524",
    "126": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/126_dinov3_vitl16_qkvb_task7_whole352_token_headonly_5fold_20260524",
    "149": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/149_dinov3_vitl16_qkvb_task7_whole352_stylelight_lastblock_5fold_20260524",
}

EXTERNAL_PREDS = {
    "strict_external": {
        "105": "experiments/base_model_expansion_20260706/outputs/strict_external_inference/105_qkvb_tta4/external_tta_predictions.csv",
        "126": "experiments/base_model_expansion_20260706/outputs/strict_external_inference/126_qkvb_headonly_tta4/external_tta_predictions.csv",
        "149": "experiments/base_model_expansion_20260706/outputs/strict_external_inference/149_qkvb_stylelight_tta4/external_tta_predictions.csv",
    },
    "new_external_160": {
        "105": "experiments/base_model_expansion_20260706/outputs/new_external_inference/105_qkvb_tta4/external_tta_predictions.csv",
        "126": "experiments/base_model_expansion_20260706/outputs/new_external_inference/126_qkvb_headonly_tta4/external_tta_predictions.csv",
        "149": "experiments/base_model_expansion_20260706/outputs/new_external_inference/149_qkvb_stylelight_tta4/external_tta_predictions.csv",
    },
}

ID_COLS = {"case_id", "domain", "label_idx", "task_l6_label"}


def positive_prob_column(df: pd.DataFrame) -> str:
    for col in ("prob_high_risk_group", "prob_1", "high_prob", "main_prob"):
        if col in df.columns:
            return col
    raise ValueError(f"No high-risk probability column found in {list(df.columns)}")


def add_probability_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prob_cols = ["p105", "p126", "p149"]
    clipped = out[prob_cols].clip(1e-6, 1 - 1e-6)
    for col in prob_cols:
        out[f"logit_{col}"] = np.log(clipped[col] / (1.0 - clipped[col]))
    out["p_avg_105_126_149"] = out[prob_cols].mean(axis=1)
    out["p_median_105_126_149"] = out[prob_cols].median(axis=1)
    out["p_min_105_126_149"] = out[prob_cols].min(axis=1)
    out["p_max_105_126_149"] = out[prob_cols].max(axis=1)
    out["p_std_105_126_149"] = out[prob_cols].std(axis=1)
    out["p_range_105_126_149"] = out["p_max_105_126_149"] - out["p_min_105_126_149"]
    out["p_avg_105_149"] = out[["p105", "p149"]].mean(axis=1)
    out["p_avg_105_126"] = out[["p105", "p126"]].mean(axis=1)
    out["p_avg_126_149"] = out[["p126", "p149"]].mean(axis=1)
    return out


def load_old_third_oof(project_root: Path) -> pd.DataFrame:
    merged = None
    meta_cols = [
        "case_id",
        "label_idx",
        "fold_id",
        "original_case_id",
        "source_dataset",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "master_fold_id",
    ]
    for tag, run_dir in RUN_DIRS.items():
        path = project_root / run_dir / "oof_case_predictions_mean.csv"
        df = pd.read_csv(path)
        prob_col = positive_prob_column(df)
        cols = [c for c in meta_cols if c in df.columns]
        keep = df[cols + [prob_col]].rename(columns={prob_col: f"p{tag}"})
        if merged is None:
            merged = keep
        else:
            key = ["case_id", "label_idx"]
            merged = merged.merge(keep[key + [f"p{tag}"]], on=key, how="inner")
    assert merged is not None
    if len(merged) != 591:
        print(f"[warn] expected 591 old+third OOF rows, got {len(merged)}")
    return add_probability_features(merged)


def load_external_predictions(project_root: Path, domain: str) -> pd.DataFrame:
    merged = None
    for tag, rel_path in EXTERNAL_PREDS[domain].items():
        df = pd.read_csv(project_root / rel_path)
        prob_col = positive_prob_column(df)
        keep_cols = [
            "case_id",
            "label_idx",
            "domain",
            "task_l6_label",
            "task_l7_label",
            "risk_label",
            prob_col,
        ]
        keep = df[[c for c in keep_cols if c in df.columns]].rename(columns={prob_col: f"p{tag}"})
        if merged is None:
            merged = keep
        else:
            merged = merged.merge(keep[["case_id", "label_idx", f"p{tag}"]], on=["case_id", "label_idx"], how="inner")
    assert merged is not None
    return add_probability_features(merged)


def load_roi_features(project_root: Path, roi_csv: Path) -> pd.DataFrame:
    roi = pd.read_csv(project_root / roi_csv)
    roi_features = [
        c
        for c in roi.columns
        if c not in ID_COLS and pd.api.types.is_numeric_dtype(roi[c])
    ]
    keep = roi[["case_id"] + roi_features].copy()
    keep = keep.rename(columns={c: f"roi__{c}" for c in roi_features})
    return keep


def select_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    values = np.unique(np.r_[0.0, 1.0, prob])
    mids = (values[:-1] + values[1:]) / 2 if len(values) > 1 else values
    grid = np.unique(np.r_[values, mids])
    best_threshold = 0.5
    best_row: Dict[str, float] | None = None
    best_key = None
    for threshold in grid:
        row = compute_metrics(y_true, prob, threshold)
        key = (
            row["balanced_accuracy"],
            row["accuracy"],
            -abs(row["high_recall"] - row["low_specificity"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_row = row
    assert best_row is not None
    return best_threshold, best_row


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) == 2 else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "high_recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "low_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def model_specs(random_state: int) -> Dict[str, Pipeline]:
    return {
        "logreg_balanced_l2": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logreg_balanced_l1": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.2,
                        class_weight="balanced",
                        max_iter=5000,
                        penalty="l1",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def cv_predict(
    x: pd.DataFrame,
    y: np.ndarray,
    model: Pipeline,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    prob = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in cv.split(x, y):
        fitted = clone(model)
        fitted.fit(x.iloc[train_idx], y[train_idx])
        prob[val_idx] = fitted.predict_proba(x.iloc[val_idx])[:, 1]
    return prob


def feature_columns(frame: pd.DataFrame, feature_set: str) -> List[str]:
    prob_cols = [
        c
        for c in frame.columns
        if c.startswith("p")
        or c.startswith("logit_p")
    ]
    roi_cols = [c for c in frame.columns if c.startswith("roi__")]
    if feature_set == "qkvb_only":
        return prob_cols
    if feature_set == "roi_only":
        return roi_cols
    if feature_set == "qkvb_roi":
        return prob_cols + roi_cols
    raise ValueError(f"unknown feature set: {feature_set}")


def evaluate_by_subtype(df: pd.DataFrame, prob: np.ndarray, threshold: float, model_tag: str, domain: str) -> List[Dict[str, object]]:
    pred = (prob >= threshold).astype(int)
    tmp = df.copy()
    tmp["pred_idx"] = pred
    tmp["locked_prob_high"] = prob
    rows: List[Dict[str, object]] = []
    if "task_l6_label" not in tmp.columns:
        return rows
    for subtype, group in tmp.groupby("task_l6_label", dropna=False):
        rows.append(
            {
                "domain": domain,
                "model_tag": model_tag,
                "task_l6_label": subtype,
                "n": int(len(group)),
                "accuracy": float(accuracy_score(group["label_idx"].astype(int), group["pred_idx"].astype(int))),
                "mean_locked_prob_high": float(group["locked_prob_high"].mean()),
                "pred_high_count": int((group["pred_idx"] == 1).sum()),
                "pred_low_count": int((group["pred_idx"] == 0).sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--roi-csv",
        type=Path,
        default=Path("experiments/base_model_expansion_20260706/outputs/roi_stats_probe/roi_stats_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/base_model_expansion_20260706/outputs/qkvb_roi_meta_fusion"),
    )
    parser.add_argument("--random-state", type=int, default=20260706)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roi = load_roi_features(project_root, args.roi_csv)
    train = load_old_third_oof(project_root).merge(roi, on="case_id", how="left")
    externals = {
        domain: load_external_predictions(project_root, domain).merge(roi, on="case_id", how="left")
        for domain in EXTERNAL_PREDS
    }

    specs = model_specs(args.random_state)
    feature_sets = ["qkvb_only", "qkvb_roi", "roi_only"]
    y_train = train["label_idx"].astype(int).to_numpy()

    rows: List[Dict[str, object]] = []
    subtype_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    selected_models = []

    for feature_set in feature_sets:
        cols = feature_columns(train, feature_set)
        if not cols:
            continue
        x_train = train[cols]
        for model_name, spec in specs.items():
            if feature_set == "roi_only" and model_name.startswith("logreg"):
                # ROI-only logistic is covered by the earlier probe; keep this run focused.
                continue
            tag = f"{feature_set}__{model_name}"
            oof_prob = cv_predict(x_train, y_train, spec, args.n_splits, args.random_state)
            locked_threshold, dev_selected = select_threshold(y_train, oof_prob)
            for threshold_source, threshold in [("fixed_0.5", 0.5), ("old_third_meta_oof_selected", locked_threshold)]:
                row = compute_metrics(y_train, oof_prob, threshold)
                row.update(
                    {
                        "domain": "old_third_meta_oof",
                        "model_tag": tag,
                        "feature_set": feature_set,
                        "base_model": model_name,
                        "threshold_source": threshold_source,
                        "n_features": len(cols),
                    }
                )
                rows.append(row)

            final_model = clone(spec)
            final_model.fit(x_train, y_train)
            joblib.dump(
                {
                    "model": final_model,
                    "feature_columns": cols,
                    "threshold": locked_threshold,
                    "feature_set": feature_set,
                    "base_model": model_name,
                },
                output_dir / f"{tag}.joblib",
            )
            selected_models.append(
                {
                    "model_tag": tag,
                    "feature_set": feature_set,
                    "base_model": model_name,
                    "threshold": locked_threshold,
                    "old_third_oof_balanced_accuracy": dev_selected["balanced_accuracy"],
                    "old_third_oof_auc": dev_selected["auc"],
                    "n_features": len(cols),
                }
            )

            base_pred = train[["case_id", "label_idx", "task_l6_label"]].copy()
            base_pred["domain"] = "old_third_meta_oof"
            base_pred["model_tag"] = tag
            base_pred["locked_prob_high"] = oof_prob
            base_pred["locked_threshold"] = locked_threshold
            base_pred["pred_idx"] = (oof_prob >= locked_threshold).astype(int)
            pred_frames.append(base_pred)
            subtype_rows.extend(evaluate_by_subtype(train, oof_prob, locked_threshold, tag, "old_third_meta_oof"))

            for domain, ext in externals.items():
                x_ext = ext[cols]
                prob = final_model.predict_proba(x_ext)[:, 1]
                for threshold_source, threshold in [("fixed_0.5", 0.5), ("old_third_meta_oof_selected", locked_threshold)]:
                    row = compute_metrics(ext["label_idx"].astype(int).to_numpy(), prob, threshold)
                    row.update(
                        {
                            "domain": domain,
                            "model_tag": tag,
                            "feature_set": feature_set,
                            "base_model": model_name,
                            "threshold_source": threshold_source,
                            "n_features": len(cols),
                        }
                    )
                    rows.append(row)
                ext_pred = ext[["case_id", "label_idx", "task_l6_label"]].copy()
                ext_pred["domain"] = domain
                ext_pred["model_tag"] = tag
                ext_pred["locked_prob_high"] = prob
                ext_pred["locked_threshold"] = locked_threshold
                ext_pred["pred_idx"] = (prob >= locked_threshold).astype(int)
                pred_frames.append(ext_pred)
                subtype_rows.extend(evaluate_by_subtype(ext, prob, locked_threshold, tag, domain))

    metrics = pd.DataFrame(rows).sort_values(["domain", "balanced_accuracy"], ascending=[True, False])
    subtype = pd.DataFrame(subtype_rows).sort_values(["domain", "model_tag", "task_l6_label"])
    predictions = pd.concat(pred_frames, ignore_index=True)
    selected = pd.DataFrame(selected_models).sort_values("old_third_oof_balanced_accuracy", ascending=False)

    metrics_path = output_dir / "qkvb_roi_meta_fusion_metrics.csv"
    subtype_path = output_dir / "qkvb_roi_meta_fusion_by_task_l6_label.csv"
    pred_path = output_dir / "qkvb_roi_meta_fusion_predictions.csv"
    selected_path = output_dir / "qkvb_roi_meta_fusion_selected_models.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    subtype.to_csv(subtype_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")

    report = {
        "method": "Meta-models trained on old+third OOF qkvb probabilities with optional ROI statistics; external labels were not used for model selection.",
        "train_rows": int(len(train)),
        "external_rows": {domain: int(len(df)) for domain, df in externals.items()},
        "outputs": {
            "metrics": str(metrics_path),
            "by_task_l6_label": str(subtype_path),
            "predictions": str(pred_path),
            "selected_models": str(selected_path),
        },
    }
    (output_dir / "qkvb_roi_meta_fusion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for domain in ["old_third_meta_oof", "strict_external", "new_external_160"]:
        print(f"\n[{domain}]")
        view = metrics[
            (metrics["domain"] == domain)
            & (metrics["threshold_source"] == "old_third_meta_oof_selected")
        ].sort_values("balanced_accuracy", ascending=False)
        cols = [
            "model_tag",
            "n",
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "auc",
            "high_recall",
            "low_specificity",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        print(view[cols].head(12).to_string(index=False))
    print(f"\n[ok] {metrics_path}")


if __name__ == "__main__":
    main()
