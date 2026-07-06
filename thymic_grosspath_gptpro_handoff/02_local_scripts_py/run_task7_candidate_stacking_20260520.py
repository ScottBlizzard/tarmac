from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_task7_stage2_auto_corrector_20260520 import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict OOF stacking over Task7 corrected candidate outputs.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument("--runs-root", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs")
    parser.add_argument(
        "--candidate-dirs",
        default=(
            "29_best_score_split_corrector_hybrid065_gb_flagtrain_20260520,"
            "36_best_hgb_score_split_corrector_20260520,"
            "38_best_hgb_refine_score_split_corrector_20260520,"
            "39_hgb_dual_tiebreak_prefer_low_20260520"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, prob))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": auc,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def split_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "logreg_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "extra_d2_l10": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=2,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_d3_l8": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "rf_d3_l8": RandomForestClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb_d1_lr05": GradientBoostingClassifier(max_depth=1, learning_rate=0.05, n_estimators=80, random_state=seed),
        "gb_d2_lr035": GradientBoostingClassifier(max_depth=2, learning_rate=0.035, n_estimators=80, random_state=seed),
        "hgb_leaf3_l2": HistGradientBoostingClassifier(max_leaf_nodes=3, l2_regularization=0.1, min_samples_leaf=20, max_iter=60, learning_rate=0.08, random_state=seed),
        "hgb_leaf5_l2": HistGradientBoostingClassifier(max_leaf_nodes=5, l2_regularization=0.1, min_samples_leaf=24, max_iter=60, learning_rate=0.08, random_state=seed),
    }
    return models


def load_candidates(df: pd.DataFrame, runs_root: Path, candidate_dirs: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    pred_cols: list[str] = []
    prob_cols: list[str] = []
    for idx, name in enumerate(candidate_dirs):
        path = runs_root / name / "best_case_outputs_full.csv"
        cand = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
        prob_col = "final_prob_high"
        pred_col = "final_pred"
        aligned = df[["case_id"]].merge(cand[["case_id", pred_col, prob_col]], on="case_id", how="left")
        prefix = f"cand{idx}_{name.split('_')[0]}"
        pred_name = f"{prefix}_pred"
        prob_name = f"{prefix}_prob"
        out[pred_name] = pd.to_numeric(aligned[pred_col], errors="coerce").fillna(0.0).astype(float)
        out[prob_name] = pd.to_numeric(aligned[prob_col], errors="coerce").fillna(0.5).astype(float)
        out[f"{prefix}_conf"] = np.where(out[pred_name] >= 0.5, out[prob_name], 1.0 - out[prob_name])
        pred_cols.append(pred_name)
        prob_cols.append(prob_name)
    pred_values = out[pred_cols].to_numpy(dtype=float)
    prob_values = out[prob_cols].to_numpy(dtype=float)
    out["cand_vote_frac"] = pred_values.mean(axis=1)
    out["cand_vote_disagree"] = ((pred_values.sum(axis=1) > 0) & (pred_values.sum(axis=1) < pred_values.shape[1])).astype(float)
    out["cand_prob_mean"] = prob_values.mean(axis=1)
    out["cand_prob_std"] = prob_values.std(axis=1)
    out["cand_prob_min"] = prob_values.min(axis=1)
    out["cand_prob_max"] = prob_values.max(axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_predict(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = clone(model)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    return (prob >= 0.5).astype(int), prob


def evaluate(df: pd.DataFrame, x_df: pd.DataFrame, model_name: str, model: object) -> tuple[dict[str, Any], pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros(len(df), dtype=float)
    x = x_df.to_numpy(dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        pred[test], prob[test] = fit_predict(model, x[train], y[train], x[test])
        row = metric_row(y[test], pred[test], prob[test])
        row["fold_id"] = int(fold)
        fold_rows.append(row)
    summary = metric_row(y, pred, prob)
    summary["model"] = model_name
    case_df = df[["case_id", "original_case_id", "fold_id", "label_idx", "pred_upper", "p_upper"]].copy()
    case_df["final_pred"] = pred
    case_df["final_prob_high"] = prob
    case_df["model"] = model_name
    case_df["base_wrong"] = (case_df["pred_upper"].astype(int) != case_df["label_idx"].astype(int)).astype(int)
    case_df["final_wrong"] = (case_df["final_pred"].astype(int) != case_df["label_idx"].astype(int)).astype(int)
    case_df["rescued"] = ((case_df["base_wrong"] == 1) & (case_df["final_wrong"] == 0)).astype(int)
    case_df["hurt"] = ((case_df["base_wrong"] == 0) & (case_df["final_wrong"] == 1)).astype(int)
    return summary, case_df, pd.DataFrame(fold_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    runs_root = Path(args.runs_root)
    candidate_dirs = split_arg(args.candidate_dirs)
    candidate_x = load_candidates(df, runs_root, candidate_dirs)
    raw_sets = build_features(df, Path(args.registry_csv))
    feature_sets = {
        "candidates_only": candidate_x,
        "candidates_plus_raw_review": pd.concat([candidate_x, raw_sets["raw_plus_review"]], axis=1),
        "candidates_plus_raw_review_gross": pd.concat([candidate_x, raw_sets["raw_plus_review_gross"]], axis=1),
    }
    models = make_models(args.seed)
    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for feature_name, x_df in feature_sets.items():
        for model_name, model in models.items():
            summary, case_df, fold_df = evaluate(df, x_df, model_name, model)
            summary["feature_set"] = feature_name
            summaries.append(summary)
            case_df["feature_set"] = feature_name
            fold_df["feature_set"] = feature_name
            fold_df["model"] = model_name
            cases.append(case_df)
            folds.append(fold_df)
    summary_df = pd.DataFrame(summaries).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(out_dir / "candidate_stacking_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "candidate_stacking_case_outputs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out_dir / "candidate_stacking_fold_metrics.csv", index=False)
    print(summary_df.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
