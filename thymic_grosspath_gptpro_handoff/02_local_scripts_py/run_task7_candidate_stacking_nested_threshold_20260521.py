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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested-threshold OOF stacking over Task7 candidate outputs.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
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
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def split_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, prob)) if prob is not None else float("nan")
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


def make_models(seed: int) -> dict[str, object]:
    return {
        "logreg_c003": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "logreg_c01": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "logreg_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "extra_d2_l10": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=2,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_d3_l8": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "rf_d3_l8": RandomForestClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb_d1_lr03": GradientBoostingClassifier(max_depth=1, learning_rate=0.03, n_estimators=120, random_state=seed),
        "gb_d1_lr05": GradientBoostingClassifier(max_depth=1, learning_rate=0.05, n_estimators=80, random_state=seed),
        "gb_d2_lr03": GradientBoostingClassifier(max_depth=2, learning_rate=0.03, n_estimators=80, random_state=seed),
        "hgb_leaf3": HistGradientBoostingClassifier(
            max_leaf_nodes=3,
            l2_regularization=0.1,
            min_samples_leaf=20,
            max_iter=60,
            learning_rate=0.08,
            random_state=seed,
        ),
        "hgb_leaf5": HistGradientBoostingClassifier(
            max_leaf_nodes=5,
            l2_regularization=0.1,
            min_samples_leaf=24,
            max_iter=60,
            learning_rate=0.08,
            random_state=seed,
        ),
    }


def load_candidates(df: pd.DataFrame, runs_root: Path, candidate_dirs: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    pred_cols: list[str] = []
    prob_cols: list[str] = []
    for idx, name in enumerate(candidate_dirs):
        path = runs_root / name / "best_case_outputs_full.csv"
        cand = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
        aligned = df[["case_id"]].merge(cand[["case_id", "final_pred", "final_prob_high"]], on="case_id", how="left")
        prefix = f"cand{idx}_{name.split('_')[0]}"
        pred_name = f"{prefix}_pred"
        prob_name = f"{prefix}_prob"
        out[pred_name] = pd.to_numeric(aligned["final_pred"], errors="coerce").fillna(0.0).astype(float)
        out[prob_name] = pd.to_numeric(aligned["final_prob_high"], errors="coerce").fillna(0.5).astype(float)
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


def build_model_output_features(df: pd.DataFrame) -> pd.DataFrame:
    p_cols = [c for c in df.columns if c.startswith("p_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    out = pd.DataFrame(index=df.index)
    for col in p_cols:
        p = pd.to_numeric(df[col], errors="coerce").fillna(0.5).astype(float)
        out[col] = p
        out[f"{col}_margin"] = (p - 0.5).abs()
    for col in pred_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    pmat = out[p_cols].to_numpy(dtype=float)
    out["raw_prob_mean"] = pmat.mean(axis=1)
    out["raw_prob_std"] = pmat.std(axis=1)
    out["raw_prob_min"] = pmat.min(axis=1)
    out["raw_prob_max"] = pmat.max(axis=1)
    votes = (pmat >= 0.5).astype(float)
    out["raw_vote_frac"] = votes.mean(axis=1)
    out["raw_vote_disagree"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def predict_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    clf = clone(model)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def threshold_candidates(prob: np.ndarray) -> np.ndarray:
    return np.unique(np.r_[np.linspace(0.25, 0.75, 101), prob])


def choose_threshold(y: np.ndarray, prob: np.ndarray, mode: str) -> tuple[float, dict[str, Any]]:
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for thr in threshold_candidates(prob):
        pred = (prob >= thr).astype(int)
        row = metric_row(y, pred, prob)
        if mode == "bacc":
            key = (row["balanced_accuracy"], row["accuracy"], row["f1"])
        elif mode == "acc":
            key = (row["accuracy"], row["balanced_accuracy"], row["f1"])
        elif mode == "f1":
            key = (row["f1"], row["balanced_accuracy"], row["accuracy"])
        elif mode == "bacc_sens80":
            penalty = min(0.0, row["sensitivity"] - 0.80)
            key = (row["balanced_accuracy"] + penalty, row["accuracy"], row["f1"])
        elif mode == "acc_sens80":
            penalty = min(0.0, row["sensitivity"] - 0.80)
            key = (row["accuracy"] + penalty, row["balanced_accuracy"], row["f1"])
        else:
            raise ValueError(mode)
        if best is None or key > best[0]:
            best = (key, float(thr), row)
    assert best is not None
    return best[1], best[2]


def nested_train_threshold(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    outer_train: np.ndarray,
    model: object,
    mode: str,
) -> tuple[float, dict[str, Any]]:
    inner_prob = np.full(len(y), np.nan, dtype=float)
    inner_folds = sorted(np.unique(folds[outer_train]))
    for inner_fold in inner_folds:
        inner_test = outer_train & (folds == inner_fold)
        inner_train = outer_train & (folds != inner_fold)
        inner_prob[inner_test] = predict_prob(model, x[inner_train], y[inner_train], x[inner_test])
    valid = outer_train & np.isfinite(inner_prob)
    threshold, row = choose_threshold(y[valid], inner_prob[valid], mode)
    row = {f"inner_{k}": v for k, v in row.items()}
    row["threshold"] = threshold
    return threshold, row


def evaluate(df: pd.DataFrame, x_df: pd.DataFrame, model_name: str, model: object, threshold_mode: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)
    final_prob = np.zeros(len(df), dtype=float)
    final_pred = np.zeros(len(df), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    for outer_fold in sorted(np.unique(folds)):
        train = folds != outer_fold
        test = folds == outer_fold
        threshold, inner_row = nested_train_threshold(x, y, folds, train, model, threshold_mode)
        prob = predict_prob(model, x[train], y[train], x[test])
        pred = (prob >= threshold).astype(int)
        final_prob[test] = prob
        final_pred[test] = pred
        row = metric_row(y[test], pred, prob)
        row["fold_id"] = int(outer_fold)
        row["threshold"] = threshold
        row.update(inner_row)
        fold_rows.append(row)
    summary = metric_row(y, final_pred, final_prob)
    summary["model"] = model_name
    summary["threshold_mode"] = threshold_mode
    case_df = df[["case_id", "original_case_id", "fold_id", "label_idx", "pred_upper", "p_upper"]].copy()
    case_df["final_pred"] = final_pred
    case_df["final_prob_high"] = final_prob
    case_df["model"] = model_name
    case_df["threshold_mode"] = threshold_mode
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
    candidates = load_candidates(df, Path(args.runs_root), split_arg(args.candidate_dirs))
    raw_outputs = build_model_output_features(df)
    feature_sets = {
        "candidates_only": candidates,
        "candidates_plus_raw_outputs": pd.concat([candidates, raw_outputs], axis=1),
    }
    models = make_models(args.seed)
    threshold_modes = ["bacc", "acc", "f1", "bacc_sens80", "acc_sens80"]
    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    folds_out: list[pd.DataFrame] = []
    for feature_name, x_df in feature_sets.items():
        for model_name, model in models.items():
            for threshold_mode in threshold_modes:
                summary, case_df, fold_df = evaluate(df, x_df, model_name, model, threshold_mode)
                summary["feature_set"] = feature_name
                summaries.append(summary)
                case_df["feature_set"] = feature_name
                fold_df["feature_set"] = feature_name
                fold_df["model"] = model_name
                fold_df["threshold_mode"] = threshold_mode
                cases.append(case_df)
                folds_out.append(fold_df)
    summary_df = pd.DataFrame(summaries).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(out_dir / "nested_threshold_stacking_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "nested_threshold_stacking_case_outputs.csv", index=False)
    pd.concat(folds_out, ignore_index=True).to_csv(out_dir / "nested_threshold_stacking_fold_metrics.csv", index=False)
    print(summary_df.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
