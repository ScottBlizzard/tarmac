from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 candidate correctness selector.")
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
            "39_hgb_dual_tiebreak_prefer_low_20260520,"
            "41_best_candidate_stacking_balanced_20260520,"
            "43_best_candidate_stacking_no_tiebreak_20260520"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pools", default="raw_only,corrected_only,raw_plus_corrected")
    parser.add_argument("--feature-sets", default="candidate_full,candidate_agg")
    parser.add_argument("--models", default="logreg_c03,logreg_c1,extra_d2_l10,extra_d3_l8,rf_d3_l8,gb_d1_lr05,gb_d2_lr03")
    parser.add_argument("--modes", default="argmax_model,weighted_vote,trainacc_weighted_vote,top3_weighted_vote")
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


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


def split_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_models(seed: int) -> dict[str, object]:
    return {
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
        "gb_d2_lr03": GradientBoostingClassifier(max_depth=2, learning_rate=0.03, n_estimators=80, random_state=seed),
    }


def build_raw_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    rows: dict[str, pd.Series] = {}
    probs: dict[str, pd.Series] = {}
    names: list[str] = []
    for pred_col in pred_cols:
        suffix = pred_col[5:]
        prob_col = f"p_{suffix}"
        if prob_col not in df.columns:
            continue
        name = f"raw_{suffix}"
        rows[name] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).astype(float)
        probs[name] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.5).astype(float)
        names.append(name)
    return pd.DataFrame(rows), pd.DataFrame(probs), names


def build_corrected_candidates(df: pd.DataFrame, runs_root: Path, candidate_dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    pred_data: dict[str, pd.Series] = {}
    prob_data: dict[str, pd.Series] = {}
    names: list[str] = []
    for name in candidate_dirs:
        path = runs_root / name / "best_case_outputs_full.csv"
        cand = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
        aligned = df[["case_id"]].merge(cand[["case_id", "final_pred", "final_prob_high"]], on="case_id", how="left")
        short = "corr_" + name.split("_")[0]
        # Avoid duplicate short names if multiple directories start with a number.
        if short in pred_data:
            short = f"{short}_{len(names)}"
        pred_data[short] = pd.to_numeric(aligned["final_pred"], errors="coerce").fillna(0.0).astype(float)
        prob_data[short] = pd.to_numeric(aligned["final_prob_high"], errors="coerce").fillna(0.5).astype(float)
        names.append(short)
    return pd.DataFrame(pred_data), pd.DataFrame(prob_data), names


def build_selector_features(preds: pd.DataFrame, probs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred_mat = preds.to_numpy(dtype=float)
    prob_mat = probs.to_numpy(dtype=float)
    conf_mat = np.where(pred_mat >= 0.5, prob_mat, 1.0 - prob_mat)
    agg = pd.DataFrame(index=preds.index)
    agg["vote_frac"] = pred_mat.mean(axis=1)
    agg["vote_std"] = pred_mat.std(axis=1)
    agg["vote_disagree"] = ((pred_mat.sum(axis=1) > 0) & (pred_mat.sum(axis=1) < pred_mat.shape[1])).astype(float)
    agg["prob_mean"] = prob_mat.mean(axis=1)
    agg["prob_std"] = prob_mat.std(axis=1)
    agg["prob_min"] = prob_mat.min(axis=1)
    agg["prob_max"] = prob_mat.max(axis=1)
    agg["conf_mean"] = conf_mat.mean(axis=1)
    agg["conf_std"] = conf_mat.std(axis=1)
    agg["conf_min"] = conf_mat.min(axis=1)
    compact = pd.concat([preds, probs, agg], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    agg_only = agg.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {"candidate_full": compact, "candidate_agg": agg_only}


def predict_correctness(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(x_test.shape[0], float(np.mean(y_train)) if len(y_train) else 0.5)
    clf = clone(model)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def evaluate(
    df: pd.DataFrame,
    x_df: pd.DataFrame,
    candidate_pred: pd.DataFrame,
    candidate_prob: pd.DataFrame,
    model_name: str,
    model: object,
    select_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    pred_mat = candidate_pred.to_numpy(dtype=int)
    prob_mat = candidate_prob.to_numpy(dtype=float)
    x = x_df.to_numpy(dtype=float)
    final_pred = np.zeros(len(df), dtype=int)
    final_prob = np.zeros(len(df), dtype=float)
    selected_idx = np.zeros(len(df), dtype=int)
    correctness_score = np.zeros((len(df), pred_mat.shape[1]), dtype=float)
    candidate_train_acc_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        score_test = np.zeros((int(test.sum()), pred_mat.shape[1]), dtype=float)
        train_acc = (pred_mat[train] == y[train, None]).mean(axis=0)
        for j in range(pred_mat.shape[1]):
            y_corr = (pred_mat[train, j] == y[train]).astype(int)
            score_test[:, j] = predict_correctness(model, x[train], y_corr, x[test])
        if select_mode == "argmax_model":
            chosen = np.argmax(score_test, axis=1)
            final_pred[test] = pred_mat[test][np.arange(int(test.sum())), chosen]
            final_prob[test] = prob_mat[test][np.arange(int(test.sum())), chosen]
        elif select_mode == "weighted_vote":
            high_score = (score_test * pred_mat[test]).sum(axis=1)
            low_score = (score_test * (1 - pred_mat[test])).sum(axis=1)
            final_pred[test] = (high_score >= low_score).astype(int)
            final_prob[test] = high_score / np.maximum(high_score + low_score, 1e-6)
            chosen = np.argmax(score_test, axis=1)
        elif select_mode == "trainacc_weighted_vote":
            weights = score_test * train_acc[None, :]
            high_score = (weights * pred_mat[test]).sum(axis=1)
            low_score = (weights * (1 - pred_mat[test])).sum(axis=1)
            final_pred[test] = (high_score >= low_score).astype(int)
            final_prob[test] = high_score / np.maximum(high_score + low_score, 1e-6)
            chosen = np.argmax(weights, axis=1)
        elif select_mode == "top3_weighted_vote":
            topk = np.argsort(score_test, axis=1)[:, -3:]
            ptest = pred_mat[test]
            high_score = np.zeros(int(test.sum()), dtype=float)
            low_score = np.zeros(int(test.sum()), dtype=float)
            for i in range(int(test.sum())):
                idx = topk[i]
                high_score[i] = (score_test[i, idx] * ptest[i, idx]).sum()
                low_score[i] = (score_test[i, idx] * (1 - ptest[i, idx])).sum()
            final_pred[test] = (high_score >= low_score).astype(int)
            final_prob[test] = high_score / np.maximum(high_score + low_score, 1e-6)
            chosen = topk[:, -1]
        else:
            raise ValueError(select_mode)
        selected_idx[test] = chosen
        correctness_score[test] = score_test
        candidate_train_acc_rows.append(
            {
                "fold_id": int(fold),
                "best_train_candidate": candidate_pred.columns[int(np.argmax(train_acc))],
                "best_train_candidate_acc": float(train_acc.max()),
                "mean_train_candidate_acc": float(train_acc.mean()),
            }
        )

    summary = metric_row(y, final_pred, final_prob)
    summary["model"] = model_name
    summary["select_mode"] = select_mode
    summary["oracle_acc"] = float((pred_mat == y[:, None]).any(axis=1).mean())
    summary["candidate_n"] = int(pred_mat.shape[1])
    case_df = df[["case_id", "original_case_id", "fold_id", "label_idx", "pred_upper", "p_upper"]].copy()
    case_df["final_pred"] = final_pred
    case_df["final_prob_high"] = final_prob
    case_df["model"] = model_name
    case_df["select_mode"] = select_mode
    case_df["selected_candidate"] = [candidate_pred.columns[i] for i in selected_idx]
    case_df["selected_correctness_score"] = correctness_score[np.arange(len(df)), selected_idx]
    case_df["base_wrong"] = (case_df["pred_upper"].astype(int) != case_df["label_idx"].astype(int)).astype(int)
    case_df["final_wrong"] = (case_df["final_pred"].astype(int) != case_df["label_idx"].astype(int)).astype(int)
    case_df["rescued"] = ((case_df["base_wrong"] == 1) & (case_df["final_wrong"] == 0)).astype(int)
    case_df["hurt"] = ((case_df["base_wrong"] == 0) & (case_df["final_wrong"] == 1)).astype(int)
    return summary, case_df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    raw_pred, raw_prob, _ = build_raw_candidates(df)
    corr_pred, corr_prob, _ = build_corrected_candidates(df, Path(args.runs_root), split_arg(args.candidate_dirs))
    pools_all = {
        "raw_only": (raw_pred, raw_prob),
        "corrected_only": (corr_pred, corr_prob),
        "raw_plus_corrected": (pd.concat([raw_pred, corr_pred], axis=1), pd.concat([raw_prob, corr_prob], axis=1)),
    }
    wanted_pools = set(split_arg(args.pools))
    pools = {name: value for name, value in pools_all.items() if name in wanted_pools}
    models_all = make_models(args.seed)
    wanted_models = set(split_arg(args.models))
    models = {name: value for name, value in models_all.items() if name in wanted_models}
    modes = split_arg(args.modes)
    wanted_feature_sets = set(split_arg(args.feature_sets))
    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    for pool_name, (preds, probs) in pools.items():
        feature_sets = {name: value for name, value in build_selector_features(preds, probs).items() if name in wanted_feature_sets}
        y = df["label_idx"].astype(int).to_numpy()
        oracle_acc = float((preds.to_numpy(dtype=int) == y[:, None]).any(axis=1).mean())
        for feature_name, x_df in feature_sets.items():
            for model_name, model in models.items():
                for mode in modes:
                    summary, case_df = evaluate(df, x_df, preds, probs, model_name, model, mode)
                    summary["pool"] = pool_name
                    summary["feature_set"] = feature_name
                    summary["oracle_acc"] = oracle_acc
                    summaries.append(summary)
                    case_df["pool"] = pool_name
                    case_df["feature_set"] = feature_name
                    cases.append(case_df)
    summary_df = pd.DataFrame(summaries).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(out_dir / "candidate_correctness_selector_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "candidate_correctness_selector_case_outputs.csv", index=False)
    print(summary_df.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
