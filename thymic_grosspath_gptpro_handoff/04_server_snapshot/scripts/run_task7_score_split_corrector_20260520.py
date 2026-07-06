from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from run_task7_stage2_auto_corrector_20260520 import build_features, make_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 hard-like score split with fold-wise automatic corrector.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scores",
        default="review_score_learn_hard_core_extra,review_score_learn_hard_core_rf,review_score_hybrid_max_allrange_fn,review_score_upper_low_conf",
    )
    parser.add_argument("--recall-targets", default="0.60,0.70,0.80,0.90")
    parser.add_argument("--feature-sets", default="raw_plus_review,raw_plus_review_gross")
    parser.add_argument("--models", default="logreg,extra,gb")
    parser.add_argument("--train-scopes", default="flag_train,all_train")
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, prob)) if prob is not None else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(score: np.ndarray, hard_true: np.ndarray, recall_target: float) -> tuple[float, dict[str, Any]]:
    thresholds = np.unique(np.r_[0.0, np.linspace(0.01, 0.99, 99), 1.0, score])
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in thresholds:
        flag = score >= threshold
        tp = int((flag & (hard_true == 1)).sum())
        fn = int(((~flag) & (hard_true == 1)).sum())
        fp = int((flag & (hard_true == 0)).sum())
        tn = int(((~flag) & (hard_true == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        pass_frac = (tn + fn) / len(hard_true)
        if recall + 1e-12 < recall_target:
            continue
        key = (pass_frac, precision, threshold)
        info = {
            "train_hard_recall": float(recall),
            "train_hard_precision": float(precision),
            "train_pass_frac": float(pass_frac),
            "train_flag_n": int(flag.sum()),
        }
        if best is None or key > best[0]:
            best = (key, float(threshold), info)
    if best is not None:
        return best[1], best[2]
    return 0.0, {"train_hard_recall": 1.0, "train_hard_precision": float(hard_true.mean()), "train_pass_frac": 0.0, "train_flag_n": len(hard_true)}


def fit_predict(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_test.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if x_train.shape[0] < 10 or len(np.unique(y_train)) < 2:
        prob = np.full(x_test.shape[0], float(np.mean(y_train)) if len(y_train) else 0.5)
    else:
        clf = clone(model)
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_test)[:, 1]
    return (prob >= 0.5).astype(int), prob


def evaluate_config(
    df: pd.DataFrame,
    x_df: pd.DataFrame,
    score_col: str,
    recall_target: float,
    model_name: str,
    model: object,
    train_scope: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    hard_true = df["hard_core"].astype(int).to_numpy()
    score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_prob = df["p_upper"].astype(float).to_numpy()
    base_pred = df["pred_upper"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    hard_flag = np.zeros(len(df), dtype=bool)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        threshold, info = select_threshold(score[train], hard_true[train], recall_target)
        test_flag = test & (score >= threshold)
        hard_flag[test_flag] = True
        if train_scope == "flag_train":
            corr_train = train & (score >= threshold)
        elif train_scope == "all_train":
            corr_train = train
        else:
            raise ValueError(train_scope)
        pred, prob = fit_predict(model, x[corr_train], y[corr_train], x[test_flag])
        final_pred[test_flag] = pred
        final_prob[test_flag] = prob
        fold_rows.append(
            {
                "fold_id": int(fold),
                "threshold": float(threshold),
                "test_flag_n": int(test_flag.sum()),
                "test_hard_recall": float(((test_flag) & (hard_true == 1)).sum() / max(1, int(((test) & (hard_true == 1)).sum()))),
                **info,
            }
        )

    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = base_wrong & (~final_wrong) & hard_flag
    hurt = (~base_wrong) & final_wrong & hard_flag
    base_metrics = metric_row(y, base_pred, base_prob)
    final_metrics = metric_row(y, final_pred, final_prob)
    hard_mask = hard_true.astype(bool)
    pass_mask = ~hard_flag
    summary = {
        "score": score_col,
        "recall_target": float(recall_target),
        "model": model_name,
        "train_scope": train_scope,
        "pass_n": int(pass_mask.sum()),
        "pass_frac": float(pass_mask.mean()),
        "pass_acc": float(accuracy_score(y[pass_mask], base_pred[pass_mask])) if pass_mask.any() else float("nan"),
        "hardlike_n": int(hard_flag.sum()),
        "hard_recall": float((hard_flag & hard_mask).sum() / max(1, int(hard_mask.sum()))),
        "hard_precision": float((hard_flag & hard_mask).sum() / max(1, int(hard_flag.sum()))),
        "base_acc": base_metrics["accuracy"],
        "base_bacc": base_metrics["balanced_accuracy"],
        "final_acc": final_metrics["accuracy"],
        "final_bacc": final_metrics["balanced_accuracy"],
        "final_auc": final_metrics["auc"],
        "final_sens": final_metrics["sensitivity"],
        "final_spec": final_metrics["specificity"],
        "final_tn": final_metrics["tn"],
        "final_fp": final_metrics["fp"],
        "final_fn": final_metrics["fn"],
        "final_tp": final_metrics["tp"],
        "hard_base_acc": float(accuracy_score(y[hard_mask], base_pred[hard_mask])),
        "hard_final_acc": float(accuracy_score(y[hard_mask], final_pred[hard_mask])),
        "rescued": int(rescued.sum()),
        "hurt": int(hurt.sum()),
        "rescued_hard": int((rescued & hard_mask).sum()),
        "hurt_hard": int((hurt & hard_mask).sum()),
    }
    case_cols = ["case_id", "original_case_id", "fold_id", "label_idx", "difficulty_fine", "hard_core", "p_upper", "pred_upper"]
    case_df = df[case_cols].copy()
    case_df["score_name"] = score_col
    case_df["score_value"] = score
    case_df["recall_target"] = float(recall_target)
    case_df["model"] = model_name
    case_df["train_scope"] = train_scope
    case_df["hardlike_flag"] = hard_flag.astype(int)
    case_df["final_prob_high"] = final_prob
    case_df["final_pred"] = final_pred
    case_df["rescued"] = rescued.astype(int)
    case_df["hurt"] = hurt.astype(int)
    return summary, case_df, pd.DataFrame(fold_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    feature_sets = build_features(df, Path(args.registry_csv))
    models = make_models(seed=20260520)
    score_cols = [item.strip() for item in args.scores.split(",") if item.strip()]
    recall_targets = [float(item.strip()) for item in args.recall_targets.split(",") if item.strip()]
    feature_names = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    train_scopes = [item.strip() for item in args.train_scopes.split(",") if item.strip()]

    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for feature_name in feature_names:
        x_df = feature_sets[feature_name]
        for score_col in score_cols:
            if score_col not in df.columns:
                continue
            for recall_target in recall_targets:
                for model_name in model_names:
                    model = models[model_name]
                    for train_scope in train_scopes:
                        summary, case_df, fold_df = evaluate_config(
                            df=df,
                            x_df=x_df,
                            score_col=score_col,
                            recall_target=recall_target,
                            model_name=model_name,
                            model=model,
                            train_scope=train_scope,
                        )
                        summary["feature_set"] = feature_name
                        summaries.append(summary)
                        case_df["feature_set"] = feature_name
                        fold_df["feature_set"] = feature_name
                        fold_df["score"] = score_col
                        fold_df["recall_target"] = recall_target
                        fold_df["model"] = model_name
                        fold_df["train_scope"] = train_scope
                        cases.append(case_df)
                        folds.append(fold_df)
    summary_df = pd.DataFrame(summaries).sort_values(["final_bacc", "final_acc", "final_auc"], ascending=False)
    summary_df.to_csv(out_dir / "score_split_corrector_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "score_split_corrector_case_outputs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out_dir / "score_split_corrector_fold_thresholds.csv", index=False)
    print(summary_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
