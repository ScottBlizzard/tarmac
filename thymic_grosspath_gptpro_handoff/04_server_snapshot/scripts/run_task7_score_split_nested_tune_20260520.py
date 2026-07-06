from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from run_task7_stage2_auto_corrector_20260520 import build_features, make_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested threshold tuning for Task7 score-split automatic corrector.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scores", default="review_score_hybrid_max_allrange_fn,review_score_upper_low_conf")
    parser.add_argument("--recall-targets", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--correction-thresholds", default="0.25:0.75:0.025")
    parser.add_argument("--blend-alphas", default="1.0", help="Routed-case probability blend: alpha*corrector + (1-alpha)*upper.")
    parser.add_argument("--feature-sets", default="raw_plus_review_gross")
    parser.add_argument("--models", default="gb,rf,extra,logreg")
    parser.add_argument("--train-scopes", default="flag_train,all_train")
    parser.add_argument("--objective", default="balanced_accuracy", choices=("accuracy", "balanced_accuracy", "f1"))
    parser.add_argument("--threshold-mode", default="global", choices=("global", "by_base_pred"))
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    text = text.strip()
    if ":" in text:
        start, stop, step = [float(item) for item in text.split(":")]
        values = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(float(current), 6))
            current += step
        return values
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def score_value(y_true: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y_true, pred))
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if objective == "f1":
        return float(f1_score(y_true, pred, zero_division=0))
    raise ValueError(objective)


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


def select_threshold_for_hard_recall(score: np.ndarray, hard_true: np.ndarray, recall_target: float) -> tuple[float, dict[str, Any]]:
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
            "hard_recall_train": float(recall),
            "hard_precision_train": float(precision),
            "pass_frac_train": float(pass_frac),
            "flag_n_train": int(flag.sum()),
        }
        if best is None or key > best[0]:
            best = (key, float(threshold), info)
    if best is not None:
        return best[1], best[2]
    return 0.0, {
        "hard_recall_train": 1.0,
        "hard_precision_train": float(hard_true.mean()),
        "pass_frac_train": 0.0,
        "flag_n_train": int(len(hard_true)),
    }


def fit_predict_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if x_test.shape[0] == 0:
        return np.array([], dtype=float)
    if x_train.shape[0] < 10 or len(np.unique(y_train)) < 2:
        return np.full(x_test.shape[0], float(np.mean(y_train)) if len(y_train) else 0.5, dtype=float)
    clf = clone(model)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def inner_select(
    df: pd.DataFrame,
    x: np.ndarray,
    model: object,
    outer_train: np.ndarray,
    score_col: str,
    recall_targets: list[float],
    correction_thresholds: list[float],
    blend_alphas: list[float],
    train_scope: str,
    objective: str,
    threshold_mode: str,
) -> dict[str, Any]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    hard_true = df["hard_core"].astype(int).to_numpy()
    route_score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_pred = df["pred_upper"].astype(int).to_numpy()
    base_prob = df["p_upper"].astype(float).to_numpy()
    candidates: list[dict[str, Any]] = []

    for recall_target in recall_targets:
        corrected_prob = np.full(len(df), np.nan, dtype=float)
        routed = np.zeros(len(df), dtype=bool)
        for inner_fold in sorted(np.unique(folds[outer_train])):
            inner_train = outer_train & (folds != inner_fold)
            inner_val = outer_train & (folds == inner_fold)
            threshold, _ = select_threshold_for_hard_recall(route_score[inner_train], hard_true[inner_train], recall_target)
            inner_val_flag = inner_val & (route_score >= threshold)
            routed[inner_val_flag] = True
            if train_scope == "flag_train":
                corrector_train = inner_train & (route_score >= threshold)
            elif train_scope == "all_train":
                corrector_train = inner_train
            else:
                raise ValueError(train_scope)
            corrected_prob[inner_val_flag] = fit_predict_prob(model, x[corrector_train], y[corrector_train], x[inner_val_flag])

        valid_train = outer_train.copy()
        valid_corrected = valid_train & routed & np.isfinite(corrected_prob)
        if threshold_mode == "global":
            threshold_pairs = [(float(t), float(t)) for t in correction_thresholds]
        elif threshold_mode == "by_base_pred":
            threshold_pairs = [(float(t_low), float(t_high)) for t_low in correction_thresholds for t_high in correction_thresholds]
        else:
            raise ValueError(threshold_mode)
        for alpha in blend_alphas:
            blended_prob = base_prob.copy()
            blended_prob[valid_corrected] = float(alpha) * corrected_prob[valid_corrected] + (1.0 - float(alpha)) * base_prob[valid_corrected]
            for threshold_low_base, threshold_high_base in threshold_pairs:
                pred = base_pred.copy()
                low_base = valid_corrected & (base_pred == 0)
                high_base = valid_corrected & (base_pred == 1)
                pred[low_base] = (blended_prob[low_base] >= threshold_low_base).astype(int)
                pred[high_base] = (blended_prob[high_base] >= threshold_high_base).astype(int)
                primary = score_value(y[valid_train], pred[valid_train], objective)
                bacc = float(balanced_accuracy_score(y[valid_train], pred[valid_train]))
                acc = float(accuracy_score(y[valid_train], pred[valid_train]))
                f1 = float(f1_score(y[valid_train], pred[valid_train], zero_division=0))
                candidates.append(
                    {
                        "recall_target": float(recall_target),
                        "blend_alpha": float(alpha),
                        "correction_threshold": float(threshold_low_base),
                        "correction_threshold_low_base": float(threshold_low_base),
                        "correction_threshold_high_base": float(threshold_high_base),
                        "inner_objective": primary,
                        "inner_bacc": bacc,
                        "inner_acc": acc,
                        "inner_f1": f1,
                        "inner_routed_n": int((valid_train & routed).sum()),
                    }
                )
    if not candidates:
        return {
            "recall_target": recall_targets[0],
            "blend_alpha": 1.0,
            "correction_threshold": 0.5,
            "correction_threshold_low_base": 0.5,
            "correction_threshold_high_base": 0.5,
            "inner_objective": float("nan"),
        }
    candidates.sort(
        key=lambda row: (
            row["inner_objective"],
            row["inner_bacc"],
            row["inner_acc"],
            -abs(row.get("blend_alpha", 1.0) - 0.5),
            -abs(row["correction_threshold_low_base"] - 0.5) - abs(row["correction_threshold_high_base"] - 0.5),
            -abs(row["recall_target"] - 0.65),
        ),
        reverse=True,
    )
    return candidates[0]


def evaluate_config(
    df: pd.DataFrame,
    x_df: pd.DataFrame,
    score_col: str,
    model_name: str,
    model: object,
    train_scope: str,
    recall_targets: list[float],
    correction_thresholds: list[float],
    blend_alphas: list[float],
    objective: str,
    threshold_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    hard_true = df["hard_core"].astype(int).to_numpy()
    hard_mask = hard_true.astype(bool)
    route_score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_prob = df["p_upper"].astype(float).to_numpy()
    base_pred = df["pred_upper"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    hardlike_flag = np.zeros(len(df), dtype=bool)
    fold_rows: list[dict[str, Any]] = []

    for fold in sorted(np.unique(folds)):
        outer_train = folds != fold
        test = folds == fold
        choice = inner_select(
            df,
            x,
            model,
            outer_train,
            score_col,
            recall_targets,
            correction_thresholds,
            blend_alphas,
            train_scope,
            objective,
            threshold_mode,
        )
        route_threshold, route_info = select_threshold_for_hard_recall(
            route_score[outer_train],
            hard_true[outer_train],
            float(choice["recall_target"]),
        )
        test_flag = test & (route_score >= route_threshold)
        hardlike_flag[test_flag] = True
        if train_scope == "flag_train":
            corrector_train = outer_train & (route_score >= route_threshold)
        elif train_scope == "all_train":
            corrector_train = outer_train
        else:
            raise ValueError(train_scope)
        prob = fit_predict_prob(model, x[corrector_train], y[corrector_train], x[test_flag])
        alpha = float(choice.get("blend_alpha", 1.0))
        final_prob[test_flag] = alpha * prob + (1.0 - alpha) * base_prob[test_flag]
        test_indices = np.where(test_flag)[0]
        low_base_indices = test_indices[base_pred[test_indices] == 0]
        high_base_indices = test_indices[base_pred[test_indices] == 1]
        if len(low_base_indices):
            low_prob = final_prob[low_base_indices]
            final_pred[low_base_indices] = (low_prob >= float(choice["correction_threshold_low_base"])).astype(int)
        if len(high_base_indices):
            high_prob = final_prob[high_base_indices]
            final_pred[high_base_indices] = (high_prob >= float(choice["correction_threshold_high_base"])).astype(int)
        fold_rows.append(
            {
                "fold_id": int(fold),
                "route_threshold": float(route_threshold),
                **route_info,
                **choice,
                "test_flag_n": int(test_flag.sum()),
                "test_hard_recall": float((test_flag & hard_mask).sum() / max(1, int((test & hard_mask).sum()))),
            }
        )

    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = base_wrong & (~final_wrong) & hardlike_flag
    hurt = (~base_wrong) & final_wrong & hardlike_flag
    base_metrics = metric_row(y, base_pred, base_prob)
    final_metrics = metric_row(y, final_pred, final_prob)
    pass_mask = ~hardlike_flag
    summary = {
        "score": score_col,
        "model": model_name,
        "train_scope": train_scope,
        "objective": objective,
        "threshold_mode": threshold_mode,
        "pass_n": int(pass_mask.sum()),
        "pass_frac": float(pass_mask.mean()),
        "pass_acc": float(accuracy_score(y[pass_mask], base_pred[pass_mask])) if pass_mask.any() else float("nan"),
        "hardlike_n": int(hardlike_flag.sum()),
        "hard_recall": float((hardlike_flag & hard_mask).sum() / max(1, int(hard_mask.sum()))),
        "hard_precision": float((hardlike_flag & hard_mask).sum() / max(1, int(hardlike_flag.sum()))),
        "base_acc": base_metrics["accuracy"],
        "base_bacc": base_metrics["balanced_accuracy"],
        "final_acc": final_metrics["accuracy"],
        "final_bacc": final_metrics["balanced_accuracy"],
        "final_auc": final_metrics["auc"],
        "final_f1": final_metrics["f1"],
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
    case_df["score_value"] = route_score
    case_df["model"] = model_name
    case_df["train_scope"] = train_scope
    case_df["hardlike_flag"] = hardlike_flag.astype(int)
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
    recall_targets = parse_float_list(args.recall_targets)
    correction_thresholds = parse_float_list(args.correction_thresholds)
    blend_alphas = parse_float_list(args.blend_alphas)
    score_cols = [item.strip() for item in args.scores.split(",") if item.strip()]
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
            for model_name in model_names:
                model = models[model_name]
                for train_scope in train_scopes:
                    summary, case_df, fold_df = evaluate_config(
                        df=df,
                        x_df=x_df,
                        score_col=score_col,
                        model_name=model_name,
                        model=model,
                        train_scope=train_scope,
                        recall_targets=recall_targets,
                        correction_thresholds=correction_thresholds,
                        blend_alphas=blend_alphas,
                        objective=args.objective,
                        threshold_mode=args.threshold_mode,
                    )
                    summary["feature_set"] = feature_name
                    summaries.append(summary)
                    case_df["feature_set"] = feature_name
                    fold_df["feature_set"] = feature_name
                    fold_df["score"] = score_col
                    fold_df["model"] = model_name
                    fold_df["train_scope"] = train_scope
                    folds.append(fold_df)
                    cases.append(case_df)

    summary_df = pd.DataFrame(summaries).sort_values(["final_bacc", "final_acc", "final_auc"], ascending=False)
    summary_df.to_csv(out_dir / "nested_tune_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "nested_tune_case_outputs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out_dir / "nested_tune_fold_choices.csv", index=False)
    print(summary_df.head(40).to_string(index=False))
    best = summary_df.iloc[0].to_dict()
    (out_dir / "best_summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
