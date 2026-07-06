from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_task7_score_split_corrector_20260520 import fit_predict, metric_row, select_threshold
from run_task7_stage2_auto_corrector_20260520 import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 direction-specific score-routed corrector.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-sets", default="raw_plus_review_gross")
    parser.add_argument("--recall-targets", default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--common-scores", default="review_score_hybrid_max_allrange_fn,review_score_learn_hard_core_extra")
    parser.add_argument(
        "--low-scores",
        default="review_score_learn_upper_fn_extra,review_score_hybrid_max_allrange_fn,review_score_upper_predlow_pupper",
    )
    parser.add_argument(
        "--high-scores",
        default="review_score_learn_upper_wrong_extra,review_score_learn_hard_core_extra,review_score_hybrid_max_allrange_fn",
    )
    parser.add_argument("--route-modes", default="common_hard,directional_error")
    parser.add_argument("--train-scopes", default="flag_train,all_train")
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def make_model_grid(seed: int) -> dict[str, object]:
    return {
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "logreg_c05": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "extra_d3_l8": ExtraTreesClassifier(
            n_estimators=420,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_d4_l6": ExtraTreesClassifier(
            n_estimators=420,
            max_depth=4,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "rf_d3_l8": RandomForestClassifier(
            n_estimators=360,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb_d1_lr05_n100": GradientBoostingClassifier(max_depth=1, learning_rate=0.05, n_estimators=100, random_state=seed),
        "gb_d2_lr035_n100": GradientBoostingClassifier(max_depth=2, learning_rate=0.035, n_estimators=100, random_state=seed),
        "gb_d2_lr05_n80": GradientBoostingClassifier(max_depth=2, learning_rate=0.05, n_estimators=80, random_state=seed),
    }


def choose_threshold(
    score: np.ndarray,
    target: np.ndarray,
    recall_target: float,
    train_mask: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    if int(train_mask.sum()) == 0 or int(target[train_mask].sum()) == 0:
        return 1.01, {
            "train_route_recall": 0.0,
            "train_route_precision": 0.0,
            "train_route_pass_frac": 1.0,
            "train_route_flag_n": 0,
        }
    threshold, info = select_threshold(score[train_mask], target[train_mask], recall_target)
    return threshold, {
        "train_route_recall": info["train_hard_recall"],
        "train_route_precision": info["train_hard_precision"],
        "train_route_pass_frac": info["train_pass_frac"],
        "train_route_flag_n": info["train_flag_n"],
    }


def evaluate_directional(
    df: pd.DataFrame,
    x_df: pd.DataFrame,
    route_mode: str,
    common_score: str,
    low_score: str,
    high_score: str,
    recall_target: float,
    model_name: str,
    model: object,
    train_scope: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    hard_true = df["hard_core"].astype(int).to_numpy()
    upper_fn = df["upper_fn"].astype(int).to_numpy()
    upper_fp = df["upper_fp"].astype(int).to_numpy()
    base_prob = df["p_upper"].astype(float).to_numpy()
    base_pred = df["pred_upper"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)

    common_arr = pd.to_numeric(df[common_score], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    low_arr = pd.to_numeric(df[low_score], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    high_arr = pd.to_numeric(df[high_score], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    route_flag = np.zeros(len(df), dtype=bool)
    fold_rows: list[dict[str, Any]] = []

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if route_mode == "common_hard":
            threshold, info = choose_threshold(common_arr, hard_true, recall_target, train)
            test_flag = test & (common_arr >= threshold)
            dir_thresholds = {0: threshold, 1: threshold}
            dir_scores = {0: common_arr, 1: common_arr}
            dir_train_flags = {0: train & (common_arr >= threshold), 1: train & (common_arr >= threshold)}
            route_info = {f"common_{k}": v for k, v in info.items()}
        elif route_mode == "directional_error":
            low_train = train & (base_pred == 0)
            high_train = train & (base_pred == 1)
            low_threshold, low_info = choose_threshold(low_arr, upper_fn, recall_target, low_train)
            high_threshold, high_info = choose_threshold(high_arr, upper_fp, recall_target, high_train)
            test_flag = test & (
                ((base_pred == 0) & (low_arr >= low_threshold))
                | ((base_pred == 1) & (high_arr >= high_threshold))
            )
            dir_thresholds = {0: low_threshold, 1: high_threshold}
            dir_scores = {0: low_arr, 1: high_arr}
            dir_train_flags = {
                0: train & (base_pred == 0) & (low_arr >= low_threshold),
                1: train & (base_pred == 1) & (high_arr >= high_threshold),
            }
            route_info = {f"low_{k}": v for k, v in low_info.items()} | {f"high_{k}": v for k, v in high_info.items()}
        else:
            raise ValueError(route_mode)

        route_flag[test_flag] = True
        for direction in [0, 1]:
            dir_test = test_flag & (base_pred == direction)
            if train_scope == "flag_train":
                dir_train = dir_train_flags[direction] & (base_pred == direction)
            elif train_scope == "all_train":
                dir_train = train & (base_pred == direction)
            else:
                raise ValueError(train_scope)
            pred, prob = fit_predict(model, x[dir_train], y[dir_train], x[dir_test])
            final_pred[dir_test] = pred
            final_prob[dir_test] = prob

        fold_rows.append(
            {
                "fold_id": int(fold),
                "route_mode": route_mode,
                "low_threshold": float(dir_thresholds[0]),
                "high_threshold": float(dir_thresholds[1]),
                "test_flag_n": int(test_flag.sum()),
                "test_flag_lowpred_n": int((test_flag & (base_pred == 0)).sum()),
                "test_flag_highpred_n": int((test_flag & (base_pred == 1)).sum()),
                "test_hard_recall": float(((test_flag) & (hard_true == 1)).sum() / max(1, int(((test) & (hard_true == 1)).sum()))),
                **route_info,
            }
        )

    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = base_wrong & (~final_wrong) & route_flag
    hurt = (~base_wrong) & final_wrong & route_flag
    hard_mask = hard_true.astype(bool)
    pass_mask = ~route_flag
    base_metrics = metric_row(y, base_pred, base_prob)
    final_metrics = metric_row(y, final_pred, final_prob)
    summary = {
        "route_mode": route_mode,
        "common_score": common_score,
        "low_score": low_score,
        "high_score": high_score,
        "recall_target": float(recall_target),
        "model": model_name,
        "train_scope": train_scope,
        "pass_n": int(pass_mask.sum()),
        "pass_frac": float(pass_mask.mean()),
        "pass_acc": float(accuracy_score(y[pass_mask], base_pred[pass_mask])) if pass_mask.any() else float("nan"),
        "review_n": int(route_flag.sum()),
        "hard_recall": float((route_flag & hard_mask).sum() / max(1, int(hard_mask.sum()))),
        "hard_precision": float((route_flag & hard_mask).sum() / max(1, int(route_flag.sum()))),
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
        "hard_base_acc": float(accuracy_score(y[hard_mask], base_pred[hard_mask])) if hard_mask.any() else float("nan"),
        "hard_final_acc": float(accuracy_score(y[hard_mask], final_pred[hard_mask])) if hard_mask.any() else float("nan"),
        "rescued": int(rescued.sum()),
        "hurt": int(hurt.sum()),
        "rescued_fn": int((rescued & (upper_fn == 1)).sum()),
        "rescued_fp": int((rescued & (upper_fp == 1)).sum()),
        "hurt_from_lowpred": int((hurt & (base_pred == 0)).sum()),
        "hurt_from_highpred": int((hurt & (base_pred == 1)).sum()),
        "rescued_hard": int((rescued & hard_mask).sum()),
        "hurt_hard": int((hurt & hard_mask).sum()),
    }
    case_cols = ["case_id", "original_case_id", "fold_id", "label_idx", "difficulty_fine", "hard_core", "p_upper", "pred_upper"]
    case_df = df[case_cols].copy()
    case_df["route_mode"] = route_mode
    case_df["common_score"] = common_score
    case_df["low_score"] = low_score
    case_df["high_score"] = high_score
    case_df["recall_target"] = float(recall_target)
    case_df["model"] = model_name
    case_df["train_scope"] = train_scope
    case_df["route_flag"] = route_flag.astype(int)
    case_df["final_prob_high"] = final_prob
    case_df["final_pred"] = final_pred
    case_df["rescued"] = rescued.astype(int)
    case_df["hurt"] = hurt.astype(int)
    case_df["route_score_used"] = np.where(base_pred == 0, dir_scores[0], dir_scores[1])
    return summary, case_df, pd.DataFrame(fold_rows)


def split_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    feature_sets = build_features(df, Path(args.registry_csv))
    feature_names = split_arg(args.feature_sets)
    common_scores = [s for s in split_arg(args.common_scores) if s in df.columns]
    low_scores = [s for s in split_arg(args.low_scores) if s in df.columns]
    high_scores = [s for s in split_arg(args.high_scores) if s in df.columns]
    recall_targets = [float(item) for item in split_arg(args.recall_targets)]
    route_modes = split_arg(args.route_modes)
    train_scopes = split_arg(args.train_scopes)
    models = make_model_grid(args.seed)

    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for feature_name in feature_names:
        x_df = feature_sets[feature_name]
        for route_mode in route_modes:
            if route_mode == "common_hard":
                score_options = [(common_score, common_score, common_score) for common_score in common_scores]
            elif route_mode == "directional_error":
                fallback_common = common_scores[0] if common_scores else "review_score_hybrid_max_allrange_fn"
                score_options = [(fallback_common, low_score, high_score) for low_score in low_scores for high_score in high_scores]
            else:
                continue
            for common_score, low_score, high_score in score_options:
                for recall_target in recall_targets:
                    for model_name, model in models.items():
                        for train_scope in train_scopes:
                            summary, case_df, fold_df = evaluate_directional(
                                df=df,
                                x_df=x_df,
                                route_mode=route_mode,
                                common_score=common_score,
                                low_score=low_score,
                                high_score=high_score,
                                recall_target=recall_target,
                                model_name=model_name,
                                model=model,
                                train_scope=train_scope,
                            )
                            summary["feature_set"] = feature_name
                            summaries.append(summary)
                            case_df["feature_set"] = feature_name
                            fold_df["feature_set"] = feature_name
                            fold_df["model"] = model_name
                            fold_df["train_scope"] = train_scope
                            cases.append(case_df)
                            folds.append(fold_df)

    summary_df = pd.DataFrame(summaries).sort_values(["final_bacc", "final_acc", "final_auc"], ascending=False)
    summary_df.to_csv(out_dir / "directional_corrector_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "directional_corrector_case_outputs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out_dir / "directional_corrector_fold_thresholds.csv", index=False)
    print(summary_df.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
