from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_task7_stage2_auto_corrector_20260520 import build_features, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 hard-core split + automatic specialist correction pipeline.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/17_hardcore_split_auto_pipeline_20260520",
    )
    parser.add_argument("--feature-set", default="raw_plus_review_gross")
    parser.add_argument("--hard-detectors", default="hard_logreg,hard_extra")
    parser.add_argument("--correctors", default="corr_logreg")
    parser.add_argument("--recall-targets", default="0.70,0.80,0.90")
    parser.add_argument("--corrector-scopes", default="true_hardcore_train,pred_hardlike_train,all_train")
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def make_hard_detectors(seed: int) -> dict[str, object]:
    return {
        "hard_logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "hard_extra": ExtraTreesClassifier(
            n_estimators=240,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hard_rf": RandomForestClassifier(
            n_estimators=220,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def make_correctors(seed: int) -> dict[str, object]:
    return {
        "corr_logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "corr_extra": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=3,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def predict_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    clf = clone(model)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def inner_oof_prob(model: object, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(np.unique(folds)):
        tr = folds != fold
        va = folds == fold
        if len(np.unique(y[tr])) < 2:
            out[va] = float(np.mean(y[tr]))
        else:
            out[va] = predict_prob(model, x[tr], y[tr], x[va])
    return np.nan_to_num(out, nan=float(np.nanmean(out)))


def select_threshold_for_hard_recall(
    hard_prob: np.ndarray,
    hard_true: np.ndarray,
    recall_target: float,
) -> tuple[float, dict[str, float | int]]:
    # Flag as hard-like if hard_prob >= threshold. Choose the highest threshold
    # that still reaches target hard-core recall, maximizing non-hard pass rate.
    thresholds = np.unique(np.r_[0.0, np.linspace(0.05, 0.95, 91), 1.0, hard_prob])
    best = None
    for threshold in thresholds:
        flag = hard_prob >= threshold
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
        if best is None or key > best[0]:
            best = (
                key,
                float(threshold),
                {
                    "train_hard_recall": float(recall),
                    "train_hard_precision": float(precision),
                    "train_pass_frac": float(pass_frac),
                    "train_flag_n": int(flag.sum()),
                    "train_pass_n": int((~flag).sum()),
                },
            )
    if best is None:
        threshold = float(np.min(thresholds))
        flag = hard_prob >= threshold
        return threshold, {
            "train_hard_recall": float(((flag & (hard_true == 1)).sum()) / max(1, int((hard_true == 1).sum()))),
            "train_hard_precision": float(((flag & (hard_true == 1)).sum()) / max(1, int(flag.sum()))),
            "train_pass_frac": float((~flag).mean()),
            "train_flag_n": int(flag.sum()),
            "train_pass_n": int((~flag).sum()),
        }
    return best[1], best[2]


def train_corrector(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    if train_mask.sum() < 10 or len(np.unique(y[train_mask])) < 2:
        return np.full(test_x.shape[0], np.nan)
    return predict_prob(model, x[train_mask], y[train_mask], test_x)


def evaluate_pipeline(
    df: pd.DataFrame,
    x_df: pd.DataFrame,
    hard_detector_name: str,
    hard_detector: object,
    corrector_name: str,
    corrector: object,
    recall_target: float,
    corrector_scope: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    base_pred = df["pred_upper"].astype(int).to_numpy()
    base_prob = df["p_upper"].astype(float).to_numpy()
    hard_true = df["hard_core"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)

    hard_prob = np.full(len(df), np.nan, dtype=float)
    hard_flag = np.zeros(len(df), dtype=bool)
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    fold_rows = []

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        train_inner_prob = inner_oof_prob(hard_detector, x[train], hard_true[train], folds[train])
        threshold, threshold_info = select_threshold_for_hard_recall(train_inner_prob, hard_true[train], recall_target)
        test_hard_prob = predict_prob(hard_detector, x[train], hard_true[train], x[test])
        hard_prob[test] = test_hard_prob
        test_flag = test_hard_prob >= threshold
        hard_flag[test] = test_flag

        if corrector_scope == "true_hardcore_train":
            corr_train = train & (hard_true == 1)
        elif corrector_scope == "pred_hardlike_train":
            # Use inner hard probabilities to mimic predicted hard-like training data.
            inner_full = np.full(len(df), np.nan, dtype=float)
            inner_full[np.where(train)[0]] = train_inner_prob
            corr_train = train & (inner_full >= threshold)
        elif corrector_scope == "all_train":
            corr_train = train
        else:
            raise ValueError(corrector_scope)

        test_indices = np.where(test)[0]
        hardlike_test_indices = test_indices[test_flag]
        if len(hardlike_test_indices):
            prob = train_corrector(corrector, x, y, corr_train, x[hardlike_test_indices])
            valid = np.isfinite(prob)
            if valid.any():
                selected_indices = hardlike_test_indices[valid]
                final_prob[selected_indices] = prob[valid]
                final_pred[selected_indices] = (prob[valid] >= 0.5).astype(int)

        fold_rows.append(
            {
                "fold_id": int(fold),
                "threshold": threshold,
                "test_flag_n": int(test_flag.sum()),
                "test_pass_n": int((~test_flag).sum()),
                "test_hard_recall": float(((test_flag) & (hard_true[test] == 1)).sum() / max(1, int((hard_true[test] == 1).sum()))),
                **threshold_info,
            }
        )

    base_metrics = metric_row(y, base_pred)
    final_metrics = metric_row(y, final_pred)
    pass_mask = ~hard_flag
    hardlike_mask = hard_flag
    hard_mask = hard_true.astype(bool)
    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = base_wrong & (~final_wrong) & hardlike_mask
    hurt = (~base_wrong) & final_wrong & hardlike_mask
    try:
        hard_auc = float(roc_auc_score(hard_true, hard_prob))
    except ValueError:
        hard_auc = np.nan
    summary = {
        "hard_detector": hard_detector_name,
        "corrector": corrector_name,
        "recall_target": float(recall_target),
        "corrector_scope": corrector_scope,
        "hard_auc": hard_auc,
        "pass_n": int(pass_mask.sum()),
        "pass_frac": float(pass_mask.mean()),
        "hardlike_n": int(hardlike_mask.sum()),
        "hardlike_frac": float(hardlike_mask.mean()),
        "pass_accuracy": float(accuracy_score(y[pass_mask], base_pred[pass_mask])) if pass_mask.any() else np.nan,
        "pass_hard_contamination": float((hard_mask & pass_mask).sum() / max(1, int(pass_mask.sum()))),
        "hard_recall_in_hardlike": float((hard_mask & hardlike_mask).sum() / max(1, int(hard_mask.sum()))),
        "hard_precision_in_hardlike": float((hard_mask & hardlike_mask).sum() / max(1, int(hardlike_mask.sum()))),
        "base_accuracy": base_metrics["accuracy"],
        "base_balanced_accuracy": base_metrics["balanced_accuracy"],
        "final_accuracy": final_metrics["accuracy"],
        "final_balanced_accuracy": final_metrics["balanced_accuracy"],
        "final_specificity_low": final_metrics["specificity_low"],
        "final_sensitivity_high": final_metrics["sensitivity_high"],
        "final_tn": final_metrics["tn"],
        "final_fp": final_metrics["fp"],
        "final_fn": final_metrics["fn"],
        "final_tp": final_metrics["tp"],
        "hardcore_base_accuracy": float(accuracy_score(y[hard_mask], base_pred[hard_mask])),
        "hardcore_final_accuracy": float(accuracy_score(y[hard_mask], final_pred[hard_mask])),
        "hardlike_base_accuracy": float(accuracy_score(y[hardlike_mask], base_pred[hardlike_mask])) if hardlike_mask.any() else np.nan,
        "hardlike_final_accuracy": float(accuracy_score(y[hardlike_mask], final_pred[hardlike_mask])) if hardlike_mask.any() else np.nan,
        "rescued_total": int(rescued.sum()),
        "hurt_total": int(hurt.sum()),
        "rescued_hardcore": int((rescued & hard_mask).sum()),
        "hurt_hardcore": int((hurt & hard_mask).sum()),
    }
    case_df = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "pred_upper",
            "p_upper",
            "difficulty_fine",
            "hard_core",
            "upper_wrong",
            "upper_fn",
            "upper_fp",
        ]
    ].copy()
    case_df["hard_detector"] = hard_detector_name
    case_df["corrector"] = corrector_name
    case_df["recall_target"] = float(recall_target)
    case_df["corrector_scope"] = corrector_scope
    case_df["hard_prob"] = hard_prob
    case_df["hardlike_flag"] = hard_flag.astype(int)
    case_df["final_prob_high"] = final_prob
    case_df["final_pred"] = final_pred
    case_df["rescued"] = rescued.astype(int)
    case_df["hurt"] = hurt.astype(int)
    fold_df = pd.DataFrame(fold_rows)
    return summary, case_df, fold_df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    feature_sets = build_features(df, Path(args.registry_csv))
    if args.feature_set not in feature_sets:
        raise KeyError(f"Unknown feature set {args.feature_set}; available={sorted(feature_sets)}")
    x_df = feature_sets[args.feature_set]
    hard_detectors = make_hard_detectors(args.seed)
    correctors = make_correctors(args.seed)
    hard_keep = {x.strip() for x in args.hard_detectors.split(",") if x.strip()}
    corr_keep = {x.strip() for x in args.correctors.split(",") if x.strip()}
    hard_detectors = {k: v for k, v in hard_detectors.items() if k in hard_keep}
    correctors = {k: v for k, v in correctors.items() if k in corr_keep}
    recall_targets = [float(x.strip()) for x in args.recall_targets.split(",") if x.strip()]
    corrector_scopes = [x.strip() for x in args.corrector_scopes.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    cases = []
    folds = []
    for hard_name, hard_model in hard_detectors.items():
        for corr_name, corr_model in correctors.items():
            for recall_target in recall_targets:
                for corr_scope in corrector_scopes:
                    summary, case_df, fold_df = evaluate_pipeline(
                        df,
                        x_df,
                        hard_name,
                        hard_model,
                        corr_name,
                        corr_model,
                        recall_target,
                        corr_scope,
                    )
                    summaries.append(summary)
                    cases.append(case_df)
                    fold_df["hard_detector"] = hard_name
                    fold_df["corrector"] = corr_name
                    fold_df["recall_target"] = recall_target
                    fold_df["corrector_scope"] = corr_scope
                    folds.append(fold_df)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["final_balanced_accuracy", "final_accuracy", "hard_recall_in_hardlike"], ascending=False
    )
    case_out = pd.concat(cases, ignore_index=True)
    fold_out = pd.concat(folds, ignore_index=True)
    summary_df.to_csv(out_dir / "hardcore_split_auto_pipeline_summary.csv", index=False)
    case_out.to_csv(out_dir / "hardcore_split_auto_pipeline_case_outputs.csv", index=False)
    fold_out.to_csv(out_dir / "hardcore_split_auto_pipeline_fold_thresholds.csv", index=False)
    print(summary_df.head(40).to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
