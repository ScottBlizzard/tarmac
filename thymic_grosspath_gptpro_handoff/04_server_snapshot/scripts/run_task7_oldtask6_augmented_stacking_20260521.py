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
    parser = argparse.ArgumentParser(description="Task7 stacking augmented with old Task6 OOF probabilities.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--old-task6-csv",
        default="outputs/task6_fair_benchmark_20260513/05_dino_vitb14_whole/oof_case_predictions_mean.csv",
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
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def split_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
        "hgb_leaf3": HistGradientBoostingClassifier(max_leaf_nodes=3, l2_regularization=0.1, min_samples_leaf=20, max_iter=60, learning_rate=0.08, random_state=seed),
    }


def build_corrected_candidate_features(df: pd.DataFrame, runs_root: Path, candidate_dirs: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    pred_names: list[str] = []
    prob_names: list[str] = []
    for idx, name in enumerate(candidate_dirs):
        path = runs_root / name / "best_case_outputs_full.csv"
        cand = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
        aligned = df[["case_id"]].merge(cand[["case_id", "final_pred", "final_prob_high"]], on="case_id", how="left")
        prefix = f"corr{idx}_{name.split('_')[0]}"
        pred_name = f"{prefix}_pred"
        prob_name = f"{prefix}_prob"
        out[pred_name] = pd.to_numeric(aligned["final_pred"], errors="coerce").fillna(0.0).astype(float)
        out[prob_name] = pd.to_numeric(aligned["final_prob_high"], errors="coerce").fillna(0.5).astype(float)
        out[f"{prefix}_conf"] = np.where(out[pred_name] >= 0.5, out[prob_name], 1.0 - out[prob_name])
        pred_names.append(pred_name)
        prob_names.append(prob_name)
    pred_mat = out[pred_names].to_numpy(dtype=float)
    prob_mat = out[prob_names].to_numpy(dtype=float)
    out["corr_vote_frac"] = pred_mat.mean(axis=1)
    out["corr_vote_disagree"] = ((pred_mat.sum(axis=1) > 0) & (pred_mat.sum(axis=1) < pred_mat.shape[1])).astype(float)
    out["corr_prob_mean"] = prob_mat.mean(axis=1)
    out["corr_prob_std"] = prob_mat.std(axis=1)
    out["corr_prob_min"] = prob_mat.min(axis=1)
    out["corr_prob_max"] = prob_mat.max(axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_old_task6_features(df: pd.DataFrame, old_csv: Path) -> pd.DataFrame:
    old = pd.read_csv(old_csv, dtype={"case_id": str})
    prob_cols = ["prob_A", "prob_AB", "prob_B1", "prob_B2", "prob_B3", "prob_TC"]
    high_cols = ["prob_B2", "prob_B3", "prob_TC"]
    old = old[["case_id", "pred_idx", *prob_cols]].copy()
    old["old_task6_prob_high"] = old[high_cols].sum(axis=1)
    old["old_task6_pred_high"] = (old["old_task6_prob_high"] >= 0.5).astype(float)
    old["old_task6_conf"] = np.maximum(old["old_task6_prob_high"], 1.0 - old["old_task6_prob_high"])
    old["old_task6_max_prob"] = old[prob_cols].max(axis=1)
    old["old_task6_entropy"] = -(old[prob_cols].clip(1e-6, 1.0).to_numpy() * np.log(old[prob_cols].clip(1e-6, 1.0).to_numpy())).sum(axis=1)
    aligned = df[["original_case_id"]].merge(old, left_on="original_case_id", right_on="case_id", how="left")
    out = pd.DataFrame(index=df.index)
    out["old_task6_covered"] = aligned["case_id"].notna().astype(float)
    for col in prob_cols + ["old_task6_prob_high", "old_task6_pred_high", "old_task6_conf", "old_task6_max_prob", "old_task6_entropy"]:
        fill = 0.0 if col == "old_task6_pred_high" else (0.5 if "prob" in col or "conf" in col else 0.0)
        out[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(fill).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_predict(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = clone(model)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    return (prob >= 0.5).astype(int), prob


def evaluate(df: pd.DataFrame, x_df: pd.DataFrame, model_name: str, model: object) -> tuple[dict[str, Any], pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros(len(df), dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        pred[test], prob[test] = fit_predict(model, x[train], y[train], x[test])
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
    return summary, case_df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    corrected = build_corrected_candidate_features(df, Path(args.runs_root), split_arg(args.candidate_dirs))
    old_task6 = build_old_task6_features(df, Path(args.old_task6_csv))
    feature_sets = {
        "corrected": corrected,
        "old_task6": old_task6,
        "corrected_plus_old_task6": pd.concat([corrected, old_task6], axis=1),
    }
    models = make_models(args.seed)
    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    for feature_name, x_df in feature_sets.items():
        for model_name, model in models.items():
            summary, case_df = evaluate(df, x_df, model_name, model)
            summary["feature_set"] = feature_name
            summaries.append(summary)
            case_df["feature_set"] = feature_name
            cases.append(case_df)
    summary_df = pd.DataFrame(summaries).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(out_dir / "oldtask6_augmented_stacking_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "oldtask6_augmented_stacking_case_outputs.csv", index=False)
    print(summary_df.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
