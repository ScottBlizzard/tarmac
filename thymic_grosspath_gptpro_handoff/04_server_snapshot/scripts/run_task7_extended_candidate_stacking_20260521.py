from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 extended OOF candidate stacking with weak gross-distill candidates.")
    parser.add_argument("--case-scores-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv")
    parser.add_argument("--gross-runs-root", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs")
    parser.add_argument("--curriculum-runs-root", default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs")
    parser.add_argument(
        "--gross-candidate-dirs",
        default="29_best_score_split_corrector_hybrid065_gb_flagtrain_20260520,36_best_hgb_score_split_corrector_20260520,38_best_hgb_refine_score_split_corrector_20260520,39_hgb_dual_tiebreak_prefer_low_20260520,41_best_candidate_stacking_balanced_20260520,43_best_candidate_stacking_no_tiebreak_20260520",
    )
    parser.add_argument(
        "--curriculum-candidate-dirs",
        default="50_gross_boundary_aux_hardcore_w010_full5fold,51_gross_boundary_aux_trusted55_w002_full5fold,52_gross_distill_trusted55_w005_full5fold,53_gross_boundary_aux_allcases_w0005_full5fold",
    )
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/61_extended_candidate_stacking_20260521")
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def split_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_models(seed: int) -> dict[str, object]:
    return {
        "logreg_c003": make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed)),
        "logreg_c01": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed)),
        "logreg_c03": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed)),
        "extra_d2_l10": ExtraTreesClassifier(n_estimators=500, max_depth=2, min_samples_leaf=10, class_weight="balanced", random_state=seed, n_jobs=-1),
        "extra_d3_l8": ExtraTreesClassifier(n_estimators=500, max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=seed, n_jobs=-1),
        "rf_d3_l8": RandomForestClassifier(n_estimators=400, max_depth=3, min_samples_leaf=8, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "gb_d1": GradientBoostingClassifier(max_depth=1, learning_rate=0.05, n_estimators=80, random_state=seed),
        "gb_d2": GradientBoostingClassifier(max_depth=2, learning_rate=0.03, n_estimators=80, random_state=seed),
    }


def add_candidate(out: pd.DataFrame, df: pd.DataFrame, name: str, pred: pd.Series, prob: pd.Series) -> None:
    pred_name = f"{name}_pred"
    prob_name = f"{name}_prob"
    out[pred_name] = pd.to_numeric(pred, errors="coerce").fillna(0.0).astype(float)
    out[prob_name] = pd.to_numeric(prob, errors="coerce").fillna(0.5).astype(float)
    out[f"{name}_conf"] = np.where(out[pred_name] >= 0.5, out[prob_name], 1.0 - out[prob_name])


def build_features(df: pd.DataFrame, args: argparse.Namespace, include_gross: bool, include_curriculum: bool, include_raw: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    names: list[str] = []
    if include_gross:
        for idx, run_name in enumerate(split_arg(args.gross_candidate_dirs)):
            path = Path(args.gross_runs_root) / run_name / "best_case_outputs_full.csv"
            cand = pd.read_csv(path, dtype={"case_id": str})
            aligned = df[["case_id"]].merge(cand[["case_id", "final_pred", "final_prob_high"]], on="case_id", how="left")
            short = f"g{idx}_{run_name.split('_')[0]}"
            add_candidate(out, df, short, aligned["final_pred"], aligned["final_prob_high"])
            names.append(short)
    if include_curriculum:
        for idx, run_name in enumerate(split_arg(args.curriculum_candidate_dirs)):
            path = Path(args.curriculum_runs_root) / run_name / "oof_case_predictions_mean.csv"
            cand = pd.read_csv(path, dtype={"case_id": str})
            aligned = df[["case_id"]].merge(cand[["case_id", "pred_idx", "prob_high_risk_group"]], on="case_id", how="left")
            short = f"c{idx}_{run_name.split('_')[0]}"
            add_candidate(out, df, short, aligned["pred_idx"], aligned["prob_high_risk_group"])
            names.append(short)
    if include_raw:
        for pred_col in [c for c in df.columns if c.startswith("pred_")]:
            suffix = pred_col[5:]
            prob_col = f"p_{suffix}"
            if prob_col not in df.columns:
                continue
            short = f"r_{suffix}"
            add_candidate(out, df, short, df[pred_col], df[prob_col])
            names.append(short)
    pred_cols = [f"{name}_pred" for name in names]
    prob_cols = [f"{name}_prob" for name in names]
    pred_mat = out[pred_cols].to_numpy(dtype=float)
    prob_mat = out[prob_cols].to_numpy(dtype=float)
    out["vote_frac"] = pred_mat.mean(axis=1)
    out["vote_disagree"] = ((pred_mat.sum(axis=1) > 0) & (pred_mat.sum(axis=1) < pred_mat.shape[1])).astype(float)
    out["prob_mean"] = prob_mat.mean(axis=1)
    out["prob_std"] = prob_mat.std(axis=1)
    out["prob_min"] = prob_mat.min(axis=1)
    out["prob_max"] = prob_mat.max(axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def evaluate(df: pd.DataFrame, x: pd.DataFrame, model_name: str, model: object) -> tuple[dict[str, object], pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    xx = x.to_numpy(dtype=float)
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros(len(df), dtype=float)
    for fold in sorted(set(folds)):
        train = folds != fold
        test = folds == fold
        clf = clone(model)
        clf.fit(xx[train], y[train])
        prob[test] = clf.predict_proba(xx[test])[:, 1]
        pred[test] = (prob[test] >= 0.5).astype(int)
    metrics = metric_row(y, pred, prob)
    metrics["model"] = model_name
    oof = df[["case_id", "original_case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].copy()
    oof["pred_idx"] = pred
    oof["prob_high_risk_group"] = prob
    oof["correct"] = pred == y
    return metrics, oof


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    feature_sets = {
        "gross_corrected": build_features(df, args, True, False, False),
        "gross_plus_curriculum": build_features(df, args, True, True, False),
        "gross_plus_curriculum_raw": build_features(df, args, True, True, True),
        "curriculum_only": build_features(df, args, False, True, False),
    }
    rows = []
    cases = []
    models = make_models(args.seed)
    for feature_name, x in feature_sets.items():
        for model_name, model in models.items():
            metrics, oof = evaluate(df, x, model_name, model)
            metrics["feature_set"] = feature_name
            rows.append(metrics)
            oof["feature_set"] = feature_name
            oof["model"] = model_name
            cases.append(oof)
    summary = pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary.to_csv(output_dir / "extended_candidate_stacking_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(cases, ignore_index=True).to_csv(output_dir / "extended_candidate_stacking_oof_all.csv", index=False, encoding="utf-8-sig")
    best = summary.iloc[0].to_dict()
    (output_dir / "best_summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
