from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from run_task7_deployable_gross_text_switch_20260521 import fit_calibrator, generate_outer_train_calib_oof
from run_task7_gross_text_stacking_20260521 import load_data, metric_dict
from run_task7_oracle_hard_gross_text_calibrator_20260521 import select_numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 learned switcher for gross/text calibrator.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/57_learned_gross_text_switch_20260521",
    )
    parser.add_argument("--registry-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv")
    parser.add_argument("--split-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv")
    parser.add_argument("--curriculum-csv", default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv")
    parser.add_argument("--review-score-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv")
    parser.add_argument("--best41-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv")
    return parser.parse_args()


def build_switch_features(
    numeric_all: pd.DataFrame,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    calib_prob: np.ndarray,
    calib_pred: np.ndarray,
) -> pd.DataFrame:
    cols = []
    for c in numeric_all.columns:
        if c.startswith("review_score_") or c.startswith("p_") or c.startswith("pred_"):
            cols.append(c)
    # Keep the hand-crafted gross fields too; they do not start with p_/review_/pred_.
    cols.extend([c for c in numeric_all.columns if c not in cols])
    x = numeric_all.loc[:, cols].copy()
    x["base_prob"] = base_prob
    x["base_margin"] = np.abs(base_prob - 0.5)
    x["calib_prob"] = calib_prob
    x["calib_margin"] = np.abs(calib_prob - 0.5)
    x["calib_minus_base"] = calib_prob - base_prob
    x["abs_calib_minus_base"] = np.abs(calib_prob - base_prob)
    x["base_pred"] = base_pred
    x["calib_pred"] = calib_pred
    x["switch_low_to_high"] = ((base_pred == 0) & (calib_pred == 1)).astype(float)
    x["switch_high_to_low"] = ((base_pred == 1) & (calib_pred == 0)).astype(float)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_model(name: str, seed: int):
    if name.startswith("logreg"):
        c = float(name.split("_c")[-1])
        return LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed), True
    if name.startswith("rf"):
        leaf = int(name.split("_l")[-1])
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=leaf,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ), False
    if name.startswith("extra"):
        leaf = int(name.split("_l")[-1])
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=leaf,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ), False
    if name.startswith("gb"):
        depth = int(name.split("_d")[-1])
        return GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=depth,
            random_state=seed,
        ), False
    raise ValueError(name)


def model_score(model, scale: bool, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    if scale:
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x_train)
        xte = scaler.transform(x_test)
    else:
        xtr = x_train
        xte = x_test
    model.fit(xtr, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(xte)[:, 1]
    return model.decision_function(xte)


def select_switcher(
    y: np.ndarray,
    folds: np.ndarray,
    outer_train: np.ndarray,
    x_switch: pd.DataFrame,
    base_pred: np.ndarray,
    calib_pred: np.ndarray,
    valid_calib: np.ndarray,
) -> dict[str, object]:
    candidate = outer_train & valid_calib & (base_pred != calib_pred)
    target = (calib_pred == y).astype(int)
    model_names = ["logreg_c0.03", "logreg_c0.1", "logreg_c0.3", "rf_l4", "rf_l8", "extra_l4", "extra_l8", "gb_d1", "gb_d2"]
    best: dict[str, object] | None = None
    for model_name in model_names:
        scores = np.full(len(y), np.nan, dtype=float)
        for inner_fold in sorted(set(folds[outer_train])):
            tr = candidate & (folds != inner_fold)
            va = candidate & (folds == inner_fold)
            if tr.sum() < 12 or va.sum() == 0 or len(np.unique(target[tr])) < 2:
                continue
            model, scale = make_model(model_name, 20260521 + int(inner_fold))
            scores[va] = model_score(model, scale, x_switch.loc[tr], target[tr], x_switch.loc[va])
        valid = candidate & ~np.isnan(scores)
        if valid.sum() < 12 or len(np.unique(target[valid])) < 2:
            continue
        for threshold in np.linspace(0.10, 0.90, 81):
            pred = base_pred.copy()
            switch = valid & (scores >= threshold)
            pred[switch] = calib_pred[switch]
            acc = float(accuracy_score(y[outer_train], pred[outer_train]))
            bacc = float(balanced_accuracy_score(y[outer_train], pred[outer_train]))
            switched = int(switch.sum())
            switch_precision = float(target[switch].mean()) if switched else 1.0
            key = (acc, bacc, switch_precision, -switched)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "model_name": model_name,
                    "threshold": float(threshold),
                    "train_acc": acc,
                    "train_bacc": bacc,
                    "train_switched": switched,
                    "train_switch_precision": switch_precision,
                    "train_candidates": int(valid.sum()),
                }
    if best is None:
        best = {
            "key": (float(accuracy_score(y[outer_train], base_pred[outer_train])), float(balanced_accuracy_score(y[outer_train], base_pred[outer_train])), 1.0, 0),
            "model_name": "none",
            "threshold": 1.0,
            "train_acc": float(accuracy_score(y[outer_train], base_pred[outer_train])),
            "train_bacc": float(balanced_accuracy_score(y[outer_train], base_pred[outer_train])),
            "train_switched": 0,
            "train_switch_precision": 1.0,
            "train_candidates": int(candidate.sum()),
        }
    return best


def main() -> None:
    args = parse_args()
    if "20260520" in args.best41_csv and "batch1_batch2" not in args.best41_csv:
        # Backward-compatible guard for accidental typo in command-line defaults.
        args.best41_csv = "outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv"
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, text, numeric_all = load_data(project_root, args)
    numeric_all = numeric_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    calib_numeric = select_numeric(numeric_all, "image_core")
    cfg = {
        "name": "image_core_text",
        "use_text": True,
        "use_numeric": True,
        "ngram_range": (2, 4),
        "min_df": 1,
        "max_df": 0.95,
        "c_grid": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    }
    y = df["label_idx"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    train_scope = df["difficulty_fine"].isin({"hard_salvage_teacher", "hard_core"}).to_numpy()
    base_prob = df["p_best41"].to_numpy(dtype=float)
    base_pred = df["pred_best41"].to_numpy(dtype=int)
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    calib_prob_all = np.full(len(df), np.nan, dtype=float)
    calib_pred_all = np.full(len(df), -1, dtype=int)
    switch_score_all = np.full(len(df), np.nan, dtype=float)
    switched = np.zeros(len(df), dtype=bool)
    fold_rows = []

    for fold in sorted(set(folds)):
        outer_train = folds != fold
        outer_test = folds == fold
        train_calib_prob, train_calib_pred = generate_outer_train_calib_oof(df, text, calib_numeric, outer_train, train_scope, cfg)
        p_test, pred_test, c, cal_threshold = fit_calibrator(df, text, calib_numeric, outer_train & train_scope, outer_test, cfg, "accuracy")
        calib_prob_for_switch = np.full(len(df), np.nan, dtype=float)
        calib_pred_for_switch = np.full(len(df), -1, dtype=int)
        calib_prob_for_switch[outer_train] = train_calib_prob[outer_train]
        calib_pred_for_switch[outer_train] = train_calib_pred[outer_train]
        calib_prob_for_switch[outer_test] = p_test
        calib_pred_for_switch[outer_test] = pred_test
        x_switch = build_switch_features(numeric_all, base_prob, base_pred, calib_prob_for_switch, calib_pred_for_switch)
        valid_calib = ~np.isnan(calib_prob_for_switch) & (calib_pred_for_switch >= 0)
        selected = select_switcher(y, folds, outer_train, x_switch, base_pred, calib_pred_for_switch, valid_calib)
        calib_prob_all[outer_test] = p_test
        calib_pred_all[outer_test] = pred_test
        test_candidate = outer_test & valid_calib & (base_pred != calib_pred_for_switch)
        if selected["model_name"] != "none" and test_candidate.sum() > 0:
            train_candidate = outer_train & valid_calib & (base_pred != calib_pred_for_switch)
            target = (calib_pred_for_switch == y).astype(int)
            model, scale = make_model(str(selected["model_name"]), 20260521 + int(fold))
            score = model_score(model, scale, x_switch.loc[train_candidate], target[train_candidate], x_switch.loc[test_candidate])
            switch_score_all[test_candidate] = score
            test_switch = test_candidate.copy()
            test_switch[test_candidate] = score >= float(selected["threshold"])
            final_pred[test_switch] = calib_pred_for_switch[test_switch]
            final_prob[test_switch] = calib_prob_for_switch[test_switch]
            switched[test_switch] = True
        fold_rows.append(
            {
                "fold_id": int(fold),
                "calib_c": c,
                "calib_threshold": cal_threshold,
                "test_n": int(outer_test.sum()),
                "test_candidates": int(test_candidate.sum()),
                "test_switched": int(switched[outer_test].sum()),
                **{k: v for k, v in selected.items() if k != "key"},
            }
        )

    oof = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
        ]
    ].copy()
    oof["base_pred_idx"] = base_pred
    oof["base_prob_high_risk_group"] = base_prob
    oof["calib_pred_idx"] = calib_pred_all
    oof["calib_prob_high_risk_group"] = calib_prob_all
    oof["switch_score"] = switch_score_all
    oof["switched"] = switched
    oof["pred_idx"] = final_pred
    oof["prob_high_risk_group"] = final_prob
    oof["base_correct"] = base_pred == y
    oof["calib_correct"] = calib_pred_all == y
    oof["final_correct"] = final_pred == y
    metrics = metric_dict(y, final_pred, final_prob)
    metrics.update(
        {
            "config": "learned_switch__train_allhard_calib_apply_all",
            "switched_n": int(switched.sum()),
            "rescue_n": int(((~oof["base_correct"]) & oof["final_correct"]).sum()),
            "hurt_n": int((oof["base_correct"] & (~oof["final_correct"])).sum()),
            "base_accuracy": float(accuracy_score(y, base_pred)),
            "base_balanced_accuracy": float(balanced_accuracy_score(y, base_pred)),
            "calib_oracle_with_base_accuracy": float(((base_pred == y) | (calib_pred_all == y)).mean()),
        }
    )
    oof.to_csv(output_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_switch_choices.csv", index=False, encoding="utf-8-sig")
    (output_dir / "overall_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(pd.DataFrame(fold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
