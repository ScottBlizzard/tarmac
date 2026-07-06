from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from run_task7_gross_text_stacking_20260521 import (
    build_sparse,
    inner_select,
    load_data,
    metric_dict,
)
from run_task7_oracle_hard_gross_text_calibrator_20260521 import select_numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 deployable switch for gross/text hard-case calibrator.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/56_deployable_gross_text_switch_20260521",
    )
    parser.add_argument("--registry-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv")
    parser.add_argument("--split-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv")
    parser.add_argument("--curriculum-csv", default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv")
    parser.add_argument("--review-score-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv")
    parser.add_argument("--best41-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv")
    return parser.parse_args()


def fit_calibrator(
    df: pd.DataFrame,
    text: pd.Series,
    numeric: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: dict[str, object],
    objective: str = "accuracy",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    y = df["label_idx"].to_numpy(dtype=int)
    c, threshold, _ = inner_select(df, text, numeric, train_mask, cfg, objective)
    xtr, xte, _ = build_sparse(text[train_mask], text[test_mask], numeric.loc[train_mask], numeric.loc[test_mask], cfg)
    clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=20260521)
    clf.fit(xtr, y[train_mask])
    prob = clf.predict_proba(xte)[:, 1]
    pred = (prob >= threshold).astype(int)
    return prob, pred, c, threshold


def generate_outer_train_calib_oof(
    df: pd.DataFrame,
    text: pd.Series,
    numeric: pd.DataFrame,
    outer_train: np.ndarray,
    train_scope: np.ndarray,
    cfg: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    folds = df["fold_id"].to_numpy(dtype=int)
    prob = np.full(len(df), np.nan, dtype=float)
    pred = np.full(len(df), -1, dtype=int)
    for inner_fold in sorted(set(folds[outer_train])):
        calib_train = outer_train & train_scope & (folds != inner_fold)
        val = outer_train & (folds == inner_fold)
        if calib_train.sum() < 16 or val.sum() == 0:
            continue
        p, pr, _, _ = fit_calibrator(df, text, numeric, calib_train, val, cfg, "accuracy")
        prob[val] = p
        pred[val] = pr
    return prob, pred


def apply_rule(
    base_pred: np.ndarray,
    base_prob: np.ndarray,
    calib_pred: np.ndarray,
    calib_prob: np.ndarray,
    params: dict[str, object],
) -> np.ndarray:
    base_margin = np.abs(base_prob - 0.5)
    calib_margin = np.abs(calib_prob - 0.5)
    disagree = base_pred != calib_pred
    switch = disagree & (calib_margin >= float(params["min_calib_margin"]))
    if params["max_base_margin"] is not None:
        switch &= base_margin <= float(params["max_base_margin"])
    if params["direction"] == "fn_only":
        switch &= (base_pred == 0) & (calib_pred == 1)
    elif params["direction"] == "fp_only":
        switch &= (base_pred == 1) & (calib_pred == 0)
    final = base_pred.copy()
    final[switch] = calib_pred[switch]
    return final


def select_rule(y: np.ndarray, base_pred: np.ndarray, base_prob: np.ndarray, calib_pred: np.ndarray, calib_prob: np.ndarray) -> dict[str, object]:
    valid = ~np.isnan(calib_prob) & (calib_pred >= 0)
    best: dict[str, object] | None = None
    for direction in ["both", "fn_only", "fp_only"]:
        for min_calib_margin in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]:
            for max_base_margin in [None, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.08, 0.05]:
                params = {
                    "direction": direction,
                    "min_calib_margin": min_calib_margin,
                    "max_base_margin": max_base_margin,
                }
                pred = base_pred.copy()
                pred[valid] = apply_rule(base_pred[valid], base_prob[valid], calib_pred[valid], calib_prob[valid], params)
                acc = float(accuracy_score(y, pred))
                bacc = float(balanced_accuracy_score(y, pred))
                switched = int((pred != base_pred).sum())
                key = (acc, bacc, -switched)
                if best is None or key > best["key"]:
                    best = {"key": key, "params": params, "train_acc": acc, "train_bacc": bacc, "train_switched": switched}
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, text, numeric_all = load_data(project_root, args)
    numeric = select_numeric(numeric_all.replace([np.inf, -np.inf], np.nan).fillna(0.0), "image_core")
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
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    calib_prob_all = np.full(len(df), np.nan, dtype=float)
    calib_pred_all = np.full(len(df), -1, dtype=int)
    switched_all = np.zeros(len(df), dtype=bool)
    fold_rows = []

    for fold in sorted(set(folds)):
        outer_train = folds != fold
        outer_test = folds == fold
        train_calib_prob, train_calib_pred = generate_outer_train_calib_oof(df, text, numeric, outer_train, train_scope, cfg)
        rule = select_rule(
            y[outer_train],
            base_pred[outer_train],
            base_prob[outer_train],
            train_calib_pred[outer_train],
            train_calib_prob[outer_train],
        )
        p, pr, c, threshold = fit_calibrator(df, text, numeric, outer_train & train_scope, outer_test, cfg, "accuracy")
        calib_prob_all[outer_test] = p
        calib_pred_all[outer_test] = pr
        fold_final = apply_rule(base_pred[outer_test], base_prob[outer_test], pr, p, rule["params"])
        final_pred[outer_test] = fold_final
        final_prob[outer_test] = np.where(fold_final == pr, p, base_prob[outer_test])
        switched_all[outer_test] = fold_final != base_pred[outer_test]
        fold_rows.append(
            {
                "fold_id": int(fold),
                "c": c,
                "threshold": threshold,
                "test_n": int(outer_test.sum()),
                "test_switched": int(switched_all[outer_test].sum()),
                **rule["params"],
                "train_acc": rule["train_acc"],
                "train_bacc": rule["train_bacc"],
                "train_switched": rule["train_switched"],
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
    oof["pred_idx"] = final_pred
    oof["prob_high_risk_group"] = final_prob
    oof["switched"] = switched_all
    oof["base_correct"] = base_pred == y
    oof["calib_correct"] = calib_pred_all == y
    oof["final_correct"] = final_pred == y
    metrics = metric_dict(y, final_pred, final_prob)
    metrics.update(
        {
            "config": "deployable_rule_switch__train_allhard_calib_apply_all",
            "switched_n": int(switched_all.sum()),
            "rescue_n": int(((~oof["base_correct"]) & oof["final_correct"]).sum()),
            "hurt_n": int((oof["base_correct"] & (~oof["final_correct"])).sum()),
            "base_accuracy": float(accuracy_score(y, base_pred)),
            "base_balanced_accuracy": float(balanced_accuracy_score(y, base_pred)),
            "calib_oracle_with_base_accuracy": float(((base_pred == y) | (calib_pred_all == y)).mean()),
        }
    )
    oof.to_csv(output_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_rule_choices.csv", index=False, encoding="utf-8-sig")
    (output_dir / "overall_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(pd.DataFrame(fold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
