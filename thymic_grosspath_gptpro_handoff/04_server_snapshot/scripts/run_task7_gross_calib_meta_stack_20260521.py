from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 strict meta-stack with best41 and gross/text calibrator OOF.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--best41-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv",
    )
    parser.add_argument(
        "--gross-calib-oof",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/57_learned_gross_text_switch_20260521/oof_case_predictions_mean.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/58_gross_calib_meta_stack_20260521",
    )
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, object]:
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


def score(y: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y, pred))
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y, pred))
    if objective == "f1":
        return float(f1_score(y, pred, zero_division=0))
    raise ValueError(objective)


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (prob >= t).astype(int)
        s = score(y, pred, objective)
        key = (s, -abs(float(t) - 0.5))
        if key > (best_s, -abs(best_t - 0.5)):
            best_s = s
            best_t = float(t)
    return best_t, best_s


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["base_prob"] = df["final_prob_high"].astype(float)
    x["base_pred"] = df["final_pred"].astype(float)
    x["base_margin"] = np.abs(x["base_prob"] - 0.5)
    x["calib_prob"] = df["calib_prob_high_risk_group"].astype(float)
    x["calib_pred"] = df["calib_pred_idx"].astype(float)
    x["calib_margin"] = np.abs(x["calib_prob"] - 0.5)
    x["calib_minus_base"] = x["calib_prob"] - x["base_prob"]
    x["abs_calib_minus_base"] = np.abs(x["calib_minus_base"])
    x["base_calib_disagree"] = (x["base_pred"] != x["calib_pred"]).astype(float)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def inner_select(x: pd.DataFrame, y: np.ndarray, folds: np.ndarray, train_mask: np.ndarray, objective: str) -> tuple[float, float]:
    best_c = 0.1
    best_t = 0.5
    best_s = -1.0
    for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        prob = np.full(len(y), np.nan, dtype=float)
        for inner_fold in sorted(set(folds[train_mask])):
            tr = train_mask & (folds != inner_fold)
            va = train_mask & (folds == inner_fold)
            if tr.sum() < 16 or va.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x.loc[tr])
            xva = scaler.transform(x.loc[va])
            clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260521)
            clf.fit(xtr, y[tr])
            prob[va] = clf.predict_proba(xva)[:, 1]
        valid = train_mask & ~np.isnan(prob)
        if valid.sum() < 16:
            continue
        threshold, current = best_threshold(y[valid], prob[valid], objective)
        key = (current, -abs(threshold - 0.5), -abs(np.log10(c)))
        if key > (best_s, -abs(best_t - 0.5), -abs(np.log10(best_c))):
            best_s = current
            best_c = float(c)
            best_t = threshold
    return best_c, best_t


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(root / args.best41_csv, dtype={"case_id": str})
    calib = pd.read_csv(root / args.gross_calib_oof, dtype={"case_id": str})
    keep = ["case_id", "calib_prob_high_risk_group", "calib_pred_idx", "difficulty", "difficulty_fine", "task_l6_label", "task_l7_label"]
    df = base.merge(calib[keep], on="case_id", how="inner")
    y = df["label_idx"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    x = build_features(df)
    rows = []
    best_oof: pd.DataFrame | None = None
    for objective in ["balanced_accuracy", "accuracy", "f1"]:
        prob = np.full(len(df), np.nan, dtype=float)
        pred = np.full(len(df), -1, dtype=int)
        choices = []
        for fold in sorted(set(folds)):
            tr = folds != fold
            te = folds == fold
            c, threshold = inner_select(x, y, folds, tr, objective)
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x.loc[tr])
            xte = scaler.transform(x.loc[te])
            clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260521 + int(fold))
            clf.fit(xtr, y[tr])
            prob[te] = clf.predict_proba(xte)[:, 1]
            pred[te] = (prob[te] >= threshold).astype(int)
            choices.append({"fold_id": int(fold), "c": c, "threshold": threshold})
        metrics = metric_dict(y, pred, prob)
        metrics["objective"] = objective
        rows.append(metrics)
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
                "final_pred",
                "final_prob_high",
                "calib_pred_idx",
                "calib_prob_high_risk_group",
            ]
        ].copy()
        oof["pred_idx"] = pred
        oof["prob_high_risk_group"] = prob
        oof["correct"] = pred == y
        run_dir = output_dir / f"objective_{objective}"
        run_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(choices).to_csv(run_dir / "fold_choices.csv", index=False)
        (run_dir / "overall_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        if best_oof is None or (metrics["balanced_accuracy"], metrics["accuracy"]) > (
            metric_dict(y, best_oof["pred_idx"].to_numpy(dtype=int), best_oof["prob_high_risk_group"].to_numpy(dtype=float))["balanced_accuracy"],
            metric_dict(y, best_oof["pred_idx"].to_numpy(dtype=int), best_oof["prob_high_risk_group"].to_numpy(dtype=float))["accuracy"],
        ):
            best_oof = oof
    summary = pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy"], ascending=False)
    summary.to_csv(output_dir / "gross_calib_meta_stack_summary.csv", index=False)
    best = summary.iloc[0].to_dict()
    (output_dir / "best_summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    if best_oof is not None:
        best_oof.to_csv(output_dir / "best_oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
