from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict fold-wise capacity test for Task7 stage-1 direct-pass gate.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/13_stage1_direct_pass_gate_20260520",
    )
    parser.add_argument("--targets", default="0.90,0.95")
    parser.add_argument("--min-train-accept", type=int, default=10)
    parser.add_argument(
        "--selection-modes",
        default="raw,wilson90,wilson95",
        help="Threshold selection modes: raw, wilson90, wilson95.",
    )
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "specificity_low": np.nan,
            "sensitivity_high": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(
    risk: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    target_acc: float,
    min_accept: int,
    selection_mode: str,
) -> dict[str, float | int]:
    order = np.argsort(risk, kind="mergesort")
    sorted_risk = risk[order]
    sorted_correct = (y[order] == pred[order]).astype(float)
    cumsum = np.cumsum(sorted_correct)
    ks = np.arange(1, len(order) + 1)
    acc = cumsum / ks
    if selection_mode == "raw":
        criterion = acc
    elif selection_mode == "wilson90":
        criterion = wilson_lower_bound(cumsum, ks, z=1.6448536269514722)
    elif selection_mode == "wilson95":
        criterion = wilson_lower_bound(cumsum, ks, z=1.959963984540054)
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    ok = (criterion >= target_acc) & (ks >= min_accept)
    if not ok.any():
        return {
            "threshold": -np.inf,
            "train_accept_n": 0,
            "train_accept_acc": np.nan,
            "train_accept_criterion": np.nan,
        }
    best_idx = np.where(ok)[0][-1]
    return {
        "threshold": float(sorted_risk[best_idx]),
        "train_accept_n": int(best_idx + 1),
        "train_accept_acc": float(acc[best_idx]),
        "train_accept_criterion": float(criterion[best_idx]),
    }


def wilson_lower_bound(successes: np.ndarray, totals: np.ndarray, z: float) -> np.ndarray:
    phat = successes / totals
    z2 = z * z
    denom = 1.0 + z2 / totals
    center = phat + z2 / (2.0 * totals)
    spread = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * totals)) / totals)
    return (center - spread) / denom


def evaluate_score(
    df: pd.DataFrame,
    score_col: str,
    target_acc: float,
    min_accept: int,
    selection_mode: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    pred = df["pred_upper"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    risk = pd.to_numeric(df[score_col], errors="coerce").fillna(1e9).to_numpy(dtype=float)
    hard_core = df["hard_core"].astype(int).to_numpy().astype(bool)
    wrong = pred != y
    upper_fn = df["upper_fn"].astype(int).to_numpy().astype(bool)

    accept = np.zeros(len(df), dtype=bool)
    fold_rows: list[dict[str, float | int | str]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        selected = select_threshold(risk[train], y[train], pred[train], target_acc, min_accept, selection_mode)
        threshold = float(selected["threshold"])
        fold_accept = risk[test] <= threshold
        accept[test] = fold_accept
        row = {
            "score": score_col.replace("review_score_", ""),
            "selection_mode": selection_mode,
            "target_accept_acc": float(target_acc),
            "fold_id": int(fold),
            **selected,
            "test_accept_n": int(fold_accept.sum()),
            "test_total_n": int(test.sum()),
        }
        if fold_accept.any():
            row.update({f"test_accept_{k}": v for k, v in metric_row(y[test][fold_accept], pred[test][fold_accept]).items()})
        fold_rows.append(row)

    accepted = accept
    reviewed = ~accept
    accept_metrics = metric_row(y[accepted], pred[accepted])
    review_base_metrics = metric_row(y[reviewed], pred[reviewed])
    final_if_review_corrected = pred.copy()
    final_if_review_corrected[reviewed] = y[reviewed]
    final_metrics = metric_row(y, final_if_review_corrected)
    row = {
        "score": score_col.replace("review_score_", ""),
        "selection_mode": selection_mode,
        "target_accept_acc": float(target_acc),
        "accept_n": int(accepted.sum()),
        "accept_frac": float(accepted.mean()),
        "review_n": int(reviewed.sum()),
        "review_frac": float(reviewed.mean()),
        "review_error_precision": float((reviewed & wrong).sum() / reviewed.sum()) if reviewed.sum() else np.nan,
        "error_recall_in_review": float((reviewed & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
        "fn_recall_in_review": float((reviewed & upper_fn).sum() / upper_fn.sum()) if upper_fn.sum() else np.nan,
        "hardcore_recall_in_review": float((reviewed & hard_core).sum() / hard_core.sum()) if hard_core.sum() else np.nan,
    }
    row.update({f"accept_{k}": v for k, v in accept_metrics.items()})
    row.update({f"review_base_{k}": v for k, v in review_base_metrics.items()})
    row.update({f"final_if_review_corrected_{k}": v for k, v in final_metrics.items()})

    case_decision = df[
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
    case_decision["score"] = score_col.replace("review_score_", "")
    case_decision["selection_mode"] = selection_mode
    case_decision["target_accept_acc"] = float(target_acc)
    case_decision["risk_score"] = risk
    case_decision["stage1_accept"] = accept.astype(int)
    case_decision["stage1_review"] = reviewed.astype(int)
    return row, pd.DataFrame(fold_rows), case_decision


def main() -> None:
    args = parse_args()
    case_scores = Path(args.case_scores_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(case_scores, dtype={"case_id": str, "original_case_id": str})
    score_cols = [c for c in df.columns if c.startswith("review_score_")]
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    selection_modes = [x.strip() for x in args.selection_modes.split(",") if x.strip()]

    summary_rows = []
    fold_rows = []
    case_rows = []
    for target in targets:
        for selection_mode in selection_modes:
            for score_col in score_cols:
                row, fold_df, case_df = evaluate_score(df, score_col, target, args.min_train_accept, selection_mode)
                summary_rows.append(row)
                fold_rows.append(fold_df)
                case_rows.append(case_df)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_accept_acc", "selection_mode", "accept_frac"], ascending=[True, True, False]
    )
    fold_summary = pd.concat(fold_rows, ignore_index=True)
    case_decisions = pd.concat(case_rows, ignore_index=True)

    summary.to_csv(out_dir / "stage1_gate_nested_summary.csv", index=False)
    fold_summary.to_csv(out_dir / "stage1_gate_nested_fold_thresholds.csv", index=False)
    case_decisions.to_csv(out_dir / "stage1_gate_nested_case_decisions.csv", index=False)

    baseline = metric_row(df["label_idx"].astype(int).to_numpy(), df["pred_upper"].astype(int).to_numpy())
    pd.DataFrame([baseline]).to_csv(out_dir / "baseline_upper_metrics.csv", index=False)

    print("Baseline upper:", baseline)
    for target in targets:
        print(f"\nTarget direct-pass accuracy >= {target:.2f}")
        cols = [
            "score",
            "selection_mode",
            "accept_n",
            "accept_frac",
            "review_frac",
            "accept_accuracy",
            "accept_balanced_accuracy",
            "accept_sensitivity_high",
            "accept_specificity_low",
            "review_error_precision",
            "error_recall_in_review",
            "fn_recall_in_review",
            "hardcore_recall_in_review",
        ]
        show = summary[summary["target_accept_acc"] == target]
        show = show.sort_values(["accept_accuracy", "accept_frac"], ascending=[False, False])
        print(show[cols].head(16).to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
