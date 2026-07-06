from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-wise greedy candidate committee for Task7.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
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
    parser.add_argument("--metrics", default="accuracy,balanced_accuracy,f1")
    parser.add_argument("--max-sizes", default="1,2,3,5,8,13,21")
    parser.add_argument("--allow-repeats", default="0,1")
    return parser.parse_args()


def split_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def split_int_arg(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, prob)) if prob is not None else float("nan")
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


def build_candidates(df: pd.DataFrame, runs_root: Path, candidate_dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_data: dict[str, pd.Series] = {}
    prob_data: dict[str, pd.Series] = {}
    for pred_col in [c for c in df.columns if c.startswith("pred_")]:
        suffix = pred_col[5:]
        prob_col = f"p_{suffix}"
        if prob_col not in df.columns:
            continue
        name = f"raw_{suffix}"
        pred_data[name] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)
        prob_data[name] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.5).astype(float)
    for idx, name in enumerate(candidate_dirs):
        path = runs_root / name / "best_case_outputs_full.csv"
        cand = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
        aligned = df[["case_id"]].merge(cand[["case_id", "final_pred", "final_prob_high"]], on="case_id", how="left")
        short = f"corr{idx}_{name.split('_')[0]}"
        pred_data[short] = pd.to_numeric(aligned["final_pred"], errors="coerce").fillna(0).astype(int)
        prob_data[short] = pd.to_numeric(aligned["final_prob_high"], errors="coerce").fillna(0.5).astype(float)
    return pd.DataFrame(pred_data), pd.DataFrame(prob_data)


def predict_committee(pred_mat: np.ndarray, selected: list[int], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if not selected:
        prob = np.full(pred_mat.shape[0], 0.5)
    else:
        prob = pred_mat[:, selected].mean(axis=1)
    return (prob >= threshold).astype(int), prob


def score_metric(y: np.ndarray, pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, pred))
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    raise ValueError(metric)


def best_threshold(y: np.ndarray, vote_prob: np.ndarray, metric: str) -> tuple[float, float]:
    best_key: tuple[float, float] | None = None
    best_thr = 0.5
    for thr in np.unique(np.r_[np.linspace(0.2, 0.8, 61), vote_prob]):
        pred = (vote_prob >= thr).astype(int)
        val = score_metric(y, pred, metric)
        acc = float(accuracy_score(y, pred))
        key = (val, acc)
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
    return best_thr, float(best_key[0])


def greedy_select(
    y_train: np.ndarray,
    pred_train: np.ndarray,
    metric: str,
    max_size: int,
    allow_repeats: bool,
) -> tuple[list[int], float, float]:
    selected: list[int] = []
    remaining = set(range(pred_train.shape[1]))
    current_score = -1.0
    current_thr = 0.5
    for _ in range(max_size):
        best: tuple[tuple[float, float], int, float] | None = None
        candidates = range(pred_train.shape[1]) if allow_repeats else sorted(remaining)
        for j in candidates:
            trial = selected + [j]
            _, prob = predict_committee(pred_train, trial, threshold=0.5)
            thr, val = best_threshold(y_train, prob, metric)
            pred = (prob >= thr).astype(int)
            key = (val, float(accuracy_score(y_train, pred)))
            if best is None or key > best[0]:
                best = (key, j, thr)
        if best is None:
            break
        best_score = best[0][0]
        if best_score + 1e-12 < current_score:
            break
        selected.append(best[1])
        remaining.discard(best[1])
        current_score = best_score
        current_thr = best[2]
        if not allow_repeats and not remaining:
            break
    return selected, current_thr, current_score


def evaluate(
    df: pd.DataFrame,
    candidate_pred: pd.DataFrame,
    metric: str,
    max_size: int,
    allow_repeats: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    pred_mat = candidate_pred.to_numpy(dtype=int)
    final_pred = np.zeros(len(df), dtype=int)
    final_prob = np.zeros(len(df), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        selected, threshold, train_score = greedy_select(
            y_train=y[train],
            pred_train=pred_mat[train],
            metric=metric,
            max_size=max_size,
            allow_repeats=allow_repeats,
        )
        pred, prob = predict_committee(pred_mat[test], selected, threshold)
        final_pred[test] = pred
        final_prob[test] = prob
        row = metric_row(y[test], pred, prob)
        row["fold_id"] = int(fold)
        row["selected"] = "|".join(candidate_pred.columns[i] for i in selected)
        row["selected_n"] = int(len(selected))
        row["threshold"] = float(threshold)
        row["train_score"] = float(train_score)
        fold_rows.append(row)
    summary = metric_row(y, final_pred, final_prob)
    summary["metric"] = metric
    summary["max_size"] = int(max_size)
    summary["allow_repeats"] = int(allow_repeats)
    summary["oracle_acc"] = float((pred_mat == y[:, None]).any(axis=1).mean())
    case_df = df[["case_id", "original_case_id", "fold_id", "label_idx", "pred_upper", "p_upper"]].copy()
    case_df["final_pred"] = final_pred
    case_df["final_prob_high"] = final_prob
    case_df["base_wrong"] = (case_df["pred_upper"].astype(int) != case_df["label_idx"].astype(int)).astype(int)
    case_df["final_wrong"] = (case_df["final_pred"].astype(int) != case_df["label_idx"].astype(int)).astype(int)
    case_df["rescued"] = ((case_df["base_wrong"] == 1) & (case_df["final_wrong"] == 0)).astype(int)
    case_df["hurt"] = ((case_df["base_wrong"] == 0) & (case_df["final_wrong"] == 1)).astype(int)
    return summary, case_df, pd.DataFrame(fold_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    pred_df, _ = build_candidates(df, Path(args.runs_root), split_arg(args.candidate_dirs))
    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for metric in split_arg(args.metrics):
        for max_size in split_int_arg(args.max_sizes):
            for allow_repeats_int in split_int_arg(args.allow_repeats):
                allow_repeats = bool(allow_repeats_int)
                summary, case_df, fold_df = evaluate(df, pred_df, metric, max_size, allow_repeats)
                summaries.append(summary)
                case_df["metric"] = metric
                case_df["max_size"] = max_size
                case_df["allow_repeats"] = int(allow_repeats)
                fold_df["metric"] = metric
                fold_df["max_size"] = max_size
                fold_df["allow_repeats"] = int(allow_repeats)
                cases.append(case_df)
                folds.append(fold_df)
    summary_df = pd.DataFrame(summaries).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(out_dir / "greedy_committee_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "greedy_committee_case_outputs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out_dir / "greedy_committee_fold_details.csv", index=False)
    print(summary_df.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
