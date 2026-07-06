#!/usr/bin/env python
"""Threshold sweep for Task 1 case-level out-of-fold predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metrics_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (score >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    accuracy = _safe_div(tp + tn, len(y_true))
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _pick_score_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested score column not found: {requested}")
        return requested

    preferred = [
        "prob_tet",
        "prob_positive",
        "positive_prob",
        "prob_1",
        "score",
    ]
    for column in preferred:
        if column in df.columns:
            return column

    prob_columns = [column for column in df.columns if column.startswith("prob_")]
    if len(prob_columns) == 2 and "prob_benign_hyperplasia" in prob_columns:
        other = [column for column in prob_columns if column != "prob_benign_hyperplasia"]
        return other[0]
    if len(prob_columns) == 1:
        return prob_columns[0]

    raise ValueError(f"Could not infer score column from columns: {list(df.columns)}")


def _write_summary(
    summary_path: Path,
    prediction_path: Path,
    score_column: str,
    default_row: pd.Series,
    best_balanced_row: pd.Series,
    best_specificity_row: pd.Series | None,
) -> None:
    def fmt(row: pd.Series) -> str:
        return (
            f"threshold={row['threshold']:.2f}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"sensitivity={row['sensitivity']:.4f}, "
            f"specificity={row['specificity']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"f1={row['f1']:.4f}, "
            f"tn/fp/fn/tp={int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tp'])}"
        )

    lines = [
        "# Task 1 threshold analysis",
        "",
        f"- Prediction file: `{prediction_path}`",
        f"- Score column: `{score_column}`",
        "- Positive class: `label_idx == 1` / TET",
        "- Threshold sweep: 0.00 to 1.00, step 0.01",
        "",
        "## Key thresholds",
        "",
        f"- Default 0.50: {fmt(default_row)}",
        f"- Best balanced accuracy: {fmt(best_balanced_row)}",
    ]
    if best_specificity_row is not None:
        lines.append(f"- Best sensitivity with specificity >= 0.70: {fmt(best_specificity_row)}")
    else:
        lines.append("- Best sensitivity with specificity >= 0.70: no threshold met this constraint")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-column", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if "label_idx" not in df.columns:
        raise ValueError("Expected a `label_idx` column in the prediction file")

    score_column = _pick_score_column(df, args.score_column)
    y_true = (df["label_idx"].to_numpy() == 1).astype(int)
    score = df[score_column].to_numpy(dtype=float)

    rows = [_metrics_at_threshold(y_true, score, threshold) for threshold in np.round(np.arange(0.0, 1.0001, 0.01), 2)]
    metrics = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "task1_whole_threshold_metrics.csv"
    summary_path = args.output_dir / "task1_whole_threshold_summary.md"
    metrics.to_csv(metrics_path, index=False)

    default_idx = (metrics["threshold"] - 0.50).abs().idxmin()
    default_row = metrics.loc[default_idx]
    best_balanced_row = metrics.sort_values(
        ["balanced_accuracy", "sensitivity", "specificity", "f1"], ascending=False
    ).iloc[0]
    constrained = metrics[metrics["specificity"] >= 0.70]
    best_specificity_row = None
    if not constrained.empty:
        best_specificity_row = constrained.sort_values(
            ["sensitivity", "balanced_accuracy", "specificity", "f1"], ascending=False
        ).iloc[0]

    _write_summary(
        summary_path=summary_path,
        prediction_path=args.predictions,
        score_column=score_column,
        default_row=default_row,
        best_balanced_row=best_balanced_row,
        best_specificity_row=best_specificity_row,
    )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
