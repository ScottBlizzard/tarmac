from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend two Task56 case-level runs using val-selected alpha.")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--class-names", required=True, help="Comma-separated class names in probability-column order.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-grid", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0")
    return parser.parse_args()


def get_prob_columns(class_names: list[str]) -> list[str]:
    return [f"prob_{name}" for name in class_names]


def read_case_predictions(run_dir: Path, fold_id: int, split: str) -> pd.DataFrame:
    path = run_dir / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing case prediction file: {path}")
    return pd.read_csv(path)


def blend_frames(frame_a: pd.DataFrame, frame_b: pd.DataFrame, prob_columns: list[str], alpha: float) -> pd.DataFrame:
    base_cols = [col for col in frame_a.columns if col not in prob_columns and col != "pred_idx"]
    merged = frame_a[base_cols + prob_columns].merge(
        frame_b[["case_id"] + prob_columns],
        on="case_id",
        suffixes=("_a", "_b"),
        how="inner",
    )
    if len(merged) != len(frame_a):
        raise ValueError("Case alignment mismatch between run_a and run_b.")
    blended = merged[base_cols].copy()
    for col in prob_columns:
        blended[col] = alpha * merged[f"{col}_a"] + (1.0 - alpha) * merged[f"{col}_b"]
    blended["pred_idx"] = blended[prob_columns].to_numpy().argmax(axis=1)
    return blended


def compute_metrics(frame: pd.DataFrame, prob_columns: list[str]) -> dict[str, float]:
    y_true = frame["label_idx"].to_numpy(dtype=int)
    y_pred = frame["pred_idx"].to_numpy(dtype=int)
    probs = frame[prob_columns].to_numpy(dtype=float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["macro_auc"] = float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    prob_columns = get_prob_columns(class_names)
    alpha_grid = [float(item.strip()) for item in args.alpha_grid.split(",") if item.strip()]

    fold_dirs = sorted(path for path in run_a.glob("fold_*") if path.is_dir())
    fold_ids = [int(path.name.split("_")[1]) for path in fold_dirs]
    fold_rows: list[dict[str, float | int]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        val_a = read_case_predictions(run_a, fold_id, "val")
        val_b = read_case_predictions(run_b, fold_id, "val")
        test_a = read_case_predictions(run_a, fold_id, "test")
        test_b = read_case_predictions(run_b, fold_id, "test")

        best = None
        for alpha in alpha_grid:
            blended_val = blend_frames(val_a, val_b, prob_columns, alpha)
            val_metrics = compute_metrics(blended_val, prob_columns)
            candidate = (
                val_metrics["macro_f1"],
                val_metrics["macro_auc"],
                alpha,
                val_metrics,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        assert best is not None
        best_alpha = best[2]
        best_val_metrics = best[3]
        blended_test = blend_frames(test_a, test_b, prob_columns, best_alpha)
        test_metrics = compute_metrics(blended_test, prob_columns)
        blended_test.insert(0, "fold_id", fold_id)
        oof_frames.append(blended_test)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "alpha_for_run_a": best_alpha,
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_macro_auc": best_val_metrics["macro_auc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_macro_auc": test_metrics["macro_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            }
        )

    oof = pd.concat(oof_frames, ignore_index=True)
    oof_metrics = compute_metrics(oof, prob_columns)
    oof.to_csv(output_dir / "oof_case_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "test_oof",
                "level": "case",
                "aggregation": "blend_case",
                **oof_metrics,
            }
        ]
    ).to_csv(output_dir / "oof_metrics.csv", index=False)

    print(f"Blended folds: {fold_ids}")
    print(
        "OOF case blend | "
        f"macro_f1={oof_metrics['macro_f1']:.4f} | "
        f"macro_auc={oof_metrics['macro_auc']:.4f} | "
        f"acc={oof_metrics['accuracy']:.4f} | "
        f"bacc={oof_metrics['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
