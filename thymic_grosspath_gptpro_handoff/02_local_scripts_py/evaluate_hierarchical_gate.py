from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


DEFAULT_FINE_CLASSES = ("A", "AB", "B1", "B2", "B3")
DEFAULT_LOW_RISK_CLASSES = ("A", "AB", "B1")
DEFAULT_HIGH_RISK_CLASSES = ("B2", "B3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fold-wise hierarchical gating from coarse binary predictions to fine 5-class predictions."
    )
    parser.add_argument("--coarse-run-dir", required=True, type=Path, help="Run dir containing fold_{k} coarse case predictions.")
    parser.add_argument("--fine-run-dir", required=True, type=Path, help="Run dir containing fold_{k} fine case predictions.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection-metric", default="macro_f1", choices=("macro_f1", "macro_auc", "accuracy", "balanced_accuracy"))
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    parser.add_argument("--mode", default="soft", choices=("soft", "hard"))
    parser.add_argument("--coarse-low-col", default="prob_low_risk")
    parser.add_argument("--coarse-high-col", default="prob_high_risk")
    parser.add_argument("--fine-classes", default="A,AB,B1,B2,B3")
    parser.add_argument("--low-risk-classes", default="A,AB,B1")
    parser.add_argument("--high-risk-classes", default="B2,B3")
    parser.add_argument("--aggregation", default="mean", help="case prediction file suffix, e.g. mean/max_prob/majority_vote")
    return parser.parse_args()


def _float_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _class_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _prob_columns(class_names: list[str]) -> list[str]:
    return [f"prob_{name}" for name in class_names]


def _load_case_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_fold_pair(
    coarse_run_dir: Path,
    fine_run_dir: Path,
    fold_id: int,
    split_name: str,
    aggregation: str,
    coarse_low_col: str,
    coarse_high_col: str,
) -> pd.DataFrame:
    coarse_path = coarse_run_dir / f"fold_{fold_id}" / f"{split_name}_case_predictions_{aggregation}.csv"
    fine_path = fine_run_dir / f"fold_{fold_id}" / f"{split_name}_case_predictions_{aggregation}.csv"
    coarse = _load_case_predictions(coarse_path)
    fine = _load_case_predictions(fine_path)

    coarse_cols = ["case_id", coarse_low_col, coarse_high_col]
    missing = [column for column in coarse_cols if column not in coarse.columns]
    if missing:
        raise ValueError(f"Missing coarse columns in {coarse_path}: {missing}")
    fine_cols = ["case_id", "label_idx", "pred_idx"] + [column for column in fine.columns if column.startswith("prob_")]
    merged = fine[fine_cols].merge(coarse[coarse_cols], on="case_id", how="inner")
    if len(merged) != len(fine):
        raise ValueError(
            f"Fold {fold_id} {split_name}: case alignment mismatch between fine ({len(fine)}) and merged ({len(merged)})."
        )
    merged["fold_id"] = fold_id
    return merged


def _apply_gate(
    frame: pd.DataFrame,
    fine_classes: list[str],
    low_risk_classes: set[str],
    high_risk_classes: set[str],
    coarse_low_col: str,
    coarse_high_col: str,
    alpha: float,
    mode: str,
) -> pd.DataFrame:
    prob_columns = _prob_columns(fine_classes)
    out = frame.copy()
    probs = out[prob_columns].to_numpy(dtype=float)
    coarse_low = out[coarse_low_col].to_numpy(dtype=float)
    coarse_high = out[coarse_high_col].to_numpy(dtype=float)

    multipliers = np.ones_like(probs)
    for idx, class_name in enumerate(fine_classes):
        if class_name in low_risk_classes:
            if mode == "soft":
                multipliers[:, idx] = np.power(np.clip(coarse_low, 1e-8, 1.0), alpha)
            else:
                multipliers[:, idx] = (coarse_low >= coarse_high).astype(float)
        elif class_name in high_risk_classes:
            if mode == "soft":
                multipliers[:, idx] = np.power(np.clip(coarse_high, 1e-8, 1.0), alpha)
            else:
                multipliers[:, idx] = (coarse_high > coarse_low).astype(float)
        else:
            raise ValueError(f"Class {class_name} was not assigned to low-risk or high-risk groups.")

    adjusted = probs * multipliers
    row_sums = adjusted.sum(axis=1, keepdims=True)
    zero_rows = np.where(row_sums.squeeze() <= 0)[0]
    if len(zero_rows) > 0:
        adjusted[zero_rows] = probs[zero_rows]
        row_sums = adjusted.sum(axis=1, keepdims=True)
    adjusted = adjusted / np.clip(row_sums, 1e-12, None)

    out[prob_columns] = adjusted
    out["pred_idx"] = adjusted.argmax(axis=1)
    out["gate_alpha"] = float(alpha)
    return out


def _summarize(frame: pd.DataFrame, fine_classes: list[str]) -> dict[str, float]:
    prob_columns = _prob_columns(fine_classes)
    y_true = frame["label_idx"].to_numpy(dtype=int)
    y_pred = frame["pred_idx"].to_numpy(dtype=int)
    prob = frame[prob_columns].to_numpy(dtype=float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auc": float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro")),
    }


def _select_best_alpha(
    val_frame: pd.DataFrame,
    fine_classes: list[str],
    low_risk_classes: set[str],
    high_risk_classes: set[str],
    coarse_low_col: str,
    coarse_high_col: str,
    alpha_grid: list[float],
    mode: str,
    selection_metric: str,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    rows = []
    best_alpha = None
    best_metrics = None
    best_frame = None
    for alpha in alpha_grid:
        gated = _apply_gate(
            frame=val_frame,
            fine_classes=fine_classes,
            low_risk_classes=low_risk_classes,
            high_risk_classes=high_risk_classes,
            coarse_low_col=coarse_low_col,
            coarse_high_col=coarse_high_col,
            alpha=alpha,
            mode=mode,
        )
        metrics = _summarize(gated, fine_classes)
        row = {"alpha": float(alpha), **metrics}
        rows.append(row)
        if best_metrics is None or row[selection_metric] > best_metrics[selection_metric]:
            best_alpha = float(alpha)
            best_metrics = row
            best_frame = gated
    if best_alpha is None or best_frame is None:
        raise RuntimeError("Failed to select any alpha.")
    return best_alpha, pd.DataFrame(rows), best_frame


def main() -> None:
    args = parse_args()
    fine_classes = _class_list(args.fine_classes)
    low_risk_classes = set(_class_list(args.low_risk_classes))
    high_risk_classes = set(_class_list(args.high_risk_classes))
    alpha_grid = _float_grid(args.alpha_grid)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(path for path in args.fine_run_dir.glob("fold_*") if path.is_dir())
    fold_ids = [int(path.name.split("_")[1]) for path in fold_dirs]
    if not fold_ids:
        raise ValueError(f"No fold directories found under {args.fine_run_dir}.")

    fold_summary_rows = []
    oof_frames = []

    for fold_id in fold_ids:
        fold_output = args.output_dir / f"fold_{fold_id}"
        fold_output.mkdir(parents=True, exist_ok=True)

        val_frame = _load_fold_pair(
            coarse_run_dir=args.coarse_run_dir,
            fine_run_dir=args.fine_run_dir,
            fold_id=fold_id,
            split_name="val",
            aggregation=args.aggregation,
            coarse_low_col=args.coarse_low_col,
            coarse_high_col=args.coarse_high_col,
        )
        test_frame = _load_fold_pair(
            coarse_run_dir=args.coarse_run_dir,
            fine_run_dir=args.fine_run_dir,
            fold_id=fold_id,
            split_name="test",
            aggregation=args.aggregation,
            coarse_low_col=args.coarse_low_col,
            coarse_high_col=args.coarse_high_col,
        )

        best_alpha, sweep_df, best_val_frame = _select_best_alpha(
            val_frame=val_frame,
            fine_classes=fine_classes,
            low_risk_classes=low_risk_classes,
            high_risk_classes=high_risk_classes,
            coarse_low_col=args.coarse_low_col,
            coarse_high_col=args.coarse_high_col,
            alpha_grid=alpha_grid,
            mode=args.mode,
            selection_metric=args.selection_metric,
        )
        sweep_df.to_csv(fold_output / "val_alpha_sweep.csv", index=False)
        best_val_frame.to_csv(fold_output / "val_best_gated_predictions.csv", index=False)

        test_best_frame = _apply_gate(
            frame=test_frame,
            fine_classes=fine_classes,
            low_risk_classes=low_risk_classes,
            high_risk_classes=high_risk_classes,
            coarse_low_col=args.coarse_low_col,
            coarse_high_col=args.coarse_high_col,
            alpha=best_alpha,
            mode=args.mode,
        )
        test_best_frame.to_csv(fold_output / "test_best_gated_predictions.csv", index=False)

        baseline_test_frame = _apply_gate(
            frame=test_frame,
            fine_classes=fine_classes,
            low_risk_classes=low_risk_classes,
            high_risk_classes=high_risk_classes,
            coarse_low_col=args.coarse_low_col,
            coarse_high_col=args.coarse_high_col,
            alpha=0.0,
            mode="soft",
        )
        baseline_test_metrics = _summarize(baseline_test_frame, fine_classes)
        gated_test_metrics = _summarize(test_best_frame, fine_classes)
        fold_summary_rows.append(
            {
                "fold_id": fold_id,
                "selected_alpha": best_alpha,
                "val_best_" + args.selection_metric: float(sweep_df[args.selection_metric].max()),
                "test_baseline_macro_f1": baseline_test_metrics["macro_f1"],
                "test_baseline_macro_auc": baseline_test_metrics["macro_auc"],
                "test_gated_macro_f1": gated_test_metrics["macro_f1"],
                "test_gated_macro_auc": gated_test_metrics["macro_auc"],
                "test_gated_accuracy": gated_test_metrics["accuracy"],
                "test_gated_balanced_accuracy": gated_test_metrics["balanced_accuracy"],
            }
        )
        oof_frames.append(test_best_frame)

    fold_summary = pd.DataFrame(fold_summary_rows).sort_values("fold_id")
    fold_summary.to_csv(args.output_dir / "cv_fold_summary.csv", index=False)

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    oof_df.to_csv(args.output_dir / "oof_case_predictions_gated.csv", index=False)

    oof_metrics = _summarize(oof_df, fine_classes)
    pd.DataFrame([{"mode": args.mode, "aggregation": args.aggregation, **oof_metrics}]).to_csv(
        args.output_dir / "oof_metrics.csv", index=False
    )

    baseline_rows = []
    for fold_id in fold_ids:
        test_frame = _load_fold_pair(
            coarse_run_dir=args.coarse_run_dir,
            fine_run_dir=args.fine_run_dir,
            fold_id=fold_id,
            split_name="test",
            aggregation=args.aggregation,
            coarse_low_col=args.coarse_low_col,
            coarse_high_col=args.coarse_high_col,
        )
        baseline_rows.append(
            _apply_gate(
                frame=test_frame,
                fine_classes=fine_classes,
                low_risk_classes=low_risk_classes,
                high_risk_classes=high_risk_classes,
                coarse_low_col=args.coarse_low_col,
                coarse_high_col=args.coarse_high_col,
                alpha=0.0,
                mode="soft",
            )
        )
    baseline_oof = pd.concat(baseline_rows, axis=0, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    baseline_oof.to_csv(args.output_dir / "oof_case_predictions_baseline.csv", index=False)
    baseline_metrics = _summarize(baseline_oof, fine_classes)

    lines = [
        "# Hierarchical Gate Evaluation",
        "",
        f"- Coarse run dir: `{args.coarse_run_dir}`",
        f"- Fine run dir: `{args.fine_run_dir}`",
        f"- Mode: `{args.mode}`",
        f"- Aggregation: `{args.aggregation}`",
        f"- Selection metric: `{args.selection_metric}`",
        f"- Alpha grid: `{args.alpha_grid}`",
        "",
        "## Baseline fine OOF",
        "",
        f"- accuracy = {baseline_metrics['accuracy']:.4f}",
        f"- balanced_accuracy = {baseline_metrics['balanced_accuracy']:.4f}",
        f"- macro_f1 = {baseline_metrics['macro_f1']:.4f}",
        f"- macro_auc = {baseline_metrics['macro_auc']:.4f}",
        "",
        "## Gated fine OOF",
        "",
        f"- accuracy = {oof_metrics['accuracy']:.4f}",
        f"- balanced_accuracy = {oof_metrics['balanced_accuracy']:.4f}",
        f"- macro_f1 = {oof_metrics['macro_f1']:.4f}",
        f"- macro_auc = {oof_metrics['macro_auc']:.4f}",
        "",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved hierarchical gating evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
