from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate class-wise DINO+PLIP probability fusion for Task4.")
    parser.add_argument("--dino-dir", required=True)
    parser.add_argument("--plip-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-grid", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0")
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def prob_cols(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("prob_")]


def summarize_multiclass(df: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true = df["label_idx"].to_numpy(dtype=np.int64)
    y_pred = df["pred_idx"].to_numpy(dtype=np.int64)
    probs = df.loc[:, prob_cols(df)].to_numpy(dtype=np.float64)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        metrics["macro_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["macro_auc"] = float("nan")
    return metrics


def align_frames(left_df: pd.DataFrame, right_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left_df.sort_values(["case_id"]).reset_index(drop=True)
    right = right_df.sort_values(["case_id"]).reset_index(drop=True)
    if left["case_id"].astype(str).tolist() != right["case_id"].astype(str).tolist():
        raise ValueError("Case ids are not aligned between DINO and PLIP frames.")
    if left["label_idx"].astype(int).tolist() != right["label_idx"].astype(int).tolist():
        raise ValueError("Labels are not aligned between DINO and PLIP frames.")
    return left, right


def blend_with_alpha_vector(dino_df: pd.DataFrame, plip_df: pd.DataFrame, alpha_vec: np.ndarray) -> pd.DataFrame:
    cols = prob_cols(dino_df)
    dino_probs = dino_df.loc[:, cols].to_numpy(dtype=np.float64)
    plip_probs = plip_df.loc[:, cols].to_numpy(dtype=np.float64)
    mixed = alpha_vec[None, :] * dino_probs + (1.0 - alpha_vec[None, :]) * plip_probs
    row_sums = mixed.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    mixed = mixed / row_sums

    out = dino_df.copy()
    out.loc[:, cols] = mixed
    out["pred_idx"] = mixed.argmax(axis=1)
    return out


def select_best_global_alpha(dino_val_df: pd.DataFrame, plip_val_df: pd.DataFrame, alpha_grid: list[float]) -> tuple[float, float]:
    best_alpha = float("nan")
    best_metric = float("-inf")
    for alpha in alpha_grid:
        alpha_vec = np.full(len(prob_cols(dino_val_df)), float(alpha), dtype=np.float64)
        blended = blend_with_alpha_vector(dino_val_df, plip_val_df, alpha_vec)
        metric_value = summarize_multiclass(blended)["macro_f1"]
        if metric_value > best_metric:
            best_alpha = float(alpha)
            best_metric = float(metric_value)
    return best_alpha, best_metric


def coordinate_search_alpha_vector(
    dino_val_df: pd.DataFrame,
    plip_val_df: pd.DataFrame,
    alpha_grid: list[float],
    rounds: int,
) -> tuple[np.ndarray, float]:
    initial_alpha, best_metric = select_best_global_alpha(dino_val_df, plip_val_df, alpha_grid)
    alpha_vec = np.full(len(prob_cols(dino_val_df)), initial_alpha, dtype=np.float64)

    for _ in range(rounds):
        improved = False
        for class_idx in range(len(alpha_vec)):
            local_best_alpha = alpha_vec[class_idx]
            local_best_metric = best_metric
            for alpha in alpha_grid:
                trial = alpha_vec.copy()
                trial[class_idx] = float(alpha)
                blended = blend_with_alpha_vector(dino_val_df, plip_val_df, trial)
                metric_value = summarize_multiclass(blended)["macro_f1"]
                if metric_value > local_best_metric:
                    local_best_metric = float(metric_value)
                    local_best_alpha = float(alpha)
            if local_best_metric > best_metric:
                alpha_vec[class_idx] = local_best_alpha
                best_metric = local_best_metric
                improved = True
        if not improved:
            break
    return alpha_vec, best_metric


def metrics_frame(split_name: str, mode: str, metrics: dict[str, float]) -> pd.DataFrame:
    row = {"split": split_name, "level": "case", "aggregation": mode}
    row.update(metrics)
    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()
    dino_dir = Path(args.dino_dir)
    plip_dir = Path(args.plip_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = parse_grid(args.alpha_grid)

    fold_dirs = sorted(path for path in dino_dir.glob("fold_*") if path.is_dir())
    summary_rows: list[dict[str, Any]] = []
    mode_to_oof_frames: dict[str, list[pd.DataFrame]] = {}

    for dino_fold_dir in fold_dirs:
        fold_id = int(dino_fold_dir.name.split("_")[-1])
        plip_fold_dir = plip_dir / dino_fold_dir.name
        dino_val_df, plip_val_df = align_frames(
            pd.read_csv(dino_fold_dir / "val_case_predictions_mean.csv"),
            pd.read_csv(plip_fold_dir / "val_case_predictions_mean.csv"),
        )
        dino_test_df, plip_test_df = align_frames(
            pd.read_csv(dino_fold_dir / "test_case_predictions_mean.csv"),
            pd.read_csv(plip_fold_dir / "test_case_predictions_mean.csv"),
        )

        global_alpha, global_val_metric = select_best_global_alpha(dino_val_df, plip_val_df, alpha_grid)
        global_alpha_vec = np.full(len(prob_cols(dino_val_df)), global_alpha, dtype=np.float64)
        class_alpha_vec, class_val_metric = coordinate_search_alpha_vector(
            dino_val_df=dino_val_df,
            plip_val_df=plip_val_df,
            alpha_grid=alpha_grid,
            rounds=args.rounds,
        )

        mode_to_case_df = {
            "dino_case": dino_test_df,
            "plip_case": plip_test_df,
            "blend_case": blend_with_alpha_vector(dino_test_df, plip_test_df, global_alpha_vec),
            "classblend_case": blend_with_alpha_vector(dino_test_df, plip_test_df, class_alpha_vec),
        }

        fold_out_dir = output_dir / dino_fold_dir.name
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        metric_frames: list[pd.DataFrame] = []
        test_metrics_by_mode: dict[str, dict[str, float]] = {}
        for mode_name, case_df in mode_to_case_df.items():
            case_df.to_csv(fold_out_dir / f"test_case_predictions_{mode_name}.csv", index=False)
            case_df["fold_id"] = fold_id
            mode_to_oof_frames.setdefault(mode_name, []).append(case_df)
            metrics = summarize_multiclass(case_df.drop(columns=["fold_id"]))
            test_metrics_by_mode[mode_name] = metrics
            metric_frames.append(metrics_frame("test", mode_name, metrics))
        pd.concat(metric_frames, ignore_index=True).to_csv(fold_out_dir / "test_metrics_by_mode.csv", index=False)

        summary = {
            "fold_id": int(fold_id),
            "global_alpha": float(global_alpha),
            "global_alpha_val_macro_f1": float(global_val_metric),
            "class_alpha_vec": class_alpha_vec.tolist(),
            "class_alpha_val_macro_f1": float(class_val_metric),
            "test_metrics_by_mode": test_metrics_by_mode,
        }
        summary_rows.append(summary)
        (fold_out_dir / "fold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(summary_rows).to_csv(output_dir / "cv_fold_summary.csv", index=False)

    oof_metric_frames: list[pd.DataFrame] = []
    for mode_name, frames in sorted(mode_to_oof_frames.items()):
        oof_df = pd.concat(frames, ignore_index=True)
        oof_df.to_csv(output_dir / f"oof_case_predictions_{mode_name}.csv", index=False)
        metrics = summarize_multiclass(oof_df.drop(columns=["fold_id"]))
        oof_metric_frames.append(metrics_frame("test_oof", mode_name, metrics))
    pd.concat(oof_metric_frames, ignore_index=True).to_csv(output_dir / "oof_metrics_by_mode.csv", index=False)


if __name__ == "__main__":
    main()
