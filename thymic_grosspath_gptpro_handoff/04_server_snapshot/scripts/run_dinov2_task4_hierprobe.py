from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_dinov2_frozen_probe import (  # type: ignore
    build_prediction_df,
    extract_dataset_features,
    load_available_folds,
    load_dinov2_model,
    load_task_image_df,
    save_split_outputs,
    trim_image_df,
    write_json,
)
from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS, get_task
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import format_metric_value, metrics_to_frame, resolve_device


LOW_GROUP = (0, 1, 2)
HIGH_GROUP = (3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hierarchical DINOv2 probe for Task4.")
    parser.add_argument("--registry-csv", required=True, help="Case-level registry CSV.")
    parser.add_argument("--split-csv", required=True, help="Case-level split assignment CSV.")
    parser.add_argument("--images-root", required=True, help="Root directory containing all image files.")
    parser.add_argument("--task", default="task4_who_5class", help="Task key defined in thymic_baseline.config.")
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all", help="Fold id 1-5, or 'all' to run 5-fold CV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports and predictions.")
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2", help="Local path to the cloned DINOv2 repo.")
    parser.add_argument("--model-name", default="dinov2_vits14", help="torch.hub model entry, e.g. dinov2_vits14.")
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0,100.0", help="Comma-separated logistic-regression C values.")
    parser.add_argument(
        "--alpha-grid",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Weight grid for selected = alpha * flat + (1-alpha) * hierarchical.",
    )
    parser.add_argument(
        "--aggregate-methods",
        default="mean,max_prob,majority_vote",
        help="Comma-separated aggregation methods for case-level evaluation.",
    )
    parser.add_argument("--max-train-images", type=int, default=None, help="Optional image-level smoke limit.")
    parser.add_argument("--max-val-images", type=int, default=None, help="Optional image-level smoke limit.")
    parser.add_argument("--max-test-images", type=int, default=None, help="Optional image-level smoke limit.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_case_primary_metric(prediction_df: pd.DataFrame, class_names: tuple[str, ...]) -> float:
    case_df = aggregate_case_predictions(prediction_df, class_names, method="mean")
    metrics = summarize_prediction_frame(case_df, class_names)
    if len(class_names) == 2:
        for key in ("balanced_accuracy", "f1", "auc", "accuracy"):
            value = metrics.get(key)
            if value is not None and not math.isnan(float(value)):
                return float(value)
        return float("nan")
    return float(metrics.get("macro_f1", float("nan")))


def fit_multiclass_logreg(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    class_names: tuple[str, ...],
    c_grid: list[float],
    seed: int,
) -> tuple[Pipeline, float, float]:
    best_model: Pipeline | None = None
    best_c = float("nan")
    best_metric: float | None = None
    for c_value in c_grid:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        max_iter=4000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(train_features, train_labels)
        val_probs = model.predict_proba(val_features)
        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=val_labels,
            case_ids=val_case_ids,
            image_names=val_image_names,
            class_names=class_names,
        )
        metric_value = compute_case_primary_metric(val_predictions, class_names)
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_model = model
            best_c = float(c_value)
            best_metric = float(metric_value)
    if best_model is None or best_metric is None:
        raise RuntimeError("Failed to fit hierarchical logistic probe.")
    return best_model, best_c, best_metric


def remap_labels(labels: np.ndarray, original: tuple[int, ...]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(original)}
    return np.asarray([mapping[int(item)] for item in labels], dtype=np.int64)


def build_hierarchical_probs(
    coarse_probs: np.ndarray,
    low_probs: np.ndarray,
    high_probs: np.ndarray,
) -> np.ndarray:
    final_probs = np.zeros((coarse_probs.shape[0], 5), dtype=np.float64)
    final_probs[:, 0:3] = coarse_probs[:, [0]] * low_probs
    final_probs[:, 3:5] = coarse_probs[:, [1]] * high_probs
    return final_probs


def evaluate_modes(
    flat_probs: np.ndarray,
    hier_probs: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    image_names: list[str],
    class_names: tuple[str, ...],
    alpha_grid: list[float],
) -> tuple[float, dict[str, pd.DataFrame], dict[str, float]]:
    flat_predictions = build_prediction_df(flat_probs, labels, case_ids, image_names, class_names)
    hier_predictions = build_prediction_df(hier_probs, labels, case_ids, image_names, class_names)
    prediction_frames = {
        "flat": flat_predictions,
        "hier": hier_predictions,
    }
    metric_by_mode = {
        "flat": compute_case_primary_metric(flat_predictions, class_names),
        "hier": compute_case_primary_metric(hier_predictions, class_names),
    }

    best_alpha = 1.0
    best_metric = float("-inf")
    best_predictions = flat_predictions
    for alpha in alpha_grid:
        blend_probs = alpha * flat_probs + (1.0 - alpha) * hier_probs
        blend_predictions = build_prediction_df(blend_probs, labels, case_ids, image_names, class_names)
        metric_value = compute_case_primary_metric(blend_predictions, class_names)
        prediction_frames[f"blend_alpha_{alpha:g}"] = blend_predictions
        metric_by_mode[f"blend_alpha_{alpha:g}"] = float(metric_value)
        if not math.isnan(metric_value) and metric_value > best_metric:
            best_metric = float(metric_value)
            best_alpha = float(alpha)
            best_predictions = blend_predictions

    prediction_frames["selected"] = best_predictions
    metric_by_mode["selected"] = best_metric
    return best_alpha, prediction_frames, metric_by_mode


def save_mode_diagnostics(
    fold_dir: Path,
    split_name: str,
    prediction_frames: dict[str, pd.DataFrame],
    task_class_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    rows: list[pd.DataFrame] = []
    metrics_by_mode: dict[str, dict[str, float]] = {}
    for mode_name, prediction_df in prediction_frames.items():
        case_df = aggregate_case_predictions(prediction_df, task_class_names, method="mean")
        metrics = summarize_prediction_frame(case_df, task_class_names)
        metrics_by_mode[mode_name] = metrics
        rows.append(metrics_to_frame(split_name, "case", mode_name, metrics))
    pd.concat(rows, ignore_index=True).to_csv(fold_dir / f"{split_name}_metrics_by_mode.csv", index=False)
    return metrics_by_mode


def aggregate_selected_outputs(root_output_dir: Path, task_key: str, fold_ids: list[int]) -> None:
    task = get_task(task_key)
    fold_rows: list[dict[str, Any]] = []
    all_test_images: list[pd.DataFrame] = []
    all_test_cases: list[pd.DataFrame] = []
    all_metric_frames: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        fold_dir = root_output_dir / f"fold_{fold_id}"
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "flat_best_c": float(summary["flat_best_c"]),
                "coarse_best_c": float(summary["coarse_best_c"]),
                "low_best_c": float(summary["low_best_c"]),
                "high_best_c": float(summary["high_best_c"]),
                "selected_alpha": float(summary["selected_alpha"]),
                "best_val_primary_metric": float(summary["selected_val_case_mean"].get(task.primary_metric, float("nan"))),
                "test_case_mean_primary_metric": float(summary["selected_test_case_mean"].get(task.primary_metric, float("nan"))),
                "test_case_mean_accuracy": float(summary["selected_test_case_mean"].get("accuracy", float("nan"))),
                "test_case_mean_balanced_accuracy": float(summary["selected_test_case_mean"].get("balanced_accuracy", float("nan"))),
            }
        )

        image_df = pd.read_csv(fold_dir / "test_image_predictions.csv")
        image_df["fold_id"] = fold_id
        all_test_images.append(image_df)

        case_df = pd.read_csv(fold_dir / "test_case_predictions_mean.csv")
        case_df["fold_id"] = fold_id
        all_test_cases.append(case_df)

        metrics_df = pd.read_csv(fold_dir / "test_metrics_by_mode.csv")
        metrics_df["fold_id"] = fold_id
        all_metric_frames.append(metrics_df)

    pd.DataFrame(fold_rows).to_csv(root_output_dir / "cv_fold_summary.csv", index=False)

    image_predictions_df = pd.concat(all_test_images, ignore_index=True)
    image_predictions_df.to_csv(root_output_dir / "oof_image_predictions.csv", index=False)
    case_predictions_df = pd.concat(all_test_cases, ignore_index=True)
    case_predictions_df.to_csv(root_output_dir / "oof_case_predictions_mean.csv", index=False)

    metrics_frames = [
        metrics_to_frame("test_oof", "image", "none", summarize_prediction_frame(image_predictions_df, task.class_names)),
        metrics_to_frame("test_oof", "case", "mean", summarize_prediction_frame(case_predictions_df, task.class_names)),
    ]
    pd.concat(metrics_frames, ignore_index=True).to_csv(root_output_dir / "oof_metrics.csv", index=False)
    pd.concat(all_metric_frames, ignore_index=True).to_csv(root_output_dir / "oof_metrics_by_mode.csv", index=False)


def run_single_fold(
    args: argparse.Namespace,
    fold_id: int,
    available_folds: list[int],
    image_df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    task = get_task(args.task)
    if task.key != "task4_who_5class":
        raise ValueError("Hierarchical probe currently only supports task4_who_5class.")

    fold_dir = Path(args.output_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
    train_df = trim_image_df(train_df, args.max_train_images)
    val_df = trim_image_df(val_df, args.max_val_images)
    test_df = trim_image_df(test_df, args.max_test_images)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
    alpha_grid = [float(item.strip()) for item in args.alpha_grid.split(",") if item.strip()]

    run_config = {
        "task": task.key,
        "input_variant": args.input_variant,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "model_name": args.model_name,
        "fold_id": fold_id,
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "seed": int(args.seed),
        "c_grid": c_grid,
        "alpha_grid": alpha_grid,
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "test_images": int(len(test_df)),
    }
    write_json(fold_dir / "run_config.json", run_config)

    print(
        f"\n=== HierProbe Fold {fold_id}/{len(available_folds)} | task={task.key} | variant={args.input_variant} | model={args.model_name} ===",
        flush=True,
    )
    print(
        f"train_cases={train_df['case_id'].nunique()}, val_cases={val_df['case_id'].nunique()}, test_cases={test_df['case_id'].nunique()}, "
        f"train_images={len(train_df)}, val_images={len(val_df)}, test_images={len(test_df)}",
        flush=True,
    )

    train_features, train_labels, _, _ = extract_dataset_features(
        image_df=train_df,
        input_variant=args.input_variant,
        feature_mode="cls",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )
    val_features, val_labels, val_case_ids, val_image_names = extract_dataset_features(
        image_df=val_df,
        input_variant=args.input_variant,
        feature_mode="cls",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )
    test_features, test_labels, test_case_ids, test_image_names = extract_dataset_features(
        image_df=test_df,
        input_variant=args.input_variant,
        feature_mode="cls",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )

    flat_model, flat_best_c, flat_val_metric = fit_multiclass_logreg(
        train_features,
        train_labels,
        val_features,
        val_labels,
        val_case_ids,
        val_image_names,
        task.class_names,
        c_grid,
        args.seed,
    )

    train_coarse = np.isin(train_labels, HIGH_GROUP).astype(np.int64)
    val_coarse = np.isin(val_labels, HIGH_GROUP).astype(np.int64)
    coarse_model, coarse_best_c, coarse_val_metric = fit_multiclass_logreg(
        train_features,
        train_coarse,
        val_features,
        val_coarse,
        val_case_ids,
        val_image_names,
        ("low", "high"),
        c_grid,
        args.seed,
    )

    low_train_mask = np.isin(train_labels, LOW_GROUP)
    low_val_mask = np.isin(val_labels, LOW_GROUP)
    low_model, low_best_c, low_val_metric = fit_multiclass_logreg(
        train_features[low_train_mask],
        remap_labels(train_labels[low_train_mask], LOW_GROUP),
        val_features[low_val_mask],
        remap_labels(val_labels[low_val_mask], LOW_GROUP),
        [item for item, keep in zip(val_case_ids, low_val_mask) if keep],
        [item for item, keep in zip(val_image_names, low_val_mask) if keep],
        ("A", "AB", "B1"),
        c_grid,
        args.seed,
    )

    high_train_mask = np.isin(train_labels, HIGH_GROUP)
    high_val_mask = np.isin(val_labels, HIGH_GROUP)
    high_model, high_best_c, high_val_metric = fit_multiclass_logreg(
        train_features[high_train_mask],
        remap_labels(train_labels[high_train_mask], HIGH_GROUP),
        val_features[high_val_mask],
        remap_labels(val_labels[high_val_mask], HIGH_GROUP),
        [item for item, keep in zip(val_case_ids, high_val_mask) if keep],
        [item for item, keep in zip(val_image_names, high_val_mask) if keep],
        ("B2", "B3"),
        c_grid,
        args.seed,
    )

    val_flat_probs = flat_model.predict_proba(val_features)
    test_flat_probs = flat_model.predict_proba(test_features)
    val_hier_probs = build_hierarchical_probs(
        coarse_model.predict_proba(val_features),
        low_model.predict_proba(val_features),
        high_model.predict_proba(val_features),
    )
    test_hier_probs = build_hierarchical_probs(
        coarse_model.predict_proba(test_features),
        low_model.predict_proba(test_features),
        high_model.predict_proba(test_features),
    )

    selected_alpha, val_prediction_frames, _ = evaluate_modes(
        flat_probs=val_flat_probs,
        hier_probs=val_hier_probs,
        labels=val_labels,
        case_ids=val_case_ids,
        image_names=val_image_names,
        class_names=task.class_names,
        alpha_grid=alpha_grid,
    )
    _, test_prediction_frames, _ = evaluate_modes(
        flat_probs=test_flat_probs,
        hier_probs=test_hier_probs,
        labels=test_labels,
        case_ids=test_case_ids,
        image_names=test_image_names,
        class_names=task.class_names,
        alpha_grid=[selected_alpha],
    )

    val_mode_metrics = save_mode_diagnostics(fold_dir, "val", {k: v for k, v in val_prediction_frames.items() if k in ("flat", "hier", "selected")}, task.class_names)
    test_mode_metrics = save_mode_diagnostics(fold_dir, "test", {k: v for k, v in test_prediction_frames.items() if k in ("flat", "hier", "selected")}, task.class_names)

    selected_val_predictions = val_prediction_frames["selected"]
    selected_test_predictions = test_prediction_frames["selected"]
    selected_val_case_metrics = save_split_outputs(fold_dir, "val", selected_val_predictions, task, aggregate_methods)
    selected_test_case_metrics = save_split_outputs(fold_dir, "test", selected_test_predictions, task, aggregate_methods)

    selected_mode_name = "flat" if math.isclose(selected_alpha, 1.0) else ("hier" if math.isclose(selected_alpha, 0.0) else "blend")
    fold_summary = {
        "flat_best_c": flat_best_c,
        "flat_val_primary_metric": flat_val_metric,
        "coarse_best_c": coarse_best_c,
        "coarse_val_primary_metric": coarse_val_metric,
        "low_best_c": low_best_c,
        "low_val_primary_metric": low_val_metric,
        "high_best_c": high_best_c,
        "high_val_primary_metric": high_val_metric,
        "selected_alpha": selected_alpha,
        "selected_mode": selected_mode_name,
        "val_case_by_mode": val_mode_metrics,
        "test_case_by_mode": test_mode_metrics,
        "selected_val_case_mean": selected_val_case_metrics["mean"],
        "selected_test_case_mean": selected_test_case_metrics["mean"],
    }
    write_json(fold_dir / "fold_summary.json", fold_summary)
    print(
        f"[Fold {fold_id}] alpha={selected_alpha:g} ({selected_mode_name}) | "
        f"val_selected_macro_f1={format_metric_value(selected_val_case_metrics['mean'].get('macro_f1', float('nan')))} | "
        f"test_selected_macro_f1={format_metric_value(selected_test_case_metrics['mean'].get('macro_f1', float('nan')))} | "
        f"test_selected_acc={format_metric_value(selected_test_case_metrics['mean'].get('accuracy', float('nan')))}",
        flush=True,
    )
    return fold_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(args.task)
    if task.key != "task4_who_5class":
        raise ValueError("This script currently supports only task4_who_5class.")

    image_df = load_task_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    print(
        f"Starting hierarchical DINOv2 probe: task={task.key}, variant={args.input_variant}, model={args.model_name}, device={device}, folds={fold_ids}",
        flush=True,
    )
    model = load_dinov2_model(Path(args.repo_dir), args.model_name, device)
    for fold_id in fold_ids:
        run_single_fold(
            args=args,
            fold_id=fold_id,
            available_folds=available_folds,
            image_df=image_df,
            model=model,
            device=device,
        )
    if len(fold_ids) > 1:
        aggregate_selected_outputs(output_dir, args.task, fold_ids)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
