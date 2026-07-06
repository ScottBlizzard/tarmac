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
    extract_dataset_features as extract_dino_dataset_features,
    load_available_folds,
    load_dinov2_model,
    load_task_image_df,
    save_split_outputs,
    trim_image_df,
    write_json,
)
from run_plip_caseprobe import (  # type: ignore
    aggregate_features_to_cases,
    extract_dataset_features as extract_plip_dataset_features,
    load_plip_model,
)
from thymic_baseline.config import DEFAULT_RANDOM_SEED, get_task
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import format_metric_value, metrics_to_frame, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case-level concatenated DINO + PLIP frozen probe.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", default="task4_who_5class")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--dino-feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--plip-model-dir", default="third_party/round3/local_models/plip")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_selection_metric(task, prediction_df: pd.DataFrame) -> float:
    case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean")
    metrics = summarize_prediction_frame(case_df, task.class_names)
    return float(metrics.get(task.primary_metric, float("nan")))


def align_case_features(
    feature_blocks: list[tuple[np.ndarray, np.ndarray, list[str], list[str]]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    base_features, base_labels, base_case_ids, base_image_names = feature_blocks[0]
    combined = [base_features]
    for features, labels, case_ids, image_names in feature_blocks[1:]:
        if case_ids != base_case_ids:
            raise ValueError("Case ids are not aligned across feature blocks.")
        if image_names != base_image_names:
            raise ValueError("Case image-name traces are not aligned across feature blocks.")
        if not np.array_equal(labels, base_labels):
            raise ValueError("Case labels are not aligned across feature blocks.")
        combined.append(features)
    return np.concatenate(combined, axis=1), base_labels, base_case_ids, base_image_names


def fit_logreg_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
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
        val_predictions = build_prediction_df(val_probs, val_labels, val_case_ids, val_image_names, task.class_names)
        metric_value = compute_selection_metric(task, val_predictions)
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_model = model
            best_c = float(c_value)
            best_metric = float(metric_value)
    if best_model is None or best_metric is None:
        raise RuntimeError("Failed to fit combined case probe.")
    return best_model, best_c, best_metric


def save_mode_metrics(
    fold_dir: Path,
    split_name: str,
    mode_to_predictions: dict[str, pd.DataFrame],
    task,
) -> dict[str, dict[str, float]]:
    rows: list[pd.DataFrame] = []
    metrics_by_mode: dict[str, dict[str, float]] = {}
    for mode_name, prediction_df in mode_to_predictions.items():
        case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean")
        metrics = summarize_prediction_frame(case_df, task.class_names)
        metrics_by_mode[mode_name] = metrics
        rows.append(metrics_to_frame(split_name, "case", mode_name, metrics))
    pd.concat(rows, ignore_index=True).to_csv(fold_dir / f"{split_name}_metrics_by_mode.csv", index=False)
    return metrics_by_mode


def aggregate_run_outputs(root_output_dir: Path, task, fold_ids: list[int], aggregate_methods: list[str]) -> None:
    fold_rows: list[dict[str, Any]] = []
    all_test_images: list[pd.DataFrame] = []
    all_test_cases: dict[str, list[pd.DataFrame]] = {method: [] for method in aggregate_methods}
    all_mode_metrics: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        fold_dir = root_output_dir / f"fold_{fold_id}"
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "best_c": float(summary["best_c"]),
                "best_val_primary_metric": float(summary["best_val_primary_metric"]),
                "test_case_mean_primary_metric": float(summary["test_case_mean"].get(task.primary_metric, float("nan"))),
                "test_case_mean_accuracy": float(summary["test_case_mean"].get("accuracy", float("nan"))),
                "test_case_mean_balanced_accuracy": float(summary["test_case_mean"].get("balanced_accuracy", float("nan"))),
            }
        )

        image_df = pd.read_csv(fold_dir / "test_image_predictions.csv")
        image_df["fold_id"] = fold_id
        all_test_images.append(image_df)

        for method in aggregate_methods:
            case_df = pd.read_csv(fold_dir / f"test_case_predictions_{method}.csv")
            case_df["fold_id"] = fold_id
            all_test_cases[method].append(case_df)

        mode_df = pd.read_csv(fold_dir / "test_metrics_by_mode.csv")
        mode_df["fold_id"] = fold_id
        all_mode_metrics.append(mode_df)

    pd.DataFrame(fold_rows).to_csv(root_output_dir / "cv_fold_summary.csv", index=False)

    image_predictions_df = pd.concat(all_test_images, ignore_index=True)
    image_predictions_df.to_csv(root_output_dir / "oof_image_predictions.csv", index=False)
    metric_frames = [metrics_to_frame("test_oof", "image", "none", summarize_prediction_frame(image_predictions_df, task.class_names))]

    for method, frames in all_test_cases.items():
        case_predictions_df = pd.concat(frames, ignore_index=True)
        case_predictions_df.to_csv(root_output_dir / f"oof_case_predictions_{method}.csv", index=False)
        metric_frames.append(metrics_to_frame("test_oof", "case", method, summarize_prediction_frame(case_predictions_df, task.class_names)))

    pd.concat(metric_frames, ignore_index=True).to_csv(root_output_dir / "oof_metrics.csv", index=False)
    pd.concat(all_mode_metrics, ignore_index=True).to_csv(root_output_dir / "oof_metrics_by_mode.csv", index=False)


def run_single_fold(
    args: argparse.Namespace,
    task,
    image_df: pd.DataFrame,
    dino_model: torch.nn.Module,
    plip_model,
    plip_processor,
    device: torch.device,
    fold_id: int,
    available_folds: list[int],
) -> dict[str, Any]:
    fold_dir = Path(args.output_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
    train_df = trim_image_df(train_df, args.max_train_images)
    val_df = trim_image_df(val_df, args.max_val_images)
    test_df = trim_image_df(test_df, args.max_test_images)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]

    run_config = {
        "task": task.key,
        "dino_model_name": args.dino_model_name,
        "dino_feature_mode": args.dino_feature_mode,
        "plip_model_dir": str(Path(args.plip_model_dir).resolve()),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "fold_id": int(fold_id),
    }
    write_json(fold_dir / "run_config.json", run_config)

    print(
        f"\n=== DINO+PLIP CaseProbe Fold {fold_id}/{len(available_folds)} | task={task.key} | dino={args.dino_model_name} ===",
        flush=True,
    )
    print(
        f"train_cases={train_df['case_id'].nunique()}, val_cases={val_df['case_id'].nunique()}, test_cases={test_df['case_id'].nunique()}, "
        f"train_images={len(train_df)}, val_images={len(val_df)}, test_images={len(test_df)}",
        flush=True,
    )

    def extract_split_features(df: pd.DataFrame):
        dino_case = aggregate_features_to_cases(
            *extract_dino_dataset_features(
                image_df=df,
                input_variant="whole",
                feature_mode=args.dino_feature_mode,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                model=dino_model,
                device=device,
            ),
            agg="mean",
        )
        plip_case = aggregate_features_to_cases(
            *extract_plip_dataset_features(
                image_df=df,
                input_variant="whole",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                processor=plip_processor,
                model=plip_model,
                device=device,
            ),
            agg="mean",
        )
        return dino_case, plip_case

    train_dino_case, train_plip_case = extract_split_features(train_df)
    val_dino_case, val_plip_case = extract_split_features(val_df)
    test_dino_case, test_plip_case = extract_split_features(test_df)

    train_combo, train_labels, train_case_ids, train_image_names = align_case_features([train_dino_case, train_plip_case])
    val_combo, val_labels, val_case_ids, val_image_names = align_case_features([val_dino_case, val_plip_case])
    test_combo, test_labels, test_case_ids, test_image_names = align_case_features([test_dino_case, test_plip_case])

    combo_model, best_c, best_val_metric = fit_logreg_probe(
        train_combo,
        train_labels,
        val_combo,
        val_labels,
        val_case_ids,
        val_image_names,
        task,
        c_grid,
        args.seed,
    )

    dino_only_model, _, _ = fit_logreg_probe(
        train_dino_case[0],
        train_dino_case[1],
        val_dino_case[0],
        val_dino_case[1],
        val_dino_case[2],
        val_dino_case[3],
        task,
        c_grid,
        args.seed,
    )
    plip_only_model, _, _ = fit_logreg_probe(
        train_plip_case[0],
        train_plip_case[1],
        val_plip_case[0],
        val_plip_case[1],
        val_plip_case[2],
        val_plip_case[3],
        task,
        c_grid,
        args.seed,
    )

    val_combo_probs = combo_model.predict_proba(val_combo)
    test_combo_probs = combo_model.predict_proba(test_combo)
    val_dino_probs = dino_only_model.predict_proba(val_dino_case[0])
    test_dino_probs = dino_only_model.predict_proba(test_dino_case[0])
    val_plip_probs = plip_only_model.predict_proba(val_plip_case[0])
    test_plip_probs = plip_only_model.predict_proba(test_plip_case[0])

    val_predictions = build_prediction_df(val_combo_probs, val_labels, val_case_ids, val_image_names, task.class_names)
    test_predictions = build_prediction_df(test_combo_probs, test_labels, test_case_ids, test_image_names, task.class_names)
    val_case_metrics = save_split_outputs(fold_dir, "val", val_predictions, task, aggregate_methods)
    test_case_metrics = save_split_outputs(fold_dir, "test", test_predictions, task, aggregate_methods)

    val_mode_metrics = save_mode_metrics(
        fold_dir,
        "val",
        {
            "dino_case": build_prediction_df(val_dino_probs, val_dino_case[1], val_dino_case[2], val_dino_case[3], task.class_names),
            "plip_case": build_prediction_df(val_plip_probs, val_plip_case[1], val_plip_case[2], val_plip_case[3], task.class_names),
            "concat_case": val_predictions,
        },
        task,
    )
    test_mode_metrics = save_mode_metrics(
        fold_dir,
        "test",
        {
            "dino_case": build_prediction_df(test_dino_probs, test_dino_case[1], test_dino_case[2], test_dino_case[3], task.class_names),
            "plip_case": build_prediction_df(test_plip_probs, test_plip_case[1], test_plip_case[2], test_plip_case[3], task.class_names),
            "concat_case": test_predictions,
        },
        task,
    )

    fold_summary = {
        "best_c": best_c,
        "best_val_primary_metric": best_val_metric,
        "val_case_by_mode": val_mode_metrics,
        "test_case_by_mode": test_mode_metrics,
        "val_case_mean": val_case_metrics["mean"],
        "test_case_mean": test_case_metrics["mean"],
    }
    write_json(fold_dir / "fold_summary.json", fold_summary)
    print(
        f"[Fold {fold_id}] best_c={best_c:.4g} | "
        f"val_{task.primary_metric}={format_metric_value(val_case_metrics['mean'].get(task.primary_metric, float('nan')))} | "
        f"test_{task.primary_metric}={format_metric_value(test_case_metrics['mean'].get(task.primary_metric, float('nan')))} | "
        f"test_acc={format_metric_value(test_case_metrics['mean'].get('accuracy', float('nan')))}",
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
        raise ValueError("This script currently targets task4_who_5class.")
    image_df = load_task_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    print(
        f"Starting DINO+PLIP case probe: task={task.key}, dino={args.dino_model_name}, device={device}, folds={fold_ids}",
        flush=True,
    )
    dino_model = load_dinov2_model(Path(args.dino_repo_dir), args.dino_model_name, device)
    plip_model, plip_processor = load_plip_model(Path(args.plip_model_dir), device)
    for fold_id in fold_ids:
        run_single_fold(args, task, image_df, dino_model, plip_model, plip_processor, device, fold_id, available_folds)
    if len(fold_ids) > 1:
        aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
        aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
