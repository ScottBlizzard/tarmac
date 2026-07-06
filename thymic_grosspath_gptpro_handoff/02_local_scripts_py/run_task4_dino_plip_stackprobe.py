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
    aggregate_features_to_cases as aggregate_dino_features_to_cases,
    build_prediction_df,
    extract_dataset_features as extract_dino_dataset_features,
    load_available_folds,
    load_dinov2_model,
    load_task_image_df,
    trim_image_df,
    write_json,
)
from run_plip_caseprobe import (  # type: ignore
    extract_dataset_features as extract_plip_dataset_features,
    load_plip_model,
)
from thymic_baseline.config import DEFAULT_RANDOM_SEED, get_task
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import format_metric_value, metrics_to_frame, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task4 DINO+PLIP case-level stacking probe.")
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
    parser.add_argument("--base-c-grid", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--meta-c-grid", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--alpha-grid", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def compute_selection_metric(task, prediction_df: pd.DataFrame) -> float:
    metrics = summarize_prediction_frame(prediction_df, task.class_names)
    return float(metrics.get(task.primary_metric, float("nan")))


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
        metric_value = compute_selection_metric(task, aggregate_case_predictions(val_predictions, task.class_names, method="mean"))
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_model = model
            best_c = float(c_value)
            best_metric = float(metric_value)
    if best_model is None or best_metric is None:
        raise RuntimeError("Failed to fit logistic probe.")
    return best_model, best_c, best_metric


def case_prob_feature_block(
    prediction_df: pd.DataFrame,
    task,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DataFrame]:
    case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean").copy()
    case_df = case_df.sort_values(["case_id"]).reset_index(drop=True)
    feature_cols = [f"prob_{name}" for name in task.class_names]
    features = case_df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    labels = case_df["label_idx"].to_numpy(dtype=np.int64)
    case_ids = case_df["case_id"].astype(str).tolist()
    image_names = case_df.get("image_name", pd.Series([""] * len(case_df))).astype(str).tolist()
    return features, labels, case_ids, image_names, case_df


def align_case_blocks(
    left_block: tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DataFrame],
    right_block: tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    left_features, left_labels, left_case_ids, left_image_names, left_df = left_block
    right_features, right_labels, right_case_ids, right_image_names, right_df = right_block
    if left_case_ids != right_case_ids:
        raise ValueError("Case ids are not aligned across probability blocks.")
    if not np.array_equal(left_labels, right_labels):
        raise ValueError("Labels are not aligned across probability blocks.")
    return (
        np.concatenate([left_features, right_features], axis=1),
        left_labels,
        left_case_ids,
        left_image_names if left_image_names else right_image_names,
        left_df,
        right_df,
    )


def blend_case_predictions(dino_case_df: pd.DataFrame, plip_case_df: pd.DataFrame, alpha: float, task) -> pd.DataFrame:
    prob_cols = [f"prob_{name}" for name in task.class_names]
    blended = dino_case_df.copy()
    blended.loc[:, prob_cols] = alpha * dino_case_df.loc[:, prob_cols].to_numpy() + (1.0 - alpha) * plip_case_df.loc[:, prob_cols].to_numpy()
    blended["pred_idx"] = blended.loc[:, prob_cols].to_numpy().argmax(axis=1)
    return blended


def select_best_blend_alpha(
    dino_case_df: pd.DataFrame,
    plip_case_df: pd.DataFrame,
    alpha_grid: list[float],
    task,
) -> tuple[float, float]:
    best_alpha = float("nan")
    best_metric: float | None = None
    for alpha in alpha_grid:
        case_df = blend_case_predictions(dino_case_df, plip_case_df, alpha=alpha, task=task)
        metric_value = compute_selection_metric(task, case_df)
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_alpha = float(alpha)
            best_metric = float(metric_value)
    if best_metric is None or math.isnan(best_alpha):
        raise RuntimeError("Failed to select blend alpha.")
    return best_alpha, best_metric


def summarize_modes(
    fold_dir: Path,
    split_name: str,
    mode_to_case_df: dict[str, pd.DataFrame],
    task,
) -> dict[str, dict[str, float]]:
    frames: list[pd.DataFrame] = []
    out: dict[str, dict[str, float]] = {}
    for mode_name, case_df in mode_to_case_df.items():
        metrics = summarize_prediction_frame(case_df, task.class_names)
        out[mode_name] = metrics
        frames.append(metrics_to_frame(split_name, "case", mode_name, metrics))
    pd.concat(frames, ignore_index=True).to_csv(fold_dir / f"{split_name}_metrics_by_mode.csv", index=False)
    return out


def aggregate_run_outputs(root_output_dir: Path, task, fold_ids: list[int]) -> None:
    summary_rows: list[dict[str, Any]] = []
    mode_to_frames: dict[str, list[pd.DataFrame]] = {}
    for fold_id in fold_ids:
        fold_dir = root_output_dir / f"fold_{fold_id}"
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        summary_rows.append(summary)
        for case_file in fold_dir.glob("test_case_predictions_*.csv"):
            mode_name = case_file.stem.replace("test_case_predictions_", "")
            frame = pd.read_csv(case_file)
            frame["fold_id"] = fold_id
            mode_to_frames.setdefault(mode_name, []).append(frame)
    pd.DataFrame(summary_rows).to_csv(root_output_dir / "cv_fold_summary.csv", index=False)

    metric_frames: list[pd.DataFrame] = []
    for mode_name, frames in sorted(mode_to_frames.items()):
        case_df = pd.concat(frames, ignore_index=True)
        case_df.to_csv(root_output_dir / f"oof_case_predictions_{mode_name}.csv", index=False)
        metrics = summarize_prediction_frame(case_df, task.class_names)
        metric_frames.append(metrics_to_frame("test_oof", "case", mode_name, metrics))
    pd.concat(metric_frames, ignore_index=True).to_csv(root_output_dir / "oof_metrics_by_mode.csv", index=False)

    best_mode = "stack_case"
    if best_mode in mode_to_frames:
        case_df = pd.concat(mode_to_frames[best_mode], ignore_index=True)
        pd.concat([metrics_to_frame("test_oof", "case", best_mode, summarize_prediction_frame(case_df, task.class_names))], ignore_index=True).to_csv(
            root_output_dir / "oof_metrics.csv", index=False
        )


def run_single_fold(
    args: argparse.Namespace,
    task,
    image_df: pd.DataFrame,
    dino_model: torch.nn.Module,
    plip_model,
    plip_processor,
    device: torch.device,
    fold_id: int,
    fold_count: int,
) -> dict[str, Any]:
    fold_dir = Path(args.output_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=fold_count)
    train_df = trim_image_df(train_df, args.max_train_images)
    val_df = trim_image_df(val_df, args.max_val_images)
    test_df = trim_image_df(test_df, args.max_test_images)

    base_c_grid = parse_float_grid(args.base_c_grid)
    meta_c_grid = parse_float_grid(args.meta_c_grid)
    alpha_grid = parse_float_grid(args.alpha_grid)

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
        f"\n=== DINO+PLIP Stack Fold {fold_id}/{fold_count} | task={task.key} | dino={args.dino_model_name} ===",
        flush=True,
    )
    print(
        f"train_cases={train_df['case_id'].nunique()}, val_cases={val_df['case_id'].nunique()}, test_cases={test_df['case_id'].nunique()}, "
        f"train_images={len(train_df)}, val_images={len(val_df)}, test_images={len(test_df)}",
        flush=True,
    )

    def extract_dino(df: pd.DataFrame):
        return extract_dino_dataset_features(
            image_df=df,
            input_variant="whole",
            feature_mode=args.dino_feature_mode,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model=dino_model,
            device=device,
        )

    def extract_plip(df: pd.DataFrame):
        return extract_plip_dataset_features(
            image_df=df,
            input_variant="whole",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            processor=plip_processor,
            model=plip_model,
            device=device,
        )

    dino_train = extract_dino(train_df)
    dino_val = extract_dino(val_df)
    dino_test = extract_dino(test_df)
    plip_train = extract_plip(train_df)
    plip_val = extract_plip(val_df)
    plip_test = extract_plip(test_df)

    dino_model_probe, dino_best_c, dino_best_val_metric = fit_logreg_probe(
        train_features=dino_train[0],
        train_labels=dino_train[1],
        val_features=dino_val[0],
        val_labels=dino_val[1],
        val_case_ids=dino_val[2],
        val_image_names=dino_val[3],
        task=task,
        c_grid=base_c_grid,
        seed=args.seed,
    )
    plip_model_probe, plip_best_c, plip_best_val_metric = fit_logreg_probe(
        train_features=plip_train[0],
        train_labels=plip_train[1],
        val_features=plip_val[0],
        val_labels=plip_val[1],
        val_case_ids=plip_val[2],
        val_image_names=plip_val[3],
        task=task,
        c_grid=base_c_grid,
        seed=args.seed,
    )

    def make_case_block(model_probe, split_block):
        probs = model_probe.predict_proba(split_block[0])
        image_df = build_prediction_df(probs, split_block[1], split_block[2], split_block[3], task.class_names)
        return case_prob_feature_block(image_df, task)

    dino_train_block = make_case_block(dino_model_probe, dino_train)
    dino_val_block = make_case_block(dino_model_probe, dino_val)
    dino_test_block = make_case_block(dino_model_probe, dino_test)
    plip_train_block = make_case_block(plip_model_probe, plip_train)
    plip_val_block = make_case_block(plip_model_probe, plip_val)
    plip_test_block = make_case_block(plip_model_probe, plip_test)

    train_meta, train_labels, train_case_ids, train_image_names, dino_train_case_df, plip_train_case_df = align_case_blocks(dino_train_block, plip_train_block)
    val_meta, val_labels, val_case_ids, val_image_names, dino_val_case_df, plip_val_case_df = align_case_blocks(dino_val_block, plip_val_block)
    test_meta, test_labels, test_case_ids, test_image_names, dino_test_case_df, plip_test_case_df = align_case_blocks(dino_test_block, plip_test_block)

    meta_probe, meta_best_c, meta_best_val_metric = fit_logreg_probe(
        train_features=train_meta,
        train_labels=train_labels,
        val_features=val_meta,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
        c_grid=meta_c_grid,
        seed=args.seed,
    )
    best_alpha, best_alpha_val_metric = select_best_blend_alpha(dino_val_case_df, plip_val_case_df, alpha_grid=alpha_grid, task=task)

    stack_val_probs = meta_probe.predict_proba(val_meta)
    stack_test_probs = meta_probe.predict_proba(test_meta)
    stack_val_case_df = build_prediction_df(stack_val_probs, val_labels, val_case_ids, val_image_names, task.class_names)
    stack_test_case_df = build_prediction_df(stack_test_probs, test_labels, test_case_ids, test_image_names, task.class_names)
    blend_val_case_df = blend_case_predictions(dino_val_case_df, plip_val_case_df, alpha=best_alpha, task=task)
    blend_test_case_df = blend_case_predictions(dino_test_case_df, plip_test_case_df, alpha=best_alpha, task=task)

    val_modes = {
        "dino_case": dino_val_case_df,
        "plip_case": plip_val_case_df,
        "blend_case": blend_val_case_df,
        "stack_case": stack_val_case_df,
    }
    test_modes = {
        "dino_case": dino_test_case_df,
        "plip_case": plip_test_case_df,
        "blend_case": blend_test_case_df,
        "stack_case": stack_test_case_df,
    }
    val_metrics_by_mode = summarize_modes(fold_dir, "val", val_modes, task)
    test_metrics_by_mode = summarize_modes(fold_dir, "test", test_modes, task)

    for mode_name, case_df in test_modes.items():
        case_df.to_csv(fold_dir / f"test_case_predictions_{mode_name}.csv", index=False)
    for mode_name, case_df in val_modes.items():
        case_df.to_csv(fold_dir / f"val_case_predictions_{mode_name}.csv", index=False)

    fold_summary = {
        "fold_id": int(fold_id),
        "dino_best_c": float(dino_best_c),
        "plip_best_c": float(plip_best_c),
        "meta_best_c": float(meta_best_c),
        "best_alpha": float(best_alpha),
        "dino_best_val_primary_metric": float(dino_best_val_metric),
        "plip_best_val_primary_metric": float(plip_best_val_metric),
        "meta_best_val_primary_metric": float(meta_best_val_metric),
        "alpha_best_val_primary_metric": float(best_alpha_val_metric),
        "test_metrics_by_mode": test_metrics_by_mode,
    }
    write_json(fold_dir / "fold_summary.json", fold_summary)

    stack_metrics = test_metrics_by_mode["stack_case"]
    print(
        f"[Fold {fold_id}] stack_case: "
        + " | ".join(f"{key}={format_metric_value(value)}" for key, value in stack_metrics.items()),
        flush=True,
    )
    return fold_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    if len(task.class_names) != 5:
        raise ValueError("This script currently targets the 5-class Task4 setting.")

    device = resolve_device(args.device)
    image_df = load_task_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    available_folds = load_available_folds(args.split_csv)
    if args.fold == "all":
        fold_ids = available_folds
    else:
        fold_id = int(args.fold)
        if fold_id not in available_folds:
            raise ValueError(f"Fold {fold_id} is not available in split CSV.")
        fold_ids = [fold_id]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dino_model = load_dinov2_model(Path(args.dino_repo_dir), args.dino_model_name, device=device)
    plip_model, plip_processor = load_plip_model(Path(args.plip_model_dir), device=device)

    summaries = []
    for fold_id in fold_ids:
        summaries.append(
            run_single_fold(
                args=args,
                task=task,
                image_df=image_df,
                dino_model=dino_model,
                plip_model=plip_model,
                plip_processor=plip_processor,
                device=device,
                fold_id=fold_id,
                fold_count=len(available_folds),
            )
        )

    if args.fold == "all":
        aggregate_run_outputs(output_dir, task, fold_ids)

    print(f"\nCompleted {len(summaries)} fold(s). Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
