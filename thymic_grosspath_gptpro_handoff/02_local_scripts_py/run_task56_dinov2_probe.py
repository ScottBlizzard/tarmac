from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from run_dinov2_frozen_probe import (
    aggregate_features_to_cases,
    aggregate_run_outputs,
    build_prediction_df,
    extract_dataset_features,
    fit_probe,
    format_metric_value,
    load_available_folds,
    load_dinov2_model,
    save_split_outputs,
    set_seed,
    trim_image_df,
    write_json,
)
from thymic_baseline.config import DEFAULT_IMAGE_SIZE, DEFAULT_RANDOM_SEED, INPUT_VARIANTS
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import (
    expand_registry_to_images,
    filter_registry_for_task,
    load_registry,
    load_split_assignments,
    merge_registry_with_splits,
    subset_by_fold,
)
from thymic_baseline.train import resolve_device


@dataclass(frozen=True)
class Task56Config:
    key: str
    label_column: str
    class_names: tuple[str, ...]
    paper_role: str
    primary_metric: str

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2


TASK56: dict[str, Task56Config] = {
    "task5_threeclass": Task56Config(
        key="task5_threeclass",
        label_column="task_l5_label",
        class_names=("A_AB", "B123", "TC"),
        paper_role="new_primary_candidate",
        primary_metric="macro_f1",
    ),
    "task6_sixclass": Task56Config(
        key="task6_sixclass",
        label_column="task_l6_label",
        class_names=("A", "AB", "B1", "B2", "B3", "TC"),
        paper_role="new_exploratory_candidate",
        primary_metric="macro_f1",
    ),
    "task7_lowhigh_tc": Task56Config(
        key="task7_lowhigh_tc",
        label_column="task_l7_label",
        class_names=("low_risk_group", "high_risk_group"),
        paper_role="new_primary_candidate",
        primary_metric="auc",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv2 frozen probe for Task5/Task6/Task7.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASK56))
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--probe-type", default="logreg", choices=("logreg", "mlp", "lda"))
    parser.add_argument("--fit-level", default="image", choices=("image", "case"))
    parser.add_argument("--case-feature-agg", default="mean", choices=("mean",))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-epochs", type=int, default=60)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument("--lda-shrinkage", default="auto")
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    return parser.parse_args()


def get_task(task_key: str) -> Task56Config:
    if task_key not in TASK56:
        raise KeyError(f"Unknown task: {task_key}. Available: {', '.join(TASK56)}")
    return TASK56[task_key]


def load_task56_image_df(registry_csv: str, split_csv: str, images_root: str, task: Task56Config) -> pd.DataFrame:
    registry = load_registry(registry_csv)
    if task.label_column not in registry.columns:
        raise ValueError(f"Registry CSV does not contain required column: {task.label_column}")
    split_df = load_split_assignments(split_csv)
    merged = merge_registry_with_splits(registry, split_df)
    filtered = filter_registry_for_task(merged, task)
    return expand_registry_to_images(filtered, task=task, images_root=images_root)


def run_single_fold(
    args: argparse.Namespace,
    task: Task56Config,
    image_df: pd.DataFrame,
    model: torch.nn.Module,
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
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

    run_config = {
        "task": task.key,
        "input_variant": args.input_variant,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "model_name": args.model_name,
        "feature_mode": args.feature_mode,
        "probe_type": args.probe_type,
        "fit_level": args.fit_level,
        "case_feature_agg": args.case_feature_agg,
        "fold_id": fold_id,
        "device": str(device),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "test_images": int(len(test_df)),
        "train_cases": int(train_df["case_id"].nunique()),
        "val_cases": int(val_df["case_id"].nunique()),
        "test_cases": int(test_df["case_id"].nunique()),
        "c_grid": [float(item) for item in args.c_grid.split(",") if item.strip()],
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== Task56 DINOv2 Fold {fold_id}/{fold_count} | task={task.key} | variant={args.input_variant} | model={args.model_name} ===",
        flush=True,
    )
    print(
        f"feature_mode={args.feature_mode}, train_cases={run_config['train_cases']}, val_cases={run_config['val_cases']}, "
        f"test_cases={run_config['test_cases']}, train_images={run_config['train_images']}, "
        f"val_images={run_config['val_images']}, test_images={run_config['test_images']}",
        flush=True,
    )

    train_features, train_labels, _, _ = extract_dataset_features(
        train_df,
        input_variant=args.input_variant,
        feature_mode=args.feature_mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )
    val_features, val_labels, val_case_ids, val_image_names = extract_dataset_features(
        val_df,
        input_variant=args.input_variant,
        feature_mode=args.feature_mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )
    test_features, test_labels, test_case_ids, test_image_names = extract_dataset_features(
        test_df,
        input_variant=args.input_variant,
        feature_mode=args.feature_mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )

    if args.fit_level == "case":
        train_features, train_labels, _, _ = aggregate_features_to_cases(
            train_features, train_labels, train_df["case_id"].astype(str).tolist(), train_df["image_name"].astype(str).tolist(), args.case_feature_agg
        )
        val_features, val_labels, val_case_ids, val_image_names = aggregate_features_to_cases(
            val_features, val_labels, val_case_ids, val_image_names, args.case_feature_agg
        )
        test_features, test_labels, test_case_ids, test_image_names = aggregate_features_to_cases(
            test_features, test_labels, test_case_ids, test_image_names, args.case_feature_agg
        )

    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
    probe, best_c, best_val_metric = fit_probe(
        args=args,
        train_features=train_features,
        train_labels=train_labels,
        train_sample_weights=None,
        val_features=val_features,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
        c_grid=c_grid,
        seed=args.seed,
    )

    val_predictions = build_prediction_df(
        probs=probe.predict_proba(val_features),
        labels=val_labels,
        case_ids=val_case_ids,
        image_names=val_image_names,
        class_names=task.class_names,
    )
    test_predictions = build_prediction_df(
        probs=probe.predict_proba(test_features),
        labels=test_labels,
        case_ids=test_case_ids,
        image_names=test_image_names,
        class_names=task.class_names,
    )

    val_case_metrics = save_split_outputs(fold_dir, "val", val_predictions, task, aggregate_methods)
    test_case_metrics = save_split_outputs(fold_dir, "test", test_predictions, task, aggregate_methods)
    fold_summary = {
        "best_c": float(best_c),
        "best_val_primary_metric": float(best_val_metric),
        "val_case_mean": val_case_metrics["mean"],
        "test_case_mean": test_case_metrics["mean"],
    }
    write_json(fold_dir / "fold_summary.json", fold_summary)
    print(
        f"[Fold {fold_id}] best_c={best_c:.4g} | "
        f"val_{task.primary_metric}={format_metric_value(val_case_metrics['mean'].get(task.primary_metric))} | "
        f"test_{task.primary_metric}={format_metric_value(test_case_metrics['mean'].get(task.primary_metric))} | "
        f"test_acc={format_metric_value(test_case_metrics['mean'].get('accuracy'))}",
        flush=True,
    )
    return fold_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    task = get_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]

    print(
        "Resolved inputs: "
        f"registry_csv={args.registry_csv}, "
        f"split_csv={args.split_csv}, "
        f"images_root={args.images_root}, "
        f"output_dir={args.output_dir}",
        flush=True,
    )

    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    repo_dir = Path(args.repo_dir)
    model = load_dinov2_model(repo_dir=repo_dir, model_name=args.model_name, device=device)

    print(
        f"Starting Task56 DINOv2 probe: task={task.key}, variant={args.input_variant}, "
        f"model={args.model_name}, feature_mode={args.feature_mode}, device={device}, folds={fold_ids}",
        flush=True,
    )
    for fold_id in fold_ids:
        run_single_fold(
            args=args,
            task=task,
            image_df=image_df,
            model=model,
            device=device,
            fold_id=fold_id,
            fold_count=len(available_folds),
        )

    if len(fold_ids) > 1:
        aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
        aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
