from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_dinov2_frozen_probe import (
    aggregate_weights_to_cases,
    aggregate_features_to_cases,
    aggregate_run_outputs,
    build_sample_weights,
    build_prediction_df,
    extract_dataset_features,
    fit_probe,
    format_metric_value,
    load_available_folds,
    load_dinov2_model,
    parse_subtype_weight_map,
    save_split_outputs,
    set_seed,
    trim_image_df,
    write_json,
)
from run_task56_dinov2_probe import TASK56, get_task, load_task56_image_df
from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


DEFAULT_DINO_IMAGE_SIZE = 518


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concatenated DINOv2 frozen probe for Task5/Task6/Task7.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASK56))
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", required=True, help="Comma-separated two model names.")
    parser.add_argument("--feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--probe-type", default="logreg", choices=("logreg", "mlp", "lda"))
    parser.add_argument("--fit-level", default="image", choices=("image", "case"))
    parser.add_argument("--case-feature-agg", default="mean", choices=("mean",))
    parser.add_argument(
        "--selection-metric",
        default="primary",
        choices=("primary", "accuracy", "balanced_accuracy", "f1"),
        help="Validation metric used to select the best probe/checkpoint. 'primary' means task primary metric.",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_DINO_IMAGE_SIZE)
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
    parser.add_argument(
        "--logreg-class-weight",
        default="balanced",
        choices=("balanced", "none"),
        help="Class-weight mode for logistic regression probe.",
    )
    parser.add_argument(
        "--subtype-weight-map",
        default="",
        help="Optional subtype sample-weight map, e.g. 'thymoma_B2:1.5,thymic_carcinoma:1.4'.",
    )
    return parser.parse_args()


def extract_concat_features(
    image_df,
    model_names: list[str],
    models: list[torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
):
    features_list = []
    labels_ref = None
    case_ids_ref = None
    image_names_ref = None
    for model_name, model in zip(model_names, models):
        features, labels, case_ids, image_names = extract_dataset_features(
            image_df,
            input_variant=args.input_variant,
            feature_mode=args.feature_mode,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model=model,
            device=device,
        )
        if labels_ref is None:
            labels_ref = labels
            case_ids_ref = case_ids
            image_names_ref = image_names
        else:
            if not np.array_equal(labels_ref, labels):
                raise ValueError(f"Label mismatch while concatenating features for {model_name}.")
            if case_ids_ref != case_ids or image_names_ref != image_names:
                raise ValueError(f"Sample order mismatch while concatenating features for {model_name}.")
        features_list.append(features)
    return np.concatenate(features_list, axis=1), labels_ref, case_ids_ref, image_names_ref


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if len(model_names) != 2:
        raise ValueError("--model-names must contain exactly two model names.")

    task = get_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    repo_dir = Path(args.repo_dir)
    models = [load_dinov2_model(repo_dir=repo_dir, model_name=name, device=device) for name in model_names]

    print(
        f"Starting Task56 concat probe: task={task.key}, models={model_names}, "
        f"feature_mode={args.feature_mode}, device={device}, folds={fold_ids}",
        flush=True,
    )
    subtype_weight_map = parse_subtype_weight_map(args.subtype_weight_map)

    for fold_id in fold_ids:
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
        train_df = trim_image_df(train_df, args.max_train_images)
        val_df = trim_image_df(val_df, args.max_val_images)
        test_df = trim_image_df(test_df, args.max_test_images)
        aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

        run_config = {
            "task": task.key,
            "input_variant": args.input_variant,
            "repo_dir": str(repo_dir.resolve()),
            "model_names": model_names,
            "feature_mode": args.feature_mode,
            "probe_type": args.probe_type,
            "fit_level": args.fit_level,
            "case_feature_agg": args.case_feature_agg,
            "selection_metric": args.selection_metric,
            "logreg_class_weight": args.logreg_class_weight,
            "subtype_weight_map": subtype_weight_map,
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
            f"\n=== Task56 Concat Fold {fold_id}/{len(available_folds)} | task={task.key} | models={model_names} ===",
            flush=True,
        )

        train_features, train_labels, _, _ = extract_concat_features(train_df, model_names, models, args, device)
        val_features, val_labels, val_case_ids, val_image_names = extract_concat_features(val_df, model_names, models, args, device)
        test_features, test_labels, test_case_ids, test_image_names = extract_concat_features(test_df, model_names, models, args, device)
        train_sample_weights = build_sample_weights(train_df["who_type_raw"].astype(str).tolist(), subtype_weight_map)

        if args.fit_level == "case":
            train_features, train_labels, _, _ = aggregate_features_to_cases(
                train_features,
                train_labels,
                train_df["case_id"].astype(str).tolist(),
                train_df["image_name"].astype(str).tolist(),
                args.case_feature_agg,
            )
            val_features, val_labels, val_case_ids, val_image_names = aggregate_features_to_cases(
                val_features, val_labels, val_case_ids, val_image_names, args.case_feature_agg
            )
            test_features, test_labels, test_case_ids, test_image_names = aggregate_features_to_cases(
                test_features, test_labels, test_case_ids, test_image_names, args.case_feature_agg
            )
            train_sample_weights = aggregate_weights_to_cases(
                train_sample_weights,
                train_df["case_id"].astype(str).tolist(),
            )

        c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
        probe, best_c, best_val_metric = fit_probe(
            args=args,
            train_features=train_features,
            train_labels=train_labels,
            train_sample_weights=train_sample_weights,
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

    if len(fold_ids) > 1:
        aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
        aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
