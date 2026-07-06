from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from run_dinov2_frozen_probe import (
    StandardizedMLPProbe,
    aggregate_features_to_cases,
    aggregate_run_outputs,
    build_prediction_df,
    compute_selection_metric,
    extract_dataset_features,
    format_metric_value,
    load_available_folds,
    load_dinov2_model,
    save_split_outputs,
    set_seed,
    trim_image_df,
    write_json,
)
from run_task56_dinov2_probe import TASK56, get_task, load_task56_image_df
from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS
from thymic_baseline.metrics import summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


DEFAULT_DINO_IMAGE_SIZE = 518


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task7 curriculum case-level MLP on dual DINO frozen concat features.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", default="task7_lowhigh_tc", choices=tuple(TASK56))
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", required=True, help="Comma-separated two model names.")
    parser.add_argument("--feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--case-feature-agg", default="mean", choices=("mean",))
    parser.add_argument("--selection-metric", default="accuracy", choices=("accuracy", "balanced_accuracy", "f1"))
    parser.add_argument("--image-size", type=int, default=DEFAULT_DINO_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage1-patience", type=int, default=8)
    parser.add_argument("--stage2-epochs", type=int, default=35)
    parser.add_argument("--stage2-patience", type=int, default=6)
    parser.add_argument("--stage3-epochs", type=int, default=35)
    parser.add_argument("--stage3-patience", type=int, default=6)
    parser.add_argument("--teacher-oof-csvs", required=True, help="Comma-separated OOF case prediction CSVs used to derive easy/medium/hard.")
    parser.add_argument("--easy-min-correct", type=int, default=3)
    parser.add_argument("--easy-mean-true-prob", type=float, default=0.70)
    parser.add_argument("--easy-min-true-prob", type=float, default=0.60)
    parser.add_argument("--medium-min-correct", type=int, default=2)
    parser.add_argument("--medium-mean-true-prob", type=float, default=0.55)
    parser.add_argument("--hard-weight", type=float, default=0.50)
    parser.add_argument("--salvage-hard-weight", type=float, default=None)
    parser.add_argument(
        "--stage3-hard-mode",
        default="all",
        choices=("all", "salvage"),
        help="Use all hard cases in stage3, or only already-correct plus teacher-salvage hard cases.",
    )
    parser.add_argument(
        "--stop-after-stage",
        default="stage3_all",
        choices=("stage1_easy", "stage2_easy_medium", "stage3_all"),
        help="Stop training after the specified curriculum stage and use that checkpoint for final evaluation.",
    )
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    return parser.parse_args()


def extract_concat_features(
    image_df: pd.DataFrame,
    model_names: list[str],
    models: list[torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    features_list = []
    labels_ref = None
    case_ids_ref = None
    image_names_ref = None
    for model_name, model in zip(model_names, models):
        features, labels, case_ids, image_names = extract_dataset_features(
            image_df=image_df,
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


def load_curriculum_table(teacher_oof_csvs: list[Path], args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for idx, path in enumerate(teacher_oof_csvs):
        name = f"t{idx+1}"
        df = pd.read_csv(path)
        need = ["case_id", "label_idx", "pred_idx", "prob_low_risk_group", "prob_high_risk_group", "fold_id"]
        missing = [col for col in need if col not in df.columns]
        if missing:
            raise KeyError(f"{path} missing columns: {missing}")
        df = df[need].copy()
        df = df.rename(columns={col: f"{name}_{col}" for col in need if col != "case_id"})
        frames.append(df)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="case_id", how="inner")
    merged["label_idx"] = merged["t1_label_idx"].astype(int)
    merged["fold_id"] = merged["t1_fold_id"].astype(int)
    teacher_names = [f"t{idx+1}" for idx in range(len(teacher_oof_csvs))]
    for name in teacher_names:
        merged[f"{name}_correct"] = (merged[f"{name}_pred_idx"].astype(int) == merged["label_idx"]).astype(int)
        merged[f"{name}_true_prob"] = np.where(
            merged["label_idx"].to_numpy() == 1,
            merged[f"{name}_prob_high_risk_group"].to_numpy(),
            merged[f"{name}_prob_low_risk_group"].to_numpy(),
        )
        merged[f"{name}_margin"] = np.abs(merged[f"{name}_prob_high_risk_group"].to_numpy() - 0.5) * 2.0
    merged["correct_count"] = merged[[f"{name}_correct" for name in teacher_names]].sum(axis=1)
    merged["mean_true_prob"] = merged[[f"{name}_true_prob" for name in teacher_names]].mean(axis=1)
    merged["min_true_prob"] = merged[[f"{name}_true_prob" for name in teacher_names]].min(axis=1)
    merged["mean_margin"] = merged[[f"{name}_margin" for name in teacher_names]].mean(axis=1)
    pred_cols = [f"{name}_pred_idx" for name in teacher_names]
    merged["all_agree_pred"] = merged[pred_cols].nunique(axis=1) == 1

    easy_mask = (
        (merged["correct_count"] >= args.easy_min_correct)
        & merged["all_agree_pred"]
        & (merged["mean_true_prob"] >= args.easy_mean_true_prob)
        & (merged["min_true_prob"] >= args.easy_min_true_prob)
    )
    medium_mask = (
        ~easy_mask
        & (merged["correct_count"] >= args.medium_min_correct)
        & (merged["mean_true_prob"] >= args.medium_mean_true_prob)
    )
    merged["difficulty"] = np.where(easy_mask, "easy", np.where(medium_mask, "medium", "hard"))
    hard_salvage_mask = (
        (merged["difficulty"] == "hard")
        & (merged["correct_count"] >= 1)
        & (merged["mean_true_prob"] >= 0.35)
    )
    hard_core_mask = merged["difficulty"] == "hard"
    merged["difficulty_fine"] = merged["difficulty"]
    merged.loc[hard_core_mask, "difficulty_fine"] = "hard_core"
    merged.loc[hard_salvage_mask, "difficulty_fine"] = "hard_salvage_teacher"
    return merged[
        [
            "case_id",
            "fold_id",
            "label_idx",
            "difficulty",
            "difficulty_fine",
            "correct_count",
            "mean_true_prob",
            "min_true_prob",
            "mean_margin",
        ]
    ].copy()


def build_case_feature_df(
    features: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    image_names: list[str],
    agg: str,
) -> pd.DataFrame:
    case_features, case_labels, case_ids, case_image_names = aggregate_features_to_cases(
        features,
        labels,
        case_ids,
        image_names,
        agg,
    )
    return pd.DataFrame(
        {
            "case_id": case_ids,
            "label_idx": case_labels.astype(int),
            "image_name": case_image_names,
            "feature_idx": np.arange(len(case_ids), dtype=int),
        }
    ), case_features


def train_stage(
    stage_name: str,
    model: StandardizedMLPProbe,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_case_ids: list[str],
    train_weights: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
    selection_metric: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_inputs = torch.from_numpy(train_x.astype(np.float32, copy=False))
    train_targets = torch.from_numpy(train_y.astype(np.int64, copy=False))
    train_sample_weights = torch.from_numpy(train_weights.astype(np.float32, copy=False))
    val_inputs = torch.from_numpy(val_x.astype(np.float32, copy=False)).to(device)

    class_counts = np.bincount(train_y, minlength=len(task.class_names)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weight_tensor = torch.from_numpy(class_weights).to(device)

    batch_size = min(64, len(train_inputs))
    rng = np.random.default_rng(seed)
    best_state = None
    best_metric = None
    best_epoch = 0
    best_metrics_dict = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = rng.permutation(len(train_inputs))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_inputs = train_inputs[batch_idx].to(device)
            batch_targets = train_targets[batch_idx].to(device)
            batch_weights = train_sample_weights[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            ce = F.cross_entropy(logits, batch_targets, weight=class_weight_tensor, reduction="none")
            loss = (ce * batch_weights).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model(val_inputs), dim=1).cpu().numpy().astype(np.float64)
        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=val_y,
            case_ids=val_case_ids,
            image_names=val_image_names,
            class_names=task.class_names,
        )
        metric_value = compute_selection_metric(task, val_predictions, selection_metric)
        val_mean = summarize_prediction_frame(val_predictions, task.class_names)
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_metric = float(metric_value)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics_dict = val_mean
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None or best_metrics_dict is None:
        raise RuntimeError(f"{stage_name}: failed to train a valid stage.")
    return best_state, {
        "best_epoch": int(best_epoch),
        "best_val_primary_metric": float(best_metric),
        "best_val_metrics": {k: float(v) for k, v in best_metrics_dict.items()},
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if len(model_names) != 2:
        raise ValueError("--model-names must contain exactly two model names.")
    teacher_oof_csvs = [Path(item.strip()) for item in args.teacher_oof_csvs.split(",") if item.strip()]
    if not teacher_oof_csvs:
        raise ValueError("No teacher OOF CSVs provided.")

    difficulty_df = load_curriculum_table(teacher_oof_csvs, args)
    difficulty_df.to_csv(output_dir / "curriculum_case_table.csv", index=False)
    diff_summary = (
        difficulty_df.groupby(["fold_id", "difficulty"]).size().unstack(fill_value=0).reset_index()
    )
    diff_summary.to_csv(output_dir / "curriculum_fold_counts.csv", index=False)

    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    repo_dir = Path(args.repo_dir)
    models = [load_dinov2_model(repo_dir=repo_dir, model_name=name, device=device) for name in model_names]

    print(
        f"Starting Task7 curriculum probe: task={task.key}, models={model_names}, device={device}, folds={fold_ids}",
        flush=True,
    )

    for fold_id in fold_ids:
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
        train_df = trim_image_df(train_df, args.max_train_images)
        val_df = trim_image_df(val_df, args.max_val_images)
        test_df = trim_image_df(test_df, args.max_test_images)

        run_config = {
            "task": task.key,
            "model_names": model_names,
            "feature_mode": args.feature_mode,
            "fit_level": "case",
            "selection_metric": args.selection_metric,
            "teacher_oof_csvs": [str(p) for p in teacher_oof_csvs],
            "easy_rule": {
                "min_correct": int(args.easy_min_correct),
                "mean_true_prob": float(args.easy_mean_true_prob),
                "min_true_prob": float(args.easy_min_true_prob),
            },
            "medium_rule": {
                "min_correct": int(args.medium_min_correct),
                "mean_true_prob": float(args.medium_mean_true_prob),
            },
            "hard_weight": float(args.hard_weight),
            "stage3_hard_mode": args.stage3_hard_mode,
            "fold_id": int(fold_id),
        }
        write_json(fold_dir / "run_config.json", run_config)
        print(f"\n=== Curriculum Fold {fold_id}/{len(available_folds)} | task={task.key} ===", flush=True)

        train_features, train_labels, train_case_ids, train_image_names = extract_concat_features(train_df, model_names, models, args, device)
        val_features, val_labels, val_case_ids, val_image_names = extract_concat_features(val_df, model_names, models, args, device)
        test_features, test_labels, test_case_ids, test_image_names = extract_concat_features(test_df, model_names, models, args, device)

        train_case_df, train_case_features = build_case_feature_df(
            train_features, train_labels, train_case_ids, train_image_names, args.case_feature_agg
        )
        val_case_df, val_case_features = build_case_feature_df(
            val_features, val_labels, val_case_ids, val_image_names, args.case_feature_agg
        )
        test_case_df, test_case_features = build_case_feature_df(
            test_features, test_labels, test_case_ids, test_image_names, args.case_feature_agg
        )

        train_case_df = train_case_df.merge(difficulty_df[["case_id", "difficulty", "difficulty_fine"]], on="case_id", how="left")
        val_case_df = val_case_df.merge(difficulty_df[["case_id", "difficulty", "difficulty_fine"]], on="case_id", how="left")
        test_case_df = test_case_df.merge(difficulty_df[["case_id", "difficulty", "difficulty_fine"]], on="case_id", how="left")
        for df in (train_case_df, val_case_df, test_case_df):
            df["difficulty"] = df["difficulty"].fillna("hard")
            df["difficulty_fine"] = df["difficulty_fine"].fillna("hard_core")

        scaler_mean = train_case_features.mean(axis=0, keepdims=True)
        scaler_std = train_case_features.std(axis=0, keepdims=True)
        scaler_std = np.where(scaler_std < 1e-6, 1.0, scaler_std)
        train_case_features_z = ((train_case_features - scaler_mean) / scaler_std).astype(np.float32)
        val_case_features_z = ((val_case_features - scaler_mean) / scaler_std).astype(np.float32)
        test_case_features_z = ((test_case_features - scaler_mean) / scaler_std).astype(np.float32)

        input_dim = train_case_features_z.shape[1]
        model = StandardizedMLPProbe(
            input_dim=input_dim,
            hidden_dim=args.mlp_hidden_dim,
            num_classes=len(task.class_names),
            dropout=args.mlp_dropout,
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        stage3_train_fine_levels = None
        stage3_val_fine_levels = None
        if args.stage3_hard_mode == "salvage":
            stage3_train_fine_levels = ["easy", "medium", "hard_salvage_teacher"]
            stage3_val_fine_levels = ["easy", "medium", "hard_salvage_teacher"]

        stage_defs = [
            ("stage1_easy", ["easy"], ["easy"], None, None, args.stage1_epochs, args.stage1_patience, 1.0),
            ("stage2_easy_medium", ["easy", "medium"], ["easy", "medium"], None, None, args.stage2_epochs, args.stage2_patience, 1.0),
            ("stage3_all", ["easy", "medium", "hard"], ["easy", "medium", "hard"], stage3_train_fine_levels, stage3_val_fine_levels, args.stage3_epochs, args.stage3_patience, args.hard_weight),
        ]

        stage_summaries = {}
        final_state = None
        final_stage_name = None
        for stage_idx, (stage_name, train_levels, val_levels, train_fine_levels, val_fine_levels, epochs, patience, hard_weight) in enumerate(stage_defs, start=1):
            if train_fine_levels is not None:
                train_mask = train_case_df["difficulty_fine"].isin(train_fine_levels).to_numpy()
            else:
                train_mask = train_case_df["difficulty"].isin(train_levels).to_numpy()
            if val_fine_levels is not None:
                val_mask = val_case_df["difficulty_fine"].isin(val_fine_levels).to_numpy()
            else:
                val_mask = val_case_df["difficulty"].isin(val_levels).to_numpy()
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                raise RuntimeError(f"{stage_name}: empty train or val subset.")
            train_weights = np.ones(int(train_mask.sum()), dtype=np.float32)
            if "hard" in train_levels:
                stage_train_diff = train_case_df.loc[train_mask, "difficulty"].to_numpy()
                train_weights = np.where(stage_train_diff == "hard", float(hard_weight), 1.0).astype(np.float32)
            if args.stage3_hard_mode == "salvage" and stage_name == "stage3_all" and args.salvage_hard_weight is not None:
                stage_train_fine = train_case_df.loc[train_mask, "difficulty_fine"].to_numpy()
                train_weights = np.where(
                    stage_train_fine == "hard_salvage_teacher",
                    float(args.salvage_hard_weight),
                    train_weights,
                ).astype(np.float32)

            state, summary = train_stage(
                stage_name=stage_name,
                model=model,
                train_x=train_case_features_z[train_mask],
                train_y=train_case_df.loc[train_mask, "label_idx"].to_numpy(dtype=np.int64),
                train_case_ids=train_case_df.loc[train_mask, "case_id"].astype(str).tolist(),
                train_weights=train_weights,
                val_x=val_case_features_z[val_mask],
                val_y=val_case_df.loc[val_mask, "label_idx"].to_numpy(dtype=np.int64),
                val_case_ids=val_case_df.loc[val_mask, "case_id"].astype(str).tolist(),
                val_image_names=val_case_df.loc[val_mask, "image_name"].astype(str).tolist(),
                task=task,
                selection_metric="accuracy" if stage_idx == 1 else args.selection_metric,
                epochs=epochs,
                patience=patience,
                lr=args.mlp_lr,
                weight_decay=args.mlp_weight_decay,
                seed=args.seed + stage_idx * 97 + fold_id,
            )
            model.load_state_dict(state)
            stage_summaries[stage_name] = summary
            final_state = state
            final_stage_name = stage_name
            if stage_name == args.stop_after_stage:
                break

        if final_state is None or final_stage_name is None:
            raise RuntimeError("No final curriculum stage was selected.")
        model.load_state_dict(final_state)

        easy_val_mask = (val_case_df["difficulty"] == "easy").to_numpy()
        with torch.no_grad():
            easy_val_probs = F.softmax(model(torch.from_numpy(val_case_features_z[easy_val_mask]).to(next(model.parameters()).device)), dim=1).cpu().numpy().astype(np.float64)
            full_val_probs = F.softmax(model(torch.from_numpy(val_case_features_z).to(next(model.parameters()).device)), dim=1).cpu().numpy().astype(np.float64)
            test_probs = F.softmax(model(torch.from_numpy(test_case_features_z).to(next(model.parameters()).device)), dim=1).cpu().numpy().astype(np.float64)

        easy_val_predictions = build_prediction_df(
            probs=easy_val_probs,
            labels=val_case_df.loc[easy_val_mask, "label_idx"].to_numpy(dtype=np.int64),
            case_ids=val_case_df.loc[easy_val_mask, "case_id"].astype(str).tolist(),
            image_names=val_case_df.loc[easy_val_mask, "image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )
        full_val_predictions = build_prediction_df(
            probs=full_val_probs,
            labels=val_case_df["label_idx"].to_numpy(dtype=np.int64),
            case_ids=val_case_df["case_id"].astype(str).tolist(),
            image_names=val_case_df["image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )
        test_predictions = build_prediction_df(
            probs=test_probs,
            labels=test_case_df["label_idx"].to_numpy(dtype=np.int64),
            case_ids=test_case_df["case_id"].astype(str).tolist(),
            image_names=test_case_df["image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )
        easy_val_metrics = save_split_outputs(fold_dir, "val_easy", easy_val_predictions, task, ["mean"])["mean"]
        val_case_metrics = save_split_outputs(fold_dir, "val", full_val_predictions, task, ["mean"])
        test_case_metrics = save_split_outputs(fold_dir, "test", test_predictions, task, ["mean"])

        fold_summary = {
            "best_c": float("nan"),
            "best_val_primary_metric": float(stage_summaries[final_stage_name]["best_val_primary_metric"]),
            "final_stage": final_stage_name,
            "curriculum_counts": {
                "train": train_case_df["difficulty"].value_counts().to_dict(),
                "val": val_case_df["difficulty"].value_counts().to_dict(),
                "test": test_case_df["difficulty"].value_counts().to_dict(),
            },
            "curriculum_fine_counts": {
                "train": train_case_df["difficulty_fine"].value_counts().to_dict(),
                "val": val_case_df["difficulty_fine"].value_counts().to_dict(),
                "test": test_case_df["difficulty_fine"].value_counts().to_dict(),
            },
            "stage_summaries": stage_summaries,
            "val_easy_mean": easy_val_metrics,
            "val_case_mean": val_case_metrics["mean"],
            "test_case_mean": test_case_metrics["mean"],
        }
        write_json(fold_dir / "fold_summary.json", fold_summary)
        print(
            f"[Fold {fold_id}] easy_val_acc={format_metric_value(easy_val_metrics.get('accuracy'))} | "
            f"test_acc={format_metric_value(test_case_metrics['mean'].get('accuracy'))} | "
            f"test_bacc={format_metric_value(test_case_metrics['mean'].get('balanced_accuracy'))}",
            flush=True,
        )

    if len(fold_ids) > 1:
        aggregate_run_outputs(output_dir, task, fold_ids, ["mean"])
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
