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
    aggregate_run_outputs,
    build_prediction_df,
    compute_selection_metric,
    format_metric_value,
    load_available_folds,
    load_dinov2_model,
    save_split_outputs,
    set_seed,
    trim_image_df,
    write_json,
)
from run_task56_dinov2_probe import TASK56, get_task, load_task56_image_df
from run_task7_concat_curriculum_probe import (
    DEFAULT_DINO_IMAGE_SIZE,
    build_case_feature_df,
    extract_concat_features,
    load_curriculum_table,
)
from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS
from thymic_baseline.metrics import summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 curriculum MLP with gross-calibrated soft-teacher distillation.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", default="task7_lowhigh_tc", choices=tuple(TASK56))
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", required=True)
    parser.add_argument("--feature-mode", default="cls_patchmean", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--case-feature-agg", default="mean", choices=("mean",))
    parser.add_argument("--selection-metric", default="balanced_accuracy", choices=("accuracy", "balanced_accuracy", "f1"))
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
    parser.add_argument("--teacher-oof-csvs", required=True)
    parser.add_argument("--easy-min-correct", type=int, default=3)
    parser.add_argument("--easy-mean-true-prob", type=float, default=0.70)
    parser.add_argument("--easy-min-true-prob", type=float, default=0.60)
    parser.add_argument("--easy-mean-margin", type=float, default=0.0)
    parser.add_argument("--easy-exclude-label-idxs", default="")
    parser.add_argument("--medium-min-correct", type=int, default=2)
    parser.add_argument("--medium-mean-true-prob", type=float, default=0.60)
    parser.add_argument("--medium-mean-margin", type=float, default=0.0)
    parser.add_argument("--hard-salvage-mean-true-prob", type=float, default=0.35)
    parser.add_argument("--hard-salvage-mean-margin", type=float, default=0.0)
    parser.add_argument("--hard-weight", type=float, default=0.50)
    parser.add_argument("--stage3-hard-mode", default="salvage_labeled", choices=("all", "salvage", "salvage_labeled"))
    parser.add_argument("--stop-after-stage", default="stage3_all", choices=("stage1_easy", "stage2_easy_medium", "stage3_all"))
    parser.add_argument("--gross-teacher-csv", required=True)
    parser.add_argument("--gross-teacher-prob-column", default="gross_calibrated_prob_high_risk_group")
    parser.add_argument("--gross-teacher-weight", type=float, default=0.05)
    parser.add_argument("--gross-teacher-start-stage", default="stage3", choices=("stage1", "stage2", "stage3"))
    parser.add_argument("--gross-teacher-statuses", default="rescued_by_gross,stable_correct_after_gross")
    parser.add_argument("--gross-teacher-clip", type=float, default=0.05)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    return parser.parse_args()


def robust_load_curriculum(teacher_oof_csvs: list[Path], args: argparse.Namespace, task) -> pd.DataFrame:
    try:
        return load_curriculum_table(teacher_oof_csvs, args, task)
    except TypeError:
        return load_curriculum_table(teacher_oof_csvs, args)


def load_teacher_table(path: Path, prob_col: str, statuses: set[str], clip: float) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str}).fillna("")
    if statuses and "gross_delta_status" in df.columns:
        df = df[df["gross_delta_status"].isin(statuses)].copy()
    if prob_col not in df.columns:
        raise KeyError(f"{path} missing gross teacher probability column: {prob_col}")
    out = df[["case_id", prob_col]].copy()
    out[prob_col] = pd.to_numeric(out[prob_col], errors="coerce")
    out = out.dropna(subset=[prob_col])
    out[prob_col] = out[prob_col].clip(float(clip), 1.0 - float(clip))
    return out.rename(columns={prob_col: "gross_teacher_prob"})


def build_teacher_probs(case_ids: list[str], teacher_df: pd.DataFrame) -> np.ndarray:
    lookup = dict(zip(teacher_df["case_id"].astype(str), teacher_df["gross_teacher_prob"].astype(float)))
    arr = np.full(len(case_ids), np.nan, dtype=np.float32)
    for idx, case_id in enumerate(case_ids):
        value = lookup.get(str(case_id))
        if value is not None:
            arr[idx] = float(value)
    return arr


def train_stage_distill(
    stage_name: str,
    model: StandardizedMLPProbe,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_teacher_prob: np.ndarray,
    train_weights: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
    selection_metric: str,
    gross_teacher_weight: float,
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
    teacher_probs = torch.from_numpy(train_teacher_prob.astype(np.float32, copy=False))
    teacher_mask = torch.isfinite(teacher_probs)
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
            batch_teacher = teacher_probs[batch_idx].to(device)
            batch_teacher_mask = teacher_mask[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            ce = F.cross_entropy(logits, batch_targets, weight=class_weight_tensor, reduction="none")
            loss = (ce * batch_weights).mean()
            if gross_teacher_weight > 0 and bool(batch_teacher_mask.any()):
                high_logit = logits[:, 1] - logits[:, 0]
                distill = F.binary_cross_entropy_with_logits(
                    high_logit[batch_teacher_mask],
                    batch_teacher[batch_teacher_mask],
                    reduction="mean",
                )
                loss = loss + float(gross_teacher_weight) * distill
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
        raise RuntimeError(f"{stage_name}: failed to train.")
    return best_state, {
        "best_epoch": int(best_epoch),
        "best_val_primary_metric": float(best_metric),
        "best_val_metrics": {k: float(v) for k, v in best_metrics_dict.items()},
        "gross_teacher_coverage": int(np.isfinite(train_teacher_prob).sum()),
        "gross_teacher_weight": float(gross_teacher_weight),
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
    gross_statuses = {item.strip() for item in args.gross_teacher_statuses.split(",") if item.strip()}
    gross_teacher_df = load_teacher_table(
        Path(args.gross_teacher_csv),
        prob_col=args.gross_teacher_prob_column,
        statuses=gross_statuses,
        clip=args.gross_teacher_clip,
    )
    gross_teacher_df.to_csv(output_dir / "gross_teacher_cases_used.csv", index=False)

    difficulty_df = robust_load_curriculum(teacher_oof_csvs, args, task)
    if "difficulty_fine" not in difficulty_df.columns:
        difficulty_df["difficulty_fine"] = difficulty_df["difficulty"]
        hard_mask = difficulty_df["difficulty"] == "hard"
        salvage_mask = hard_mask & (difficulty_df["correct_count"] >= 1) & (difficulty_df["mean_true_prob"] >= float(args.hard_salvage_mean_true_prob))
        difficulty_df.loc[hard_mask, "difficulty_fine"] = "hard_core"
        difficulty_df.loc[salvage_mask, "difficulty_fine"] = "hard_salvage_teacher"
    difficulty_df.to_csv(output_dir / "curriculum_case_table.csv", index=False)

    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    models = [load_dinov2_model(repo_dir=Path(args.repo_dir), model_name=name, device=device) for name in model_names]
    print(f"Starting gross distill Task7: folds={fold_ids}, teacher_cases={len(gross_teacher_df)}", flush=True)

    teacher_case_ids = set(gross_teacher_df["case_id"].astype(str))
    distill_stage_threshold = {"stage1": 1, "stage2": 2, "stage3": 3}[args.gross_teacher_start_stage]

    for fold_id in fold_ids:
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
        train_df = trim_image_df(train_df, args.max_train_images)
        val_df = trim_image_df(val_df, args.max_val_images)
        test_df = trim_image_df(test_df, args.max_test_images)
        print(f"\n=== GrossDistill Fold {fold_id}/{len(available_folds)} ===", flush=True)

        train_features, train_labels, train_case_ids, train_image_names = extract_concat_features(train_df, model_names, models, args, device)
        val_features, val_labels, val_case_ids, val_image_names = extract_concat_features(val_df, model_names, models, args, device)
        test_features, test_labels, test_case_ids, test_image_names = extract_concat_features(test_df, model_names, models, args, device)

        train_case_df, train_case_features = build_case_feature_df(train_features, train_labels, train_case_ids, train_image_names, args.case_feature_agg)
        val_case_df, val_case_features = build_case_feature_df(val_features, val_labels, val_case_ids, val_image_names, args.case_feature_agg)
        test_case_df, test_case_features = build_case_feature_df(test_features, test_labels, test_case_ids, test_image_names, args.case_feature_agg)

        for df in (train_case_df, val_case_df, test_case_df):
            df[["difficulty", "difficulty_fine"]] = df.merge(
                difficulty_df[["case_id", "difficulty", "difficulty_fine"]], on="case_id", how="left"
            )[["difficulty", "difficulty_fine"]].fillna({"difficulty": "hard", "difficulty_fine": "hard_core"})

        scaler_mean = train_case_features.mean(axis=0, keepdims=True)
        scaler_std = train_case_features.std(axis=0, keepdims=True)
        scaler_std = np.where(scaler_std < 1e-6, 1.0, scaler_std)
        train_x = ((train_case_features - scaler_mean) / scaler_std).astype(np.float32)
        val_x = ((val_case_features - scaler_mean) / scaler_std).astype(np.float32)
        test_x = ((test_case_features - scaler_mean) / scaler_std).astype(np.float32)

        model = StandardizedMLPProbe(
            input_dim=train_x.shape[1],
            hidden_dim=args.mlp_hidden_dim,
            num_classes=len(task.class_names),
            dropout=args.mlp_dropout,
        ).to(device)

        stage3_train_fine_levels = ["easy", "medium", "hard_salvage_teacher"] if args.stage3_hard_mode in {"salvage", "salvage_labeled"} else None
        stage_defs = [
            ("stage1_easy", ["easy"], None, args.stage1_epochs, args.stage1_patience, 1.0),
            ("stage2_easy_medium", ["easy", "medium"], None, args.stage2_epochs, args.stage2_patience, 1.0),
            ("stage3_all", ["easy", "medium", "hard"], stage3_train_fine_levels, args.stage3_epochs, args.stage3_patience, args.hard_weight),
        ]
        stage_summaries = {}
        final_state = None
        final_stage_name = None
        for stage_idx, (stage_name, train_levels, train_fine_levels, epochs, patience, hard_weight) in enumerate(stage_defs, start=1):
            train_mask = train_case_df["difficulty_fine"].isin(train_fine_levels).to_numpy() if train_fine_levels is not None else train_case_df["difficulty"].isin(train_levels).to_numpy()
            val_mask = val_case_df["difficulty_fine"].isin(train_fine_levels).to_numpy() if train_fine_levels is not None else val_case_df["difficulty"].isin(train_levels).to_numpy()
            if args.stage3_hard_mode == "salvage_labeled" and stage_name == "stage3_all":
                train_labeled_hard = train_case_df["case_id"].astype(str).isin(teacher_case_ids) & (train_case_df["difficulty"] == "hard")
                train_mask = train_mask | train_labeled_hard.to_numpy()
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                raise RuntimeError(f"{stage_name}: empty train or val subset.")
            train_weights = np.ones(int(train_mask.sum()), dtype=np.float32)
            if "hard" in train_levels:
                diff = train_case_df.loc[train_mask, "difficulty"].to_numpy()
                train_weights = np.where(diff == "hard", float(hard_weight), 1.0).astype(np.float32)
            stage_teacher_weight = float(args.gross_teacher_weight) if stage_idx >= distill_stage_threshold else 0.0
            stage_teacher_prob = build_teacher_probs(train_case_df.loc[train_mask, "case_id"].astype(str).tolist(), gross_teacher_df)
            state, summary = train_stage_distill(
                stage_name=stage_name,
                model=model,
                train_x=train_x[train_mask],
                train_y=train_case_df.loc[train_mask, "label_idx"].to_numpy(dtype=np.int64),
                train_teacher_prob=stage_teacher_prob,
                train_weights=train_weights,
                val_x=val_x[val_mask],
                val_y=val_case_df.loc[val_mask, "label_idx"].to_numpy(dtype=np.int64),
                val_case_ids=val_case_df.loc[val_mask, "case_id"].astype(str).tolist(),
                val_image_names=val_case_df.loc[val_mask, "image_name"].astype(str).tolist(),
                task=task,
                selection_metric="accuracy" if stage_idx == 1 else args.selection_metric,
                gross_teacher_weight=stage_teacher_weight,
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

        assert final_state is not None and final_stage_name is not None
        model.load_state_dict(final_state)
        with torch.no_grad():
            val_probs = F.softmax(model(torch.from_numpy(val_x).to(device)), dim=1).cpu().numpy().astype(np.float64)
            test_probs = F.softmax(model(torch.from_numpy(test_x).to(device)), dim=1).cpu().numpy().astype(np.float64)

        val_predictions = build_prediction_df(val_probs, val_case_df["label_idx"].to_numpy(dtype=np.int64), val_case_df["case_id"].astype(str).tolist(), val_case_df["image_name"].astype(str).tolist(), task.class_names)
        test_predictions = build_prediction_df(test_probs, test_case_df["label_idx"].to_numpy(dtype=np.int64), test_case_df["case_id"].astype(str).tolist(), test_case_df["image_name"].astype(str).tolist(), task.class_names)
        val_case_metrics = save_split_outputs(fold_dir, "val", val_predictions, task, ["mean"])
        test_case_metrics = save_split_outputs(fold_dir, "test", test_predictions, task, ["mean"])
        fold_summary = {
            "best_c": float("nan"),
            "best_val_primary_metric": float(stage_summaries[final_stage_name]["best_val_primary_metric"]),
            "final_stage": final_stage_name,
            "stage_summaries": stage_summaries,
            "val_case_mean": val_case_metrics["mean"],
            "test_case_mean": test_case_metrics["mean"],
            "gross_teacher_weight": float(args.gross_teacher_weight),
            "gross_teacher_statuses": sorted(gross_statuses),
        }
        write_json(fold_dir / "fold_summary.json", fold_summary)
        print(
            f"[Fold {fold_id}] test_acc={format_metric_value(test_case_metrics['mean'].get('accuracy'))} "
            f"test_bacc={format_metric_value(test_case_metrics['mean'].get('balanced_accuracy'))}",
            flush=True,
        )

    if len(fold_ids) > 1:
        aggregate_run_outputs(output_dir, task, fold_ids, ["mean"])
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
