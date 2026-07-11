from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS, TaskConfig, get_task
from thymic_baseline.data import ThymicImageDataset
from thymic_baseline.metrics import summarize_prediction_frame
from thymic_baseline.registry import filter_registry_for_task, load_registry, load_split_assignments, merge_registry_with_splits, subset_by_fold
from thymic_baseline.train import (
    build_class_weights,
    build_dataloader,
    evaluate_model,
    format_metric_value,
    is_better_metric,
    metric_summary_text,
    metrics_to_frame,
    resolve_device,
    resolve_selection_metric,
    save_evaluation_outputs,
    write_json,
)


DEFAULT_BASE = PROJECT_ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_adaptation_runs"
DEFAULT_INPUT = DEFAULT_BASE / "45_old_third_all_balanced_finetune_inputs_20260523"
SUBTYPE_CLASS_NAMES = ("A", "AB", "B1", "B2", "B3", "TC")
SUBTYPE_TO_IDX = {name: idx for idx, name in enumerate(SUBTYPE_CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv3/timm on old+third Task7 with a six-subtype auxiliary head."
    )
    parser.add_argument("--registry-csv", default=str(DEFAULT_INPUT / "registry.csv"))
    parser.add_argument("--split-csv", default=str(DEFAULT_INPUT / "split.csv"))
    parser.add_argument("--task", default="task7_lowrisk_vs_highrisk_tc")
    parser.add_argument("--input-variant", default="whole_plus_crop", choices=INPUT_VARIANTS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--global-pool", default="token", choices=("token", "avg"))
    parser.add_argument("--fold", default="all")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--backbone-lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--aux-loss-weight", type=float, default=0.25)
    parser.add_argument("--aux-label-smoothing", type=float, default=0.03)
    parser.add_argument("--sample-loss-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-risk-loss-weight", type=float, default=1.35)
    parser.add_argument("--subtype-a-loss-weight", type=float, default=1.10)
    parser.add_argument("--subtype-ab-loss-weight", type=float, default=1.20)
    parser.add_argument("--subtype-b1-loss-weight", type=float, default=1.60)
    parser.add_argument("--subtype-b2-loss-weight", type=float, default=1.00)
    parser.add_argument("--subtype-b3-loss-weight", type=float, default=1.00)
    parser.add_argument("--subtype-tc-loss-weight", type=float, default=1.00)
    parser.add_argument("--selection-metric", default="balanced_accuracy")
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-scope", default="last_block", choices=("head_only", "last_block", "last2_blocks", "full"))
    parser.add_argument("--head-type", default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_image_df(registry_csv: str, split_csv: str, task: TaskConfig) -> pd.DataFrame:
    registry = load_registry(registry_csv)
    split_df = load_split_assignments(split_csv)
    merged = merge_registry_with_splits(registry, split_df)
    filtered = filter_registry_for_task(merged, task)
    class_to_idx = {name: idx for idx, name in enumerate(task.class_names)}
    rows: list[dict[str, Any]] = []
    for row in filtered.to_dict(orient="records"):
        image_path = str(row.get("training_image_path", "")).strip()
        if not image_path:
            image_path = str(row.get("selected_original_image_path", "")).strip()
        if not image_path:
            raise ValueError(f"Missing training image path for case {row['case_id']}")
        image_name = str(row.get("selected_original_image_name", "")).strip() or Path(image_path).name
        item = {
            "case_id": str(row["case_id"]),
            "patient_id": str(row.get("patient_id", row["case_id"])),
            "original_case_id": str(row.get("original_case_id", row["case_id"])),
            "label_name": str(row[task.label_column]),
            "label_idx": int(class_to_idx[str(row[task.label_column])]),
            "subtype_label": str(row.get("task_l6_label", "")).strip(),
            "image_name": image_name,
            "image_path": image_path,
            "master_fold_id": int(row["master_fold_id"]),
            "who_type_raw": str(row.get("who_type_raw", "")),
            "source_dataset": str(row.get("source_dataset", "")),
            "source_folder": str(row.get("source_folder", row.get("source_case_folder", ""))),
            "task_l6_label": str(row.get("task_l6_label", "")),
            "task_l7_label": str(row.get("task_l7_label", "")),
        }
        for key, value in row.items():
            key_text = str(key)
            if key_text.startswith("v2_") or key_text in {
                "domain",
                "third_split",
                "view_type_final",
                "quality_group",
                "multi_image_group",
                "brightness_mean_quartile",
                "saturation_mean_quartile",
                "subject_area_proxy_quartile",
                "red_tissue_ratio_quartile",
            }:
                item[key_text] = value
        rows.append(item)
    image_df = pd.DataFrame(rows)
    unknown_subtypes = sorted(set(image_df["subtype_label"]) - set(SUBTYPE_CLASS_NAMES))
    if unknown_subtypes:
        raise ValueError(f"Unsupported subtype labels for auxiliary head: {unknown_subtypes}")
    image_df["subtype_idx"] = image_df["subtype_label"].map(SUBTYPE_TO_IDX).astype(int)
    missing_paths = [p for p in image_df["image_path"].tolist() if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing image files: {missing_paths[:10]}")
    return image_df.reset_index(drop=True)


def load_available_folds(split_csv: str) -> list[int]:
    split_df = load_split_assignments(split_csv)
    return sorted(int(item) for item in split_df["master_fold_id"].dropna().unique().tolist())


def prepare_fold_data(registry_csv: str, split_csv: str, task: TaskConfig, fold_id: int, num_folds: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_df = make_image_df(registry_csv=registry_csv, split_csv=split_csv, task=task)
    return subset_by_fold(image_df, test_fold=fold_id, num_folds=num_folds)


def attach_sample_loss_weights(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    weighted = frame.copy()
    weights = np.ones(len(weighted), dtype=np.float32)
    if bool(args.sample_loss_weighting):
        subtype_weights = {
            "A": float(args.subtype_a_loss_weight),
            "AB": float(args.subtype_ab_loss_weight),
            "B1": float(args.subtype_b1_loss_weight),
            "B2": float(args.subtype_b2_loss_weight),
            "B3": float(args.subtype_b3_loss_weight),
            "TC": float(args.subtype_tc_loss_weight),
        }
        low_mask = weighted["label_idx"].astype(int).to_numpy() == 0
        weights[low_mask] *= float(args.low_risk_loss_weight)
        weights *= weighted["subtype_label"].map(subtype_weights).astype(float).to_numpy(dtype=np.float32)
    weighted["sample_loss_weight"] = weights
    return weighted


class AuxThymicImageDataset(ThymicImageDataset):
    def __getitem__(self, index: int):
        inputs, label, case_id, image_name = super().__getitem__(index)
        row = self.image_df.iloc[index]
        subtype = torch.tensor(int(row["subtype_idx"]), dtype=torch.long)
        sample_weight = torch.tensor(float(row.get("sample_loss_weight", 1.0)), dtype=torch.float32)
        return inputs, label, subtype, sample_weight, case_id, image_name


def build_aux_dataloader(
    image_df: pd.DataFrame,
    input_variant: str,
    image_size: int,
    is_train: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = AuxThymicImageDataset(
        image_df=image_df,
        input_variant=input_variant,
        image_size=image_size,
        is_train=is_train,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": is_train,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def build_subtype_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["subtype_idx"].value_counts().sort_index()
    weights = []
    total = float(len(train_df))
    for idx in range(len(SUBTYPE_CLASS_NAMES)):
        count = float(counts.get(idx, 0))
        weights.append(0.0 if count <= 0 else total / (len(SUBTYPE_CLASS_NAMES) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


class DINOv3MultitaskSubtypeModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        global_pool: str,
        input_variant: str,
        num_classes: int,
        dropout: float,
        head_type: str,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.input_variant = input_variant
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool=global_pool)
        feature_dim = int(getattr(self.encoder, "num_features", 0))
        if feature_dim <= 0:
            raise ValueError(f"Backbone {model_name} does not expose num_features.")
        head_dim = feature_dim * 2 if input_variant == "whole_plus_crop" else feature_dim
        self.head = self._make_head(head_dim, num_classes, dropout, head_type, hidden_dim)
        self.aux_head = self._make_head(head_dim, len(SUBTYPE_CLASS_NAMES), dropout, head_type, hidden_dim)

    @staticmethod
    def _make_head(head_dim: int, num_classes: int, dropout: float, head_type: str, hidden_dim: int) -> nn.Sequential:
        if head_type == "linear":
            return nn.Sequential(
                nn.LayerNorm(head_dim),
                nn.Dropout(dropout),
                nn.Linear(head_dim, num_classes),
            )
        return nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def extract_features(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            features = self.encode(inputs)
        elif isinstance(inputs, (tuple, list)):
            whole, crop = inputs
            features = torch.cat([self.encode(whole), self.encode(crop)], dim=1)
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)!r}")
        return features

    def forward(self, inputs: Any, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(inputs)
        logits = self.head(features)
        if return_aux:
            return logits, self.aux_head(features)
        return logits

    def binary_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.state_dict().items() if not key.startswith("aux_head.")}


def selected_block_prefixes(encoder: nn.Module, tune_scope: str) -> set[str]:
    if tune_scope == "head_only":
        return set()

    vit_block_ids: set[int] = set()
    vit_pattern = re.compile(r"^blocks\.(\d+)\.")
    convnext_block_ids: set[tuple[int, int]] = set()
    convnext_pattern = re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.")
    for name, _ in encoder.named_parameters():
        vit_match = vit_pattern.match(name)
        if vit_match:
            vit_block_ids.add(int(vit_match.group(1)))
            continue
        convnext_match = convnext_pattern.match(name)
        if convnext_match:
            convnext_block_ids.add((int(convnext_match.group(1)), int(convnext_match.group(2))))

    if vit_block_ids:
        ordered = sorted(vit_block_ids)
        selected = ordered[-1:] if tune_scope == "last_block" else ordered[-2:]
        if tune_scope == "full":
            selected = ordered
        return {f"blocks.{idx}." for idx in selected}

    if convnext_block_ids:
        ordered = sorted(convnext_block_ids)
        selected = ordered[-1:] if tune_scope == "last_block" else ordered[-2:]
        if tune_scope == "full":
            selected = ordered
        return {f"stages.{stage_idx}.blocks.{block_idx}." for stage_idx, block_idx in selected}

    if tune_scope == "full":
        return set()
    return set()


def configure_trainable_parameters(model: DINOv3MultitaskSubtypeModel, tune_scope: str) -> dict[str, int]:
    for param in model.encoder.parameters():
        param.requires_grad = False

    chosen_prefixes = selected_block_prefixes(model.encoder, tune_scope)
    if tune_scope == "full":
        for param in model.encoder.parameters():
            param.requires_grad = True
    elif chosen_prefixes:
        for name, param in model.encoder.named_parameters():
            in_selected_block = any(name.startswith(prefix) for prefix in chosen_prefixes)
            is_final_norm = name.startswith("norm.") or name.startswith("fc_norm.") or name.startswith("head.norm.")
            if in_selected_block or is_final_norm:
                param.requires_grad = True
    elif tune_scope != "head_only":
        raise ValueError(f"Could not locate blocks for tune_scope={tune_scope}")

    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.aux_head.parameters():
        param.requires_grad = True

    return {
        "trainable_backbone": int(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)),
        "trainable_head": int(sum(p.numel() for p in model.head.parameters() if p.requires_grad)),
        "trainable_aux_head": int(sum(p.numel() for p in model.aux_head.parameters() if p.requires_grad)),
        "total": int(sum(p.numel() for p in model.parameters())),
    }


def build_optimizer(model: DINOv3MultitaskSubtypeModel, head_lr: float, backbone_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params = [p for module in (model.head, model.aux_head) for p in module.parameters() if p.requires_grad]
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, tuple):
        return tuple(move_to_device(item, device) for item in inputs)
    if isinstance(inputs, list):
        return [move_to_device(item, device) for item in inputs]
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def train_one_epoch_amp(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    primary_criterion: nn.Module,
    aux_criterion: nn.Module,
    aux_loss_weight: float,
    device: torch.device,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    max_batches: int | None,
    progress_desc: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(loader, total=total_batches, desc=progress_desc, leave=False, dynamic_ncols=True, file=sys.stdout)
    for batch_idx, (inputs, labels, subtype_labels, sample_weights, _, _) in enumerate(progress, start=1):
        inputs = move_to_device(inputs, device)
        labels = labels.to(device, non_blocking=True)
        subtype_labels = subtype_labels.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits, subtype_logits = model(inputs, return_aux=True)
            primary_loss = primary_criterion(logits, labels)
            primary_loss = (primary_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            aux_loss = aux_criterion(subtype_logits, subtype_labels)
            loss = primary_loss + float(aux_loss_weight) * aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{(total_loss / max(total_items, 1)):.4f}", main=f"{float(primary_loss.item()):.4f}", aux=f"{float(aux_loss.item()):.4f}")
        if max_batches is not None and batch_idx >= max_batches:
            break
    progress.close()
    return total_loss / max(total_items, 1)


def run_single_fold(args: argparse.Namespace, task: TaskConfig, fold_id: int, output_dir: Path) -> dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = load_available_folds(args.split_csv)
    train_df, val_df, test_df = prepare_fold_data(args.registry_csv, args.split_csv, task, fold_id, len(fold_ids))
    train_df = attach_sample_loss_weights(train_df, args)

    train_loader = build_aux_dataloader(train_df, args.input_variant, args.image_size, True, args.batch_size, args.num_workers)
    val_loader = build_dataloader(val_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers)
    test_loader = build_dataloader(test_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers)

    device = resolve_device(args.device)
    model = DINOv3MultitaskSubtypeModel(
        model_name=args.model_name,
        global_pool=args.global_pool,
        input_variant=args.input_variant,
        num_classes=task.num_classes,
        dropout=args.dropout,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
    )
    trainable = configure_trainable_parameters(model, args.tune_scope)
    model.to(device)

    class_weights = build_class_weights(train_df, task.num_classes, device=device) if args.class_weighting else None
    subtype_class_weights = build_subtype_class_weights(train_df, device=device) if args.class_weighting else None
    train_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing, reduction="none")
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    aux_criterion = nn.CrossEntropyLoss(weight=subtype_class_weights, label_smoothing=args.aux_label_smoothing)
    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    amp_enabled = bool(args.amp and device.type == "cuda")

    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
    selection_metric_name = resolve_selection_metric(task, args.selection_metric)
    best_selection_metric: float | None = None
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []

    run_config = {
        "task": task.key,
        "task_class_names": list(task.class_names),
        "input_variant": args.input_variant,
        "model_name": args.model_name,
        "global_pool": args.global_pool,
        "tune_scope": args.tune_scope,
        "head_type": args.head_type,
        "head_lr": float(args.head_lr),
        "backbone_lr": float(args.backbone_lr),
        "aux_loss_weight": float(args.aux_loss_weight),
        "sample_loss_weighting": bool(args.sample_loss_weighting),
        "low_risk_loss_weight": float(args.low_risk_loss_weight),
        "subtype_loss_weights": {
            "A": float(args.subtype_a_loss_weight),
            "AB": float(args.subtype_ab_loss_weight),
            "B1": float(args.subtype_b1_loss_weight),
            "B2": float(args.subtype_b2_loss_weight),
            "B3": float(args.subtype_b3_loss_weight),
            "TC": float(args.subtype_tc_loss_weight),
        },
        "train_sample_loss_weight_mean": float(train_df["sample_loss_weight"].mean()),
        "train_sample_loss_weight_max": float(train_df["sample_loss_weight"].max()),
        "fold_id": int(fold_id),
        "device": str(device),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "train_cases": int(train_df["case_id"].nunique()),
        "val_cases": int(val_df["case_id"].nunique()),
        "test_cases": int(test_df["case_id"].nunique()),
        "trainable": trainable,
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== DINOv3 subtype-aux FT Fold {fold_id}/{len(fold_ids)} | task={task.key} | variant={args.input_variant} | "
        f"model={args.model_name} | pool={args.global_pool} | tune={args.tune_scope} | trainable={trainable} ===",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Fold {fold_id}] Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_one_epoch_amp(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            primary_criterion=train_criterion,
            aux_criterion=aux_criterion,
            aux_loss_weight=float(args.aux_loss_weight),
            device=device,
            grad_clip=args.grad_clip,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=args.max_train_batches,
            progress_desc=f"dinov3 fold{fold_id} train e{epoch}",
        )
        val_results = evaluate_model(
            model=model,
            loader=val_loader,
            task=task,
            device=device,
            aggregate_methods=aggregate_methods,
            criterion=eval_criterion,
            max_batches=args.max_eval_batches,
            progress_desc=f"dinov3 fold{fold_id} val e{epoch}",
        )
        scheduler.step()
        val_primary = float(val_results["case_metrics"]["mean"].get(task.primary_metric, float("nan")))
        val_selection = float(val_results["case_metrics"]["mean"].get(selection_metric_name, float("nan")))
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_results["loss"]),
                "val_primary_metric": val_primary,
                "val_selection_metric": val_selection,
                "head_lr": float(optimizer.param_groups[0]["lr"]),
                "backbone_lr": float(optimizer.param_groups[-1]["lr"]),
            }
        )
        if is_better_metric(val_selection, best_selection_metric):
            best_selection_metric = val_selection
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), fold_dir / "best_multitask_model.pt")
            torch.save(model.binary_state_dict(), fold_dir / "best_model.pt")
            best_flag = "yes"
        else:
            stale_epochs += 1
            best_flag = "no"
        print(
            f"[Fold {fold_id}] train_loss={train_loss:.4f} | val_loss={val_results['loss']:.4f} | "
            f"val_{selection_metric_name}={format_metric_value(val_selection)} | "
            f"val_{task.primary_metric}={format_metric_value(val_primary)} | "
            f"val_acc={format_metric_value(val_results['case_metrics']['mean'].get('accuracy'))} | "
            f"best_updated={best_flag} | stale_epochs={stale_epochs}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch} (best_epoch={best_epoch}).", flush=True)
            break

    pd.DataFrame(history_rows).to_csv(fold_dir / "history.csv", index=False)
    if best_epoch < 1:
        raise RuntimeError("Fine-tuning finished without a valid checkpoint.")
    try:
        best_state = torch.load(fold_dir / "best_multitask_model.pt", map_location=device, weights_only=True)
    except TypeError:
        best_state = torch.load(fold_dir / "best_multitask_model.pt", map_location=device)
    model.load_state_dict(best_state)

    val_results = evaluate_model(model, val_loader, task, device, aggregate_methods, eval_criterion, args.max_eval_batches, f"dinov3 fold{fold_id} val best")
    test_results = evaluate_model(model, test_loader, task, device, aggregate_methods, eval_criterion, args.max_eval_batches, f"dinov3 fold{fold_id} test best")
    save_evaluation_outputs(fold_dir, "val", val_results, aggregate_methods)
    save_evaluation_outputs(fold_dir, "test", test_results, aggregate_methods)

    summary = {
        "fold_id": int(fold_id),
        "best_epoch": int(best_epoch),
        "selection_metric": selection_metric_name,
        "best_val_selection_metric": best_selection_metric,
        "val_case_mean": val_results["case_metrics"]["mean"],
        "test_case_mean": test_results["case_metrics"]["mean"],
        "test_image": test_results["image_metrics"],
    }
    write_json(fold_dir / "fold_summary.json", summary)
    print(
        f"[Fold {fold_id}] best_epoch={best_epoch} | "
        f"val_case_mean: {metric_summary_text(val_results['case_metrics']['mean'], [selection_metric_name, task.primary_metric, 'accuracy', 'balanced_accuracy'])} | "
        f"test_case_mean: {metric_summary_text(test_results['case_metrics']['mean'], [task.primary_metric, 'accuracy', 'balanced_accuracy'])}",
        flush=True,
    )
    return {
        "summary_row": {
            "fold_id": int(fold_id),
            "best_epoch": int(best_epoch),
            "best_val_selection_metric": float(best_selection_metric) if best_selection_metric is not None else float("nan"),
            "test_case_mean_primary_metric": float(test_results["case_metrics"]["mean"].get(task.primary_metric, float("nan"))),
            "test_case_mean_accuracy": float(test_results["case_metrics"]["mean"].get("accuracy", float("nan"))),
            "test_case_mean_balanced_accuracy": float(test_results["case_metrics"]["mean"].get("balanced_accuracy", float("nan"))),
        },
        "test_image_predictions": test_results["image_predictions"],
        "test_case_predictions": test_results["case_predictions"],
    }


def add_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "master_fold_id",
    ]
    available = [c for c in keep_cols if c in metadata.columns]
    return predictions.merge(metadata[available].drop_duplicates("case_id"), on="case_id", how="left")


def domain_metrics(predictions: pd.DataFrame, task: TaskConfig) -> pd.DataFrame:
    frames = [metrics_to_frame("test_oof", "case", "mean", summarize_prediction_frame(predictions, task.class_names))]
    for domain, group in predictions.groupby("source_dataset", dropna=False):
        frames.append(metrics_to_frame(f"test_oof:{domain}", "case", "mean", summarize_prediction_frame(group, task.class_names)))
    old_mask = ~predictions["source_dataset"].astype(str).str.startswith("third_batch", na=False)
    for name, mask in [("old", old_mask), ("third", ~old_mask)]:
        group = predictions[mask].copy()
        if not group.empty:
            frames.append(metrics_to_frame(f"test_oof:{name}", "case", "mean", summarize_prediction_frame(group, task.class_names)))
    return pd.concat(frames, ignore_index=True)


def run_all_folds(args: argparse.Namespace, task: TaskConfig, output_dir: Path) -> None:
    fold_ids = load_available_folds(args.split_csv)
    metadata = make_image_df(args.registry_csv, args.split_csv, task)
    fold_summaries: list[dict[str, Any]] = []
    all_test_images: list[pd.DataFrame] = []
    all_test_cases: dict[str, list[pd.DataFrame]] = {
        item.strip(): [] for item in args.aggregate_methods.split(",") if item.strip()
    }
    print(f"Running DINOv3 fine-tune folds={fold_ids} task={task.key}", flush=True)
    for fold_idx, fold_id in enumerate(fold_ids, start=1):
        print(f"\n##### CV Progress {fold_idx}/{len(fold_ids)}: fold {fold_id} #####", flush=True)
        fold_result = run_single_fold(args, task, fold_id, output_dir)
        fold_summaries.append(fold_result["summary_row"])
        image_df = fold_result["test_image_predictions"].copy()
        image_df["fold_id"] = int(fold_id)
        all_test_images.append(image_df)
        for method, case_df in fold_result["test_case_predictions"].items():
            fold_case_df = case_df.copy()
            fold_case_df["fold_id"] = int(fold_id)
            all_test_cases[method].append(fold_case_df)

    pd.DataFrame(fold_summaries).to_csv(output_dir / "cv_fold_summary.csv", index=False)
    image_predictions_df = add_metadata(pd.concat(all_test_images, ignore_index=True), metadata)
    image_predictions_df.to_csv(output_dir / "oof_image_predictions.csv", index=False, encoding="utf-8-sig")

    metrics_frames = [metrics_to_frame("test_oof", "image", "none", summarize_prediction_frame(image_predictions_df, task.class_names))]
    for method, frames in all_test_cases.items():
        case_predictions_df = add_metadata(pd.concat(frames, ignore_index=True), metadata)
        case_predictions_df.to_csv(output_dir / f"oof_case_predictions_{method}.csv", index=False, encoding="utf-8-sig")
        metrics_frames.append(metrics_to_frame("test_oof", "case", method, summarize_prediction_frame(case_predictions_df, task.class_names)))
        if method == "mean":
            domain_metrics(case_predictions_df, task).to_csv(output_dir / "oof_domain_metrics_mean.csv", index=False, encoding="utf-8-sig")
    pd.concat(metrics_frames, ignore_index=True).to_csv(output_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[done] outputs saved to {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))
    if args.fold == "all":
        run_all_folds(args, task, output_dir)
        return
    fold_id = int(args.fold)
    run_single_fold(args, task, fold_id, output_dir)


if __name__ == "__main__":
    main()
