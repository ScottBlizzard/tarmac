from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thymic_baseline.config import DEFAULT_IMAGE_SIZE, DEFAULT_RANDOM_SEED, TaskConfig, get_task
from thymic_baseline.train import (
    build_class_weights,
    build_dataloader,
    evaluate_model,
    format_metric_value,
    is_better_metric,
    load_available_folds,
    metric_summary_text,
    prepare_fold_data,
    resolve_device,
    resolve_selection_metric,
    save_evaluation_outputs,
    train_one_epoch,
    write_json,
)


SUPPORTED_VARIANTS = ("whole", "crop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-backbone DINOv2 fine-tuning on thymic tasks.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--input-variant", default="whole", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", default="dinov2_vits14,dinov2_vitb14")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="primary")
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-scope", default="last_block", choices=("head_only", "last_block", "last2_blocks", "full"))
    parser.add_argument("--head-type", default="linear", choices=("linear", "mlp"))
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DINOv2DualFineTuneModel(nn.Module):
    def __init__(
        self,
        repo_dir: Path,
        model_names: list[str],
        num_classes: int,
        dropout: float,
        head_type: str,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(model_names) != 2:
            raise ValueError("Dual DINO fine-tune requires exactly two model names.")
        self.model_names = list(model_names)
        self.backbone_a = torch.hub.load(str(repo_dir), model_names[0], source="local", pretrained=True)
        self.backbone_b = torch.hub.load(str(repo_dir), model_names[1], source="local", pretrained=True)
        dim_a = int(getattr(self.backbone_a, "embed_dim", 384))
        dim_b = int(getattr(self.backbone_b, "embed_dim", 768))
        feature_dim = dim_a + dim_b
        if head_type == "linear":
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs_a = self.backbone_a.forward_features(inputs)
        outputs_b = self.backbone_b.forward_features(inputs)
        cls_a = outputs_a["x_norm_clstoken"]
        cls_b = outputs_b["x_norm_clstoken"]
        fused = torch.cat([cls_a, cls_b], dim=1)
        return self.head(fused)


def load_init_checkpoint(model: DINOv2DualFineTuneModel, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint}")
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded init checkpoint: {checkpoint} | "
        f"missing_keys={len(missing)} | unexpected_keys={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"  missing sample: {missing[:5]}", flush=True)
    if unexpected:
        print(f"  unexpected sample: {unexpected[:5]}", flush=True)


def _configure_single_backbone(backbone: nn.Module, tune_scope: str) -> None:
    for param in backbone.parameters():
        param.requires_grad = False
    if tune_scope == "head_only":
        return
    if tune_scope == "last_block":
        for name, param in backbone.named_parameters():
            if name.startswith("blocks.11") or name.startswith("norm"):
                param.requires_grad = True
        return
    if tune_scope == "last2_blocks":
        for name, param in backbone.named_parameters():
            if name.startswith("blocks.10") or name.startswith("blocks.11") or name.startswith("norm"):
                param.requires_grad = True
        return
    if tune_scope == "full":
        for param in backbone.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unsupported tune scope: {tune_scope}")


def configure_trainable_parameters(model: DINOv2DualFineTuneModel, tune_scope: str) -> None:
    _configure_single_backbone(model.backbone_a, tune_scope)
    _configure_single_backbone(model.backbone_b, tune_scope)
    for param in model.head.parameters():
        param.requires_grad = True


def build_optimizer(
    model: DINOv2DualFineTuneModel,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    backbone_params = [
        param
        for param in list(model.backbone_a.parameters()) + list(model.backbone_b.parameters())
        if param.requires_grad
    ]
    param_groups: list[dict[str, Any]] = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def run_single_fold(args: argparse.Namespace, task: TaskConfig, fold_id: int, root_output_dir: Path) -> dict[str, Any]:
    fold_dir = root_output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    available_folds = load_available_folds(args.split_csv)
    train_df, val_df, test_df = prepare_fold_data(
        registry_csv=args.registry_csv,
        split_csv=args.split_csv,
        images_root=args.images_root,
        task=task,
        fold_id=fold_id,
        num_folds=len(available_folds),
    )

    train_loader = build_dataloader(train_df, args.input_variant, args.image_size, True, args.batch_size, args.num_workers)
    val_loader = build_dataloader(val_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers)
    test_loader = build_dataloader(test_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers)

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    device = resolve_device(args.device)
    model = DINOv2DualFineTuneModel(
        repo_dir=Path(args.repo_dir),
        model_names=model_names,
        num_classes=task.num_classes,
        dropout=args.dropout,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
    )
    load_init_checkpoint(model, args.init_checkpoint)
    configure_trainable_parameters(model, args.tune_scope)
    model = model.to(device)

    class_weights = build_class_weights(train_df, task.num_classes, device=device) if args.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
    selection_metric_name = resolve_selection_metric(task, args.selection_metric)
    history_rows: list[dict[str, Any]] = []
    best_selection_metric: float | None = None
    best_epoch = 0
    stale_epochs = 0

    run_config = {
        "task": task.key,
        "input_variant": args.input_variant,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "model_names": model_names,
        "init_checkpoint": args.init_checkpoint,
        "tune_scope": args.tune_scope,
        "head_type": args.head_type,
        "head_lr": float(args.head_lr),
        "backbone_lr": float(args.backbone_lr),
        "fold_id": fold_id,
        "device": str(device),
        "train_cases": int(train_df["case_id"].nunique()),
        "val_cases": int(val_df["case_id"].nunique()),
        "test_cases": int(test_df["case_id"].nunique()),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "test_images": int(len(test_df)),
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== DINOv2 Dual FT Fold {fold_id}/{len(available_folds)} | task={task.key} | variant={args.input_variant} | "
        f"models={model_names} | tune_scope={args.tune_scope} | head={args.head_type} ===",
        flush=True,
    )
    print(
        f"train_cases={run_config['train_cases']}, val_cases={run_config['val_cases']}, "
        f"test_cases={run_config['test_cases']}, train_images={run_config['train_images']}, "
        f"val_images={run_config['val_images']}, test_images={run_config['test_images']}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Fold {fold_id}] Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
            progress_desc=f"dinodual fold{fold_id} train e{epoch}",
        )
        val_results = evaluate_model(
            model=model,
            loader=val_loader,
            task=task,
            device=device,
            aggregate_methods=aggregate_methods,
            criterion=criterion,
            max_batches=args.max_eval_batches,
            progress_desc=f"dinodual fold{fold_id} val e{epoch}",
        )
        scheduler.step()

        val_primary = float(val_results["case_metrics"]["mean"].get(task.primary_metric, float("nan")))
        val_selection = float(val_results["case_metrics"]["mean"].get(selection_metric_name, float("nan")))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_results["loss"],
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
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
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
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    try:
        best_state = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        best_state = torch.load(fold_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state)

    val_results = evaluate_model(model, val_loader, task, device, aggregate_methods, criterion, args.max_eval_batches, f"dinodual fold{fold_id} val best")
    test_results = evaluate_model(model, test_loader, task, device, aggregate_methods, criterion, args.max_eval_batches, f"dinodual fold{fold_id} test best")
    save_evaluation_outputs(fold_dir, "val", val_results, aggregate_methods)
    save_evaluation_outputs(fold_dir, "test", test_results, aggregate_methods)

    summary = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
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
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    print(
        f"Starting DINOv2 dual fine-tune: task={task.key}, variant={args.input_variant}, "
        f"models={args.model_names}, tune_scope={args.tune_scope}, head={args.head_type}, folds={fold_ids}",
        flush=True,
    )
    for fold_id in fold_ids:
        run_single_fold(args, task, fold_id, output_dir)


if __name__ == "__main__":
    main()
