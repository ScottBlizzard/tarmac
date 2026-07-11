from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_finetune_old_third_20260523 import (  # noqa: E402
    DEFAULT_INPUT,
    DINOv3FineTuneModel,
    build_optimizer,
    configure_trainable_parameters,
    load_available_folds,
    make_image_df,
    prepare_fold_data,
    selected_block_prefixes,
)
from thymic_baseline.config import INPUT_VARIANTS, TaskConfig, get_task  # noqa: E402
from thymic_baseline.cropping import extract_specimen_crop  # noqa: E402
from thymic_baseline.data import build_transform  # noqa: E402
from thymic_baseline.metrics import summarize_prediction_frame  # noqa: E402
from thymic_baseline.train import (  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 DINOv3 fine-tuning with supervised two-view domain-consistency regularization."
    )
    parser.add_argument("--registry-csv", default=str(DEFAULT_INPUT / "registry.csv"))
    parser.add_argument("--split-csv", default=str(DEFAULT_INPUT / "split.csv"))
    parser.add_argument("--task", default="task7_lowrisk_vs_highrisk_tc")
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="vit_large_patch16_dinov3_qkvb.lvd1689m")
    parser.add_argument("--global-pool", default="token", choices=("token", "avg"))
    parser.add_argument("--fold", default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--selection-metric", default="balanced_accuracy")
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-scope", default="last_block", choices=("head_only", "last_block", "last2_blocks", "full"))
    parser.add_argument("--head-type", default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--view-a-profile", default="style_light")
    parser.add_argument("--view-b-profile", default="domain_robust")
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--consistency-temperature", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def transform_for_profile(image_size: int, profile: str):
    previous = os.environ.get("THYMIC_AUG_PROFILE")
    os.environ["THYMIC_AUG_PROFILE"] = profile
    try:
        return build_transform(image_size=image_size, is_train=True)
    finally:
        if previous is None:
            os.environ.pop("THYMIC_AUG_PROFILE", None)
        else:
            os.environ["THYMIC_AUG_PROFILE"] = previous


class TwoViewThymicDataset(Dataset):
    def __init__(
        self,
        image_df: pd.DataFrame,
        input_variant: str,
        image_size: int,
        view_a_profile: str,
        view_b_profile: str,
    ) -> None:
        self.image_df = image_df.reset_index(drop=True)
        self.input_variant = input_variant
        self.transform_a = transform_for_profile(image_size, view_a_profile)
        self.transform_b = transform_for_profile(image_size, view_b_profile)

    def __len__(self) -> int:
        return len(self.image_df)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _make_inputs(self, image: Image.Image, transform) -> Any:
        if self.input_variant == "whole":
            return transform(image)
        if self.input_variant == "crop":
            return transform(extract_specimen_crop(image))
        if self.input_variant == "whole_plus_crop":
            return (transform(image), transform(extract_specimen_crop(image)))
        raise ValueError(f"Unsupported input variant: {self.input_variant}")

    def __getitem__(self, index: int):
        row = self.image_df.iloc[index]
        image = self._load_image(str(row["image_path"]))
        inputs_a = self._make_inputs(image, self.transform_a)
        inputs_b = self._make_inputs(image, self.transform_b)
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return inputs_a, inputs_b, label, str(row["case_id"]), str(row["image_name"])


def build_two_view_dataloader(
    image_df: pd.DataFrame,
    input_variant: str,
    image_size: int,
    view_a_profile: str,
    view_b_profile: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = TwoViewThymicDataset(
        image_df=image_df,
        input_variant=input_variant,
        image_size=image_size,
        view_a_profile=view_a_profile,
        view_b_profile=view_b_profile,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, tuple):
        return tuple(move_to_device(item, device) for item in inputs)
    if isinstance(inputs, list):
        return [move_to_device(item, device) for item in inputs]
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def symmetric_kl_consistency(logits_a: torch.Tensor, logits_b: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    log_pa = F.log_softmax(logits_a / temp, dim=1)
    log_pb = F.log_softmax(logits_b / temp, dim=1)
    pa = log_pa.exp().detach()
    pb = log_pb.exp().detach()
    loss_ab = F.kl_div(log_pa, pb, reduction="batchmean")
    loss_ba = F.kl_div(log_pb, pa, reduction="batchmean")
    return 0.5 * (loss_ab + loss_ba) * (temp * temp)


def train_one_epoch_consistency(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    consistency_weight: float,
    consistency_temperature: float,
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
    for batch_idx, (inputs_a, inputs_b, labels, _, _) in enumerate(progress, start=1):
        inputs_a = move_to_device(inputs_a, device)
        inputs_b = move_to_device(inputs_b, device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits_a = model(inputs_a)
            logits_b = model(inputs_b)
            supervised = 0.5 * (criterion(logits_a, labels) + criterion(logits_b, labels))
            consistency = symmetric_kl_consistency(logits_a, logits_b, consistency_temperature)
            loss = supervised + float(consistency_weight) * consistency
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        progress.set_postfix(
            loss=f"{(total_loss / max(total_items, 1)):.4f}",
            ce=f"{float(supervised.item()):.4f}",
            cons=f"{float(consistency.item()):.4f}",
        )
        if max_batches is not None and batch_idx >= max_batches:
            break
    progress.close()
    return total_loss / max(total_items, 1)


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


def run_single_fold(args: argparse.Namespace, task: TaskConfig, fold_id: int, output_dir: Path) -> dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = load_available_folds(args.split_csv)
    train_df, val_df, test_df = prepare_fold_data(args.registry_csv, args.split_csv, task, fold_id, len(fold_ids))

    train_loader = build_two_view_dataloader(
        train_df,
        input_variant=args.input_variant,
        image_size=args.image_size,
        view_a_profile=args.view_a_profile,
        view_b_profile=args.view_b_profile,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(val_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers)
    test_loader = build_dataloader(test_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers)

    device = resolve_device(args.device)
    model = DINOv3FineTuneModel(
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
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
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
        "view_a_profile": str(args.view_a_profile),
        "view_b_profile": str(args.view_b_profile),
        "consistency_weight": float(args.consistency_weight),
        "consistency_temperature": float(args.consistency_temperature),
        "class_weighting": bool(args.class_weighting),
        "fold_id": int(fold_id),
        "device": str(device),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "train_cases": int(train_df["case_id"].nunique()),
        "val_cases": int(val_df["case_id"].nunique()),
        "test_cases": int(test_df["case_id"].nunique()),
        "trainable": trainable,
        "selected_block_prefixes": sorted(selected_block_prefixes(model.encoder, args.tune_scope)),
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== Domain-consistency Fold {fold_id}/{len(fold_ids)} | task={task.key} | "
        f"variant={args.input_variant} | model={args.model_name} | "
        f"views={args.view_a_profile}+{args.view_b_profile} | trainable={trainable} ===",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Fold {fold_id}] Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_one_epoch_consistency(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            consistency_weight=float(args.consistency_weight),
            consistency_temperature=float(args.consistency_temperature),
            device=device,
            grad_clip=args.grad_clip,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=args.max_train_batches,
            progress_desc=f"cons fold{fold_id} train e{epoch}",
        )
        val_results = evaluate_model(
            model=model,
            loader=val_loader,
            task=task,
            device=device,
            aggregate_methods=aggregate_methods,
            criterion=criterion,
            max_batches=args.max_eval_batches,
            progress_desc=f"cons fold{fold_id} val e{epoch}",
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
        raise RuntimeError("Fine-tuning finished without a valid checkpoint.")
    try:
        best_state = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        best_state = torch.load(fold_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state)

    val_results = evaluate_model(model, val_loader, task, device, aggregate_methods, criterion, args.max_eval_batches, f"cons fold{fold_id} val best")
    test_results = evaluate_model(model, test_loader, task, device, aggregate_methods, criterion, args.max_eval_batches, f"cons fold{fold_id} test best")
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


def run_all_folds(args: argparse.Namespace, task: TaskConfig, output_dir: Path) -> None:
    fold_ids = load_available_folds(args.split_csv)
    metadata = make_image_df(args.registry_csv, args.split_csv, task)
    fold_summaries: list[dict[str, Any]] = []
    all_test_images: list[pd.DataFrame] = []
    all_test_cases: dict[str, list[pd.DataFrame]] = {
        item.strip(): [] for item in args.aggregate_methods.split(",") if item.strip()
    }
    print(f"Running domain-consistency folds={fold_ids} task={task.key}", flush=True)
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
    run_single_fold(args, task, int(args.fold), output_dir)


if __name__ == "__main__":
    main()
