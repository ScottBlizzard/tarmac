from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_finetune_old_third_20260523 import (  # noqa: E402
    DEFAULT_INPUT,
    load_available_folds,
    prepare_fold_data,
)
from run_task7_lora_dense_finetune_20260711 import dense_tokens, inject_lora  # noqa: E402
from thymic_baseline.config import get_task  # noqa: E402
from thymic_baseline.cropping import extract_specimen_crop  # noqa: E402
from thymic_baseline.data import build_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foldwise label-free VICReg adaptation of Task7 LoRA layers.")
    parser.add_argument("--registry-csv", default=str(DEFAULT_INPUT / "registry.csv"))
    parser.add_argument("--split-csv", default=str(DEFAULT_INPUT / "split.csv"))
    parser.add_argument("--task", default="task7_lowrisk_vs_highrisk_tc")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="vit_large_patch16_dinov3_qkvb.lvd1689m")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--input-variant", choices=("whole", "crop"), default="whole")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--view-a-profile", default="style_light")
    parser.add_argument("--view-b-profile", default="domain_robust")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-last-blocks", type=int, default=2)
    parser.add_argument("--lora-targets", default="qkv,proj")
    parser.add_argument("--train-final-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-hidden-dim", type=int, default=1024)
    parser.add_argument("--invariance-weight", type=float, default=25.0)
    parser.add_argument("--variance-weight", type=float, default=25.0)
    parser.add_argument("--covariance-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--projector-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class TwoViewDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        input_variant: str,
        image_size: int,
        profile_a: str,
        profile_b: str,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.input_variant = input_variant
        self.transform_a = transform_for_profile(image_size, profile_a)
        self.transform_b = transform_for_profile(image_size, profile_b)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        if self.input_variant == "crop":
            image = extract_specimen_crop(image)
        return self.transform_a(image), self.transform_b(image)


def make_loader(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    training: bool,
) -> DataLoader:
    dataset = TwoViewDataset(
        frame,
        input_variant=args.input_variant,
        image_size=args.image_size,
        profile_a=args.view_a_profile,
        profile_b=args.view_b_profile,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": training,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": training,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


class LoRAVICReg(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.encoder = timm.create_model(args.model_name, pretrained=True, num_classes=0, global_pool="")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        targets = {item.strip() for item in args.lora_targets.split(",") if item.strip()}
        self.lora_modules = inject_lora(
            self.encoder,
            last_blocks=args.lora_last_blocks,
            targets=targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        if args.train_final_norm:
            for name, parameter in self.encoder.named_parameters():
                if name.startswith("norm.") or name.startswith("fc_norm."):
                    parameter.requires_grad = True
        feature_dim = int(self.encoder.num_features)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, args.projection_hidden_dim),
            nn.BatchNorm1d(args.projection_hidden_dim),
            nn.GELU(),
            nn.Linear(args.projection_hidden_dim, args.projection_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = dense_tokens(self.encoder, inputs)
        pooled = tokens.mean(dim=1)
        return self.projector(pooled)


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    invariance = F.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-4)
    second_std = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (F.relu(1.0 - first_std).mean() + F.relu(1.0 - second_std).mean())
    if first.shape[0] > 1:
        first_cov = (first_centered.T @ first_centered) / (first.shape[0] - 1)
        second_cov = (second_centered.T @ second_centered) / (second.shape[0] - 1)
        covariance = (
            off_diagonal(first_cov).pow(2).sum() + off_diagonal(second_cov).pow(2).sum()
        ) / first.shape[1]
    else:
        covariance = first.sum() * 0.0
    total = (
        float(invariance_weight) * invariance
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
    )
    return total, {
        "total": float(total.detach()),
        "invariance": float(invariance.detach()),
        "variance": float(variance.detach()),
        "covariance": float(covariance.detach()),
    }


def encoder_trainable_state(model: LoRAVICReg) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.encoder.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.encoder.state_dict().items()
        if name in names
    }


def run_epoch(
    model: LoRAVICReg,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, list[float]] = {}
    max_batches = args.max_train_batches if training else args.max_eval_batches
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_idx, (first, second) in enumerate(loader, start=1):
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"
            ):
                first_projection = model(first)
                second_projection = model(second)
                loss, parts = vicreg_loss(
                    first_projection,
                    second_projection,
                    args.invariance_weight,
                    args.variance_weight,
                    args.covariance_weight,
                )
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
            for key, value in parts.items():
                totals.setdefault(key, []).append(value)
            if max_batches is not None and batch_idx >= max_batches:
                break
    return {key: float(np.mean(values)) for key, values in totals.items()}


def run_fold(args: argparse.Namespace, fold_id: int, output_dir: Path) -> None:
    set_seed(args.seed + fold_id)
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = load_available_folds(args.split_csv)
    task = get_task(args.task)
    train_df, val_df, _ = prepare_fold_data(args.registry_csv, args.split_csv, task, fold_id, len(fold_ids))
    train_loader = make_loader(train_df, args, training=True)
    val_loader = make_loader(val_df, args, training=False)
    device = torch.device(args.device)
    model = LoRAVICReg(args).to(device)
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.lr},
            {"params": model.projector.parameters(), "lr": args.projector_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history = []
    config = vars(args).copy()
    config.update(
        {
            "fold_id": fold_id,
            "train_cases": int(train_df["case_id"].nunique()),
            "val_cases": int(val_df["case_id"].nunique()),
            "lora_modules": model.lora_modules,
            "encoder_trainable_parameters": int(sum(p.numel() for p in encoder_parameters)),
        }
    )
    (fold_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        train_parts = run_epoch(model, train_loader, args, device, optimizer)
        val_parts = run_epoch(model, val_loader, args, device, None)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_parts.items()},
                **{f"val_{key}": value for key, value in val_parts.items()},
            }
        )
        val_total = float(val_parts["total"])
        if val_total < best_val - 1e-8:
            best_val = val_total
            best_epoch = epoch
            stale = 0
            torch.save(encoder_trainable_state(model), fold_dir / "encoder_lora_state.pt")
        else:
            stale += 1
        print(
            f"ssl fold={fold_id} epoch={epoch} train={train_parts['total']:.4f} "
            f"val={val_total:.4f} best={best_val:.4f} stale={stale}",
            flush=True,
        )
        if stale >= args.patience:
            break
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    (fold_dir / "fold_summary.json").write_text(
        json.dumps({"fold_id": fold_id, "best_epoch": best_epoch, "best_val_vicreg": best_val}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    folds = load_available_folds(args.split_csv) if args.fold == "all" else [int(args.fold)]
    for fold_id in folds:
        run_fold(args, fold_id, output_dir)


if __name__ == "__main__":
    main()
