from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (
    _SCRIPT_PROJECT_ROOT
    if (_SCRIPT_PROJECT_ROOT / "thymic_baseline").exists()
    else Path("/workspace/thymic_project")
)
for item in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_h3b_masked_gated_20260713 import MaskedDenseGatedHead  # noqa: E402
from run_task7_spatial_relational_20260713 import (  # noqa: E402
    EXPECTED_SOURCES,
    fold_partitions,
    load_metadata,
    metric_record,
)

EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked PE final-block LoRA adaptation from the H12 prefix bank."
    )
    parser.add_argument("--prefix-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--h3-reference-root", required=True)
    parser.add_argument("--h3-reference-oof", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-code-dir", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-mode", choices=("source_lodo", "fivefold"), required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--lora-weight-decay", type=float, default=1e-2)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None, help="Engineering smoke override")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Engineering smoke override")
    return parser.parse_args()


def validate_locked_args(args: argparse.Namespace) -> None:
    expected = {
        "rank": 8,
        "alpha": 8.0,
        "hidden_dim": 128,
        "attention_dim": 64,
        "dropout": 0.10,
        "epochs": 24,
        "patience": 5,
        "batch_size": 4,
        "effective_batch_size": 8,
        "num_workers": 0,
        "lora_lr": 1e-4,
        "head_lr": 3e-5,
        "lora_weight_decay": 1e-2,
        "head_weight_decay": 1e-4,
        "grad_clip": 1.0,
        "reproduction_tolerance": 0.005,
    }
    if args.max_epochs is None and args.max_train_batches is None:
        mismatch = {
            key: (getattr(args, key), value)
            for key, value in expected.items()
            if getattr(args, key) != value
        }
        if mismatch:
            raise ValueError(f"Locked H12 arguments changed: {mismatch}")
    if args.seed not in (20260715, 20260716):
        raise ValueError("Only the primary and conditional confirmation H12 seeds are allowed")
    if args.effective_batch_size % args.batch_size:
        raise ValueError("Effective batch size must be divisible by physical batch size")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def sha256_file(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


class PrefixDataset(Dataset):
    def __init__(
        self,
        prefix: np.ndarray,
        spatial_shapes: np.ndarray,
        metadata: pd.DataFrame,
        indices: np.ndarray,
    ) -> None:
        self.prefix = prefix
        self.spatial_shapes = spatial_shapes
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "prefix": torch.from_numpy(
                np.array(self.prefix[index], dtype=np.float16, copy=True)
            ),
            "spatial_shapes": torch.from_numpy(
                np.array(self.spatial_shapes[index], dtype=np.int16, copy=True)
            ),
            "label": torch.tensor(
                int(self.metadata.iloc[index]["label_idx"]), dtype=torch.long
            ),
            "index": torch.tensor(index, dtype=torch.long),
        }


def make_loader(
    prefix: np.ndarray,
    shapes: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        PrefixDataset(prefix, shapes, metadata, indices),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def source_risk_sampler(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    seed: int,
) -> WeightedRandomSampler:
    subset = metadata.iloc[indices]
    groups = list(
        zip(subset["source_dataset"].astype(str), subset["label_idx"].astype(int))
    )
    counts = pd.Series(groups, dtype="object").value_counts().to_dict()
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )


class LowRankWeightDelta(nn.Module):
    def __init__(self, output_dim: int, input_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim, rank))
        self.scale = float(alpha / rank)
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + (self.b @ self.a) * self.scale


def register_lora(block: nn.Module, rank: int, alpha: float) -> list[nn.Parameter]:
    for parameter in block.parameters():
        parameter.requires_grad = False
    targets = (
        (block.attn, "in_proj_weight"),
        (block.attn.out_proj, "weight"),
        (block.mlp.c_fc, "weight"),
        (block.mlp.c_proj, "weight"),
    )
    for module, parameter_name in targets:
        weight = getattr(module, parameter_name)
        if weight.ndim != 2:
            raise ValueError(f"LoRA target {parameter_name} is not a matrix")
        parametrize.register_parametrization(
            module,
            parameter_name,
            LowRankWeightDelta(weight.shape[0], weight.shape[1], rank, alpha),
        )
    trainable = [parameter for parameter in block.parameters() if parameter.requires_grad]
    expected = 131072 if rank == 8 else None
    count = int(sum(parameter.numel() for parameter in trainable))
    if expected is not None and count != expected:
        raise ValueError(f"Unexpected rank-8 LoRA parameter count: {count} != {expected}")
    return trainable


def trainable_state(block: nn.Module, head: nn.Module) -> dict[str, Any]:
    return {
        "lora": {
            name: parameter.detach().cpu().clone()
            for name, parameter in block.named_parameters()
            if parameter.requires_grad
        },
        "head": {
            name: value.detach().cpu().clone() for name, value in head.state_dict().items()
        },
    }


def load_trainable_state(block: nn.Module, head: nn.Module, state: dict[str, Any]) -> None:
    named_parameters = dict(block.named_parameters())
    if set(state["lora"]) != {
        name for name, parameter in named_parameters.items() if parameter.requires_grad
    }:
        raise ValueError("Saved LoRA parameter names do not match the locked model")
    with torch.no_grad():
        for name, value in state["lora"].items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
    head.load_state_dict(state["head"], strict=True)


def load_base_block(args: argparse.Namespace) -> nn.Module:
    code_dir = Path(args.model_code_dir).resolve(strict=True)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from core.vision_encoder.pe import VisionTransformer

    model = VisionTransformer.from_config(
        "PE-Spatial-L14-448",
        pretrained=True,
        checkpoint_path=str(Path(args.model_id).resolve(strict=True)),
    ).eval()
    if not model.use_cls_token or model.layers != 24 or model.width != 1024:
        raise ValueError("Unexpected PE-Spatial architecture")
    block = copy.deepcopy(model.transformer.resblocks[-1]).cpu().eval()
    block.h12_ln_post = copy.deepcopy(model.ln_post).cpu().eval()
    del model
    gc.collect()
    return block


def create_components(
    base_block: nn.Module,
    args: argparse.Namespace,
    fold_id: int,
    device: torch.device,
) -> tuple[nn.Module, MaskedDenseGatedHead, list[nn.Parameter]]:
    block = copy.deepcopy(base_block)
    register_lora(block, args.rank, args.alpha)
    block = block.to(device)
    lora_parameters = [
        parameter for parameter in block.parameters() if parameter.requires_grad
    ]
    head = MaskedDenseGatedHead(
        feature_dim=1024,
        num_views=len(EXPECTED_VIEWS),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    checkpoint_path = Path(args.h3_reference_root) / f"fold_{fold_id}" / "best_head.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head = head.to(device)
    return block, head, lora_parameters


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def adapted_dense_features(
    block: nn.Module,
    prefix: torch.Tensor,
    spatial_shapes: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_views, _, feature_dim = prefix.shape
    if num_views != len(EXPECTED_VIEWS) or feature_dim != 1024:
        raise ValueError(f"Unexpected H12 prefix batch shape: {tuple(prefix.shape)}")
    case_features: list[torch.Tensor] = []
    case_masks: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        view_features: list[torch.Tensor] = []
        view_masks: list[torch.Tensor] = []
        for view_index in range(num_views):
            rows = int(spatial_shapes[batch_index, view_index, 0])
            columns = int(spatial_shapes[batch_index, view_index, 1])
            patch_count = rows * columns
            if not (1 <= patch_count <= 1024):
                raise ValueError(f"Invalid PE patch grid: {rows}x{columns}")
            tokens = prefix[
                batch_index, view_index, : patch_count + 1
            ].to(device, non_blocking=True).unsqueeze(0)
            block.attn.rope.update_grid(device, rows, columns)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = block.h12_ln_post(block(tokens))[:, 1:, :]
            output = output.to(torch.float16 if device.type == "cuda" else torch.float32)
            output = F.pad(output, (0, 0, 0, 1024 - patch_count)).squeeze(0)
            view_features.append(output)
            mask = torch.arange(1024, device=device) < patch_count
            view_masks.append(mask)
        case_features.append(torch.stack(view_features, dim=0))
        case_masks.append(torch.stack(view_masks, dim=0))
    return torch.stack(case_features, dim=0), torch.stack(case_masks, dim=0)


@torch.no_grad()
def predict(
    block: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block.eval()
    head.eval()
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        features, masks = adapted_dense_features(
            block, batch["prefix"], batch["spatial_shapes"], device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = head(features, masks)
        probability = torch.softmax(logits.float(), dim=-1)[:, 1]
        indices.append(batch["index"].numpy())
        labels.append(batch["label"].numpy())
        probabilities.append(probability.cpu().numpy())
    return (
        np.concatenate(indices).astype(int),
        np.concatenate(labels).astype(int),
        np.concatenate(probabilities).astype(float),
    )


def prediction_frame(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    probabilities: np.ndarray,
    fold_id: int,
    held_source: str,
    candidate: str,
) -> pd.DataFrame:
    frame = metadata.iloc[indices][
        ["feature_row", "case_id", "source_dataset", "task_l6_label", "label_idx"]
    ].copy()
    frame["fold_id"] = int(fold_id)
    frame["held_out_source"] = held_source
    frame["candidate"] = candidate
    frame["split_role"] = "test"
    frame["prob_high"] = probabilities
    frame["pred_idx"] = (probabilities >= 0.5).astype(int)
    frame["correct"] = frame["pred_idx"].to_numpy() == frame["label_idx"].to_numpy()
    return frame


def load_bank(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    bank_dir = Path(args.prefix_bank_dir)
    config_path = bank_dir / "prefix_bank_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("complete") is not True:
        raise ValueError("H12 prefix bank is incomplete")
    if tuple(config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("H12 prefix view order differs")
    if config.get("model_weight_sha256") != sha256_file(args.model_id):
        raise ValueError("PE model weight hash differs from H12 prefix bank")
    if config.get("model_code_revision") != args.code_revision:
        raise ValueError("PE source revision differs from H12 prefix bank")
    prefix_path = bank_dir / "prefix_features.float16.npy"
    shapes_path = bank_dir / "spatial_shapes.int16.npy"
    if prefix_path.stat().st_size != int(config["prefix_file_bytes"]):
        raise ValueError("H12 prefix file size differs from locked config")
    prefix = np.load(prefix_path, mmap_mode="r")
    shapes = np.load(shapes_path, mmap_mode="r")
    metadata = load_metadata(bank_dir, Path(args.split_csv))
    if prefix.shape != (591, 6, 1025, 1024) or shapes.shape != (591, 6, 2):
        raise ValueError(f"Unexpected H12 bank shapes: {prefix.shape}, {shapes.shape}")
    return prefix, shapes, metadata, config


def selected_folds(args: argparse.Namespace) -> list[int]:
    maximum = 3 if args.split_mode == "source_lodo" else 5
    if args.fold == "all":
        return list(range(1, maximum + 1))
    fold = int(args.fold)
    if fold not in range(1, maximum + 1):
        raise ValueError(f"Invalid {args.split_mode} fold: {fold}")
    return [fold]


def reproduction_preflight(
    args: argparse.Namespace,
    base_block: nn.Module,
    prefix: np.ndarray,
    shapes: np.ndarray,
    metadata: pd.DataFrame,
    folds: list[int],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    reference = pd.read_csv(args.h3_reference_oof, dtype={"case_id": str})
    reference = reference.set_index("case_id")
    frames: list[pd.DataFrame] = []
    by_fold: dict[str, Any] = {}
    for fold_id in folds:
        _, _, test_indices, held_source = fold_partitions(
            metadata, args.split_mode, fold_id
        )
        block, head, _ = create_components(base_block, args, fold_id, device)
        loader = make_loader(
            prefix,
            shapes,
            metadata,
            test_indices,
            batch_size=8,
            num_workers=args.num_workers,
        )
        indices, labels, probabilities = predict(block, head, loader, device)
        frame = prediction_frame(
            metadata,
            indices,
            probabilities,
            fold_id,
            held_source,
            "H12_ZERO_LORA_H3_REPRODUCTION",
        )
        expected = reference.loc[frame["case_id"]]
        if not np.array_equal(frame["label_idx"].to_numpy(), expected["label_idx"].to_numpy()):
            raise ValueError(f"H3 label mismatch in fold {fold_id}")
        frame["reference_prob_high"] = expected["prob_high"].to_numpy(dtype=float)
        frame["reference_pred_idx"] = expected["pred_idx"].to_numpy(dtype=int)
        frame["absolute_probability_error"] = (
            frame["prob_high"] - frame["reference_prob_high"]
        ).abs()
        frame["prediction_match"] = frame["pred_idx"] == frame["reference_pred_idx"]
        by_fold[str(fold_id)] = {
            "held_out_source": held_source,
            "n": int(len(frame)),
            "prediction_match_count": int(frame["prediction_match"].sum()),
            "max_absolute_probability_error": float(frame["absolute_probability_error"].max()),
            "mean_absolute_probability_error": float(frame["absolute_probability_error"].mean()),
        }
        frames.append(frame)
        del block, head
        torch.cuda.empty_cache()
    combined = pd.concat(frames, ignore_index=True)
    complete_expected = 591 if args.fold == "all" else len(combined)
    passed = (
        len(combined) == complete_expected
        and bool(combined["prediction_match"].all())
        and float(combined["absolute_probability_error"].max())
        <= args.reproduction_tolerance
    )
    gate = {
        "passed": passed,
        "n": int(len(combined)),
        "expected_n": int(complete_expected),
        "classification_match": bool(combined["prediction_match"].all()),
        "max_absolute_probability_error": float(
            combined["absolute_probability_error"].max()
        ),
        "tolerance": args.reproduction_tolerance,
        "by_fold": by_fold,
    }
    write_csv(output_dir / "h12_zero_lora_reproduction_predictions.csv", combined)
    write_json(output_dir / "h12_zero_lora_reproduction_gate.json", gate)
    if not passed:
        raise RuntimeError(f"H12 zero-LoRA reproduction gate failed: {gate}")
    return gate


def score_is_better(
    metrics: dict[str, Any],
    best_bacc: float,
    best_auc: float,
) -> bool:
    bacc = float(metrics["balanced_accuracy"])
    auc = float(metrics["auc"])
    return bacc > best_bacc + 1e-12 or (
        abs(bacc - best_bacc) <= 1e-12 and auc > best_auc + 1e-12
    )


def run_fold(
    args: argparse.Namespace,
    base_block: nn.Module,
    prefix: np.ndarray,
    shapes: np.ndarray,
    metadata: pd.DataFrame,
    fold_id: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    test_path = fold_dir / "test_predictions.csv"
    summary_path = fold_dir / "fold_summary.json"
    best_path = fold_dir / "best_adapter_head.pt"
    if test_path.exists() and summary_path.exists() and best_path.exists():
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            pd.read_csv(test_path, dtype={"case_id": str}),
        )

    train_indices, val_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    fold_seed = args.seed + 1000 * fold_id
    set_seed(fold_seed)
    block, head, lora_parameters = create_components(base_block, args, fold_id, device)
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": args.lora_lr,
                "weight_decay": args.lora_weight_decay,
            },
            {
                "params": head_parameters,
                "lr": args.head_lr,
                "weight_decay": args.head_weight_decay,
            },
        ]
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation_steps = args.effective_batch_size // args.batch_size
    val_loader = make_loader(
        prefix, shapes, metadata, val_indices, 8, args.num_workers
    )
    test_loader = make_loader(
        prefix, shapes, metadata, test_indices, 8, args.num_workers
    )
    resume_path = fold_dir / "last_state.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    stale = 0
    best_epoch = 0
    best_bacc = -math.inf
    best_auc = -math.inf
    best_state: dict[str, Any] | None = None

    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        load_trainable_state(block, head, resume["current_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        move_optimizer_state(optimizer, device)
        scaler.load_state_dict(resume["scaler_state"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
        stale = int(resume["stale"])
        best_epoch = int(resume["best_epoch"])
        best_bacc = float(resume["best_bacc"])
        best_auc = float(resume["best_auc"])
        best_state = resume["best_state"]
    else:
        val_indices_pred, val_labels, val_probability = predict(
            block, head, val_loader, device
        )
        del val_indices_pred
        val_metrics = metric_record(val_labels, val_probability)
        best_bacc = float(val_metrics["balanced_accuracy"])
        best_auc = float(val_metrics["auc"])
        best_state = trainable_state(block, head)
        history.append(
            {
                "epoch": 0,
                "train_loss": math.nan,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "selected": True,
            }
        )
        write_csv(fold_dir / "training_history.csv", pd.DataFrame(history))

    total_epochs = args.epochs if args.max_epochs is None else min(args.epochs, args.max_epochs)
    train_started = time.monotonic()
    all_trainable = lora_parameters + head_parameters
    for epoch in range(start_epoch, total_epochs + 1):
        sampler = source_risk_sampler(metadata, train_indices, fold_seed + epoch)
        train_loader = make_loader(
            prefix,
            shapes,
            metadata,
            train_indices,
            args.batch_size,
            args.num_workers,
            sampler,
        )
        block.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        batch_count = len(train_loader)
        if args.max_train_batches is not None:
            batch_count = min(batch_count, int(args.max_train_batches))
        for batch_number, batch in enumerate(train_loader, start=1):
            if batch_number > batch_count:
                break
            labels = batch["label"].to(device, non_blocking=True)
            features, masks = adapted_dense_features(
                block, batch["prefix"], batch["spatial_shapes"], device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = head(features, masks)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss / accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))
            should_step = batch_number % accumulation_steps == 0 or batch_number == batch_count
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        _, val_labels, val_probability = predict(block, head, val_loader, device)
        val_metrics = metric_record(val_labels, val_probability)
        selected = score_is_better(val_metrics, best_bacc, best_auc)
        if selected:
            best_bacc = float(val_metrics["balanced_accuracy"])
            best_auc = float(val_metrics["auc"])
            best_epoch = epoch
            best_state = trainable_state(block, head)
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "selected": selected,
            }
        )
        write_csv(fold_dir / "training_history.csv", pd.DataFrame(history))
        torch.save(
            {
                "epoch": epoch,
                "stale": stale,
                "best_epoch": best_epoch,
                "best_bacc": best_bacc,
                "best_auc": best_auc,
                "best_state": best_state,
                "current_state": trainable_state(block, head),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
            },
            resume_path,
        )
        print(
            f"[H12 {args.split_mode} fold={fold_id}] epoch={epoch} "
            f"loss={np.mean(losses):.5f} val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} selected={selected} stale={stale}",
            flush=True,
        )
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("H12 training did not retain an epoch-0 or adapted state")
    load_trainable_state(block, head, best_state)
    val_pred_indices, val_labels, val_probability = predict(block, head, val_loader, device)
    test_pred_indices, test_labels, test_probability = predict(block, head, test_loader, device)
    val_frame = prediction_frame(
        metadata,
        val_pred_indices,
        val_probability,
        fold_id,
        held_source,
        "H12_PE_FINAL_BLOCK_LORA",
    )
    val_frame["split_role"] = "validation"
    test_frame = prediction_frame(
        metadata,
        test_pred_indices,
        test_probability,
        fold_id,
        held_source,
        "H12_PE_FINAL_BLOCK_LORA",
    )
    val_metrics = metric_record(val_labels, val_probability)
    test_metrics = metric_record(test_labels, test_probability)
    checkpoint = {
        "experiment": "H12_PE_FINAL_BLOCK_LORA_ADAPTATION_20260715",
        "split_mode": args.split_mode,
        "fold_id": fold_id,
        "held_out_source": held_source,
        "best_epoch": best_epoch,
        "best_val_bacc": best_bacc,
        "best_val_auc": best_auc,
        "rank": args.rank,
        "alpha": args.alpha,
        "trainable_parameter_count": int(sum(p.numel() for p in all_trainable)),
        "state": best_state,
    }
    torch.save(checkpoint, best_path)
    write_csv(fold_dir / "validation_predictions.csv", val_frame)
    write_csv(test_path, test_frame)
    summary = {
        "fold_id": fold_id,
        "held_out_source": held_source,
        "train_n": int(len(train_indices)),
        "validation_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": best_epoch,
        "best_val_bacc": best_bacc,
        "best_val_auc": best_auc,
        "adapted": best_epoch > 0,
        "train_elapsed_seconds": time.monotonic() - train_started,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    write_json(summary_path, summary)
    return summary, test_frame


def grouped_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby(group_column, dropna=False, sort=True):
        rows.append(
            {
                group_column: group_name,
                **metric_record(group["label_idx"], group["prob_high"]),
                "correct_count": int(group["correct"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    validate_locked_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix, shapes, metadata, bank_config = load_bank(args)
    folds = selected_folds(args)
    base_block = load_base_block(args)

    run_config = {
        "experiment": "H12_PE_FINAL_BLOCK_LORA_ADAPTATION_20260715",
        "trainer_sha256": sha256_file(__file__),
        "args": vars(args),
        "prefix_bank_dir": str(Path(args.prefix_bank_dir).resolve()),
        "prefix_bank_config_sha256": sha256_file(
            Path(args.prefix_bank_dir) / "prefix_bank_config.json"
        ),
        "prefix_sha256": bank_config["prefix_sha256"],
        "split_sha256": sha256_file(args.split_csv),
        "h3_reference_oof_sha256": sha256_file(args.h3_reference_oof),
        "model_weight_sha256": sha256_file(args.model_id),
        "folds": folds,
        "selection_threshold": 0.5,
        "coverage_required": 591 if args.fold == "all" else None,
        "uses_source_as_input": False,
        "uses_doctor_concepts": False,
        "uses_historical_probabilities_as_input": False,
    }
    write_json(output_dir / "run_config.json", run_config)
    reproduction = reproduction_preflight(
        args,
        base_block,
        prefix,
        shapes,
        metadata,
        folds,
        output_dir,
        device,
    )
    print(json.dumps(reproduction, ensure_ascii=False, indent=2), flush=True)

    summaries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for fold_id in folds:
        summary, frame = run_fold(
            args,
            base_block,
            prefix,
            shapes,
            metadata,
            fold_id,
            output_dir,
            device,
        )
        summaries.append(summary)
        frames.append(frame)
        write_json(output_dir / "fold_summaries.partial.json", summaries)

    oof = pd.concat(frames, ignore_index=True).sort_values("feature_row").reset_index(drop=True)
    if args.fold == "all":
        if len(oof) != 591 or oof["case_id"].duplicated().any():
            raise RuntimeError(f"H12 OOF coverage failure: rows={len(oof)}")
    write_csv(output_dir / "oof_predictions.csv", oof)
    overall = metric_record(oof["label_idx"], oof["prob_high"])
    write_json(output_dir / "overall_metrics.json", overall)
    write_json(output_dir / "fold_summaries.json", summaries)
    write_csv(
        output_dir / "source_metrics.csv",
        grouped_metrics(oof, "source_dataset"),
    )
    write_csv(
        output_dir / "subtype_metrics.csv",
        grouped_metrics(oof, "task_l6_label"),
    )
    (output_dir / "RUN.status").write_text("COMPLETE\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
