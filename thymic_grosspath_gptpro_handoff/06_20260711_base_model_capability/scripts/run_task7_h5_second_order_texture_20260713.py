from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from run_task7_h3_summary_gated_20260713 import prediction_frame, sha256_file
from run_task7_h3b_masked_gated_20260713 import (
    EXPECTED_VIEWS,
    MaskedDenseDataset,
    MaskedGatedPool,
    validate_bank,
)
from run_task7_spatial_relational_20260713 import (
    EXPECTED_SOURCES,
    fold_partitions,
    load_metadata,
    metric_record,
    set_seed,
    source_risk_sampler,
    summarize_predictions,
    write_json,
)


CANDIDATE = "pe_spatial_lowrank_covariance_v1"
EXPECTED_FEATURE_SHA256 = (
    "e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked H5 low-rank second-order PE texture head."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--texture-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override")
    return parser.parse_args()


def validate_locked_args(args: argparse.Namespace) -> None:
    expected = {
        "candidate": CANDIDATE,
        "hidden_dim": 128,
        "attention_dim": 64,
        "texture_dim": 64,
        "dropout": 0.10,
        "epochs": 80,
        "patience": 12,
        "batch_size": 4,
        "num_workers": 0,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
    }
    mismatches = {
        name: (getattr(args, name), value)
        for name, value in expected.items()
        if getattr(args, name) != value
    }
    if mismatches:
        raise ValueError(f"Locked H5 arguments changed: {mismatches}")
    if args.seed not in (20260713, 20260714):
        raise ValueError("Only the primary and conditional confirmation seeds are allowed")
    if args.max_epochs not in (None, 1):
        raise ValueError("Only a one-epoch engineering smoke override is allowed")


def make_loader(
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None,
) -> DataLoader:
    return DataLoader(
        MaskedDenseDataset(features, masks, metadata, indices),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


class LowRankSecondOrderHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        attention_dim: int,
        texture_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)

        self.first_project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.first_view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        nn.init.normal_(self.first_view_embeddings, std=0.02)
        self.first_pool = MaskedGatedPool(hidden_dim, attention_dim)

        self.texture_project = nn.Linear(feature_dim, texture_dim, bias=False)
        upper_indices = torch.triu_indices(texture_dim, texture_dim)
        self.register_buffer("upper_row", upper_indices[0], persistent=False)
        self.register_buffer("upper_col", upper_indices[1], persistent=False)
        covariance_dim = texture_dim * (texture_dim + 1) // 2
        self.texture_encoder = nn.Sequential(
            nn.LayerNorm(covariance_dim),
            nn.Linear(covariance_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.texture_view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        nn.init.normal_(self.texture_view_embeddings, std=0.02)
        self.texture_pool = MaskedGatedPool(hidden_dim, attention_dim)

        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def second_order_tokens(
        self,
        normalized_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.texture_project(normalized_features)
        mask = valid_mask.unsqueeze(-1)
        counts = mask.sum(dim=-2, keepdim=True)
        if torch.any(counts < 2):
            raise ValueError("Every H5 view must contain at least two valid tokens")

        projected_float = projected.float()
        mask_float = mask.float()
        mean = (projected_float * mask_float).sum(dim=-2, keepdim=True) / counts.float()
        centered = (projected_float - mean) * mask_float
        covariance = torch.einsum("bvtd,bvte->bvde", centered, centered)
        covariance = covariance / (counts.float().squeeze(-1) - 1.0).unsqueeze(-1)
        upper = covariance[..., self.upper_row, self.upper_col]
        upper = torch.sign(upper) * torch.sqrt(torch.abs(upper).clamp_min(1e-6))
        upper = F.normalize(upper, p=2.0, dim=-1, eps=1e-6)
        return self.texture_encoder(upper)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, token_count, _ = features.shape
        normalized = self.input_norm(features)

        first_tokens = self.first_project(normalized)
        first_tokens = first_tokens + self.first_view_embeddings[:num_views].view(
            1, num_views, 1, -1
        )
        first_case = self.first_pool(
            first_tokens.reshape(batch_size, num_views * token_count, -1),
            valid_mask.reshape(batch_size, num_views * token_count),
        )

        texture_tokens = self.second_order_tokens(normalized, valid_mask)
        texture_tokens = texture_tokens + self.texture_view_embeddings[:num_views].view(
            1, num_views, -1
        )
        texture_case = self.texture_pool(texture_tokens, valid_mask.any(dim=-1))

        fused = self.fusion(torch.cat([first_case, texture_case], dim=-1))
        return self.classifier(fused)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True)
        if not use_bfloat16:
            feature = feature.float()
        mask = batch["valid_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            logits = model(feature, mask)
        probability = torch.softmax(logits.float(), dim=-1)[:, 1]
        indices.append(batch["index"].numpy())
        labels.append(batch["label"].numpy())
        probabilities.append(probability.cpu().numpy())
    return (
        np.concatenate(indices).astype(int),
        np.concatenate(labels).astype(int),
        np.concatenate(probabilities).astype(float),
    )


@dataclass
class TrainedModel:
    model: nn.Module
    best_epoch: int
    best_val_bacc: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainedModel:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    total_epochs = args.epochs if args.max_epochs is None else args.max_epochs
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_bacc = -math.inf
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, total_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            feature = batch["feature"].to(device, non_blocking=True)
            if not use_bfloat16:
                feature = feature.float()
            mask = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                logits = model(feature, mask)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite H5 training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite H5 gradient norm")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        _, val_labels, val_probability = predict(model, val_loader, device)
        val_bacc = float(metric_record(val_labels, val_probability)["balanced_accuracy"])
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_bacc": val_bacc,
        }
        history.append(row)
        print(
            f"[{args.candidate}] epoch={epoch} loss={row['loss']:.5f} "
            f"val_bacc={val_bacc:.4f}",
            flush=True,
        )
        if val_bacc > best_bacc + 1e-12:
            best_bacc = val_bacc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("H5 training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save(
        {"state_dict": best_state, "best_epoch": best_epoch, "best_val_bacc": best_bacc},
        output_dir / "best_head.pt",
    )
    return TrainedModel(model=model, best_epoch=best_epoch, best_val_bacc=best_bacc)


def run_fold(
    fold_id: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    test_path = fold_dir / "test_predictions.csv"
    validation_path = fold_dir / "validation_predictions.csv"
    summary_path = fold_dir / "fold_summary.json"
    if test_path.exists() and validation_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_csv(
            test_path, dtype={"case_id": str}
        )

    train_indices, validation_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    fold_seed = args.seed + 1000 * fold_id
    set_seed(fold_seed)
    device = torch.device(args.device)
    model = LowRankSecondOrderHead(
        feature_dim=int(features.shape[-1]),
        num_views=int(features.shape[1]),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        texture_dim=args.texture_dim,
        dropout=args.dropout,
    )
    sampler = source_risk_sampler(metadata, train_indices, fold_seed + 1)
    loaders = [
        make_loader(
            features,
            masks,
            metadata,
            indices,
            args.batch_size,
            args.num_workers,
            sample,
        )
        for indices, sample in (
            (train_indices, sampler),
            (validation_indices, None),
            (test_indices, None),
        )
    ]
    trained = train_model(model, loaders[0], loaders[1], fold_dir, args, device)
    val_index, val_label, val_probability = predict(trained.model, loaders[1], device)
    test_index, test_label, test_probability = predict(trained.model, loaders[2], device)
    validation_frame = prediction_frame(
        metadata,
        val_index,
        val_label,
        val_probability,
        fold_id,
        held_source,
        args.candidate,
        "validation",
    )
    test_frame = prediction_frame(
        metadata,
        test_index,
        test_label,
        test_probability,
        fold_id,
        held_source,
        args.candidate,
        "test",
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    validation_frame.to_csv(validation_path, index=False, encoding="utf-8-sig")
    test_frame.to_csv(test_path, index=False, encoding="utf-8-sig")
    summary = {
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "candidate": args.candidate,
        "train_n": int(len(train_indices)),
        "val_n": int(len(validation_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": int(trained.best_epoch),
        "best_validation_bacc": float(trained.best_val_bacc),
        "validation_metrics": metric_record(val_label, val_probability),
        "test_metrics": metric_record(test_label, test_probability),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    write_json(summary_path, summary)
    del trained, model, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, test_frame


def main() -> None:
    args = parse_args()
    validate_locked_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)

    bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = bank_dir / "dense_bank_config.json"
    bank_config = json.loads(config_path.read_text(encoding="utf-8"))
    if bank_config.get("complete") is not True:
        raise ValueError("PE dense bank is incomplete")
    if tuple(bank_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("PE dense bank view order differs from the H5 lock")

    feature_path = bank_dir / "dense_features.float16.npy"
    actual_feature_hash = sha256_file(feature_path)
    if actual_feature_hash != EXPECTED_FEATURE_SHA256:
        raise ValueError("PE dense feature hash differs from the H5 preregistration")

    metadata = load_metadata(bank_dir, Path(args.split_csv))
    features = np.load(feature_path, mmap_mode="r")
    masks = np.load(bank_dir / "valid_token_mask.uint8.npy", mmap_mode="r")
    shapes = np.load(bank_dir / "spatial_shapes.int16.npy", mmap_mode="r")
    if features.shape != (591, len(EXPECTED_VIEWS), 1024, 1024):
        raise ValueError(f"Unexpected H5 feature shape: {features.shape}")
    if features.dtype != np.float16 or masks.shape != features.shape[:-1]:
        raise ValueError("Unexpected H5 feature or mask dtype/shape")
    validate_bank(features, masks, shapes)

    run_config = vars(args).copy()
    run_config.update(
        {
            "feature_shape": list(features.shape),
            "feature_sha256": actual_feature_hash,
            "bank_config_sha256": sha256_file(config_path),
            "source_counts": metadata["source_dataset"].value_counts().sort_index().to_dict(),
            "subtype_counts": metadata["task_l6_label"].value_counts().to_dict(),
            "expected_sources": list(EXPECTED_SOURCES),
            "selection_threshold": 0.5,
            "coverage": 1.0,
            "second_order_normalization": "upper_triangle_signed_sqrt_l2",
            "compute_precision": (
                "bfloat16_autocast"
                if args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else "float32"
            ),
        }
    )
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        if json.loads(run_config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("Existing H5 output has a different configuration")
    else:
        write_json(run_config_path, run_config)

    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    if args.fold != "all":
        requested_fold = int(args.fold)
        if requested_fold not in folds:
            raise ValueError(f"Invalid H5 fold {requested_fold}")
        folds = [requested_fold]

    summaries: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold_id in folds:
        summary, frame = run_fold(
            fold_id, features, masks, metadata, output_dir, args
        )
        summaries.append(summary)
        predictions.append(frame)
    pd.DataFrame(summaries).to_json(
        output_dir / "fold_summaries.json", orient="records", indent=2
    )
    expected_fold_count = 3 if args.split_mode == "source_lodo" else 5
    if len(folds) == expected_fold_count:
        summarize_predictions(pd.concat(predictions, ignore_index=True), output_dir)
        (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    else:
        (output_dir / "RUN.status").write_text("partial\n", encoding="utf-8")


if __name__ == "__main__":
    main()
