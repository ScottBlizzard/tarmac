from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from run_task7_h3_summary_gated_20260713 import prediction_frame, sha256_file
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


EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked masked H3B gated head.")
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override")
    return parser.parse_args()


class MaskedDenseDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        metadata: pd.DataFrame,
        indices: np.ndarray,
    ) -> None:
        self.features = features
        self.masks = masks
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "feature": torch.from_numpy(
                np.array(self.features[index], dtype=np.float16, copy=True)
            ),
            "valid_mask": torch.from_numpy(
                np.array(self.masks[index], dtype=np.uint8, copy=True)
            ).bool(),
            "label": torch.tensor(
                int(self.metadata.iloc[index]["label_idx"]), dtype=torch.long
            ),
            "index": torch.tensor(index, dtype=torch.long),
        }


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


class MaskedGatedPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.tanh = nn.Linear(hidden_dim, attention_dim)
        self.sigmoid = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if not torch.all(valid_mask.any(dim=-1)):
            raise ValueError("Every case must contain at least one valid token")
        scores = self.score(
            torch.tanh(self.tanh(tokens)) * torch.sigmoid(self.sigmoid(tokens))
        ).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, -torch.inf)
        weights = torch.softmax(scores.float(), dim=-1).to(tokens.dtype)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=-2)


class MaskedDenseGatedHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.pool = MaskedGatedPool(hidden_dim, attention_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, token_count, _ = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        tokens = tokens.reshape(batch_size, num_views * token_count, -1)
        mask = valid_mask.reshape(batch_size, num_views * token_count)
        pooled = self.pool(tokens, mask)
        return self.classifier(self.output_norm(pooled))


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
    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True)
        if device.type != "cuda":
            feature = feature.float()
        mask = batch["valid_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
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
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_epochs = args.epochs if args.max_epochs is None else min(args.epochs, args.max_epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_bacc = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, total_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            feature = batch["feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                feature = feature.float()
            mask = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(feature, mask)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        _, val_labels, val_probability = predict(model, val_loader, device)
        val_bacc = float(metric_record(val_labels, val_probability)["balanced_accuracy"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_bacc": val_bacc})
        print(
            f"[{args.candidate}] epoch={epoch} loss={np.mean(losses):.5f} "
            f"val_bacc={val_bacc:.4f}",
            flush=True,
        )
        if val_bacc > best_bacc + 1e-12:
            best_bacc = val_bacc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
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
    val_path = fold_dir / "validation_predictions.csv"
    summary_path = fold_dir / "fold_summary.json"
    if test_path.exists() and val_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_csv(
            test_path, dtype={"case_id": str}
        )
    train_indices, val_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    fold_seed = args.seed + 1000 * fold_id
    set_seed(fold_seed)
    device = torch.device(args.device)
    model = MaskedDenseGatedHead(
        feature_dim=int(features.shape[-1]),
        num_views=int(features.shape[1]),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    sampler = source_risk_sampler(metadata, train_indices, fold_seed + 1)
    loaders = [
        make_loader(features, masks, metadata, indices, args.batch_size, args.num_workers, sample)
        for indices, sample in (
            (train_indices, sampler),
            (val_indices, None),
            (test_indices, None),
        )
    ]
    trained = train_model(model, loaders[0], loaders[1], fold_dir, args, device)
    val_index, val_label, val_probability = predict(trained.model, loaders[1], device)
    test_index, test_label, test_probability = predict(trained.model, loaders[2], device)
    val_frame = prediction_frame(
        metadata, val_index, val_label, val_probability, fold_id, held_source,
        args.candidate, "validation"
    )
    test_frame = prediction_frame(
        metadata, test_index, test_label, test_probability, fold_id, held_source,
        args.candidate, "test"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    val_frame.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_frame.to_csv(test_path, index=False, encoding="utf-8-sig")
    summary = {
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "candidate": args.candidate,
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": trained.best_epoch,
        "best_val_bacc": trained.best_val_bacc,
        "validation_metrics": metric_record(val_label, val_probability),
        "test_metrics": metric_record(test_label, test_probability),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    write_json(summary_path, summary)
    del trained, model, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, test_frame


def validate_bank(features: np.ndarray, masks: np.ndarray, shapes: np.ndarray) -> None:
    for start in range(0, len(features), 8):
        stop = min(len(features), start + 8)
        feature = np.asarray(features[start:stop])
        mask = np.asarray(masks[start:stop]).astype(bool)
        if not np.isfinite(feature).all():
            raise ValueError(f"Non-finite dense features in rows {start}:{stop}")
        if np.any(feature[~mask] != 0):
            raise ValueError(f"Nonzero padded features in rows {start}:{stop}")
    counts = np.asarray(masks).sum(axis=-1)
    grid_counts = np.asarray(shapes, dtype=np.int64).prod(axis=-1)
    if not np.array_equal(counts, grid_counts):
        raise ValueError("Dense mask counts differ from spatial grid sizes")


def main() -> None:
    args = parse_args()
    if args.seed not in (20260713, 20260714):
        raise ValueError("Only the preregistered primary and confirmation seeds are allowed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    feature_bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = feature_bank_dir / "dense_bank_config.json"
    bank_config = json.loads(config_path.read_text(encoding="utf-8"))
    if bank_config.get("complete") is not True:
        raise ValueError("Dense feature bank is incomplete")
    if tuple(bank_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("Dense feature bank view order differs from H3 lock")
    metadata = load_metadata(feature_bank_dir, Path(args.split_csv))
    features = np.load(feature_bank_dir / "dense_features.float16.npy", mmap_mode="r")
    masks = np.load(feature_bank_dir / "valid_token_mask.uint8.npy", mmap_mode="r")
    shapes = np.load(feature_bank_dir / "spatial_shapes.int16.npy", mmap_mode="r")
    if features.shape[:3] != (591, len(EXPECTED_VIEWS), 1024):
        raise ValueError(f"Unexpected dense shape: {features.shape}")
    if features.dtype != np.float16 or masks.shape != features.shape[:-1]:
        raise ValueError("Dense feature/mask dtype or shape mismatch")
    validate_bank(features, masks, shapes)
    run_config = vars(args).copy()
    run_config.update(
        {
            "feature_shape": list(features.shape),
            "bank_config_sha256": sha256_file(config_path),
            "valid_tokens_min": int(np.asarray(masks).sum(axis=-1).min()),
            "valid_tokens_max": int(np.asarray(masks).sum(axis=-1).max()),
            "source_counts": metadata["source_dataset"].value_counts().sort_index().to_dict(),
            "subtype_counts": metadata["task_l6_label"].value_counts().to_dict(),
            "expected_sources": list(EXPECTED_SOURCES),
            "selection_threshold": 0.5,
            "coverage": 1.0,
        }
    )
    locked_config = output_dir / "run_config.json"
    if locked_config.exists():
        if json.loads(locked_config.read_text(encoding="utf-8")) != run_config:
            raise ValueError("Existing H3B output has a different configuration")
    else:
        write_json(locked_config, run_config)
    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    if args.fold != "all":
        requested = int(args.fold)
        if requested not in folds:
            raise ValueError(f"Invalid fold {requested}")
        folds = [requested]
    summaries = []
    predictions = []
    for fold_id in folds:
        summary, frame = run_fold(fold_id, features, masks, metadata, output_dir, args)
        summaries.append(summary)
        predictions.append(frame)
    pd.DataFrame(summaries).to_json(
        output_dir / "fold_summaries.json", orient="records", indent=2
    )
    expected = 3 if args.split_mode == "source_lodo" else 5
    if len(folds) == expected:
        summarize_predictions(pd.concat(predictions, ignore_index=True), output_dir)
        (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    else:
        (output_dir / "RUN.status").write_text("partial\n", encoding="utf-8")


if __name__ == "__main__":
    main()
