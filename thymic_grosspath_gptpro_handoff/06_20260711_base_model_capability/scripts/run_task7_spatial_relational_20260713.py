from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


EXPECTED_SOURCES = ("batch1", "batch2", "third_batch")
EXPECTED_SUBTYPES = {"A": 44, "AB": 262, "B1": 62, "B2": 89, "B3": 24, "TC": 110}
EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
LOW_RISK = {"A", "AB", "B1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked canonical-coordinate spatial-relational Task7 experiment."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--coordinate-dir", required=True)
    parser.add_argument("--integrity-manifest", required=True)
    parser.add_argument("--expected-integrity-sha256", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=("matched_gated", "relational_permuted", "relational"),
        required=True,
    )
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--global-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--permutation-seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_source(value: object) -> str:
    text = str(value)
    if text.startswith("third_batch"):
        return "third_batch"
    return text


def metric_record(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_metadata(feature_bank_dir: Path, split_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        feature_bank_dir / "metadata.csv",
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    metadata.columns = [str(column).lstrip("\ufeff") for column in metadata.columns]
    split = pd.read_csv(split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split.columns = [str(column).lstrip("\ufeff") for column in split.columns]
    split = split[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    authoritative = metadata[["case_id"]].merge(split, on="case_id", how="left")["master_fold_id"]
    fallback = pd.to_numeric(metadata.get("master_fold_id"), errors="coerce")
    metadata["master_fold_id"] = pd.to_numeric(authoritative, errors="coerce").fillna(fallback)
    metadata["source_dataset"] = metadata["source_dataset"].map(canonical_source)
    metadata["task_l6_label"] = metadata["task_l6_label"].astype(str)
    metadata["label_idx"] = (~metadata["task_l6_label"].isin(LOW_RISK)).astype(int)
    metadata["feature_row"] = np.arange(len(metadata), dtype=int)
    if len(metadata) != 591 or metadata["case_id"].duplicated().any():
        raise ValueError("Expected 591 unique feature metadata rows")
    if metadata["master_fold_id"].isna().any():
        raise ValueError("Missing locked master folds")
    metadata["master_fold_id"] = metadata["master_fold_id"].astype(int)
    if tuple(sorted(metadata["source_dataset"].unique().tolist())) != tuple(sorted(EXPECTED_SOURCES)):
        raise ValueError("Unexpected canonical source set")
    if metadata["task_l6_label"].value_counts().to_dict() != EXPECTED_SUBTYPES:
        raise ValueError("Unexpected subtype totals")
    return metadata


def validate_locked_args(args: argparse.Namespace) -> None:
    expected = {
        "hidden_dim": 128,
        "attention_dim": 64,
        "window_size": 4,
        "local_layers": 2,
        "global_layers": 2,
        "attention_heads": 4,
        "dropout": 0.10,
        "epochs": 80,
        "patience": 12,
        "batch_size": 4,
        "num_workers": 0,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "permutation_seed": 20260713,
    }
    if args.max_epochs is None:
        mismatches = {
            key: (getattr(args, key), value)
            for key, value in expected.items()
            if getattr(args, key) != value
        }
        if mismatches:
            raise ValueError(f"Locked primary arguments changed: {mismatches}")
    if args.seed not in (20260713, 20260714):
        raise ValueError("Only the primary and conditional confirmation seeds are allowed")


def validate_assets(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.integrity_manifest).resolve(strict=True)
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != args.expected_integrity_sha256:
        raise ValueError("Integrity manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("case_count") != 591:
        raise ValueError("Integrity manifest is incomplete")
    expected_feature = Path(args.feature_bank_dir).resolve() / "dense_features.float16.npy"
    feature_record = manifest["assets"]["dense_features"]
    if Path(feature_record["path"]).resolve() != expected_feature:
        raise ValueError("Feature path differs from the immutable manifest")
    if expected_feature.stat().st_size != int(feature_record["bytes"]):
        raise ValueError("Feature size differs from the immutable manifest")
    small_assets = ("feature_config", "metadata", "split_csv", "view_bounds", "coordinate_metadata")
    for key in small_assets:
        record = manifest["assets"][key]
        path = Path(record["path"])
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Immutable asset changed: {key}")
    current_code = Path(__file__).resolve()
    matching_code_records = [
        record
        for key, record in manifest["assets"].items()
        if key.startswith("code_") and Path(record["path"]).resolve() == current_code
    ]
    if not matching_code_records or matching_code_records[0]["sha256"] != sha256_file(current_code):
        raise ValueError("Trainer code is not locked in the integrity manifest")
    return manifest


def deterministic_permutations(
    case_ids: Iterable[str],
    num_views: int,
    token_count: int,
    seed: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    result = np.empty((len(case_ids), num_views, token_count), dtype=np.uint16)
    for case_index, case_id in enumerate(case_ids):
        for view_index in range(num_views):
            digest = hashlib.sha256(
                f"{seed}|{case_id}|{view_index}".encode("utf-8")
            ).digest()
            rng_seed = int.from_bytes(digest[:8], "little", signed=False)
            result[case_index, view_index] = np.random.default_rng(rng_seed).permutation(
                token_count
            )
    return result


class DenseCoordinateDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        view_bounds: np.ndarray,
        metadata: pd.DataFrame,
        indices: np.ndarray,
        permutations: np.ndarray | None,
    ) -> None:
        self.features = features
        self.view_bounds = view_bounds
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=int)
        self.permutations = permutations

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        feature = np.array(self.features[index], dtype=np.float16, copy=True)
        if self.permutations is not None:
            feature = np.take_along_axis(
                feature,
                self.permutations[index, :, :, None],
                axis=1,
            ).copy()
        return {
            "feature": torch.from_numpy(feature),
            "view_bounds": torch.from_numpy(
                np.array(self.view_bounds[index], dtype=np.float32, copy=True)
            ),
            "label": torch.tensor(int(self.metadata.iloc[index]["label_idx"]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def source_risk_sampler(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    seed: int,
) -> WeightedRandomSampler:
    subset = metadata.iloc[indices]
    groups = list(zip(subset["source_dataset"].astype(str), subset["label_idx"].astype(int)))
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


class GatedTokenPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.tanh = nn.Linear(hidden_dim, attention_dim)
        self.sigmoid = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = self.score(
            torch.tanh(self.tanh(tokens)) * torch.sigmoid(self.sigmoid(tokens))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=-2)


class MatchedGatedHead(nn.Module):
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
        self.pool = GatedTokenPool(hidden_dim, attention_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, features: torch.Tensor, view_bounds: torch.Tensor) -> torch.Tensor:
        del view_bounds
        batch_size, num_views, token_count, _ = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        pooled = self.pool(tokens.reshape(batch_size, num_views * token_count, -1))
        return self.classifier(self.output_norm(pooled))


class WindowSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, window_size: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        relative_size = 2 * window_size - 1
        self.relative_bias = nn.Parameter(torch.zeros(heads, relative_size * relative_size))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        axis = torch.arange(window_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1).reshape(-1, 2)
        relative = coordinates[:, None, :] - coordinates[None, :, :]
        relative = relative + window_size - 1
        relative_index = relative[..., 0] * relative_size + relative[..., 1]
        self.register_buffer("relative_index", relative_index.long(), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, hidden_dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size, token_count, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias = self.relative_bias[:, self.relative_index.reshape(-1)].reshape(
            self.heads, token_count, token_count
        )
        attention = torch.softmax(attention + bias.unsqueeze(0), dim=-1)
        attention = self.attention_dropout(attention)
        output = torch.matmul(attention, v).transpose(1, 2).reshape(
            batch_size, token_count, hidden_dim
        )
        return self.output_dropout(self.projection(output))


class LocalWindowBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, window_size: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = WindowSelfAttention(hidden_dim, heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class GlobalBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class SpatialRelationalHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        window_size: int,
        local_layers: int,
        global_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.window_size = window_size
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_blocks = nn.ModuleList(
            [
                LocalWindowBlock(hidden_dim, heads, window_size, dropout)
                for _ in range(local_layers)
            ]
        )
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.scale_embedding = nn.Linear(1, hidden_dim)
        self.case_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.case_token, std=0.02)
        self.global_blocks = nn.ModuleList(
            [GlobalBlock(hidden_dim, heads, dropout) for _ in range(global_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        frequencies = torch.arange(1, 17, dtype=torch.float32) * (2.0 * math.pi)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def scalar_embedding(self, values: torch.Tensor) -> torch.Tensor:
        angles = values.unsqueeze(-1) * self.frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def forward(self, features: torch.Tensor, view_bounds: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, token_count, _ = features.shape
        grid_size = math.isqrt(token_count)
        if grid_size != 32 or grid_size * grid_size != token_count:
            raise ValueError(f"Expected a 32x32 patch grid, received {token_count} tokens")
        if grid_size % self.window_size:
            raise ValueError("Patch grid is not divisible by the locked window size")
        windows_per_axis = grid_size // self.window_size
        tokens = self.project(self.input_norm(features))
        tokens = tokens.reshape(
            batch_size,
            num_views,
            windows_per_axis,
            self.window_size,
            windows_per_axis,
            self.window_size,
            self.hidden_dim,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        windows = tokens.reshape(
            batch_size * num_views * windows_per_axis * windows_per_axis,
            self.window_size * self.window_size,
            self.hidden_dim,
        )
        for block in self.local_blocks:
            windows = block(windows)
        windows = windows.mean(dim=1).reshape(
            batch_size,
            num_views,
            windows_per_axis,
            windows_per_axis,
            self.hidden_dim,
        )

        centers = (torch.arange(windows_per_axis, device=features.device, dtype=torch.float32) + 0.5) / windows_per_axis
        local_y, local_x = torch.meshgrid(centers, centers, indexing="ij")
        local_x = local_x.view(1, 1, windows_per_axis, windows_per_axis).expand(
            batch_size, num_views, -1, -1
        )
        local_y = local_y.view(1, 1, windows_per_axis, windows_per_axis).expand_as(local_x)
        x1 = view_bounds[:, :, 0].view(batch_size, num_views, 1, 1)
        y1 = view_bounds[:, :, 1].view(batch_size, num_views, 1, 1)
        width = (view_bounds[:, :, 2] - view_bounds[:, :, 0]).clamp_min(1e-6).view(
            batch_size, num_views, 1, 1
        )
        height = (view_bounds[:, :, 3] - view_bounds[:, :, 1]).clamp_min(1e-6).view(
            batch_size, num_views, 1, 1
        )
        canonical_x = x1 + local_x * width
        canonical_y = y1 + local_y * height
        coordinates = torch.cat(
            (
                self.scalar_embedding(local_x),
                self.scalar_embedding(local_y),
                self.scalar_embedding(canonical_x),
                self.scalar_embedding(canonical_y),
            ),
            dim=-1,
        )
        if coordinates.shape[-1] != self.hidden_dim:
            raise ValueError("Locked coordinate encoding does not match hidden_dim")
        log_scale = (0.5 * (torch.log(width) + torch.log(height))).unsqueeze(-1)
        scale_embedding = self.scale_embedding(log_scale)
        view_embedding = self.view_embeddings[:num_views].view(
            1, num_views, 1, 1, self.hidden_dim
        )
        windows = windows + coordinates + scale_embedding + view_embedding
        windows = windows.reshape(batch_size, num_views * windows_per_axis**2, self.hidden_dim)
        case_token = self.case_token.expand(batch_size, -1, -1)
        global_tokens = torch.cat((case_token, windows), dim=1)
        for block in self.global_blocks:
            global_tokens = block(global_tokens)
        embedding = self.output_norm(global_tokens[:, 0])
        return self.classifier(embedding)


def create_model(args: argparse.Namespace, feature_dim: int, num_views: int) -> nn.Module:
    if args.variant == "matched_gated":
        return MatchedGatedHead(
            feature_dim,
            num_views,
            args.hidden_dim,
            args.attention_dim,
            args.dropout,
        )
    return SpatialRelationalHead(
        feature_dim,
        num_views,
        args.hidden_dim,
        args.window_size,
        args.local_layers,
        args.global_layers,
        args.attention_heads,
        args.dropout,
    )


def fold_partitions(
    metadata: pd.DataFrame,
    split_mode: str,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    fold_values = metadata["master_fold_id"].to_numpy(dtype=int)
    val_fold = (fold_id % 5) + 1
    if split_mode == "source_lodo":
        held_source = EXPECTED_SOURCES[fold_id - 1]
        source_values = metadata["source_dataset"].astype(str).to_numpy()
        test_mask = source_values == held_source
        val_mask = (~test_mask) & (fold_values == val_fold)
        train_mask = (~test_mask) & (~val_mask)
    else:
        held_source = ""
        test_mask = fold_values == fold_id
        val_mask = fold_values == val_fold
        train_mask = ~(test_mask | val_mask)
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    test_indices = np.flatnonzero(test_mask)
    if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
        raise ValueError(f"Empty outer partition for fold {fold_id}")
    return train_indices, val_indices, test_indices, held_source


def make_loader(
    features: np.ndarray,
    view_bounds: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    permutations: np.ndarray | None,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None,
) -> DataLoader:
    dataset = DenseCoordinateDataset(features, view_bounds, metadata, indices, permutations)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


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
    use_amp = device.type == "cuda"
    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True)
        if device.type != "cuda":
            feature = feature.float()
        bounds = batch["view_bounds"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(feature, bounds)
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
    history: pd.DataFrame


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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_epochs = args.epochs if args.max_epochs is None else min(args.epochs, args.max_epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_bacc = -math.inf
    stale = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, total_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            feature = batch["feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                feature = feature.float()
            bounds = batch["view_bounds"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(feature, bounds)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        _, val_labels, val_probability = predict(model, val_loader, device)
        val_metrics = metric_record(val_labels, val_probability)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        print(
            f"[{args.variant}] epoch={epoch} loss={np.mean(losses):.5f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if float(val_metrics["balanced_accuracy"]) > best_bacc + 1e-8:
            best_bacc = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": args.variant,
            "best_epoch": best_epoch,
            "best_val_bacc": best_bacc,
            "state_dict": best_state,
        },
        output_dir / "best_checkpoint.pt",
    )
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "training_history.csv", index=False)
    return TrainedModel(model, best_epoch, best_bacc, history)


def run_fold(
    fold_id: int,
    features: np.ndarray,
    view_bounds: np.ndarray,
    metadata: pd.DataFrame,
    permutations: np.ndarray | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    prediction_path = fold_dir / "test_predictions.csv"
    summary_path = fold_dir / "fold_summary.json"
    if prediction_path.exists() and summary_path.exists():
        print(f"[resume] {args.variant} {args.split_mode} fold {fold_id}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_csv(
            prediction_path, dtype={"case_id": str}
        )

    train_indices, val_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    fold_seed = args.seed + 1000 * fold_id
    set_seed(fold_seed)
    device = torch.device(args.device)
    model = create_model(args, int(features.shape[-1]), int(features.shape[1]))
    sampler = source_risk_sampler(metadata, train_indices, fold_seed + 1)
    train_loader = make_loader(
        features,
        view_bounds,
        metadata,
        train_indices,
        permutations,
        args.batch_size,
        args.num_workers,
        sampler,
    )
    val_loader = make_loader(
        features,
        view_bounds,
        metadata,
        val_indices,
        permutations,
        args.batch_size,
        args.num_workers,
        None,
    )
    test_loader = make_loader(
        features,
        view_bounds,
        metadata,
        test_indices,
        permutations,
        args.batch_size,
        args.num_workers,
        None,
    )
    trained = train_model(model, train_loader, val_loader, fold_dir, args, device)
    predicted_indices, labels, probability = predict(trained.model, test_loader, device)
    if set(predicted_indices.tolist()) != set(test_indices.tolist()):
        raise RuntimeError("Test prediction indices are incomplete")
    order = np.argsort(predicted_indices)
    predicted_indices = predicted_indices[order]
    labels = labels[order]
    probability = probability[order]
    result = metadata.iloc[predicted_indices][
        ["feature_row", "case_id", "source_dataset", "task_l6_label", "label_idx"]
    ].copy()
    if not np.array_equal(labels, result["label_idx"].to_numpy(dtype=int)):
        raise RuntimeError("Predicted labels do not align with metadata")
    result["fold_id"] = fold_id
    result["held_out_source"] = held_source
    result["variant"] = args.variant
    result["prob_high"] = probability
    result["pred_idx"] = (probability >= 0.5).astype(int)
    result["correct"] = result["pred_idx"].eq(result["label_idx"])
    result.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    summary = {
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "variant": args.variant,
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": trained.best_epoch,
        "best_val_bacc": trained.best_val_bacc,
        "parameter_count": int(sum(parameter.numel() for parameter in trained.model.parameters())),
        "test_metrics": metric_record(labels, probability),
    }
    write_json(summary_path, summary)
    del trained, model, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, result


def summarize_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    predictions = predictions.sort_values("feature_row").reset_index(drop=True)
    if len(predictions) != 591 or predictions["case_id"].duplicated().any():
        raise ValueError("Complete OOF predictions must contain 591 unique cases")
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_dir / "overall_metrics.json",
        metric_record(predictions["label_idx"], predictions["prob_high"]),
    )
    source_rows = []
    for source, group in predictions.groupby("source_dataset", sort=False):
        source_rows.append(
            {
                "source_dataset": source,
                **metric_record(group["label_idx"], group["prob_high"]),
            }
        )
    pd.DataFrame(source_rows).to_csv(output_dir / "source_metrics.csv", index=False)
    subtype_rows = []
    for subtype, group in predictions.groupby("task_l6_label", sort=False):
        subtype_rows.append(
            {
                "subtype": subtype,
                "n": int(len(group)),
                "risk_accuracy": float(group["correct"].mean()),
                "predicted_high_n": int(group["pred_idx"].sum()),
            }
        )
    pd.DataFrame(subtype_rows).to_csv(output_dir / "subtype_metrics.csv", index=False)


def main() -> None:
    args = parse_args()
    validate_locked_args(args)
    set_seed(args.seed)
    manifest = validate_assets(args)
    feature_bank_dir = Path(args.feature_bank_dir)
    coordinate_dir = Path(args.coordinate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    metadata = load_metadata(feature_bank_dir, Path(args.split_csv))
    feature_config = json.loads(
        (feature_bank_dir / "feature_bank_config.json").read_text(encoding="utf-8")
    )
    if tuple(feature_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("Feature view order is not locked")
    features = np.load(feature_bank_dir / "dense_features.float16.npy", mmap_mode="r")
    view_bounds = np.load(coordinate_dir / "view_bounds.float32.npy", mmap_mode="r")
    if features.shape != (591, 6, 1024, 1024) or features.dtype != np.float16:
        raise ValueError("Feature bank shape or dtype changed")
    if view_bounds.shape != (591, 6, 4) or view_bounds.dtype != np.float32:
        raise ValueError("Coordinate manifest shape or dtype changed")
    case_id_hash = hashlib.sha256(
        "\n".join(metadata["case_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    if case_id_hash != manifest["case_id_order_sha256"]:
        raise ValueError("Metadata order differs from the immutable manifest")
    permutations = None
    if args.variant == "relational_permuted":
        case_ids = metadata["case_id"].astype(str).tolist()
        permutations = deterministic_permutations(
            case_ids, features.shape[1], features.shape[2], args.permutation_seed
        )
    run_config = vars(args).copy()
    run_config.update(
        {
            "integrity_manifest_actual_sha256": sha256_file(Path(args.integrity_manifest)),
            "feature_shape": list(features.shape),
            "coordinate_shape": list(view_bounds.shape),
            "source_counts": metadata["source_dataset"].value_counts().sort_index().to_dict(),
            "subtype_counts": metadata["task_l6_label"].value_counts().to_dict(),
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in create_model(args, features.shape[-1], features.shape[1]).parameters()
                )
            ),
        }
    )
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise ValueError("Existing output directory has a different locked configuration")
    else:
        write_json(config_path, run_config)
    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    if args.fold != "all":
        requested_fold = int(args.fold)
        if requested_fold not in folds:
            raise ValueError(
                f"Fold {requested_fold} is invalid for split mode {args.split_mode}"
            )
        folds = [requested_fold]
    summaries = []
    prediction_frames = []
    for fold_id in folds:
        summary, frame = run_fold(
            fold_id,
            features,
            view_bounds,
            metadata,
            permutations,
            output_dir,
            args,
        )
        summaries.append(summary)
        prediction_frames.append(frame)
    write_json(output_dir / "fold_summaries.json", summaries)
    if args.fold == "all":
        summarize_predictions(pd.concat(prediction_frames, ignore_index=True), output_dir)
    print(f"[complete] {args.variant} {args.split_mode} {output_dir}", flush=True)


if __name__ == "__main__":
    main()
