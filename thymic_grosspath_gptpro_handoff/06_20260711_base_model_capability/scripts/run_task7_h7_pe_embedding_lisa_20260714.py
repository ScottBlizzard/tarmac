from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from run_task7_h3_summary_gated_20260713 import prediction_frame, sha256_file
from run_task7_h3b_masked_gated_20260713 import (
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
)


EXPERIMENT = "H7_PE_EMBEDDING_LISA_20260714"
EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
EXPECTED_FEATURE_SHAPE = (591, 6, 1024, 1024)
EXPECTED_HASHES = {
    "dense_features": "e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34",
    "valid_token_mask": "af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c",
    "spatial_shapes": "14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f",
}

HIDDEN_DIM = 128
ATTENTION_DIM = 64
DROPOUT = 0.10
EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
MIX_ALPHA = 2.0

MODE_CROSS_SOURCE_SAME_RISK = 0
MODE_WITHIN_SOURCE_OPPOSITE_RISK = 1
MODE_NAMES = {
    MODE_CROSS_SOURCE_SAME_RISK: "cross_source_same_risk",
    MODE_WITHIN_SOURCE_OPPOSITE_RISK: "within_source_opposite_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded post-stop H7 PE-embedding LISA exploration."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--shared-integrity-manifest", required=True)
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Engineering smoke override; never use for reported H7 results.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def set_deterministic(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def validate_shared_integrity(
    manifest_path: Path,
    feature_bank_dir: Path,
    split_csv: Path,
) -> str:
    manifest_hash = sha256_file(manifest_path)
    sidecar = manifest_path.with_name("integrity.sha256")
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing integrity sidecar: {sidecar}")
    expected_manifest_hash = sidecar.read_text(encoding="utf-8").split()[0]
    if manifest_hash != expected_manifest_hash:
        raise ValueError("Shared integrity manifest differs from its sidecar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("case_count") != 591:
        raise ValueError("Shared H3/H6 integrity manifest is incomplete")
    assets = manifest.get("assets", {})
    expected_paths = {
        "dense_features": feature_bank_dir / "dense_features.float16.npy",
        "valid_token_mask": feature_bank_dir / "valid_token_mask.uint8.npy",
        "spatial_shapes": feature_bank_dir / "spatial_shapes.int16.npy",
        "bank_metadata": feature_bank_dir / "metadata.csv",
        "bank_config": feature_bank_dir / "dense_bank_config.json",
        "split_csv": split_csv,
    }
    for key, path in expected_paths.items():
        if key not in assets:
            raise ValueError(f"Shared manifest is missing {key}")
        resolved = path.resolve(strict=True)
        record = assets[key]
        if Path(record["path"]).resolve(strict=True) != resolved:
            raise ValueError(f"Shared manifest path differs for {key}")
        if int(record["bytes"]) != resolved.stat().st_size:
            raise ValueError(f"Shared manifest byte count differs for {key}")
        actual_hash = sha256_file(resolved)
        if actual_hash != record["sha256"]:
            raise ValueError(f"Shared immutable asset changed: {key}")
        if key in EXPECTED_HASHES and actual_hash != EXPECTED_HASHES[key]:
            raise ValueError(f"Regenerated PE bank hash differs for {key}")
    return manifest_hash


def make_loader(
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    sampler: WeightedRandomSampler | None,
) -> DataLoader:
    return DataLoader(
        MaskedDenseDataset(features, masks, metadata, indices),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


class H3LISAHead(nn.Module):
    def __init__(self, feature_dim: int, num_views: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, HIDDEN_DIM))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.pool = MaskedGatedPool(HIDDEN_DIM, ATTENTION_DIM)
        self.output_norm = nn.LayerNorm(HIDDEN_DIM)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 2)
        )

    def embed(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, token_count, _ = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(
            1, num_views, 1, -1
        )
        tokens = tokens.reshape(batch_size, num_views * token_count, -1)
        mask = valid_mask.reshape(batch_size, num_views * token_count)
        return self.output_norm(self.pool(tokens, mask))

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.classify(self.embed(features, valid_mask))


class LISAPartnerSelector:
    def __init__(
        self,
        metadata: pd.DataFrame,
        train_indices: np.ndarray,
        seed: int,
    ) -> None:
        self.metadata = metadata
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.train_set = set(self.train_indices.tolist())
        self.rng = np.random.default_rng(seed)
        self.extra_mode = MODE_CROSS_SOURCE_SAME_RISK
        train = metadata.iloc[self.train_indices]
        self.sources = train["source_dataset"].astype(str).to_numpy()
        self.labels = train["label_idx"].to_numpy(dtype=int)
        self.pools: dict[tuple[int, str, int], np.ndarray] = {}
        for source in sorted(set(self.sources)):
            for risk in (0, 1):
                self.pools[(MODE_CROSS_SOURCE_SAME_RISK, source, risk)] = (
                    self.train_indices[(self.labels == risk) & (self.sources != source)]
                )
                self.pools[(MODE_WITHIN_SOURCE_OPPOSITE_RISK, source, risk)] = (
                    self.train_indices[(self.labels != risk) & (self.sources == source)]
                )
        empty = [key for key, values in self.pools.items() if len(values) == 0]
        if empty:
            raise ValueError(f"LISA partner pool is empty: {empty}")
        self.actual_counts = {name: 0 for name in MODE_NAMES.values()}
        self.pair_counts: dict[str, int] = {}

    def _balanced_modes(self, size: int) -> np.ndarray:
        half = size // 2
        counts = [half, half]
        if size % 2:
            counts[self.extra_mode] += 1
            self.extra_mode = 1 - self.extra_mode
        modes = np.asarray(
            [MODE_CROSS_SOURCE_SAME_RISK] * counts[0]
            + [MODE_WITHIN_SOURCE_OPPOSITE_RISK] * counts[1],
            dtype=np.int64,
        )
        return modes[self.rng.permutation(size)]

    def select(
        self, anchor_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchors = np.asarray(anchor_indices, dtype=np.int64)
        if any(int(index) not in self.train_set for index in anchors):
            raise ValueError("LISA received an anchor outside outer training")
        modes = self._balanced_modes(len(anchors))
        partners = np.empty(len(anchors), dtype=np.int64)
        lambdas = self.rng.beta(MIX_ALPHA, MIX_ALPHA, size=len(anchors)).astype(
            np.float32
        )
        for position, (anchor, mode) in enumerate(zip(anchors, modes, strict=True)):
            row = self.metadata.iloc[int(anchor)]
            source = str(row["source_dataset"])
            risk = int(row["label_idx"])
            candidates = self.pools[(int(mode), source, risk)]
            partner = int(candidates[int(self.rng.integers(len(candidates)))])
            partners[position] = partner
            partner_row = self.metadata.iloc[partner]
            mode_name = MODE_NAMES[int(mode)]
            self.actual_counts[mode_name] += 1
            pair_key = (
                f"{mode_name}|{source}:{risk}->"
                f"{partner_row['source_dataset']}:{int(partner_row['label_idx'])}"
            )
            self.pair_counts[pair_key] = self.pair_counts.get(pair_key, 0) + 1
        return partners, modes, lambdas

    def preflight(self) -> dict[str, Any]:
        return {
            "outer_training_n": int(len(self.train_indices)),
            "pool_count": int(len(self.pools)),
            "minimum_partner_pool_n": int(min(map(len, self.pools.values()))),
            "maximum_partner_pool_n": int(max(map(len, self.pools.values()))),
            "pool_sizes": {
                f"{MODE_NAMES[mode]}|{source}|risk{risk}": int(len(values))
                for (mode, source, risk), values in sorted(self.pools.items())
            },
        }

    def diagnostics(self) -> dict[str, Any]:
        total = sum(self.actual_counts.values())
        return {
            **self.preflight(),
            "actual_anchor_pair_count": int(total),
            "actual_mode_counts": dict(self.actual_counts),
            "actual_cross_source_fraction": (
                float(self.actual_counts[MODE_NAMES[0]] / total) if total else None
            ),
            "actual_pair_counts": dict(sorted(self.pair_counts.items())),
        }


def symmetric_lisa_mix(
    first: torch.Tensor,
    second: torch.Tensor,
    first_labels: torch.Tensor,
    second_labels: torch.Tensor,
    lambdas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("LISA embeddings must be aligned rank-2 tensors")
    if lambdas.ndim != 1 or len(lambdas) != len(first):
        raise ValueError("LISA lambda vector is not batch aligned")
    lam_x = lambdas.to(first.dtype).unsqueeze(1)
    first_targets = F.one_hot(first_labels, num_classes=2).float()
    second_targets = F.one_hot(second_labels, num_classes=2).float()
    lam_y = lambdas.float().unsqueeze(1)
    mixed_embeddings = torch.cat(
        [
            lam_x * first + (1.0 - lam_x) * second,
            lam_x * second + (1.0 - lam_x) * first,
        ],
        dim=0,
    )
    mixed_targets = torch.cat(
        [
            lam_y * first_targets + (1.0 - lam_y) * second_targets,
            lam_y * second_targets + (1.0 - lam_y) * first_targets,
        ],
        dim=0,
    )
    return mixed_embeddings, mixed_targets


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def load_feature_batch(
    features: np.ndarray,
    masks: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature = torch.from_numpy(
        np.array(features[np.asarray(indices, dtype=int)], dtype=np.float16, copy=True)
    ).to(device, non_blocking=True)
    if device.type != "cuda":
        feature = feature.float()
    mask = torch.from_numpy(
        np.array(masks[np.asarray(indices, dtype=int)], dtype=np.uint8, copy=True)
    ).bool().to(device, non_blocking=True)
    return feature, mask


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
    model: H3LISAHead
    best_epoch: int
    best_validation_bacc: float
    best_validation_sensitivity: float
    pairing_diagnostics: dict[str, Any]


def train_model(
    model: H3LISAHead,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    fold_dir: Path,
    seed: int,
    max_epochs: int | None,
    device: torch.device,
) -> TrainedModel:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_epochs = EPOCHS if max_epochs is None else min(EPOCHS, max_epochs)
    selector = LISAPartnerSelector(metadata, train_indices, seed + 2)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_bacc = -math.inf
    best_sensitivity = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, total_epochs + 1):
        model.train()
        losses: list[float] = []
        lambda_values: list[np.ndarray] = []
        mode_counts = {name: 0 for name in MODE_NAMES.values()}
        for batch in train_loader:
            anchor_indices = batch["index"].numpy().astype(int)
            partner_indices, modes, lambda_numpy = selector.select(anchor_indices)
            anchor_feature = batch["feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                anchor_feature = anchor_feature.float()
            anchor_mask = batch["valid_mask"].to(device, non_blocking=True)
            anchor_labels = batch["label"].to(device, non_blocking=True)
            partner_feature, partner_mask = load_feature_batch(
                features, masks, partner_indices, device
            )
            partner_labels = torch.as_tensor(
                metadata.iloc[partner_indices]["label_idx"].to_numpy(dtype=int),
                dtype=torch.long,
                device=device,
            )
            lambdas = torch.as_tensor(lambda_numpy, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                anchor_embedding = model.embed(anchor_feature, anchor_mask)
                partner_embedding = model.embed(partner_feature, partner_mask)
                mixed_embedding, mixed_targets = symmetric_lisa_mix(
                    anchor_embedding,
                    partner_embedding,
                    anchor_labels,
                    partner_labels,
                    lambdas,
                )
                logits = model.classify(mixed_embedding)
                loss = soft_cross_entropy(logits, mixed_targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            lambda_values.append(lambda_numpy)
            for mode, name in MODE_NAMES.items():
                mode_counts[name] += int(np.sum(modes == mode))
        _, validation_labels, validation_probability = predict(
            model, validation_loader, device
        )
        validation_metrics = metric_record(validation_labels, validation_probability)
        validation_bacc = float(validation_metrics["balanced_accuracy"])
        validation_sensitivity = float(validation_metrics["sensitivity"])
        epoch_lambdas = np.concatenate(lambda_values)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_bacc": validation_bacc,
                "validation_sensitivity": validation_sensitivity,
                "lambda_mean": float(np.mean(epoch_lambdas)),
                "lambda_std": float(np.std(epoch_lambdas)),
                **{f"pairs_{key}": value for key, value in mode_counts.items()},
            }
        )
        print(
            f"epoch={epoch} loss={np.mean(losses):.5f} "
            f"val_bacc={validation_bacc:.4f} val_sens={validation_sensitivity:.4f} "
            f"pair_modes={mode_counts}",
            flush=True,
        )
        better = validation_bacc > best_bacc + 1e-8
        tied_better_sensitivity = (
            abs(validation_bacc - best_bacc) <= 1e-8
            and validation_sensitivity > best_sensitivity + 1e-8
        )
        if better or tied_better_sensitivity:
            best_bacc = validation_bacc
            best_sensitivity = validation_sensitivity
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("H7 training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    fold_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(fold_dir / "training_history.csv", pd.DataFrame(history))
    pairing_diagnostics = selector.diagnostics()
    atomic_write_json(fold_dir / "pairing_diagnostics.json", pairing_diagnostics)
    checkpoint = fold_dir / "best_head.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_bacc": best_bacc,
            "best_validation_sensitivity": best_sensitivity,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    return TrainedModel(
        model=model,
        best_epoch=best_epoch,
        best_validation_bacc=best_bacc,
        best_validation_sensitivity=best_sensitivity,
        pairing_diagnostics=pairing_diagnostics,
    )


def completed_fold(
    fold_dir: Path, run_config_hash: str, integrity_hash: str
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    artifact_path = fold_dir / "fold_artifacts.json"
    if not artifact_path.exists():
        return None
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("run_config_sha256") != run_config_hash:
        raise ValueError(f"Completed H7 fold configuration changed: {fold_dir}")
    if artifact.get("integrity_sha256") != integrity_hash:
        raise ValueError(f"Completed H7 fold integrity changed: {fold_dir}")
    for filename, expected_hash in artifact["files"].items():
        path = fold_dir / filename
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"Completed H7 artifact changed: {path}")
    return (
        json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8")),
        pd.read_csv(
            fold_dir / "test_predictions.csv",
            dtype={"case_id": str},
            encoding="utf-8-sig",
        ),
    )


def run_fold(
    fold_id: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    run_config_hash: str,
    integrity_hash: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    recovered = completed_fold(fold_dir, run_config_hash, integrity_hash)
    if recovered is not None:
        print(f"fold={fold_id} status=already_complete", flush=True)
        return recovered
    train_indices, validation_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    fold_seed = args.seed + 1000 * fold_id
    set_deterministic(fold_seed)
    model = H3LISAHead(int(features.shape[-1]), int(features.shape[1]))
    sampler = source_risk_sampler(metadata, train_indices, fold_seed + 1)
    train_loader = make_loader(features, masks, metadata, train_indices, sampler)
    validation_loader = make_loader(
        features, masks, metadata, validation_indices, None
    )
    test_loader = make_loader(features, masks, metadata, test_indices, None)
    device = torch.device(args.device)
    trained = train_model(
        model,
        train_loader,
        validation_loader,
        features,
        masks,
        metadata,
        train_indices,
        fold_dir,
        fold_seed,
        args.max_epochs,
        device,
    )
    validation_index, validation_label, validation_probability = predict(
        trained.model, validation_loader, device
    )
    test_index, test_label, test_probability = predict(
        trained.model, test_loader, device
    )
    candidate = f"h7_pe_embedding_lisa_{args.split_mode}_seed{args.seed}"
    validation_frame = prediction_frame(
        metadata,
        validation_index,
        validation_label,
        validation_probability,
        fold_id,
        held_source,
        candidate,
        "validation",
    )
    test_frame = prediction_frame(
        metadata,
        test_index,
        test_label,
        test_probability,
        fold_id,
        held_source,
        candidate,
        "test",
    )
    summary = {
        "experiment": EXPERIMENT,
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "seed": args.seed,
        "fold_seed": fold_seed,
        "train_n": int(len(train_indices)),
        "validation_n": int(len(validation_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": trained.best_epoch,
        "best_validation_bacc": trained.best_validation_bacc,
        "best_validation_sensitivity": trained.best_validation_sensitivity,
        "validation_metrics": metric_record(validation_label, validation_probability),
        "test_metrics": metric_record(test_label, test_probability),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "pairing_diagnostics": trained.pairing_diagnostics,
    }
    atomic_write_csv(fold_dir / "validation_predictions.csv", validation_frame)
    atomic_write_csv(fold_dir / "test_predictions.csv", test_frame)
    atomic_write_json(fold_dir / "fold_summary.json", summary)
    tracked = (
        "best_head.pt",
        "training_history.csv",
        "pairing_diagnostics.json",
        "validation_predictions.csv",
        "test_predictions.csv",
        "fold_summary.json",
    )
    atomic_write_json(
        fold_dir / "fold_artifacts.json",
        {
            "run_config_sha256": run_config_hash,
            "integrity_sha256": integrity_hash,
            "files": {name: sha256_file(fold_dir / name) for name in tracked},
        },
    )
    del trained, model, train_loader, validation_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, test_frame


def main() -> None:
    args = parse_args()
    if args.seed not in (20260716, 20260717):
        raise ValueError("Only the locked H7 primary and confirmation seeds are allowed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.max_epochs is not None:
        print("WARNING: max-epochs creates a smoke test only", flush=True)
    set_deterministic(args.seed)
    bank_dir = Path(args.feature_bank_dir)
    split_csv = Path(args.split_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    integrity_hash = validate_shared_integrity(
        Path(args.shared_integrity_manifest), bank_dir, split_csv
    )
    bank_config_path = bank_dir / "dense_bank_config.json"
    bank_config = json.loads(bank_config_path.read_text(encoding="utf-8"))
    if bank_config.get("complete") is not True:
        raise ValueError("PE dense bank is incomplete")
    if tuple(bank_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("PE view order differs from H3")
    metadata = load_metadata(bank_dir, split_csv)
    features = np.load(bank_dir / "dense_features.float16.npy", mmap_mode="r")
    masks = np.load(bank_dir / "valid_token_mask.uint8.npy", mmap_mode="r")
    shapes = np.load(bank_dir / "spatial_shapes.int16.npy", mmap_mode="r")
    if features.shape != EXPECTED_FEATURE_SHAPE or features.dtype != np.float16:
        raise ValueError(f"Unexpected PE bank: {features.shape}/{features.dtype}")
    if masks.shape != features.shape[:-1] or masks.dtype != np.uint8:
        raise ValueError("PE valid mask differs from H3")
    validate_bank(features, masks, shapes)
    run_config = {
        "experiment": EXPERIMENT,
        "post_h6_stop_rule_exploration": True,
        "split_mode": args.split_mode,
        "fold": args.fold,
        "seed": args.seed,
        "device": args.device,
        "feature_bank_dir": str(bank_dir.resolve()),
        "split_csv": str(split_csv.resolve()),
        "shared_integrity_manifest": str(
            Path(args.shared_integrity_manifest).resolve()
        ),
        "shared_integrity_sha256": integrity_hash,
        "trainer_sha256": sha256_file(Path(__file__)),
        "feature_shape": list(features.shape),
        "views": list(EXPECTED_VIEWS),
        "hidden_dim": HIDDEN_DIM,
        "attention_dim": ATTENTION_DIM,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "mix_layer": "post_pooling_128d_case_embedding",
        "mix_alpha": MIX_ALPHA,
        "pairing": "50pct_cross_source_same_risk_and_50pct_within_source_opposite_risk",
        "symmetric_mixed_examples": True,
        "original_erm_loss_added": False,
        "sampler": "source_x_risk_inverse_frequency_with_replacement",
        "checkpoint_selection": "validation_bacc_then_sensitivity_then_earlier_epoch",
        "threshold": 0.5,
        "coverage": 1.0,
        "max_epochs_smoke_override": args.max_epochs,
    }
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        if json.loads(run_config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("Existing H7 output has a different immutable configuration")
    else:
        atomic_write_json(run_config_path, run_config)
    run_config_hash = sha256_file(run_config_path)
    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    if args.fold != "all":
        requested = int(args.fold)
        if requested not in folds:
            raise ValueError(f"Invalid fold {requested} for {args.split_mode}")
        folds = [requested]
    summaries = []
    predictions = []
    for fold_id in folds:
        summary, frame = run_fold(
            fold_id,
            features,
            masks,
            metadata,
            output_dir,
            args,
            run_config_hash,
            integrity_hash,
        )
        summaries.append(summary)
        predictions.append(frame)
    atomic_write_json(output_dir / "fold_summaries.json", summaries)
    expected = 3 if args.split_mode == "source_lodo" else 5
    if len(folds) == expected:
        summarize_predictions(pd.concat(predictions, ignore_index=True), output_dir)
        (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    else:
        (output_dir / "RUN.status").write_text("partial\n", encoding="utf-8")


if __name__ == "__main__":
    main()
