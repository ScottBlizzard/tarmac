from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import math
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


EXPERIMENT = "H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714"
UMBRELLA_CONFIGURATION = "H8_C1_H3_CONCAT_MLP16"
CONFIGURATIONS = (
    "C1_ONLY_PADDED",
    "H3_ONLY_PADDED",
    "C1_H3_EXACT",
    "C1_H3_SAME_SOURCE_DERANGED",
)
SOURCE_ORDER = ("batch1", "batch2", "third_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the locked H8 direct case-embedding heads.")
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--split-mode", choices=("source_lodo", "fivefold"), required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--grad-clip", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=None, help="Synthetic smoke-test override only")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_record(record: dict[str, Any]) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise ValueError(f"Embedding asset changed: {path}")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip().split()[0] != sha256_file(path):
        raise ValueError("Embedding manifest sidecar is missing or invalid")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != EXPERIMENT or manifest.get("complete") is not True:
        raise ValueError("Embedding manifest does not describe a complete H8 extraction")
    if max(manifest["maximum_branch_probability_abs_error"].values()) > float(manifest["reproduction_tolerance"]):
        raise ValueError("Embedding manifest failed branch reproduction")
    return manifest


def canonical_source(value: Any) -> str:
    text = str(value)
    if text in {"batch1", "batch2", "third_batch"}:
        return text
    if text.startswith("third_batch"):
        return "third_batch"
    raise ValueError(f"Unexpected source: {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    locked = {
        "configuration": UMBRELLA_CONFIGURATION,
        "hidden_dim": 16,
        "dropout": 0.10,
        "epochs": 80,
        "patience": 12,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
    }
    mismatches = {
        key: (getattr(args, key), value)
        for key, value in locked.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Locked H8 arguments changed: {mismatches}")
    if args.seed not in {20260714, 20260715}:
        raise ValueError("Only the preregistered H8 seeds are permitted")
    if args.max_epochs is not None and os.environ.get("H8_ALLOW_SYNTHETIC_SMOKE") != "1":
        raise ValueError("--max-epochs is restricted to explicit synthetic smoke tests")


def load_inputs(
    manifest: dict[str, Any], split_csv: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata_path = verify_record(manifest["assets"]["embedding_metadata"])
    embedding_path = verify_record(manifest["assets"]["fold_embeddings"])
    metadata = pd.read_csv(metadata_path, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig")
    metadata["source_dataset"] = metadata["source_dataset"].map(canonical_source)
    split = pd.read_csv(split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split = split[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    joined = metadata[["case_id"]].merge(split, on="case_id", how="left", validate="one_to_one")
    locked_folds = pd.to_numeric(joined["master_fold_id"], errors="raise").astype(int)
    if not np.array_equal(locked_folds.to_numpy(), metadata["master_fold_id"].to_numpy(int)):
        raise ValueError("Embedding metadata no longer matches the locked split")
    with np.load(embedding_path, allow_pickle=False) as values:
        c1 = np.array(values["c1"], dtype=np.float32, copy=True)
        h3 = np.array(values["h3"], dtype=np.float32, copy=True)
    expected_folds = 3 if manifest["split_mode"] == "source_lodo" else 5
    if c1.shape != (expected_folds, 591, 256) or h3.shape != (expected_folds, 591, 128):
        raise ValueError(f"Unexpected embedding shapes: C1={c1.shape}, H3={h3.shape}")
    if len(metadata) != 591 or metadata["case_id"].duplicated().any():
        raise ValueError("Embedding metadata is not a unique 591-case table")
    if not np.isfinite(c1).all() or not np.isfinite(h3).all():
        raise ValueError("Nonfinite frozen embedding")
    return metadata, c1, h3


def l2_normalize(values: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
    return (values / denominator).astype(np.float32)


def fold_partitions(metadata: pd.DataFrame, split_mode: str, fold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    master = metadata["master_fold_id"].to_numpy(int)
    val_fold = (fold_id % 5) + 1
    if split_mode == "source_lodo":
        held_source = SOURCE_ORDER[fold_id - 1]
        source = metadata["source_dataset"].astype(str).to_numpy()
        test_mask = source == held_source
        val_mask = (~test_mask) & (master == val_fold)
        train_mask = (~test_mask) & (~val_mask)
    else:
        held_source = ""
        test_mask = master == fold_id
        val_mask = master == val_fold
        train_mask = ~(test_mask | val_mask)
    train = np.flatnonzero(train_mask)
    val = np.flatnonzero(val_mask)
    test = np.flatnonzero(test_mask)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"Empty H8 partition in fold {fold_id}")
    return train, val, test, held_source


def source_preserving_derangement(
    metadata: pd.DataFrame,
    fold_id: int,
    partitions: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    permutation = np.arange(len(metadata), dtype=int)
    rows: list[dict[str, Any]] = []
    for split_role, indices in partitions.items():
        for source in sorted(metadata.iloc[indices]["source_dataset"].astype(str).unique()):
            group = [int(index) for index in indices if str(metadata.iloc[index]["source_dataset"]) == source]
            if len(group) < 2:
                raise ValueError(f"Derangement group too small: fold={fold_id}, split={split_role}, source={source}")
            ordered = sorted(
                group,
                key=lambda index: hashlib.sha256(
                    f"H8|{fold_id}|{split_role}|{metadata.iloc[index]['case_id']}".encode("utf-8")
                ).hexdigest(),
            )
            shifted = ordered[1:] + ordered[:1]
            for target, donor in zip(ordered, shifted):
                if target == donor:
                    raise ValueError("Derangement retained a same-case pair")
                permutation[target] = donor
                rows.append(
                    {
                        "fold_id": fold_id,
                        "split_role": split_role,
                        "source_dataset": source,
                        "case_id": metadata.iloc[target]["case_id"],
                        "h3_case_id": metadata.iloc[donor]["case_id"],
                    }
                )
    return permutation, pd.DataFrame(rows)


def source_risk_sampler(metadata: pd.DataFrame, indices: np.ndarray, seed: int) -> WeightedRandomSampler:
    subset = metadata.iloc[indices]
    groups = list(zip(subset["source_dataset"].astype(str), subset["label_idx"].astype(int)))
    counts = pd.Series(groups, dtype="object").value_counts().to_dict()
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True, generator=generator)


class DirectFusionHead(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(384, 16)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(16, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(16, 2)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.linear1(features))
        hidden = self.norm(hidden)
        return self.linear2(self.dropout(hidden))


def make_loader(
    x: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    sampler: WeightedRandomSampler | None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x[indices], dtype=np.float32)),
        torch.from_numpy(np.asarray(labels[indices], dtype=np.int64)),
        torch.from_numpy(np.asarray(indices, dtype=np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices = []
    labels = []
    probabilities = []
    for features, target, row_index in loader:
        logits = model(features.to(device, non_blocking=True))
        probability = torch.softmax(logits, dim=-1)[:, 1]
        indices.append(row_index.numpy())
        labels.append(target.numpy())
        probabilities.append(probability.cpu().numpy())
    return (
        np.concatenate(indices).astype(int),
        np.concatenate(labels).astype(int),
        np.concatenate(probabilities).astype(float),
    )


@dataclass
class TrainedHead:
    model: DirectFusionHead
    best_epoch: int
    best_val_bacc: float
    elapsed_seconds: float


def train_head(
    x: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
) -> TrainedHead:
    set_seed(args.seed)
    model = DirectFusionHead(args.dropout).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 6226:
        raise ValueError(f"H8 head parameter count changed: {parameter_count}")
    sampler = source_risk_sampler(metadata, train_indices, args.seed)
    train_loader = make_loader(x, metadata["label_idx"].to_numpy(int), train_indices, args.batch_size, sampler)
    val_loader = make_loader(x, metadata["label_idx"].to_numpy(int), val_indices, args.batch_size, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    max_epochs = args.epochs if args.max_epochs is None else min(args.epochs, args.max_epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_bacc = -math.inf
    best_epoch = 0
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for features, target, _ in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, val_labels, val_probability = predict(model, val_loader, device)
        val_bacc = float(balanced_accuracy_score(val_labels, (val_probability >= 0.5).astype(int)))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_bacc": val_bacc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_bacc > best_bacc + 1e-12:
            best_bacc = val_bacc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("H8 training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    atomic_torch_save(
        fold_dir / "best_head.pt",
        {"state_dict": best_state, "best_epoch": best_epoch, "best_val_bacc": best_bacc},
    )
    return TrainedHead(model, best_epoch, best_bacc, time.monotonic() - started)


def prediction_frame(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    fold_id: int,
    held_source: str,
    configuration: str,
    split_role: str,
) -> pd.DataFrame:
    result = metadata.iloc[indices][
        ["case_id", "original_case_id", "source_dataset", "task_l6_label", "label_idx", "master_fold_id"]
    ].copy()
    if not np.array_equal(result["label_idx"].to_numpy(int), labels):
        raise ValueError("Prediction labels no longer align with metadata")
    result.insert(0, "feature_row", indices)
    result["fold_id"] = fold_id
    result["held_out_source"] = held_source
    result["configuration"] = configuration
    result["split_role"] = split_role
    result["prob_high"] = probability
    result["pred_idx"] = (probability >= 0.5).astype(int)
    result["correct"] = result["pred_idx"].to_numpy(int) == labels
    return result


def build_configuration_input(
    configuration: str,
    c1: np.ndarray,
    h3: np.ndarray,
    derangement: np.ndarray,
) -> np.ndarray:
    zeros_c1 = np.zeros_like(c1)
    zeros_h3 = np.zeros_like(h3)
    if configuration == "C1_ONLY_PADDED":
        return np.concatenate([c1, zeros_h3], axis=1).astype(np.float32)
    if configuration == "H3_ONLY_PADDED":
        return np.concatenate([zeros_c1, h3], axis=1).astype(np.float32)
    if configuration == "C1_H3_EXACT":
        return np.concatenate([c1, h3], axis=1).astype(np.float32)
    if configuration == "C1_H3_SAME_SOURCE_DERANGED":
        return np.concatenate([c1, h3[derangement]], axis=1).astype(np.float32)
    raise ValueError(configuration)


def run_fold(
    fold_id: int,
    metadata: pd.DataFrame,
    c1_fold: np.ndarray,
    h3_fold: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    train_indices, val_indices, test_indices, held_source = fold_partitions(metadata, args.split_mode, fold_id)
    partitions = {"train": train_indices, "validation": val_indices, "test": test_indices}
    derangement, mapping = source_preserving_derangement(metadata, fold_id, partitions)
    mapping_dir = output_dir / "derangements"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_dir / f"fold_{fold_id}.csv", index=False, encoding="utf-8-sig")
    c1 = l2_normalize(c1_fold)
    h3 = l2_normalize(h3_fold)
    summaries = []
    predictions: dict[str, pd.DataFrame] = {}
    for configuration in CONFIGURATIONS:
        print(f"[H8] fold={fold_id} configuration={configuration}", flush=True)
        x = build_configuration_input(configuration, c1, h3, derangement)
        fold_dir = output_dir / configuration / f"fold_{fold_id}"
        trained = train_head(x, metadata, train_indices, val_indices, args, device, fold_dir)
        val_loader = make_loader(x, metadata["label_idx"].to_numpy(int), val_indices, args.batch_size, None)
        test_loader = make_loader(x, metadata["label_idx"].to_numpy(int), test_indices, args.batch_size, None)
        val_index, val_label, val_probability = predict(trained.model, val_loader, device)
        test_index, test_label, test_probability = predict(trained.model, test_loader, device)
        val_frame = prediction_frame(
            metadata, val_index, val_label, val_probability, fold_id, held_source, configuration, "validation"
        )
        test_frame = prediction_frame(
            metadata, test_index, test_label, test_probability, fold_id, held_source, configuration, "test"
        )
        val_frame.to_csv(fold_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        test_frame.to_csv(fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        summary = {
            "fold_id": fold_id,
            "configuration": configuration,
            "held_out_source": held_source,
            "train_n": int(len(train_indices)),
            "val_n": int(len(val_indices)),
            "test_n": int(len(test_indices)),
            "best_epoch": trained.best_epoch,
            "best_val_bacc": trained.best_val_bacc,
            "parameter_count": 6226,
            "elapsed_seconds": trained.elapsed_seconds,
        }
        atomic_write_text(fold_dir / "fold_summary.json", json.dumps(summary, indent=2))
        summaries.append(summary)
        predictions[configuration] = test_frame
        del trained, x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries, predictions


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    set_seed(args.seed)
    manifest = load_manifest(Path(args.embedding_manifest))
    if manifest["split_mode"] != args.split_mode:
        raise ValueError("Embedding and training split modes differ")
    metadata, c1, h3 = load_inputs(manifest, Path(args.split_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment": EXPERIMENT,
        "embedding_manifest_sha256": sha256_file(Path(args.embedding_manifest)),
        **vars(args),
        "configurations": list(CONFIGURATIONS),
        "parameter_count": 6226,
        "threshold": 0.5,
        "coverage": 1.0,
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != run_config:
        raise ValueError("Existing H8 run directory has a different configuration")
    atomic_write_text(config_path, json.dumps(run_config, ensure_ascii=False, indent=2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    folds = range(1, 4) if args.split_mode == "source_lodo" else range(1, 6)
    all_summaries = []
    all_predictions: dict[str, list[pd.DataFrame]] = {key: [] for key in CONFIGURATIONS}
    for fold_id in folds:
        summaries, predictions = run_fold(
            fold_id, metadata, c1[fold_id - 1], h3[fold_id - 1], args, output_dir, device
        )
        all_summaries.extend(summaries)
        for configuration, frame in predictions.items():
            all_predictions[configuration].append(frame)
    for configuration, frames in all_predictions.items():
        oof = pd.concat(frames, ignore_index=True).sort_values("feature_row").reset_index(drop=True)
        if len(oof) != 591 or oof["case_id"].duplicated().any():
            raise ValueError(f"Incomplete H8 OOF predictions for {configuration}")
        config_dir = output_dir / configuration
        oof.to_csv(config_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_summaries).to_csv(output_dir / "fold_summaries.csv", index=False)
    summary = {
        "experiment": EXPERIMENT,
        "complete": True,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "elapsed_seconds": float(time.monotonic() - started),
        "parameter_count_per_configuration": 6226,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_resident_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    atomic_write_text(output_dir / "run_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output_dir / "RUN.status", "complete\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "RUN.status", "running\n")
    try:
        run(args)
    except Exception as error:
        atomic_write_text(output_dir / "RUN.status", f"failed: {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
