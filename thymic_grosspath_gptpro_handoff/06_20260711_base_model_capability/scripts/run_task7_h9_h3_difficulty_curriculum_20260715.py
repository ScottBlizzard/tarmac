from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import resource
except ImportError:  # Windows-only local smoke tests
    resource = None


PROJECT_SCRIPTS = Path(
    os.environ.get("THYMIC_PROJECT_SCRIPTS", "/workspace/thymic_project/scripts")
)
if not PROJECT_SCRIPTS.is_dir():
    PROJECT_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_SCRIPTS))

from run_task7_h3_summary_gated_20260713 import prediction_frame, sha256_file
from run_task7_h3b_masked_gated_20260713 import (
    MaskedDenseGatedHead,
    make_loader,
    predict,
    validate_bank,
)
from run_task7_spatial_relational_20260713 import (
    EXPECTED_SOURCES,
    fold_partitions,
    load_metadata,
    metric_record,
    set_seed,
    write_json,
)


EXPERIMENT = "H9_H3_DIFFICULTY_BALANCED_CURRICULUM_20260715"
EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
CONFIGURATIONS = (
    "SOURCE_RISK_SUBTYPE_TEMPERED",
    "SOURCE_RISK_SUBTYPE_CURRICULUM",
)
PARAMETER_COUNT = 151107


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked H9 subtype-tempered and dynamic curriculum H3 heads."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-mode", choices=("source_lodo", "fivefold"), required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--attention-dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--minimum-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--grad-clip", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
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


def validate_args(args: argparse.Namespace) -> None:
    locked = {
        "hidden_dim": 128,
        "attention_dim": 64,
        "dropout": 0.10,
        "epochs": 80,
        "patience": 12,
        "minimum_epochs": 24,
        "batch_size": 8,
        "num_workers": 0,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
    }
    mismatch = {
        key: (getattr(args, key), expected)
        for key, expected in locked.items()
        if getattr(args, key) != expected
    }
    if mismatch:
        raise ValueError(f"Locked H9 arguments changed: {mismatch}")
    if args.seed not in {20260713, 20260715}:
        raise ValueError("Only preregistered H9 seeds are permitted")
    if args.max_epochs is not None and os.environ.get("H9_ALLOW_SYNTHETIC_SMOKE") != "1":
        raise ValueError("--max-epochs is restricted to explicit H9 smoke tests")


def median_clip_normalize(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Invalid sampler weights")
    positive = values[values > 0]
    if len(positive) != len(values):
        raise ValueError("Sampler weights must be strictly positive")
    median = float(np.median(positive))
    values = np.clip(values, 0.2 * median, 5.0 * median)
    values /= values.mean()
    return values


def subtype_tempered_weights(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    subset = metadata.iloc[indices][
        ["source_dataset", "label_idx", "task_l6_label"]
    ].reset_index(drop=True)
    weights = np.zeros(len(subset), dtype=np.float64)
    grouped = subset.groupby(["source_dataset", "label_idx"], sort=True)
    if grouped.ngroups == 0:
        raise ValueError("No source-risk cells in H9 training subset")
    cell_mass = 1.0 / grouped.ngroups
    for _, cell in grouped:
        subtype_counts = cell["task_l6_label"].value_counts(sort=False)
        target = np.sqrt(subtype_counts.astype(float))
        target /= target.sum()
        for subtype, count in subtype_counts.items():
            positions = cell.index[cell["task_l6_label"] == subtype].to_numpy(int)
            weights[positions] = cell_mass * float(target[subtype]) / int(count)
    return median_clip_normalize(weights)


def difficulty_from_probability(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    true_probability = np.where(labels == 1, probability, 1.0 - probability)
    result = np.full(len(labels), "hard", dtype="<U6")
    result[true_probability >= 0.50] = "medium"
    result[true_probability >= 0.80] = "easy"
    return result


def curriculum_stage(epoch: int) -> tuple[str, dict[str, float]]:
    if epoch <= 8:
        return "balanced_warmup", {"easy": 1.0, "medium": 1.0, "hard": 1.0}
    if epoch <= 16:
        return "boundary_bridge", {"easy": 1.25, "medium": 1.50, "hard": 0.75}
    return "hard_replay", {"easy": 0.75, "medium": 1.25, "hard": 2.0}


def epoch_weights(
    configuration: str,
    base_weights: np.ndarray,
    difficulty: np.ndarray,
    epoch: int,
) -> tuple[np.ndarray, str]:
    if configuration == "SOURCE_RISK_SUBTYPE_TEMPERED":
        return base_weights.copy(), "static_tempered"
    if configuration != "SOURCE_RISK_SUBTYPE_CURRICULUM":
        raise ValueError(configuration)
    stage, factors = curriculum_stage(epoch)
    multiplier = np.asarray([factors[str(value)] for value in difficulty], dtype=np.float64)
    return median_clip_normalize(base_weights * multiplier), stage


def make_weighted_loader(
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> DataLoader:
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )
    return make_loader(
        features,
        masks,
        metadata,
        indices,
        args.batch_size,
        args.num_workers,
        sampler,
    )


@dataclass
class TrainedCandidate:
    model: nn.Module
    best_epoch: int
    best_val_bacc: float
    elapsed_seconds: float


def train_candidate(
    configuration: str,
    fold_seed: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainedCandidate:
    set_seed(fold_seed)
    model = MaskedDenseGatedHead(
        feature_dim=int(features.shape[-1]),
        num_views=int(features.shape[1]),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PARAMETER_COUNT:
        raise ValueError(f"H9 parameter count changed: {parameter_count}")

    base_weights = subtype_tempered_weights(metadata, train_indices)
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(fold_seed + 1)
    val_loader = make_loader(
        features, masks, metadata, val_indices, args.batch_size, args.num_workers, None
    )
    train_eval_loader = make_loader(
        features, masks, metadata, train_indices, args.batch_size, args.num_workers, None
    )
    train_labels = metadata.iloc[train_indices]["label_idx"].to_numpy(int)
    difficulty = np.full(len(train_indices), "medium", dtype="<U6")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    maximum_epochs = args.epochs if args.max_epochs is None else min(args.epochs, args.max_epochs)
    minimum_epochs = min(args.minimum_epochs, maximum_epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_bacc = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.monotonic()

    for epoch in range(1, maximum_epochs + 1):
        weights, stage = epoch_weights(configuration, base_weights, difficulty, epoch)
        train_loader = make_weighted_loader(
            features,
            masks,
            metadata,
            train_indices,
            weights,
            args,
            sampler_generator,
        )
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            feature = batch["feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                feature = feature.float()
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(feature, valid_mask)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        _, val_labels, val_probability = predict(model, val_loader, device)
        val_bacc = float(metric_record(val_labels, val_probability)["balanced_accuracy"])

        if configuration == "SOURCE_RISK_SUBTYPE_CURRICULUM":
            train_rows, observed_train_labels, train_probability = predict(
                model, train_eval_loader, device
            )
            if not np.array_equal(train_rows, train_indices):
                raise ValueError("H9 train-evaluation order changed")
            if not np.array_equal(observed_train_labels, train_labels):
                raise ValueError("H9 train-evaluation labels changed")
            difficulty = difficulty_from_probability(train_labels, train_probability)

        counts = pd.Series(difficulty).value_counts().to_dict()
        effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "loss": float(np.mean(losses)),
                "val_bacc": val_bacc,
                "easy_n": int(counts.get("easy", 0)),
                "medium_n": int(counts.get("medium", 0)),
                "hard_n": int(counts.get("hard", 0)),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "effective_sample_size": effective_n,
            }
        )
        print(
            f"[H9] {configuration} epoch={epoch} stage={stage} "
            f"loss={np.mean(losses):.5f} val_bacc={val_bacc:.4f} "
            f"difficulty={counts}",
            flush=True,
        )
        if val_bacc > best_bacc + 1e-12:
            best_bacc = val_bacc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= minimum_epochs and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("H9 training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    train_rows, observed_train_labels, final_train_probability = predict(
        model, train_eval_loader, device
    )
    if not np.array_equal(train_rows, train_indices) or not np.array_equal(
        observed_train_labels, train_labels
    ):
        raise ValueError("H9 final train difficulty alignment changed")
    final_difficulty = difficulty_from_probability(train_labels, final_train_probability)
    difficulty_frame = metadata.iloc[train_indices][
        ["case_id", "source_dataset", "task_l6_label", "label_idx"]
    ].copy()
    difficulty_frame["prob_high"] = final_train_probability
    difficulty_frame["difficulty"] = final_difficulty
    difficulty_frame.to_csv(
        output_dir / "final_training_difficulty_server_only.csv",
        index=False,
        encoding="utf-8-sig",
    )
    atomic_torch_save(
        output_dir / "best_head.pt",
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_bacc": best_bacc,
            "configuration": configuration,
        },
    )
    return TrainedCandidate(
        model=model,
        best_epoch=best_epoch,
        best_val_bacc=best_bacc,
        elapsed_seconds=time.monotonic() - started,
    )


def run_fold(
    fold_id: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    train_indices, val_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    summaries: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    fold_seed = args.seed + 1000 * fold_id
    for configuration in CONFIGURATIONS:
        fold_dir = output_dir / configuration / f"fold_{fold_id}"
        summary_path = fold_dir / "fold_summary.json"
        test_path = fold_dir / "test_predictions.csv"
        val_path = fold_dir / "validation_predictions.csv"
        if summary_path.is_file() and test_path.is_file() and val_path.is_file():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            predictions[configuration] = pd.read_csv(
                test_path, dtype={"case_id": str}, encoding="utf-8-sig"
            )
            continue

        print(f"[H9] fold={fold_id} configuration={configuration}", flush=True)
        trained = train_candidate(
            configuration,
            fold_seed,
            features,
            masks,
            metadata,
            train_indices,
            val_indices,
            fold_dir,
            args,
            device,
        )
        val_loader = make_loader(
            features, masks, metadata, val_indices, args.batch_size, args.num_workers, None
        )
        test_loader = make_loader(
            features, masks, metadata, test_indices, args.batch_size, args.num_workers, None
        )
        val_rows, val_labels, val_probability = predict(trained.model, val_loader, device)
        test_rows, test_labels, test_probability = predict(trained.model, test_loader, device)
        val_frame = prediction_frame(
            metadata,
            val_rows,
            val_labels,
            val_probability,
            fold_id,
            held_source,
            configuration,
            "validation",
        )
        test_frame = prediction_frame(
            metadata,
            test_rows,
            test_labels,
            test_probability,
            fold_id,
            held_source,
            configuration,
            "test",
        )
        val_frame.to_csv(val_path, index=False, encoding="utf-8-sig")
        test_frame.to_csv(test_path, index=False, encoding="utf-8-sig")
        summary = {
            "fold_id": fold_id,
            "configuration": configuration,
            "split_mode": args.split_mode,
            "held_out_source": held_source,
            "train_n": int(len(train_indices)),
            "val_n": int(len(val_indices)),
            "test_n": int(len(test_indices)),
            "best_epoch": trained.best_epoch,
            "best_val_bacc": trained.best_val_bacc,
            "elapsed_seconds": trained.elapsed_seconds,
            "validation_metrics": metric_record(val_labels, val_probability),
            "test_metrics": metric_record(test_labels, test_probability),
            "parameter_count": PARAMETER_COUNT,
        }
        write_json(summary_path, summary)
        summaries.append(summary)
        predictions[configuration] = test_frame
        del trained, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries, predictions


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    feature_bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    config_path = feature_bank_dir / "dense_bank_config.json"
    bank_config = json.loads(config_path.read_text(encoding="utf-8"))
    if bank_config.get("complete") is not True:
        raise ValueError("H9 dense bank is incomplete")
    if tuple(bank_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("H9 dense bank view order changed")
    if bank_config.get("canonical_model_id") != "facebook/PE-Spatial-L14-448":
        raise ValueError("H9 dense bank model changed")

    metadata = load_metadata(feature_bank_dir, Path(args.split_csv))
    features = np.load(feature_bank_dir / "dense_features.float16.npy", mmap_mode="r")
    masks = np.load(feature_bank_dir / "valid_token_mask.uint8.npy", mmap_mode="r")
    shapes = np.load(feature_bank_dir / "spatial_shapes.int16.npy", mmap_mode="r")
    if features.shape != (591, 6, 1024, 1024) or features.dtype != np.float16:
        raise ValueError(f"Unexpected H9 dense bank: {features.shape} {features.dtype}")
    if masks.shape != features.shape[:-1] or masks.dtype != np.uint8:
        raise ValueError("Unexpected H9 mask bank")
    validate_bank(features, masks, shapes)

    run_config = vars(args).copy()
    run_config.update(
        {
            "experiment": EXPERIMENT,
            "configurations": list(CONFIGURATIONS),
            "parameter_count": PARAMETER_COUNT,
            "feature_shape": list(features.shape),
            "bank_config_sha256": sha256_file(config_path),
            "source_counts": metadata["source_dataset"].value_counts().sort_index().to_dict(),
            "subtype_counts": metadata["task_l6_label"].value_counts().to_dict(),
            "difficulty_thresholds": {"easy": 0.80, "medium": 0.50},
            "curriculum_stages": {
                "1-8": {"easy": 1.0, "medium": 1.0, "hard": 1.0},
                "9-16": {"easy": 1.25, "medium": 1.50, "hard": 0.75},
                "17-80": {"easy": 0.75, "medium": 1.25, "hard": 2.0},
            },
            "expected_sources": list(EXPECTED_SOURCES),
            "threshold": 0.5,
            "coverage": 1.0,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output = output_dir / "run_config.json"
    if config_output.is_file():
        if json.loads(config_output.read_text(encoding="utf-8")) != run_config:
            raise ValueError("Existing H9 run has a different configuration")
    else:
        write_json(config_output, run_config)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    all_summaries: list[dict[str, Any]] = []
    all_predictions: dict[str, list[pd.DataFrame]] = {
        configuration: [] for configuration in CONFIGURATIONS
    }
    for fold_id in folds:
        summaries, predictions = run_fold(
            fold_id, features, masks, metadata, output_dir, args, device
        )
        all_summaries.extend(summaries)
        for configuration, frame in predictions.items():
            all_predictions[configuration].append(frame)

    for configuration, frames in all_predictions.items():
        oof = pd.concat(frames, ignore_index=True).sort_values("feature_row").reset_index(drop=True)
        if len(oof) != 591 or oof["case_id"].duplicated().any():
            raise ValueError(f"Incomplete H9 OOF predictions for {configuration}")
        oof.to_csv(
            output_dir / configuration / "oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    pd.DataFrame(all_summaries).to_csv(output_dir / "fold_summaries.csv", index=False)
    summary = {
        "experiment": EXPERIMENT,
        "complete": True,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "elapsed_seconds": float(time.monotonic() - started),
        "parameter_count_per_configuration": PARAMETER_COUNT,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_resident_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if resource is not None
        else 0,
    }
    write_json(output_dir / "run_summary.json", summary)
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
        atomic_write_text(
            output_dir / "RUN.status", f"failed: {type(error).__name__}: {error}\n"
        )
        raise


if __name__ == "__main__":
    main()
