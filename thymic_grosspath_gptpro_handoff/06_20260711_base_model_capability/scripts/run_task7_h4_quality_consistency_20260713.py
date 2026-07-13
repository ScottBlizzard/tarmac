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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from run_task7_h3_summary_gated_20260713 import prediction_frame, sha256_file
from run_task7_h3b_masked_gated_20260713 import (
    EXPECTED_VIEWS,
    MaskedDenseGatedHead,
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


CONSISTENCY_WEIGHT = 0.10
AUGMENTATION_PROFILE = "quality_domain_randomization_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked H4 clean/quality-randomized consistency head."
    )
    parser.add_argument("--clean-bank-dir", required=True)
    parser.add_argument("--augmented-bank-dir", required=True)
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


class PairedDenseDataset(Dataset):
    def __init__(
        self,
        clean_features: np.ndarray,
        augmented_features: np.ndarray,
        masks: np.ndarray,
        metadata: pd.DataFrame,
        indices: np.ndarray,
    ) -> None:
        self.clean_features = clean_features
        self.augmented_features = augmented_features
        self.masks = masks
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "clean_feature": torch.from_numpy(
                np.array(self.clean_features[index], dtype=np.float16, copy=True)
            ),
            "augmented_feature": torch.from_numpy(
                np.array(self.augmented_features[index], dtype=np.float16, copy=True)
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
    clean_features: np.ndarray,
    augmented_features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None,
) -> DataLoader:
    return DataLoader(
        PairedDenseDataset(
            clean_features, augmented_features, masks, metadata, indices
        ),
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
    feature_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        feature = batch[feature_key].to(device, non_blocking=True)
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


def jensen_shannon(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    probability_a = torch.softmax(logits_a.float(), dim=-1).clamp_min(1e-7)
    probability_b = torch.softmax(logits_b.float(), dim=-1).clamp_min(1e-7)
    midpoint = ((probability_a + probability_b) / 2.0).clamp_min(1e-7)
    kl_a = (probability_a * (probability_a.log() - midpoint.log())).sum(dim=-1)
    kl_b = (probability_b * (probability_b.log() - midpoint.log())).sum(dim=-1)
    return ((kl_a + kl_b) / 2.0).mean()


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
        total_losses: list[float] = []
        clean_losses: list[float] = []
        augmented_losses: list[float] = []
        consistency_losses: list[float] = []
        for batch in train_loader:
            clean_feature = batch["clean_feature"].to(device, non_blocking=True)
            augmented_feature = batch["augmented_feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                clean_feature = clean_feature.float()
                augmented_feature = augmented_feature.float()
            mask = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                clean_logits = model(clean_feature, mask)
                augmented_logits = model(augmented_feature, mask)
                clean_loss = F.cross_entropy(clean_logits, labels)
                augmented_loss = F.cross_entropy(augmented_logits, labels)
                consistency_loss = jensen_shannon(clean_logits, augmented_logits)
                loss = (
                    0.5 * (clean_loss + augmented_loss)
                    + CONSISTENCY_WEIGHT * consistency_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_losses.append(float(loss.detach().cpu()))
            clean_losses.append(float(clean_loss.detach().cpu()))
            augmented_losses.append(float(augmented_loss.detach().cpu()))
            consistency_losses.append(float(consistency_loss.detach().cpu()))

        _, val_labels, val_probability = predict(
            model, val_loader, device, "clean_feature"
        )
        val_bacc = float(metric_record(val_labels, val_probability)["balanced_accuracy"])
        row = {
            "epoch": epoch,
            "loss": float(np.mean(total_losses)),
            "clean_ce": float(np.mean(clean_losses)),
            "augmented_ce": float(np.mean(augmented_losses)),
            "js_consistency": float(np.mean(consistency_losses)),
            "clean_val_bacc": val_bacc,
        }
        history.append(row)
        print(
            f"[{args.candidate}] epoch={epoch} loss={row['loss']:.5f} "
            f"clean_ce={row['clean_ce']:.5f} aug_ce={row['augmented_ce']:.5f} "
            f"js={row['js_consistency']:.5f} clean_val_bacc={val_bacc:.4f}",
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
    clean_features: np.ndarray,
    augmented_features: np.ndarray,
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
        feature_dim=int(clean_features.shape[-1]),
        num_views=int(clean_features.shape[1]),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    sampler = source_risk_sampler(metadata, train_indices, fold_seed + 1)
    loaders = [
        make_loader(
            clean_features,
            augmented_features,
            masks,
            metadata,
            indices,
            args.batch_size,
            args.num_workers,
            sample,
        )
        for indices, sample in (
            (train_indices, sampler),
            (val_indices, None),
            (test_indices, None),
        )
    ]
    trained = train_model(model, loaders[0], loaders[1], fold_dir, args, device)
    val_index, val_label, val_probability = predict(
        trained.model, loaders[1], device, "clean_feature"
    )
    test_index, test_label, test_probability = predict(
        trained.model, loaders[2], device, "clean_feature"
    )
    aug_val_index, aug_val_label, aug_val_probability = predict(
        trained.model, loaders[1], device, "augmented_feature"
    )
    aug_test_index, aug_test_label, aug_test_probability = predict(
        trained.model, loaders[2], device, "augmented_feature"
    )
    if not (
        np.array_equal(val_index, aug_val_index)
        and np.array_equal(val_label, aug_val_label)
        and np.array_equal(test_index, aug_test_index)
        and np.array_equal(test_label, aug_test_label)
    ):
        raise RuntimeError("Clean and augmented prediction rows do not align")

    val_frame = prediction_frame(
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
    augmented_val_frame = prediction_frame(
        metadata,
        aug_val_index,
        aug_val_label,
        aug_val_probability,
        fold_id,
        held_source,
        f"{args.candidate}_augmented_diagnostic",
        "validation_augmented_diagnostic",
    )
    augmented_test_frame = prediction_frame(
        metadata,
        aug_test_index,
        aug_test_label,
        aug_test_probability,
        fold_id,
        held_source,
        f"{args.candidate}_augmented_diagnostic",
        "test_augmented_diagnostic",
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    val_frame.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_frame.to_csv(test_path, index=False, encoding="utf-8-sig")
    augmented_val_frame.to_csv(
        fold_dir / "augmented_validation_predictions_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    augmented_test_frame.to_csv(
        fold_dir / "augmented_test_predictions_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "candidate": args.candidate,
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": int(trained.best_epoch),
        "best_clean_val_bacc": float(trained.best_val_bacc),
        "clean_validation_metrics": metric_record(val_label, val_probability),
        "clean_test_metrics": metric_record(test_label, test_probability),
        "augmented_validation_metrics_diagnostic": metric_record(
            aug_val_label, aug_val_probability
        ),
        "augmented_test_metrics_diagnostic": metric_record(
            aug_test_label, aug_test_probability
        ),
        "validation_clean_aug_mean_abs_probability_delta": float(
            np.mean(np.abs(val_probability - aug_val_probability))
        ),
        "test_clean_aug_mean_abs_probability_delta": float(
            np.mean(np.abs(test_probability - aug_test_probability))
        ),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    write_json(summary_path, summary)
    return summary, test_frame


def main() -> None:
    args = parse_args()
    if args.seed not in (20260713, 20260714):
        raise ValueError("Only the preregistered primary and confirmation seeds are allowed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    clean_dir = Path(args.clean_bank_dir)
    augmented_dir = Path(args.augmented_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_config_path = clean_dir / "dense_bank_config.json"
    augmented_config_path = augmented_dir / "dense_bank_config.json"
    clean_config = json.loads(clean_config_path.read_text(encoding="utf-8"))
    augmented_config = json.loads(augmented_config_path.read_text(encoding="utf-8"))
    if clean_config.get("complete") is not True or augmented_config.get("complete") is not True:
        raise ValueError("Clean or augmented dense bank is incomplete")
    if tuple(clean_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("Clean view order differs from H4 lock")
    if tuple(augmented_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("Augmented view order differs from H4 lock")
    if augmented_config.get("augmentation_profile_name") != AUGMENTATION_PROFILE:
        raise ValueError("Unexpected H4 augmentation profile")
    if clean_config.get("canonical_model_id") != augmented_config.get("canonical_model_id"):
        raise ValueError("Clean and augmented encoders differ")
    if clean_config.get("weight_sha256") != augmented_config.get("weight_sha256"):
        raise ValueError("Clean and augmented encoder weights differ")

    clean_metadata = load_metadata(clean_dir, Path(args.split_csv))
    augmented_metadata = load_metadata(augmented_dir, Path(args.split_csv))
    metadata_columns = [
        "case_id",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        "master_fold_id",
    ]
    if not clean_metadata[metadata_columns].equals(augmented_metadata[metadata_columns]):
        raise ValueError("Clean and augmented bank metadata do not align")

    clean_features = np.load(clean_dir / "dense_features.float16.npy", mmap_mode="r")
    augmented_features = np.load(
        augmented_dir / "dense_features.float16.npy", mmap_mode="r"
    )
    clean_masks = np.load(clean_dir / "valid_token_mask.uint8.npy", mmap_mode="r")
    augmented_masks = np.load(
        augmented_dir / "valid_token_mask.uint8.npy", mmap_mode="r"
    )
    clean_shapes = np.load(clean_dir / "spatial_shapes.int16.npy", mmap_mode="r")
    augmented_shapes = np.load(
        augmented_dir / "spatial_shapes.int16.npy", mmap_mode="r"
    )
    expected_shape = (591, len(EXPECTED_VIEWS), 1024, 1024)
    if clean_features.shape != expected_shape or augmented_features.shape != expected_shape:
        raise ValueError(
            f"Unexpected clean/augmented shapes: {clean_features.shape}, {augmented_features.shape}"
        )
    if clean_features.dtype != np.float16 or augmented_features.dtype != np.float16:
        raise ValueError("Clean and augmented features must be float16")
    validate_bank(clean_features, clean_masks, clean_shapes)
    validate_bank(augmented_features, augmented_masks, augmented_shapes)
    if not np.array_equal(np.asarray(clean_masks), np.asarray(augmented_masks)):
        raise ValueError("Clean and augmented valid-token masks differ")
    if not np.array_equal(np.asarray(clean_shapes), np.asarray(augmented_shapes)):
        raise ValueError("Clean and augmented spatial shapes differ")
    smoke_delta = float(
        np.mean(
            np.abs(
                np.asarray(clean_features[:2], dtype=np.float32)
                - np.asarray(augmented_features[:2], dtype=np.float32)
            )
        )
    )
    if not np.isfinite(smoke_delta) or smoke_delta <= 1e-5:
        raise ValueError("Clean and augmented features are unexpectedly identical")

    run_config = vars(args).copy()
    run_config.update(
        {
            "clean_feature_shape": list(clean_features.shape),
            "augmented_feature_shape": list(augmented_features.shape),
            "clean_bank_config_sha256": sha256_file(clean_config_path),
            "augmented_bank_config_sha256": sha256_file(augmented_config_path),
            "augmentation_profile": AUGMENTATION_PROFILE,
            "objective": "0.5*(CE_clean+CE_augmented)+0.10*JS(clean,augmented)",
            "consistency_weight": CONSISTENCY_WEIGHT,
            "selection_predictions": "clean_validation_only",
            "final_predictions": "clean_test_only",
            "clean_aug_smoke_mean_abs_feature_delta": smoke_delta,
            "source_counts": clean_metadata["source_dataset"].value_counts().sort_index().to_dict(),
            "subtype_counts": clean_metadata["task_l6_label"].value_counts().to_dict(),
            "expected_sources": list(EXPECTED_SOURCES),
            "selection_threshold": 0.5,
            "coverage": 1.0,
        }
    )
    locked_config = output_dir / "run_config.json"
    if locked_config.exists():
        if json.loads(locked_config.read_text(encoding="utf-8")) != run_config:
            raise ValueError("Existing H4 output has a different configuration")
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
        summary, frame = run_fold(
            fold_id,
            clean_features,
            augmented_features,
            clean_masks,
            clean_metadata,
            output_dir,
            args,
        )
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
