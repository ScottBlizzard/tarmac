from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_SCRIPTS = Path(
    os.environ.get("THYMIC_PROJECT_SCRIPTS", "/workspace/thymic_project/scripts")
)
if not PROJECT_SCRIPTS.is_dir():
    PROJECT_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_task7_h10_internal_sampler_20260715 import (
    SAMPLERS,
    build_sampler,
    risk_balanced_weights,
)
from run_task7_h3_summary_gated_20260713 import prediction_frame, sha256_file
from run_task7_h3b_masked_gated_20260713 import (
    MaskedDenseGatedHead,
    make_loader,
    predict,
    validate_bank,
)
from run_task7_spatial_relational_20260713 import (
    fold_partitions,
    load_metadata,
    metric_record,
    set_seed,
    summarize_predictions,
    write_json,
)


EXPERIMENT = "H10_STAGE2_NESTED_PHENOTYPE_CURRICULUM_20260715"
CANDIDATE = "H10_NESTED_PHENOTYPE_CURRICULUM"
EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
PARAMETER_COUNT = 151107
HIGH_RISK_CONCEPTS = [
    "boundary_unclear",
    "capsule_absent",
    "capsule_involved",
    "invasion",
    "fat_involved_or_attached",
    "lung_attached",
    "pericardium_attached",
    "pleura_attached",
    "necrosis",
    "hemorrhage",
    "cystic_change",
    "nodular_lobulated",
    "texture_tough",
]
LOW_RISK_CONCEPTS = [
    "boundary_clear",
    "capsule_complete",
    "texture_soft",
    "homogeneous",
]
ROLE_FACTORS = {
    "anchor_warmup": {
        "canonical_anchor": 1.50,
        "stable_noncanonical": 1.00,
        "learnable_boundary": 0.75,
        "persistent_canonical_failure": 0.75,
        "persistent_mimic_failure": 0.50,
        "persistent_sparse_or_missing": 0.50,
    },
    "boundary_bridge": {
        "canonical_anchor": 1.00,
        "stable_noncanonical": 0.75,
        "learnable_boundary": 1.50,
        "persistent_canonical_failure": 1.00,
        "persistent_mimic_failure": 0.75,
        "persistent_sparse_or_missing": 0.50,
    },
    "targeted_replay": {
        "canonical_anchor": 0.75,
        "stable_noncanonical": 0.50,
        "learnable_boundary": 1.50,
        "persistent_canonical_failure": 1.50,
        "persistent_mimic_failure": 1.00,
        "persistent_sparse_or_missing": 0.50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leak-free nested H10 curriculum.")
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--concept-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--teacher-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--minimum-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    locked = {
        "inner_folds": 3,
        "teacher_epochs": 12,
        "epochs": 80,
        "minimum_epochs": 24,
        "patience": 12,
        "batch_size": 8,
        "num_workers": 0,
        "hidden_dim": 128,
        "attention_dim": 64,
        "dropout": 0.10,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "seed": 20260713,
    }
    smoke = os.environ.get("H10_STAGE2_ALLOW_SMOKE") == "1"
    mismatch = {
        key: (getattr(args, key), value)
        for key, value in locked.items()
        if getattr(args, key) != value
    }
    if mismatch and not smoke:
        raise ValueError(f"Locked H10 Stage-2 arguments changed: {mismatch}")
    if args.max_epochs is not None and not smoke:
        raise ValueError("--max-epochs is restricted to explicit smoke tests")


def new_model(features: np.ndarray, args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = MaskedDenseGatedHead(
        feature_dim=int(features.shape[-1]),
        num_views=int(features.shape[1]),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    ).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PARAMETER_COUNT:
        raise ValueError(f"H10 Stage-2 parameter count changed: {count}")
    return model


def physician_pattern(row: pd.Series) -> str:
    if not bool(row["physician_concept_available"]):
        return "missing"
    has_high = int(row["physician_high_concept_count"]) > 0
    has_low = int(row["physician_low_concept_count"]) > 0
    if has_high and has_low:
        return "mixed"
    if not has_high and not has_low:
        return "uninformative"
    points_high = has_high and not has_low
    return "canonical" if points_high == bool(row["label_idx"]) else "discordant"


def add_physician_patterns(metadata: pd.DataFrame, concept_path: Path) -> pd.DataFrame:
    columns = [
        "original_case_id",
        "gross_highrisk_score",
        *HIGH_RISK_CONCEPTS,
        *LOW_RISK_CONCEPTS,
    ]
    concept = pd.read_csv(
        concept_path,
        usecols=columns,
        dtype={"original_case_id": str},
        encoding="utf-8-sig",
    ).drop_duplicates("original_case_id")
    enriched = metadata.merge(
        concept, on="original_case_id", how="left", sort=False, validate="one_to_one"
    )
    if not enriched["case_id"].equals(metadata["case_id"]):
        raise ValueError("Physician concept merge changed feature order")
    enriched["physician_concept_available"] = enriched["gross_highrisk_score"].notna()
    enriched["physician_high_concept_count"] = enriched[HIGH_RISK_CONCEPTS].fillna(0).sum(axis=1)
    enriched["physician_low_concept_count"] = enriched[LOW_RISK_CONCEPTS].fillna(0).sum(axis=1)
    enriched["physician_pattern"] = enriched.apply(physician_pattern, axis=1)
    if int(enriched["physician_concept_available"].sum()) != 589:
        raise ValueError("Expected physician concepts for 589/591 internal cases")
    return enriched


def diagnostic_role(pattern: str, correct_count: int) -> str:
    if correct_count == 3:
        return "canonical_anchor" if pattern == "canonical" else "stable_noncanonical"
    if correct_count in {1, 2}:
        return "learnable_boundary"
    if pattern == "canonical":
        return "persistent_canonical_failure"
    if pattern in {"mixed", "discordant"}:
        return "persistent_mimic_failure"
    return "persistent_sparse_or_missing"


def train_fixed_teacher(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    final_loss = math.nan
    for _ in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            features = batch["feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                features = features.float()
            masks = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(features, masks)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
    return final_loss


def generate_nested_roles(
    fold_id: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    output_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    if output_path.is_file():
        cached = pd.read_csv(output_path, dtype={"case_id": str}, encoding="utf-8-sig")
        expected_ids = metadata.iloc[train_indices]["case_id"].astype(str).tolist()
        if cached["case_id"].astype(str).tolist() != expected_ids:
            raise ValueError("Cached nested roles do not match outer training order")
        return cached

    outer = metadata.iloc[train_indices].reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=args.inner_folds,
        shuffle=True,
        random_state=args.seed + 10000 * fold_id,
    )
    splits = list(splitter.split(np.zeros(len(outer)), outer["task_l6_label"]))
    teacher_probability: dict[str, np.ndarray] = {}
    teacher_rows = []
    for sampler_index, sampler_name in enumerate(SAMPLERS):
        probability = np.full(len(outer), np.nan, dtype=float)
        for inner_id, (inner_train_rows, inner_test_rows) in enumerate(splits, start=1):
            inner_train = train_indices[np.asarray(inner_train_rows, dtype=int)]
            inner_test = train_indices[np.asarray(inner_test_rows, dtype=int)]
            teacher_seed = (
                args.seed + fold_id * 100000 + sampler_index * 10000 + inner_id * 1000
            )
            set_seed(teacher_seed)
            model = new_model(features, args, device)
            sampler = build_sampler(sampler_name)(metadata, inner_train, teacher_seed + 1)
            train_loader = make_loader(
                features,
                masks,
                metadata,
                inner_train,
                args.batch_size,
                args.num_workers,
                sampler,
            )
            test_loader = make_loader(
                features,
                masks,
                metadata,
                inner_test,
                args.batch_size,
                args.num_workers,
                None,
            )
            started = time.monotonic()
            final_loss = train_fixed_teacher(
                model, train_loader, args.teacher_epochs, args, device
            )
            observed_rows, observed_labels, observed_probability = predict(
                model, test_loader, device
            )
            if not np.array_equal(observed_rows, inner_test):
                raise ValueError("Nested teacher prediction order changed")
            if not np.array_equal(
                observed_labels, metadata.iloc[inner_test]["label_idx"].to_numpy(int)
            ):
                raise ValueError("Nested teacher labels changed")
            probability[np.asarray(inner_test_rows, dtype=int)] = observed_probability
            teacher_rows.append(
                {
                    "outer_fold": fold_id,
                    "sampler": sampler_name,
                    "inner_fold": inner_id,
                    "train_n": int(len(inner_train)),
                    "test_n": int(len(inner_test)),
                    "fixed_epochs": int(args.teacher_epochs),
                    "final_loss": final_loss,
                    "inner_test_bacc": metric_record(
                        observed_labels, observed_probability
                    )["balanced_accuracy"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            del model, train_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not np.isfinite(probability).all():
            raise ValueError(f"Incomplete nested predictions for {sampler_name}")
        teacher_probability[sampler_name] = probability

    role_frame = outer[
        [
            "case_id",
            "original_case_id",
            "task_l6_label",
            "label_idx",
            "physician_pattern",
            "physician_concept_available",
            "physician_high_concept_count",
            "physician_low_concept_count",
        ]
    ].copy()
    labels = role_frame["label_idx"].to_numpy(int)
    correct_columns = []
    for sampler_name in SAMPLERS:
        role_frame[f"teacher_prob_{sampler_name}"] = teacher_probability[sampler_name]
        column = f"teacher_correct_{sampler_name}"
        role_frame[column] = (
            (teacher_probability[sampler_name] >= 0.5).astype(int) == labels
        ).astype(int)
        correct_columns.append(column)
    role_frame["teacher_correct_count"] = role_frame[correct_columns].sum(axis=1)
    role_frame["training_role"] = [
        diagnostic_role(pattern, int(count))
        for pattern, count in zip(
            role_frame["physician_pattern"], role_frame["teacher_correct_count"]
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    role_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(teacher_rows).to_csv(
        output_path.with_name("nested_teacher_fold_metrics.csv"), index=False
    )
    return role_frame


def median_clip_normalize(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Invalid curriculum weights")
    if np.any(values <= 0):
        raise ValueError("Every outer-training case must keep positive weight")
    median = float(np.median(values))
    values = np.clip(values, 0.2 * median, 5.0 * median)
    values /= values.mean()
    return values


def curriculum_stage(epoch: int) -> str:
    if epoch <= 8:
        return "anchor_warmup"
    if epoch <= 16:
        return "boundary_bridge"
    return "targeted_replay"


def curriculum_weights(base: np.ndarray, roles: np.ndarray, epoch: int) -> tuple[np.ndarray, str]:
    stage = curriculum_stage(epoch)
    factors = ROLE_FACTORS[stage]
    multiplier = np.asarray([factors[str(role)] for role in roles], dtype=float)
    return median_clip_normalize(base * multiplier), stage


@dataclass
class TrainedModel:
    model: nn.Module
    best_epoch: int
    best_val_bacc: float
    elapsed_seconds: float


def train_outer_model(
    fold_id: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    roles: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainedModel:
    fold_seed = args.seed + fold_id * 1000
    set_seed(fold_seed)
    model = new_model(features, args, device)
    base_weights = risk_balanced_weights(metadata, train_indices)
    role_values = roles["training_role"].astype(str).to_numpy()
    if len(role_values) != len(train_indices):
        raise ValueError("Nested roles and outer training rows differ")
    val_loader = make_loader(
        features, masks, metadata, val_indices, args.batch_size, args.num_workers, None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    maximum_epochs = args.epochs if args.max_epochs is None else min(args.epochs, args.max_epochs)
    minimum_epochs = min(args.minimum_epochs, maximum_epochs)
    checkpoint_eligible_epoch = minimum_epochs
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_bacc = -math.inf
    stale = 0
    history = []
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(fold_seed + 1)
    started = time.monotonic()

    for epoch in range(1, maximum_epochs + 1):
        weights, stage = curriculum_weights(base_weights, role_values, epoch)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
            generator=sampler_generator,
        )
        train_loader = make_loader(
            features,
            masks,
            metadata,
            train_indices,
            args.batch_size,
            args.num_workers,
            sampler,
        )
        model.train()
        losses = []
        for batch in train_loader:
            batch_features = batch["feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                batch_features = batch_features.float()
            batch_masks = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(batch_features, batch_masks)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        _, val_labels, val_probability = predict(model, val_loader, device)
        val_bacc = float(metric_record(val_labels, val_probability)["balanced_accuracy"])
        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "loss": float(np.mean(losses)),
                "val_bacc": val_bacc,
                "checkpoint_eligible": epoch >= checkpoint_eligible_epoch,
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
            }
        )
        print(
            f"[H10_STAGE2] fold={fold_id} epoch={epoch} stage={stage} "
            f"loss={np.mean(losses):.5f} val_bacc={val_bacc:.4f}",
            flush=True,
        )
        if epoch < checkpoint_eligible_epoch:
            continue
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
        raise RuntimeError("H10 Stage-2 training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_bacc": best_bacc,
            "candidate": CANDIDATE,
            "checkpoint_eligible_epoch": checkpoint_eligible_epoch,
        },
        output_dir / "best_head.pt",
    )
    return TrainedModel(model, best_epoch, best_bacc, time.monotonic() - started)


def run_fold(
    fold_id: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    summary_path = fold_dir / "fold_summary.json"
    test_path = fold_dir / "test_predictions.csv"
    val_path = fold_dir / "validation_predictions.csv"
    if summary_path.is_file() and test_path.is_file() and val_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_csv(
            test_path, dtype={"case_id": str}, encoding="utf-8-sig"
        )

    train_indices, val_indices, test_indices, held_source = fold_partitions(
        metadata, "fivefold", fold_id
    )
    roles = generate_nested_roles(
        fold_id,
        features,
        masks,
        metadata,
        train_indices,
        fold_dir / "nested_training_roles_server_only.csv",
        args,
        device,
    )
    role_counts = roles["training_role"].value_counts().to_dict()
    role_subtypes = pd.crosstab(roles["training_role"], roles["task_l6_label"])
    role_subtypes.to_csv(fold_dir / "nested_role_by_subtype.csv")
    trained = train_outer_model(
        fold_id,
        features,
        masks,
        metadata,
        train_indices,
        val_indices,
        roles,
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
        CANDIDATE,
        "validation",
    )
    test_frame = prediction_frame(
        metadata,
        test_rows,
        test_labels,
        test_probability,
        fold_id,
        held_source,
        CANDIDATE,
        "test",
    )
    val_frame.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_frame.to_csv(test_path, index=False, encoding="utf-8-sig")
    summary = {
        "fold_id": fold_id,
        "candidate": CANDIDATE,
        "split_mode": "fivefold",
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": trained.best_epoch,
        "best_val_bacc": trained.best_val_bacc,
        "elapsed_seconds_final_training": trained.elapsed_seconds,
        "role_counts": {str(key): int(value) for key, value in role_counts.items()},
        "validation_metrics": metric_record(val_labels, val_probability),
        "test_metrics": metric_record(test_labels, test_probability),
        "parameter_count": PARAMETER_COUNT,
    }
    write_json(summary_path, summary)
    del trained, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, test_frame


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    feature_bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = feature_bank_dir / "dense_bank_config.json"
    bank_config = json.loads(config_path.read_text(encoding="utf-8"))
    if bank_config.get("complete") is not True:
        raise ValueError("H10 Stage-2 dense bank is incomplete")
    if tuple(bank_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("H10 Stage-2 dense view order changed")
    metadata = load_metadata(feature_bank_dir, Path(args.split_csv))
    metadata = add_physician_patterns(metadata, Path(args.concept_csv))
    features = np.load(feature_bank_dir / "dense_features.float16.npy", mmap_mode="r")
    masks = np.load(feature_bank_dir / "valid_token_mask.uint8.npy", mmap_mode="r")
    shapes = np.load(feature_bank_dir / "spatial_shapes.int16.npy", mmap_mode="r")
    if features.shape != (591, 6, 1024, 1024) or features.dtype != np.float16:
        raise ValueError(f"Unexpected H10 Stage-2 dense bank: {features.shape} {features.dtype}")
    if masks.shape != features.shape[:-1] or masks.dtype != np.uint8:
        raise ValueError("Unexpected H10 Stage-2 mask")
    validate_bank(features, masks, shapes)

    run_config = vars(args).copy()
    run_config.update(
        {
            "experiment": EXPERIMENT,
            "candidate": CANDIDATE,
            "feature_shape": list(features.shape),
            "bank_config_sha256": sha256_file(config_path),
            "split_sha256": sha256_file(Path(args.split_csv)),
            "concept_sha256": sha256_file(Path(args.concept_csv)),
            "physician_concept_coverage": int(metadata["physician_concept_available"].sum()),
            "role_factors": ROLE_FACTORS,
            "selection_threshold": 0.5,
            "coverage": 1.0,
            "source_used_for_training": False,
        }
    )
    config_output = output_dir / "run_config.json"
    if config_output.is_file():
        if json.loads(config_output.read_text(encoding="utf-8")) != run_config:
            raise ValueError("Existing Stage-2 output has a different configuration")
    else:
        write_json(config_output, run_config)

    folds = [1, 2, 3, 4, 5]
    if args.fold != "all":
        requested = int(args.fold)
        if requested not in folds:
            raise ValueError(f"Invalid H10 Stage-2 fold: {requested}")
        folds = [requested]
    summaries = []
    predictions = []
    for fold_id in folds:
        summary, frame = run_fold(
            fold_id, features, masks, metadata, output_dir, args, device
        )
        summaries.append(summary)
        predictions.append(frame)
    pd.DataFrame(summaries).to_json(
        output_dir / "fold_summaries.json", orient="records", indent=2
    )
    if len(folds) == 5:
        summarize_predictions(pd.concat(predictions, ignore_index=True), output_dir)
        (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    else:
        (output_dir / "RUN.status").write_text("partial\n", encoding="utf-8")


if __name__ == "__main__":
    main()
