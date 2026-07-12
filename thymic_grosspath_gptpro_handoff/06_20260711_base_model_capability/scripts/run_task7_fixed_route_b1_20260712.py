from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_task7_native_detail_a1_20260712 import (  # noqa: E402
    DEFAULT_REGISTRY,
    DEFAULT_SPLIT,
    FeatureDataset,
    NATIVE_VIEW_INDICES,
    NativeDetailHead,
    extract_features_to_ram,
    load_records,
    set_seed,
    write_json,
)


DEFAULT_STAGE1_ROOT = Path(
    "/workspace/thymic_project/experiments/base_model_capability_20260711/"
    "phase2_siglipl512_local_pyramid_screen"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-route same-case B1 visual cascade mechanism pilot.")
    parser.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--split-csv", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", default="old_data,third_batch")
    parser.add_argument("--model-name", default="vit_large_patch16_siglip_512.v2_webli")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fixed-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--route-fraction", type=float, default=0.40)
    parser.add_argument("--split-modes", default="fivefold,source_lodo")
    parser.add_argument(
        "--stage1-fivefold",
        default=str(
            DEFAULT_STAGE1_ROOT
            / "347_siglipl512_localpyramid6_gated_fivefold_cw_20260711"
            / "oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--stage1-lodo",
        default=str(
            DEFAULT_STAGE1_ROOT
            / "348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711"
            / "oof_predictions.csv"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-route-iterations", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def metric_summary(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probability))
        bacc = float(balanced_accuracy_score(y_true, prediction))
    else:
        auc = float("nan")
        bacc = float(accuracy_score(y_true, prediction))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": bacc,
        "auc": auc,
        "sensitivity": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_stage1(path: str, records: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame = frame[["case_id", "prob_high"]].drop_duplicates("case_id")
    aligned = records[["case_id"]].merge(frame, on="case_id", how="left", validate="one_to_one")
    if aligned["prob_high"].isna().any():
        missing = aligned.loc[aligned["prob_high"].isna(), "case_id"].head(10).tolist()
        raise ValueError(f"Stage-1 predictions are missing cases: {missing}")
    return aligned["prob_high"].to_numpy(dtype=float)


def route_threshold(probability: np.ndarray, fraction: float) -> float:
    return float(np.quantile(np.abs(logit(probability)), fraction))


def route_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    return np.abs(logit(probability)) <= threshold


def matched_control_indices(
    records: pd.DataFrame, train_indices: np.ndarray, routed: np.ndarray, margins: np.ndarray
) -> np.ndarray:
    train_frame = records.iloc[train_indices].copy()
    train_frame["index_value"] = train_indices
    train_frame["routed"] = routed
    train_frame["margin"] = margins
    selected: list[int] = []
    group_columns = ["source_dataset", "label_idx", "task_l6_label"]
    for _, routed_group in train_frame[train_frame["routed"]].groupby(group_columns, dropna=False):
        key = tuple(routed_group.iloc[0][column] for column in group_columns)
        candidates = train_frame[~train_frame["routed"]]
        for column, value in zip(group_columns, key):
            candidates = candidates[candidates[column] == value]
        if candidates.empty:
            candidates = train_frame[
                (~train_frame["routed"])
                & (train_frame["source_dataset"] == key[0])
                & (train_frame["label_idx"] == key[1])
            ]
        if candidates.empty:
            continue
        count = min(len(routed_group), len(candidates))
        nearest = candidates.sort_values(["margin", "index_value"]).head(count)
        selected.extend(nearest["index_value"].astype(int).tolist())
    if not selected:
        nonrouted = train_indices[~routed]
        selected = nonrouted[np.argsort(margins[~routed])[: max(1, len(nonrouted) // 3)]].tolist()
    return np.asarray(sorted(set(selected)), dtype=int)


def add_balanced_pool_mass(
    weights: np.ndarray,
    records: pd.DataFrame,
    pool_indices: np.ndarray,
    total_mass: float,
) -> None:
    if not len(pool_indices):
        return
    frame = records.iloc[pool_indices]
    keys = frame["source_dataset"].astype(str) + "|" + frame["label_idx"].astype(str)
    groups = sorted(keys.unique().tolist())
    mass_per_group = total_mass / len(groups)
    for group in groups:
        members = pool_indices[keys.to_numpy() == group]
        weights[members] += mass_per_group / len(members)


def training_sampler(
    records: pd.DataFrame,
    train_indices: np.ndarray,
    stage1_probability: np.ndarray,
    route_fraction: float,
    seed: int,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    threshold = route_threshold(stage1_probability[train_indices], route_fraction)
    margins = np.abs(logit(stage1_probability[train_indices]))
    routed = margins <= threshold
    routed_indices = train_indices[routed]
    matched_indices = matched_control_indices(records, train_indices, routed, margins)
    weights_global = np.zeros(len(records), dtype=np.float64)
    add_balanced_pool_mass(weights_global, records, routed_indices, 0.50)
    add_balanced_pool_mass(weights_global, records, matched_indices, 0.25)
    add_balanced_pool_mass(weights_global, records, train_indices, 0.25)
    local_weights = weights_global[train_indices].copy()
    if not np.isfinite(local_weights).all() or local_weights.sum() <= 0:
        raise ValueError("Invalid stage-2 sampling weights")
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(local_weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )
    return sampler, {
        "route_threshold": threshold,
        "routed_train_cases": int(len(routed_indices)),
        "matched_control_cases": int(len(matched_indices)),
        "full_train_cases": int(len(train_indices)),
    }


def make_loader(
    features: np.ndarray,
    view_metadata: np.ndarray,
    records: pd.DataFrame,
    indices: np.ndarray,
    batch_size: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    dataset = FeatureDataset(features, view_metadata, records, indices, NATIVE_VIEW_INDICES)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def train_image_reader(
    args: argparse.Namespace,
    features: np.ndarray,
    view_metadata: np.ndarray,
    records: pd.DataFrame,
    stage1_probability: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
) -> tuple[NativeDetailHead, list[dict[str, float]], dict[str, Any]]:
    set_seed(seed)
    sampler, sampling_summary = training_sampler(
        records, train_indices, stage1_probability, args.route_fraction, seed
    )
    loader = make_loader(
        features, view_metadata, records, train_indices, args.head_batch_size, sampler
    )
    device = torch.device(args.device)
    model = NativeDetailHead(
        feature_dim=features.shape[-1],
        num_views=len(NATIVE_VIEW_INDICES),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        architecture="hier_mil",
        use_metadata=True,
        specimen_view_index=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fixed_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    history = []
    for epoch in range(1, args.fixed_epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            feature = batch["feature"].to(device, non_blocking=True)
            metadata = batch["view_metadata"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"
            ):
                logits = model(feature, metadata)
                loss = F.cross_entropy(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"]}
        )
    return model, history, sampling_summary


def predict_image_reader(
    args: argparse.Namespace,
    model: NativeDetailHead,
    features: np.ndarray,
    view_metadata: np.ndarray,
    records: pd.DataFrame,
    indices: np.ndarray,
) -> np.ndarray:
    loader = make_loader(features, view_metadata, records, indices, args.head_batch_size)
    device = torch.device(args.device)
    model.eval()
    probabilities = np.empty(len(indices), dtype=float)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            feature = batch["feature"].to(device, non_blocking=True)
            metadata = batch["view_metadata"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"
            ):
                logits = model(feature, metadata)
            values = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            probabilities[cursor : cursor + len(values)] = values
            cursor += len(values)
    return probabilities


def fit_behavior_model(probability: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260712
    )
    model.fit(logit(probability).reshape(-1, 1), labels)
    return model


def fit_fusion_model(
    stage1_probability: np.ndarray, stage2_probability: np.ndarray, labels: np.ndarray
) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260712
    )
    features = np.column_stack([logit(stage1_probability), logit(stage2_probability)])
    model.fit(features, labels)
    return model


def outer_splits(records: pd.DataFrame, split_mode: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    splits = []
    if split_mode == "fivefold":
        for fold in sorted(records["master_fold_id"].unique().astype(int).tolist()):
            test = np.flatnonzero(records["master_fold_id"].to_numpy(dtype=int) == fold)
            train = np.setdiff1d(np.arange(len(records)), test)
            splits.append((str(fold), train, test))
    elif split_mode == "source_lodo":
        sources = records["source_dataset"].astype(str).to_numpy()
        for source in sorted(np.unique(sources).tolist()):
            test = np.flatnonzero(sources == source)
            train = np.flatnonzero(sources != source)
            splits.append((source, train, test))
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    return splits


def inner_crossfit_stage2(
    args: argparse.Namespace,
    features: np.ndarray,
    view_metadata: np.ndarray,
    records: pd.DataFrame,
    stage1_probability: np.ndarray,
    outer_train: np.ndarray,
    outer_key: str,
    split_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    strata = (
        records.iloc[outer_train]["source_dataset"].astype(str)
        + "|"
        + records.iloc[outer_train]["label_idx"].astype(str)
    ).to_numpy()
    splitter = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)
    predictions = np.empty(len(outer_train), dtype=float)
    summaries = []
    for inner_fold, (train_position, test_position) in enumerate(
        splitter.split(np.zeros(len(outer_train)), strata), start=1
    ):
        inner_train = outer_train[train_position]
        inner_test = outer_train[test_position]
        seed_material = f"{split_mode}|{outer_key}|inner{inner_fold}"
        seed = args.seed + int(hashlib.sha256(seed_material.encode()).hexdigest()[:6], 16)
        model, history, sampling = train_image_reader(
            args,
            features,
            view_metadata,
            records,
            stage1_probability,
            inner_train,
            seed,
        )
        predictions[test_position] = predict_image_reader(
            args, model, features, view_metadata, records, inner_test
        )
        summaries.append(
            {
                "inner_fold": inner_fold,
                "train_n": len(inner_train),
                "test_n": len(inner_test),
                "final_train_loss": history[-1]["train_loss"],
                **sampling,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions, summaries


def evaluate_fold(
    args: argparse.Namespace,
    features: np.ndarray,
    view_metadata: np.ndarray,
    records: pd.DataFrame,
    stage1_probability: np.ndarray,
    split_mode: str,
    outer_position: int,
    outer_key: str,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fold_dir = output_dir / split_mode / f"fold_{outer_position + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    inner_stage2, inner_summaries = inner_crossfit_stage2(
        args,
        features,
        view_metadata,
        records,
        stage1_probability,
        outer_train,
        outer_key,
        split_mode,
    )
    labels = records["label_idx"].to_numpy(dtype=int)
    behavior_model = fit_behavior_model(stage1_probability[outer_train], labels[outer_train])
    fusion_model = fit_fusion_model(
        stage1_probability[outer_train], inner_stage2, labels[outer_train]
    )
    seed_material = f"{split_mode}|{outer_key}|final"
    final_seed = args.seed + int(hashlib.sha256(seed_material.encode()).hexdigest()[:6], 16)
    final_model, final_history, final_sampling = train_image_reader(
        args,
        features,
        view_metadata,
        records,
        stage1_probability,
        outer_train,
        final_seed,
    )
    stage2_test = predict_image_reader(
        args, final_model, features, view_metadata, records, outer_test
    )
    stage1_test = stage1_probability[outer_test]
    behavior_test = behavior_model.predict_proba(logit(stage1_test).reshape(-1, 1))[:, 1]
    fusion_test = fusion_model.predict_proba(
        np.column_stack([logit(stage1_test), logit(stage2_test)])
    )[:, 1]
    threshold = route_threshold(stage1_probability[outer_train], args.route_fraction)
    routed = route_mask(stage1_test, threshold)
    frame = records.iloc[outer_test][
        [
            "feature_row",
            "case_id",
            "original_case_id",
            "domain",
            "source_dataset",
            "task_l6_label",
            "label_idx",
            "master_fold_id",
        ]
    ].copy()
    frame["split_mode"] = split_mode
    frame["outer_key"] = outer_key
    frame["route_threshold"] = threshold
    frame["routed"] = routed.astype(int)
    frame["prob_m0_c1"] = stage1_test
    frame["prob_m1_behavior"] = behavior_test
    frame["prob_m2_image"] = stage2_test
    frame["prob_m3_fusion"] = fusion_test
    for model in ["m1_behavior", "m2_image", "m3_fusion"]:
        probability = frame[f"prob_{model}"].to_numpy(dtype=float)
        frame[f"prob_cascade_{model}"] = np.where(routed, probability, stage1_test)
    frame.to_csv(fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final_history).to_csv(fold_dir / "final_stage2_history.csv", index=False)
    pd.DataFrame(inner_summaries).to_csv(fold_dir / "inner_crossfit_summary.csv", index=False)
    torch.save(final_model.state_dict(), fold_dir / "final_stage2_head.pt")
    summary = {
        "split_mode": split_mode,
        "outer_position": outer_position + 1,
        "outer_key": outer_key,
        "train_n": len(outer_train),
        "test_n": len(outer_test),
        "routed_test_n": int(routed.sum()),
        "route_threshold": threshold,
        "inner_folds": args.inner_folds,
        "fixed_epochs": args.fixed_epochs,
        **final_sampling,
    }
    write_json(fold_dir / "fold_config.json", summary)
    metric_rows = []
    for scope, subset in [("full", frame), ("routed", frame[frame["routed"] == 1])]:
        for model, probability_column in [
            ("m0_c1", "prob_m0_c1"),
            ("m1_behavior", "prob_m1_behavior" if scope == "routed" else "prob_cascade_m1_behavior"),
            ("m2_image", "prob_m2_image" if scope == "routed" else "prob_cascade_m2_image"),
            ("m3_fusion", "prob_m3_fusion" if scope == "routed" else "prob_cascade_m3_fusion"),
        ]:
            metric_rows.append(
                {
                    "split_mode": split_mode,
                    "outer_key": outer_key,
                    "scope": scope,
                    "model": model,
                    **metric_summary(subset["label_idx"], subset[probability_column]),
                }
            )
    pd.DataFrame(metric_rows).to_csv(fold_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    del final_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame, metric_rows


def paired_bootstrap(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels = frame["label_idx"].to_numpy(dtype=int)
    baseline = frame[baseline_column].to_numpy(dtype=float)
    candidate = frame[candidate_column].to_numpy(dtype=float)
    strata = [group.index.to_numpy(dtype=int) for _, group in frame.groupby(["source_dataset", "label_idx"])]
    differences = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        differences[iteration] = balanced_accuracy_score(
            labels[sampled], (candidate[sampled] >= 0.5).astype(int)
        ) - balanced_accuracy_score(labels[sampled], (baseline[sampled] >= 0.5).astype(int))
    baseline_metric = metric_summary(labels, baseline)
    candidate_metric = metric_summary(labels, candidate)
    baseline_correct = (baseline >= 0.5).astype(int) == labels
    candidate_correct = (candidate >= 0.5).astype(int) == labels
    return {
        "n": len(frame),
        "baseline_bacc": baseline_metric["balanced_accuracy"],
        "candidate_bacc": candidate_metric["balanced_accuracy"],
        "delta_bacc": float(candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"]),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "rescued": int((~baseline_correct & candidate_correct).sum()),
        "harmed": int((baseline_correct & ~candidate_correct).sum()),
    }


def random_route_analysis(
    frame: pd.DataFrame, iterations: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels = frame["label_idx"].to_numpy(dtype=int)
    baseline = frame["prob_m0_c1"].to_numpy(dtype=float)
    stage2 = frame["prob_m2_image"].to_numpy(dtype=float)
    actual_route = frame["routed"].to_numpy(dtype=bool)
    baseline_bacc = metric_summary(labels, baseline)["balanced_accuracy"]
    actual_probability = np.where(actual_route, stage2, baseline)
    actual_delta = metric_summary(labels, actual_probability)["balanced_accuracy"] - baseline_bacc
    working = frame.copy()
    working["stage1_correct"] = ((working["prob_m0_c1"] >= 0.5).astype(int) == working["label_idx"]).astype(int)
    group_columns = ["outer_key", "source_dataset", "label_idx", "stage1_correct"]
    groups = []
    for _, group in working.groupby(group_columns, dropna=False):
        indices = group.index.to_numpy(dtype=int)
        target_count = int(group["routed"].sum())
        groups.append((indices, target_count))
    random_deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        random_route = np.zeros(len(frame), dtype=bool)
        for indices, count in groups:
            if count:
                random_route[rng.choice(indices, size=count, replace=False)] = True
        probability = np.where(random_route, stage2, baseline)
        random_deltas[iteration] = metric_summary(labels, probability)["balanced_accuracy"] - baseline_bacc
    percentile95 = float(np.quantile(random_deltas, 0.95))
    return {
        "iterations": iterations,
        "actual_route_delta_bacc": float(actual_delta),
        "matched_random_mean_delta_bacc": float(random_deltas.mean()),
        "matched_random_95th_delta_bacc": percentile95,
        "actual_exceeds_random_95th": bool(actual_delta > percentile95),
    }


def summarize_split(
    frame: pd.DataFrame, split_mode: str, args: argparse.Namespace, output_dir: Path
) -> dict[str, Any]:
    metric_rows = []
    model_columns = {
        "m0_c1": "prob_m0_c1",
        "m1_behavior": "prob_cascade_m1_behavior",
        "m2_image": "prob_cascade_m2_image",
        "m3_fusion": "prob_cascade_m3_fusion",
    }
    for group_type, column in [("overall", None), ("source_dataset", "source_dataset"), ("task_l6_label", "task_l6_label")]:
        groups = [("all", frame)] if column is None else frame.groupby(column, dropna=False)
        for group_name, group in groups:
            for model, probability_column in model_columns.items():
                metric_rows.append(
                    {
                        "split_mode": split_mode,
                        "scope": "full",
                        "group_type": group_type,
                        "group": str(group_name),
                        "model": model,
                        **metric_summary(group["label_idx"], group[probability_column]),
                    }
                )
    routed = frame[frame["routed"] == 1].reset_index(drop=True)
    for model, probability_column in {
        "m0_c1": "prob_m0_c1",
        "m1_behavior": "prob_m1_behavior",
        "m2_image": "prob_m2_image",
        "m3_fusion": "prob_m3_fusion",
    }.items():
        metric_rows.append(
            {
                "split_mode": split_mode,
                "scope": "routed",
                "group_type": "overall",
                "group": "all",
                "model": model,
                **metric_summary(routed["label_idx"], routed[probability_column]),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / split_mode / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    comparisons = []
    for scope, subset, columns in [
        ("routed", routed, {"m1_behavior": "prob_m1_behavior", "m2_image": "prob_m2_image", "m3_fusion": "prob_m3_fusion"}),
        ("full", frame.reset_index(drop=True), {"m1_behavior": "prob_cascade_m1_behavior", "m2_image": "prob_cascade_m2_image", "m3_fusion": "prob_cascade_m3_fusion"}),
    ]:
        baseline_column = "prob_m0_c1"
        for model, candidate_column in columns.items():
            result = paired_bootstrap(
                subset,
                baseline_column,
                candidate_column,
                args.bootstrap_iterations,
                args.seed + len(comparisons),
            )
            comparisons.append({"split_mode": split_mode, "scope": scope, "baseline": "m0_c1", "candidate": model, **result})
    routed_m3_vs_m1 = paired_bootstrap(
        routed,
        "prob_m1_behavior",
        "prob_m3_fusion",
        args.bootstrap_iterations,
        args.seed + 99,
    )
    comparisons.append(
        {"split_mode": split_mode, "scope": "routed", "baseline": "m1_behavior", "candidate": "m3_fusion", **routed_m3_vs_m1}
    )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / split_mode / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    random_result = random_route_analysis(frame.reset_index(drop=True), args.random_route_iterations, args.seed)
    write_json(output_dir / split_mode / "matched_random_route_analysis.json", random_result)
    return {
        "metrics": metrics,
        "comparisons": comparison_frame,
        "random_route": random_result,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.route_fraction < 1.0:
        raise ValueError("route_fraction must lie strictly between zero and one")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)
    run_config = vars(args).copy()
    run_config.update(
        {
            "records": len(records),
            "stage2_diagnostic_inputs": ["native image tokens", "tile coordinates", "tile scale", "tissue coverage"],
            "stage2_prohibited_inputs": ["stage1 probability", "confidence", "source", "difficulty", "error target"],
            "stage2_target": "direct Task7 disease label",
            "router": "fixed outer-train 40th percentile of absolute locked-C1 logit margin",
            "stage2_training_mixture": "50% routed + 25% matched controls + 25% full-cohort controls",
            "fusion_policy": "M3 logistic fusion is fit only on inner-crossfit stage2 predictions",
        }
    )
    write_json(output_dir / "run_config.json", run_config)
    records.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    features, view_metadata, feature_config = extract_features_to_ram(args, records, output_dir)
    all_results = {}
    split_modes = [item.strip() for item in args.split_modes.split(",") if item.strip()]
    for split_mode in split_modes:
        split_dir = output_dir / split_mode
        split_dir.mkdir(parents=True, exist_ok=True)
        stage1_path = args.stage1_fivefold if split_mode == "fivefold" else args.stage1_lodo
        stage1_probability = load_stage1(stage1_path, records)
        fold_frames = []
        fold_metrics = []
        for outer_position, (outer_key, outer_train, outer_test) in enumerate(outer_splits(records, split_mode)):
            fold_dir = output_dir / split_mode / f"fold_{outer_position + 1}"
            prediction_path = fold_dir / "test_predictions.csv"
            metric_path = fold_dir / "fold_metrics.csv"
            if args.resume and prediction_path.is_file() and metric_path.is_file():
                cached_frame = pd.read_csv(
                    prediction_path,
                    dtype={"case_id": str, "original_case_id": str, "outer_key": str},
                    encoding="utf-8-sig",
                )
                cached_metrics = pd.read_csv(metric_path, encoding="utf-8-sig")
                expected_cases = set(records.iloc[outer_test]["case_id"].astype(str))
                if set(cached_frame["case_id"].astype(str)) != expected_cases:
                    raise ValueError(f"Resume case mismatch for {split_mode} outer={outer_key}")
                if len(cached_metrics) != 8:
                    raise ValueError(f"Resume metric row mismatch for {split_mode} outer={outer_key}")
                print(f"[resume] split={split_mode} outer={outer_key}", flush=True)
                fold_frames.append(cached_frame)
                fold_metrics.extend(cached_metrics.to_dict(orient="records"))
                continue
            print(
                f"[B1] split={split_mode} outer={outer_key} train={len(outer_train)} test={len(outer_test)}",
                flush=True,
            )
            frame, rows = evaluate_fold(
                args,
                features,
                view_metadata,
                records,
                stage1_probability,
                split_mode,
                outer_position,
                outer_key,
                outer_train,
                outer_test,
                output_dir,
            )
            fold_frames.append(frame)
            fold_metrics.extend(rows)
        predictions = pd.concat(fold_frames, ignore_index=True)
        predictions.to_csv(split_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_metrics).to_csv(split_dir / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
        all_results[split_mode] = summarize_split(predictions, split_mode, args, output_dir)

    if set(split_modes) >= {"fivefold", "source_lodo"}:
        decisions = {}
        for split_mode, result in all_results.items():
            comparisons = result["comparisons"]
            routed_m2 = comparisons[(comparisons["scope"] == "routed") & (comparisons["candidate"] == "m2_image")].iloc[0]
            full_m2 = comparisons[(comparisons["scope"] == "full") & (comparisons["candidate"] == "m2_image")].iloc[0]
            m3_vs_m1 = comparisons[(comparisons["baseline"] == "m1_behavior") & (comparisons["candidate"] == "m3_fusion")].iloc[0]
            decisions[split_mode] = {
                "routed_m2_minus_m0": float(routed_m2["delta_bacc"]),
                "routed_m2_ci_low": float(routed_m2["ci_low"]),
                "routed_m3_minus_m1": float(m3_vs_m1["delta_bacc"]),
                "full_m2_minus_m0": float(full_m2["delta_bacc"]),
                **result["random_route"],
            }
        pass_gate = bool(
            decisions["fivefold"]["routed_m2_minus_m0"] >= 0.06
            and decisions["fivefold"]["routed_m2_ci_low"] > 0.0
            and decisions["fivefold"]["routed_m3_minus_m1"] >= 0.03
            and decisions["fivefold"]["full_m2_minus_m0"] >= 0.02
            and decisions["source_lodo"]["full_m2_minus_m0"] >= 0.015
            and decisions["fivefold"]["actual_exceeds_random_95th"]
        )
        decisions["passes_predeclared_b1_gate"] = pass_gate
        write_json(output_dir / "b1_advancement_decision.json", decisions)
    feature_config["completed_b1"] = True
    write_json(output_dir / "ram_feature_config.json", feature_config)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
