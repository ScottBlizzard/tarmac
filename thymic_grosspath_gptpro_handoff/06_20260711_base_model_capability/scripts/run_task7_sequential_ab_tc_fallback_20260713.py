from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DEFAULT_SPLIT_CSV = (
    "/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/"
    "task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
)
SUBTYPES = ("A", "AB", "B1", "B2", "B3", "TC")
CORE_SUBTYPES = {"A", "B1", "B2", "B3"}
LOW_RISK = {"A", "AB", "B1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential AB/TC visual experts with a six-subtype-covered binary fallback."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), default="fivefold")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--anchor-sample-count", type=int, default=40)
    parser.add_argument("--ab-risk-purity", type=float, default=0.98)
    parser.add_argument("--tc-risk-purity", type=float, default=0.95)
    parser.add_argument("--min-route-n", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_summary(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_metadata(feature_bank_dir: Path, split_csv: str) -> pd.DataFrame:
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
    registry_fold = pd.to_numeric(metadata.get("master_fold_id"), errors="coerce")
    locked_fold = pd.to_numeric(authoritative, errors="coerce")
    metadata["master_fold_id"] = locked_fold.fillna(registry_fold)
    if metadata["master_fold_id"].isna().any():
        raise ValueError("Missing locked master_fold_id values")
    metadata["master_fold_id"] = metadata["master_fold_id"].astype(int)
    metadata["task_l6_label"] = metadata["task_l6_label"].astype(str)
    unknown = sorted(set(metadata["task_l6_label"]) - set(SUBTYPES))
    if unknown:
        raise ValueError(f"Unknown subtype labels: {unknown}")
    metadata["risk_label"] = (~metadata["task_l6_label"].isin(LOW_RISK)).astype(int)
    metadata["ab_label"] = (metadata["task_l6_label"] == "AB").astype(int)
    metadata["tc_label"] = (metadata["task_l6_label"] == "TC").astype(int)
    metadata["feature_row"] = np.arange(len(metadata), dtype=int)
    if metadata["case_id"].duplicated().any():
        raise ValueError("Feature metadata must contain one row per case")
    return metadata


class DenseFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, indices: np.ndarray, targets: np.ndarray) -> None:
        self.features = features
        self.indices = np.asarray(indices, dtype=int)
        self.targets = np.asarray(targets, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        feature = torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True))
        return {
            "feature": feature,
            "target": torch.tensor(int(self.targets[index]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class GatedTokenPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.tanh = nn.Linear(hidden_dim, attention_dim)
        self.sigmoid = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(self.tanh(tokens)) * torch.sigmoid(self.sigmoid(tokens))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=-2), weights


class DenseBinaryHead(nn.Module):
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
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, token_count, _ = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        pooled, weights = self.pool(tokens.reshape(batch_size, num_views * token_count, -1))
        return self.head(self.embedding_norm(pooled)), weights


def make_loader(
    features: np.ndarray,
    indices: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    num_workers: int,
    training: bool,
    balance_binary: bool,
) -> DataLoader:
    indices = np.asarray(indices, dtype=int)
    dataset = DenseFeatureDataset(features, indices, targets)
    sampler = None
    shuffle = training
    if training and balance_binary:
        labels = targets[indices]
        counts = np.bincount(labels, minlength=2)
        if np.all(counts > 0):
            weights = 1.0 / counts[labels]
            sampler = WeightedRandomSampler(
                torch.tensor(weights, dtype=torch.double), num_samples=len(indices), replacement=True
            )
            shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def predict(
    model: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    loader = make_loader(
        features,
        indices,
        targets,
        args.batch_size,
        args.num_workers,
        training=False,
        balance_binary=False,
    )
    model.eval()
    rows: list[dict[str, float | int]] = []
    device = torch.device(args.device)
    for batch in loader:
        logits, _ = model(batch["feature"].to(device, non_blocking=True))
        probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for row_index, value in zip(batch["index"].numpy(), probability):
            rows.append({"feature_row": int(row_index), "probability": float(value)})
    return pd.DataFrame(rows)


@dataclass
class TrainedHead:
    model: DenseBinaryHead
    best_epoch: int
    best_val_bacc: float
    history: pd.DataFrame


def build_model(features: np.ndarray, args: argparse.Namespace) -> DenseBinaryHead:
    _, num_views, _, feature_dim = features.shape
    return DenseBinaryHead(
        feature_dim=feature_dim,
        num_views=num_views,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    ).to(torch.device(args.device))


def train_head(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    seed: int,
    balance_binary: bool,
    epoch_indices: Callable[[int], np.ndarray] | None = None,
) -> TrainedHead:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(features, args)
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    max_epochs = int(args.max_epochs or args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_metric = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        selected = np.asarray(epoch_indices(epoch) if epoch_indices else train_indices, dtype=int)
        loader = make_loader(
            features,
            selected,
            targets,
            args.batch_size,
            args.num_workers,
            training=True,
            balance_binary=balance_binary,
        )
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["feature"].to(device, non_blocking=True))
            loss = F.cross_entropy(logits, batch["target"].to(device, non_blocking=True))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val = predict(model, features, val_indices, targets, args)
        val_metric = float(
            balanced_accuracy_score(targets[val["feature_row"].to_numpy(dtype=int)], val["probability"] >= 0.5)
        )
        row = {
            "epoch": epoch,
            "train_n": int(len(selected)),
            "train_loss": float(np.mean(losses)),
            "val_balanced_accuracy": val_metric,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if val_metric > best_metric + 1e-12:
            best_metric = val_metric
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            stale += 1
        print(
            f"[{name}] epoch={epoch} train_n={len(selected)} loss={row['train_loss']:.4f} "
            f"val_bacc={val_metric:.4f} best={best_metric:.4f} stale={stale}",
            flush=True,
        )
        if stale >= args.patience:
            break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    try:
        state = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    return TrainedHead(model, best_epoch, best_metric, history_frame)


def train_fixed_head(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    epochs: int,
) -> DenseBinaryHead:
    set_seed(seed)
    model = build_model(features, args)
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    for epoch in range(1, max(epochs, 1) + 1):
        loader = make_loader(
            features,
            train_indices,
            targets,
            args.batch_size,
            args.num_workers,
            training=True,
            balance_binary=True,
        )
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["feature"].to(device, non_blocking=True))
            loss = F.cross_entropy(logits, batch["target"].to(device, non_blocking=True))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()
        print(f"[{name}] fixed_epoch={epoch}/{epochs}", flush=True)
    return model


def robust_inner_strata(metadata: pd.DataFrame, indices: np.ndarray, targets: np.ndarray, n_splits: int) -> np.ndarray:
    subtype = metadata.iloc[indices]["task_l6_label"].astype(str).to_numpy()
    counts = pd.Series(subtype).value_counts()
    if len(counts) and int(counts.min()) >= n_splits:
        return subtype
    labels = targets[indices].astype(str)
    if int(pd.Series(labels).value_counts().min()) < n_splits:
        raise ValueError("Not enough samples per binary target for inner cross-fitting")
    return labels


def inner_oof_probabilities(
    name: str,
    features: np.ndarray,
    metadata: pd.DataFrame,
    targets: np.ndarray,
    eligible_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    epochs: int,
) -> np.ndarray:
    eligible_indices = np.asarray(eligible_indices, dtype=int)
    probabilities = np.full(len(metadata), np.nan, dtype=float)
    strata = robust_inner_strata(metadata, eligible_indices, targets, args.inner_folds)
    splitter = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=seed)
    for inner_fold, (train_position, test_position) in enumerate(
        splitter.split(np.zeros(len(eligible_indices)), strata), start=1
    ):
        inner_train = eligible_indices[train_position]
        inner_test = eligible_indices[test_position]
        model = train_fixed_head(
            f"{name}-inner{inner_fold}",
            features,
            targets,
            inner_train,
            args,
            seed + inner_fold,
            epochs,
        )
        frame = predict(model, features, inner_test, targets, args)
        probabilities[frame["feature_row"].to_numpy(dtype=int)] = frame["probability"].to_numpy()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if np.isnan(probabilities[eligible_indices]).any():
        raise RuntimeError(f"{name} inner OOF predictions are incomplete")
    return probabilities


def choose_route_threshold(
    probability: np.ndarray,
    risk_label: np.ndarray,
    desired_risk: int,
    minimum_purity: float,
    minimum_n: int,
) -> tuple[float, dict[str, float | int | bool]]:
    probability = np.asarray(probability, dtype=float)
    risk_label = np.asarray(risk_label, dtype=int)
    finite = np.isfinite(probability)
    candidates = np.unique(probability[finite])
    best: tuple[int, float, float] | None = None
    for threshold in candidates:
        routed = finite & (probability >= threshold)
        count = int(routed.sum())
        if count < minimum_n:
            continue
        purity = float(np.mean(risk_label[routed] == desired_risk))
        if purity + 1e-12 < minimum_purity:
            continue
        candidate = (count, -float(threshold), purity)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return float("inf"), {
            "enabled": False,
            "threshold": None,
            "route_n": 0,
            "risk_purity": float("nan"),
        }
    route_n, negative_threshold, purity = best
    threshold = -negative_threshold
    return threshold, {
        "enabled": True,
        "threshold": threshold,
        "route_n": route_n,
        "risk_purity": purity,
    }


def stratified_take(
    candidates: np.ndarray,
    metadata: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=int)
    if count <= 0 or not len(candidates):
        return np.empty(0, dtype=int)
    if len(candidates) <= count:
        return candidates.copy()
    pools: dict[str, list[int]] = {}
    for index in candidates:
        group = str(metadata.iloc[int(index)]["source_dataset"])
        pools.setdefault(group, []).append(int(index))
    for values in pools.values():
        rng.shuffle(values)
    selected: list[int] = []
    group_order = sorted(pools)
    while len(selected) < count:
        progress = False
        for group in group_order:
            if pools[group] and len(selected) < count:
                selected.append(pools[group].pop())
                progress = True
        if not progress:
            break
    return np.asarray(selected, dtype=int)


def sample_anchor_cases(
    all_candidates: np.ndarray,
    missed_candidates: np.ndarray,
    metadata: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    all_candidates = np.asarray(all_candidates, dtype=int)
    count = min(int(count), len(all_candidates))
    hard_target = min(count // 2, len(missed_candidates))
    selected_hard = stratified_take(missed_candidates, metadata, hard_target, rng)
    remaining = np.setdiff1d(all_candidates, selected_hard, assume_unique=False)
    selected_random = stratified_take(remaining, metadata, count - len(selected_hard), rng)
    return np.concatenate([selected_hard, selected_random]).astype(int)


def fold_partitions(
    metadata: pd.DataFrame,
    split_mode: str,
    fold_id: int,
    group_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    val_fold = (fold_id % 5) + 1
    fold_values = metadata["master_fold_id"].to_numpy(dtype=int)
    if split_mode == "source_lodo":
        held_source = group_names[fold_id - 1]
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
        raise ValueError(f"Empty partition for fold {fold_id}")
    return train_indices, val_indices, test_indices, held_source


def probability_array(frame: pd.DataFrame, length: int) -> np.ndarray:
    values = np.full(length, np.nan, dtype=float)
    values[frame["feature_row"].to_numpy(dtype=int)] = frame["probability"].to_numpy(dtype=float)
    return values


def summarize_fold_predictions(frame: pd.DataFrame) -> dict[str, object]:
    risk = frame["risk_label"].to_numpy(dtype=int)
    final_metrics = metric_summary(risk, frame["prob_high"].to_numpy())
    fallback_metrics = metric_summary(risk, frame["prob_fallback"].to_numpy())
    residual = frame["task_l6_label"].isin(CORE_SUBTYPES)
    residual_metrics = metric_summary(
        frame.loc[residual, "risk_label"], frame.loc[residual, "prob_fallback"]
    )
    ab_exclusive = frame["route_decision"] == "ab"
    tc_exclusive = frame["route_decision"] == "tc"
    ab_cases = frame["task_l6_label"] == "AB"
    tc_cases = frame["task_l6_label"] == "TC"
    return {
        "final": final_metrics,
        "fallback_all": fallback_metrics,
        "fallback_core_four": residual_metrics,
        "routing": {
            "ab_exclusive_n": int(ab_exclusive.sum()),
            "ab_exclusive_low_risk_purity": float(
                np.mean(frame.loc[ab_exclusive, "risk_label"] == 0)
            ) if ab_exclusive.any() else float("nan"),
            "ab_target_recall": float(np.mean(ab_exclusive[ab_cases])) if ab_cases.any() else float("nan"),
            "ab_high_risk_absorbed": int(((frame["risk_label"] == 1) & ab_exclusive).sum()),
            "tc_exclusive_n": int(tc_exclusive.sum()),
            "tc_exclusive_high_risk_purity": float(
                np.mean(frame.loc[tc_exclusive, "risk_label"] == 1)
            ) if tc_exclusive.any() else float("nan"),
            "tc_target_recall": float(np.mean(tc_exclusive[tc_cases])) if tc_cases.any() else float("nan"),
            "tc_low_risk_absorbed": int(((frame["risk_label"] == 0) & tc_exclusive).sum()),
            "conflict_n": int((frame["route_decision"] == "fallback_conflict").sum()),
            "fallback_n": int(frame["route_decision"].str.startswith("fallback").sum()),
            "ab_missed_fallback_correct": int(
                (ab_cases & ~ab_exclusive & (frame["pred_idx"] == frame["risk_label"])).sum()
            ),
            "tc_missed_fallback_correct": int(
                (tc_cases & ~tc_exclusive & (frame["pred_idx"] == frame["risk_label"])).sum()
            ),
        },
    }


def train_outer_fold(
    fold_id: int,
    features: np.ndarray,
    metadata: pd.DataFrame,
    group_names: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_indices, val_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id, group_names
    )
    risk_target = metadata["risk_label"].to_numpy(dtype=int)
    ab_target = metadata["ab_label"].to_numpy(dtype=int)
    tc_target = metadata["tc_label"].to_numpy(dtype=int)
    subtype = metadata["task_l6_label"].astype(str).to_numpy()
    seed = args.seed + 1000 * fold_id

    # Stage 1: AB versus all other subtypes.
    ab_head = train_head(
        "E-AB",
        features,
        ab_target,
        train_indices,
        val_indices,
        fold_dir / "e_ab",
        args,
        seed + 10,
        balance_binary=True,
    )
    ab_val_frame = predict(ab_head.model, features, val_indices, ab_target, args)
    ab_val_prob = probability_array(ab_val_frame, len(metadata))
    ab_threshold, ab_threshold_info = choose_route_threshold(
        ab_val_prob[val_indices],
        risk_target[val_indices],
        desired_risk=0,
        minimum_purity=args.ab_risk_purity,
        minimum_n=min(args.min_route_n, len(val_indices)),
    )
    ab_inner_prob = inner_oof_probabilities(
        "E-AB",
        features,
        metadata,
        ab_target,
        train_indices,
        args,
        seed + 100,
        ab_head.best_epoch,
    )
    ab_missed_train = train_indices[
        (subtype[train_indices] == "AB") & (ab_inner_prob[train_indices] < ab_threshold)
    ]

    # Stage 2: TC versus the four core subtypes plus training-side AB misses.
    tc_train_indices = np.concatenate(
        [train_indices[np.isin(subtype[train_indices], list(CORE_SUBTYPES | {"TC"}))], ab_missed_train]
    )
    tc_train_indices = np.unique(tc_train_indices)
    ab_val_route = ab_val_prob[val_indices] >= ab_threshold
    tc_val_indices = val_indices[(subtype[val_indices] != "AB") | (~ab_val_route)]
    tc_head = train_head(
        "E-TC",
        features,
        tc_target,
        tc_train_indices,
        tc_val_indices,
        fold_dir / "e_tc",
        args,
        seed + 20,
        balance_binary=True,
    )
    tc_val_frame = predict(tc_head.model, features, val_indices, tc_target, args)
    tc_val_prob = probability_array(tc_val_frame, len(metadata))
    tc_threshold, tc_threshold_info = choose_route_threshold(
        tc_val_prob[val_indices][~ab_val_route],
        risk_target[val_indices][~ab_val_route],
        desired_risk=1,
        minimum_purity=args.tc_risk_purity,
        minimum_n=min(args.min_route_n, int((~ab_val_route).sum())),
    )
    tc_inner_prob = inner_oof_probabilities(
        "E-TC",
        features,
        metadata,
        tc_target,
        tc_train_indices,
        args,
        seed + 200,
        tc_head.best_epoch,
    )

    # Stage 3: risk fallback over all core cases plus dynamic AB/TC coverage.
    core_train = train_indices[np.isin(subtype[train_indices], list(CORE_SUBTYPES))]
    ab_train = train_indices[subtype[train_indices] == "AB"]
    tc_train = train_indices[subtype[train_indices] == "TC"]
    missed_ab = ab_train[ab_inner_prob[ab_train] < ab_threshold]
    missed_tc = tc_train[
        np.isnan(tc_inner_prob[tc_train]) | (tc_inner_prob[tc_train] < tc_threshold)
    ]

    def fallback_epoch_indices(epoch: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 300 + epoch)
        selected_ab = sample_anchor_cases(
            ab_train, missed_ab, metadata, args.anchor_sample_count, rng
        )
        selected_tc = sample_anchor_cases(
            tc_train, missed_tc, metadata, args.anchor_sample_count, rng
        )
        indices = np.concatenate([core_train, selected_ab, selected_tc])
        rng.shuffle(indices)
        return indices

    fallback_head = train_head(
        "E-Fallback",
        features,
        risk_target,
        fallback_epoch_indices(1),
        val_indices,
        fold_dir / "e_fallback",
        args,
        seed + 30,
        balance_binary=False,
        epoch_indices=fallback_epoch_indices,
    )

    # All heads run on every test case; conflicts fall back to the risk head.
    ab_test = predict(ab_head.model, features, test_indices, ab_target, args)
    tc_test = predict(tc_head.model, features, test_indices, tc_target, args)
    fallback_test = predict(fallback_head.model, features, test_indices, risk_target, args)
    ab_probability = probability_array(ab_test, len(metadata))[test_indices]
    tc_probability = probability_array(tc_test, len(metadata))[test_indices]
    fallback_probability = probability_array(fallback_test, len(metadata))[test_indices]
    route_ab = ab_probability >= ab_threshold
    route_tc = tc_probability >= tc_threshold
    route_decision = np.full(len(test_indices), "fallback_none", dtype=object)
    route_decision[route_ab & ~route_tc] = "ab"
    route_decision[route_tc & ~route_ab] = "tc"
    route_decision[route_ab & route_tc] = "fallback_conflict"
    final_probability = fallback_probability.copy()
    final_probability[route_ab & ~route_tc] = 0.0
    final_probability[route_tc & ~route_ab] = 1.0
    prediction = (final_probability >= 0.5).astype(int)

    result = metadata.iloc[test_indices].copy().reset_index(drop=True)
    result["fold_id"] = fold_id
    result["held_out_source"] = held_source
    result["prob_ab"] = ab_probability
    result["threshold_ab"] = ab_threshold
    result["route_ab"] = route_ab
    result["prob_tc"] = tc_probability
    result["threshold_tc"] = tc_threshold
    result["route_tc"] = route_tc
    result["prob_fallback"] = fallback_probability
    result["route_decision"] = route_decision
    result["prob_high"] = final_probability
    result["pred_idx"] = prediction
    result["correct"] = (prediction == result["risk_label"].to_numpy(dtype=int)).astype(int)
    result.to_csv(fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "ab_best_epoch": ab_head.best_epoch,
        "ab_best_val_bacc": ab_head.best_val_bacc,
        "ab_threshold_validation": ab_threshold_info,
        "ab_inner_missed_n": int(len(ab_missed_train)),
        "tc_train_n": int(len(tc_train_indices)),
        "tc_best_epoch": tc_head.best_epoch,
        "tc_best_val_bacc": tc_head.best_val_bacc,
        "tc_threshold_validation": tc_threshold_info,
        "fallback_best_epoch": fallback_head.best_epoch,
        "fallback_best_val_bacc": fallback_head.best_val_bacc,
        "fallback_ab_missed_pool_n": int(len(missed_ab)),
        "fallback_tc_missed_pool_n": int(len(missed_tc)),
        **summarize_fold_predictions(result),
    }
    write_json(fold_dir / "fold_summary.json", summary)
    del ab_head, tc_head, fallback_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, result


def summarize_complete(predictions: pd.DataFrame, output_dir: Path) -> None:
    overall = summarize_fold_predictions(predictions)
    write_json(output_dir / "summary.json", overall)
    pd.DataFrame([overall["final"]]).to_csv(output_dir / "metrics.csv", index=False)
    subtype_rows = []
    for subtype_name, frame in predictions.groupby("task_l6_label", sort=False):
        values = metric_summary(frame["risk_label"], frame["prob_high"])
        values.update({"subtype": subtype_name, "risk_accuracy": float(frame["correct"].mean())})
        subtype_rows.append(values)
    pd.DataFrame(subtype_rows).to_csv(output_dir / "subtype_metrics.csv", index=False)
    source_rows = []
    for source, frame in predictions.groupby("source_dataset", sort=False):
        values = metric_summary(frame["risk_label"], frame["prob_high"])
        values["source_dataset"] = source
        source_rows.append(values)
    pd.DataFrame(source_rows).to_csv(output_dir / "source_metrics.csv", index=False)
    pd.DataFrame([overall["routing"]]).to_csv(output_dir / "routing_metrics.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.5 <= args.ab_risk_purity <= 1.0:
        raise ValueError("ab-risk-purity must be in [0.5, 1]")
    if not 0.5 <= args.tc_risk_purity <= 1.0:
        raise ValueError("tc-risk-purity must be in [0.5, 1]")
    if args.anchor_sample_count <= 0 or args.inner_folds < 2:
        raise ValueError("anchor-sample-count and inner-folds must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    feature_bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_config = json.loads((feature_bank_dir / "feature_bank_config.json").read_text(encoding="utf-8"))
    if not feature_config.get("complete"):
        raise RuntimeError("Feature bank is incomplete")
    metadata = load_metadata(feature_bank_dir, args.split_csv)
    features = np.load(feature_bank_dir / "dense_features.float16.npy", mmap_mode="r")
    if len(features) != len(metadata):
        raise ValueError("Feature and metadata row counts do not match")
    group_names = sorted(metadata["source_dataset"].astype(str).unique().tolist())
    run_config = vars(args).copy()
    run_config.update(
        {
            "feature_shape": list(features.shape),
            "feature_bank_config": feature_config,
            "group_names": group_names,
            "subtype_counts": metadata["task_l6_label"].value_counts().to_dict(),
            "locked_inference_rule": {
                "ab_only": "low_risk",
                "tc_only": "high_risk",
                "both_or_neither": "fallback",
            },
            "prohibited_inputs": [
                "source_dataset",
                "doctor_text",
                "model_correctness",
                "difficulty",
                "external_labels",
            ],
        }
    )
    write_json(output_dir / "run_config.json", run_config)
    folds = list(range(1, len(group_names) + 1)) if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    if args.fold != "all":
        folds = [int(args.fold)]
    summaries = []
    prediction_frames = []
    for fold_id in folds:
        summary, frame = train_outer_fold(
            fold_id, features, metadata, group_names, output_dir, args
        )
        summaries.append(summary)
        prediction_frames.append(frame)
    write_json(output_dir / "fold_summaries.json", summaries)
    if args.fold == "all":
        summarize_complete(pd.concat(prediction_frames, ignore_index=True), output_dir)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
