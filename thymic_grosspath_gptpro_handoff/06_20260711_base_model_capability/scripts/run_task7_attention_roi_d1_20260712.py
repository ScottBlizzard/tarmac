from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = (
    _SCRIPT_DIR.parent
    if (_SCRIPT_DIR.parent / "thymic_baseline").exists()
    else Path("/workspace/thymic_project")
)
for item in reversed((_SCRIPT_DIR, _PROJECT_ROOT, _PROJECT_ROOT / "scripts")):
    while str(item) in sys.path:
        sys.path.remove(str(item))
    sys.path.insert(0, str(item))

import run_task7_native_detail_a1_20260712 as a1  # noqa: E402
from extract_task7_dense_token_bank_20260711 import (  # noqa: E402
    create_backbone,
    extract_feature_tokens,
    model_image_size,
    model_normalization,
)


BASE_VIEW_NAMES = ["whole", "c1_crop", "c1_q0", "c1_q1", "c1_q2", "c1_q3"]
Q70_WINDOWS = [
    (0.0, 0.0, 0.7, 0.7),
    (0.3, 0.0, 1.0, 0.7),
    (0.0, 0.3, 0.7, 1.0),
    (0.3, 0.3, 1.0, 1.0),
]
VARIANTS = ("base_only", "attention_roi", "matched_random_roi")


@dataclass(frozen=True)
class CaseRoiPlan:
    attention: tuple[a1.TileWindow, ...]
    random: tuple[a1.TileWindow, ...]
    fallback_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe weakly supervised attention ROI direct-model experiment."
    )
    parser.add_argument("--registry-csv", default=str(a1.DEFAULT_REGISTRY))
    parser.add_argument("--split-csv", default=str(a1.DEFAULT_SPLIT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", default="old_data,third_batch")
    parser.add_argument("--model-name", default="vit_large_patch16_siglip_512.v2_webli")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--selector-epochs", type=int, default=60)
    parser.add_argument("--selector-patience", type=int, default=10)
    parser.add_argument("--diagnostic-epochs", type=int, default=80)
    parser.add_argument("--diagnostic-patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--split-modes", default="fivefold,source_lodo")
    parser.add_argument("--roi-centers", type=int, default=2)
    parser.add_argument("--roi-scales", default="0.40,0.25")
    parser.add_argument("--minimum-tissue", type=float, default=0.60)
    parser.add_argument("--minimum-center-distance", type=float, default=0.20)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class BaseViewDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        views, metadata = a1.build_all_views(image)
        tensors = torch.stack([self.transform(view) for view in views[:6]], dim=0)
        return index, tensors, torch.from_numpy(metadata[:6])


def extract_base_features(
    args: argparse.Namespace, records: pd.DataFrame, output_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    device = torch.device(args.device)
    model = create_backbone(args.model_name, "dense").eval().to(device)
    image_size = model_image_size(model, args.image_size)
    mean, std = model_normalization(model)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            dummy_tokens = extract_feature_tokens(model, dummy, "dense")
    token_count, feature_dim = map(int, dummy_tokens.shape[1:])
    shape = (len(records), len(BASE_VIEW_NAMES), token_count, feature_dim)
    estimated_gib = float(np.prod(shape) * np.dtype(np.float16).itemsize / (1024**3))
    print(f"[base-ram-bank] shape={shape} estimated_gib={estimated_gib:.2f}", flush=True)
    features = np.empty(shape, dtype=np.float16)
    view_metadata = np.empty((len(records), len(BASE_VIEW_NAMES), 4), dtype=np.float32)
    loader_kwargs: dict[str, Any] = {
        "dataset": BaseViewDataset(records, transform),
        "batch_size": args.extract_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    progress = tqdm(loader, desc="D1 base tokens", dynamic_ncols=True, file=sys.stdout)
    for row_indices, view_batch, metadata_batch in progress:
        batch_size, num_views, channels, height, width = view_batch.shape
        inputs = view_batch.reshape(batch_size * num_views, channels, height, width).to(
            device, non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            tokens = extract_feature_tokens(model, inputs, "dense")
        tokens = tokens.reshape(batch_size, num_views, token_count, feature_dim)
        indices = row_indices.numpy().astype(int)
        features[indices] = tokens.detach().cpu().numpy().astype(np.float16, copy=False)
        view_metadata[indices] = metadata_batch.numpy().astype(np.float32, copy=False)
        progress.set_postfix(cases=int(indices[-1]) + 1, total=len(records))
    progress.close()
    config = {
        "model_name": args.model_name,
        "image_size": image_size,
        "view_names": BASE_VIEW_NAMES,
        "feature_shape": list(shape),
        "estimated_ram_gib": estimated_gib,
        "disk_feature_bank_written": False,
    }
    write_json(output_dir / "base_ram_feature_config.json", config)
    del model, dummy_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, view_metadata, config


class GatedDirectHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.token_pool = a1.GatedPool(hidden_dim, attention_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, token_count, _ = feature.shape
        tokens = self.project(self.input_norm(feature))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        pooled, weights = self.token_pool(tokens.reshape(batch_size, num_views * token_count, -1))
        logits = self.classifier(self.embedding_norm(pooled))
        return logits, weights.reshape(batch_size, num_views, token_count)


class DirectFeatureDataset(Dataset):
    def __init__(
        self,
        base_features: np.ndarray,
        records: pd.DataFrame,
        indices: np.ndarray,
        roi_features: np.ndarray | None = None,
    ) -> None:
        self.base_features = base_features
        self.roi_features = roi_features
        self.records = records
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        feature = self.base_features[index]
        if self.roi_features is not None:
            feature = np.concatenate([feature, self.roi_features[index]], axis=0)
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "feature": torch.from_numpy(np.asarray(feature, dtype=np.float16)),
            "label": torch.tensor(int(self.records.iloc[index]["label_idx"]), dtype=torch.long),
        }


def balanced_sampler(records: pd.DataFrame, indices: np.ndarray, seed: int) -> WeightedRandomSampler:
    frame = records.iloc[indices]
    keys = frame["source_dataset"].astype(str) + "|" + frame["label_idx"].astype(str)
    counts = keys.value_counts()
    weights = keys.map(lambda key: 1.0 / counts[key]).to_numpy(dtype=np.float64).copy()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )


def make_loader(
    base_features: np.ndarray,
    records: pd.DataFrame,
    indices: np.ndarray,
    roi_features: np.ndarray | None,
    batch_size: int,
    train: bool,
    seed: int,
) -> DataLoader:
    dataset = DirectFeatureDataset(base_features, records, indices, roi_features)
    sampler = balanced_sampler(records, indices, seed) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def evaluate_head(
    model: GatedDirectHead,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    return_attention: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    model.eval()
    rows: list[dict[str, Any]] = []
    attention_rows: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.inference_mode():
        for batch in loader:
            feature = batch["feature"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                logits, attention = model(feature)
            probability = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            indices = batch["index"].numpy().astype(int)
            labels = batch["label"].numpy().astype(int)
            for index, label, prob in zip(indices, labels, probability):
                rows.append(
                    {"feature_row": index, "label_idx": label, "prob_high": float(prob)}
                )
            if return_attention:
                attention_rows.append((indices, attention.float().cpu().numpy()))
    frame = pd.DataFrame(rows)
    frame["pred_idx"] = (frame["prob_high"] >= 0.5).astype(int)
    if not return_attention:
        return frame, None
    attention_shape = attention_rows[0][1].shape[1:]
    attention_array = np.empty((len(loader.dataset.records), *attention_shape), dtype=np.float32)
    for indices, values in attention_rows:
        attention_array[indices] = values
    return frame, attention_array


def train_head(
    args: argparse.Namespace,
    base_features: np.ndarray,
    records: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    roi_features: np.ndarray | None,
    epochs: int,
    patience: int,
    seed: int,
    output_dir: Path,
) -> tuple[GatedDirectHead, dict[str, Any], pd.DataFrame]:
    a1.set_seed(seed)
    num_views = base_features.shape[1] + (0 if roi_features is None else roi_features.shape[1])
    device = torch.device(args.device)
    model = GatedDirectHead(
        feature_dim=base_features.shape[-1],
        num_views=num_views,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    ).to(device)
    train_loader = make_loader(
        base_features,
        records,
        train_indices,
        roi_features,
        args.head_batch_size,
        True,
        seed,
    )
    val_loader = make_loader(
        base_features,
        records,
        val_indices,
        roi_features,
        args.head_batch_size,
        False,
        seed,
    )
    test_loader = make_loader(
        base_features,
        records,
        test_indices,
        roi_features,
        args.head_batch_size,
        False,
        seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            feature = batch["feature"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                logits, _ = model(feature)
                loss = F.cross_entropy(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation, _ = evaluate_head(model, val_loader, device, args.amp)
        val_bacc = float(
            balanced_accuracy_score(validation["label_idx"], validation["pred_idx"])
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_balanced_accuracy": val_bacc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_bacc > best_metric + 1e-12:
            best_metric = val_bacc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No best head state was retained")
    model.load_state_dict(best_state)
    predictions, _ = evaluate_head(model, test_loader, device, args.amp)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    torch.save(best_state, output_dir / "best_head.pt")
    summary = {
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_metric,
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "num_views": num_views,
    }
    write_json(output_dir / "fold_summary.json", summary)
    return model, summary, predictions


def attention_heatmaps(attention: np.ndarray, heatmap_size: int) -> np.ndarray:
    _, num_views, token_count = attention.shape
    grid_size = math.isqrt(token_count)
    if grid_size * grid_size != token_count or num_views != 6:
        raise ValueError(
            f"Expected six square-token views, received views={num_views} tokens={token_count}"
        )
    heatmaps = np.zeros((len(attention), heatmap_size, heatmap_size), dtype=np.float32)
    counts = np.zeros_like(heatmaps)
    patch_y, patch_x = np.meshgrid(
        (np.arange(grid_size) + 0.5) / grid_size,
        (np.arange(grid_size) + 0.5) / grid_size,
        indexing="ij",
    )
    for view_offset, (x1, y1, x2, y2) in enumerate(Q70_WINDOWS):
        specimen_x = x1 + patch_x * (x2 - x1)
        specimen_y = y1 + patch_y * (y2 - y1)
        heat_x = np.clip((specimen_x * heatmap_size).astype(int), 0, heatmap_size - 1)
        heat_y = np.clip((specimen_y * heatmap_size).astype(int), 0, heatmap_size - 1)
        values = attention[:, view_offset + 2].reshape(len(attention), grid_size, grid_size)
        for row in range(len(attention)):
            np.add.at(heatmaps[row], (heat_y, heat_x), values[row])
            np.add.at(counts[row], (heat_y, heat_x), 1.0)
    heatmaps /= np.maximum(counts, 1.0)
    tensor = torch.from_numpy(heatmaps[:, None])
    smoothed = F.avg_pool2d(tensor, kernel_size=5, stride=1, padding=2)
    return smoothed[:, 0].numpy()


def specimen_and_mask(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    thumbnail = image.copy()
    thumbnail.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    thumb_rgb = np.asarray(thumbnail, dtype=np.uint8)
    raw_box = a1.detect_specimen_bbox(thumb_rgb)
    box2 = a1.expand_box(raw_box, thumb_rgb.shape[:2], 0.02)
    original_box = a1.map_thumb_box_to_original(box2, thumbnail.size, image.size)
    specimen = image.crop(original_box)
    tissue_mask = a1.detect_specimen_mask(thumb_rgb) > 0
    x1, y1, x2, y2 = box2
    specimen_mask = tissue_mask[y1:y2, x1:x2]
    if specimen_mask.size == 0:
        specimen_mask = np.ones((8, 8), dtype=bool)
    return specimen, specimen_mask


def centered_window(
    mask: np.ndarray, center_x: float, center_y: float, scale: float
) -> a1.TileWindow:
    half = scale / 2.0
    center_x = min(1.0 - half, max(half, center_x))
    center_y = min(1.0 - half, max(half, center_y))
    bounds = (center_x - half, center_y - half, center_x + half, center_y + half)
    return a1.TileWindow(*bounds, a1.normalized_window_coverage(mask, bounds))


def choose_attention_centers(
    heatmap: np.ndarray,
    mask: np.ndarray,
    count: int,
    detail_scale: float,
    minimum_tissue: float,
    minimum_distance: float,
) -> tuple[list[tuple[float, float]], int]:
    height, width = heatmap.shape
    ranked = np.argsort(heatmap.ravel())[::-1]
    selected: list[tuple[float, float]] = []
    for flat_index in ranked:
        row, column = divmod(int(flat_index), width)
        center = ((column + 0.5) / width, (row + 0.5) / height)
        window = centered_window(mask, *center, detail_scale)
        if window.tissue_coverage < minimum_tissue:
            continue
        if any(math.dist(center, previous) < minimum_distance for previous in selected):
            continue
        selected.append(center)
        if len(selected) == count:
            return selected, 0
    fallback = a1.select_spatial_tiles(mask, detail_scale, grid_size=7, count=count)
    for tile in fallback:
        center = (tile.center_x, tile.center_y)
        if all(math.dist(center, previous) >= minimum_distance for previous in selected):
            selected.append(center)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected.extend([(0.5, 0.5)] * (count - len(selected)))
    return selected[:count], count


def choose_matched_random_centers(
    mask: np.ndarray,
    attention_centers: list[tuple[float, float]],
    detail_scale: float,
    minimum_tissue: float,
    minimum_distance: float,
    seed: int,
) -> list[tuple[float, float]]:
    candidates = [
        item
        for item in a1.candidate_windows(mask, detail_scale, grid_size=9)
        if item.tissue_coverage >= minimum_tissue
    ]
    if len(candidates) < len(attention_centers):
        candidates = a1.candidate_windows(mask, detail_scale, grid_size=9)
    rng = np.random.default_rng(seed)
    selected: list[tuple[float, float]] = []
    remaining = list(candidates)
    for target_center in attention_centers:
        target_coverage = centered_window(mask, *target_center, detail_scale).tissue_coverage
        eligible = [
            item
            for item in remaining
            if all(
                math.dist((item.center_x, item.center_y), previous) >= minimum_distance
                for previous in selected
            )
        ]
        if not eligible:
            eligible = remaining
        eligible.sort(key=lambda item: abs(item.tissue_coverage - target_coverage))
        short = eligible[: min(8, len(eligible))]
        choice = short[int(rng.integers(0, len(short)))]
        selected.append((choice.center_x, choice.center_y))
        remaining.remove(choice)
    return selected


def make_case_plan(
    image: Image.Image,
    heatmap: np.ndarray,
    scales: list[float],
    args: argparse.Namespace,
    seed: int,
) -> tuple[CaseRoiPlan, Image.Image]:
    specimen, mask = specimen_and_mask(image)
    detail_scale = min(scales)
    attention_centers, fallback_count = choose_attention_centers(
        heatmap,
        mask,
        args.roi_centers,
        detail_scale,
        args.minimum_tissue,
        args.minimum_center_distance,
    )
    random_centers = choose_matched_random_centers(
        mask,
        attention_centers,
        detail_scale,
        args.minimum_tissue,
        args.minimum_center_distance,
        seed,
    )
    attention_windows = tuple(
        centered_window(mask, center_x, center_y, scale)
        for center_x, center_y in attention_centers
        for scale in scales
    )
    random_windows = tuple(
        centered_window(mask, center_x, center_y, scale)
        for center_x, center_y in random_centers
        for scale in scales
    )
    return CaseRoiPlan(attention_windows, random_windows, fallback_count), specimen


def build_roi_plans(
    args: argparse.Namespace,
    records: pd.DataFrame,
    heatmaps: np.ndarray,
    split_mode: str,
    fold_value: int | str,
    output_dir: Path,
) -> list[CaseRoiPlan]:
    scales = [float(item) for item in args.roi_scales.split(",") if item.strip()]
    plans: list[CaseRoiPlan] = []
    rows: list[dict[str, Any]] = []
    for index, row in records.iterrows():
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        plan, _ = make_case_plan(
            image,
            heatmaps[index],
            scales,
            args,
            stable_seed(args.seed, split_mode, fold_value, row["case_id"], "random_roi"),
        )
        plans.append(plan)
        for route_name, windows in (("attention", plan.attention), ("random", plan.random)):
            for roi_index, window in enumerate(windows):
                rows.append(
                    {
                        "feature_row": int(index),
                        "case_id": str(row["case_id"]),
                        "route": route_name,
                        "roi_index": roi_index,
                        "center_x": window.center_x,
                        "center_y": window.center_y,
                        "scale": window.scale,
                        "tissue_coverage": window.tissue_coverage,
                        "attention_fallback_count": plan.fallback_count,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "roi_plan.csv", index=False, encoding="utf-8-sig")
    return plans


class RoiViewDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        plans: list[CaseRoiPlan],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.plans = plans
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        specimen, _ = specimen_and_mask(image)
        plan = self.plans[index]
        attention = [self.transform(a1.crop_relative(specimen, item)) for item in plan.attention]
        random_views = [self.transform(a1.crop_relative(specimen, item)) for item in plan.random]
        return index, torch.stack(attention + random_views, dim=0)


def extract_roi_features(
    args: argparse.Namespace,
    records: pd.DataFrame,
    plans: list[CaseRoiPlan],
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    model = create_backbone(args.model_name, "dense").eval().to(device)
    image_size = model_image_size(model, args.image_size)
    mean, std = model_normalization(model)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            dummy_tokens = extract_feature_tokens(model, dummy, "dense")
    token_count, feature_dim = map(int, dummy_tokens.shape[1:])
    roi_views = len(plans[0].attention)
    attention_features = np.empty(
        (len(records), roi_views, token_count, feature_dim), dtype=np.float16
    )
    random_features = np.empty_like(attention_features)
    loader_kwargs: dict[str, Any] = {
        "dataset": RoiViewDataset(records, plans, transform),
        "batch_size": args.extract_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    progress = tqdm(loader, desc="D1 attention/random ROI tokens", dynamic_ncols=True, file=sys.stdout)
    for row_indices, view_batch in progress:
        batch_size, num_views, channels, height, width = view_batch.shape
        inputs = view_batch.reshape(batch_size * num_views, channels, height, width).to(
            device, non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            tokens = extract_feature_tokens(model, inputs, "dense")
        tokens = tokens.reshape(batch_size, num_views, token_count, feature_dim)
        indices = row_indices.numpy().astype(int)
        array = tokens.detach().cpu().numpy().astype(np.float16, copy=False)
        attention_features[indices] = array[:, :roi_views]
        random_features[indices] = array[:, roi_views:]
        progress.set_postfix(cases=int(indices[-1]) + 1, total=len(records))
    progress.close()
    del model, dummy_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return attention_features, random_features


def metric_summary(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probability))
        balanced_accuracy = float(balanced_accuracy_score(y_true, predicted))
    else:
        auc = float("nan")
        balanced_accuracy = float(accuracy_score(y_true, predicted))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize_variant(
    records: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    merged = predictions.merge(records, on="feature_row", how="left", suffixes=("", "_meta"))
    merged.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    rows = [
        {
            "group_type": "overall",
            "group": "all",
            **metric_summary(merged["label_idx"], merged["prob_high"]),
        }
    ]
    for column in ("source_dataset", "task_l6_label", "master_fold_id"):
        for group_name, group in merged.groupby(column, dropna=False):
            rows.append(
                {
                    "group_type": column,
                    "group": str(group_name),
                    **metric_summary(group["label_idx"], group["prob_high"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    return rows[0]


def main() -> None:
    args = parse_args()
    a1.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = a1.load_records(args)
    metadata_path = output_dir / "metadata.csv"
    if metadata_path.exists():
        metadata_path.unlink()
    records.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "records": len(records),
            "base_views": BASE_VIEW_NAMES,
            "variants": VARIANTS,
            "selector_input": "image tokens only",
            "diagnostic_input": "base and optional raw-pixel ROI tokens only",
            "prohibited_inputs": [
                "stage1_probability",
                "confidence",
                "difficulty",
                "source",
                "batch",
                "true_subtype",
                "error_label",
            ],
            "selection_policy": "Fold-specific selector trained only on outer train; matched random ROI control is mandatory.",
            "disk_feature_bank_written": False,
        },
    )
    base_features, _, _ = extract_base_features(args, records, output_dir)
    if args.extract_only:
        print("[done] D1 base extraction smoke completed", flush=True)
        return

    all_summaries: list[dict[str, Any]] = []
    split_modes = [item.strip() for item in args.split_modes.split(",") if item.strip()]
    for split_mode in split_modes:
        if split_mode == "fivefold":
            fold_values: list[int | str] = sorted(
                records["master_fold_id"].unique().astype(int).tolist()
            )
        elif split_mode == "source_lodo":
            fold_values = sorted(records["source_dataset"].astype(str).unique().tolist())
        else:
            raise ValueError(f"Unsupported split mode: {split_mode}")
        if args.max_folds is not None:
            fold_values = fold_values[: args.max_folds]
        for fold_position, fold_value in enumerate(fold_values):
            fold_root = output_dir / split_mode / f"fold_{fold_position + 1}"
            complete = all(
                (fold_root / variant / "test_predictions.csv").is_file() for variant in VARIANTS
            )
            if complete:
                print(f"[skip] complete {split_mode} {fold_value}", flush=True)
                continue
            if fold_root.exists():
                shutil.rmtree(fold_root)
            train_indices, val_indices, test_indices, held_source = a1.fold_indices(
                records, split_mode, fold_position, fold_value
            )
            selector_dir = fold_root / "selector"
            selector_seed = stable_seed(args.seed, split_mode, fold_value, "selector")
            selector_model, selector_summary, _ = train_head(
                args,
                base_features,
                records,
                train_indices,
                val_indices,
                test_indices,
                None,
                args.selector_epochs,
                args.selector_patience,
                selector_seed,
                selector_dir,
            )
            all_loader = make_loader(
                base_features,
                records,
                np.arange(len(records)),
                None,
                args.head_batch_size,
                False,
                selector_seed,
            )
            _, attention = evaluate_head(
                selector_model,
                all_loader,
                torch.device(args.device),
                args.amp,
                return_attention=True,
            )
            if attention is None:
                raise RuntimeError("Selector did not return attention")
            heatmaps = attention_heatmaps(attention, args.heatmap_size)
            np.save(selector_dir / "attention_heatmaps.float16.npy", heatmaps.astype(np.float16))
            del selector_model, attention
            if args.device == "cuda":
                torch.cuda.empty_cache()
            plans = build_roi_plans(
                args, records, heatmaps, split_mode, fold_value, selector_dir
            )
            attention_features, random_features = extract_roi_features(args, records, plans)
            variant_features = {
                "base_only": None,
                "attention_roi": attention_features,
                "matched_random_roi": random_features,
            }
            diagnostic_seed = stable_seed(args.seed, split_mode, fold_value, "diagnostic")
            for variant, roi_features in variant_features.items():
                model, head_summary, predictions = train_head(
                    args,
                    base_features,
                    records,
                    train_indices,
                    val_indices,
                    test_indices,
                    roi_features,
                    args.diagnostic_epochs,
                    args.diagnostic_patience,
                    diagnostic_seed,
                    fold_root / variant,
                )
                head_summary.update(
                    {
                        "variant": variant,
                        "split_mode": split_mode,
                        "fold_value": str(fold_value),
                        "held_source": held_source,
                        "selector_best_epoch": selector_summary["best_epoch"],
                    }
                )
                write_json(fold_root / variant / "fold_summary.json", head_summary)
                del model, predictions
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            del heatmaps, plans, attention_features, random_features

        for variant in VARIANTS:
            frames = []
            for fold_position, _ in enumerate(fold_values):
                path = (
                    output_dir
                    / split_mode
                    / f"fold_{fold_position + 1}"
                    / variant
                    / "test_predictions.csv"
                )
                if path.is_file():
                    frames.append(pd.read_csv(path, encoding="utf-8-sig"))
            if not frames:
                continue
            predictions = pd.concat(frames, ignore_index=True)
            variant_dir = output_dir / split_mode / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            overall = summarize_variant(records, predictions, variant_dir)
            all_summaries.append(
                {"split_mode": split_mode, "variant": variant, **overall}
            )
    summary_path = output_dir / "D1_ATTENTION_ROI_SUMMARY.csv"
    if summary_path.exists():
        summary_path.unlink()
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_json(output_dir / "D1_ATTENTION_ROI_COMPLETE.json", {"complete": True})


if __name__ == "__main__":
    main()
