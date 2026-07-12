from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent if (_SCRIPT_DIR.parent / "thymic_baseline").exists() else Path("/workspace/thymic_project")
for item in reversed((_SCRIPT_DIR, _PROJECT_ROOT, _PROJECT_ROOT / "scripts")):
    while str(item) in sys.path:
        sys.path.remove(str(item))
    sys.path.insert(0, str(item))

from extract_task7_dense_token_bank_20260711 import (  # noqa: E402
    create_backbone,
    detect_specimen_bbox,
    detect_specimen_mask,
    extract_feature_tokens,
    model_image_size,
    model_normalization,
)


DEFAULT_REGISTRY = (
    _PROJECT_ROOT
    / "experiments"
    / "base_model_expansion_20260706"
    / "outputs"
    / "registry"
    / "task7_four_domain_master_registry.csv"
)
DEFAULT_SPLIT = (
    _PROJECT_ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "45_old_third_all_balanced_finetune_inputs_20260523"
    / "split.csv"
)

ALL_VIEW_NAMES = [
    "whole",
    "c1_crop",
    "c1_q0",
    "c1_q1",
    "c1_q2",
    "c1_q3",
    "native_specimen",
    "medium_0",
    "medium_1",
    "medium_2",
    "medium_3",
    "local_0",
    "local_1",
    "local_2",
    "local_3",
    "local_4",
    "local_5",
    "local_6",
    "local_7",
]
C1_VIEW_INDICES = list(range(6))
NATIVE_VIEW_INDICES = [0, *range(6, 19)]


@dataclass(frozen=True)
class TileWindow:
    x1: float
    y1: float
    x2: float
    y2: float
    tissue_coverage: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def scale(self) -> float:
        return max(self.x2 - self.x1, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 A1 native-detail frozen-token pilot without a disk feature bank."
    )
    parser.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--split-csv", default=str(DEFAULT_SPLIT))
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--variants",
        default="c1_hier_mil,native_hier_mil,native_cross_attention",
        help="Fixed families; each is reported separately and is not selected on pooled OOF.",
    )
    parser.add_argument("--split-modes", default="fivefold,source_lodo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def expand_box(
    box: tuple[int, int, int, int], shape: tuple[int, int], margin_ratio: float
) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = box
    margin_x = int(round((x2 - x1) * margin_ratio))
    margin_y = int(round((y2 - y1) * margin_ratio))
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )


def normalized_window_coverage(mask: np.ndarray, window: tuple[float, float, float, float]) -> float:
    height, width = mask.shape
    x1, y1, x2, y2 = window
    left = min(width - 1, max(0, int(math.floor(x1 * width))))
    top = min(height - 1, max(0, int(math.floor(y1 * height))))
    right = min(width, max(left + 1, int(math.ceil(x2 * width))))
    bottom = min(height, max(top + 1, int(math.ceil(y2 * height))))
    return float(mask[top:bottom, left:right].mean())


def candidate_windows(mask: np.ndarray, scale: float, grid_size: int) -> list[TileWindow]:
    centers = np.linspace(scale / 2.0, 1.0 - scale / 2.0, grid_size)
    candidates = []
    for center_y in centers:
        for center_x in centers:
            window = (
                float(center_x - scale / 2.0),
                float(center_y - scale / 2.0),
                float(center_x + scale / 2.0),
                float(center_y + scale / 2.0),
            )
            candidates.append(TileWindow(*window, normalized_window_coverage(mask, window)))
    return candidates


def select_spatial_tiles(
    mask: np.ndarray, scale: float, grid_size: int, count: int, minimum_tissue: float = 0.60
) -> list[TileWindow]:
    candidates = candidate_windows(mask, scale, grid_size)
    eligible = [item for item in candidates if item.tissue_coverage >= minimum_tissue]
    pool = eligible if len(eligible) >= count else candidates
    center = np.array([0.5, 0.5])
    first = max(
        pool,
        key=lambda item: (
            item.tissue_coverage,
            -float(np.linalg.norm(np.array([item.center_x, item.center_y]) - center)),
            -item.center_y,
            -item.center_x,
        ),
    )
    selected = [first]
    remaining = [item for item in pool if item != first]
    while len(selected) < count and remaining:
        next_item = max(
            remaining,
            key=lambda item: (
                min(
                    (item.center_x - chosen.center_x) ** 2 + (item.center_y - chosen.center_y) ** 2
                    for chosen in selected
                ),
                item.tissue_coverage,
                -item.center_y,
                -item.center_x,
            ),
        )
        selected.append(next_item)
        remaining.remove(next_item)
    if len(selected) != count:
        raise RuntimeError(f"Could not select {count} deterministic tiles at scale={scale}")
    return sorted(selected, key=lambda item: (item.center_y, item.center_x))


def crop_relative(image: Image.Image, window: TileWindow) -> Image.Image:
    width, height = image.size
    left = min(width - 1, max(0, int(round(window.x1 * width))))
    top = min(height - 1, max(0, int(round(window.y1 * height))))
    right = min(width, max(left + 1, int(round(window.x2 * width))))
    bottom = min(height, max(top + 1, int(round(window.y2 * height))))
    return image.crop((left, top, right, bottom))


def map_thumb_box_to_original(
    box: tuple[int, int, int, int], thumb_size: tuple[int, int], original_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    thumb_width, thumb_height = thumb_size
    original_width, original_height = original_size
    scale_x = original_width / thumb_width
    scale_y = original_height / thumb_height
    x1, y1, x2, y2 = box
    return (
        max(0, int(math.floor(x1 * scale_x))),
        max(0, int(math.floor(y1 * scale_y))),
        min(original_width, int(math.ceil(x2 * scale_x))),
        min(original_height, int(math.ceil(y2 * scale_y))),
    )


def view_metadata(window: TileWindow) -> list[float]:
    return [
        float(window.center_x * 2.0 - 1.0),
        float(window.center_y * 2.0 - 1.0),
        float(window.scale),
        float(window.tissue_coverage),
    ]


def build_all_views(image: Image.Image) -> tuple[list[Image.Image], np.ndarray]:
    thumbnail = image.copy()
    thumbnail.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    thumb_rgb = np.asarray(thumbnail, dtype=np.uint8)
    raw_box = detect_specimen_bbox(thumb_rgb)
    box2 = expand_box(raw_box, thumb_rgb.shape[:2], 0.02)
    box12 = expand_box(raw_box, thumb_rgb.shape[:2], 0.12)
    original_box2 = map_thumb_box_to_original(box2, thumbnail.size, image.size)
    original_box12 = map_thumb_box_to_original(box12, thumbnail.size, image.size)
    specimen = image.crop(original_box2)
    c1_crop = image.crop(original_box12)

    tissue_mask = detect_specimen_mask(thumb_rgb) > 0
    x1, y1, x2, y2 = box2
    specimen_mask = tissue_mask[y1:y2, x1:x2]
    if specimen_mask.size == 0:
        specimen_mask = np.ones((8, 8), dtype=bool)
    specimen_coverage = float(specimen_mask.mean())

    q70 = [
        TileWindow(0.0, 0.0, 0.7, 0.7, normalized_window_coverage(specimen_mask, (0.0, 0.0, 0.7, 0.7))),
        TileWindow(0.3, 0.0, 1.0, 0.7, normalized_window_coverage(specimen_mask, (0.3, 0.0, 1.0, 0.7))),
        TileWindow(0.0, 0.3, 0.7, 1.0, normalized_window_coverage(specimen_mask, (0.0, 0.3, 0.7, 1.0))),
        TileWindow(0.3, 0.3, 1.0, 1.0, normalized_window_coverage(specimen_mask, (0.3, 0.3, 1.0, 1.0))),
    ]
    medium = select_spatial_tiles(specimen_mask, scale=0.50, grid_size=3, count=4)
    local = select_spatial_tiles(specimen_mask, scale=0.25, grid_size=5, count=8)

    whole_window = TileWindow(0.0, 0.0, 1.0, 1.0, float(tissue_mask.mean()))
    specimen_window = TileWindow(0.0, 0.0, 1.0, 1.0, specimen_coverage)
    views = [
        image,
        c1_crop,
        *[crop_relative(specimen, item) for item in q70],
        specimen,
        *[crop_relative(specimen, item) for item in medium],
        *[crop_relative(specimen, item) for item in local],
    ]
    metadata = np.asarray(
        [
            view_metadata(whole_window),
            view_metadata(specimen_window),
            *[view_metadata(item) for item in q70],
            view_metadata(specimen_window),
            *[view_metadata(item) for item in medium],
            *[view_metadata(item) for item in local],
        ],
        dtype=np.float32,
    )
    if len(views) != len(ALL_VIEW_NAMES) or metadata.shape != (len(ALL_VIEW_NAMES), 4):
        raise RuntimeError("Native-detail view construction produced an invalid shape")
    return views, metadata


def load_records(args: argparse.Namespace) -> pd.DataFrame:
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    records = pd.read_csv(
        args.registry_csv, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig"
    )
    records = records[records["domain"].astype(str).isin(domains)].copy()
    if "image_exists" in records:
        records = records[records["image_exists"].astype(str).str.lower().isin(["true", "1", "yes"])]
    records = records.sort_values(["domain", "case_id"]).drop_duplicates("case_id").reset_index(drop=True)
    locked = pd.read_csv(args.split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    locked.columns = [str(column).lstrip("\ufeff") for column in locked.columns]
    locked = locked[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    records = records.drop(columns=["master_fold_id"], errors="ignore").merge(locked, on="case_id", how="left")
    if records["master_fold_id"].isna().any():
        missing = records.loc[records["master_fold_id"].isna(), "case_id"].head(10).tolist()
        raise ValueError(f"Locked fold is missing for cases: {missing}")
    records["master_fold_id"] = records["master_fold_id"].astype(int)
    records["label_idx"] = pd.to_numeric(records["label_idx"], errors="raise").astype(int)
    if args.max_cases is not None:
        records = records.head(int(args.max_cases)).copy()
    missing_paths = [path for path in records["image_path"].astype(str) if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing image paths: {missing_paths[:10]}")
    if records["original_case_id"].duplicated().any():
        raise ValueError("A1 registry must contain one primary image per original_case_id")
    records.insert(0, "feature_row", np.arange(len(records), dtype=int))
    return records


class NativeViewDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        views, metadata = build_all_views(image)
        tensors = torch.stack([self.transform(view) for view in views], dim=0)
        return index, tensors, torch.from_numpy(metadata)


def extract_features_to_ram(
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
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            dummy_tokens = extract_feature_tokens(model, dummy, "dense")
    token_count = int(dummy_tokens.shape[1])
    feature_dim = int(dummy_tokens.shape[2])
    shape = (len(records), len(ALL_VIEW_NAMES), token_count, feature_dim)
    estimated_gib = float(np.prod(shape) * np.dtype(np.float16).itemsize / (1024**3))
    print(f"[ram-bank] shape={shape} estimated_gib={estimated_gib:.2f}", flush=True)
    features = np.empty(shape, dtype=np.float16)
    view_metadata_array = np.empty((len(records), len(ALL_VIEW_NAMES), 4), dtype=np.float32)

    dataset = NativeViewDataset(records, transform)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.extract_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    progress = tqdm(loader, desc="native-detail tokens", dynamic_ncols=True, file=sys.stdout)
    for row_indices, view_batch, view_meta in progress:
        batch_size, num_views, channels, height, width = view_batch.shape
        inputs = view_batch.reshape(batch_size * num_views, channels, height, width).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"
        ):
            tokens = extract_feature_tokens(model, inputs, "dense")
        tokens = tokens.reshape(batch_size, num_views, token_count, feature_dim)
        indices = row_indices.numpy().astype(int)
        features[indices] = tokens.detach().cpu().numpy().astype(np.float16, copy=False)
        view_metadata_array[indices] = view_meta.numpy().astype(np.float32, copy=False)
        progress.set_postfix(cases=int(indices[-1]) + 1, total=len(records))
    progress.close()

    config = {
        "model_name": args.model_name,
        "image_size": image_size,
        "view_names": ALL_VIEW_NAMES,
        "c1_view_indices": C1_VIEW_INDICES,
        "native_view_indices": NATIVE_VIEW_INDICES,
        "feature_shape": list(shape),
        "view_metadata_shape": list(view_metadata_array.shape),
        "estimated_ram_gib": estimated_gib,
        "dtype": "float16",
        "disk_feature_bank_written": False,
    }
    write_json(output_dir / "ram_feature_config.json", config)
    del model, dummy_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, view_metadata_array, config


class FeatureDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        view_metadata_array: np.ndarray,
        records: pd.DataFrame,
        indices: np.ndarray,
        view_indices: list[int],
    ) -> None:
        self.features = features
        self.view_metadata_array = view_metadata_array
        self.records = records
        self.indices = np.asarray(indices, dtype=int)
        self.view_indices = np.asarray(view_indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        feature = np.array(self.features[index, self.view_indices], dtype=np.float16, copy=True)
        metadata = np.array(self.view_metadata_array[index, self.view_indices], dtype=np.float32, copy=True)
        return {
            "feature": torch.from_numpy(feature),
            "view_metadata": torch.from_numpy(metadata),
            "label": torch.tensor(int(self.records.iloc[index]["label_idx"]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class GatedPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.tanh = nn.Linear(hidden_dim, attention_dim)
        self.sigmoid = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score(torch.tanh(self.tanh(tokens)) * torch.sigmoid(self.sigmoid(tokens))).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        return torch.sum(tokens * weight.unsqueeze(-1), dim=-2), weight


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, context: torch.Tensor, local: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attended, _ = self.attention(
            context, local, local, key_padding_mask=key_padding_mask, need_weights=False
        )
        context = self.norm1(context + attended)
        return self.norm2(context + self.ffn(context))


class NativeDetailHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
        architecture: str,
        use_metadata: bool,
        specimen_view_index: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.use_metadata = use_metadata
        self.specimen_view_index = specimen_view_index
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.token_pool = GatedPool(hidden_dim, attention_dim)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.view_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, 1),
        )
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, dropout) for _ in range(2)])
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, feature: torch.Tensor, view_metadata_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, token_count, _ = feature.shape
        tokens = self.project(self.input_norm(feature))
        pooled, _ = self.token_pool(tokens.reshape(batch_size * num_views, token_count, -1))
        views = pooled.reshape(batch_size, num_views, -1)
        views = views + self.view_embeddings[:num_views].unsqueeze(0)
        valid_views = torch.ones((batch_size, num_views), dtype=torch.bool, device=views.device)
        if self.use_metadata:
            views = views + self.metadata_mlp(view_metadata_tensor)
            scale = view_metadata_tensor[..., 2]
            tissue_coverage = view_metadata_tensor[..., 3]
            valid_views = (scale >= 0.60) | (tissue_coverage >= 0.60)
        if self.architecture == "cross_attention":
            context = views[:, self.specimen_view_index : self.specimen_view_index + 1]
            local_indices = [index for index in range(num_views) if index != self.specimen_view_index]
            local = views[:, local_indices]
            local_padding_mask = ~valid_views[:, local_indices]
            for block in self.cross_blocks:
                context = block(context, local, key_padding_mask=local_padding_mask)
            embedding = context.squeeze(1)
        else:
            global_context = views[:, : min(2, num_views)].mean(dim=1, keepdim=True).expand_as(views)
            scores = self.view_gate(torch.cat([views, global_context], dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~valid_views, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            embedding = torch.sum(views * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.embedding_norm(embedding))


def variant_specification(name: str) -> tuple[list[int], str, bool, int]:
    if name == "c1_hier_mil":
        return C1_VIEW_INDICES, "hier_mil", False, 1
    if name == "native_hier_mil":
        return NATIVE_VIEW_INDICES, "hier_mil", True, 1
    if name == "native_cross_attention":
        return NATIVE_VIEW_INDICES, "cross_attention", True, 1
    raise ValueError(f"Unknown fixed A1 variant: {name}")


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
    features: np.ndarray,
    view_metadata_array: np.ndarray,
    records: pd.DataFrame,
    indices: np.ndarray,
    view_indices: list[int],
    batch_size: int,
    train: bool,
    seed: int,
) -> DataLoader:
    dataset = FeatureDataset(features, view_metadata_array, records, indices, view_indices)
    sampler = balanced_sampler(records, indices, seed) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            feature = batch["feature"].to(device, non_blocking=True)
            metadata = batch["view_metadata"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(feature, metadata)
            probability = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            for index, label, prob in zip(batch["index"].numpy(), batch["label"].numpy(), probability):
                rows.append({"feature_row": int(index), "label_idx": int(label), "prob_high": float(prob)})
    result = pd.DataFrame(rows)
    result["pred_idx"] = (result["prob_high"] >= 0.5).astype(int)
    return result


def metric_summary(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, probability))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fold_indices(
    records: pd.DataFrame, split_mode: str, fold_position: int, fold_value: int | str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    master_fold = records["master_fold_id"].to_numpy(dtype=int)
    if split_mode == "fivefold":
        test_fold = int(fold_value)
        val_fold = (test_fold % 5) + 1
        test_mask = master_fold == test_fold
        val_mask = master_fold == val_fold
        held_source = ""
    elif split_mode == "source_lodo":
        held_source = str(fold_value)
        source = records["source_dataset"].astype(str).to_numpy()
        test_mask = source == held_source
        val_fold = (fold_position % 5) + 1
        val_mask = (~test_mask) & (master_fold == val_fold)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    train_mask = ~(test_mask | val_mask)
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask), held_source


def train_fold(
    args: argparse.Namespace,
    features: np.ndarray,
    view_metadata_array: np.ndarray,
    records: pd.DataFrame,
    output_dir: Path,
    variant: str,
    split_mode: str,
    fold_position: int,
    fold_value: int | str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_seed = args.seed + fold_position + int(hashlib.sha256(f"{variant}|{split_mode}".encode()).hexdigest()[:6], 16)
    set_seed(fold_seed)
    view_indices, architecture, use_metadata, specimen_view_index = variant_specification(variant)
    train_indices, val_indices, test_indices, held_source = fold_indices(
        records, split_mode, fold_position, fold_value
    )
    if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
        raise ValueError(f"Empty partition for {variant} {split_mode} {fold_value}")
    train_loader = make_loader(
        features,
        view_metadata_array,
        records,
        train_indices,
        view_indices,
        args.head_batch_size,
        True,
        fold_seed,
    )
    val_loader = make_loader(
        features,
        view_metadata_array,
        records,
        val_indices,
        view_indices,
        args.head_batch_size,
        False,
        fold_seed,
    )
    test_loader = make_loader(
        features,
        view_metadata_array,
        records,
        test_indices,
        view_indices,
        args.head_batch_size,
        False,
        fold_seed,
    )
    device = torch.device(args.device)
    model = NativeDetailHead(
        feature_dim=features.shape[-1],
        num_views=len(view_indices),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        architecture=architecture,
        use_metadata=use_metadata,
        specimen_view_index=specimen_view_index,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
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
        validation = evaluate(model, val_loader, device, args.amp)
        validation_metric = float(balanced_accuracy_score(validation["label_idx"], validation["pred_idx"]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_balanced_accuracy": validation_metric,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_metric > best_metric + 1e-12:
            best_metric = validation_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"[{variant} {split_mode} {fold_value}] epoch={epoch} "
            f"val_bacc={validation_metric:.4f} best={best_metric:.4f} stale={stale}",
            flush=True,
        )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No best model state was retained")
    model.load_state_dict(best_state)
    predictions = evaluate(model, test_loader, device, args.amp)
    metrics = metric_summary(predictions["label_idx"], predictions["prob_high"])
    fold_dir = output_dir / variant / split_mode / f"fold_{fold_position + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    predictions.to_csv(fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    torch.save(best_state, fold_dir / "best_head.pt")
    summary = {
        "variant": variant,
        "split_mode": split_mode,
        "fold_position": fold_position + 1,
        "fold_value": str(fold_value),
        "held_source": held_source,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_metric,
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    write_json(fold_dir / "fold_summary.json", summary)
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, predictions


def summarize_predictions(
    predictions: pd.DataFrame, records: pd.DataFrame, output_dir: Path, variant: str, split_mode: str
) -> dict[str, Any]:
    merged = predictions.merge(records, on="feature_row", how="left", suffixes=("", "_meta"))
    merged.to_csv(output_dir / variant / split_mode / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    rows = [{"group_type": "overall", "group": "all", **metric_summary(merged["label_idx"], merged["prob_high"])}]
    for column in ["domain", "source_dataset", "task_l6_label"]:
        for group_name, group in merged.groupby(column, dropna=False):
            rows.append(
                {
                    "group_type": column,
                    "group": str(group_name),
                    **metric_summary(group["label_idx"], group["prob_high"]),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / variant / split_mode / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    overall = rows[0]
    return {"variant": variant, "split_mode": split_mode, **overall}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)
    metadata_columns = [
        "feature_row",
        "domain",
        "dataset_role",
        "case_id",
        "original_case_id",
        "source_dataset",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "image_name",
        "image_path",
        "master_fold_id",
    ]
    records[[column for column in metadata_columns if column in records]].to_csv(
        output_dir / "metadata.csv", index=False, encoding="utf-8-sig"
    )
    run_config = vars(args).copy()
    run_config.update(
        {
            "records": len(records),
            "view_names": ALL_VIEW_NAMES,
            "prohibited_inference_features": [
                "source",
                "batch",
                "image_count",
                "selection_rule",
                "stage1_probability",
                "confidence",
                "difficulty",
            ],
            "selection_policy": "Each fixed family is reported separately; no pooled-OOF winner replacement.",
            "tissue_tile_policy": "Medium/local slots below 60% tissue coverage are masked from view aggregation.",
        }
    )
    write_json(output_dir / "run_config.json", run_config)
    features, view_metadata_array, feature_config = extract_features_to_ram(args, records, output_dir)
    tile_rows = []
    for row_index, row in records.iterrows():
        for view_index, view_name in enumerate(ALL_VIEW_NAMES):
            center_x, center_y, scale, tissue_coverage = view_metadata_array[row_index, view_index]
            tile_rows.append(
                {
                    "feature_row": int(row_index),
                    "case_id": str(row["case_id"]),
                    "view_index": view_index,
                    "view_name": view_name,
                    "center_x_normalized": float(center_x),
                    "center_y_normalized": float(center_y),
                    "scale_relative_to_specimen": float(scale),
                    "tissue_coverage": float(tissue_coverage),
                }
            )
    pd.DataFrame(tile_rows).to_csv(output_dir / "tile_metadata.csv", index=False, encoding="utf-8-sig")
    if args.extract_only:
        print("[done] extraction smoke test completed; RAM bank released on exit", flush=True)
        return

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    split_modes = [item.strip() for item in args.split_modes.split(",") if item.strip()]
    summaries = []
    for variant in variants:
        variant_specification(variant)
        for split_mode in split_modes:
            if split_mode == "fivefold":
                fold_values: list[int | str] = sorted(records["master_fold_id"].unique().astype(int).tolist())
            elif split_mode == "source_lodo":
                fold_values = sorted(records["source_dataset"].astype(str).unique().tolist())
            else:
                raise ValueError(f"Unknown split mode: {split_mode}")
            fold_summaries = []
            fold_predictions = []
            for fold_position, fold_value in enumerate(fold_values):
                summary, prediction = train_fold(
                    args,
                    features,
                    view_metadata_array,
                    records,
                    output_dir,
                    variant,
                    split_mode,
                    fold_position,
                    fold_value,
                )
                fold_summaries.append(summary)
                prediction["outer_fold"] = str(fold_value)
                fold_predictions.append(prediction)
            run_dir = output_dir / variant / split_mode
            pd.DataFrame(fold_summaries).to_csv(run_dir / "cv_fold_summary.csv", index=False, encoding="utf-8-sig")
            summaries.append(
                summarize_predictions(
                    pd.concat(fold_predictions, ignore_index=True), records, output_dir, variant, split_mode
                )
            )
    pd.DataFrame(summaries).to_csv(output_dir / "a1_fixed_family_summary.csv", index=False, encoding="utf-8-sig")
    feature_config["completed_training"] = True
    write_json(output_dir / "ram_feature_config.json", feature_config)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
