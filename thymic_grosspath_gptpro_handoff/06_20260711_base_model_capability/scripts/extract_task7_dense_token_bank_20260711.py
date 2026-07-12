from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageFilter, ImageOps
from timm.models import load_checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

_SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (
    _SCRIPT_PROJECT_ROOT
    if (_SCRIPT_PROJECT_ROOT / "thymic_baseline").exists()
    else Path("/workspace/thymic_project")
)
for item in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from thymic_baseline.cropping import (  # noqa: E402
    detect_specimen_bbox,
    detect_specimen_mask,
    expand_bbox,
    extract_specimen_crop,
)

DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "experiments"
    / "base_model_expansion_20260706"
    / "outputs"
    / "registry"
    / "task7_four_domain_master_registry.csv"
)
LOCAL_DINOV2_REPO = PROJECT_ROOT / "third_party" / "round3" / "dinov2"
LOCAL_VITAMIN_CHECKPOINT = Path(
    "/root/.cache/huggingface/hub/models--jienengchen--ViTamin-L-384px/"
    "snapshots/a8bd536320237a2a0fc65480bcdfc4a67c00133d/pytorch_model.bin"
)


class DINOv2PatchWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.patch_embed = backbone.patch_embed
        self.num_features = int(getattr(backbone, "embed_dim"))
        self.num_prefix_tokens = 0
        self.pretrained_cfg = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "input_size": (3, 518, 518),
        }

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(inputs)["x_norm_patchtokens"]


def create_backbone(
    model_name: str, feature_mode: str, checkpoint_path: str = ""
) -> torch.nn.Module:
    if model_name.startswith("local_dinov2_"):
        if feature_mode != "dense":
            raise ValueError("Local DINOv2 wrapper only supports dense features.")
        hub_name = model_name.removeprefix("local_")
        if not LOCAL_DINOV2_REPO.exists():
            raise FileNotFoundError(f"Missing local DINOv2 repository: {LOCAL_DINOV2_REPO}")
        backbone = torch.hub.load(
            str(LOCAL_DINOV2_REPO), hub_name, source="local", pretrained=True
        )
        return DINOv2PatchWrapper(backbone)
    if model_name == "local_vitamin_large_384":
        if not LOCAL_VITAMIN_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing local ViTamin checkpoint: {LOCAL_VITAMIN_CHECKPOINT}")
        model_kwargs: dict[str, Any] = {"pretrained": False, "num_classes": 0}
        if feature_mode == "dense":
            model_kwargs["global_pool"] = ""
        model = timm.create_model("vitamin_large_384.datacomp1b_clip", **model_kwargs)
        state = torch.load(LOCAL_VITAMIN_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
        trunk_state = {
            key.removeprefix("visual.trunk."): value
            for key, value in state.items()
            if key.startswith("visual.trunk.")
        }
        if feature_mode == "dense":
            for suffix in ("weight", "bias"):
                pooled_key = f"fc_norm.{suffix}"
                dense_key = f"norm.{suffix}"
                if pooled_key in trunk_state and dense_key not in trunk_state:
                    trunk_state[dense_key] = trunk_state.pop(pooled_key)
        missing, unexpected = model.load_state_dict(trunk_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"ViTamin checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        return model
    model_kwargs = {"pretrained": not bool(checkpoint_path), "num_classes": 0}
    if feature_mode == "dense":
        model_kwargs["global_pool"] = ""
    model = timm.create_model(model_name, **model_kwargs)
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing local backbone checkpoint: {checkpoint}")
        incompatible = load_checkpoint(model, str(checkpoint), strict=False)
        allowed_fragments = ("head", "classifier", "fc.", "attn_mask", "attn_pool")
        missing = [
            key
            for key in incompatible.missing_keys
            if not any(fragment in key for fragment in allowed_fragments)
        ]
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not any(fragment in key for fragment in allowed_fragments)
        ]
        if missing or unexpected:
            raise ValueError(
                "Local checkpoint trunk mismatch: "
                f"missing={missing[:20]} unexpected={unexpected[:20]}"
            )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract resumable dense Task7 feature banks from timm backbones.")
    parser.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Optional local timm checkpoint; avoids remote weight download.",
    )
    parser.add_argument("--feature-mode", choices=("dense", "pooled"), default="dense")
    parser.add_argument("--domains", default="old_data,third_batch")
    parser.add_argument("--views", default="whole,crop")
    parser.add_argument("--image-size", type=int, default=0, help="0 uses the model's configured input size")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grayworld_balance(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    channel_means = array.reshape(-1, 3).mean(axis=0)
    target = float(channel_means.mean())
    scales = target / np.maximum(channel_means, 1.0)
    balanced = np.clip(array * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    return Image.fromarray(balanced, mode="RGB")


def make_view(image: Image.Image, view_name: str) -> Image.Image:
    if view_name == "whole":
        return image
    if view_name == "crop":
        return extract_specimen_crop(image)
    if view_name == "crop_autocontrast":
        return ImageOps.autocontrast(extract_specimen_crop(image), cutoff=1)
    if view_name == "crop_unsharp":
        return extract_specimen_crop(image).filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3)
        )
    if view_name == "crop_tight":
        return extract_specimen_crop(image, margin_ratio=0.02)
    if view_name == "masked_gray":
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask = detect_specimen_mask(rgb) > 0
        masked = np.full_like(rgb, 127)
        masked[mask] = rgb[mask]
        return Image.fromarray(masked, mode="RGB")
    if view_name == "background_only":
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        mask = detect_specimen_mask(rgb) > 0
        rgb[mask] = 127
        return Image.fromarray(rgb, mode="RGB")
    if view_name == "grayworld":
        return grayworld_balance(image)
    if view_name == "crop_grayworld":
        return grayworld_balance(extract_specimen_crop(image))
    if view_name == "autocontrast":
        return ImageOps.autocontrast(image)
    if view_name.startswith("crop_q"):
        quadrant = int(view_name[-1])
        if quadrant not in range(4):
            raise ValueError(f"Unsupported crop quadrant: {view_name}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bbox = expand_bbox(detect_specimen_bbox(rgb), rgb.shape[:2], margin_ratio=0.02)
        x1, y1, x2, y2 = bbox
        specimen = rgb[y1:y2, x1:x2]
        height, width = specimen.shape[:2]
        tile_height = max(1, int(round(height * 0.70)))
        tile_width = max(1, int(round(width * 0.70)))
        start_y = 0 if quadrant < 2 else max(0, height - tile_height)
        start_x = 0 if quadrant % 2 == 0 else max(0, width - tile_width)
        tile = specimen[start_y : start_y + tile_height, start_x : start_x + tile_width]
        return Image.fromarray(tile, mode="RGB")
    raise ValueError(f"Unsupported view: {view_name}")


class DenseFeatureDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        row_indices: list[int],
        view_names: list[str],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.row_indices = row_indices
        self.view_names = view_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int):
        row_index = int(self.row_indices[index])
        row = self.records.iloc[row_index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        view_tensors = [self.transform(make_view(image, name)) for name in self.view_names]
        return row_index, torch.stack(view_tensors, dim=0)


def model_image_size(model: torch.nn.Module, override: int) -> int:
    if override > 0:
        return int(override)
    configured = getattr(getattr(model, "patch_embed", None), "img_size", None)
    if isinstance(configured, tuple):
        configured = configured[0]
    if configured:
        return int(configured)
    return int(getattr(model, "default_cfg", {}).get("input_size", (3, 224, 224))[-1])


def model_normalization(model: torch.nn.Module) -> tuple[list[float], list[float]]:
    config = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
    mean = [float(item) for item in config.get("mean", (0.485, 0.456, 0.406))]
    std = [float(item) for item in config.get("std", (0.229, 0.224, 0.225))]
    return mean, std


def unwrap_feature_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        preferred = ["x_norm_patchtokens", "last_hidden_state", "features", "x"]
        for key in preferred:
            if key in value and isinstance(value[key], torch.Tensor):
                return value[key]
        tensors = [item for item in value.values() if isinstance(item, torch.Tensor)]
        if tensors:
            return tensors[-1]
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if tensors:
            return tensors[-1]
    raise TypeError(f"Could not identify dense tensor in forward_features output {type(value)!r}")


def dense_patch_tokens(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    dense = unwrap_feature_tensor(model.forward_features(inputs))
    feature_dim = int(getattr(model, "num_features", 0))
    if dense.ndim == 3:
        prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
        if prefix_tokens:
            dense = dense[:, prefix_tokens:, :]
        return dense
    if dense.ndim == 4:
        if dense.shape[1] == feature_dim:
            return dense.flatten(2).transpose(1, 2)
        if dense.shape[-1] == feature_dim:
            return dense.reshape(dense.shape[0], -1, dense.shape[-1])
    raise ValueError(f"Unsupported dense feature shape: {tuple(dense.shape)}")


def extract_feature_tokens(model: torch.nn.Module, inputs: torch.Tensor, feature_mode: str) -> torch.Tensor:
    if feature_mode == "dense":
        return dense_patch_tokens(model, inputs)
    pooled = model(inputs)
    if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
        raise ValueError(f"Unsupported pooled feature output: {type(pooled)!r} shape={getattr(pooled, 'shape', None)}")
    return pooled.unsqueeze(1)


def load_records(registry_csv: str, domains: list[str], max_cases: int | None) -> pd.DataFrame:
    records = pd.read_csv(registry_csv, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig")
    records = records[records["domain"].astype(str).isin(domains)].copy()
    if "image_exists" in records.columns:
        truthy = records["image_exists"].astype(str).str.lower().isin(["true", "1", "yes"])
        records = records[truthy].copy()
    records = records.sort_values(["domain", "case_id"]).drop_duplicates("case_id").reset_index(drop=True)
    if max_cases is not None:
        records = records.head(int(max_cases)).copy()
    if records.empty:
        raise ValueError("No registry rows remain after domain filtering.")
    if records["case_id"].duplicated().any():
        raise ValueError("Feature registry must have one row per case_id.")
    missing = [path for path in records["image_path"].astype(str) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing image paths: {missing[:10]}")
    return records


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    view_names = [item.strip() for item in args.views.split(",") if item.strip()]
    if not view_names:
        raise ValueError("At least one view is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "dense_features.float16.npy"
    processed_path = output_dir / "processed.uint8.npy"
    metadata_path = output_dir / "metadata.csv"
    config_path = output_dir / "feature_bank_config.json"
    if args.overwrite:
        for path in (feature_path, processed_path, metadata_path, config_path):
            if path.exists():
                path.unlink()

    records = load_records(args.registry_csv, domains, args.max_cases)
    metadata_columns = [
        "domain",
        "dataset_role",
        "case_id",
        "original_case_id",
        "source_dataset",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "risk_label",
        "image_name",
        "image_path",
        "master_fold_id",
        "source_case_folder",
        "source_folder",
        "selection_rule",
        "image_count",
        "original_image_count",
        "is_frozen_external",
    ]
    available_columns = [column for column in metadata_columns if column in records.columns]
    metadata = records[available_columns].copy()
    metadata.insert(0, "feature_row", np.arange(len(metadata), dtype=int))

    device = torch.device(args.device)
    model = create_backbone(args.model_name, args.feature_mode, args.checkpoint_path)
    model.eval().to(device)
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
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            dummy_tokens = extract_feature_tokens(model, dummy, args.feature_mode)
    token_count = int(dummy_tokens.shape[1])
    feature_dim = int(dummy_tokens.shape[2])
    expected_shape = (len(metadata), len(view_names), token_count, feature_dim)
    config = {
        "registry_csv": str(Path(args.registry_csv).resolve()),
        "domains": domains,
        "model_name": args.model_name,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()) if args.checkpoint_path else "",
        "feature_mode": args.feature_mode,
        "local_checkpoint": str(LOCAL_VITAMIN_CHECKPOINT) if args.model_name == "local_vitamin_large_384" else "",
        "views": view_names,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "feature_shape": list(expected_shape),
        "dtype": "float16",
        "seed": int(args.seed),
        "complete": False,
    }

    if feature_path.exists() or processed_path.exists():
        if not feature_path.exists() or not processed_path.exists() or not config_path.exists():
            raise RuntimeError("Incomplete feature-bank files found; use --overwrite to rebuild.")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        keys = [
            "domains",
            "model_name",
            "checkpoint_path",
            "feature_mode",
            "views",
            "image_size",
            "feature_shape",
            "dtype",
        ]
        mismatches = {key: (existing_config.get(key), config.get(key)) for key in keys if existing_config.get(key) != config.get(key)}
        if mismatches:
            raise ValueError(f"Existing feature bank configuration mismatch: {mismatches}")
        feature_map = np.lib.format.open_memmap(feature_path, mode="r+")
        processed_map = np.lib.format.open_memmap(processed_path, mode="r+")
    else:
        feature_map = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=expected_shape)
        processed_map = np.lib.format.open_memmap(processed_path, mode="w+", dtype=np.uint8, shape=(len(metadata),))
        processed_map[:] = 0
        feature_map.flush()
        processed_map.flush()
        metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
        write_json(config_path, config)

    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int).tolist()
    print(
        f"Feature bank model={args.model_name} cases={len(metadata)} remaining={len(remaining)} "
        f"views={view_names} image_size={image_size} tokens={token_count} dim={feature_dim}",
        flush=True,
    )
    if remaining:
        dataset = DenseFeatureDataset(records, remaining, view_names, transform)
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": int(args.batch_size),
            "shuffle": False,
            "num_workers": int(args.num_workers),
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(**loader_kwargs)
        progress = tqdm(loader, desc="dense features", dynamic_ncols=True, file=sys.stdout)
        for row_indices, view_batch in progress:
            batch_size, num_views, channels, height, width = view_batch.shape
            flat_inputs = view_batch.reshape(batch_size * num_views, channels, height, width).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"
            ):
                tokens = extract_feature_tokens(model, flat_inputs, args.feature_mode)
            tokens = tokens.reshape(batch_size, num_views, token_count, feature_dim)
            indices = row_indices.detach().cpu().numpy().astype(int)
            feature_map[indices] = tokens.detach().cpu().numpy().astype(np.float16, copy=False)
            processed_map[indices] = 1
            feature_map.flush()
            processed_map.flush()
            progress.set_postfix(done=int(processed_map.sum()), total=len(processed_map))
        progress.close()

    config["complete"] = bool(np.asarray(processed_map).all())
    config["completed_cases"] = int(np.asarray(processed_map).sum())
    write_json(config_path, config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if not config["complete"]:
        raise RuntimeError("Feature bank extraction finished with unprocessed rows.")


if __name__ == "__main__":
    main()
