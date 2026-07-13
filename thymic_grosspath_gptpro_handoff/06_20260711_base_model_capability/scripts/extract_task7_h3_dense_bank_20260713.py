from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from extract_task7_h3_representation_bank_20260713 import (
    DEFAULT_VIEWS,
    choose_patch_grid,
    image_to_unit_tensor,
    load_local_radio_model,
    load_records,
    make_view,
    metadata_from_records,
    set_seed,
    validate_views,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resumable masked H3B dense-token bank.")
    parser.add_argument(
        "--backend", choices=("siglip2", "siglip_fixed", "radio", "pe"), required=True
    )
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--canonical-model-id", required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--model-code-dir", default="")
    parser.add_argument("--code-revision", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", default="old_data,third_batch")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--max-num-patches", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


@dataclass
class ViewFeatures:
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    spatial_shape: tuple[int, int]


class Adapter(Protocol):
    feature_dim: int
    patch_size: int
    model_parameter_count: int
    model_dtype: str
    adapter_config: dict[str, Any]

    def extract(self, image: Image.Image) -> ViewFeatures: ...


class Siglip2Adapter:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from transformers import Siglip2ImageProcessor, Siglip2VisionModel

        self.device = device
        self.max_num_patches = int(args.max_num_patches)
        self.processor = Siglip2ImageProcessor.from_pretrained(
            args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
        self.model = Siglip2VisionModel.from_pretrained(
            args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).eval().to(device)
        self.feature_dim = int(self.model.config.hidden_size)
        self.patch_size = int(self.processor.patch_size)
        self.model_parameter_count = int(sum(p.numel() for p in self.model.parameters()))
        self.model_dtype = str(next(self.model.parameters()).dtype)
        self.adapter_config = {
            "input_geometry": "official SigLIP2 NaFlex native aspect",
            "processor_class": self.processor.__class__.__name__,
        }

    def extract(self, image: Image.Image) -> ViewFeatures:
        batch = self.processor(
            images=image,
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
        )
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if key in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        }
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(**inputs)
        tokens = output.last_hidden_state[0].float()
        mask = inputs["pixel_attention_mask"][0].bool()
        tokens = tokens.masked_fill(~mask[:, None], 0.0)
        shape = tuple(int(value) for value in inputs["spatial_shapes"][0].tolist())
        return ViewFeatures(tokens.cpu(), mask.cpu(), shape)


class FixedSiglipAdapter:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.device = device
        self.processor = SiglipImageProcessor.from_pretrained(
            args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
        self.model = SiglipVisionModel.from_pretrained(
            args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).eval().to(device)
        self.feature_dim = int(self.model.config.hidden_size)
        self.patch_size = int(self.model.config.patch_size)
        self.image_size = int(self.model.config.image_size)
        self.grid_side = self.image_size // self.patch_size
        self.model_parameter_count = int(sum(p.numel() for p in self.model.parameters()))
        self.model_dtype = str(next(self.model.parameters()).dtype)
        self.adapter_config = {
            "input_geometry": "official fixed-square preprocessing",
            "image_size": self.image_size,
            "processor_class": self.processor.__class__.__name__,
        }

    def extract(self, image: Image.Image) -> ViewFeatures:
        batch = self.processor(images=image, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            tokens = self.model(pixel_values=pixel_values).last_hidden_state[0].float()
        mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        return ViewFeatures(tokens.cpu(), mask.cpu(), (self.grid_side, self.grid_side))


class RadioAdapter:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.device = device
        self.max_num_patches = int(args.max_num_patches)
        self.model = load_local_radio_model(
            args.model_id,
            args.revision,
            args.local_files_only,
            torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).eval().to(device)
        self.patch_size = int(self.model.patch_size)
        self.feature_dim = int(self.model.model.embed_dim)
        self.max_side_patches = int(self.model.max_resolution) // self.patch_size
        self.model_parameter_count = int(sum(p.numel() for p in self.model.parameters()))
        self.model_dtype = str(next(self.model.parameters()).dtype)
        self.adapter_config = {
            "input_geometry": "native aspect with at least 94% token utilization",
            "resize_interpolation": "PIL.Image.Resampling.BICUBIC",
            "input_range": "0_to_1; official RADIO internal conditioner",
        }

    def extract(self, image: Image.Image) -> ViewFeatures:
        rows, columns = choose_patch_grid(
            image.width,
            image.height,
            self.max_num_patches,
            self.max_side_patches,
        )
        model_input = image_to_unit_tensor(
            image, rows, columns, self.patch_size, self.device
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            tokens = self.model(model_input).features[0].float()
        mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        return ViewFeatures(tokens.cpu(), mask.cpu(), (rows, columns))


class PeAdapter:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        checkpoint = Path(args.model_id).resolve()
        code_dir = Path(args.model_code_dir).resolve()
        if not args.code_revision:
            raise ValueError("--code-revision is required for PE")
        if not (code_dir / "core" / "vision_encoder" / "pe.py").is_file():
            raise FileNotFoundError(f"Missing official PE source: {code_dir}")
        sys.path.insert(0, str(code_dir))
        from core.vision_encoder.pe import VisionTransformer

        self.device = device
        self.max_num_patches = int(args.max_num_patches)
        self.model = VisionTransformer.from_config(
            "PE-Spatial-L14-448",
            pretrained=True,
            checkpoint_path=str(checkpoint),
        ).eval().to(device)
        self.patch_size = int(self.model.patch_size)
        self.feature_dim = int(self.model.width)
        self.max_side_patches = max(32, int(round(self.max_num_patches**0.5)) * 4)
        self.model_parameter_count = int(sum(p.numel() for p in self.model.parameters()))
        self.model_dtype = str(next(self.model.parameters()).dtype)
        self.adapter_config = {
            "input_geometry": "native aspect with at least 94% token utilization",
            "resize_interpolation": "PIL.Image.Resampling.BILINEAR",
            "input_normalization": "(RGB/255 - 0.5) / 0.5",
            "feature_layer": "PE-Spatial aligned final dense layer",
            "model_code_dir": str(code_dir),
            "code_revision": args.code_revision,
        }

    def extract(self, image: Image.Image) -> ViewFeatures:
        rows, columns = choose_patch_grid(
            image.width,
            image.height,
            self.max_num_patches,
            self.max_side_patches,
        )
        model_input = image_to_unit_tensor(
            image,
            rows,
            columns,
            self.patch_size,
            self.device,
            resample=Image.Resampling.BILINEAR,
        ).sub(0.5).div(0.5)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            tokens = self.model.forward_features(
                model_input,
                norm=True,
                strip_cls_token=True,
            )[0].float()
        mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        return ViewFeatures(tokens.cpu(), mask.cpu(), (rows, columns))


def make_adapter(args: argparse.Namespace, device: torch.device) -> Adapter:
    if args.backend == "siglip2":
        return Siglip2Adapter(args, device)
    if args.backend == "siglip_fixed":
        return FixedSiglipAdapter(args, device)
    if args.backend == "radio":
        return RadioAdapter(args, device)
    return PeAdapter(args, device)


def remove_outputs(output_dir: Path) -> None:
    for name in (
        "dense_features.float16.npy",
        "valid_token_mask.uint8.npy",
        "spatial_shapes.int16.npy",
        "processed.uint8.npy",
        "metadata.csv",
        "dense_bank_config.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def open_maps(
    output_dir: Path,
    metadata,
    config: dict[str, Any],
    overwrite: bool,
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        remove_outputs(output_dir)
    paths = {
        "dense": output_dir / "dense_features.float16.npy",
        "mask": output_dir / "valid_token_mask.uint8.npy",
        "shape": output_dir / "spatial_shapes.int16.npy",
        "processed": output_dir / "processed.uint8.npy",
        "metadata": output_dir / "metadata.csv",
        "config": output_dir / "dense_bank_config.json",
    }
    if any(path.exists() for path in paths.values()):
        if not all(path.exists() for path in paths.values()):
            raise RuntimeError("Partial dense bank found; rerun with --overwrite")
        existing = json.loads(paths["config"].read_text(encoding="utf-8"))
        keys = (
            "backend",
            "canonical_model_id",
            "weight_sha256",
            "views",
            "max_num_patches",
            "feature_dim",
            "dense_shape",
        )
        mismatch = {
            key: (existing.get(key), config.get(key))
            for key in keys
            if existing.get(key) != config.get(key)
        }
        if mismatch:
            raise ValueError(f"Existing dense bank differs: {mismatch}")
        return (
            np.lib.format.open_memmap(paths["dense"], mode="r+"),
            np.lib.format.open_memmap(paths["mask"], mode="r+"),
            np.lib.format.open_memmap(paths["shape"], mode="r+"),
            np.lib.format.open_memmap(paths["processed"], mode="r+"),
        )

    dense_shape = tuple(config["dense_shape"])
    required_bytes = int(np.prod(dense_shape)) * np.dtype(np.float16).itemsize
    required_bytes += int(np.prod(dense_shape[:-1])) * np.dtype(np.uint8).itemsize
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < int(required_bytes * 1.10):
        raise RuntimeError(
            f"Dense bank needs about {required_bytes / 2**30:.2f} GiB plus headroom; "
            f"only {free_bytes / 2**30:.2f} GiB is free"
        )
    dense = np.lib.format.open_memmap(
        paths["dense"], mode="w+", dtype=np.float16, shape=dense_shape
    )
    mask = np.lib.format.open_memmap(
        paths["mask"], mode="w+", dtype=np.uint8, shape=dense_shape[:-1]
    )
    shape = np.lib.format.open_memmap(
        paths["shape"],
        mode="w+",
        dtype=np.int16,
        shape=(len(metadata), len(DEFAULT_VIEWS), 2),
    )
    processed = np.lib.format.open_memmap(
        paths["processed"], mode="w+", dtype=np.uint8, shape=(len(metadata),)
    )
    mask[:] = 0
    shape[:] = 0
    processed[:] = 0
    for item in (mask, shape, processed):
        item.flush()
    metadata.to_csv(paths["metadata"], index=False, encoding="utf-8-sig")
    write_json(paths["config"], config)
    return dense, mask, shape, processed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    views = validate_views(args.views)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    records = load_records(args.registry_csv, domains, args.max_cases)
    metadata = metadata_from_records(records)
    started_at = time.monotonic()
    adapter = make_adapter(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    config: dict[str, Any] = {
        "backend": args.backend,
        "model_id": args.model_id,
        "canonical_model_id": args.canonical_model_id,
        "weight_sha256": args.weight_sha256,
        "revision": args.revision,
        "registry_csv": str(Path(args.registry_csv).resolve()),
        "domains": domains,
        "views": views,
        "max_num_patches": int(args.max_num_patches),
        "feature_dim": int(adapter.feature_dim),
        "patch_size": int(adapter.patch_size),
        "model_parameter_count": int(adapter.model_parameter_count),
        "model_dtype": adapter.model_dtype,
        "dense_shape": [
            len(metadata),
            len(views),
            int(args.max_num_patches),
            int(adapter.feature_dim),
        ],
        "dtype": "float16",
        "mask_dtype": "uint8",
        "seed": int(args.seed),
        "adapter": adapter.adapter_config,
        "complete": False,
    }
    maps = open_maps(Path(args.output_dir), metadata, config, args.overwrite)
    dense_map, mask_map, shape_map, processed_map = maps
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    progress = tqdm(remaining, desc=f"H3B {args.canonical_model_id}", dynamic_ncols=True)
    for progress_index, index in enumerate(progress, start=1):
        row = records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        dense_map[index] = 0
        mask_map[index] = 0
        for view_index, view_name in enumerate(views):
            extracted = adapter.extract(make_view(image, view_name))
            token_count = int(extracted.tokens.shape[0])
            if extracted.tokens.shape[1] != adapter.feature_dim:
                raise RuntimeError(f"Unexpected feature dim: {tuple(extracted.tokens.shape)}")
            if token_count > args.max_num_patches or len(extracted.valid_mask) != token_count:
                raise RuntimeError(f"Invalid token output: {token_count}")
            valid_count = int(extracted.valid_mask.sum())
            if valid_count != int(np.prod(extracted.spatial_shape)):
                raise RuntimeError(
                    f"Mask/grid mismatch: {valid_count} vs {extracted.spatial_shape}"
                )
            dense_map[index, view_index, :token_count] = (
                extracted.tokens.numpy().astype(np.float16)
            )
            mask_map[index, view_index, :token_count] = (
                extracted.valid_mask.numpy().astype(np.uint8)
            )
            shape_map[index, view_index] = extracted.spatial_shape
        processed_map[index] = 1
        if progress_index % 5 == 0:
            for item in maps:
                item.flush()
    for item in maps:
        item.flush()
    config["complete"] = bool(np.asarray(processed_map).all())
    config["completed_cases"] = int(np.asarray(processed_map).sum())
    config["elapsed_seconds"] = float(time.monotonic() - started_at)
    config["valid_token_count_min"] = int(np.asarray(mask_map).sum(axis=-1).min())
    config["valid_token_count_max"] = int(np.asarray(mask_map).sum(axis=-1).max())
    if device.type == "cuda":
        config["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        config["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    write_json(Path(args.output_dir) / "dense_bank_config.json", config)
    print(json.dumps({"status": "complete", "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
