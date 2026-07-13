from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from extract_task7_dense_token_bank_20260711 import load_records, make_view


DEFAULT_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
STATISTICS = ("mean", "std", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact, resumable H3 representation-screen feature banks."
    )
    parser.add_argument(
        "--backend",
        choices=("existing_dense", "siglip2", "siglip_fixed", "radio", "pe"),
        required=True,
    )
    parser.add_argument("--registry-csv", default="")
    parser.add_argument("--source-feature-bank", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-code-dir", default="")
    parser.add_argument("--code-revision", default="")
    parser.add_argument("--canonical-model-id", default="")
    parser.add_argument("--weight-sha256", default="")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", default="old_data,third_batch")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--max-num-patches", type=int, default=1024)
    parser.add_argument("--view-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metadata_from_records(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    available = [column for column in columns if column in records.columns]
    metadata = records[available].copy()
    metadata.insert(0, "feature_row", np.arange(len(metadata), dtype=int))
    return metadata


def validate_views(raw: str) -> list[str]:
    views = [item.strip() for item in raw.split(",") if item.strip()]
    if tuple(views) != DEFAULT_VIEWS:
        raise ValueError(f"H3 view order is locked to {DEFAULT_VIEWS}, received {views}")
    return views


def remove_outputs(output_dir: Path) -> None:
    for name in (
        "summary_features.float16.npy",
        "valid_patch_counts.uint16.npy",
        "spatial_shapes.int16.npy",
        "processed.uint8.npy",
        "metadata.csv",
        "representation_bank_config.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def open_maps(
    output_dir: Path,
    metadata: pd.DataFrame,
    config: dict[str, Any],
    overwrite: bool,
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        remove_outputs(output_dir)
    summary_path = output_dir / "summary_features.float16.npy"
    count_path = output_dir / "valid_patch_counts.uint16.npy"
    shape_path = output_dir / "spatial_shapes.int16.npy"
    processed_path = output_dir / "processed.uint8.npy"
    config_path = output_dir / "representation_bank_config.json"
    metadata_path = output_dir / "metadata.csv"
    required = (summary_path, count_path, shape_path, processed_path, config_path, metadata_path)
    if any(path.exists() for path in required):
        if not all(path.exists() for path in required):
            raise RuntimeError("Partial representation bank found; rerun with --overwrite")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        keys = (
            "backend",
            "model_id",
            "canonical_model_id",
            "weight_sha256",
            "code_revision",
            "revision",
            "views",
            "statistics",
            "max_num_patches",
            "summary_shape",
            "feature_dim",
        )
        mismatch = {
            key: (existing.get(key), config.get(key))
            for key in keys
            if existing.get(key) != config.get(key)
        }
        if mismatch:
            raise ValueError(f"Existing representation bank differs: {mismatch}")
        return (
            np.lib.format.open_memmap(summary_path, mode="r+"),
            np.lib.format.open_memmap(count_path, mode="r+"),
            np.lib.format.open_memmap(shape_path, mode="r+"),
            np.lib.format.open_memmap(processed_path, mode="r+"),
        )

    summary_map = np.lib.format.open_memmap(
        summary_path,
        mode="w+",
        dtype=np.float16,
        shape=tuple(config["summary_shape"]),
    )
    count_map = np.lib.format.open_memmap(
        count_path,
        mode="w+",
        dtype=np.uint16,
        shape=(len(metadata), len(DEFAULT_VIEWS)),
    )
    shape_map = np.lib.format.open_memmap(
        shape_path,
        mode="w+",
        dtype=np.int16,
        shape=(len(metadata), len(DEFAULT_VIEWS), 2),
    )
    processed_map = np.lib.format.open_memmap(
        processed_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(metadata),),
    )
    summary_map[:] = 0
    count_map[:] = 0
    shape_map[:] = 0
    processed_map[:] = 0
    for item in (summary_map, count_map, shape_map, processed_map):
        item.flush()
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    write_json(config_path, config)
    return summary_map, count_map, shape_map, processed_map


def summarize_numpy(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float32)
    return np.stack(
        [values.mean(axis=0), values.std(axis=0), values.max(axis=0)],
        axis=0,
    ).astype(np.float16)


def extract_existing_dense(args: argparse.Namespace, views: list[str]) -> None:
    source_dir = Path(args.source_feature_bank)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source feature bank: {source_dir}")
    source_config = json.loads(
        (source_dir / "feature_bank_config.json").read_text(encoding="utf-8")
    )
    if tuple(source_config.get("views", [])) != tuple(views):
        raise ValueError("Source feature bank view order differs from H3")
    features = np.load(source_dir / "dense_features.float16.npy", mmap_mode="r")
    metadata = pd.read_csv(
        source_dir / "metadata.csv",
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    if args.max_cases is not None:
        metadata = metadata.head(int(args.max_cases)).copy()
        features = features[: len(metadata)]
    metadata = metadata.reset_index(drop=True)
    metadata["feature_row"] = np.arange(len(metadata), dtype=int)
    feature_dim = int(features.shape[-1])
    config = {
        "backend": "existing_dense",
        "model_id": source_config.get("model_name", "existing_dense"),
        "revision": "existing-immutable-bank",
        "source_feature_bank": str(source_dir.resolve()),
        "views": views,
        "statistics": list(STATISTICS),
        "max_num_patches": int(features.shape[2]),
        "feature_dim": feature_dim,
        "summary_shape": [len(metadata), len(views), len(STATISTICS), feature_dim],
        "dtype": "float16",
        "seed": int(args.seed),
        "complete": False,
    }
    maps = open_maps(Path(args.output_dir), metadata, config, args.overwrite)
    summary_map, count_map, shape_map, processed_map = maps
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    token_count = int(features.shape[2])
    side = int(round(token_count**0.5))
    spatial_shape = (side, side) if side * side == token_count else (1, token_count)
    for index in tqdm(remaining, desc="H3 existing dense summary", dynamic_ncols=True):
        for view_index in range(len(views)):
            summary_map[index, view_index] = summarize_numpy(features[index, view_index])
            count_map[index, view_index] = token_count
            shape_map[index, view_index] = spatial_shape
        processed_map[index] = 1
        if (index + 1) % 20 == 0:
            for item in maps:
                item.flush()
    for item in maps:
        item.flush()
    config["complete"] = bool(np.asarray(processed_map).all())
    config["completed_cases"] = int(np.asarray(processed_map).sum())
    write_json(Path(args.output_dir) / "representation_bank_config.json", config)


def masked_statistics(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = tokens.float()
    valid = mask.to(dtype=values.dtype).unsqueeze(-1)
    count = valid.sum(dim=1).clamp_min(1.0)
    mean = (values * valid).sum(dim=1) / count
    second = (values.square() * valid).sum(dim=1) / count
    std = (second - mean.square()).clamp_min(0.0).sqrt()
    maximum = values.masked_fill(~mask.bool().unsqueeze(-1), -torch.inf).amax(dim=1)
    return torch.stack((mean, std, maximum), dim=1)


def choose_patch_grid(
    width: int,
    height: int,
    max_num_patches: int,
    max_side_patches: int,
) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {(width, height)}")
    aspect = float(width) / float(height)
    candidates: list[tuple[float, int, int, int]] = []
    max_rows = min(max_side_patches, max_num_patches)
    for rows in range(1, max_rows + 1):
        max_columns = min(max_side_patches, max_num_patches // rows)
        if max_columns < 1:
            continue
        ideal = rows * aspect
        columns_to_try = {
            1,
            max_columns,
            max(1, min(max_columns, int(np.floor(ideal)))),
            max(1, min(max_columns, int(np.ceil(ideal)))),
            max(1, min(max_columns, int(round(ideal)))),
        }
        for columns in columns_to_try:
            tokens = rows * columns
            aspect_error = abs(np.log((float(columns) / float(rows)) / aspect))
            candidates.append((float(aspect_error), -tokens, rows, columns))
    if not candidates:
        raise RuntimeError("Could not construct a valid native-aspect patch grid")
    minimum_tokens = int(np.ceil(0.94 * max_num_patches))
    high_utilization = [item for item in candidates if -item[1] >= minimum_tokens]
    if high_utilization:
        best = min(high_utilization)
    else:
        best = min(
            candidates,
            key=lambda item: (
                item[0] + 0.05 * (1.0 + item[1] / float(max_num_patches)),
                item[1],
                item[2],
                item[3],
            ),
        )
    return best[2], best[3]


def image_to_unit_tensor(
    image: Image.Image,
    rows: int,
    columns: int,
    patch_size: int,
    device: torch.device,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> torch.Tensor:
    resized = image.resize(
        (columns * patch_size, rows * patch_size),
        resample=resample,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device, non_blocking=True)


def load_local_radio_model(
    model_dir: str,
    revision: str,
    local_files_only: bool,
    dtype: torch.dtype,
) -> torch.nn.Module:
    directory = Path(model_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"RADIO local model directory is missing: {directory}")
    package_name = "_h3_radio_model_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.hf_model")
    config = module.RADIOConfig.from_pretrained(
        directory,
        revision=revision,
        local_files_only=local_files_only,
    )
    return module.RADIOModel.from_pretrained(
        directory,
        config=config,
        revision=revision,
        local_files_only=local_files_only,
        dtype=dtype,
    )


def extract_siglip2(args: argparse.Namespace, views: list[str]) -> None:
    started_at = time.monotonic()
    if not args.registry_csv:
        raise ValueError("--registry-csv is required for siglip2 extraction")
    if not args.model_id:
        raise ValueError("--model-id is required for siglip2 extraction")
    if "siglip2" not in args.model_id.lower():
        raise ValueError("The siglip2 backend requires an official SigLIP2 checkpoint")
    if args.view_batch_size < 1:
        raise ValueError("--view-batch-size must be positive")

    from transformers import Siglip2ImageProcessor, Siglip2VisionModel

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    records = load_records(args.registry_csv, domains, args.max_cases)
    metadata = metadata_from_records(records)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    processor = Siglip2ImageProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = Siglip2VisionModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feature_dim = int(model.config.hidden_size)
    config = {
        "backend": "siglip2",
        "model_id": args.model_id,
        "canonical_model_id": args.canonical_model_id or args.model_id,
        "weight_sha256": args.weight_sha256 or None,
        "revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_processor_commit": getattr(processor, "_commit_hash", None),
        "registry_csv": str(Path(args.registry_csv).resolve()),
        "domains": domains,
        "views": views,
        "statistics": list(STATISTICS),
        "patch_size": int(processor.patch_size),
        "max_num_patches": int(args.max_num_patches),
        "feature_dim": feature_dim,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "summary_shape": [len(metadata), len(views), len(STATISTICS), feature_dim],
        "dtype": "float16",
        "model_dtype": str(next(model.parameters()).dtype),
        "seed": int(args.seed),
        "complete": False,
    }
    maps = open_maps(Path(args.output_dir), metadata, config, args.overwrite)
    summary_map, count_map, shape_map, processed_map = maps
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    progress = tqdm(remaining, desc=f"H3 {args.model_id}", dynamic_ncols=True)
    for progress_index, index in enumerate(progress, start=1):
        row = records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        view_images = [make_view(image, name) for name in views]
        for start in range(0, len(view_images), args.view_batch_size):
            stop = min(len(view_images), start + args.view_batch_size)
            batch = processor(
                images=view_images[start:stop],
                return_tensors="pt",
                max_num_patches=args.max_num_patches,
            )
            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
            }
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(**model_inputs)
            stats = masked_statistics(
                output.last_hidden_state,
                model_inputs["pixel_attention_mask"],
            )
            summary_map[index, start:stop] = stats.cpu().numpy().astype(np.float16)
            count_map[index, start:stop] = (
                model_inputs["pixel_attention_mask"].sum(dim=1).cpu().numpy().astype(np.uint16)
            )
            shape_map[index, start:stop] = (
                model_inputs["spatial_shapes"].cpu().numpy().astype(np.int16)
            )
            del batch, model_inputs, output, stats
        processed_map[index] = 1
        if progress_index % 10 == 0:
            for item in maps:
                item.flush()
    for item in maps:
        item.flush()
    config["complete"] = bool(np.asarray(processed_map).all())
    config["completed_cases"] = int(np.asarray(processed_map).sum())
    config["elapsed_seconds"] = float(time.monotonic() - started_at)
    if device.type == "cuda":
        config["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        config["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    write_json(Path(args.output_dir) / "representation_bank_config.json", config)


def extract_siglip_fixed(args: argparse.Namespace, views: list[str]) -> None:
    started_at = time.monotonic()
    if not args.registry_csv:
        raise ValueError("--registry-csv is required for fixed SigLIP extraction")
    if not args.model_id or "siglip" not in args.model_id.lower():
        raise ValueError("The siglip_fixed backend requires a SigLIP-family checkpoint")
    if args.view_batch_size != 1:
        raise ValueError("Fixed SigLIP extraction is locked to --view-batch-size 1")

    from transformers import SiglipImageProcessor, SiglipVisionModel

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    records = load_records(args.registry_csv, domains, args.max_cases)
    metadata = metadata_from_records(records)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    processor = SiglipImageProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = SiglipVisionModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feature_dim = int(model.config.hidden_size)
    image_size = int(model.config.image_size)
    patch_size = int(model.config.patch_size)
    grid_side = image_size // patch_size
    if grid_side * patch_size != image_size:
        raise ValueError(f"Image size {image_size} is not divisible by patch size {patch_size}")
    config = {
        "backend": "siglip_fixed",
        "model_id": args.model_id,
        "canonical_model_id": args.canonical_model_id or args.model_id,
        "weight_sha256": args.weight_sha256 or None,
        "revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_processor_commit": getattr(processor, "_commit_hash", None),
        "registry_csv": str(Path(args.registry_csv).resolve()),
        "domains": domains,
        "views": views,
        "statistics": list(STATISTICS),
        "image_size": image_size,
        "patch_size": patch_size,
        "max_num_patches": grid_side * grid_side,
        "input_geometry": "official fixed-square preprocessing",
        "processor_class": processor.__class__.__name__,
        "feature_dim": feature_dim,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "summary_shape": [len(metadata), len(views), len(STATISTICS), feature_dim],
        "dtype": "float16",
        "model_dtype": str(next(model.parameters()).dtype),
        "seed": int(args.seed),
        "complete": False,
    }
    maps = open_maps(Path(args.output_dir), metadata, config, args.overwrite)
    summary_map, count_map, shape_map, processed_map = maps
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    progress = tqdm(remaining, desc=f"H3 {args.model_id}", dynamic_ncols=True)
    for progress_index, index in enumerate(progress, start=1):
        row = records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        for view_index, view_name in enumerate(views):
            view_image = make_view(image, view_name)
            batch = processor(images=view_image, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                tokens = model(pixel_values=pixel_values).last_hidden_state
            expected_tokens = grid_side * grid_side
            if tokens.shape != (1, expected_tokens, feature_dim):
                raise RuntimeError(
                    f"Unexpected fixed SigLIP tokens {tuple(tokens.shape)}; "
                    f"expected {(1, expected_tokens, feature_dim)}"
                )
            mask = torch.ones((1, expected_tokens), dtype=torch.bool, device=device)
            stats = masked_statistics(tokens, mask)
            summary_map[index, view_index] = stats[0].cpu().numpy().astype(np.float16)
            count_map[index, view_index] = expected_tokens
            shape_map[index, view_index] = (grid_side, grid_side)
            del batch, pixel_values, tokens, mask, stats
        processed_map[index] = 1
        if progress_index % 10 == 0:
            for item in maps:
                item.flush()
    for item in maps:
        item.flush()
    config["complete"] = bool(np.asarray(processed_map).all())
    config["completed_cases"] = int(np.asarray(processed_map).sum())
    config["elapsed_seconds"] = float(time.monotonic() - started_at)
    if device.type == "cuda":
        config["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        config["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    write_json(Path(args.output_dir) / "representation_bank_config.json", config)


def extract_radio(args: argparse.Namespace, views: list[str]) -> None:
    started_at = time.monotonic()
    if not args.registry_csv:
        raise ValueError("--registry-csv is required for RADIO extraction")
    if not args.model_id or "radio" not in args.model_id.lower():
        raise ValueError("The radio backend requires an official RADIO checkpoint")
    if args.view_batch_size != 1:
        raise ValueError("Native-aspect RADIO extraction is locked to --view-batch-size 1")

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    records = load_records(args.registry_csv, domains, args.max_cases)
    metadata = metadata_from_records(records)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_local_radio_model(
        args.model_id,
        args.revision,
        args.local_files_only,
        torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    patch_size = int(model.patch_size)
    max_resolution = int(model.max_resolution)
    max_side_patches = max_resolution // patch_size
    feature_dim = int(model.model.embed_dim)
    config = {
        "backend": "radio",
        "model_id": args.model_id,
        "canonical_model_id": args.canonical_model_id or args.model_id,
        "weight_sha256": args.weight_sha256 or None,
        "revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "registry_csv": str(Path(args.registry_csv).resolve()),
        "domains": domains,
        "views": views,
        "statistics": list(STATISTICS),
        "patch_size": patch_size,
        "max_num_patches": int(args.max_num_patches),
        "max_resolution": max_resolution,
        "preferred_resolution": list(model.preferred_resolution),
        "native_grid_rule": "min-aspect-error-with-at-least-94pct-token-utilization",
        "resize_interpolation": "PIL.Image.Resampling.BICUBIC",
        "input_range": "0_to_1; official RADIO internal conditioner",
        "feature_dim": feature_dim,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "summary_shape": [len(metadata), len(views), len(STATISTICS), feature_dim],
        "dtype": "float16",
        "model_dtype": str(next(model.parameters()).dtype),
        "seed": int(args.seed),
        "complete": False,
    }
    maps = open_maps(Path(args.output_dir), metadata, config, args.overwrite)
    summary_map, count_map, shape_map, processed_map = maps
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    progress = tqdm(remaining, desc=f"H3 {args.model_id}", dynamic_ncols=True)
    for progress_index, index in enumerate(progress, start=1):
        row = records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        for view_index, view_name in enumerate(views):
            view_image = make_view(image, view_name)
            grid_rows, grid_columns = choose_patch_grid(
                view_image.width,
                view_image.height,
                args.max_num_patches,
                max_side_patches,
            )
            model_input = image_to_unit_tensor(
                view_image,
                grid_rows,
                grid_columns,
                patch_size,
                device,
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(model_input)
            tokens = output.features
            expected_tokens = grid_rows * grid_columns
            if tokens.shape != (1, expected_tokens, feature_dim):
                raise RuntimeError(
                    f"Unexpected RADIO tokens {tuple(tokens.shape)} for grid "
                    f"{(grid_rows, grid_columns)}"
                )
            mask = torch.ones((1, expected_tokens), dtype=torch.bool, device=device)
            stats = masked_statistics(tokens, mask)
            summary_map[index, view_index] = stats[0].cpu().numpy().astype(np.float16)
            count_map[index, view_index] = expected_tokens
            shape_map[index, view_index] = (grid_rows, grid_columns)
            del model_input, output, tokens, mask, stats
        processed_map[index] = 1
        if progress_index % 10 == 0:
            for item in maps:
                item.flush()
    for item in maps:
        item.flush()
    config["complete"] = bool(np.asarray(processed_map).all())
    config["completed_cases"] = int(np.asarray(processed_map).sum())
    config["elapsed_seconds"] = float(time.monotonic() - started_at)
    if device.type == "cuda":
        config["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        config["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    write_json(Path(args.output_dir) / "representation_bank_config.json", config)


def extract_pe(args: argparse.Namespace, views: list[str]) -> None:
    started_at = time.monotonic()
    if not args.registry_csv:
        raise ValueError("--registry-csv is required for PE extraction")
    checkpoint_path = Path(args.model_id).resolve()
    if not checkpoint_path.is_file() or "pe-spatial-l14-448" not in checkpoint_path.name.lower():
        raise ValueError("The PE backend requires the official PE-Spatial-L14-448 checkpoint")
    code_dir = Path(args.model_code_dir).resolve()
    if not (code_dir / "core" / "vision_encoder" / "pe.py").is_file():
        raise FileNotFoundError(f"Official Perception Models source is missing: {code_dir}")
    if not args.code_revision:
        raise ValueError("--code-revision is required for PE provenance")
    if args.view_batch_size != 1:
        raise ValueError("Native-aspect PE extraction is locked to --view-batch-size 1")

    sys.path.insert(0, str(code_dir))
    from core.vision_encoder.pe import VisionTransformer

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    records = load_records(args.registry_csv, domains, args.max_cases)
    metadata = metadata_from_records(records)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = VisionTransformer.from_config(
        "PE-Spatial-L14-448",
        pretrained=True,
        checkpoint_path=str(checkpoint_path),
    )
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    patch_size = int(model.patch_size)
    feature_dim = int(model.width)
    max_side_patches = max(32, int(round(args.max_num_patches**0.5)) * 4)
    config = {
        "backend": "pe",
        "model_id": str(checkpoint_path),
        "canonical_model_id": args.canonical_model_id or checkpoint_path.name,
        "weight_sha256": args.weight_sha256 or None,
        "revision": args.revision,
        "model_code_dir": str(code_dir),
        "code_revision": args.code_revision,
        "registry_csv": str(Path(args.registry_csv).resolve()),
        "domains": domains,
        "views": views,
        "statistics": list(STATISTICS),
        "patch_size": patch_size,
        "max_num_patches": int(args.max_num_patches),
        "max_side_patches": int(max_side_patches),
        "native_grid_rule": "min-aspect-error-with-at-least-94pct-token-utilization",
        "resize_interpolation": "PIL.Image.Resampling.BILINEAR",
        "input_normalization": "(RGB/255 - 0.5) / 0.5",
        "feature_layer": "PE-Spatial aligned final dense layer",
        "feature_dim": feature_dim,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "summary_shape": [len(metadata), len(views), len(STATISTICS), feature_dim],
        "dtype": "float16",
        "model_dtype": str(next(model.parameters()).dtype),
        "seed": int(args.seed),
        "complete": False,
    }
    maps = open_maps(Path(args.output_dir), metadata, config, args.overwrite)
    summary_map, count_map, shape_map, processed_map = maps
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    progress = tqdm(remaining, desc=f"H3 {checkpoint_path.name}", dynamic_ncols=True)
    for progress_index, index in enumerate(progress, start=1):
        row = records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        for view_index, view_name in enumerate(views):
            view_image = make_view(image, view_name)
            grid_rows, grid_columns = choose_patch_grid(
                view_image.width,
                view_image.height,
                args.max_num_patches,
                max_side_patches,
            )
            model_input = image_to_unit_tensor(
                view_image,
                grid_rows,
                grid_columns,
                patch_size,
                device,
                resample=Image.Resampling.BILINEAR,
            )
            model_input = model_input.sub(0.5).div(0.5)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                tokens = model.forward_features(
                    model_input,
                    norm=True,
                    strip_cls_token=True,
                )
            expected_tokens = grid_rows * grid_columns
            if tokens.shape != (1, expected_tokens, feature_dim):
                raise RuntimeError(
                    f"Unexpected PE tokens {tuple(tokens.shape)} for grid "
                    f"{(grid_rows, grid_columns)}"
                )
            mask = torch.ones((1, expected_tokens), dtype=torch.bool, device=device)
            stats = masked_statistics(tokens, mask)
            summary_map[index, view_index] = stats[0].cpu().numpy().astype(np.float16)
            count_map[index, view_index] = expected_tokens
            shape_map[index, view_index] = (grid_rows, grid_columns)
            del model_input, tokens, mask, stats
        processed_map[index] = 1
        if progress_index % 10 == 0:
            for item in maps:
                item.flush()
    for item in maps:
        item.flush()
    config["complete"] = bool(np.asarray(processed_map).all())
    config["completed_cases"] = int(np.asarray(processed_map).sum())
    config["elapsed_seconds"] = float(time.monotonic() - started_at)
    if device.type == "cuda":
        config["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        config["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    write_json(Path(args.output_dir) / "representation_bank_config.json", config)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    views = validate_views(args.views)
    if args.backend == "existing_dense":
        if not args.source_feature_bank:
            raise ValueError("--source-feature-bank is required for existing_dense")
        extract_existing_dense(args, views)
    elif args.backend == "siglip2":
        extract_siglip2(args, views)
    elif args.backend == "siglip_fixed":
        extract_siglip_fixed(args, views)
    elif args.backend == "radio":
        extract_radio(args, views)
    else:
        extract_pe(args, views)
    print(
        json.dumps(
            {"status": "complete", "backend": args.backend, "output_dir": args.output_dir},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
