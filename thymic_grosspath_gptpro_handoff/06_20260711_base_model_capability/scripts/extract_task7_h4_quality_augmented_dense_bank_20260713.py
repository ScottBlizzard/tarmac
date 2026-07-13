from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from tqdm.auto import tqdm

from extract_task7_h3_dense_bank_20260713 import PeAdapter, open_maps
from extract_task7_h3_representation_bank_20260713 import (
    DEFAULT_VIEWS,
    load_records,
    make_view,
    metadata_from_records,
    set_seed,
    validate_views,
    write_json,
)


PROFILE_NAME = "quality_domain_randomization_v1"
PROFILE = {
    "downsample_scale_uniform": [0.60, 0.90],
    "downsample_resample": "PIL.Image.Resampling.BILINEAR",
    "gaussian_blur_probability": 0.75,
    "gaussian_blur_radius_uniform": [0.20, 1.00],
    "jpeg_quality_integer_uniform_inclusive": [55, 90],
    "brightness_factor_uniform": [0.85, 1.15],
    "contrast_factor_uniform": [0.85, 1.15],
    "saturation_factor_uniform": [0.85, 1.15],
    "white_balance_channel_gain_uniform": [0.92, 1.08],
    "white_balance_gain_normalization": "divide_by_mean_gain",
    "geometry_preserved": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the preregistered H4 quality-randomized PE dense bank."
    )
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--canonical-model-id", required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--model-code-dir", required=True)
    parser.add_argument("--code-revision", required=True)
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


def deterministic_rng(seed: int, case_id: str, view_name: str) -> np.random.Generator:
    payload = f"{PROFILE_NAME}|{seed}|{case_id}|{view_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))


def quality_randomize(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size

    scale = float(rng.uniform(0.60, 0.90))
    reduced = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.BILINEAR,
    )
    image = reduced.resize((width, height), Image.Resampling.BILINEAR)

    if float(rng.random()) < 0.75:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.20, 1.00))))

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=int(rng.integers(55, 91)),
        optimize=False,
        progressive=False,
        subsampling=2,
    )
    buffer.seek(0)
    with Image.open(buffer) as jpeg_image:
        image = jpeg_image.convert("RGB").copy()

    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
    image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.85, 1.15)))

    gains = rng.uniform(0.92, 1.08, size=3).astype(np.float32)
    gains /= float(gains.mean())
    array = np.asarray(image, dtype=np.float32) * gains.reshape(1, 1, 3)
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    args = parse_args()
    if args.seed != 20260713:
        raise ValueError("The H4 augmented bank is locked to seed 20260713")
    set_seed(args.seed)
    views = validate_views(args.views)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    records = load_records(args.registry_csv, domains, args.max_cases)
    metadata = metadata_from_records(records)
    started_at = time.monotonic()
    adapter = PeAdapter(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config: dict[str, Any] = {
        "backend": "pe_h4_quality_augmented",
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
        "augmentation_profile_name": PROFILE_NAME,
        "augmentation_profile": PROFILE,
        "augmentation_seed_rule": "sha256(profile|seed|case_id|view_name) first 64 bits",
        "complete": False,
    }
    maps = open_maps(Path(args.output_dir), metadata, config, args.overwrite)
    dense_map, mask_map, shape_map, processed_map = maps
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    progress = tqdm(remaining, desc=f"H4 {args.canonical_model_id}", dynamic_ncols=True)
    for progress_index, index in enumerate(progress, start=1):
        row = records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        dense_map[index] = 0
        mask_map[index] = 0
        for view_index, view_name in enumerate(views):
            clean_view = make_view(image, view_name)
            augmented_view = quality_randomize(
                clean_view,
                deterministic_rng(int(args.seed), str(row["case_id"]), view_name),
            )
            if augmented_view.size != clean_view.size:
                raise RuntimeError("H4 augmentation changed view geometry")
            extracted = adapter.extract(augmented_view)
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
