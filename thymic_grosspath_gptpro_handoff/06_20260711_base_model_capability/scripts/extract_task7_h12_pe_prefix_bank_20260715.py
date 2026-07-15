from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
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

from extract_task7_dense_token_bank_20260711 import make_view  # noqa: E402
from extract_task7_h3_dense_bank_20260713 import (  # noqa: E402
    choose_patch_grid,
    image_to_unit_tensor,
)

VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
EXPECTED_SOURCES = {"batch1": 117, "batch2": 168, "third_batch": 306}
EXPECTED_SUBTYPES = {"A": 44, "AB": 262, "B1": 62, "B2": 89, "B3": 24, "TC": 110}
LOW_RISK = {"A", "AB", "B1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the locked PE prefix immediately before its final transformer block."
    )
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-code-dir", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-num-patches", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def canonical_source(value: object) -> str:
    text = str(value)
    return "third_batch" if text.startswith("third_batch") else text


def load_records(path: str | Path, max_cases: int | None) -> pd.DataFrame:
    records = pd.read_csv(
        path,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    records.columns = [str(column).lstrip("\ufeff") for column in records.columns]
    required = {
        "case_id",
        "original_case_id",
        "source_dataset",
        "task_l6_label",
        "image_name",
        "image_path",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Registry is missing columns: {missing}")
    records["source_dataset"] = records["source_dataset"].map(canonical_source)
    records["task_l6_label"] = records["task_l6_label"].astype(str)
    records["label_idx"] = (~records["task_l6_label"].isin(LOW_RISK)).astype(int)
    if records["case_id"].duplicated().any():
        raise ValueError("Registry contains duplicate case IDs")
    if len(records) != 591:
        raise ValueError(f"Expected 591 internal cases, found {len(records)}")
    if records["source_dataset"].value_counts().to_dict() != EXPECTED_SOURCES:
        raise ValueError("Unexpected source counts")
    if records["task_l6_label"].value_counts().to_dict() != EXPECTED_SUBTYPES:
        raise ValueError("Unexpected subtype counts")
    missing_images = [str(path) for path in records["image_path"] if not Path(str(path)).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} cached images")
    records = records.sort_values("case_id").reset_index(drop=True)
    records["feature_row"] = np.arange(len(records), dtype=int)
    if max_cases is not None:
        records = records.head(int(max_cases)).copy()
    return records


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "prefix": output_dir / "prefix_features.float16.npy",
        "shapes": output_dir / "spatial_shapes.int16.npy",
        "processed": output_dir / "processed.uint8.npy",
        "metadata": output_dir / "metadata.csv",
        "config": output_dir / "prefix_bank_config.json",
    }


def remove_existing(paths: dict[str, Path]) -> None:
    for path in paths.values():
        if path.exists():
            path.unlink()


def open_maps(
    paths: dict[str, Path],
    records: pd.DataFrame,
    config: dict[str, Any],
    overwrite: bool,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    if overwrite:
        remove_existing(paths)
    existing = [path.exists() for path in paths.values()]
    if any(existing):
        if not all(existing):
            raise RuntimeError("Partial H12 prefix bank found; use --overwrite or restore all files")
        previous = json.loads(paths["config"].read_text(encoding="utf-8"))
        keys = (
            "registry_sha256",
            "model_weight_sha256",
            "model_code_revision",
            "views",
            "prefix_shape",
            "prefix_layer_idx",
        )
        mismatch = {
            key: (previous.get(key), config.get(key))
            for key in keys
            if previous.get(key) != config.get(key)
        }
        if mismatch:
            raise ValueError(f"Existing H12 prefix bank differs: {mismatch}")
        return (
            np.lib.format.open_memmap(paths["prefix"], mode="r+"),
            np.lib.format.open_memmap(paths["shapes"], mode="r+"),
            np.lib.format.open_memmap(paths["processed"], mode="r+"),
        )

    prefix_shape = tuple(int(value) for value in config["prefix_shape"])
    required_bytes = int(np.prod(prefix_shape)) * np.dtype(np.float16).itemsize
    free_bytes = shutil.disk_usage(paths["prefix"].parent).free
    if free_bytes < int(required_bytes * 1.25):
        raise RuntimeError(
            f"H12 prefix bank needs {required_bytes / 2**30:.2f} GiB plus headroom; "
            f"only {free_bytes / 2**30:.2f} GiB is free"
        )
    prefix = np.lib.format.open_memmap(
        paths["prefix"], mode="w+", dtype=np.float16, shape=prefix_shape
    )
    shapes = np.lib.format.open_memmap(
        paths["shapes"],
        mode="w+",
        dtype=np.int16,
        shape=(len(records), len(VIEWS), 2),
    )
    processed = np.lib.format.open_memmap(
        paths["processed"], mode="w+", dtype=np.uint8, shape=(len(records),)
    )
    prefix[:] = 0
    shapes[:] = 0
    processed[:] = 0
    for item in (prefix, shapes, processed):
        item.flush()
    records.to_csv(paths["metadata"], index=False, encoding="utf-8-sig")
    write_json(paths["config"], config)
    return prefix, shapes, processed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.seed != 20260715:
        raise ValueError("H12 prefix extraction is locked to seed 20260715")
    if args.max_num_patches != 1024:
        raise ValueError("H12 is locked to 1024 patch tokens")
    if not args.code_revision:
        raise ValueError("--code-revision is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    code_dir = Path(args.model_code_dir).resolve(strict=True)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from core.vision_encoder.pe import VisionTransformer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    records = load_records(args.registry_csv, args.max_cases)
    model_weight_sha256 = sha256_file(args.model_id)
    registry_sha256 = sha256_file(args.registry_csv)
    prefix_token_count = args.max_num_patches + 1
    config: dict[str, Any] = {
        "experiment": "H12_PE_FINAL_BLOCK_LORA_ADAPTATION_20260715",
        "extractor_sha256": sha256_file(__file__),
        "registry_csv": str(Path(args.registry_csv).resolve()),
        "registry_sha256": registry_sha256,
        "model_id": str(Path(args.model_id).resolve()),
        "model_weight_sha256": model_weight_sha256,
        "model_code_dir": str(code_dir),
        "model_code_revision": args.code_revision,
        "views": list(VIEWS),
        "max_num_patch_tokens": args.max_num_patches,
        "prefix_includes_cls": True,
        "prefix_layer_idx": -2,
        "completed_block_indices": list(range(23)),
        "remaining_block_indices": [23],
        "feature_dim": 1024,
        "prefix_shape": [len(records), len(VIEWS), prefix_token_count, 1024],
        "prefix_dtype": "float16",
        "seed": args.seed,
        "complete": False,
    }
    prefix_map, shape_map, processed_map = open_maps(
        paths, records, config, args.overwrite
    )

    model = VisionTransformer.from_config(
        "PE-Spatial-L14-448",
        pretrained=True,
        checkpoint_path=str(Path(args.model_id).resolve()),
    ).eval().to(device)
    if not model.use_cls_token or model.layers != 24 or model.width != 1024:
        raise ValueError(
            f"Unexpected PE architecture: cls={model.use_cls_token} layers={model.layers} width={model.width}"
        )
    max_side_patches = max(32, int(round(args.max_num_patches**0.5)) * 4)
    remaining = np.flatnonzero(np.asarray(processed_map) == 0).astype(int)
    started = time.monotonic()
    progress = tqdm(remaining, desc="H12 PE prefix", dynamic_ncols=True)
    for completed, index in enumerate(progress, start=1):
        row = records.iloc[int(index)]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        prefix_map[index] = 0
        shape_map[index] = 0
        for view_index, view_name in enumerate(VIEWS):
            view = make_view(image, view_name)
            rows, columns = choose_patch_grid(
                view.width,
                view.height,
                args.max_num_patches,
                max_side_patches,
            )
            model_input = image_to_unit_tensor(
                view,
                rows,
                columns,
                model.patch_size,
                device,
                resample=Image.Resampling.BILINEAR,
            ).sub(0.5).div(0.5)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                tokens = model.forward_features(
                    model_input,
                    norm=False,
                    layer_idx=-2,
                    strip_cls_token=False,
                )[0].float()
            expected_tokens = rows * columns + 1
            if tokens.shape != (expected_tokens, model.width):
                raise ValueError(
                    f"Unexpected prefix shape for {row['case_id']} {view_name}: "
                    f"{tuple(tokens.shape)} != {(expected_tokens, model.width)}"
                )
            if expected_tokens > prefix_token_count:
                raise ValueError("PE prefix exceeds locked token capacity")
            prefix_map[index, view_index, :expected_tokens] = tokens.cpu().numpy()
            shape_map[index, view_index] = (rows, columns)
        processed_map[index] = 1
        if completed % 5 == 0 or completed == len(remaining):
            prefix_map.flush()
            shape_map.flush()
            processed_map.flush()
        progress.set_postfix(done=int(np.asarray(processed_map).sum()))

    if not np.all(np.asarray(processed_map) == 1):
        raise RuntimeError("H12 prefix extraction ended incomplete")
    config.update(
        {
            "complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "prefix_file_bytes": paths["prefix"].stat().st_size,
            "prefix_sha256": sha256_file(paths["prefix"]),
            "spatial_shapes_sha256": sha256_file(paths["shapes"]),
            "metadata_sha256": sha256_file(paths["metadata"]),
        }
    )
    write_json(paths["config"], config)
    (output_dir / "RUN.status").write_text("COMPLETE\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
