from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resized server-local Task7 image cache for training I/O.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-registry", required=True)
    parser.add_argument("--max-long-side", type=int, default=2048)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--minimum-free-gib", type=float, default=4.0)
    return parser.parse_args()


def target_name(case_id: str, source: Path, max_long_side: int) -> str:
    safe_case = "".join(character if character.isalnum() or character in "-_" else "_" for character in case_id)
    digest = hashlib.sha1(str(source).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{safe_case}__{digest}__max{max_long_side}.jpg"


def valid_cached_image(path: Path, max_long_side: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return max(image.size) <= max_long_side
    except OSError:
        return False


def main() -> None:
    args = parse_args()
    source_registry = Path(args.registry_csv)
    cache_dir = Path(args.cache_dir).resolve()
    output_registry = Path(args.output_registry).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_registry.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source_registry, dtype={"case_id": str}, encoding="utf-8-sig")
    if frame["case_id"].duplicated().any():
        raise ValueError("The internal registry must contain one row per case.")
    sources = [Path(str(value)) for value in frame["training_image_path"]]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source images: {missing[:10]}")

    free_bytes = shutil.disk_usage(cache_dir).free
    conservative_required = min(
        sum(path.stat().st_size for path in sources),
        len(sources) * 2 * 1024**2,
    )
    minimum_free = int(float(args.minimum_free_gib) * 1024**3)
    if free_bytes - conservative_required < minimum_free:
        raise RuntimeError(
            f"Resized cache may leave too little disk space: free={free_bytes}, "
            f"estimate={conservative_required}, minimum_free={minimum_free}"
        )

    cached_paths: list[str] = []
    copied = 0
    for index, (case_id, source) in enumerate(zip(frame["case_id"].astype(str), sources), start=1):
        target = cache_dir / target_name(case_id, source, args.max_long_side)
        if not valid_cached_image(target, args.max_long_side):
            temporary = target.with_suffix(f".{os.getpid()}.tmp.jpg")
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail(
                    (args.max_long_side, args.max_long_side),
                    resample=Image.Resampling.LANCZOS,
                )
                image.save(
                    temporary,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    subsampling=0,
                    optimize=False,
                )
            os.replace(temporary, target)
            copied += 1
        cached_paths.append(str(target))
        if index % 50 == 0 or index == len(frame):
            print(f"[resize-cache] {index}/{len(frame)} created={copied}", flush=True)

    cached_frame = frame.copy()
    cached_frame["training_image_path"] = cached_paths
    cached_frame.to_csv(output_registry, index=False, encoding="utf-8-sig")
    total_bytes = sum(Path(path).stat().st_size for path in cached_paths)
    manifest = {
        "complete": True,
        "source_registry": str(source_registry),
        "output_registry": str(output_registry),
        "cache_dir": str(cache_dir),
        "case_count": len(cached_frame),
        "max_long_side": args.max_long_side,
        "jpeg_quality": args.jpeg_quality,
        "total_bytes": total_bytes,
        "created_files_this_run": copied,
    }
    output_registry.with_suffix(".cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
