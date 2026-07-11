from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a server-local read cache for the internal Task7 images.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-registry", required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=4.0)
    return parser.parse_args()


def cache_name(case_id: str, source: Path) -> str:
    safe_case = "".join(character if character.isalnum() or character in "-_" else "_" for character in case_id)
    digest = hashlib.sha1(str(source).encode("utf-8", errors="ignore")).hexdigest()[:10]
    suffix = source.suffix.lower() or ".img"
    return f"{safe_case}__{digest}{suffix}"


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_csv)
    cache_dir = Path(args.cache_dir).resolve()
    output_registry = Path(args.output_registry).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_registry.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(registry_path, dtype={"case_id": str}, encoding="utf-8-sig")
    if "training_image_path" not in frame.columns:
        raise ValueError(f"training_image_path is absent from {registry_path}")
    if frame["case_id"].duplicated().any():
        raise ValueError("The internal registry must contain one row per case.")

    sources = [Path(str(value)) for value in frame["training_image_path"]]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source images: {missing[:10]}")
    required_bytes = sum(path.stat().st_size for path in sources)
    free_bytes = shutil.disk_usage(cache_dir).free
    minimum_free = int(float(args.minimum_free_gib) * 1024**3)
    existing_bytes = sum(path.stat().st_size for path in cache_dir.iterdir() if path.is_file())
    additional_required = max(0, required_bytes - existing_bytes)
    if free_bytes - additional_required < minimum_free:
        raise RuntimeError(
            f"Local cache would leave too little disk space: free={free_bytes}, "
            f"additional_required={additional_required}, minimum_free={minimum_free}"
        )

    cached_paths: list[str] = []
    copied = 0
    for index, (case_id, source) in enumerate(zip(frame["case_id"].astype(str), sources), start=1):
        target = cache_dir / cache_name(case_id, source)
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            copied += 1
        cached_paths.append(str(target))
        if index % 50 == 0 or index == len(frame):
            print(f"[cache] {index}/{len(frame)} copied={copied}", flush=True)

    cached_frame = frame.copy()
    cached_frame["training_image_path"] = cached_paths
    cached_frame.to_csv(output_registry, index=False, encoding="utf-8-sig")
    manifest = {
        "complete": True,
        "source_registry": str(registry_path),
        "output_registry": str(output_registry),
        "cache_dir": str(cache_dir),
        "case_count": len(cached_frame),
        "required_bytes": required_bytes,
        "copied_files_this_run": copied,
    }
    output_registry.with_suffix(".cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
