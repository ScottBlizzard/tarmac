from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_SOURCES = ("batch1", "batch2", "third_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate source-case-folder image availability without emitting identifiers."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def canonical_source(value: object) -> str:
    text = str(value)
    return "third_batch" if text.startswith("third_batch") else text


def count_images(folder_text: object, recursive: bool) -> tuple[int, bool]:
    if pd.isna(folder_text) or not str(folder_text).strip():
        return 0, False
    folder = Path(str(folder_text))
    if not folder.is_dir():
        return 0, False
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    count = sum(
        1
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return count, True


def distribution(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(dtype=int)
    return {
        "n": int(len(array)),
        "zero": int((array == 0).sum()),
        "one": int((array == 1).sum()),
        "two": int((array == 2).sum()),
        "three_or_more": int((array >= 3).sum()),
        "more_than_one": int((array > 1).sum()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": int(array.max(initial=0)),
    }


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry, dtype={"case_id": str}, encoding="utf-8-sig")
    registry.columns = [str(column).lstrip("\ufeff") for column in registry.columns]
    required = {"case_id", "source_dataset", "source_case_folder", "image_path"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"Missing registry columns: {sorted(missing)}")
    if len(registry) != 591 or registry["case_id"].nunique() != 591:
        raise ValueError("Expected 591 unique internal cases")

    registry["source_dataset"] = registry["source_dataset"].map(canonical_source)
    if set(registry["source_dataset"].unique()) != set(EXPECTED_SOURCES):
        raise ValueError("Unexpected source set")

    direct_counts: list[int] = []
    recursive_counts: list[int] = []
    folder_exists: list[bool] = []
    folder_keys: list[str] = []
    selected_exists: list[bool] = []
    for row in registry.itertuples(index=False):
        folder_value = getattr(row, "source_case_folder")
        direct_count, exists = count_images(folder_value, recursive=False)
        recursive_count, recursive_exists = count_images(folder_value, recursive=True)
        if exists != recursive_exists:
            raise RuntimeError("Direct and recursive folder existence checks disagree")
        direct_counts.append(direct_count)
        recursive_counts.append(recursive_count)
        folder_exists.append(exists)
        folder_keys.append(str(Path(str(folder_value)).resolve()) if exists else "")
        selected_exists.append(Path(str(getattr(row, "image_path"))).is_file())

    audit = registry[["source_dataset"]].copy()
    audit["direct_image_count"] = direct_counts
    audit["recursive_image_count"] = recursive_counts
    audit["folder_exists"] = folder_exists
    audit["folder_key"] = folder_keys
    audit["selected_image_exists"] = selected_exists

    folder_reuse = Counter(key for key in folder_keys if key)
    audit["folder_shared_by_multiple_cases"] = [
        bool(key and folder_reuse[key] > 1) for key in folder_keys
    ]

    result: dict[str, Any] = {
        "registry_rows": int(len(registry)),
        "unique_cases": int(registry["case_id"].nunique()),
        "privacy": "aggregate_only_no_case_ids_or_paths",
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "overall": {
            "folder_exists": int(audit["folder_exists"].sum()),
            "selected_image_exists": int(audit["selected_image_exists"].sum()),
            "shared_folder_rows": int(audit["folder_shared_by_multiple_cases"].sum()),
            "unique_existing_folders": int(
                audit.loc[audit["folder_exists"], "folder_key"].nunique()
            ),
            "direct_counts": distribution(audit["direct_image_count"]),
            "recursive_counts": distribution(audit["recursive_image_count"]),
        },
        "by_source": {},
    }
    for source in EXPECTED_SOURCES:
        group = audit[audit["source_dataset"] == source]
        result["by_source"][source] = {
            "n": int(len(group)),
            "folder_exists": int(group["folder_exists"].sum()),
            "selected_image_exists": int(group["selected_image_exists"].sum()),
            "shared_folder_rows": int(group["folder_shared_by_multiple_cases"].sum()),
            "direct_counts": distribution(group["direct_image_count"]),
            "recursive_counts": distribution(group["recursive_image_count"]),
        }

    if "original_image_count" in registry.columns:
        original = pd.to_numeric(registry["original_image_count"], errors="coerce")
        result["registry_original_image_count"] = {
            "nonmissing": int(original.notna().sum()),
            "missing": int(original.isna().sum()),
            "value_counts": {
                str(float(value)): int(count)
                for value, count in original.dropna().value_counts().sort_index().items()
            },
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
