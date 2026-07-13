from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_SOURCES = ("batch1", "batch2", "third_batch")
INTERNAL_DOMAINS = {"old_data", "third_batch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit source-root and case-bag image availability without emitting identifiers."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--case-bag-registry", required=True)
    parser.add_argument("--batch1-root", required=True)
    parser.add_argument("--batch2-root", required=True)
    parser.add_argument("--third-batch-root", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def canonical_source(value: object) -> str:
    text = str(value)
    return "third_batch" if text.startswith("third_batch") else text


def source_folder_value(row: object, source: str) -> object:
    return getattr(row, "source_folder") if source == "third_batch" else getattr(row, "source_case_folder")


def resolve_source_folder(root: Path, value: object) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    folder = Path(str(value).strip())
    return folder if folder.is_absolute() else root / folder


def count_root_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def value_counts(values: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        str(float(value)): int(count)
        for value, count in numeric.dropna().value_counts().sort_index().items()
    }


def summarize_case_bag(case_bag: pd.DataFrame) -> dict[str, Any]:
    required = {"domain", "case_id", "image_path"}
    missing = required - set(case_bag.columns)
    if missing:
        raise ValueError(f"Missing case-bag columns: {sorted(missing)}")

    internal = case_bag[case_bag["domain"].isin(INTERNAL_DOMAINS)].copy()
    if internal["case_id"].nunique() != 591:
        raise ValueError("Expected 591 unique internal cases in the case-bag registry")
    internal["image_exists_now"] = internal["image_path"].map(lambda value: Path(str(value)).is_file())
    case_sizes = internal.groupby(["domain", "case_id"], sort=False).size()

    result: dict[str, Any] = {
        "cases": int(internal["case_id"].nunique()),
        "image_rows": int(len(internal)),
        "existing_image_rows": int(internal["image_exists_now"].sum()),
        "missing_image_rows": int((~internal["image_exists_now"]).sum()),
        "multi_image_cases": int((case_sizes > 1).sum()),
        "maximum_images_per_case": int(case_sizes.max()),
        "by_domain": {},
    }
    for domain in sorted(INTERNAL_DOMAINS):
        group = internal[internal["domain"] == domain]
        domain_sizes = group.groupby("case_id", sort=False).size()
        result["by_domain"][domain] = {
            "cases": int(group["case_id"].nunique()),
            "image_rows": int(len(group)),
            "existing_image_rows": int(group["image_exists_now"].sum()),
            "multi_image_cases": int((domain_sizes > 1).sum()),
            "maximum_images_per_case": int(domain_sizes.max()),
        }
    return result


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry, dtype={"case_id": str}, encoding="utf-8-sig")
    registry.columns = [str(column).lstrip("\ufeff") for column in registry.columns]
    required = {
        "case_id",
        "source_dataset",
        "source_case_folder",
        "source_folder",
        "image_name",
        "image_path",
    }
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"Missing registry columns: {sorted(missing)}")
    if len(registry) != 591 or registry["case_id"].nunique() != 591:
        raise ValueError("Expected 591 unique internal cases")

    registry["source_dataset"] = registry["source_dataset"].map(canonical_source)
    if set(registry["source_dataset"].unique()) != set(EXPECTED_SOURCES):
        raise ValueError("Unexpected source set")

    roots = {
        "batch1": Path(args.batch1_root).resolve(),
        "batch2": Path(args.batch2_root).resolve(),
        "third_batch": Path(args.third_batch_root).resolve(),
    }
    missing_roots = [source for source, root in roots.items() if not root.is_dir()]
    if missing_roots:
        raise FileNotFoundError(f"Missing dataset roots: {missing_roots}")

    rows: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        source = str(getattr(row, "source_dataset"))
        folder = resolve_source_folder(roots[source], source_folder_value(row, source))
        folder_exists = bool(folder is not None and folder.is_dir())
        name = str(getattr(row, "image_name")).strip()
        direct_exists = bool(folder_exists and (folder / name).is_file())
        recursive_fallback_unique = False
        recursive_ambiguous = False
        if folder_exists and name and not direct_exists:
            matches = [path for path in folder.rglob(name) if path.is_file()]
            recursive_fallback_unique = len(matches) == 1
            recursive_ambiguous = len(matches) > 1
        rows.append(
            {
                "source": source,
                "folder_exists": folder_exists,
                "folder_key": str(folder.resolve()) if folder_exists else "",
                "selected_cache_exists": Path(str(getattr(row, "image_path"))).is_file(),
                "selected_original_direct_exists": direct_exists,
                "selected_original_recursive_fallback_unique": recursive_fallback_unique,
                "selected_original_recursive_ambiguous": recursive_ambiguous,
                "selected_original_resolved": direct_exists or recursive_fallback_unique,
            }
        )

    audit = pd.DataFrame(rows)
    folder_reuse = Counter(key for key in audit["folder_key"] if key)
    audit["folder_shared_by_multiple_cases"] = audit["folder_key"].map(
        lambda key: bool(key and folder_reuse[key] > 1)
    )

    case_bag = pd.read_csv(args.case_bag_registry, dtype={"case_id": str}, encoding="utf-8-sig")
    result: dict[str, Any] = {
        "registry_rows": int(len(registry)),
        "unique_cases": int(registry["case_id"].nunique()),
        "privacy": "aggregate_only_no_case_ids_or_paths",
        "path_interpretation": "source folder fields are relative to dataset roots",
        "overall": {
            "source_folder_exists_rows": int(audit["folder_exists"].sum()),
            "unique_existing_source_folders": int(
                audit.loc[audit["folder_exists"], "folder_key"].nunique()
            ),
            "shared_source_folder_rows": int(audit["folder_shared_by_multiple_cases"].sum()),
            "selected_cache_exists": int(audit["selected_cache_exists"].sum()),
            "selected_original_resolved": int(audit["selected_original_resolved"].sum()),
            "selected_original_direct_exists": int(
                audit["selected_original_direct_exists"].sum()
            ),
            "selected_original_recursive_fallback_unique": int(
                audit["selected_original_recursive_fallback_unique"].sum()
            ),
            "selected_original_recursive_ambiguous": int(
                audit["selected_original_recursive_ambiguous"].sum()
            ),
        },
        "case_bag": summarize_case_bag(case_bag),
        "by_source": {},
    }
    for source in EXPECTED_SOURCES:
        group = audit[audit["source"] == source]
        source_registry = registry[registry["source_dataset"] == source]
        result["by_source"][source] = {
            "cases": int(len(group)),
            "unique_source_folders": int(group.loc[group["folder_exists"], "folder_key"].nunique()),
            "source_folder_exists_rows": int(group["folder_exists"].sum()),
            "selected_cache_exists": int(group["selected_cache_exists"].sum()),
            "selected_original_resolved": int(group["selected_original_resolved"].sum()),
            "selected_original_direct_exists": int(
                group["selected_original_direct_exists"].sum()
            ),
            "selected_original_recursive_fallback_unique": int(
                group["selected_original_recursive_fallback_unique"].sum()
            ),
            "raw_image_files_in_dataset_root": count_root_images(roots[source]),
            "registry_original_image_count": value_counts(source_registry["original_image_count"])
            if "original_image_count" in source_registry.columns
            else {},
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
