#!/usr/bin/env python3
"""Build standardized full-data label and image manifests.

The first supported source is the current pilot registry, which keeps images in
class folders. The output follows the new full-data schema without copying or
renaming original images.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


WHO_TO_DIAGNOSIS = {
    "benign_hyperplasia": "benign_hyperplasia",
    "thymoma_A": "thymoma_A",
    "thymoma_AB": "thymoma_AB",
    "thymoma_B1": "thymoma_B1",
    "thymoma_B2": "thymoma_B2",
    "thymoma_B3": "thymoma_B3",
    "micronodular_thymoma": "micronodular_thymoma",
    "thymic_carcinoma": "thymic_carcinoma",
}

WHO_TO_MAJOR = {
    "benign_hyperplasia": "benign_hyperplasia",
    "thymoma_A": "thymoma",
    "thymoma_AB": "thymoma",
    "thymoma_B1": "thymoma",
    "thymoma_B2": "thymoma",
    "thymoma_B3": "thymoma",
    "micronodular_thymoma": "thymoma",
    "thymic_carcinoma": "thymic_carcinoma",
}

WHO_TO_FINE = {
    "thymoma_A": "A",
    "thymoma_AB": "AB",
    "thymoma_B1": "B1",
    "thymoma_B2": "B2",
    "thymoma_B3": "B3",
}

WHO_TO_DIAGNOSIS_CN = {
    "benign_hyperplasia": "良性胸腺增生",
    "thymoma_A": "A型胸腺瘤",
    "thymoma_AB": "AB型胸腺瘤",
    "thymoma_B1": "B1型胸腺瘤",
    "thymoma_B2": "B2型胸腺瘤",
    "thymoma_B3": "B3型胸腺瘤",
    "micronodular_thymoma": "微结节性胸腺瘤",
    "thymic_carcinoma": "胸腺癌",
}


LABEL_FIELDNAMES = [
    "case_id",
    "patient_id",
    "source_center",
    "source_dataset",
    "source_case_folder",
    "image_count",
    "original_filenames",
    "diagnosis_original",
    "diagnosis_standardized",
    "who_type",
    "is_thymic_hyperplasia",
    "is_tet",
    "is_thymoma",
    "is_thymic_carcinoma",
    "major_category",
    "coarse_label_candidate",
    "fine_label_candidate",
    "hierarchical_level1",
    "hierarchical_level2",
    "hierarchical_level3",
    "hierarchical_level4",
    "label_confidence",
    "label_source",
    "need_review",
    "reviewed_by",
    "review_date",
    "include_main_study",
    "exclude_reason",
    "split_stratification_class",
    "review_note",
]

IMAGE_FIELDNAMES = [
    "case_id",
    "patient_id",
    "who_type",
    "source_dataset",
    "source_case_folder",
    "image_name",
    "image_path",
    "image_index",
    "file_exists",
    "file_size_bytes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_filenames(value: str) -> list[str]:
    return [name.strip() for name in str(value).split(";") if name.strip()]


def resolve_image_path(images_root: Path, source_folder: str, image_name: str) -> Path:
    direct = images_root / source_folder / image_name
    if direct.exists():
        return direct
    matches = sorted((images_root / source_folder).rglob(image_name))
    if matches:
        return matches[0]
    return direct


def build_from_current_160(registry_csv: Path, images_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    registry = read_csv(registry_csv)
    labels: list[dict[str, object]] = []
    images: list[dict[str, object]] = []

    for row in registry:
        case_id = row["case_id"].strip()
        who = row["who_type_raw"].strip()
        filenames = split_filenames(row["image_filenames"])
        source_folder = row["source_case_folder"].strip()
        fine_label = WHO_TO_FINE.get(who, "na")
        major = WHO_TO_MAJOR.get(who, "other_or_uncertain")
        diagnosis = WHO_TO_DIAGNOSIS.get(who, "other_or_uncertain")
        diagnosis_original = WHO_TO_DIAGNOSIS_CN.get(who, diagnosis)

        labels.append(
            {
                "case_id": case_id,
                "patient_id": case_id,
                "source_center": "unknown",
                "source_dataset": row.get("source_dataset", "pilot_160").strip() or "pilot_160",
                "source_case_folder": source_folder,
                "image_count": len(filenames),
                "original_filenames": ";".join(filenames),
                "diagnosis_original": diagnosis_original,
                "diagnosis_standardized": diagnosis,
                "who_type": who,
                "is_thymic_hyperplasia": row.get("is_thymic_hyperplasia", ""),
                "is_tet": row.get("is_tet", ""),
                "is_thymoma": row.get("is_thymoma", ""),
                "is_thymic_carcinoma": row.get("is_thymic_carcinoma", ""),
                "major_category": major,
                "coarse_label_candidate": row.get("low_high_risk_group", "na"),
                "fine_label_candidate": fine_label,
                "hierarchical_level1": row.get("task_l1_label", "na"),
                "hierarchical_level2": row.get("task_l2_label", "na"),
                "hierarchical_level3": row.get("task_l3_label", "na"),
                "hierarchical_level4": row.get("task_l4_label", "na"),
                "label_confidence": "high",
                "label_source": "folder_name_registry",
                "need_review": 0,
                "reviewed_by": "",
                "review_date": "",
                "include_main_study": 1,
                "exclude_reason": "",
                "split_stratification_class": row.get("split_stratification_class", who),
                "review_note": "Converted from pilot_160 registry; diagnosis derived from source folder.",
            }
        )

        for idx, image_name in enumerate(filenames, start=1):
            image_path = resolve_image_path(images_root, source_folder, image_name)
            images.append(
                {
                    "case_id": case_id,
                    "patient_id": case_id,
                    "who_type": who,
                    "source_dataset": row.get("source_dataset", "pilot_160").strip() or "pilot_160",
                    "source_case_folder": source_folder,
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "image_index": idx,
                    "file_exists": int(image_path.exists()),
                    "file_size_bytes": image_path.stat().st_size if image_path.exists() else "",
                }
            )

    return labels, images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-data compatible labels and image manifests.")
    parser.add_argument("--source", choices=["current_160"], required=True)
    parser.add_argument("--registry-csv", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--labels-out", required=True, type=Path)
    parser.add_argument("--case-registry-out", required=True, type=Path)
    parser.add_argument("--image-manifest-out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels, images = build_from_current_160(args.registry_csv, args.images_root)
    write_csv(args.labels_out, labels, LABEL_FIELDNAMES)
    write_csv(args.case_registry_out, labels, LABEL_FIELDNAMES)
    write_csv(args.image_manifest_out, images, IMAGE_FIELDNAMES)
    missing_images = sum(1 for row in images if int(row["file_exists"]) == 0)
    print(f"Saved labels: {args.labels_out} ({len(labels)} cases)")
    print(f"Saved case registry: {args.case_registry_out}")
    print(f"Saved image manifest: {args.image_manifest_out} ({len(images)} images)")
    print(f"Missing images: {missing_images}")
    return 0 if missing_images == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
