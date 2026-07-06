#!/usr/bin/env python3
"""Preflight checks for the full thymic gross image dataset.

This script validates the standardized case label table against the raw image
directory and writes lightweight QC summaries before any model training.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ALLOWED_WHO_TYPES = {
    "benign_hyperplasia",
    "thymoma_A",
    "thymoma_AB",
    "thymoma_B1",
    "thymoma_B2",
    "thymoma_B3",
    "micronodular_thymoma",
    "thymic_carcinoma",
    "other_or_uncertain",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def list_images(case_dir: Path) -> list[Path]:
    if not case_dir.exists() or not case_dir.is_dir():
        return []
    return sorted(p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight full thymic dataset labels and raw image folders.")
    parser.add_argument("--labels-csv", required=True, type=Path, help="Standardized case label CSV.")
    parser.add_argument("--raw-images-root", required=True, type=Path, help="Root directory containing {case_id}/image files, or the base used with --image-manifest-csv.")
    parser.add_argument("--image-manifest-csv", type=Path, help="Optional image manifest with case_id and image_path columns.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for QC summaries.")
    args = parser.parse_args()

    rows = read_rows(args.labels_csv)
    case_ids = [r.get("case_id", "").strip() for r in rows]
    duplicate_ids = {case_id for case_id, n in Counter(case_ids).items() if case_id and n > 1}

    manifest_by_case: dict[str, list[dict[str, str]]] = {}
    if args.image_manifest_csv:
        for image_row in read_rows(args.image_manifest_csv):
            manifest_by_case.setdefault(image_row.get("case_id", "").strip(), []).append(image_row)

    case_qc_rows: list[dict[str, object]] = []
    class_counter: Counter[str] = Counter()
    included_counter: Counter[str] = Counter()

    for row in rows:
        case_id = row.get("case_id", "").strip()
        who_type = row.get("who_type", "").strip()
        include_main = row.get("include_main_study", "").strip()
        need_review = row.get("need_review", "").strip()
        diagnosis_original = row.get("diagnosis_original", "").strip()
        expected_count_raw = row.get("image_count", "").strip()

        flags: list[str] = []
        if not case_id:
            flags.append("missing_case_id")
        if case_id in duplicate_ids:
            flags.append("duplicate_case_id")
        if not diagnosis_original:
            flags.append("missing_diagnosis_original")
        if who_type not in ALLOWED_WHO_TYPES:
            flags.append("invalid_who_type")
        if include_main not in {"0", "1"}:
            flags.append("invalid_include_main_study")
        if need_review not in {"0", "1"}:
            flags.append("invalid_need_review")

        case_dir = args.raw_images_root / case_id if case_id else args.raw_images_root / "__missing_case_id__"
        if args.image_manifest_csv:
            manifest_images = manifest_by_case.get(case_id, [])
            images = [Path(r.get("image_path", "")) for r in manifest_images]
            missing_manifest_files = [p for p in images if not p.exists()]
            if not manifest_images:
                flags.append("no_manifest_images_found")
            if missing_manifest_files:
                flags.append("manifest_image_file_missing")
        else:
            images = list_images(case_dir)
            if not case_dir.exists():
                flags.append("missing_case_image_dir")
            if not images:
                flags.append("no_images_found")

        try:
            expected_count = int(expected_count_raw)
        except ValueError:
            expected_count = None
            flags.append("invalid_image_count")
        if expected_count is not None and expected_count != len(images):
            flags.append("image_count_mismatch")

        if who_type:
            class_counter[who_type] += 1
            if include_main == "1":
                included_counter[who_type] += 1

        case_qc_rows.append(
            {
                "case_id": case_id,
                "who_type": who_type,
                "include_main_study": include_main,
                "need_review": need_review,
                "declared_image_count": expected_count_raw,
                "found_image_count": len(images),
                "case_dir": str(case_dir),
                "qc_flags": ";".join(flags),
            }
        )

    class_rows = []
    for who_type in sorted(set(class_counter) | set(included_counter)):
        class_rows.append(
            {
                "who_type": who_type,
                "total_cases": class_counter[who_type],
                "included_cases": included_counter[who_type],
            }
        )

    write_rows(
        args.output_dir / "preflight_case_qc_summary.csv",
        case_qc_rows,
        [
            "case_id",
            "who_type",
            "include_main_study",
            "need_review",
            "declared_image_count",
            "found_image_count",
            "case_dir",
            "qc_flags",
        ],
    )
    write_rows(args.output_dir / "preflight_class_distribution.csv", class_rows, ["who_type", "total_cases", "included_cases"])

    problem_count = sum(1 for r in case_qc_rows if r["qc_flags"])
    print(f"Checked cases: {len(case_qc_rows)}")
    print(f"Cases with QC flags: {problem_count}")
    print(f"Outputs saved to: {args.output_dir}")
    return 0 if problem_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
