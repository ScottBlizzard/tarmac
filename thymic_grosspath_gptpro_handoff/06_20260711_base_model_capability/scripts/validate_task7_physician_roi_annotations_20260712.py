from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import build_task7_physician_roi_oracle_manifest_20260712 as builder


BOX_PREFIXES = (
    "specimen_extent",
    "cut_surface",
    "probable_tumor",
    "capsule_interface",
    "hemorrhage_necrosis_artifact",
    "roi1",
    "roi2",
    "roi3",
)
ALLOWED = {
    "annotation_status": {"complete"},
    "image_quality": {"adequate", "limited", "nondiagnostic"},
    "image_sufficient_for_low_high_judgment": {"yes", "no", "uncertain"},
    "physician_risk_judgment": {"low", "high", "indeterminate"},
    "no_visually_diagnostic_roi": {"yes", "no"},
    "recommended_additional_view": {
        "none",
        "whole",
        "cut_surface_closeup",
        "capsule_interface_closeup",
        "multiple_views",
        "other",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and cryptographically lock two blinded physician ROI forms."
    )
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--reader-1-csv", default=None)
    parser.add_argument("--reader-2-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def coordinate_values(row: pd.Series, prefix: str) -> list[float | None]:
    values: list[float | None] = []
    for suffix in ("x1_norm", "y1_norm", "x2_norm", "y2_norm"):
        value = row[f"{prefix}_{suffix}"]
        if pd.isna(value) or clean_text(value) == "":
            values.append(None)
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            values.append(numeric)
    return values


def validate_box(
    row: pd.Series, prefix: str, required: bool, errors: list[dict[str, Any]]
) -> bool:
    values = coordinate_values(row, prefix)
    present = [value is not None for value in values]
    location = {"oracle_id": row["oracle_id"], "reader_id": row["reader_id"]}
    if not any(present):
        if required:
            errors.append({**location, "field": prefix, "error": "required box is blank"})
        return False
    if not all(present):
        errors.append({**location, "field": prefix, "error": "partial coordinate box"})
        return False
    x1, y1, x2, y2 = [float(value) for value in values]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        errors.append({**location, "field": prefix, "error": "coordinates must be finite in [0,1]"})
        return False
    if not (x1 < x2 and y1 < y2):
        errors.append({**location, "field": prefix, "error": "box must satisfy x1<x2 and y1<y2"})
        return False
    return True


def validate_reader(
    path: Path,
    reader_id: str,
    expected_ids: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = pd.read_csv(path, dtype={"oracle_id": str, "reader_id": str}, encoding="utf-8-sig")
    expected_columns = builder.annotation_columns()
    errors: list[dict[str, Any]] = []
    missing = [column for column in expected_columns if column not in frame]
    prohibited = [
        column
        for column in frame
        if column.lower()
        in {
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "c1_prob_high",
            "c1_pred",
            "c1_correct",
            "case_id",
            "source_dataset",
        }
    ]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if prohibited:
        raise ValueError(f"{path} contains prohibited blinded columns: {prohibited}")
    frame = frame[expected_columns].copy()
    if len(frame) != len(expected_ids) or frame["oracle_id"].duplicated().any():
        raise ValueError(f"{path} must contain one row per blinded image")
    if set(frame["oracle_id"]) != expected_ids:
        raise ValueError(f"{path} oracle IDs do not match the blinded image manifest")
    if set(frame["reader_id"].astype(str)) != {reader_id}:
        raise ValueError(f"{path} must contain reader_id={reader_id} only")

    for _, row in frame.iterrows():
        location = {"oracle_id": row["oracle_id"], "reader_id": reader_id}
        for column, allowed in ALLOWED.items():
            value = clean_text(row[column]).lower()
            if value not in allowed:
                errors.append(
                    {
                        **location,
                        "field": column,
                        "error": f"value must be one of {sorted(allowed)}",
                    }
                )
        confidence = pd.to_numeric(
            pd.Series([row["physician_confidence_1_to_5"]]), errors="coerce"
        ).iloc[0]
        if pd.isna(confidence) or float(confidence) not in {1, 2, 3, 4, 5}:
            errors.append(
                {
                    **location,
                    "field": "physician_confidence_1_to_5",
                    "error": "confidence must be an integer from 1 to 5",
                }
            )
        validate_box(row, "specimen_extent", True, errors)
        for prefix in BOX_PREFIXES[1:5]:
            validate_box(row, prefix, False, errors)
        no_roi = clean_text(row["no_visually_diagnostic_roi"]).lower()
        roi_present = [validate_box(row, prefix, False, errors) for prefix in BOX_PREFIXES[5:]]
        if no_roi == "yes" and any(roi_present):
            errors.append(
                {**location, "field": "roi1-roi3", "error": "ROI boxes must be blank when no_visually_diagnostic_roi=yes"}
            )
        if no_roi == "no" and not roi_present[0]:
            errors.append(
                {**location, "field": "roi1", "error": "ROI1 is required when no_visually_diagnostic_roi=no"}
            )
        for roi_index, present in enumerate(roi_present, start=1):
            reason = clean_text(row[f"roi{roi_index}_reason"])
            if present and not reason:
                errors.append(
                    {**location, "field": f"roi{roi_index}_reason", "error": "reason is required for a populated ROI"}
                )
            if not present and reason:
                errors.append(
                    {**location, "field": f"roi{roi_index}_reason", "error": "reason must be blank when ROI coordinates are blank"}
                )
    return frame, errors


def main() -> None:
    args = parse_args()
    packet_root = Path(args.packet_root)
    blinded_dir = packet_root / "blinded_packet"
    manifest_path = blinded_dir / "BLINDED_IMAGE_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path, dtype={"oracle_id": str}, encoding="utf-8-sig")
    if len(manifest) != 120 or manifest["oracle_id"].duplicated().any():
        raise ValueError("Blinded image manifest must contain 120 unique oracle IDs")
    expected_ids = set(manifest["oracle_id"])
    reader_paths = {
        "reader_1": Path(args.reader_1_csv)
        if args.reader_1_csv
        else blinded_dir / "READER_1_ANNOTATION.csv",
        "reader_2": Path(args.reader_2_csv)
        if args.reader_2_csv
        else blinded_dir / "READER_2_ANNOTATION.csv",
    }
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else packet_root / "secure_do_not_share" / "locked_annotations"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite annotation lock: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    all_errors = []
    for reader_id, path in reader_paths.items():
        frame, errors = validate_reader(path, reader_id, expected_ids)
        frames.append(frame)
        all_errors.extend(errors)
    if all_errors:
        error_path = output_dir / "ANNOTATION_VALIDATION_ERRORS.csv"
        pd.DataFrame(all_errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        raise ValueError(f"Annotation validation failed with {len(all_errors)} errors: {error_path}")

    locked = pd.concat(frames, ignore_index=True).sort_values(
        ["oracle_id", "reader_id"]
    )
    locked_path = output_dir / "LOCKED_BLINDED_ANNOTATIONS.csv"
    locked.to_csv(locked_path, index=False, encoding="utf-8-sig")
    for reader_id, path in reader_paths.items():
        shutil.copy2(path, output_dir / f"ORIGINAL_{reader_id.upper()}.csv")
    lock = {
        "state": "locked",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "cases": 120,
        "readers": 2,
        "rows": int(len(locked)),
        "reader_file_sha256": {
            reader_id: sha256_file(path) for reader_id, path in reader_paths.items()
        },
        "locked_annotation_sha256": sha256_file(locked_path),
        "blinding_note": "No secure label/model key was loaded during validation or lock creation.",
    }
    (output_dir / "ANNOTATION_LOCK.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(json.dumps(lock, indent=2), flush=True)


if __name__ == "__main__":
    main()
