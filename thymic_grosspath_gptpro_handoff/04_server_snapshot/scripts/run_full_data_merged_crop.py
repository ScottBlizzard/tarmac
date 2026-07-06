#!/usr/bin/env python3
"""Wrapper for no-padding merged-crop manifest generation."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_manual_review_csv(path: Path) -> Path:
    if path.exists():
        return path
    write_csv(path, [], ["case_id", "image_name", "training_crop_policy", "notes"])
    return path


def safe_name(case_id: str, image_name: str) -> str:
    return f"{case_id}__{Path(image_name).name}"


def stage_images_from_manifest(image_manifest_csv: Path, staging_root: Path) -> dict[str, list[str]]:
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_by_case: dict[str, list[str]] = {}
    for row in read_csv(image_manifest_csv):
        case_id = row.get("case_id", "").strip()
        image_name = row.get("image_name", "").strip()
        image_path = Path(row.get("image_path", "").strip())
        if not case_id or not image_name or not image_path.exists():
            continue
        staged_name = safe_name(case_id, image_name)
        staged_path = staging_root / staged_name
        if not staged_path.exists():
            try:
                staged_path.symlink_to(image_path.resolve())
            except OSError:
                import shutil

                shutil.copy2(image_path, staged_path)
        staged_by_case.setdefault(case_id, []).append(staged_name)
    return staged_by_case


def build_legacy_registry(
    labels_csv: Path,
    split_csv: Path,
    output_csv: Path,
    image_manifest_csv: Path | None = None,
    staging_root: Path | None = None,
) -> tuple[Path, Path | None]:
    labels = {row["case_id"]: row for row in read_csv(labels_csv)}
    splits = {row["case_id"]: row for row in read_csv(split_csv)}
    staged_by_case: dict[str, list[str]] = {}
    if image_manifest_csv is not None:
        if staging_root is None:
            raise ValueError("staging_root is required when image_manifest_csv is provided.")
        staged_by_case = stage_images_from_manifest(image_manifest_csv, staging_root)
    rows = []
    for case_id, label in labels.items():
        split = splits.get(case_id)
        if not split:
            continue
        image_filenames = ";".join(staged_by_case.get(case_id, [])) if staged_by_case else label.get("original_filenames", "")
        rows.append(
            {
                "case_id": case_id,
                "patient_id": label.get("patient_id", case_id) or case_id,
                "source_dataset": label.get("source_dataset", ""),
                "source_case_folder": label.get("source_case_folder", ""),
                "who_type_raw": label.get("who_type", ""),
                "image_count": len(staged_by_case.get(case_id, [])) if staged_by_case else label.get("image_count", ""),
                "image_filenames": image_filenames,
                "is_thymic_hyperplasia": label.get("is_thymic_hyperplasia", ""),
                "is_tet": label.get("is_tet", ""),
                "is_thymoma": label.get("is_thymoma", ""),
                "is_thymic_carcinoma": label.get("is_thymic_carcinoma", ""),
                "low_high_risk_group": label.get("coarse_label_candidate", ""),
                "task_l1_label": label.get("hierarchical_level1", ""),
                "task_l2_label": label.get("hierarchical_level2", ""),
                "task_l3_label": label.get("hierarchical_level3", ""),
                "task_l4_label": label.get("hierarchical_level4", ""),
                "rare_subtype_role": "none",
                "split_stratification_class": label.get("split_stratification_class", label.get("who_type", "")),
                "include_main_study": 1,
            }
        )
    write_csv(output_csv, rows, list(rows[0].keys()) if rows else ["case_id"])
    return output_csv, staging_root if staged_by_case else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merged crop manifest generation with safer defaults.")
    parser.add_argument("--registry-csv", type=Path, help="Legacy-compatible registry CSV.")
    parser.add_argument("--labels-csv", type=Path, help="Standardized labels CSV; used to create a temporary legacy registry if --registry-csv is omitted.")
    parser.add_argument("--image-manifest-csv", type=Path, help="Optional manifest; creates a staged unique-name image root for crop generation.")
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manual-review-csv", type=Path)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--expand-ratio", type=float, default=0.2)
    parser.add_argument("--min-area-frac", type=float, default=0.003)
    parser.add_argument("--panel-size", type=int, default=352)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write command without generating crops.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registry_csv = args.registry_csv
    staged_images_root: Path | None = None
    if registry_csv is None:
        if args.labels_csv is None:
            raise SystemExit("Either --registry-csv or --labels-csv is required.")
        registry_csv, staged_images_root = build_legacy_registry(
            args.labels_csv,
            args.split_csv,
            args.output_dir / "legacy_registry_for_crop.csv",
            args.image_manifest_csv,
            args.output_dir / "staged_images",
        )

    manual_review_csv = args.manual_review_csv or (args.output_dir / "manual_review_empty.csv")
    ensure_manual_review_csv(manual_review_csv)

    cmd = [
        sys.executable,
        "-m",
        "thymic_surimage.generate_crop_manifest",
        "--registry-csv",
        str(registry_csv),
        "--split-csv",
        str(args.split_csv),
        "--images-root",
        str(staged_images_root or args.images_root),
        "--manual-review-csv",
        str(manual_review_csv),
        "--output-dir",
        str(args.output_dir),
        "--top-k",
        str(args.top_k),
        "--expand-ratio",
        str(args.expand_ratio),
        "--min-area-frac",
        str(args.min_area_frac),
        "--panel-size",
        str(args.panel_size),
        "--crop-strategy",
        "merged",
    ]
    (args.output_dir / "crop_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    if args.dry_run:
        report = [
            "# Merged Crop Dry Run",
            "",
            f"- registry_csv: `{registry_csv}`",
            f"- split_csv: `{args.split_csv}`",
            f"- images_root: `{args.images_root}`",
            f"- effective_images_root: `{staged_images_root or args.images_root}`",
            f"- manual_review_csv: `{manual_review_csv}`",
            "",
            "Command was written to `crop_command.txt`; crops were not generated.",
        ]
        (args.output_dir / "crop_generation_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Dry run complete. Command saved to {args.output_dir / 'crop_command.txt'}")
        return 0

    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=project_root)
    report = [
        "# Merged Crop Generation Summary",
        "",
        f"- registry_csv: `{registry_csv}`",
        f"- split_csv: `{args.split_csv}`",
        f"- images_root: `{args.images_root}`",
        f"- effective_images_root: `{staged_images_root or args.images_root}`",
        f"- output_dir: `{args.output_dir}`",
        "",
        "Merged crop generation completed.",
    ]
    (args.output_dir / "crop_generation_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
