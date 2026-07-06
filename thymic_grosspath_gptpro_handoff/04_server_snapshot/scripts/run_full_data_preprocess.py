#!/usr/bin/env python3
"""One-command preflight and task feasibility summary for full thymic data."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_image_manifest(labels_csv: Path, raw_images_root: Path, output_csv: Path) -> Path:
    labels = read_csv(labels_csv)
    rows: list[dict[str, object]] = []
    for label in labels:
        case_id = label.get("case_id", "").strip()
        patient_id = label.get("patient_id", "").strip() or case_id
        case_dir = raw_images_root / case_id
        image_paths = sorted(p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if case_dir.exists() else []
        for idx, image_path in enumerate(image_paths, start=1):
            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "who_type": label.get("who_type", "").strip(),
                    "source_dataset": label.get("source_dataset", "").strip(),
                    "source_case_folder": label.get("source_case_folder", "").strip(),
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "image_index": idx,
                    "file_exists": int(image_path.exists()),
                    "file_size_bytes": image_path.stat().st_size if image_path.exists() else "",
                }
            )
    write_csv(
        output_csv,
        rows,
        [
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
        ],
    )
    return output_csv


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-data preflight and candidate task summaries.")
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--raw-images-root", required=True, type=Path)
    parser.add_argument("--mapping-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--image-manifest-csv", type=Path, help="Use an existing image manifest instead of scanning raw_images/{case_id}.")
    parser.add_argument("--split-csv", type=Path, help="Optional split CSV for fold balance QC.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True, exist_ok=True)

    image_manifest = args.image_manifest_csv
    if image_manifest is None:
        image_manifest = args.output_root / "image_manifest.csv"
        build_image_manifest(args.labels_csv, args.raw_images_root, image_manifest)

    preflight_dir = args.output_root / "preflight"
    task_dir = args.output_root / "task_feasibility"
    preflight_cmd = [
        sys.executable,
        str(project_root / "scripts/full_data_preflight.py"),
        "--labels-csv",
        str(args.labels_csv),
        "--raw-images-root",
        str(args.raw_images_root),
        "--image-manifest-csv",
        str(image_manifest),
        "--output-dir",
        str(preflight_dir),
    ]
    run(preflight_cmd)

    task_cmd = [
        sys.executable,
        str(project_root / "scripts/summarize_candidate_tasks.py"),
        "--labels-csv",
        str(args.labels_csv),
        "--mapping-csv",
        str(args.mapping_csv),
        "--output-dir",
        str(task_dir),
    ]
    if args.split_csv:
        task_cmd.extend(["--split-csv", str(args.split_csv)])
    run(task_cmd)

    report = [
        "# Full Data Preprocess Summary",
        "",
        f"- labels_csv: `{args.labels_csv}`",
        f"- raw_images_root: `{args.raw_images_root}`",
        f"- image_manifest_csv: `{image_manifest}`",
        f"- preflight_dir: `{preflight_dir}`",
        f"- task_feasibility_dir: `{task_dir}`",
        "",
        "Run completed. Check preflight QC before generating splits or crops.",
    ]
    (args.output_root / "preprocess_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved preprocess summary to {args.output_root / 'preprocess_summary.md'}")


if __name__ == "__main__":
    main()

