from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Task5/6/7 frozen-input variants for multi-image strategy comparison."
    )
    parser.add_argument("--base-input-dir", required=True, help="Existing frozen_inputs directory.")
    parser.add_argument("--batch1-root", required=True, help="Original batch1 image root.")
    parser.add_argument("--batch2-root", required=True, help="Original batch2 image root.")
    parser.add_argument("--output-dir", required=True, help="Output variant directory.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("first", "second", "all"),
        help="For multi-image cases: use first image, second image, or all images.",
    )
    return parser.parse_args()


def natural_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def case_id_from_path(path: Path) -> str:
    match = re.match(r"(\d+)", path.stem)
    if match:
        return match.group(1)
    match = re.match(r"(\d+)", path.parent.name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot infer case_id from path: {path}")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def group_images_by_case(images_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            grouped[case_id_from_path(path)].append(path)
    for case_id, paths in grouped.items():
        grouped[case_id] = sorted(paths, key=lambda p: natural_key(str(p.relative_to(images_root))))
    return grouped


def select_paths(paths: list[Path], strategy: str) -> tuple[list[Path], str]:
    if not paths:
        raise ValueError("No image paths to select from.")
    if len(paths) == 1:
        return [paths[0]], "single_image_use_first"
    if strategy == "first":
        return [paths[0]], "multi_image_use_first"
    if strategy == "second":
        return [paths[1]], "multi_image_use_second"
    return paths, "multi_image_use_all"


def main() -> None:
    args = parse_args()
    base_input_dir = Path(args.base_input_dir)
    batch1_root = Path(args.batch1_root)
    batch2_root = Path(args.batch2_root)
    output_dir = Path(args.output_dir)
    selected_images_dir = output_dir / "selected_images"
    selected_images_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = read_csv_dicts(base_input_dir / "combined_case_registry.csv")
    manifest_rows = read_csv_dicts(base_input_dir / "selected_image_manifest.csv")
    manifest_by_training_case = {row["training_case_id"]: row for row in manifest_rows}

    image_groups = {
        "batch1": group_images_by_case(batch1_root),
        "batch2": group_images_by_case(batch2_root),
    }

    new_registry_rows: list[dict[str, object]] = []
    new_manifest_rows: list[dict[str, object]] = []

    for row in registry_rows:
        training_case_id = row["case_id"]
        original_case_id = row.get("original_case_id", "")
        source_dataset = row["source_dataset"]
        case_groups = image_groups[source_dataset]
        if original_case_id not in case_groups:
            raise FileNotFoundError(f"Case {original_case_id} not found under {source_dataset}.")
        selected_paths, selection_rule = select_paths(case_groups[original_case_id], args.strategy)

        training_names: list[str] = []
        for src_path in selected_paths:
            training_name = f"{training_case_id}_{src_path.name}"
            training_names.append(training_name)
            shutil.copy2(src_path, selected_images_dir / training_name)

        primary_src = selected_paths[0]
        source_root = batch1_root if source_dataset == "batch1" else batch2_root
        primary_rel = str(primary_src.relative_to(source_root)).replace("\\", "/")

        new_row = dict(row)
        new_row["image_count"] = str(len(training_names))
        new_row["image_filenames"] = ";".join(training_names)
        new_row["selected_original_image_name"] = primary_src.name
        new_row["selected_original_image_relpath"] = primary_rel
        new_row["selected_original_image_path"] = str(primary_src)
        new_row["training_image_path"] = str(selected_images_dir / training_names[0])
        new_row["selection_rule"] = selection_rule
        new_registry_rows.append(new_row)

        old_manifest = manifest_by_training_case.get(training_case_id, {})
        manifest_row = dict(old_manifest) if old_manifest else {
            "training_case_id": training_case_id,
            "case_id": original_case_id,
            "source_dataset": source_dataset,
            "who_type_raw": row["who_type_raw"],
            "source_case_folder": row["source_case_folder"],
        }
        manifest_row["original_image_count"] = str(len(case_groups[original_case_id]))
        manifest_row["selected_image_name"] = primary_src.name
        manifest_row["selected_image_relpath"] = primary_rel
        manifest_row["selected_image_path"] = str(primary_src)
        manifest_row["training_image_name"] = training_names[0]
        manifest_row["training_image_path"] = str(selected_images_dir / training_names[0])
        manifest_row["selection_rule"] = selection_rule
        manifest_row["selected_training_image_names"] = ";".join(training_names)
        new_manifest_rows.append(manifest_row)

    registry_fields = list(new_registry_rows[0].keys())
    manifest_field_order = [
        "training_case_id",
        "case_id",
        "source_dataset",
        "who_type_raw",
        "source_case_folder",
        "original_image_count",
        "selected_image_name",
        "selected_image_relpath",
        "selected_image_path",
        "training_image_name",
        "training_image_path",
        "selection_rule",
        "selected_training_image_names",
    ]

    write_csv(output_dir / "combined_case_registry.csv", registry_fields, new_registry_rows)
    write_csv(output_dir / "selected_image_manifest.csv", manifest_field_order, new_manifest_rows)

    for name in [
        "combined_5fold_assignments.csv",
        "combined_experience_label_soft.csv",
        "combined_task567_registry.csv",
        "combined_task567_summary.csv",
        "excluded_from_task567.csv",
        "missing_experience_rows.csv",
        "preflight_report.json",
        "preflight_summary.md",
    ]:
        src = base_input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    summary_lines = [
        "# Task567 Multi-view Variant Summary",
        "",
        f"- Base input dir: `{base_input_dir}`",
        f"- Strategy: `{args.strategy}`",
        f"- Cases: `{len(new_registry_rows)}`",
        f"- Selected image files copied: `{sum(int(row['image_count']) for row in new_registry_rows)}`",
        f"- Multi-image cases in source data: `{sum(1 for row in new_manifest_rows if int(row['original_image_count']) > 1)}`",
        "",
        "Selection rules:",
        "- `single_image_use_first`: source case had only one image",
        "- `multi_image_use_first`: source case had multiple images, used the first one",
        "- `multi_image_use_second`: source case had multiple images, used the second one",
        "- `multi_image_use_all`: source case had multiple images, all images retained for training/evaluation",
    ]
    (output_dir / "variant_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote variant inputs to: {output_dir}")


if __name__ == "__main__":
    main()
