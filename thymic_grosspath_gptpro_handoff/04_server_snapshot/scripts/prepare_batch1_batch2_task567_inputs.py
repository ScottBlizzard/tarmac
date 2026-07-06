from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
WHO_TO_TASK5 = {
    "thymoma_A": "A_AB",
    "thymoma_AB": "A_AB",
    "thymoma_B1": "B123",
    "thymoma_B2": "B123",
    "thymoma_B3": "B123",
    "thymic_carcinoma": "TC",
}
WHO_TO_TASK6 = {
    "thymoma_A": "A",
    "thymoma_AB": "AB",
    "thymoma_B1": "B1",
    "thymoma_B2": "B2",
    "thymoma_B3": "B3",
    "thymic_carcinoma": "TC",
}
WHO_TO_TASK7 = {
    "thymoma_A": "low_risk_group",
    "thymoma_AB": "low_risk_group",
    "thymoma_B1": "low_risk_group",
    "thymoma_B2": "high_risk_group",
    "thymoma_B3": "high_risk_group",
    "thymic_carcinoma": "high_risk_group",
}
TASK6_ORDER = ("A", "AB", "B1", "B2", "B3", "TC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare combined batch1+batch2 one-image-per-case inputs for Task5/6/7."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch1-root", default="datasets/thymic_gross_images")
    parser.add_argument("--batch2-root", default="datasets/thymic_gross_images_batch2")
    parser.add_argument("--batch1-registry", default="datasets/thymic_case_registry_seed_from_folders.csv")
    parser.add_argument(
        "--batch1-experience-csv",
        default="reports/ThymicGross/experience_labeling/task56_experience_label_train_candidates_soft.csv",
    )
    parser.add_argument(
        "--batch2-experience-csv",
        default="reports/ThymicGross/experience_labeling/batch2_experience_label_train_candidates_soft.csv",
    )
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/frozen_inputs")
    parser.add_argument("--fold-count", type=int, default=5)
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


def who_from_folder(folder_name: str) -> str | None:
    if "AB型胸腺瘤" in folder_name:
        return "thymoma_AB"
    if "A型胸腺瘤" in folder_name:
        return "thymoma_A"
    if "B1型胸腺瘤" in folder_name:
        return "thymoma_B1"
    if "B2型胸腺瘤" in folder_name:
        return "thymoma_B2"
    if "B3型胸腺瘤" in folder_name:
        return "thymoma_B3"
    if "胸腺癌" in folder_name:
        return "thymic_carcinoma"
    if "微结节" in folder_name:
        return "micronodular_thymoma"
    if "增生" in folder_name:
        return "benign_hyperplasia"
    return None


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


def index_files_by_name(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    duplicate: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in out:
                duplicate[path.name].extend([str(out[path.name]), str(path)])
            out[path.name] = path
    if duplicate:
        sample = next(iter(duplicate.items()))
        raise ValueError(f"Duplicate image names inside {root}: {sample}")
    return out


def image_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: natural_key(str(p.relative_to(root))),
    )


def select_case_image(paths: list[Path]) -> tuple[Path, str]:
    ordered = sorted(paths, key=lambda p: natural_key(p.name))
    if len(ordered) == 1:
        return ordered[0], "single_image_use_first"
    return ordered[1], "multi_image_use_second"


def load_batch1_cases(batch1_root: Path, registry_csv: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_name = index_files_by_name(batch1_root)
    cases: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for row in read_csv_dicts(registry_csv):
        filenames = [name.strip() for name in row["image_filenames"].split(";") if name.strip()]
        paths = [by_name[name] for name in filenames if name in by_name]
        if not paths:
            excluded.append(
                {
                    "case_id": row["case_id"],
                    "source_dataset": "batch1",
                    "who_type_raw": row["who_type_raw"],
                    "exclude_reason": "no_existing_image",
                }
            )
            continue
        selected, rule = select_case_image(paths)
        rec = dict(row)
        rec.update(
            {
                "source_dataset": "batch1",
                "source_root": str(batch1_root),
                "original_image_count": len(paths),
                "selected_image_name": selected.name,
                "selected_image_relpath": str(selected.relative_to(batch1_root)),
                "selected_image_path": str(selected),
                "selection_rule": rule,
            }
        )
        if rec["who_type_raw"] not in WHO_TO_TASK6:
            rec["exclude_reason"] = "not_in_task567"
            excluded.append(rec)
            continue
        cases.append(rec)
    return cases, excluded


def load_batch2_cases(batch2_root: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in image_files(batch2_root):
        rel = path.relative_to(batch2_root)
        top_folder = rel.parts[0]
        who = who_from_folder(top_folder)
        if who is None:
            raise ValueError(f"Unrecognized batch2 class folder: {top_folder}")
        grouped[(case_id_from_path(path), who)].append(path)

    cases: list[dict[str, object]] = []
    for (case_id, who), paths in sorted(grouped.items(), key=lambda item: (item[0][1], natural_key(item[0][0]))):
        selected, rule = select_case_image(paths)
        source_case_folder = str(selected.relative_to(batch2_root).parent)
        cases.append(
            {
                "case_id": case_id,
                "source_dataset": "batch2",
                "source_root": str(batch2_root),
                "source_case_folder": source_case_folder,
                "who_type_raw": who,
                "original_image_count": len(paths),
                "selected_image_name": selected.name,
                "selected_image_relpath": str(selected.relative_to(batch2_root)),
                "selected_image_path": str(selected),
                "selection_rule": rule,
                "is_thymic_hyperplasia": "0",
                "is_tet": "1",
                "is_thymoma": "0" if who == "thymic_carcinoma" else "1",
                "is_thymic_carcinoma": "1" if who == "thymic_carcinoma" else "0",
                "low_high_risk_group": "low_risk" if WHO_TO_TASK7[who] == "low_risk_group" else "high_risk",
                "task_l1_label": "tet",
                "task_l2_label": "thymic_carcinoma" if who == "thymic_carcinoma" else "thymoma",
                "task_l3_label": "high_risk" if WHO_TO_TASK7[who] == "high_risk_group" else "low_risk",
                "task_l4_label": WHO_TO_TASK6[who] if who != "thymic_carcinoma" else "na",
                "rare_subtype_role": "none",
                "split_stratification_class": who,
            }
        )
    return cases


def unique_link_name(row: dict[str, object]) -> str:
    source = str(row["source_dataset"])
    case_id = str(row["case_id"])
    selected = Path(str(row["selected_image_name"]))
    return f"{source}_{case_id}_{selected.name}"


def create_selected_image_links(cases: list[dict[str, object]], selected_dir: Path) -> None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    for existing in selected_dir.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
    for row in cases:
        link_name = unique_link_name(row)
        src = Path(str(row["selected_image_path"])).resolve()
        dst = selected_dir / link_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        row["training_image_name"] = link_name
        row["training_image_path"] = str(dst)
        row["training_image_materialization"] = "copied_from_original"


def build_registry_rows(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in cases:
        who = str(row["who_type_raw"])
        out = {
            "case_id": f"{row['source_dataset']}_{row['case_id']}",
            "patient_id": f"{row['source_dataset']}_{row['case_id']}",
            "original_case_id": row["case_id"],
            "source_dataset": row["source_dataset"],
            "source_case_folder": row["source_case_folder"],
            "who_type_raw": who,
            "image_count": 1,
            "image_filenames": row["training_image_name"],
            "selected_original_image_name": row["selected_image_name"],
            "selected_original_image_relpath": row["selected_image_relpath"],
            "selected_original_image_path": row["selected_image_path"],
            "training_image_path": row["training_image_path"],
            "selection_rule": row["selection_rule"],
            "original_image_count": row["original_image_count"],
            "is_thymic_hyperplasia": row.get("is_thymic_hyperplasia", "0"),
            "is_tet": row.get("is_tet", "1"),
            "is_thymoma": row.get("is_thymoma", "0" if who == "thymic_carcinoma" else "1"),
            "is_thymic_carcinoma": row.get("is_thymic_carcinoma", "1" if who == "thymic_carcinoma" else "0"),
            "low_high_risk_group": row.get("low_high_risk_group", ""),
            "task_l1_label": row.get("task_l1_label", "tet"),
            "task_l2_label": row.get("task_l2_label", "thymic_carcinoma" if who == "thymic_carcinoma" else "thymoma"),
            "task_l3_label": row.get("task_l3_label", "high_risk" if WHO_TO_TASK7[who] == "high_risk_group" else "low_risk"),
            "task_l4_label": row.get("task_l4_label", WHO_TO_TASK6[who] if who != "thymic_carcinoma" else "na"),
            "rare_subtype_role": row.get("rare_subtype_role", "none"),
            "split_stratification_class": who,
            "task_l5_label": WHO_TO_TASK5[who],
            "task_l6_label": WHO_TO_TASK6[who],
            "task_l7_label": WHO_TO_TASK7[who],
            "include_main_study": 1,
        }
        rows.append(out)
    return sorted(rows, key=lambda r: (str(r["source_dataset"]), TASK6_ORDER.index(str(r["task_l6_label"])), natural_key(str(r["original_case_id"]))))


def assign_folds(registry_rows: list[dict[str, object]], fold_count: int) -> list[dict[str, object]]:
    fold_sizes = [0 for _ in range(fold_count)]
    out: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in registry_rows:
        grouped[str(row["task_l6_label"])].append(row)
    for label in TASK6_ORDER:
        rows = grouped[label]
        for idx, row in enumerate(rows):
            preferred = idx % fold_count
            min_size = min(fold_sizes)
            candidates = [i for i, size in enumerate(fold_sizes) if size == min_size]
            fold_idx = preferred if preferred in candidates else candidates[0]
            fold_sizes[fold_idx] += 1
            out.append(
                {
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "original_case_id": row["original_case_id"],
                    "source_dataset": row["source_dataset"],
                    "who_type_raw": row["who_type_raw"],
                    "include_main_study": 1,
                    "task_l5_label": row["task_l5_label"],
                    "task_l6_label": row["task_l6_label"],
                    "task_l7_label": row["task_l7_label"],
                    "split_stratification_class": row["split_stratification_class"],
                    "master_fold_id": fold_idx + 1,
                    "notes": "batch1_batch2_task567_rebalanced_by_task6",
                }
            )
    return sorted(out, key=lambda r: natural_key(str(r["case_id"])))


def summarize_by_task(registry_rows: list[dict[str, object]], fold_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fold_by_case = {row["case_id"]: int(row["master_fold_id"]) for row in fold_rows}
    summary: list[dict[str, object]] = []
    for task_name, label_col in [
        ("task5_threeclass", "task_l5_label"),
        ("task6_sixclass", "task_l6_label"),
        ("task7_lowhigh_tc", "task_l7_label"),
    ]:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in registry_rows:
            grouped[str(row[label_col])].append(row)
        for label, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
            out = {
                "task": task_name,
                "label_name": label,
                "num_cases": len(rows),
                "num_images": len(rows),
            }
            for fold_id in range(1, 6):
                out[f"fold_{fold_id}_cases"] = sum(1 for r in rows if fold_by_case[r["case_id"]] == fold_id)
            summary.append(out)
    return summary


def load_experience_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                row["_experience_source_csv"] = str(path)
                rows.append(row)
    return rows


def combine_experience_rows(
    experience_rows: list[dict[str, str]],
    registry_rows: list[dict[str, object]],
    selected_manifest_by_case: dict[tuple[str, str], dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    exp_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in experience_rows:
        source_dataset = "batch2" if "batch2" in row["_experience_source_csv"] else "batch1"
        exp_by_key[(source_dataset, row["case_id"], row["image_name"])] = row

    fieldnames = [name for name in experience_rows[0].keys() if name != "_experience_source_csv"]
    out: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for reg in registry_rows:
        source_dataset = str(reg["source_dataset"])
        original_case_id = str(reg["original_case_id"])
        selected = selected_manifest_by_case[(source_dataset, original_case_id)]
        selected_name = str(selected["selected_image_name"])
        exp = exp_by_key.get((source_dataset, original_case_id, selected_name))
        if exp is None:
            matching_case = [
                row
                for key, row in exp_by_key.items()
                if key[0] == source_dataset and key[1] == original_case_id
            ]
            if matching_case:
                exp = dict(matching_case[0])
                exp["exp_round3_notes"] = (
                    exp.get("exp_round3_notes", "")
                    + " | experience row existed for same case but not selected image; reused case-level row"
                ).strip(" |")
            else:
                missing.append(
                    {
                        "case_id": reg["case_id"],
                        "source_dataset": source_dataset,
                        "original_case_id": original_case_id,
                        "selected_image_name": selected_name,
                        "reason": "no_experience_row_for_case",
                    }
                )
                continue
        new_row = {name: exp.get(name, "") for name in fieldnames}
        new_row["case_id"] = reg["case_id"]
        new_row["task5_label"] = reg["task_l5_label"]
        new_row["task6_label"] = reg["task_l6_label"]
        new_row["who_type_raw"] = reg["who_type_raw"]
        new_row["image_name"] = reg["image_filenames"]
        new_row["image_relpath"] = reg["image_filenames"]
        new_row["local_image_path"] = reg["training_image_path"]
        new_row["exp_round3_notes"] = (
            new_row.get("exp_round3_notes", "")
            + f" | combined_batch1_batch2 selected_original={selected_name}"
        ).strip(" |")
        out.append(new_row)
    return out, missing


def write_preflight_report(
    path: Path,
    registry_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    experience_rows: list[dict[str, object]],
    missing_experience: list[dict[str, object]],
    excluded_rows: list[dict[str, object]],
) -> dict[str, object]:
    labels = Counter(str(r["task_l6_label"]) for r in registry_rows)
    folds = Counter(int(r["master_fold_id"]) for r in fold_rows)
    source_counts = Counter(str(r["source_dataset"]) for r in registry_rows)
    multi_selected = sum(1 for r in selected_rows if int(r["original_image_count"]) > 1)
    missing_images = [r for r in registry_rows if not Path(str(r["training_image_path"])).exists()]
    duplicate_training_names = [
        name for name, count in Counter(str(r["image_filenames"]) for r in registry_rows).items() if count > 1
    ]
    fold_label_counts: dict[str, dict[str, int]] = {}
    for fold_id in range(1, 6):
        sub = [r for r in fold_rows if int(r["master_fold_id"]) == fold_id]
        fold_label_counts[str(fold_id)] = dict(Counter(str(r["task_l6_label"]) for r in sub))

    report = {
        "status": "pass" if not missing_images and not duplicate_training_names else "fail",
        "num_task567_cases": len(registry_rows),
        "num_selected_images": len(selected_rows),
        "num_experience_rows": len(experience_rows),
        "num_missing_experience_rows": len(missing_experience),
        "num_excluded_rows": len(excluded_rows),
        "num_multi_image_cases_selected_second": multi_selected,
        "source_counts": dict(source_counts),
        "task6_counts": dict(labels),
        "fold_counts": dict(folds),
        "fold_task6_counts": fold_label_counts,
        "missing_training_images": missing_images[:20],
        "duplicate_training_image_names": duplicate_training_names[:20],
        "missing_experience_rows_sample": missing_experience[:20],
        "excluded_rows_by_reason": dict(Counter(str(r.get("exclude_reason", "")) for r in excluded_rows)),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    project = Path(args.project_root).resolve()
    batch1_root = project / args.batch1_root
    batch2_root = project / args.batch2_root
    out_dir = project / args.output_dir
    selected_dir = out_dir / "selected_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch1_cases, excluded = load_batch1_cases(batch1_root, project / args.batch1_registry)
    batch2_cases = load_batch2_cases(batch2_root)
    all_cases = batch1_cases + batch2_cases
    duplicate_ids = [case_id for case_id, count in Counter(str(r["case_id"]) for r in all_cases).items() if count > 1]
    if duplicate_ids:
        # IDs can overlap across batches; training case_id is source-prefixed below.
        pass

    create_selected_image_links(all_cases, selected_dir)
    registry_rows = build_registry_rows(all_cases)

    selected_fields = [
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
    ]
    selected_rows = sorted(
        [
            {
                **{field: row.get(field, "") for field in selected_fields},
                "training_case_id": f"{row['source_dataset']}_{row['case_id']}",
            }
            for row in all_cases
        ],
        key=lambda r: natural_key(str(r["training_case_id"])),
    )
    write_csv(out_dir / "selected_image_manifest.csv", ["training_case_id"] + selected_fields, selected_rows)

    registry_fields = [
        "case_id",
        "patient_id",
        "original_case_id",
        "source_dataset",
        "source_case_folder",
        "who_type_raw",
        "image_count",
        "image_filenames",
        "selected_original_image_name",
        "selected_original_image_relpath",
        "selected_original_image_path",
        "training_image_path",
        "selection_rule",
        "original_image_count",
        "is_thymic_hyperplasia",
        "is_tet",
        "is_thymoma",
        "is_thymic_carcinoma",
        "low_high_risk_group",
        "task_l1_label",
        "task_l2_label",
        "task_l3_label",
        "task_l4_label",
        "rare_subtype_role",
        "split_stratification_class",
        "task_l5_label",
        "task_l6_label",
        "task_l7_label",
        "include_main_study",
    ]
    write_csv(out_dir / "combined_case_registry.csv", registry_fields, registry_rows)
    write_csv(out_dir / "combined_task567_registry.csv", registry_fields, registry_rows)

    fold_rows = assign_folds(registry_rows, args.fold_count)
    fold_fields = [
        "case_id",
        "patient_id",
        "original_case_id",
        "source_dataset",
        "who_type_raw",
        "include_main_study",
        "task_l5_label",
        "task_l6_label",
        "task_l7_label",
        "split_stratification_class",
        "master_fold_id",
        "notes",
    ]
    write_csv(out_dir / "combined_5fold_assignments.csv", fold_fields, fold_rows)
    summary_rows = summarize_by_task(registry_rows, fold_rows)
    summary_fields = ["task", "label_name", "num_cases", "num_images"] + [f"fold_{i}_cases" for i in range(1, 6)]
    write_csv(out_dir / "combined_task567_summary.csv", summary_fields, summary_rows)

    selected_by_case = {(str(r["source_dataset"]), str(r["case_id"])): r for r in all_cases}
    exp_rows = load_experience_rows([project / args.batch1_experience_csv, project / args.batch2_experience_csv])
    combined_exp, missing_exp = combine_experience_rows(exp_rows, registry_rows, selected_by_case)
    exp_fields = [name for name in exp_rows[0].keys() if name != "_experience_source_csv"]
    write_csv(out_dir / "combined_experience_label_soft.csv", exp_fields, combined_exp)
    write_csv(
        out_dir / "missing_experience_rows.csv",
        ["case_id", "source_dataset", "original_case_id", "selected_image_name", "reason"],
        missing_exp,
    )
    write_csv(
        out_dir / "excluded_from_task567.csv",
        sorted({key for row in excluded for key in row.keys()}),
        excluded,
    )
    report = write_preflight_report(
        out_dir / "preflight_report.json",
        registry_rows,
        fold_rows,
        selected_rows,
        combined_exp,
        missing_exp,
        excluded,
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Batch1 + Batch2 Task5/6/7 Inputs",
                "",
                "Generated one representative image per case.",
                "",
                "- Single-image case: use the first image.",
                "- Multi-image case: use the second image.",
                "- `selected_images/` contains unique-name copied representative images used as training image root.",
                "- `combined_case_registry.csv` and `combined_5fold_assignments.csv` are the training inputs.",
                "",
                "Use `selected_images` as `--images-root` for Task5/6/7 runs.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
