from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOW_HIGH_MAP = {"low_risk_group": 0, "high_risk_group": 1, "low_risk": 0, "high_risk": 1}


def label_idx(value: object) -> int:
    text = str(value).strip()
    if text not in LOW_HIGH_MAP:
        raise ValueError(f"Unknown Task7 label: {text!r}")
    return int(LOW_HIGH_MAP[text])


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def index_images(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = re.search(r"(\d{7})", path.name)
        if not match:
            continue
        index.setdefault(match.group(1), []).append(path.resolve())
    for key in list(index):
        index[key] = sorted(index[key], key=lambda p: str(p))
    return index


def candidate_images_from_index(
    case_id: str,
    search_root: Path,
    root_index: dict[str, list[Path]],
    folder: str | None,
    fallback: Path | None,
) -> list[Path]:
    candidates = list(root_index.get(str(case_id), []))
    if folder:
        folder_root = (search_root / str(folder)).resolve()
        folder_candidates = [p for p in candidates if str(p).startswith(str(folder_root))]
        if folder_candidates:
            candidates = folder_candidates
    if not candidates and fallback is not None and fallback.exists():
        candidates = [fallback.resolve()]
    return sorted(candidates, key=lambda p: str(p))


def append_image_rows(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    image_paths: list[Path],
    selected_path: Path | None,
    hash_files: bool,
) -> None:
    selected_resolved = selected_path.resolve() if selected_path is not None and selected_path.exists() else None
    for idx, path in enumerate(image_paths, start=1):
        try:
            stat = path.stat()
            file_size = int(stat.st_size)
            file_sha1 = sha1_file(path) if hash_files else ""
            exists = True
        except OSError:
            file_size = -1
            file_sha1 = ""
            exists = False
        rows.append(
            {
                **base,
                "bag_image_index": idx,
                "bag_n_images": len(image_paths),
                "image_name": path.name,
                "image_path": str(path),
                "image_exists": exists,
                "file_size": file_size,
                "file_sha1": file_sha1,
                "is_selected_image": bool(selected_resolved is not None and path.resolve() == selected_resolved),
            }
        )


def normalize_base(row: pd.Series, domain: str, dataset_role: str) -> dict[str, Any]:
    l7 = str(row.get("task_l7_label", row.get("risk_label", ""))).strip()
    idx = int(row.get("label_idx", label_idx(l7)))
    return {
        "domain": domain,
        "dataset_role": dataset_role,
        "case_id": str(row["case_id"]),
        "original_case_id": str(row.get("original_case_id", row["case_id"])),
        "source_dataset": str(row.get("source_dataset", "")),
        "source_case_folder": str(row.get("source_case_folder", "")),
        "source_folder": str(row.get("source_folder", "")),
        "task_l6_label": str(row.get("task_l6_label", "")),
        "task_l7_label": l7,
        "label_idx": idx,
        "risk_label": "high_risk_group" if idx == 1 else "low_risk_group",
        "master_fold_id": row.get("master_fold_id", ""),
        "is_frozen_external": domain in {"strict_external", "new_external_160"},
    }


def build_old_third(project_root: Path, hash_files: bool) -> list[dict[str, Any]]:
    registry_csv = (
        project_root
        / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/registry.csv"
    )
    registry = pd.read_csv(registry_csv, dtype=str)
    rows: list[dict[str, Any]] = []
    root_map = {
        "batch1": project_root / "datasets/thymic_gross_images",
        "batch2": project_root / "datasets/thymic_gross_images_batch2",
        "third_batch_adapt72_highfocus": project_root / "datasets/third_batch_306_20260521",
        "third_batch_holdout": project_root / "datasets/third_batch_306_20260521",
    }
    root_indexes = {root: index_images(root) for root in set(root_map.values())}
    for _, row in registry.iterrows():
        source = str(row.get("source_dataset", ""))
        domain = "third_batch" if source.startswith("third_batch") else "old_data"
        role = "same_system_adaptation_development" if domain == "third_batch" else "internal_development_oof"
        root = root_map.get(source)
        if root is None:
            raise ValueError(f"Unsupported source_dataset={source!r}")
        selected = Path(str(row.get("selected_original_image_path", "")))
        fallback = selected if selected.exists() else Path(str(row.get("training_image_path", "")))
        images = candidate_images_from_index(
            case_id=str(row.get("original_case_id", row["case_id"])),
            search_root=root,
            root_index=root_indexes[root],
            folder=str(row.get("source_case_folder", "")).strip() or None,
            fallback=fallback,
        )
        append_image_rows(rows, normalize_base(row, domain, role), images, selected, hash_files)
    return rows


def build_external(project_root: Path, registry_rel: str, domain: str, dataset_role: str, hash_files: bool) -> list[dict[str, Any]]:
    registry = pd.read_csv(project_root / registry_rel, dtype=str)
    rows: list[dict[str, Any]] = []
    for _, row in registry.iterrows():
        path = Path(str(row["image_path"]))
        append_image_rows(rows, normalize_base(row, domain, dataset_role), [path], path, hash_files)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task7 all-image case-bag registry for MIL experiments.")
    parser.add_argument("--project-root", default="/workspace/thymic_project")
    parser.add_argument("--out-dir", default="experiments/base_model_expansion_20260706/outputs/case_bag_registry")
    parser.add_argument("--hash-files", action="store_true", help="Compute SHA1 hashes. Slower; off by default.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_old_third(project_root, hash_files=bool(args.hash_files))
    rows.extend(
        build_external(
            project_root,
            "experiments/base_model_expansion_20260706/outputs/registry/strict_external_registry_for_inference.csv",
            "strict_external",
            "frozen_external_stress_test",
            hash_files=bool(args.hash_files),
        )
    )
    rows.extend(
        build_external(
            project_root,
            "experiments/base_model_expansion_20260706/outputs/registry/new_external_160_registry_for_inference.csv",
            "new_external_160",
            "frozen_external_confirmatory",
            hash_files=bool(args.hash_files),
        )
    )

    image_df = pd.DataFrame(rows)
    image_df.to_csv(out_dir / "task7_case_bag_image_registry.csv", index=False, encoding="utf-8-sig")

    case_summary = (
        image_df.groupby(
            [
                "domain",
                "dataset_role",
                "case_id",
                "original_case_id",
                "source_dataset",
                "task_l6_label",
                "task_l7_label",
                "label_idx",
                "risk_label",
                "master_fold_id",
                "is_frozen_external",
            ],
            dropna=False,
        )
        .agg(
            bag_n_images=("image_path", "size"),
            missing_images=("image_exists", lambda s: int((~s.astype(bool)).sum())),
            selected_images=("is_selected_image", "sum"),
            unique_sha1=("file_sha1", "nunique"),
        )
        .reset_index()
    )
    case_summary.to_csv(out_dir / "task7_case_bag_case_summary.csv", index=False, encoding="utf-8-sig")

    audit = {
        "n_cases": int(case_summary["case_id"].nunique()),
        "n_image_rows": int(len(image_df)),
        "missing_images": int((~image_df["image_exists"].astype(bool)).sum()),
        "multi_image_cases": int((case_summary["bag_n_images"] > 1).sum()),
        "domains": {},
        "duplicate_sha1_rows": int(image_df["file_sha1"].duplicated().sum()) if bool(args.hash_files) else None,
        "hash_files": bool(args.hash_files),
    }
    for domain, group in case_summary.groupby("domain", sort=True):
        audit["domains"][domain] = {
            "cases": int(len(group)),
            "image_rows": int(image_df[image_df["domain"] == domain].shape[0]),
            "multi_image_cases": int((group["bag_n_images"] > 1).sum()),
            "max_bag_n_images": int(group["bag_n_images"].max()),
            "low": int((group["label_idx"].astype(int) == 0).sum()),
            "high": int((group["label_idx"].astype(int) == 1).sum()),
        }
    (out_dir / "task7_case_bag_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
