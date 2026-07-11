from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


LOW_HIGH_MAP = {
    "low_risk_group": 0,
    "high_risk_group": 1,
    "low_risk": 0,
    "high_risk": 1,
}


def label_idx_from_l7(value: object) -> int:
    text = str(value)
    if text not in LOW_HIGH_MAP:
        raise ValueError(f"Unknown task_l7_label: {text!r}")
    return LOW_HIGH_MAP[text]


def normalize_common(df: pd.DataFrame, domain: str, dataset_role: str) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = domain
    out["dataset_role"] = dataset_role
    if "label_idx" not in out.columns:
        out["label_idx"] = out["task_l7_label"].map(lambda x: label_idx_from_l7(x))
    out["label_idx"] = out["label_idx"].astype(int)
    if "image_path" not in out.columns:
        for candidate in ["training_image_path", "staging_image_path", "selected_original_image_path"]:
            if candidate in out.columns:
                out["image_path"] = out[candidate]
                break
    if "image_name" not in out.columns:
        for candidate in ["staging_image_name", "selected_original_image_name", "canonical_filename"]:
            if candidate in out.columns:
                out["image_name"] = out[candidate]
                break
    if "image_name" not in out.columns and "image_path" in out.columns:
        out["image_name"] = out["image_path"].map(lambda p: Path(str(p)).name)
    out["risk_label"] = out["label_idx"].map({0: "low_risk_group", 1: "high_risk_group"})
    out["is_frozen_external"] = domain.isin(["strict_external", "new_external_160"]) if hasattr(domain, "isin") else domain in {"strict_external", "new_external_160"}
    return out


def load_old(project_root: Path) -> pd.DataFrame:
    old_csv = project_root / "outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry.csv"
    folds_csv = project_root / "outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv"
    old = pd.read_csv(old_csv, dtype=str)
    if "include_main_study" in old.columns:
        old = old[old["include_main_study"].astype(str) == "1"].copy()
    old = normalize_common(old, "old_data", "internal_development_oof")
    old["image_path"] = old["training_image_path"]
    old["image_name"] = old["selected_original_image_name"]
    if folds_csv.exists():
        folds = pd.read_csv(folds_csv, dtype=str)[["case_id", "master_fold_id"]]
        old = old.merge(folds, on="case_id", how="left")
    return old


def load_third(project_root: Path) -> pd.DataFrame:
    sys.path.insert(0, str(project_root / "scripts"))
    from run_task7_external_third_batch_64style_20260521 import build_third_registry

    third_root = project_root / "datasets/third_batch_306_20260521"
    third = build_third_registry(third_root)
    third = normalize_common(third, "third_batch", "same_system_adaptation_development")
    third["master_fold_id"] = ""
    return third


def load_strict(project_root: Path) -> pd.DataFrame:
    sys.path.insert(0, str(project_root / "scripts"))
    from run_task7_external_thymoma_carcinoma_folder_20260522 import build_external_registry

    strict_root = project_root / "datasets/external_thymoma_carcinoma_20260522"
    strict = build_external_registry(strict_root)
    strict = normalize_common(strict, "strict_external", "frozen_external_stress_test")
    strict["master_fold_id"] = ""
    return strict


def load_new_external(project_root: Path) -> pd.DataFrame:
    new_csv = (
        project_root
        / "experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/new_external_160_case_manifest.csv"
    )
    new = pd.read_csv(new_csv, dtype=str)
    new = normalize_common(new, "new_external_160", "frozen_external_confirmatory")
    new["image_path"] = new["staging_image_path"]
    new["image_name"] = new["staging_image_name"]
    new["master_fold_id"] = ""
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/thymic_project")
    parser.add_argument("--out-dir", default="experiments/base_model_expansion_20260706/outputs/registry")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "old_data": load_old(project_root),
        "third_batch": load_third(project_root),
        "strict_external": load_strict(project_root),
        "new_external_160": load_new_external(project_root),
    }
    common_cols = [
        "domain",
        "dataset_role",
        "case_id",
        "original_case_id",
        "source_dataset",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "risk_label",
        "image_name",
        "image_path",
        "master_fold_id",
        "source_case_folder",
        "source_folder",
        "selection_rule",
        "image_count",
        "original_image_count",
        "is_frozen_external",
    ]

    normalized = []
    for frame in frames.values():
        for col in common_cols:
            if col not in frame.columns:
                frame[col] = ""
        normalized.append(frame[common_cols].copy())
    master = pd.concat(normalized, ignore_index=True)
    master["image_exists"] = master["image_path"].map(lambda p: Path(str(p)).exists())

    master_csv = out_dir / "task7_four_domain_master_registry.csv"
    master.to_csv(master_csv, index=False, encoding="utf-8-sig")

    summary = {
        "total_cases": int(len(master)),
        "domains": {
            domain: {
                "n": int(len(g)),
                "low": int((g["label_idx"] == 0).sum()),
                "high": int((g["label_idx"] == 1).sum()),
                "missing_images": int((~g["image_exists"]).sum()),
                "subtypes": g["task_l6_label"].value_counts(dropna=False).sort_index().to_dict(),
            }
            for domain, g in master.groupby("domain", sort=True)
        },
    }
    (out_dir / "task7_four_domain_master_registry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {master_csv}")


if __name__ == "__main__":
    main()
