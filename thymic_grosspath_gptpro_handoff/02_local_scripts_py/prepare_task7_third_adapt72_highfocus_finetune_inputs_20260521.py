from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd


PROFILE = {"AB": 24, "B1": 4, "B2": 22, "TC": 22}
WHO_MAP = {"AB": "thymoma_AB", "B1": "thymoma_B1", "B2": "thymoma_B2", "TC": "thymic_carcinoma"}
L5_MAP = {"AB": "A_AB", "B1": "B123", "B2": "B123", "TC": "TC"}
L7_MAP = {"AB": "low_risk_group", "B1": "low_risk_group", "B2": "high_risk_group", "TC": "high_risk_group"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task7 third-batch adapt72 high-focus finetune input registry.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-registry-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv")
    parser.add_argument("--third-registry-csv", default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521/third_batch_task7_registry.csv")
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/06_adapt72_highfocus_finetune_inputs_20260521")
    parser.add_argument("--adapt-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    return parser.parse_args()


def stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def select_adapt(third: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = []
    for subtype, n in PROFILE.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        chosen.extend(g.sort_values("_key").head(n).index.tolist())
    mask = third.index.isin(chosen)
    return third.loc[mask].copy().reset_index(drop=True), third.loc[~mask].copy().reset_index(drop=True)


def assign_old_val_train(old: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for label, g in old.groupby("task_l7_label", sort=True):
        g = g.copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        g = g.sort_values("_key").reset_index(drop=True)
        val_n = max(8, int(round(len(g) * 0.15)))
        for i, row in g.iterrows():
            fold = 2 if i < val_n else 3 + ((i - val_n) % 3)
            rows.append({"case_id": row["case_id"], "patient_id": row.get("patient_id", row["case_id"]), "master_fold_id": fold, "notes": "old_val" if fold == 2 else "old_train"})
    return pd.DataFrame(rows)


def old_registry_rows(root: Path, old: pd.DataFrame, images_dir: Path, copy: bool) -> pd.DataFrame:
    rows = []
    for row in old.to_dict("records"):
        src = Path(row["training_image_path"])
        if not src.exists():
            src = root / str(row["training_image_path"])
        dst_name = safe_name(src.name)
        link_or_copy(src, images_dir / dst_name, copy)
        out = dict(row)
        out["image_filenames"] = dst_name
        out["include_main_study"] = 1
        rows.append(out)
    return pd.DataFrame(rows)


def third_registry_rows(third: pd.DataFrame, images_dir: Path, split_name: str, repeat: int, copy: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg_rows = []
    split_rows = []
    for row in third.to_dict("records"):
        subtype = str(row["task_l6_label"])
        src = Path(row["image_path"])
        base_name = f"third_{subtype}_{row['original_case_id']}_{Path(row['image_name']).name}"
        base_name = safe_name(base_name)
        repeats = repeat if split_name == "adapt_train" else 1
        for rep in range(repeats):
            case_id = str(row["case_id"]) if rep == 0 else f"{row['case_id']}_rep{rep}"
            image_name = base_name if rep == 0 else safe_name(f"rep{rep}_{base_name}")
            link_or_copy(src, images_dir / image_name, copy)
            low_high = L7_MAP[subtype]
            reg_rows.append(
                {
                    "case_id": case_id,
                    "patient_id": case_id,
                    "original_case_id": str(row["original_case_id"]),
                    "source_dataset": "third_batch_adapt72_highfocus" if split_name == "adapt_train" else "third_batch_holdout",
                    "source_case_folder": str(row["source_folder"]),
                    "who_type_raw": WHO_MAP[subtype],
                    "image_count": 1,
                    "image_filenames": image_name,
                    "selected_original_image_name": str(row["image_name"]),
                    "selected_original_image_relpath": str(row["selected_original_image_relpath"]),
                    "selected_original_image_path": str(row["image_path"]),
                    "training_image_path": str(images_dir / image_name),
                    "selection_rule": split_name,
                    "original_image_count": 1,
                    "is_thymic_hyperplasia": 0,
                    "is_tet": 1,
                    "is_thymoma": 0 if subtype == "TC" else 1,
                    "is_thymic_carcinoma": 1 if subtype == "TC" else 0,
                    "low_high_risk_group": "high_risk" if low_high == "high_risk_group" else "low_risk",
                    "task_l1_label": "tet",
                    "task_l2_label": "thymic_carcinoma" if subtype == "TC" else "thymoma",
                    "task_l3_label": "high_risk" if low_high == "high_risk_group" else "low_risk",
                    "task_l4_label": subtype,
                    "task_l5_label": L5_MAP[subtype],
                    "task_l6_label": subtype,
                    "task_l7_label": low_high,
                    "split_stratification_class": WHO_MAP[subtype],
                    "include_main_study": 1,
                    "pathology_id_norm": str(row["original_case_id"]),
                }
            )
            if split_name == "holdout":
                fold = 1
                notes = "third_holdout"
            else:
                fold = 3 + (rep % 3)
                notes = "third_adapt_train_repeated"
            split_rows.append({"case_id": case_id, "patient_id": case_id, "master_fold_id": fold, "notes": notes})
    return pd.DataFrame(reg_rows), pd.DataFrame(split_rows)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    images_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(root / args.old_registry_csv, dtype={"case_id": str, "patient_id": str, "original_case_id": str})
    third = pd.read_csv(root / args.third_registry_csv, dtype={"case_id": str, "original_case_id": str})
    adapt, holdout = select_adapt(third, args.seed)

    old_reg = old_registry_rows(root, old, images_dir, args.copy)
    adapt_reg, adapt_split = third_registry_rows(adapt, images_dir, "adapt_train", args.adapt_repeat, args.copy)
    hold_reg, hold_split = third_registry_rows(holdout, images_dir, "holdout", 1, args.copy)
    registry = pd.concat([old_reg, adapt_reg, hold_reg], ignore_index=True, sort=False)
    old_split = assign_old_val_train(old_reg, args.seed)
    split = pd.concat([old_split, adapt_split, hold_split], ignore_index=True, sort=False)

    registry.to_csv(out / "registry.csv", index=False, encoding="utf-8-sig")
    split.to_csv(out / "split.csv", index=False, encoding="utf-8-sig")
    adapt.to_csv(out / "adapt72_highfocus_source_cases.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(out / "holdout_source_cases.csv", index=False, encoding="utf-8-sig")
    split.groupby(["master_fold_id", "notes"]).size().reset_index(name="n").to_csv(out / "split_counts.csv", index=False, encoding="utf-8-sig")
    registry["task_l7_label"].value_counts().rename_axis("task_l7_label").reset_index(name="n").to_csv(out / "label_counts.csv", index=False, encoding="utf-8-sig")
    print(f"registry={len(registry)} split={len(split)} images={len(list(images_dir.iterdir()))}", flush=True)
    print(split.groupby(["master_fold_id", "notes"]).size(), flush=True)


if __name__ == "__main__":
    main()
