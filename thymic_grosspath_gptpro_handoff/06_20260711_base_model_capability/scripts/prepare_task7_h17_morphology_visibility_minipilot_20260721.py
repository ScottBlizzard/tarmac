from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


SEED = 20260721
LOW_RISK = {"A", "AB", "B1"}
QUOTAS = {
    ("A", "batch1"): 1,
    ("A", "batch2"): 2,
    ("AB", "batch1"): 1,
    ("AB", "batch2"): 1,
    ("AB", "third_batch"): 2,
    ("B1", "batch1"): 1,
    ("B1", "batch2"): 1,
    ("B1", "third_batch"): 1,
    ("B2", "batch1"): 1,
    ("B2", "batch2"): 1,
    ("B2", "third_batch"): 2,
    ("B3", "batch1"): 2,
    ("B3", "batch2"): 1,
    ("TC", "batch1"): 1,
    ("TC", "batch2"): 1,
    ("TC", "third_batch"): 1,
}
CONCEPT_COLUMNS = {
    "M1_capsule_boundary_complete_clear": "包膜或外周边界完整、连续、相对清楚",
    "M2_capsule_disrupted_unclear_irregular": "包膜中断、边界模糊、不规则或疑似跨界连续",
    "M3_cut_surface_pale_homogeneous": "主切面整体偏苍白且相对均质",
    "M4_cut_surface_heterogeneous_mottled": "主切面明显异质、斑驳或不同质地区域并存",
    "M5_internal_hemorrhage_necrosis_cystic": "明确出血、坏死、囊变或软化样内部异常",
    "M6_lobulated_nodular_septated": "分叶、结节、多结节或纤维隔样结构",
    "M7_continuity_with_adjacent_tissue": "异常组织与邻近组织连续、黏连或疑似受累",
    "M8_internal_abnormality_confined": "内部异常区局限于肿物内部，未到达外周界面",
    "M9_interfering_artifact": "阴影、反光、血液覆盖、破损或器械伪影影响判断",
    "M10_valid_cut_surface_visible": "当前图像存在有效主切面，可用于切面形态判断",
}
RELATION_COLUMNS = {
    "R1_internal_abnormality_enclosed_by_boundary": "内部异常区是否被完整边界包围",
    "R2_abnormality_internal_peripheral_or_crossing": "异质或异常区位于内部、外周或跨越外周界面",
    "R3_continuous_path_to_adjacent_tissue": "肿物异常区是否与邻近组织形成连续通路",
    "R4_lobulation_corresponds_to_heterogeneity": "分叶/结节/纤维隔与异质区是否空间对应",
    "R5_internal_damage_with_intact_capsule": "明显内部损伤与完整包膜是否同时存在",
    "R6_pale_homogeneous_with_clear_boundary": "苍白均质切面与完整边界是否同时存在",
}
REGIONS = {
    "Z0_global_specimen": "全局标本",
    "Z1_main_cut_surface": "主切面",
    "Z2_capsule_outer_interface": "外周包膜/边界界面",
    "Z3_internal_abnormality": "内部异常区",
    "Z4_adjacent_tissue_interface": "邻近组织界面",
    "Z5_artifact": "伪影区",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the locked H17 20-case morphology visibility mini-pilot."
    )
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--search-trials", type=int, default=20_000)
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def write_hash_manifest(path: Path, files: list[Path], root: Path) -> None:
    lines = [
        f"{sha256(file_path)}  {file_path.relative_to(root).as_posix()}"
        for file_path in files
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def normalize_source(value: Any) -> str:
    text = str(value).lstrip("\ufeff").strip()
    return "third_batch" if text.startswith("third_batch") else text


def external_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
        .to_numpy(dtype=bool)
    )


def load_registry(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    frame["source_dataset"] = frame["source_dataset"].map(normalize_source)
    frame = frame.loc[
        frame["dataset_role"].isin(
            {
                "internal_development_oof",
                "same_system_adaptation_development",
            }
        )
    ].copy()
    frame = frame.drop_duplicates("case_id", keep="first")
    if len(frame) != 591 or frame["case_id"].nunique() != 591:
        raise ValueError("Mini-pilot requires the locked 591-case internal cohort")
    if external_mask(frame["is_frozen_external"]).any():
        raise ValueError("Strict external cases entered the mini-pilot")
    if "image_exists" in frame:
        image_exists = frame["image_exists"]
        if image_exists.dtype == bool:
            image_exists_mask = image_exists.to_numpy(dtype=bool)
        else:
            image_exists_mask = (
                image_exists.astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "yes"})
                .to_numpy(dtype=bool)
            )
        if not image_exists_mask.all():
            raise FileNotFoundError("Registry contains missing internal images")
    missing = [
        str(path)
        for path in frame["image_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source images")
    if frame["task_l6_label"].value_counts().to_dict() != {
        "A": 44,
        "AB": 262,
        "B1": 62,
        "B2": 89,
        "B3": 24,
        "TC": 110,
    }:
        raise ValueError("Subtype totals differ from the locked cohort")
    frame["risk_group_private"] = np.where(
        frame["task_l6_label"].isin(LOW_RISK),
        "low",
        "high",
    )
    return frame.reset_index(drop=True)


def sample_once(
    registry: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (subtype, source), count in QUOTAS.items():
        stratum = registry.loc[
            registry["task_l6_label"].eq(subtype)
            & registry["source_dataset"].eq(source)
        ]
        if len(stratum) < count:
            raise ValueError(
                f"Insufficient cases for subtype={subtype} source={source}: "
                f"available={len(stratum)} required={count}"
            )
        positions = rng.choice(len(stratum), size=count, replace=False)
        rows.append(stratum.iloc[np.sort(positions)].copy())
    selected = pd.concat(rows, ignore_index=True)
    if len(selected) != 20 or selected["case_id"].duplicated().any():
        raise ValueError("Mini-pilot selection is not 20 unique cases")
    return selected


def selection_score(selected: pd.DataFrame) -> tuple[int, int, str]:
    fold_counts = (
        selected["master_fold_id"].astype(int).value_counts().reindex(range(1, 6), fill_value=0)
    )
    fold_penalty = int(np.square(fold_counts.to_numpy(dtype=int) - 4).sum())
    source_fold = pd.crosstab(
        selected["source_dataset"],
        selected["master_fold_id"].astype(int),
    ).reindex(index=["batch1", "batch2", "third_batch"], columns=range(1, 6), fill_value=0)
    empty_source_fold = int((source_fold == 0).sum().sum())
    tie = hashlib.sha256(
        "|".join(sorted(selected["case_id"].astype(str))).encode()
    ).hexdigest()
    return fold_penalty, empty_source_fold, tie


def select_cases(
    registry: pd.DataFrame,
    seed: int,
    trials: int,
) -> tuple[pd.DataFrame, tuple[int, int, str]]:
    rng = np.random.default_rng(seed)
    best: pd.DataFrame | None = None
    best_score: tuple[int, int, str] | None = None
    for _ in range(trials):
        candidate = sample_once(registry, rng)
        score = selection_score(candidate)
        if best_score is None or score < best_score:
            best = candidate
            best_score = score
            if score[:2] == (0, 0):
                break
    if best is None or best_score is None:
        raise RuntimeError("Mini-pilot selection search returned no candidate")
    return best, best_score


def annotation_columns() -> list[str]:
    columns = [
        "pilot_id",
        "reader_id",
        "annotation_status",
        "image_quality",
        "global_morphology_judgeable",
        "dominant_visibility_limitation",
    ]
    for region in REGIONS:
        columns.extend(
            [
                f"{region}_visibility",
                f"{region}_x1_norm",
                f"{region}_y1_norm",
                f"{region}_x2_norm",
                f"{region}_y2_norm",
            ]
        )
    columns.extend(CONCEPT_COLUMNS)
    columns.extend(RELATION_COLUMNS)
    columns.extend(
        [
            "additional_view_needed",
            "morphology_only_comment",
        ]
    )
    return columns


def build_instructions() -> str:
    concept_lines = "\n".join(
        f"- `{column}`：{description}" for column, description in CONCEPT_COLUMNS.items()
    )
    relation_lines = "\n".join(
        f"- `{column}`：{description}"
        for column, description in RELATION_COLUMNS.items()
    )
    region_lines = "\n".join(
        f"- `{column}`：{description}" for column, description in REGIONS.items()
    )
    return f"""# H17 Morphology-Only Visibility Mini-Pilot

本 pilot 只验证当前这张大体照片中的形态是否可见、是否可重复标注。

## 严格禁止

- 不判断低危或高危；
- 不判断六亚型；
- 不核验模型是否判错；
- 不查看模型概率、风险标签、亚型或来源；
- 不根据经验推断图外信息；
- `not_visible_or_uncertain` 不得改写成 `absent`。

## 状态值

- 区域可见性：`visible`、`not_visible`、`uncertain`；
- M1-M10 与 R1-R6：`present`、`absent`、`not_visible_or_uncertain`；
- `annotation_status`：完成后填写 `complete`；
- 坐标采用 `[0,1]` 归一化矩形框。区域不可见时坐标留空。

## 区域

{region_lines}

## 形态概念

{concept_lines}

## 空间关系

{relation_lines}

## 目标

先判断至少 4 个非伪影概念是否能在至少 70% 图像中被明确标为 present/absent，
再计算两位 reader 的一致性。该 packet 不会直接用于低危/高危训练。
"""


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError(f"H17 mini-pilot seed is locked to {SEED}")
    if args.search_trials < 1000:
        raise ValueError("Selection search requires at least 1000 trials")
    registry_path = Path(args.registry_csv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty packet: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    blinded_dir = output_dir / "blinded_packet"
    secure_dir = output_dir / "secure_do_not_share"
    image_dir = blinded_dir / "blinded_images"
    blinded_dir.mkdir(parents=True)
    secure_dir.mkdir(parents=True)
    if args.copy_images:
        image_dir.mkdir(parents=True)

    registry = load_registry(registry_path)
    selected, score = select_cases(
        registry,
        seed=args.seed,
        trials=args.search_trials,
    )
    shuffle_key = selected["case_id"].map(
        lambda value: hashlib.sha256(
            f"h17-mini-order|{args.seed}|{value}".encode()
        ).hexdigest()
    )
    selected = (
        selected.assign(_shuffle_key=shuffle_key)
        .sort_values("_shuffle_key")
        .drop(columns="_shuffle_key")
        .reset_index(drop=True)
    )
    selected["pilot_id"] = [
        f"MV{index:03d}_{hashlib.sha256(f'{args.seed}|{case_id}'.encode()).hexdigest()[:6].upper()}"
        for index, case_id in enumerate(selected["case_id"].astype(str), start=1)
    ]

    image_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        source_path = Path(str(row["image_path"]))
        suffix = source_path.suffix.lower() or ".jpg"
        filename = f"{row['pilot_id']}{suffix}"
        with Image.open(source_path) as image:
            width, height = image.size
        if args.copy_images:
            shutil.copy2(source_path, image_dir / filename)
        image_rows.append(
            {
                "pilot_id": row["pilot_id"],
                "image_filename": filename,
                "width_px": width,
                "height_px": height,
            }
        )
    image_manifest = pd.DataFrame(image_rows)
    write_csv(blinded_dir / "BLINDED_IMAGE_MANIFEST.csv", image_manifest)

    secure = selected.merge(
        image_manifest,
        on="pilot_id",
        how="left",
        validate="one_to_one",
    )
    secure_columns = [
        "pilot_id",
        "image_filename",
        "case_id",
        "original_case_id",
        "source_dataset",
        "task_l6_label",
        "risk_group_private",
        "master_fold_id",
        "image_path",
        "image_name",
        "width_px",
        "height_px",
    ]
    write_csv(
        secure_dir / "SECURE_CASE_KEY_DO_NOT_SHARE.csv",
        secure[secure_columns],
    )
    selection_summary = (
        secure.groupby(
            ["task_l6_label", "source_dataset", "risk_group_private"],
            dropna=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    write_csv(
        secure_dir / "SECURE_SELECTION_SUMMARY.csv",
        selection_summary,
    )

    base_rows = []
    for reader_id in ("reader_1", "reader_2"):
        for pilot_id in image_manifest["pilot_id"]:
            base_rows.append(
                {
                    "pilot_id": pilot_id,
                    "reader_id": reader_id,
                    "annotation_status": "pending",
                }
            )
    annotations = pd.DataFrame(base_rows).reindex(columns=annotation_columns())
    write_csv(
        blinded_dir / "BLINDED_ANNOTATION_TEMPLATE.csv",
        annotations,
    )
    reader_paths: list[Path] = []
    for reader_id in ("reader_1", "reader_2"):
        path = blinded_dir / f"{reader_id.upper()}_MORPHOLOGY.csv"
        write_csv(
            path,
            annotations.loc[annotations["reader_id"].eq(reader_id)].copy(),
        )
        reader_paths.append(path)

    instructions_path = blinded_dir / "ANNOTATION_SCHEMA.md"
    instructions_path.write_text(build_instructions(), encoding="utf-8")
    gate_text = """# H17 Mini-Pilot Gate

The packet remains morphology-only and risk-blinded.

Advance to the 60-case pilot only when:

1. At least four non-artifact concepts are judgeable as present/absent in at least 70% of images.
2. Those concepts reach Gwet AC1 or Cohen kappa >= 0.60.
3. Z1, Z2, and Z3 rough-region agreement is >= 0.60 when the region is visible.
4. `not_visible_or_uncertain` is used consistently.
5. M9 artifact is not the only consistently visible signal.

Failure stops historical-text supervision. It does not trigger risk-label review.
"""
    (blinded_dir / "PILOT_GATE.md").write_text(gate_text, encoding="utf-8")

    blinded_metadata = [
        blinded_dir / "BLINDED_IMAGE_MANIFEST.csv",
        blinded_dir / "BLINDED_ANNOTATION_TEMPLATE.csv",
        *reader_paths,
        instructions_path,
        blinded_dir / "PILOT_GATE.md",
    ]
    if args.copy_images:
        write_hash_manifest(
            blinded_dir / "BLINDED_IMAGE_SHA256.txt",
            sorted(image_dir.iterdir()),
            blinded_dir,
        )
    write_hash_manifest(
        blinded_dir / "BLINDED_PACKET_METADATA_SHA256.txt",
        blinded_metadata,
        blinded_dir,
    )
    write_hash_manifest(
        secure_dir / "SECURE_FILES_SHA256.txt",
        [
            secure_dir / "SECURE_CASE_KEY_DO_NOT_SHARE.csv",
            secure_dir / "SECURE_SELECTION_SUMMARY.csv",
        ],
        secure_dir,
    )
    write_json(
        output_dir / "packet_config.json",
        {
            "experiment": "H17_MORPHOLOGY_VISIBILITY_MINIPILOT_20260721",
            "registry_csv": str(registry_path),
            "registry_sha256": sha256(registry_path),
            "seed": args.seed,
            "cases": 20,
            "readers": 2,
            "copy_images": args.copy_images,
            "search_trials": args.search_trials,
            "selection_score": {
                "fold_penalty": score[0],
                "empty_source_fold_cells": score[1],
            },
            "quotas": {
                f"{subtype}|{source}": count
                for (subtype, source), count in QUOTAS.items()
            },
            "selection_uses_model_output": False,
            "selection_uses_model_correctness": False,
            "selection_uses_model_confidence": False,
            "annotator_blinding": [
                "risk_group",
                "histologic_subtype",
                "source",
                "model_output",
                "model_correctness",
            ],
            "risk_judgment_fields_present": False,
            "strict_external_read": False,
        },
    )
    (output_dir / "RUN.status").write_text(
        "COMPLETE_PENDING_MORPHOLOGY_ANNOTATION\n",
        encoding="utf-8",
    )
    try:
        secure_dir.chmod(0o700)
        for path in secure_dir.iterdir():
            path.chmod(0o600)
    except OSError:
        pass
    print(selection_summary.to_string(index=False), flush=True)
    print(
        json.dumps(
            {
                "status": "COMPLETE_PENDING_MORPHOLOGY_ANNOTATION",
                "cases": 20,
                "risk_balance": secure["risk_group_private"].value_counts().to_dict(),
                "source_balance": secure["source_dataset"].value_counts().to_dict(),
                "fold_balance": secure["master_fold_id"].value_counts().sort_index().to_dict(),
                "selection_score": score[:2],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
