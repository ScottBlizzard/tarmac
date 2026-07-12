from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_REGISTRY = Path(
    "/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/"
    "task7_four_domain_master_registry.csv"
)
DEFAULT_C1_OOF = Path(
    "/workspace/thymic_project/experiments/base_model_capability_20260711/"
    "phase2_siglipl512_local_pyramid_screen/"
    "347_siglipl512_localpyramid6_gated_fivefold_cw_20260711/oof_predictions.csv"
)

SOURCE_THIRD = "third_batch_306_20260521"
QUOTAS = {
    "A": {"batch1": 10, "batch2": 10},
    "AB": {"batch1": 7, "batch2": 7, SOURCE_THIRD: 6},
    "B1": {"batch1": 7, "batch2": 7, SOURCE_THIRD: 6},
    "B2": {"batch1": 7, "batch2": 7, SOURCE_THIRD: 6},
    "B3": {"batch1": 16, "batch2": 4},
    "TC": {"batch1": 7, "batch2": 7, SOURCE_THIRD: 6},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a label/model-blinded 120-case physician ROI oracle packet.")
    parser.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--c1-oof-csv", default=str(DEFAULT_C1_OOF))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hash_manifest(path: Path, files: list[Path], relative_to: Path) -> None:
    lines = [f"{sha256_file(item)}  {item.relative_to(relative_to).as_posix()}" for item in files]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def spread_select(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0:
        return frame.iloc[0:0].copy()
    if len(frame) <= count:
        return frame.copy()
    ordered = frame.sort_values(["c1_abs_logit", "case_id"]).reset_index(drop=True)
    raw_positions = np.linspace(0, len(ordered) - 1, count)
    positions = []
    for value in raw_positions:
        candidate = int(round(value))
        if candidate not in positions:
            positions.append(candidate)
    if len(positions) < count:
        positions.extend(index for index in range(len(ordered)) if index not in positions)
    return ordered.iloc[positions[:count]].copy()


def select_correct_error_mix(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    wrong = frame[frame["c1_correct"] == 0]
    correct = frame[frame["c1_correct"] == 1]
    wrong_target = min(len(wrong), count // 2)
    correct_target = min(len(correct), count - wrong_target)
    remaining = count - wrong_target - correct_target
    if remaining:
        extra_wrong = min(len(wrong) - wrong_target, remaining)
        wrong_target += extra_wrong
        remaining -= extra_wrong
    if remaining:
        correct_target += min(len(correct) - correct_target, remaining)
    selected = pd.concat(
        [spread_select(wrong, wrong_target), spread_select(correct, correct_target)],
        ignore_index=True,
    )
    if len(selected) != count:
        raise ValueError(f"Could not select requested correct/error mixture: requested={count} selected={len(selected)}")
    return selected


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    registry = pd.read_csv(
        args.registry_csv, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig"
    )
    registry = registry[registry["domain"].isin(["old_data", "third_batch"])].copy()
    registry = registry.sort_values(["source_dataset", "task_l6_label", "case_id"]).drop_duplicates("case_id")
    c1 = pd.read_csv(args.c1_oof_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    c1 = c1[["case_id", "prob_high"]].drop_duplicates("case_id").rename(columns={"prob_high": "c1_prob_high"})
    merged = registry.merge(c1, on="case_id", how="left", validate="one_to_one")
    if merged["c1_prob_high"].isna().any():
        missing = merged.loc[merged["c1_prob_high"].isna(), "case_id"].head(10).tolist()
        raise ValueError(f"C1 OOF predictions are missing cases: {missing}")
    probability = np.clip(merged["c1_prob_high"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    merged["c1_logit"] = np.log(probability / (1.0 - probability))
    merged["c1_abs_logit"] = np.abs(merged["c1_logit"])
    merged["c1_pred"] = (probability >= 0.5).astype(int)
    merged["c1_correct"] = (merged["c1_pred"] == merged["label_idx"].astype(int)).astype(int)
    return merged


def select_cases(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for subtype, source_quotas in QUOTAS.items():
        for source, count in source_quotas.items():
            stratum = candidates[
                (candidates["task_l6_label"] == subtype)
                & (candidates["source_dataset"] == source)
            ]
            if len(stratum) < count:
                raise ValueError(
                    f"Insufficient cases for subtype={subtype} source={source}: available={len(stratum)} needed={count}"
                )
            chosen = select_correct_error_mix(stratum, count)
            chosen["quota_subtype"] = subtype
            chosen["quota_source"] = source
            selected.append(chosen)
    result = pd.concat(selected, ignore_index=True)
    if len(result) != 120 or result["case_id"].duplicated().any():
        raise ValueError("ROI oracle selection must contain 120 unique cases")
    return result


def annotation_columns() -> list[str]:
    return [
        "oracle_id",
        "reader_id",
        "annotation_status",
        "image_quality",
        "image_sufficient_for_low_high_judgment",
        "physician_risk_judgment",
        "physician_confidence_1_to_5",
        "specimen_extent_x1_norm",
        "specimen_extent_y1_norm",
        "specimen_extent_x2_norm",
        "specimen_extent_y2_norm",
        "cut_surface_x1_norm",
        "cut_surface_y1_norm",
        "cut_surface_x2_norm",
        "cut_surface_y2_norm",
        "probable_tumor_x1_norm",
        "probable_tumor_y1_norm",
        "probable_tumor_x2_norm",
        "probable_tumor_y2_norm",
        "capsule_interface_x1_norm",
        "capsule_interface_y1_norm",
        "capsule_interface_x2_norm",
        "capsule_interface_y2_norm",
        "hemorrhage_necrosis_artifact_x1_norm",
        "hemorrhage_necrosis_artifact_y1_norm",
        "hemorrhage_necrosis_artifact_x2_norm",
        "hemorrhage_necrosis_artifact_y2_norm",
        "roi1_x1_norm",
        "roi1_y1_norm",
        "roi1_x2_norm",
        "roi1_y2_norm",
        "roi1_reason",
        "roi2_x1_norm",
        "roi2_y1_norm",
        "roi2_x2_norm",
        "roi2_y2_norm",
        "roi2_reason",
        "roi3_x1_norm",
        "roi3_y1_norm",
        "roi3_x2_norm",
        "roi3_y2_norm",
        "roi3_reason",
        "no_visually_diagnostic_roi",
        "recommended_additional_view",
        "free_text_comment",
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty blinded packet: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    blinded_dir = output_dir / "blinded_packet"
    secure_dir = output_dir / "secure_do_not_share"
    blinded_dir.mkdir(parents=True, exist_ok=True)
    secure_dir.mkdir(parents=True, exist_ok=True)
    image_dir = blinded_dir / "blinded_images"
    if args.copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args)
    selected = select_cases(candidates)
    rng = np.random.default_rng(args.seed)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected["oracle_id"] = [f"ROI{index:03d}_{secrets.token_hex(3).upper()}" for index in range(1, 121)]

    image_rows = []
    for _, row in selected.iterrows():
        source_path = Path(str(row["image_path"]))
        extension = source_path.suffix.lower() or ".jpg"
        image_filename = f"{row['oracle_id']}{extension}"
        with Image.open(source_path) as image:
            width, height = image.size
        if args.copy_images:
            shutil.copy2(source_path, image_dir / image_filename)
        image_rows.append(
            {
                "oracle_id": row["oracle_id"],
                "image_filename": image_filename,
                "width_px": width,
                "height_px": height,
            }
        )
    image_manifest = pd.DataFrame(image_rows)
    image_manifest.to_csv(blinded_dir / "BLINDED_IMAGE_MANIFEST.csv", index=False, encoding="utf-8-sig")

    secure = selected.merge(image_manifest, on="oracle_id", how="left", validate="one_to_one")
    secure_columns = [
        "oracle_id",
        "image_filename",
        "case_id",
        "original_case_id",
        "domain",
        "source_dataset",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "c1_prob_high",
        "c1_pred",
        "c1_correct",
        "c1_abs_logit",
        "image_path",
        "width_px",
        "height_px",
    ]
    secure_key_path = secure_dir / "SECURE_LABEL_KEY_DO_NOT_SHARE.csv"
    secure[secure_columns].to_csv(secure_key_path, index=False, encoding="utf-8-sig")

    annotation = pd.DataFrame(
        [
            {"oracle_id": oracle_id, "reader_id": reader_id, "annotation_status": "pending"}
            for oracle_id in image_manifest["oracle_id"]
            for reader_id in ["reader_1", "reader_2"]
        ],
        columns=annotation_columns(),
    )
    annotation.to_csv(blinded_dir / "BLINDED_ANNOTATION_TEMPLATE.csv", index=False, encoding="utf-8-sig")

    summary = (
        secure.groupby(["task_l6_label", "source_dataset"], dropna=False)
        .agg(n=("case_id", "size"), c1_wrong=("c1_correct", lambda values: int((values == 0).sum())))
        .reset_index()
    )
    summary.to_csv(secure_dir / "SECURE_SELECTION_SUMMARY.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"cases": len(secure), "independent_readers": 2, "annotation_rows": len(annotation)}]).to_csv(
        blinded_dir / "BLINDED_PACKET_SIZE_SUMMARY.csv", index=False, encoding="utf-8-sig"
    )

    instructions = """# Blinded Physician ROI Oracle Packet

Only share this `blinded_packet` directory with readers. Do not share the sibling `secure_do_not_share` directory.

Each of two physicians independently reviews all 120 neutral image files. They must not receive model outputs, Task7 labels, or final histologic subtypes. Coordinates are normalized to [0, 1] relative to the displayed image: x from left to right and y from top to bottom.

Required annotations:

1. Specimen extent.
2. Cut surface.
3. Probable tumor.
4. Capsule/interface.
5. Hemorrhage, necrosis, or artifact.
6. Up to three regions worth inspecting at higher magnification and a reason for each.
7. Explicit `no_visually_diagnostic_roi=yes` when appropriate.
8. Image sufficiency, blinded low/high-risk judgment, confidence, and the additional view that would be needed.

Readers work independently. Do not reconcile annotations until both files are locked. The secure key is used only after annotation lock to calculate agreement, manual-ROI oracle performance, B1/B2 rescue/harm, and matched-random-ROI controls.
"""
    (blinded_dir / "README_BLINDED_ANNOTATION.md").write_text(instructions, encoding="utf-8")
    write_json(
        output_dir / "packet_config.json",
        {
            "registry_csv": args.registry_csv,
            "c1_oof_csv": args.c1_oof_csv,
            "seed": args.seed,
            "cases": len(selected),
            "readers": 2,
            "copy_images": args.copy_images,
            "quota": QUOTAS,
            "selection_note": "Subtype/source stratified; up to half C1 errors per stratum where available; margins spread deterministically.",
            "blinding": ["Task7 label", "histologic subtype", "C1 output", "C1 correctness"],
        },
    )
    if args.copy_images:
        write_hash_manifest(
            blinded_dir / "BLINDED_IMAGE_SHA256.txt",
            sorted(image_dir.iterdir()),
            blinded_dir,
        )
    write_hash_manifest(
        blinded_dir / "BLINDED_PACKET_METADATA_SHA256.txt",
        [
            blinded_dir / "BLINDED_IMAGE_MANIFEST.csv",
            blinded_dir / "BLINDED_ANNOTATION_TEMPLATE.csv",
            blinded_dir / "BLINDED_PACKET_SIZE_SUMMARY.csv",
            blinded_dir / "README_BLINDED_ANNOTATION.md",
        ],
        blinded_dir,
    )
    write_hash_manifest(
        secure_dir / "SECURE_FILES_SHA256.txt",
        [secure_key_path, secure_dir / "SECURE_SELECTION_SUMMARY.csv"],
        secure_dir,
    )
    try:
        secure_dir.chmod(0o700)
        secure_key_path.chmod(0o600)
    except OSError:
        pass
    print(summary.to_string(index=False), flush=True)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
