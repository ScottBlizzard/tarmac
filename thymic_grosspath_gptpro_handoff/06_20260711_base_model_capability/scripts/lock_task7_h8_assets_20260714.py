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


EXPERIMENT = "H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714"
EXPECTED_VIEWS = ["whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3"]
EXPECTED_SOURCES = {"batch1": 117, "batch2": 168, "third_batch": 306}
EXPECTED_SUBTYPES = {"A": 44, "AB": 262, "B1": 62, "B2": 89, "B3": 24, "TC": 110}
LOW_RISK = {"A", "AB", "B1"}
EXPECTED_REGISTRY_SHA256 = "ef2d4e16b041038e36bd165c80d460b527c0c09eba19250b38f50d67199c62af"
EXPECTED_SPLIT_SHA256 = "48610c996298f8af317d547681665e0be20e4361aa61de66675efc51bb954545"
EXPECTED_C1_DENSE_SHA256 = "ff10fbf255cb0da08be19566b321947479c321ac1c7e5dee2e5158c8cc68c3ee"
EXPECTED_C1_CONFIG_SHA256 = "6788fd540a6e2f54bd3d5f204528804625f1f1868524042e7a6c0dbf3d9edd55"
EXPECTED_C1_METADATA_SHA256 = "8bf1a1746ddd6b2b3000160761093283ac5f10f2c33b5dc42fde0f456118f176"
EXPECTED_PE_SOURCE_TREE_SHA256 = "4e00f80f27fc360591f94eb125ed8ea18b86237ae59e5284abf3bde2ddebdbea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock all immutable H8 assets offline.")
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--split-mode", choices=("source_lodo", "fivefold"), required=True)
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--c1-feature-bank", required=True)
    parser.add_argument("--c1-root", required=True)
    parser.add_argument("--c1-predictions", required=True)
    parser.add_argument("--c2-predictions", required=True)
    parser.add_argument("--h3-root", required=True)
    parser.add_argument("--h3-predictions", required=True)
    parser.add_argument("--pe-checkpoint", required=True)
    parser.add_argument("--expected-pe-sha256", required=True)
    parser.add_argument("--pe-source-root", required=True)
    parser.add_argument("--expected-pe-source-revision", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def asset_record(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {resolved}: {digest} != {expected_sha256}")
    return {"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": digest}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def canonical_source(value: Any) -> str:
    text = str(value).strip()
    if text == "batch1":
        return "batch1"
    if text == "batch2":
        return "batch2"
    if text.startswith("third_batch"):
        return "third_batch"
    raise ValueError(f"Unexpected source label: {value!r}")


def source_tree_sha256(path: Path) -> tuple[str, int]:
    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
        and ".git" not in item.parts
    ]
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def load_registry(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    frame = frame[frame["domain"].astype(str).isin(["old_data", "third_batch"])].copy()
    if "image_exists" in frame.columns:
        exists = frame["image_exists"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[exists].copy()
    frame = frame.sort_values(["domain", "case_id"]).drop_duplicates("case_id").reset_index(drop=True)
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError("Registry must contain exactly 591 unique internal cases")
    frame["source_dataset"] = frame["source_dataset"].map(canonical_source)
    frame["task_l6_label"] = frame["task_l6_label"].astype(str)
    frame["label_idx"] = (~frame["task_l6_label"].isin(LOW_RISK)).astype(int)
    if frame["source_dataset"].value_counts().sort_index().to_dict() != EXPECTED_SOURCES:
        raise ValueError("Source totals differ from the locked cohort")
    if frame["task_l6_label"].value_counts().to_dict() != EXPECTED_SUBTYPES:
        raise ValueError("Subtype totals differ from the locked cohort")
    if frame["label_idx"].value_counts().sort_index().to_dict() != {0: 368, 1: 223}:
        raise ValueError("Risk totals differ from 368 low and 223 high")
    missing = [value for value in frame["image_path"].astype(str) if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing locked image paths: {missing[:10]}")
    return frame


def validate_split(path: Path, registry: pd.DataFrame) -> pd.DataFrame:
    split = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    split.columns = [str(column).lstrip("\ufeff") for column in split.columns]
    split = split[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    joined = registry[["case_id"]].merge(split, on="case_id", how="left", validate="one_to_one")
    if joined["master_fold_id"].isna().any():
        raise ValueError("The locked split is missing internal cases")
    joined["master_fold_id"] = pd.to_numeric(joined["master_fold_id"], errors="raise").astype(int)
    if set(joined["master_fold_id"]) != {1, 2, 3, 4, 5}:
        raise ValueError("The locked split must contain master folds 1-5")
    return joined


def validate_prediction_file(path: Path, registry: pd.DataFrame, fold_count: int) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {"case_id", "label_idx", "prob_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prediction columns in {path}: {sorted(required - set(frame.columns))}")
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError(f"Prediction file is not a complete 591-case table: {path}")
    aligned = registry[["case_id", "label_idx", "task_l6_label", "source_dataset"]].merge(
        frame, on="case_id", how="left", validate="one_to_one", suffixes=("_locked", "_pred")
    )
    if aligned["prob_high"].isna().any() or not np.isfinite(aligned["prob_high"].to_numpy(float)).all():
        raise ValueError(f"Missing or nonfinite probabilities in {path}")
    if not np.array_equal(
        aligned["label_idx_locked"].to_numpy(int), aligned["label_idx_pred"].to_numpy(int)
    ):
        raise ValueError(f"Risk-label mismatch in {path}")
    if "fold_id" in frame.columns:
        folds = set(pd.to_numeric(frame["fold_id"], errors="raise").astype(int))
        if folds != set(range(1, fold_count + 1)):
            raise ValueError(f"Unexpected fold IDs in {path}: {sorted(folds)}")
    labels = aligned["label_idx_locked"].to_numpy(int)
    predicted = (aligned["prob_high"].to_numpy(float) >= 0.5).astype(int)
    low = labels == 0
    high = labels == 1
    return {
        "balanced_accuracy": float((np.mean(predicted[low] == 0) + np.mean(predicted[high] == 1)) / 2),
        "sensitivity": float(np.mean(predicted[high] == 1)),
        "specificity": float(np.mean(predicted[low] == 0)),
    }


def validate_run_root(root: Path, split_mode: str) -> dict[str, Any]:
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("split_mode") != split_mode:
        raise ValueError(f"Run split mode mismatch at {root}")
    return config


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.registry_csv)
    split_path = Path(args.split_csv)
    c1_bank = Path(args.c1_feature_bank)
    c1_root = Path(args.c1_root).resolve(strict=True)
    h3_root = Path(args.h3_root).resolve(strict=True)
    fold_count = 3 if args.split_mode == "source_lodo" else 5

    print("[lock] hashing registry and split", flush=True)
    registry_record = asset_record(registry_path, EXPECTED_REGISTRY_SHA256)
    split_record = asset_record(split_path, EXPECTED_SPLIT_SHA256)
    registry = load_registry(registry_path)
    split = validate_split(split_path, registry)

    print("[lock] validating exact C1 dense bank", flush=True)
    c1_config_path = c1_bank / "feature_bank_config.json"
    c1_metadata_path = c1_bank / "metadata.csv"
    c1_dense_path = c1_bank / "dense_features.float16.npy"
    c1_config_record = asset_record(c1_config_path, EXPECTED_C1_CONFIG_SHA256)
    c1_metadata_record = asset_record(c1_metadata_path, EXPECTED_C1_METADATA_SHA256)
    c1_dense_record = asset_record(c1_dense_path, EXPECTED_C1_DENSE_SHA256)
    c1_config = json.loads(c1_config_path.read_text(encoding="utf-8"))
    if c1_config.get("complete") is not True or c1_config.get("views") != EXPECTED_VIEWS:
        raise ValueError("C1 feature-bank completeness or view order changed")
    c1_features = np.load(c1_dense_path, mmap_mode="r")
    if c1_features.shape != (591, 6, 1024, 1024) or c1_features.dtype != np.float16:
        raise ValueError(f"Unexpected C1 dense tensor: {c1_features.shape}, {c1_features.dtype}")
    c1_metadata = pd.read_csv(c1_metadata_path, dtype={"case_id": str}, encoding="utf-8-sig")
    if c1_metadata["case_id"].astype(str).tolist() != registry["case_id"].astype(str).tolist():
        raise ValueError("C1 dense-bank case order differs from the authoritative registry")
    del c1_features

    c1_config_run = validate_run_root(c1_root, args.split_mode)
    h3_config_run = validate_run_root(h3_root, args.split_mode)
    if c1_config_run.get("feature_bank_config", {}).get("views") != EXPECTED_VIEWS:
        raise ValueError("C1 run view order changed")
    if h3_config_run.get("feature_shape") != [591, 6, 1024, 1024]:
        raise ValueError("H3 run feature shape changed")

    print("[lock] hashing fold checkpoints and prediction references", flush=True)
    c1_checkpoints = []
    h3_checkpoints = []
    for fold_id in range(1, fold_count + 1):
        c1_checkpoints.append(asset_record(c1_root / f"fold_{fold_id}" / "best_model.pt"))
        h3_checkpoints.append(asset_record(h3_root / f"fold_{fold_id}" / "best_head.pt"))
    prediction_paths = {
        "c1": Path(args.c1_predictions),
        "c2": Path(args.c2_predictions),
        "h3": Path(args.h3_predictions),
    }
    prediction_records = {key: asset_record(path) for key, path in prediction_paths.items()}
    reference_metrics = {
        key: validate_prediction_file(path, registry, fold_count)
        for key, path in prediction_paths.items()
    }

    print("[lock] hashing PE checkpoint and source snapshot", flush=True)
    pe_record = asset_record(Path(args.pe_checkpoint), args.expected_pe_sha256)
    pe_source = Path(args.pe_source_root).resolve(strict=True)
    if not (pe_source / "core" / "vision_encoder" / "pe.py").is_file():
        raise FileNotFoundError("Official PE source snapshot is incomplete")
    tree_hash, tree_files = source_tree_sha256(pe_source)
    if tree_hash != EXPECTED_PE_SOURCE_TREE_SHA256:
        raise ValueError(f"PE source tree changed: {tree_hash}")

    case_order_hash = hashlib.sha256(
        "\n".join(registry["case_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    disk = shutil.disk_usage(output_path.parent)
    manifest = {
        "experiment": EXPERIMENT,
        "complete": True,
        "evidence_commit": args.evidence_commit,
        "split_mode": args.split_mode,
        "fold_count": fold_count,
        "case_count": 591,
        "case_id_order_sha256": case_order_hash,
        "source_counts": EXPECTED_SOURCES,
        "subtype_counts": EXPECTED_SUBTYPES,
        "risk_counts": {"0": 368, "1": 223},
        "master_fold_counts": {
            str(key): int(value)
            for key, value in split["master_fold_id"].value_counts().sort_index().items()
        },
        "all_image_paths_accessible": True,
        "c1_operational_amendment": (
            "Consume and byte-lock the extant exact C1 dense bank; the assumed timm cache is absent."
        ),
        "h3_extraction_mode": (
            "Stream PE tokens with the original H3 bfloat16 autocast path; do not persist dense tokens."
        ),
        "pe_source_revision": args.expected_pe_source_revision,
        "pe_source_revision_mode": "declared revision of revision-named source snapshot",
        "pe_source_tree_sha256": tree_hash,
        "pe_source_file_count": tree_files,
        "reference_metrics_at_threshold_0_5": reference_metrics,
        "disk_free_bytes_at_lock": int(disk.free),
        "privacy": "Images, paths, IDs, embeddings, predictions, and weights remain server-only.",
        "assets": {
            "registry": registry_record,
            "split": split_record,
            "c1_dense_features": c1_dense_record,
            "c1_feature_config": c1_config_record,
            "c1_metadata": c1_metadata_record,
            "c1_run_config": asset_record(c1_root / "run_config.json"),
            "h3_run_config": asset_record(h3_root / "run_config.json"),
            "c1_predictions": prediction_records["c1"],
            "c2_predictions": prediction_records["c2"],
            "h3_predictions": prediction_records["h3"],
            "c1_checkpoints": c1_checkpoints,
            "h3_checkpoints": h3_checkpoints,
            "pe_checkpoint": pe_record,
            "pe_source_root": str(pe_source),
        },
    }
    atomic_write_text(
        output_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
    )
    digest = sha256_file(output_path)
    atomic_write_text(output_path.with_suffix(output_path.suffix + ".sha256"), f"{digest}  {output_path.name}\n")
    print(json.dumps({"status": "complete", "manifest": str(output_path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
