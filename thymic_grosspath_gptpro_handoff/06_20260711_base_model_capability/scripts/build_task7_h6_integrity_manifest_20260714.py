from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "H6_NUISANCE_ANCHORED_CSD_20260714"
EXPECTED_VIEWS = ["whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3"]
EXPECTED_HASHES = {
    "dense_features": "e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34",
    "valid_token_mask": "af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c",
    "spatial_shapes": "14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f",
}
EXPECTED_PE_SOURCE_TREE_SHA256 = (
    "4e00f80f27fc360591f94eb125ed8ea18b86237ae59e5284abf3bde2ddebdbea"
)
EXPECTED_REFERENCE_METRICS = {
    "h3_oof": (0.8003265743809709, 0.8071748878923767, 0.7934782608695652),
    "h3_lodo": (0.7538506531487619, 0.6816143497757847, 0.8260869565217391),
    "c2_oof": (0.7514074380970950, 0.7174887892376681, 0.7853260869565217),
    "c2_lodo": (0.7440717001364788, 0.7354260089686099, 0.7527173913043478),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock all immutable H6 assets.")
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--frequency-features", required=True)
    parser.add_argument("--frequency-metadata", required=True)
    parser.add_argument("--h3-oof", required=True)
    parser.add_argument("--h3-lodo", required=True)
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--analyzer", required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--model-code-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def canonical_source(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"^third_batch.*", "third_batch", regex=True)


def strict_bool(values: pd.Series, name: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"Unsupported boolean values in {name}: {unknown}")
    return normalized.map(mapping).astype(bool)


def validate_bank_metadata(registry: pd.DataFrame, bank_metadata: pd.DataFrame) -> None:
    if len(bank_metadata) != 591 or bank_metadata["case_id"].duplicated().any():
        raise ValueError("PE bank metadata is not 591 unique cases")
    if set(bank_metadata["case_id"].astype(str)) != set(
        registry["case_id"].astype(str)
    ):
        raise ValueError("PE bank and authoritative registry case sets differ")
    if "feature_row" not in bank_metadata or not np.array_equal(
        pd.to_numeric(bank_metadata["feature_row"]).to_numpy(dtype=int),
        np.arange(len(bank_metadata), dtype=int),
    ):
        raise ValueError("PE bank feature_row does not match array row order")

    aligned_registry = registry.set_index("case_id").loc[
        bank_metadata["case_id"].astype(str)
    ].reset_index()
    common_columns = sorted(
        (set(registry.columns) & set(bank_metadata.columns)) - {"case_id"}
    )
    numeric_columns = {
        "label_idx",
        "master_fold_id",
        "image_count",
        "original_image_count",
    }
    for column in common_columns:
        expected = aligned_registry[column]
        observed = bank_metadata[column]
        if column == "is_frozen_external":
            equal = strict_bool(expected, column).equals(strict_bool(observed, column))
        elif column in numeric_columns:
            equal = np.array_equal(
                pd.to_numeric(expected).to_numpy(),
                pd.to_numeric(observed).to_numpy(),
                equal_nan=True,
            )
        else:
            equal = expected.fillna("").astype(str).reset_index(drop=True).equals(
                observed.fillna("").astype(str).reset_index(drop=True)
            )
        if not equal:
            raise ValueError(f"PE bank metadata differs from registry: {column}")


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["label_idx"].to_numpy(dtype=int)
    probability = frame["prob_high"].to_numpy(dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "n": int(len(frame)),
        "accuracy": float(np.mean(labels == predicted)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def load_prediction(path: Path, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {"case_id", "label_idx", "source_dataset", "task_l6_label", "prob_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Reference prediction columns are incomplete: {name}")
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError(f"Reference prediction is not 591 unique cases: {name}")
    if not np.isfinite(frame["prob_high"].to_numpy(dtype=float)).all():
        raise ValueError(f"Reference prediction contains non-finite probabilities: {name}")
    frame["source_dataset"] = canonical_source(frame["source_dataset"])
    result = metrics(frame)
    expected = EXPECTED_REFERENCE_METRICS[name]
    observed = (
        result["balanced_accuracy"],
        result["sensitivity"],
        result["specificity"],
    )
    if not np.allclose(observed, expected, atol=1e-12, rtol=0.0):
        raise ValueError(f"Locked reference metrics changed for {name}: {observed}")
    return frame, result


def validate_case_alignment(frames: dict[str, pd.DataFrame]) -> None:
    canonical = frames["h3_oof"].sort_values("case_id").reset_index(drop=True)
    columns = ["case_id", "label_idx", "source_dataset", "task_l6_label"]
    for name, frame in frames.items():
        aligned = frame.sort_values("case_id").reset_index(drop=True)
        if not canonical[columns].equals(aligned[columns]):
            raise ValueError(f"Reference case metadata differs: {name}")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


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


def resolve_source_revision(path: Path, expected_revision: str) -> tuple[str, str]:
    if (path / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = result.stdout.strip()
        mode = "git_head"
    else:
        if not path.name.endswith(expected_revision[:8]):
            raise ValueError("PE source snapshot directory does not encode revision")
        revision = expected_revision
        mode = "revision_named_snapshot"
    if revision != expected_revision:
        raise ValueError("Local PE source checkout revision changed")
    return revision, mode


def main() -> None:
    args = parse_args()
    bank = Path(args.feature_bank_dir).resolve(strict=True)
    bank_paths = {
        "dense_features": bank / "dense_features.float16.npy",
        "valid_token_mask": bank / "valid_token_mask.uint8.npy",
        "spatial_shapes": bank / "spatial_shapes.int16.npy",
        "processed": bank / "processed.uint8.npy",
        "bank_metadata": bank / "metadata.csv",
        "bank_config": bank / "dense_bank_config.json",
    }
    config = json.loads(bank_paths["bank_config"].read_text(encoding="utf-8"))
    if config.get("complete") is not True or config.get("completed_cases") != 591:
        raise ValueError("PE dense bank is not complete")
    if config.get("dense_shape") != [591, 6, 1024, 1024]:
        raise ValueError("PE dense bank shape differs from H3")
    if config.get("views") != EXPECTED_VIEWS:
        raise ValueError("PE dense bank view order differs from H3")
    if config.get("weight_sha256") != (
        "47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1"
    ):
        raise ValueError("PE checkpoint hash differs from H3")
    if config.get("adapter", {}).get("code_revision") != (
        "3e352cca660658d4b5c90f42a7808b11469e4c66"
    ):
        raise ValueError("PE code revision differs from H3")

    assets: dict[str, Any] = {}
    for key, path in bank_paths.items():
        print(f"hashing {key}: {path}", flush=True)
        assets[key] = record(path)
    for key, expected_hash in EXPECTED_HASHES.items():
        if assets[key]["sha256"] != expected_hash:
            raise ValueError(f"Regenerated PE bank hash mismatch: {key}")

    registry = pd.read_csv(
        args.registry_csv,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    registry.columns = [str(column).lstrip("\ufeff") for column in registry.columns]
    if len(registry) != 591 or registry["case_id"].duplicated().any():
        raise ValueError("H6 must use the authoritative 591-case internal registry")
    if set(registry["domain"].astype(str)) != {"old_data", "third_batch"}:
        raise ValueError("Registry contains a non-development domain")
    if "is_frozen_external" in registry and strict_bool(
        registry["is_frozen_external"], "is_frozen_external"
    ).any():
        raise ValueError("Registry unexpectedly contains frozen external cases")
    if "image_exists" in registry and not strict_bool(
        registry["image_exists"], "image_exists"
    ).all():
        raise ValueError("Registry contains inaccessible image paths")

    bank_metadata = pd.read_csv(
        bank_paths["bank_metadata"],
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    validate_bank_metadata(registry, bank_metadata)

    frequency = np.load(args.frequency_features, mmap_mode="r")
    frequency_metadata = pd.read_csv(
        args.frequency_metadata, dtype={"case_id": str}, encoding="utf-8-sig"
    )
    if frequency.shape != (591, 156) or frequency.dtype != np.float32:
        raise ValueError("Fixed Haar matrix shape/dtype changed")
    if len(frequency_metadata) != 591 or frequency_metadata["case_id"].duplicated().any():
        raise ValueError("Fixed Haar metadata is not 591 unique cases")
    if set(frequency_metadata["case_id"]) != set(registry["case_id"].astype(str)):
        raise ValueError("Fixed Haar and authoritative registry case sets differ")

    prediction_paths = {
        "h3_oof": Path(args.h3_oof),
        "h3_lodo": Path(args.h3_lodo),
        "c2_oof": Path(args.c2_oof),
        "c2_lodo": Path(args.c2_lodo),
    }
    frames: dict[str, pd.DataFrame] = {}
    reference_metrics: dict[str, Any] = {}
    for name, path in prediction_paths.items():
        frames[name], reference_metrics[name] = load_prediction(path, name)
    validate_case_alignment(frames)

    additional_paths = {
        "registry_csv": Path(args.registry_csv),
        "split_csv": Path(args.split_csv),
        "frequency_features": Path(args.frequency_features),
        "frequency_metadata": Path(args.frequency_metadata),
        "h3_oof_predictions": prediction_paths["h3_oof"],
        "h3_lodo_predictions": prediction_paths["h3_lodo"],
        "c2_oof_predictions": prediction_paths["c2_oof"],
        "c2_lodo_predictions": prediction_paths["c2_lodo"],
        "trainer": Path(args.trainer),
        "analyzer": Path(args.analyzer),
        "extractor": Path(args.extractor),
        "model_checkpoint": Path(args.model_checkpoint),
    }
    for key, path in additional_paths.items():
        print(f"hashing {key}: {path}", flush=True)
        assets[key] = record(path)
    if assets["model_checkpoint"]["sha256"] != config["weight_sha256"]:
        raise ValueError("Model checkpoint no longer matches dense bank configuration")
    model_code_dir = Path(args.model_code_dir).resolve(strict=True)
    revision, revision_mode = resolve_source_revision(
        model_code_dir, config["adapter"]["code_revision"]
    )
    source_tree_hash, source_file_count = source_tree_sha256(model_code_dir)
    if source_tree_hash != EXPECTED_PE_SOURCE_TREE_SHA256:
        raise ValueError("Local PE source snapshot content changed")

    split = pd.read_csv(args.split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    if len(split) != 591 or split["case_id"].duplicated().any():
        raise ValueError("Locked split is not 591 unique cases")
    if set(split["case_id"]) != set(registry["case_id"].astype(str)):
        raise ValueError("Locked split and registry case sets differ")
    if set(pd.to_numeric(split["master_fold_id"]).astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Locked split does not contain five master folds")

    case_order_hash = hashlib.sha256(
        "\n".join(bank_metadata["case_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    manifest = {
        "experiment": EXPERIMENT,
        "complete": True,
        "case_count": 591,
        "case_id_order_sha256": case_order_hash,
        "privacy": "Images, case-level features, predictions, and weights remain server-only.",
        "authoritative_registry_reason": (
            "Exact 591-case registry recorded by the H3 dense-bank configuration; "
            "the 861-row four-domain registry is prohibited because it includes 270 consumed stress-test cases."
        ),
        "pe_source_revision": revision,
        "pe_source_revision_mode": revision_mode,
        "pe_source_tree_sha256": source_tree_hash,
        "pe_source_file_count": source_file_count,
        "reference_metrics_at_threshold_0_5": reference_metrics,
        "assets": assets,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "integrity.json"
    atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    digest = sha256_file(manifest_path)
    atomic_write(output_dir / "integrity.sha256", f"{digest}  integrity.json\n")
    print(json.dumps({"status": "complete", "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
