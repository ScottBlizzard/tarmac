from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_SCRIPTS = Path("/workspace/thymic_project/scripts")
if not PROJECT_SCRIPTS.is_dir():
    raise FileNotFoundError(f"Missing locked project script directory: {PROJECT_SCRIPTS}")
sys.path.insert(0, str(PROJECT_SCRIPTS))

from extract_task7_h3_dense_bank_20260713 import PeAdapter
from extract_task7_h3_representation_bank_20260713 import DEFAULT_VIEWS, make_view
from run_task7_dense_feature_cv_20260711 import DenseTask7Model
from run_task7_h3b_masked_gated_20260713 import MaskedDenseGatedHead


EXPERIMENT = "H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714"
EXPECTED_VIEWS = tuple(DEFAULT_VIEWS)
SOURCE_ORDER = ("batch1", "batch2", "third_batch")
LOW_RISK = {"A", "AB", "B1"}
MAX_REPRODUCTION_ERROR = 1e-5
H3_SHARD_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fold-specific frozen C1 and H3 case embeddings.")
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--split-mode", choices=("source_lodo", "fivefold"), required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--c1-image-size", type=int, required=True)
    parser.add_argument("--h3-image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
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
    return {"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": sha256_file(resolved)}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = sha256_file(path)
    atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")


def verify_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    return expected == sha256_file(path)


def load_locked_json(path: Path) -> tuple[dict[str, Any], str]:
    if not verify_sidecar(path):
        raise ValueError(f"Missing or invalid manifest sidecar: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("complete") is not True:
        raise ValueError("Asset manifest is incomplete")
    return value, sha256_file(path)


def canonical_source(value: Any) -> str:
    text = str(value).strip()
    if text in {"batch1", "batch2"}:
        return text
    if text.startswith("third_batch"):
        return "third_batch"
    raise ValueError(f"Unexpected source: {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(manifest: dict[str, Any]) -> pd.DataFrame:
    registry_path = Path(manifest["assets"]["registry"]["path"])
    split_path = Path(manifest["assets"]["split"]["path"])
    frame = pd.read_csv(registry_path, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig")
    frame = frame[frame["domain"].astype(str).isin(["old_data", "third_batch"])].copy()
    if "image_exists" in frame.columns:
        keep = frame["image_exists"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[keep].copy()
    frame = frame.sort_values(["domain", "case_id"]).drop_duplicates("case_id").reset_index(drop=True)
    frame["source_dataset"] = frame["source_dataset"].map(canonical_source)
    frame["task_l6_label"] = frame["task_l6_label"].astype(str)
    frame["label_idx"] = (~frame["task_l6_label"].isin(LOW_RISK)).astype(int)
    split = pd.read_csv(split_path, dtype={"case_id": str}, encoding="utf-8-sig")
    split = split[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    authoritative = frame[["case_id"]].merge(split, on="case_id", how="left", validate="one_to_one")
    frame["master_fold_id"] = pd.to_numeric(authoritative["master_fold_id"], errors="raise").astype(int)
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError("Expected 591 unique embedding rows")
    return frame


def load_reference(path: str, metadata: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    required = {"case_id", "fold_id", "prob_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Reference prediction schema changed: {path}")
    frame = frame[["case_id", "fold_id", "prob_high"]].copy()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    aligned = metadata[["case_id"]].merge(frame, on="case_id", how="left", validate="one_to_one")
    if aligned.isna().any().any():
        raise ValueError(f"Reference predictions do not align: {path}")
    return aligned


def shard_name(case_id: str) -> str:
    return hashlib.sha256(f"H8|{case_id}".encode("utf-8")).hexdigest() + ".npz"


def valid_shard(path: Path, fold_count: int, embedding_dim: int) -> bool:
    if not verify_sidecar(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            embedding = values["embedding"]
            probability = values["probability"]
            feature_row = values["feature_row"]
            return bool(
                embedding.shape == (fold_count, embedding_dim)
                and probability.shape == (fold_count,)
                and feature_row.shape == (1,)
                and np.isfinite(embedding).all()
                and np.isfinite(probability).all()
            )
    except Exception:
        return False


def valid_h3_shard(path: Path, fold_count: int) -> bool:
    if not valid_shard(path, fold_count, 128):
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            return bool(
                "extraction_version" in values
                and values["extraction_version"].shape == (1,)
                and int(values["extraction_version"][0]) == H3_SHARD_VERSION
            )
    except Exception:
        return False


def make_c1_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> DenseTask7Model:
    model = DenseTask7Model(
        feature_dim=1024,
        num_views=6,
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        dropout=float(config["dropout"]),
        pooling=str(config["pooling"]),
        expert_mode=str(config["expert_mode"]),
        num_concepts=0,
        num_groups=len(config["group_names"]),
        prototype_temperature=float(config["prototype_temperature"]),
        boundary_fusion_alpha=float(config["boundary_fusion_alpha"]),
        domain_adversarial_lambda=float(config["domain_adversarial_lambda"]),
        risk_from_subtype_alpha=float(config["risk_from_subtype_alpha"]),
        sentinel_fusion_alpha=float(config["sentinel_fusion_alpha"]),
        mixstyle_probability=0.0,
        mixstyle_alpha=float(config["mixstyle_alpha"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def make_h3_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> MaskedDenseGatedHead:
    model = MaskedDenseGatedHead(
        feature_dim=1024,
        num_views=6,
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


@torch.inference_mode()
def c1_embedding_and_probability(model: DenseTask7Model, features: torch.Tensor) -> tuple[np.ndarray, float]:
    embedding, _ = model.pool_features(features)
    probability = torch.softmax(model.risk_head(embedding), dim=-1)[:, 1]
    return embedding[0].float().cpu().numpy(), float(probability[0].float().cpu())


@torch.inference_mode()
def h3_embedding_and_probability(
    model: MaskedDenseGatedHead,
    features: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        batch_size, num_views, token_count, _ = features.shape
        tokens = model.project(model.input_norm(features))
        tokens = tokens + model.view_embeddings[:num_views].view(1, num_views, 1, -1)
        pooled = model.pool(
            tokens.reshape(batch_size, num_views * token_count, -1),
            mask.reshape(batch_size, num_views * token_count),
        )
        embedding = model.output_norm(pooled)
        logits = model.classifier(embedding)
    probability = torch.softmax(logits.float(), dim=-1)[:, 1]
    return embedding.float().cpu().numpy(), probability.cpu().numpy().astype(float)


def h3_split_role(metadata: pd.DataFrame, index: int, split_mode: str, fold_id: int) -> str:
    row = metadata.iloc[index]
    val_fold = (fold_id % 5) + 1
    if split_mode == "source_lodo":
        if str(row["source_dataset"]) == SOURCE_ORDER[fold_id - 1]:
            return "test"
        if int(row["master_fold_id"]) == val_fold:
            return "validation"
        return "train"
    if int(row["master_fold_id"]) == fold_id:
        return "test"
    if int(row["master_fold_id"]) == val_fold:
        return "validation"
    return "train"


def extract_c1(
    manifest: dict[str, Any],
    metadata: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
) -> tuple[float, int]:
    fold_count = int(manifest["fold_count"])
    dense_path = Path(manifest["assets"]["c1_dense_features"]["path"])
    features = np.load(dense_path, mmap_mode="r")
    config = json.loads(Path(manifest["assets"]["c1_run_config"]["path"]).read_text(encoding="utf-8"))
    models = [
        make_c1_model(config, Path(item["path"]), device)
        for item in manifest["assets"]["c1_checkpoints"]
    ]
    shard_dir = output_dir / "case_shards" / "c1"
    started = time.monotonic()
    completed = 0
    for index, row in metadata.iterrows():
        path = shard_dir / shard_name(str(row["case_id"]))
        if valid_shard(path, fold_count, 256):
            completed += 1
            continue
        tensor = torch.from_numpy(np.array(features[index], dtype=np.float32, copy=True)).unsqueeze(0).to(device)
        embeddings = []
        probabilities = []
        for model in models:
            embedding, probability = c1_embedding_and_probability(model, tensor)
            embeddings.append(embedding)
            probabilities.append(probability)
        atomic_savez(
            path,
            feature_row=np.asarray([index], dtype=np.int32),
            embedding=np.stack(embeddings).astype(np.float32),
            probability=np.asarray(probabilities, dtype=np.float32),
        )
        completed += 1
        if completed % 20 == 0 or completed == len(metadata):
            print(f"[C1] {completed}/{len(metadata)}", flush=True)
        del tensor
    elapsed = time.monotonic() - started
    del models, features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return elapsed, completed


def extract_h3(
    manifest: dict[str, Any],
    metadata: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> tuple[float, int]:
    fold_count = int(manifest["fold_count"])
    config = json.loads(Path(manifest["assets"]["h3_run_config"]["path"]).read_text(encoding="utf-8"))
    if int(config.get("batch_size", -1)) != 8:
        raise ValueError("Locked H3 branch batch size changed")
    models = [
        make_h3_model(config, Path(item["path"]), device)
        for item in manifest["assets"]["h3_checkpoints"]
    ]
    adapter_args = SimpleNamespace(
        model_id=manifest["assets"]["pe_checkpoint"]["path"],
        model_code_dir=manifest["assets"]["pe_source_root"],
        code_revision=manifest["pe_source_revision"],
        max_num_patches=1024,
    )
    # This is the exact upstream H3 numerical path. Enabling deterministic
    # algorithms here changes frozen PE values and violates branch reproduction.
    torch.use_deterministic_algorithms(False)
    set_seed(seed)
    adapter = PeAdapter(adapter_args, device)
    shard_dir = output_dir / "case_shards" / "h3"
    started = time.monotonic()
    if all(
        valid_h3_shard(shard_dir / shard_name(str(row["case_id"])), fold_count)
        for _, row in metadata.iterrows()
    ):
        del models, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return time.monotonic() - started, len(metadata)

    embeddings = np.full((fold_count, len(metadata), 128), np.nan, dtype=np.float32)
    probabilities = np.full((fold_count, len(metadata)), np.nan, dtype=np.float32)
    buffers: dict[tuple[int, str], dict[str, list[Any]]] = {
        (fold_id, split_role): {"rows": [], "dense": [], "mask": []}
        for fold_id in range(1, fold_count + 1)
        for split_role in ("train", "validation", "test")
    }

    def flush_buffer(fold_id: int, split_role: str) -> None:
        buffer = buffers[(fold_id, split_role)]
        if not buffer["rows"]:
            return
        rows = np.asarray(buffer["rows"], dtype=int)
        dense_batch = np.concatenate(buffer["dense"], axis=0)
        mask_batch = np.concatenate(buffer["mask"], axis=0)
        dense_tensor = torch.from_numpy(dense_batch).to(device)
        mask_tensor = torch.from_numpy(mask_batch).bool().to(device)
        batch_embedding, batch_probability = h3_embedding_and_probability(
            models[fold_id - 1], dense_tensor, mask_tensor, device
        )
        embeddings[fold_id - 1, rows] = batch_embedding
        probabilities[fold_id - 1, rows] = batch_probability
        buffer["rows"].clear()
        buffer["dense"].clear()
        buffer["mask"].clear()
        del dense_batch, mask_batch, dense_tensor, mask_tensor

    completed = 0
    for index, row in metadata.iterrows():
        dense = np.zeros((1, 6, 1024, 1024), dtype=np.float16)
        mask = np.zeros((1, 6, 1024), dtype=np.uint8)
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        for view_index, view_name in enumerate(EXPECTED_VIEWS):
            extracted = adapter.extract(make_view(image, view_name))
            token_count = int(extracted.tokens.shape[0])
            if extracted.tokens.shape[1] != 1024 or token_count > 1024:
                raise ValueError(f"Unexpected PE token shape for row {index}: {tuple(extracted.tokens.shape)}")
            if int(extracted.valid_mask.sum()) != int(np.prod(extracted.spatial_shape)):
                raise ValueError(f"PE mask/grid mismatch for row {index}, view {view_name}")
            dense[0, view_index, :token_count] = extracted.tokens.numpy().astype(np.float16)
            mask[0, view_index, :token_count] = extracted.valid_mask.numpy().astype(np.uint8)
        for fold_id in range(1, fold_count + 1):
            split_role = h3_split_role(metadata, index, manifest["split_mode"], fold_id)
            buffer = buffers[(fold_id, split_role)]
            buffer["rows"].append(index)
            buffer["dense"].append(dense)
            buffer["mask"].append(mask)
            if len(buffer["rows"]) == 8:
                flush_buffer(fold_id, split_role)
        completed += 1
        if completed % 10 == 0 or completed == len(metadata):
            print(f"[H3 encoder] {completed}/{len(metadata)}", flush=True)
        del dense, mask
    for fold_id in range(1, fold_count + 1):
        for split_role in ("train", "validation", "test"):
            flush_buffer(fold_id, split_role)
    if not np.isfinite(embeddings).all() or not np.isfinite(probabilities).all():
        raise ValueError("H3 partition-batched extraction left nonfinite or missing values")
    for index, row in metadata.iterrows():
        atomic_savez(
            shard_dir / shard_name(str(row["case_id"])),
            feature_row=np.asarray([index], dtype=np.int32),
            embedding=embeddings[:, index],
            probability=probabilities[:, index],
            extraction_version=np.asarray([H3_SHARD_VERSION], dtype=np.int16),
        )
    elapsed = time.monotonic() - started
    del models, adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return elapsed, completed


def consolidate(
    manifest: dict[str, Any],
    manifest_hash: str,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    fold_count = int(manifest["fold_count"])
    c1_embeddings = np.empty((fold_count, len(metadata), 256), dtype=np.float32)
    h3_embeddings = np.empty((fold_count, len(metadata), 128), dtype=np.float32)
    c1_probabilities = np.empty((fold_count, len(metadata)), dtype=np.float32)
    h3_probabilities = np.empty((fold_count, len(metadata)), dtype=np.float32)
    for index, row in metadata.iterrows():
        name = shard_name(str(row["case_id"]))
        c1_path = output_dir / "case_shards" / "c1" / name
        h3_path = output_dir / "case_shards" / "h3" / name
        if not valid_shard(c1_path, fold_count, 256) or not valid_h3_shard(h3_path, fold_count):
            raise ValueError(f"Invalid embedding shard at feature row {index}")
        with np.load(c1_path, allow_pickle=False) as values:
            if int(values["feature_row"][0]) != index:
                raise ValueError("C1 shard row mismatch")
            c1_embeddings[:, index] = values["embedding"]
            c1_probabilities[:, index] = values["probability"]
        with np.load(h3_path, allow_pickle=False) as values:
            if int(values["feature_row"][0]) != index:
                raise ValueError("H3 shard row mismatch")
            h3_embeddings[:, index] = values["embedding"]
            h3_probabilities[:, index] = values["probability"]

    c1_reference = load_reference(manifest["assets"]["c1_predictions"]["path"], metadata)
    h3_reference = load_reference(manifest["assets"]["h3_predictions"]["path"], metadata)
    c1_fold = c1_reference["fold_id"].to_numpy(int) - 1
    h3_fold = h3_reference["fold_id"].to_numpy(int) - 1
    row_index = np.arange(len(metadata))
    c1_error = np.abs(c1_probabilities[c1_fold, row_index] - c1_reference["prob_high"].to_numpy(float))
    h3_error = np.abs(h3_probabilities[h3_fold, row_index] - h3_reference["prob_high"].to_numpy(float))
    max_c1_error = float(c1_error.max())
    max_h3_error = float(h3_error.max())
    if max_c1_error > MAX_REPRODUCTION_ERROR or max_h3_error > MAX_REPRODUCTION_ERROR:
        raise ValueError(
            f"Branch reproduction failed: C1={max_c1_error:.9g}, H3={max_h3_error:.9g}"
        )

    metadata_path = output_dir / "embedding_metadata.csv"
    metadata_columns = [
        "case_id", "original_case_id", "source_dataset", "task_l6_label",
        "label_idx", "master_fold_id", "image_path",
    ]
    metadata[metadata_columns].to_csv(metadata_path, index=False, encoding="utf-8-sig")
    integrity_path = output_dir / "branch_reproduction_integrity.csv"
    pd.DataFrame(
        {
            "case_id": metadata["case_id"],
            "c1_fold_id": c1_fold + 1,
            "c1_abs_error": c1_error,
            "h3_fold_id": h3_fold + 1,
            "h3_abs_error": h3_error,
        }
    ).to_csv(integrity_path, index=False, encoding="utf-8-sig")
    embedding_path = output_dir / "fold_embeddings.npz"
    atomic_savez(
        embedding_path,
        c1=c1_embeddings,
        h3=h3_embeddings,
        c1_probability=c1_probabilities,
        h3_probability=h3_probabilities,
    )
    output_bytes = sum(item.stat().st_size for item in output_dir.rglob("*") if item.is_file())
    if output_bytes >= 1024**3:
        raise ValueError(f"H8 embedding outputs exceed the 1 GiB ceiling: {output_bytes}")
    return {
        "experiment": EXPERIMENT,
        "complete": True,
        "split_mode": manifest["split_mode"],
        "fold_count": fold_count,
        "case_count": len(metadata),
        "asset_manifest_sha256": manifest_hash,
        "embedding_shapes": {"c1": list(c1_embeddings.shape), "h3": list(h3_embeddings.shape)},
        "maximum_branch_probability_abs_error": {"c1": max_c1_error, "h3": max_h3_error},
        "reproduction_tolerance": MAX_REPRODUCTION_ERROR,
        "persistent_dense_tokens_written": False,
        "new_output_bytes": int(output_bytes),
        "assets": {
            "fold_embeddings": record(embedding_path),
            "embedding_metadata": record(metadata_path),
            "branch_reproduction_integrity": record(integrity_path),
        },
        "privacy": "This manifest points to server-only case-level artifacts.",
    }


def run(args: argparse.Namespace) -> None:
    if tuple(item.strip() for item in args.views.split(",")) != EXPECTED_VIEWS:
        raise ValueError("H8 view order is immutable")
    if args.c1_image_size != 512 or args.h3_image_size != 448:
        raise ValueError("H8 image-size declarations changed")
    if args.batch_size != 1 or args.num_workers != 0:
        raise ValueError("H8 extraction is locked to batch size 1 and zero workers")
    if args.seed not in {20260714, 20260715}:
        raise ValueError("Unregistered H8 seed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    asset_manifest, manifest_hash = load_locked_json(Path(args.asset_manifest))
    if asset_manifest["experiment"] != EXPERIMENT or asset_manifest["split_mode"] != args.split_mode:
        raise ValueError("Asset manifest does not match this extraction stage")
    metadata = load_metadata(asset_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    c1_seconds, c1_completed = extract_c1(asset_manifest, metadata, output_dir, device)
    h3_seconds, h3_completed = extract_h3(asset_manifest, metadata, output_dir, device, args.seed)
    embedding_manifest = consolidate(asset_manifest, manifest_hash, metadata, output_dir)
    embedding_manifest.update(
        {
            "seed": args.seed,
            "elapsed_seconds": float(time.monotonic() - started),
            "c1_elapsed_seconds": float(c1_seconds),
            "h3_elapsed_seconds": float(h3_seconds),
            "c1_completed_cases": c1_completed,
            "h3_completed_cases": h3_completed,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "peak_resident_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    )
    manifest_path = output_dir / "embedding_manifest.json"
    atomic_write_text(manifest_path, json.dumps(embedding_manifest, ensure_ascii=False, indent=2))
    atomic_write_text(
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n",
    )
    atomic_write_text(output_dir / "RUN.status", "complete\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "max_error": embedding_manifest["maximum_branch_probability_abs_error"],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "RUN.status", "running\n")
    try:
        run(args)
    except Exception as error:
        atomic_write_text(output_dir / "RUN.status", f"failed: {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
