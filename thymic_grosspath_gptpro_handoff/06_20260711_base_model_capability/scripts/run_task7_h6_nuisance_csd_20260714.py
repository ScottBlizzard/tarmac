from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from run_task7_h3_summary_gated_20260713 import prediction_frame, sha256_file
from run_task7_h3b_masked_gated_20260713 import MaskedGatedPool, validate_bank
from run_task7_spatial_relational_20260713 import (
    EXPECTED_SOURCES,
    fold_partitions,
    load_metadata,
    metric_record,
    set_seed,
    source_risk_sampler,
    summarize_predictions,
    write_json,
)


EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
EXPECTED_DENSE_SHA256 = "e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34"
EXPECTED_MASK_SHA256 = "af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c"
EXPECTED_SHAPES_SHA256 = "14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f"
EXPECTED_FEATURE_SHAPE = (591, 6, 1024, 1024)

HIDDEN_DIM = 128
ATTENTION_DIM = 64
DROPOUT = 0.10
EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
COMMON_LOSS_WEIGHT = 0.5
SPECIFIC_LOSS_WEIGHT = 0.5
ORTHOGONALITY_WEIGHT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preregistered H6 nuisance-anchored rank-1 CSD."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--frequency-features", required=True)
    parser.add_argument("--frequency-metadata", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--integrity-manifest", required=True)
    parser.add_argument("--variant", choices=("nuisance_csd", "source_csd"), required=True)
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Engineering smoke-test override; never use for a preregistered result.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all outer-fold environment constructions without fitting heads.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def canonical_source(value: object) -> str:
    text = str(value)
    return "third_batch" if text.startswith("third_batch") else text


def set_deterministic(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest_hash = sha256_file(path)
    sidecar = path.with_name("integrity.sha256")
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing integrity sidecar: {sidecar}")
    expected_hash = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if manifest_hash != expected_hash:
        raise ValueError("Integrity manifest SHA-256 differs from its locked sidecar")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("case_count") != 591:
        raise ValueError("H6 integrity manifest is incomplete")
    if manifest.get("experiment") != "H6_NUISANCE_ANCHORED_CSD_20260714":
        raise ValueError("Unexpected experiment identity in integrity manifest")
    return manifest, manifest_hash


def validate_manifest_assets(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    manifest_hash: str,
) -> None:
    del manifest_hash
    assets = manifest["assets"]
    expected_paths = {
        "dense_features": Path(args.feature_bank_dir) / "dense_features.float16.npy",
        "valid_token_mask": Path(args.feature_bank_dir) / "valid_token_mask.uint8.npy",
        "spatial_shapes": Path(args.feature_bank_dir) / "spatial_shapes.int16.npy",
        "bank_metadata": Path(args.feature_bank_dir) / "metadata.csv",
        "bank_config": Path(args.feature_bank_dir) / "dense_bank_config.json",
        "frequency_features": Path(args.frequency_features),
        "frequency_metadata": Path(args.frequency_metadata),
        "split_csv": Path(args.split_csv),
        "trainer": Path(__file__),
    }
    expected_hashes = {
        "dense_features": EXPECTED_DENSE_SHA256,
        "valid_token_mask": EXPECTED_MASK_SHA256,
        "spatial_shapes": EXPECTED_SHAPES_SHA256,
    }
    for key, expected_path in expected_paths.items():
        record = assets[key]
        actual_path = expected_path.resolve(strict=True)
        if Path(record["path"]).resolve(strict=True) != actual_path:
            raise ValueError(f"Manifest path differs for {key}")
        if actual_path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Manifest byte size differs for {key}")
        if key in expected_hashes and record["sha256"] != expected_hashes[key]:
            raise ValueError(f"Locked bank hash differs for {key}")
        if key not in expected_hashes and sha256_file(actual_path) != record["sha256"]:
            raise ValueError(f"Immutable asset changed: {key}")


def parse_risk(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        result = numeric.astype(int).to_numpy()
    else:
        mapping = {
            "low": 0,
            "low_risk": 0,
            "0": 0,
            "high": 1,
            "high_risk": 1,
            "1": 1,
        }
        normalized = values.astype(str).str.strip().str.lower().map(mapping)
        if normalized.isna().any():
            raise ValueError("Frequency metadata contains an unrecognized risk value")
        result = normalized.astype(int).to_numpy()
    if not set(np.unique(result)).issubset({0, 1}):
        raise ValueError("Frequency metadata risk must be binary")
    return result


def load_frequency_matrix(
    feature_path: Path,
    metadata_path: Path,
    metadata: pd.DataFrame,
) -> np.ndarray:
    features = np.load(feature_path, mmap_mode="r")
    frequency_metadata = pd.read_csv(
        metadata_path, dtype={"case_id": str}, encoding="utf-8-sig"
    )
    frequency_metadata.columns = [
        str(column).lstrip("\ufeff") for column in frequency_metadata.columns
    ]
    required = {"case_id", "source", "risk"}
    if not required.issubset(frequency_metadata.columns):
        raise ValueError("Frequency metadata is missing case_id/source/risk")
    if features.shape != (591, 156) or features.dtype != np.float32:
        raise ValueError(f"Unexpected fixed Haar feature shape/dtype: {features.shape}")
    if len(frequency_metadata) != 591 or frequency_metadata["case_id"].duplicated().any():
        raise ValueError("Expected 591 unique frequency metadata rows")
    row_lookup = pd.Series(
        np.arange(len(frequency_metadata), dtype=int),
        index=frequency_metadata["case_id"].astype(str),
    )
    if set(row_lookup.index) != set(metadata["case_id"].astype(str)):
        raise ValueError("Frequency and PE metadata case sets differ")
    order = row_lookup.loc[metadata["case_id"].astype(str)].to_numpy(dtype=int)
    aligned = np.asarray(features[order], dtype=np.float64)
    if not np.isfinite(aligned).all():
        raise ValueError("Fixed Haar features contain non-finite values")
    source = frequency_metadata.iloc[order]["source"].map(canonical_source).to_numpy(str)
    risk = parse_risk(frequency_metadata.iloc[order]["risk"])
    if not np.array_equal(source, metadata["source_dataset"].astype(str).to_numpy()):
        raise ValueError("Frequency source labels do not align with PE metadata")
    if not np.array_equal(risk, metadata["label_idx"].to_numpy(dtype=int)):
        raise ValueError("Frequency risk labels do not align with PE metadata")
    return aligned


@dataclass(frozen=True)
class EnvironmentAssignment:
    train_ids: np.ndarray
    validation_ids: np.ndarray
    names: tuple[str, ...]
    gamma_initialization: np.ndarray
    diagnostics: dict[str, Any]


def source_design(
    sources: np.ndarray,
    labels: np.ndarray,
    present_sources: tuple[str, ...],
) -> np.ndarray:
    columns = [np.ones(len(sources), dtype=np.float64), labels.astype(np.float64)]
    columns.extend((sources == source).astype(np.float64) for source in present_sources[1:])
    return np.column_stack(columns)


def _environment_counts(
    sources: np.ndarray,
    labels: np.ndarray,
    bins: np.ndarray,
    present_sources: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in present_sources:
        for nuisance_bin in (0, 1):
            selected = (sources == source) & (bins == nuisance_bin)
            low = int(np.sum(selected & (labels == 0)))
            high = int(np.sum(selected & (labels == 1)))
            rows.append(
                {
                    "environment": f"{source}|bin{nuisance_bin}",
                    "source": source,
                    "nuisance_bin": nuisance_bin,
                    "low_risk_n": low,
                    "high_risk_n": high,
                    "n": low + high,
                }
            )
    return rows


def build_source_environments(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> EnvironmentAssignment:
    train_sources = metadata.iloc[train_indices]["source_dataset"].astype(str).to_numpy()
    validation_sources = (
        metadata.iloc[validation_indices]["source_dataset"].astype(str).to_numpy()
    )
    present_sources = tuple(source for source in EXPECTED_SOURCES if source in train_sources)
    mapping = {source: index for index, source in enumerate(present_sources)}
    if any(source not in mapping for source in validation_sources):
        raise ValueError("Validation contains a source absent from outer training")
    train_ids = np.asarray([mapping[source] for source in train_sources], dtype=np.int64)
    validation_ids = np.asarray(
        [mapping[source] for source in validation_sources], dtype=np.int64
    )
    gamma = np.linspace(-1.0, 1.0, num=len(present_sources), dtype=np.float32)
    labels = metadata.iloc[train_indices]["label_idx"].to_numpy(dtype=int)
    source_rows = []
    for source in present_sources:
        selected = train_sources == source
        source_rows.append(
            {
                "environment": source,
                "low_risk_n": int(np.sum(selected & (labels == 0))),
                "high_risk_n": int(np.sum(selected & (labels == 1))),
                "n": int(np.sum(selected)),
            }
        )
    return EnvironmentAssignment(
        train_ids=train_ids,
        validation_ids=validation_ids,
        names=present_sources,
        gamma_initialization=gamma,
        diagnostics={
            "construction": "acquisition_source_only",
            "environment_counts": source_rows,
            "gamma_initialization_rule": "canonical-source-order linspace(-1,+1)",
        },
    )


def build_nuisance_environments(
    metadata: pd.DataFrame,
    frequency: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> EnvironmentAssignment:
    train = metadata.iloc[train_indices]
    validation = metadata.iloc[validation_indices]
    train_sources = train["source_dataset"].astype(str).to_numpy()
    train_labels = train["label_idx"].to_numpy(dtype=int)
    present_sources = tuple(source for source in EXPECTED_SOURCES if source in train_sources)
    design = source_design(train_sources, train_labels, present_sources)
    coefficients = np.linalg.pinv(design) @ frequency[train_indices]
    residual = frequency[train_indices] - design @ coefficients
    scale = residual.std(axis=0, ddof=0)
    standardized = np.divide(
        residual,
        scale,
        out=np.zeros_like(residual),
        where=scale >= 1e-8,
    )
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(standardized)
    orientation = -1.0 if float(pca.components_[0].sum()) < 0.0 else 1.0
    pca.components_[0] *= orientation
    train_scores = pca.transform(standardized).reshape(-1)

    validation_sources = validation["source_dataset"].astype(str).to_numpy()
    validation_labels = validation["label_idx"].to_numpy(dtype=int)
    if any(source not in present_sources for source in validation_sources):
        raise ValueError("Validation contains a source absent from nuisance fitting")
    validation_design = source_design(
        validation_sources, validation_labels, present_sources
    )
    validation_residual = frequency[validation_indices] - validation_design @ coefficients
    validation_standardized = np.divide(
        validation_residual,
        scale,
        out=np.zeros_like(validation_residual),
        where=scale >= 1e-8,
    )
    validation_scores = pca.transform(validation_standardized).reshape(-1)

    case_ids = train["case_id"].astype(str).to_numpy()
    validation_case_ids = validation["case_id"].astype(str).to_numpy()
    train_bins = np.full(len(train_indices), -1, dtype=np.int64)
    validation_bins = np.full(len(validation_indices), -1, dtype=np.int64)
    boundaries: dict[tuple[str, int], tuple[float, str]] = {}
    for source in present_sources:
        for risk in (0, 1):
            stratum = np.flatnonzero((train_sources == source) & (train_labels == risk))
            if len(stratum) < 8:
                raise ValueError(f"Too few cases in training stratum {source}/{risk}")
            order = np.lexsort((case_ids[stratum], train_scores[stratum]))
            ordered = stratum[order]
            midpoint = len(ordered) // 2
            train_bins[ordered[:midpoint]] = 0
            train_bins[ordered[midpoint:]] = 1
            first_upper = int(ordered[midpoint])
            boundaries[(source, risk)] = (
                float(train_scores[first_upper]),
                str(case_ids[first_upper]),
            )
    if np.any(train_bins < 0):
        raise RuntimeError("Nuisance bin construction left unassigned training cases")

    # This label-aware assignment is diagnostic only. Common-head validation drives
    # checkpoint selection, and test/deployment never receives an environment ID.
    for index, (source, risk, score, case_id) in enumerate(
        zip(
            validation_sources,
            validation_labels,
            validation_scores,
            validation_case_ids,
            strict=True,
        )
    ):
        boundary = boundaries[(str(source), int(risk))]
        validation_bins[index] = int((float(score), str(case_id)) >= boundary)

    names = tuple(
        f"{source}|bin{nuisance_bin}"
        for source in present_sources
        for nuisance_bin in (0, 1)
    )
    mapping = {name: index for index, name in enumerate(names)}
    train_ids = np.asarray(
        [
            mapping[f"{source}|bin{nuisance_bin}"]
            for source, nuisance_bin in zip(train_sources, train_bins, strict=True)
        ],
        dtype=np.int64,
    )
    validation_ids = np.asarray(
        [
            mapping[f"{source}|bin{nuisance_bin}"]
            for source, nuisance_bin in zip(
                validation_sources, validation_bins, strict=True
            )
        ],
        dtype=np.int64,
    )
    counts = _environment_counts(
        train_sources, train_labels, train_bins, present_sources
    )
    if any(row["low_risk_n"] < 4 or row["high_risk_n"] < 4 for row in counts):
        raise ValueError("A nuisance environment failed the preregistered 4+4 preflight")
    component_bytes = np.asarray(pca.components_[0], dtype=np.float64).tobytes()
    boundary_payload = json.dumps(
        {
            f"{source}|risk{risk}": [score, case_id]
            for (source, risk), (score, case_id) in sorted(boundaries.items())
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return EnvironmentAssignment(
        train_ids=train_ids,
        validation_ids=validation_ids,
        names=names,
        gamma_initialization=np.asarray(
            [-1.0 if name.endswith("bin0") else 1.0 for name in names],
            dtype=np.float32,
        ),
        diagnostics={
            "construction": "training_only_frequency_residual_pca_binary_bins",
            "present_sources": list(present_sources),
            "design_columns": ["intercept", "risk"]
            + [f"source_is_{source}" for source in present_sources[1:]],
            "design_rank": int(np.linalg.matrix_rank(design)),
            "zero_scale_feature_count": int(np.sum(scale < 1e-8)),
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "pca_loading_sum": float(pca.components_[0].sum()),
            "pca_loading_sha256": hashlib.sha256(component_bytes).hexdigest(),
            "bin_boundary_sha256": hashlib.sha256(boundary_payload).hexdigest(),
            "environment_counts": counts,
            "validation_specific_logits_are_label_aware_diagnostic_only": True,
        },
    )


class CSDDenseDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        metadata: pd.DataFrame,
        indices: np.ndarray,
        environment_ids: np.ndarray | None,
    ) -> None:
        self.features = features
        self.masks = masks
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=int)
        if environment_ids is None:
            self.environment_ids = np.full(len(self.indices), -1, dtype=np.int64)
        else:
            self.environment_ids = np.asarray(environment_ids, dtype=np.int64)
        if len(self.environment_ids) != len(self.indices):
            raise ValueError("Environment IDs do not align with dataset indices")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "feature": torch.from_numpy(
                np.array(self.features[index], dtype=np.float16, copy=True)
            ),
            "valid_mask": torch.from_numpy(
                np.array(self.masks[index], dtype=np.uint8, copy=True)
            ).bool(),
            "label": torch.tensor(
                int(self.metadata.iloc[index]["label_idx"]), dtype=torch.long
            ),
            "environment": torch.tensor(
                int(self.environment_ids[item]), dtype=torch.long
            ),
            "index": torch.tensor(index, dtype=torch.long),
        }


def make_loader(
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    environment_ids: np.ndarray | None,
    sampler: WeightedRandomSampler | None,
) -> DataLoader:
    return DataLoader(
        CSDDenseDataset(features, masks, metadata, indices, environment_ids),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


class RankOneCSDHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        num_environments: int,
        gamma_initialization: np.ndarray,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, HIDDEN_DIM))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.pool = MaskedGatedPool(HIDDEN_DIM, ATTENTION_DIM)
        self.output_norm = nn.LayerNorm(HIDDEN_DIM)
        # H3 applies this dropout immediately before its final linear layer. H6
        # retains it and replaces only that linear layer with the CSD readout.
        self.classifier_dropout = nn.Dropout(DROPOUT)
        self.common_weight = nn.Parameter(torch.empty(2, HIDDEN_DIM))
        self.common_bias = nn.Parameter(torch.zeros(2))
        self.specific_weight = nn.Parameter(torch.empty(2, HIDDEN_DIM))
        self.specific_bias = nn.Parameter(torch.zeros(2))
        self.gamma = nn.Parameter(torch.as_tensor(gamma_initialization).clone().float())
        if self.gamma.shape != (num_environments,):
            raise ValueError("Gamma initialization has the wrong number of environments")
        nn.init.xavier_uniform_(self.common_weight)
        nn.init.xavier_uniform_(self.specific_weight)

    def embed(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, token_count, _ = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        tokens = tokens.reshape(batch_size, num_views * token_count, -1)
        mask = valid_mask.reshape(batch_size, num_views * token_count)
        return self.output_norm(self.pool(tokens, mask))

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        environment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = self.classifier_dropout(self.embed(features, valid_mask)).float()
        with torch.autocast(device_type=features.device.type, enabled=False):
            common_logits = F.linear(
                embedding, self.common_weight.float(), self.common_bias.float()
            )
            if environment_ids is None:
                return common_logits, None
            if torch.any(environment_ids < 0):
                raise ValueError("Environment-specific logits require valid environment IDs")
            coefficient = self.gamma[environment_ids].float()
            weights = self.common_weight.float().unsqueeze(0) + coefficient.view(
                -1, 1, 1
            ) * self.specific_weight.float().unsqueeze(0)
            biases = self.common_bias.float().unsqueeze(0) + coefficient.view(
                -1, 1
            ) * self.specific_bias.float().unsqueeze(0)
            specific_logits = torch.einsum("bch,bh->bc", weights, embedding) + biases
        return common_logits, specific_logits

    def orthogonality_loss(self) -> torch.Tensor:
        basis = torch.stack(
            (self.common_weight.float(), self.specific_weight.float()), dim=-1
        )
        gram = torch.einsum("cdi,cdj->cij", basis, basis)
        identity = torch.eye(2, device=gram.device, dtype=gram.dtype).expand_as(gram)
        return torch.mean((gram - identity) ** 2)

    @torch.no_grad()
    def mechanism_record(self, environment_names: Iterable[str]) -> dict[str, Any]:
        common_norm = float(torch.linalg.vector_norm(self.common_weight.float()).cpu())
        specific_norm = float(torch.linalg.vector_norm(self.specific_weight.float()).cpu())
        gamma = self.gamma.detach().float().cpu().numpy()
        ratios = np.abs(gamma) * specific_norm / max(common_norm, 1e-12)
        return {
            "common_weight_frobenius_norm": common_norm,
            "specific_weight_frobenius_norm": specific_norm,
            "gamma": {
                name: float(value)
                for name, value in zip(environment_names, gamma, strict=True)
            },
            "specific_to_common_ratio": {
                name: float(value)
                for name, value in zip(environment_names, ratios, strict=True)
            },
            "median_specific_to_common_ratio": float(np.median(ratios)),
        }


@torch.no_grad()
def predict(
    model: RankOneCSDHead,
    loader: DataLoader,
    device: torch.device,
    include_specific: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    common_probabilities: list[np.ndarray] = []
    specific_probabilities: list[np.ndarray] = []
    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True)
        if device.type != "cuda":
            feature = feature.float()
        mask = batch["valid_mask"].to(device, non_blocking=True)
        environment = (
            batch["environment"].to(device, non_blocking=True)
            if include_specific
            else None
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            common_logits, specific_logits = model(feature, mask, environment)
        common_probability = torch.softmax(common_logits.float(), dim=-1)[:, 1]
        common_probabilities.append(common_probability.cpu().numpy())
        if include_specific:
            if specific_logits is None:
                raise RuntimeError("Specific logits were unexpectedly absent")
            specific_probability = torch.softmax(specific_logits.float(), dim=-1)[:, 1]
            specific_probabilities.append(specific_probability.cpu().numpy())
        indices.append(batch["index"].numpy())
        labels.append(batch["label"].numpy())
    specific = (
        np.concatenate(specific_probabilities).astype(float)
        if include_specific
        else None
    )
    return (
        np.concatenate(indices).astype(int),
        np.concatenate(labels).astype(int),
        np.concatenate(common_probabilities).astype(float),
        specific,
    )


@dataclass
class TrainedModel:
    model: RankOneCSDHead
    best_epoch: int
    best_validation_bacc: float
    best_validation_sensitivity: float


def optimizer_for(model: nn.Module) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or name == "gamma":
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )


def train_model(
    model: RankOneCSDHead,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    fold_dir: Path,
    max_epochs: int | None,
    device: torch.device,
) -> TrainedModel:
    model.to(device)
    optimizer = optimizer_for(model)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_epochs = EPOCHS if max_epochs is None else min(EPOCHS, max_epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_bacc = -math.inf
    best_sensitivity = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_losses: list[float] = []
        common_losses: list[float] = []
        specific_losses: list[float] = []
        orthogonality_losses: list[float] = []
        for batch in train_loader:
            feature = batch["feature"].to(device, non_blocking=True)
            if device.type != "cuda":
                feature = feature.float()
            mask = batch["valid_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            environment = batch["environment"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                common_logits, specific_logits = model(feature, mask, environment)
            if specific_logits is None:
                raise RuntimeError("Training requires environment-specific logits")
            common_loss = F.cross_entropy(common_logits.float(), labels)
            specific_loss = F.cross_entropy(specific_logits.float(), labels)
            orthogonality_loss = model.orthogonality_loss()
            loss = (
                COMMON_LOSS_WEIGHT * common_loss
                + SPECIFIC_LOSS_WEIGHT * specific_loss
                + ORTHOGONALITY_WEIGHT * orthogonality_loss
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            total_losses.append(float(loss.detach().cpu()))
            common_losses.append(float(common_loss.detach().cpu()))
            specific_losses.append(float(specific_loss.detach().cpu()))
            orthogonality_losses.append(float(orthogonality_loss.detach().cpu()))

        _, validation_labels, common_probability, specific_probability = predict(
            model, validation_loader, device, include_specific=True
        )
        if specific_probability is None:
            raise RuntimeError("Validation-specific predictions are required")
        common_metrics = metric_record(validation_labels, common_probability)
        specific_metrics = metric_record(validation_labels, specific_probability)
        validation_bacc = float(common_metrics["balanced_accuracy"])
        validation_sensitivity = float(common_metrics["sensitivity"])
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(total_losses)),
                "common_loss": float(np.mean(common_losses)),
                "specific_loss": float(np.mean(specific_losses)),
                "orthogonality_loss": float(np.mean(orthogonality_losses)),
                "common_validation_bacc": validation_bacc,
                "common_validation_sensitivity": validation_sensitivity,
                "specific_validation_bacc": float(
                    specific_metrics["balanced_accuracy"]
                ),
            }
        )
        print(
            f"epoch={epoch} loss={np.mean(total_losses):.5f} "
            f"common_val_bacc={validation_bacc:.4f} "
            f"common_val_sens={validation_sensitivity:.4f} "
            f"specific_val_bacc={specific_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        better = validation_bacc > best_bacc + 1e-8
        tied_better_sensitivity = (
            abs(validation_bacc - best_bacc) <= 1e-8
            and validation_sensitivity > best_sensitivity + 1e-8
        )
        if better or tied_better_sensitivity:
            best_bacc = validation_bacc
            best_sensitivity = validation_sensitivity
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    fold_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(fold_dir / "training_history.csv", pd.DataFrame(history))
    checkpoint_path = fold_dir / "best_head.pt"
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_bacc": best_bacc,
            "best_validation_sensitivity": best_sensitivity,
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint_path)
    return TrainedModel(
        model=model,
        best_epoch=best_epoch,
        best_validation_bacc=best_bacc,
        best_validation_sensitivity=best_sensitivity,
    )


def completed_fold(
    fold_dir: Path,
    run_config_sha256: str,
    integrity_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    artifact_path = fold_dir / "fold_artifacts.json"
    if not artifact_path.exists():
        return None
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("run_config_sha256") != run_config_sha256:
        raise ValueError(f"Completed fold configuration changed: {fold_dir}")
    if artifact.get("integrity_sha256") != integrity_sha256:
        raise ValueError(f"Completed fold integrity lock changed: {fold_dir}")
    for filename, expected_hash in artifact["files"].items():
        path = fold_dir / filename
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"Completed fold artifact hash mismatch: {path}")
    summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(
        fold_dir / "test_predictions.csv", dtype={"case_id": str}, encoding="utf-8-sig"
    )
    return summary, predictions


def build_environments(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    frequency: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> EnvironmentAssignment:
    if args.variant == "source_csd":
        return build_source_environments(metadata, train_indices, validation_indices)
    return build_nuisance_environments(
        metadata, frequency, train_indices, validation_indices
    )


def run_fold(
    fold_id: int,
    features: np.ndarray,
    masks: np.ndarray,
    metadata: pd.DataFrame,
    frequency: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    run_config_sha256: str,
    integrity_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    recovered = completed_fold(fold_dir, run_config_sha256, integrity_sha256)
    if recovered is not None:
        print(f"fold={fold_id} status=already_complete", flush=True)
        return recovered
    train_indices, validation_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    fold_seed = args.seed + 1000 * fold_id
    set_deterministic(fold_seed)
    assignment = build_environments(
        args, metadata, frequency, train_indices, validation_indices
    )
    device = torch.device(args.device)
    model = RankOneCSDHead(
        feature_dim=int(features.shape[-1]),
        num_views=int(features.shape[1]),
        num_environments=len(assignment.names),
        gamma_initialization=assignment.gamma_initialization,
    )
    sampler = source_risk_sampler(metadata, train_indices, fold_seed + 1)
    train_loader = make_loader(
        features,
        masks,
        metadata,
        train_indices,
        assignment.train_ids,
        sampler,
    )
    validation_loader = make_loader(
        features,
        masks,
        metadata,
        validation_indices,
        assignment.validation_ids,
        None,
    )
    test_loader = make_loader(
        features, masks, metadata, test_indices, None, None
    )
    trained = train_model(
        model, train_loader, validation_loader, fold_dir, args.max_epochs, device
    )
    validation_index, validation_label, validation_probability, validation_specific = predict(
        trained.model, validation_loader, device, include_specific=True
    )
    test_index, test_label, test_probability, _ = predict(
        trained.model, test_loader, device, include_specific=False
    )
    if validation_specific is None:
        raise RuntimeError("Missing diagnostic validation-specific predictions")
    candidate = f"h6_{args.variant}_{args.split_mode}_seed{args.seed}"
    validation_frame = prediction_frame(
        metadata,
        validation_index,
        validation_label,
        validation_probability,
        fold_id,
        held_source,
        candidate,
        "validation",
    )
    validation_frame["prob_high_specific_diagnostic"] = validation_specific
    validation_frame["pred_idx_specific_diagnostic"] = (
        validation_specific >= 0.5
    ).astype(int)
    test_frame = prediction_frame(
        metadata,
        test_index,
        test_label,
        test_probability,
        fold_id,
        held_source,
        candidate,
        "test",
    )
    common_validation_metrics = metric_record(
        validation_label, validation_probability
    )
    specific_validation_metrics = metric_record(
        validation_label, validation_specific
    )
    mechanism = trained.model.mechanism_record(assignment.names)
    mechanism["specific_minus_common_validation_bacc"] = float(
        specific_validation_metrics["balanced_accuracy"]
        - common_validation_metrics["balanced_accuracy"]
    )
    summary = {
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "variant": args.variant,
        "seed": args.seed,
        "fold_seed": fold_seed,
        "train_n": int(len(train_indices)),
        "validation_n": int(len(validation_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": trained.best_epoch,
        "best_common_validation_bacc": trained.best_validation_bacc,
        "best_common_validation_sensitivity": trained.best_validation_sensitivity,
        "common_validation_metrics": common_validation_metrics,
        "specific_validation_metrics_diagnostic": specific_validation_metrics,
        "test_metrics": metric_record(test_label, test_probability),
        "parameter_count": int(
            sum(parameter.numel() for parameter in trained.model.parameters())
        ),
        "environment_names": list(assignment.names),
        "environment_construction": assignment.diagnostics,
        "mechanism": mechanism,
    }
    atomic_write_csv(fold_dir / "validation_predictions.csv", validation_frame)
    atomic_write_csv(fold_dir / "test_predictions.csv", test_frame)
    atomic_write_json(fold_dir / "fold_summary.json", summary)
    tracked_files = (
        "best_head.pt",
        "training_history.csv",
        "validation_predictions.csv",
        "test_predictions.csv",
        "fold_summary.json",
    )
    atomic_write_json(
        fold_dir / "fold_artifacts.json",
        {
            "run_config_sha256": run_config_sha256,
            "integrity_sha256": integrity_sha256,
            "files": {
                filename: sha256_file(fold_dir / filename) for filename in tracked_files
            },
        },
    )
    del trained, model, train_loader, validation_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, test_frame


def preflight_all_folds(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    frequency: np.ndarray,
    output_dir: Path,
) -> None:
    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    rows = []
    for fold_id in folds:
        train_indices, validation_indices, test_indices, held_source = fold_partitions(
            metadata, args.split_mode, fold_id
        )
        assignment = build_environments(
            args, metadata, frequency, train_indices, validation_indices
        )
        rows.append(
            {
                "fold_id": fold_id,
                "held_out_source": held_source,
                "train_n": int(len(train_indices)),
                "validation_n": int(len(validation_indices)),
                "test_n": int(len(test_indices)),
                "environment_names": list(assignment.names),
                "diagnostics": assignment.diagnostics,
            }
        )
    atomic_write_json(output_dir / "environment_preflight.json", rows)
    (output_dir / "RUN.status").write_text("preflight_complete\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.seed not in (20260714, 20260715):
        raise ValueError("Only the preregistered primary and confirmation seeds are allowed")
    if args.max_epochs is not None and not args.preflight_only:
        print("WARNING: max-epochs creates a smoke test, not a preregistered result", flush=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_deterministic(args.seed)
    feature_bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_hash = read_manifest(Path(args.integrity_manifest))
    validate_manifest_assets(manifest, args, manifest_hash)
    config_path = feature_bank_dir / "dense_bank_config.json"
    bank_config = json.loads(config_path.read_text(encoding="utf-8"))
    if bank_config.get("complete") is not True:
        raise ValueError("Dense feature bank is incomplete")
    if tuple(bank_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("Dense feature bank view order differs from the H3 lock")
    metadata = load_metadata(feature_bank_dir, Path(args.split_csv))
    frequency = load_frequency_matrix(
        Path(args.frequency_features), Path(args.frequency_metadata), metadata
    )
    if args.preflight_only:
        preflight_all_folds(args, metadata, frequency, output_dir)
        return
    features = np.load(
        feature_bank_dir / "dense_features.float16.npy", mmap_mode="r"
    )
    masks = np.load(
        feature_bank_dir / "valid_token_mask.uint8.npy", mmap_mode="r"
    )
    shapes = np.load(feature_bank_dir / "spatial_shapes.int16.npy", mmap_mode="r")
    if features.shape != EXPECTED_FEATURE_SHAPE or features.dtype != np.float16:
        raise ValueError(f"Unexpected dense feature bank: {features.shape}/{features.dtype}")
    if masks.shape != features.shape[:-1] or masks.dtype != np.uint8:
        raise ValueError("Dense token mask shape or dtype differs from the H3 lock")
    validate_bank(features, masks, shapes)
    run_config = {
        "experiment": "H6_NUISANCE_ANCHORED_CSD_20260714",
        "variant": args.variant,
        "split_mode": args.split_mode,
        "fold": args.fold,
        "seed": args.seed,
        "device": args.device,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "feature_bank_dir": str(feature_bank_dir.resolve()),
        "frequency_features": str(Path(args.frequency_features).resolve()),
        "frequency_metadata": str(Path(args.frequency_metadata).resolve()),
        "split_csv": str(Path(args.split_csv).resolve()),
        "integrity_manifest": str(Path(args.integrity_manifest).resolve()),
        "integrity_sha256": manifest_hash,
        "feature_shape": list(features.shape),
        "views": list(EXPECTED_VIEWS),
        "hidden_dim": HIDDEN_DIM,
        "attention_dim": ATTENTION_DIM,
        "dropout": DROPOUT,
        "classifier_dropout_retained_from_h3": True,
        "rank": 1,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "zero_weight_decay": ["biases", "gamma"],
        "grad_clip": GRAD_CLIP,
        "common_loss_weight": COMMON_LOSS_WEIGHT,
        "specific_loss_weight": SPECIFIC_LOSS_WEIGHT,
        "orthogonality_weight": ORTHOGONALITY_WEIGHT,
        "sampler": "source_x_risk_inverse_frequency_with_replacement",
        "checkpoint_selection": "common_validation_bacc_then_sensitivity_then_earlier_epoch",
        "threshold": 0.5,
        "coverage": 1.0,
        "max_epochs_smoke_override": args.max_epochs,
    }
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise ValueError("Existing H6 output has a different immutable configuration")
    else:
        atomic_write_json(run_config_path, run_config)
    run_config_hash = sha256_file(run_config_path)
    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    if args.fold != "all":
        requested = int(args.fold)
        if requested not in folds:
            raise ValueError(f"Invalid fold {requested} for {args.split_mode}")
        folds = [requested]
    summaries = []
    predictions = []
    for fold_id in folds:
        summary, frame = run_fold(
            fold_id,
            features,
            masks,
            metadata,
            frequency,
            output_dir,
            args,
            run_config_hash,
            manifest_hash,
        )
        summaries.append(summary)
        predictions.append(frame)
    atomic_write_json(output_dir / "fold_summaries.json", summaries)
    expected_fold_count = 3 if args.split_mode == "source_lodo" else 5
    if len(folds) == expected_fold_count:
        summarize_predictions(pd.concat(predictions, ignore_index=True), output_dir)
        (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    else:
        (output_dir / "RUN.status").write_text("partial\n", encoding="utf-8")


if __name__ == "__main__":
    main()
