from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SUBTYPE_NAMES = ["A", "AB", "B1", "B2", "B3", "TC"]
SUBTYPE_TO_INDEX = {name: index for index, name in enumerate(SUBTYPE_NAMES)}
DEFAULT_SPLIT_CSV = (
    "/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/"
    "task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
)
DEFAULT_CONCEPT_COLUMNS = [
    "boundary_clear",
    "boundary_unclear",
    "capsule_any",
    "capsule_complete",
    "capsule_absent",
    "invasion",
    "hemorrhage",
    "necrosis",
    "cystic_change",
    "calcification",
    "nodular_lobulated",
    "septum",
    "homogeneous",
    "gray_white",
    "gray_yellow",
    "gray_red",
    "texture_soft",
    "texture_medium",
    "texture_tough",
    "texture_fragile",
    "cut_surface_mentioned",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated Task7 heads over frozen dense feature banks.")
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument(
        "--concept-csv",
        default="/workspace/thymic_project/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv",
    )
    parser.add_argument("--concept-columns", default=",".join(DEFAULT_CONCEPT_COLUMNS))
    parser.add_argument(
        "--pooling",
        choices=(
            "mean",
            "gated",
            "view_gated",
            "gated_stats",
            "view_gated_stats",
            "spatial_pyramid",
        ),
        default="gated",
    )
    parser.add_argument("--expert-mode", choices=("none", "boundary", "low_b2", "moe"), default="none")
    parser.add_argument("--risk-objective", choices=("ce", "rex", "group_dro"), default="ce")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sam-rho", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtype-balanced-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-loss-weight", type=float, default=0.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--concept-loss-weight", type=float, default=0.0)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--boundary-relevance-loss-weight", type=float, default=0.0)
    parser.add_argument("--boundary-triplet-weight", type=float, default=0.0)
    parser.add_argument("--boundary-triplet-margin", type=float, default=0.25)
    parser.add_argument("--boundary-fusion-alpha", type=float, default=0.65)
    parser.add_argument("--prototype-temperature", type=float, default=0.12)
    parser.add_argument("--risk-from-subtype-alpha", type=float, default=0.0)
    parser.add_argument("--rex-weight", type=float, default=1.0)
    parser.add_argument("--group-dro-eta", type=float, default=0.05)
    parser.add_argument("--moe-specialist-weight", type=float, default=0.25)
    parser.add_argument("--moe-balance-weight", type=float, default=0.02)
    parser.add_argument("--moe-gate-supervision-weight", type=float, default=0.0)
    parser.add_argument("--soft-balanced-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--visual-conflict-softening",
        type=float,
        default=0.0,
        help="Move this fraction of target mass to the opposite risk class for gross-concept conflict cases.",
    )
    parser.add_argument("--domain-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--domain-adversarial-lambda", type=float, default=0.5)
    parser.add_argument("--class-conditional-align-weight", type=float, default=0.0)
    parser.add_argument("--sentinel-fusion-alpha", type=float, default=0.0)
    parser.add_argument("--sentinel-loss-weight", type=float, default=0.0)
    parser.add_argument("--sentinel-positive-weight", type=float, default=1.5)
    parser.add_argument("--sentinel-positive-gamma", type=float, default=0.0)
    parser.add_argument("--sentinel-negative-gamma", type=float, default=2.0)
    parser.add_argument("--mixstyle-probability", type=float, default=0.0)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--view-consistency-weight", type=float, default=0.0)
    parser.add_argument("--view-supervision-weight", type=float, default=0.0)
    parser.add_argument("--load-features-to-ram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), default="fivefold")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_summary(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, probability))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def selection_value(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, (probability >= 0.5).astype(int)))


def prepare_metadata(feature_bank_dir: Path, split_csv: str, concept_csv: str, concept_columns: list[str]):
    metadata = pd.read_csv(
        feature_bank_dir / "metadata.csv",
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    metadata.columns = [str(column).lstrip("\ufeff") for column in metadata.columns]
    fold_values = pd.to_numeric(metadata["master_fold_id"], errors="coerce")
    split_frame = pd.read_csv(split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split_frame.columns = [str(column).lstrip("\ufeff") for column in split_frame.columns]
    split_frame = split_frame[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    split_lookup = metadata[["case_id"]].merge(split_frame, on="case_id", how="left")["master_fold_id"]
    split_values = pd.to_numeric(split_lookup, errors="coerce")
    conflicts = fold_values.notna() & split_values.notna() & (fold_values != split_values)
    if conflicts.any():
        print(
            f"[split] overriding {int(conflicts.sum())} registry fold values with the locked old+third split",
            flush=True,
        )
    metadata["master_fold_id"] = split_values.fillna(fold_values)
    if metadata["master_fold_id"].isna().any():
        missing_cases = metadata.loc[metadata["master_fold_id"].isna(), "case_id"].head(10).tolist()
        raise ValueError(f"Locked fold assignment missing for cases: {missing_cases}")
    metadata["master_fold_id"] = metadata["master_fold_id"].astype(int)
    metadata["label_idx"] = pd.to_numeric(metadata["label_idx"], errors="raise").astype(int)
    metadata["subtype_idx"] = metadata["task_l6_label"].map(SUBTYPE_TO_INDEX)
    if metadata["subtype_idx"].isna().any():
        missing = metadata.loc[metadata["subtype_idx"].isna(), "task_l6_label"].value_counts().to_dict()
        raise ValueError(f"Unknown Task6 subtype labels: {missing}")
    metadata["subtype_idx"] = metadata["subtype_idx"].astype(int)
    group_names = sorted(metadata["source_dataset"].astype(str).unique().tolist())
    group_to_idx = {name: idx for idx, name in enumerate(group_names)}
    metadata["group_idx"] = metadata["source_dataset"].astype(str).map(group_to_idx).astype(int)
    metadata["gross_highrisk_score"] = np.float32(0.0)
    metadata["gross_conflict_score"] = np.float32(0.0)

    concept_labels = np.zeros((len(metadata), len(concept_columns)), dtype=np.float32)
    concept_mask = np.zeros_like(concept_labels)
    if concept_columns:
        usecols = [
            "original_case_id",
            "concept_has_gross_text",
            "gross_highrisk_score",
            "gross_conflict_score",
            *concept_columns,
        ]
        concept_frame = pd.read_csv(
            concept_csv,
            usecols=usecols,
            dtype={"original_case_id": str},
            encoding="utf-8-sig",
        )
        concept_frame.columns = [str(column).lstrip("\ufeff") for column in concept_frame.columns]
        concept_frame = concept_frame.drop_duplicates("original_case_id")
        joined = metadata[["original_case_id"]].merge(concept_frame, on="original_case_id", how="left")
        has_text = pd.to_numeric(joined["concept_has_gross_text"], errors="coerce").fillna(0).to_numpy() > 0
        metadata["gross_highrisk_score"] = pd.to_numeric(
            joined["gross_highrisk_score"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)
        metadata["gross_conflict_score"] = pd.to_numeric(
            joined["gross_conflict_score"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)
        for col_idx, column in enumerate(concept_columns):
            values = pd.to_numeric(joined[column], errors="coerce")
            available = has_text & values.notna().to_numpy()
            concept_labels[available, col_idx] = values.to_numpy(dtype=np.float32, na_value=0.0)[available]
            concept_mask[available, col_idx] = 1.0
    return metadata, concept_labels, concept_mask, group_names


class FeatureDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame,
        concept_labels: np.ndarray,
        concept_mask: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.features = features
        self.metadata = metadata
        self.concept_labels = concept_labels
        self.concept_mask = concept_mask
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        row = self.metadata.iloc[index]
        feature = torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True))
        return {
            "feature": feature,
            "label": torch.tensor(int(row["label_idx"]), dtype=torch.long),
            "subtype": torch.tensor(int(row["subtype_idx"]), dtype=torch.long),
            "group": torch.tensor(int(row["group_idx"]), dtype=torch.long),
            "concept": torch.from_numpy(self.concept_labels[index].copy()),
            "concept_mask": torch.from_numpy(self.concept_mask[index].copy()),
            "visual_conflict": torch.tensor(float(row.get("gross_conflict_score", 0.0)), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


class GatedTokenPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.tanh = nn.Linear(hidden_dim, attention_dim)
        self.sigmoid = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(self.tanh(tokens)) * torch.sigmoid(self.sigmoid(tokens))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=-2)
        return pooled, weights


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, float(scale))


class DenseTask7Model(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
        pooling: str,
        expert_mode: str,
        num_concepts: int,
        num_groups: int,
        prototype_temperature: float,
        boundary_fusion_alpha: float,
        domain_adversarial_lambda: float,
        risk_from_subtype_alpha: float,
        sentinel_fusion_alpha: float,
        mixstyle_probability: float,
        mixstyle_alpha: float,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.expert_mode = expert_mode
        self.prototype_temperature = float(prototype_temperature)
        self.boundary_fusion_alpha = float(boundary_fusion_alpha)
        self.domain_adversarial_lambda = float(domain_adversarial_lambda)
        self.risk_from_subtype_alpha = float(risk_from_subtype_alpha)
        self.sentinel_fusion_alpha = float(sentinel_fusion_alpha)
        self.mixstyle_probability = float(mixstyle_probability)
        self.mixstyle_alpha = float(mixstyle_alpha)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.token_pool = GatedTokenPool(hidden_dim, attention_dim)
        self.stats_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_level_embeddings = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.spatial_level_embeddings, std=0.02)
        self.view_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, 1),
        )
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.risk_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        self.sentinel_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.subtype_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, len(SUBTYPE_NAMES)))
        self.concept_head = nn.Linear(hidden_dim, num_concepts) if num_concepts else None
        self.prototypes = nn.Parameter(torch.randn(len(SUBTYPE_NAMES), hidden_dim) * 0.02)
        self.boundary_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        self.boundary_relevance_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.moe_heads = nn.ModuleList(
            [nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2)) for _ in range(3)]
        )
        self.moe_gate = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.GELU(), nn.Linear(attention_dim, 3))
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, num_groups),
        )

    def pool_features(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, num_views, token_count, feature_dim = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        if self.training and self.mixstyle_probability > 0 and batch_size > 1:
            if torch.rand((), device=tokens.device) < self.mixstyle_probability:
                mean = tokens.mean(dim=2, keepdim=True)
                scale = tokens.var(dim=2, keepdim=True, unbiased=False).add(1e-6).sqrt()
                normalized = (tokens - mean) / scale
                permutation = torch.randperm(batch_size, device=tokens.device)
                concentration = max(self.mixstyle_alpha, 1e-3)
                beta = torch.distributions.Beta(concentration, concentration)
                mixture = beta.sample((batch_size,)).to(tokens.device).view(batch_size, 1, 1, 1)
                mixed_mean = mixture * mean + (1.0 - mixture) * mean[permutation]
                mixed_scale = mixture * scale + (1.0 - mixture) * scale[permutation]
                tokens = normalized * mixed_scale + mixed_mean
        diagnostics: dict[str, torch.Tensor] = {
            "per_view_embedding": self.embedding_norm(tokens.mean(dim=2)),
        }
        if self.pooling == "mean":
            pooled = tokens.mean(dim=(1, 2))
            return self.embedding_norm(pooled), diagnostics
        if self.pooling == "gated":
            pooled, token_weights = self.token_pool(tokens.reshape(batch_size, num_views * token_count, -1))
            diagnostics["token_weights"] = token_weights
            return self.embedding_norm(pooled), diagnostics
        if self.pooling == "gated_stats":
            flat_tokens = tokens.reshape(batch_size, num_views * token_count, -1)
            gated, token_weights = self.token_pool(flat_tokens)
            mean = flat_tokens.mean(dim=1)
            scale = flat_tokens.var(dim=1, unbiased=False).add(1e-6).sqrt()
            pooled = self.stats_fusion(torch.cat([gated, mean, scale], dim=-1))
            diagnostics["token_weights"] = token_weights.reshape(batch_size, num_views, token_count)
            return self.embedding_norm(pooled), diagnostics
        if self.pooling == "spatial_pyramid":
            grid_size = math.isqrt(token_count)
            if grid_size * grid_size != token_count:
                raise ValueError(
                    f"spatial_pyramid requires square patch tokens, received token_count={token_count}"
                )
            token_grid = tokens.reshape(batch_size * num_views, grid_size, grid_size, -1).permute(0, 3, 1, 2)
            regions = []
            for level_index, bins in enumerate((1, 2, 4)):
                level = F.adaptive_avg_pool2d(token_grid, output_size=(bins, bins))
                level = level.flatten(2).transpose(1, 2)
                level = level + self.spatial_level_embeddings[level_index].view(1, 1, -1)
                regions.append(level)
            region_tokens = torch.cat(regions, dim=1)
            region_count = region_tokens.shape[1]
            region_tokens = region_tokens.reshape(batch_size, num_views * region_count, -1)
            pooled, region_weights = self.token_pool(region_tokens)
            diagnostics["token_weights"] = region_weights.reshape(batch_size, num_views, region_count)
            return self.embedding_norm(pooled), diagnostics
        view_tokens = tokens.reshape(batch_size * num_views, token_count, -1)
        view_repr, token_weights = self.token_pool(view_tokens)
        if self.pooling == "view_gated_stats":
            view_mean = view_tokens.mean(dim=1)
            view_scale = view_tokens.var(dim=1, unbiased=False).add(1e-6).sqrt()
            view_repr = self.stats_fusion(torch.cat([view_repr, view_mean, view_scale], dim=-1))
        view_repr = view_repr.reshape(batch_size, num_views, -1)
        global_context = view_repr.mean(dim=1, keepdim=True).expand_as(view_repr)
        view_scores = self.view_gate(torch.cat([view_repr, global_context], dim=-1)).squeeze(-1)
        view_weights = torch.softmax(view_scores, dim=1)
        pooled = torch.sum(view_repr * view_weights.unsqueeze(-1), dim=1)
        diagnostics["token_weights"] = token_weights.reshape(batch_size, num_views, token_count)
        diagnostics["view_weights"] = view_weights
        return self.embedding_norm(pooled), diagnostics

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding, diagnostics = self.pool_features(features)
        general_logits = self.risk_head(embedding)
        subtype_logits = self.subtype_head(embedding)
        boundary_logits = self.boundary_head(embedding)
        output: dict[str, torch.Tensor] = {
            "embedding": embedding,
            "general_logits": general_logits,
            "subtype_logits": subtype_logits,
            "boundary_logits": boundary_logits,
            "boundary_relevance_logits": self.boundary_relevance_head(embedding).squeeze(1),
            "domain_logits": self.domain_head(gradient_reverse(embedding, self.domain_adversarial_lambda)),
            "sentinel_logit": self.sentinel_head(embedding).squeeze(1),
        }
        output.update(diagnostics)
        output["view_risk_logits"] = self.risk_head(output["per_view_embedding"])
        if self.concept_head is not None:
            output["concept_logits"] = self.concept_head(embedding)
        normalized_embedding = F.normalize(embedding, dim=1)
        normalized_prototypes = F.normalize(self.prototypes, dim=1)
        output["prototype_logits"] = normalized_embedding @ normalized_prototypes.T / max(
            self.prototype_temperature, 1e-6
        )

        if self.expert_mode in {"boundary", "low_b2"}:
            general_prob = torch.softmax(general_logits, dim=1)
            boundary_prob = torch.softmax(boundary_logits, dim=1)
            boundary_gate = self.boundary_fusion_alpha * torch.sigmoid(output["boundary_relevance_logits"])
            mixed = (1.0 - boundary_gate.unsqueeze(1)) * general_prob + boundary_gate.unsqueeze(1) * boundary_prob
            output["risk_log_prob"] = torch.log(mixed.clamp_min(1e-7))
            output["boundary_gate"] = boundary_gate
        elif self.expert_mode == "moe":
            expert_logits = torch.stack([head(embedding) for head in self.moe_heads], dim=1)
            expert_prob = torch.softmax(expert_logits, dim=-1)
            gate = torch.softmax(self.moe_gate(embedding), dim=1)
            mixed = torch.sum(expert_prob * gate.unsqueeze(-1), dim=1)
            output["risk_log_prob"] = torch.log(mixed.clamp_min(1e-7))
            output["expert_logits"] = expert_logits
            output["expert_gate"] = gate
        else:
            output["risk_log_prob"] = F.log_softmax(general_logits, dim=1)
        if self.risk_from_subtype_alpha > 0:
            subtype_probability = torch.softmax(subtype_logits, dim=1)
            subtype_high = subtype_probability[:, [
                SUBTYPE_TO_INDEX["B2"],
                SUBTYPE_TO_INDEX["B3"],
                SUBTYPE_TO_INDEX["TC"],
            ]].sum(dim=1)
            subtype_risk = torch.stack([1.0 - subtype_high, subtype_high], dim=1)
            direct_risk = output["risk_log_prob"].exp()
            alpha = min(max(self.risk_from_subtype_alpha, 0.0), 1.0)
            mixed_risk = (1.0 - alpha) * direct_risk + alpha * subtype_risk
            output["risk_log_prob"] = torch.log(mixed_risk.clamp_min(1e-7))
        if self.sentinel_fusion_alpha > 0:
            alpha = min(max(self.sentinel_fusion_alpha, 0.0), 1.0)
            base_high = output["risk_log_prob"].exp()[:, 1]
            sentinel_high = torch.sigmoid(output["sentinel_logit"])
            fused_high = (1.0 - alpha) * base_high + alpha * sentinel_high
            fused_risk = torch.stack([1.0 - fused_high, fused_high], dim=1)
            output["risk_log_prob"] = torch.log(fused_risk.clamp_min(1e-7))
        return output


def boundary_triplet_loss(embedding: torch.Tensor, subtypes: torch.Tensor, margin: float) -> torch.Tensor:
    b1 = SUBTYPE_TO_INDEX["B1"]
    b2 = SUBTYPE_TO_INDEX["B2"]
    mask = (subtypes == b1) | (subtypes == b2)
    if int(mask.sum()) < 4:
        return embedding.sum() * 0.0
    z = F.normalize(embedding[mask], dim=1)
    labels = subtypes[mask]
    similarity = z @ z.T
    losses: list[torch.Tensor] = []
    for idx in range(len(labels)):
        positive_mask = (labels == labels[idx]).clone()
        positive_mask[idx] = False
        negative_mask = labels != labels[idx]
        if not positive_mask.any() or not negative_mask.any():
            continue
        hardest_positive_distance = (1.0 - similarity[idx][positive_mask]).max()
        hardest_negative_distance = (1.0 - similarity[idx][negative_mask]).min()
        losses.append(F.relu(hardest_positive_distance - hardest_negative_distance + float(margin)))
    return torch.stack(losses).mean() if losses else embedding.sum() * 0.0


def class_conditional_alignment_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    """Align source domains within each risk class without matching class priors."""
    losses: list[torch.Tensor] = []
    for label in labels.unique():
        label_mask = labels == label
        present_groups = groups[label_mask].unique()
        for left_index in range(len(present_groups)):
            for right_index in range(left_index + 1, len(present_groups)):
                left = embedding[label_mask & (groups == present_groups[left_index])]
                right = embedding[label_mask & (groups == present_groups[right_index])]
                if len(left) < 2 or len(right) < 2:
                    continue
                mean_loss = (left.mean(dim=0) - right.mean(dim=0)).square().mean()
                scale_loss = (
                    left.var(dim=0, unbiased=False).sqrt()
                    - right.var(dim=0, unbiased=False).sqrt()
                ).square().mean()
                losses.append(mean_loss + 0.25 * scale_loss)
    return torch.stack(losses).mean() if losses else embedding.sum() * 0.0


@dataclass
class GroupDROState:
    weights: torch.Tensor
    eta: float

    def aggregate(self, per_sample_loss: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        group_losses = []
        present = []
        for group_idx in range(len(self.weights)):
            mask = groups == group_idx
            if mask.any():
                group_losses.append(per_sample_loss[mask].mean())
                present.append(group_idx)
        if not group_losses:
            return per_sample_loss.mean()
        stacked = torch.stack(group_losses)
        present_tensor = torch.tensor(present, device=per_sample_loss.device, dtype=torch.long)
        with torch.no_grad():
            self.weights[present_tensor] *= torch.exp(float(self.eta) * stacked.detach())
            self.weights /= self.weights.sum().clamp_min(1e-12)
        present_weights = self.weights[present_tensor]
        present_weights = present_weights / present_weights.sum().clamp_min(1e-12)
        return torch.sum(present_weights * stacked)


def aggregate_risk_loss(
    per_sample_loss: torch.Tensor,
    groups: torch.Tensor,
    objective: str,
    rex_weight: float,
    group_dro_state: GroupDROState | None,
) -> torch.Tensor:
    if objective == "ce":
        return per_sample_loss.mean()
    group_losses = []
    for group_idx in torch.unique(groups):
        mask = groups == group_idx
        group_losses.append(per_sample_loss[mask].mean())
    if objective == "rex":
        stacked = torch.stack(group_losses)
        variance = torch.var(stacked, unbiased=False) if len(stacked) > 1 else stacked.sum() * 0.0
        return stacked.mean() + float(rex_weight) * variance
    assert group_dro_state is not None
    return group_dro_state.aggregate(per_sample_loss, groups)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    weights = counts.sum() / np.maximum(counts * len(counts), 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def concept_pos_weights(labels: np.ndarray, mask: np.ndarray, device: torch.device) -> torch.Tensor:
    positive = (labels * mask).sum(axis=0)
    negative = ((1.0 - labels) * mask).sum(axis=0)
    weights = np.clip(negative / np.maximum(positive, 1.0), 0.25, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    features: np.ndarray,
    metadata: pd.DataFrame,
    concept_labels: np.ndarray,
    concept_mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    training: bool,
    subtype_balanced_sampler: bool,
) -> DataLoader:
    dataset = FeatureDataset(features, metadata, concept_labels, concept_mask, indices)
    sampler = None
    shuffle = training
    if training and subtype_balanced_sampler:
        subtypes = metadata.iloc[indices]["subtype_idx"].to_numpy(dtype=int)
        counts = np.bincount(subtypes, minlength=len(SUBTYPE_NAMES))
        sample_weights = 1.0 / np.maximum(counts[subtypes], 1)
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double), num_samples=len(indices), replacement=True
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    risk_weights: torch.Tensor | None,
    concept_weights: torch.Tensor,
    group_dro_state: GroupDROState | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_sample_risk = F.nll_loss(output["risk_log_prob"], batch["label"], weight=risk_weights, reduction="none")
    if args.focal_gamma > 0:
        true_probability = output["risk_log_prob"].exp().gather(
            1, batch["label"].unsqueeze(1)
        ).squeeze(1)
        per_sample_risk = per_sample_risk * (
            1.0 - true_probability
        ).clamp_min(1e-7).pow(float(args.focal_gamma))
    if args.visual_conflict_softening > 0:
        opposite_labels = 1 - batch["label"]
        opposite_loss = F.nll_loss(output["risk_log_prob"], opposite_labels, reduction="none")
        if risk_weights is not None:
            opposite_loss = opposite_loss * risk_weights[opposite_labels]
        softening = (
            float(args.visual_conflict_softening)
            * batch["visual_conflict"].clamp(min=0.0, max=1.0)
        )
        per_sample_risk = (1.0 - softening) * per_sample_risk + softening * opposite_loss
    risk_loss = aggregate_risk_loss(
        per_sample_risk,
        batch["group"],
        args.risk_objective,
        args.rex_weight,
        group_dro_state,
    )
    total = risk_loss
    parts = {"risk": float(risk_loss.detach())}

    if args.soft_balanced_loss_weight > 0:
        high_probability = output["risk_log_prob"].exp()[:, 1]
        positive_mask = batch["label"] == 1
        negative_mask = batch["label"] == 0
        if positive_mask.any() and negative_mask.any():
            soft_sensitivity = high_probability[positive_mask].mean()
            soft_specificity = (1.0 - high_probability[negative_mask]).mean()
            soft_balanced = -0.5 * (
                torch.log(soft_sensitivity.clamp_min(1e-6))
                + torch.log(soft_specificity.clamp_min(1e-6))
            )
            total = total + float(args.soft_balanced_loss_weight) * soft_balanced
            parts["soft_balanced"] = float(soft_balanced.detach())

    if args.subtype_loss_weight > 0:
        subtype_loss = F.cross_entropy(output["subtype_logits"], batch["subtype"])
        total = total + float(args.subtype_loss_weight) * subtype_loss
        parts["subtype"] = float(subtype_loss.detach())

    if args.ordinal_loss_weight > 0:
        subtype_probability = torch.softmax(output["subtype_logits"], dim=1)
        subtype_order = torch.arange(
            len(SUBTYPE_NAMES), device=subtype_probability.device, dtype=subtype_probability.dtype
        )
        expected_subtype = torch.sum(subtype_probability * subtype_order.unsqueeze(0), dim=1)
        ordinal_loss = F.smooth_l1_loss(expected_subtype, batch["subtype"].float())
        total = total + float(args.ordinal_loss_weight) * ordinal_loss
        parts["ordinal"] = float(ordinal_loss.detach())

    if args.view_consistency_weight > 0:
        view_log_probability = F.log_softmax(output["view_risk_logits"], dim=-1)
        fused_probability = output["risk_log_prob"].exp().detach().unsqueeze(1)
        consistency_loss = F.kl_div(
            view_log_probability,
            fused_probability.expand_as(view_log_probability),
            reduction="none",
        ).sum(dim=-1).mean()
        total = total + float(args.view_consistency_weight) * consistency_loss
        parts["view_consistency"] = float(consistency_loss.detach())

    if args.view_supervision_weight > 0:
        view_logits = output["view_risk_logits"]
        view_targets = batch["label"].unsqueeze(1).expand(-1, view_logits.shape[1]).reshape(-1)
        view_loss = F.cross_entropy(
            view_logits.reshape(-1, 2),
            view_targets,
            weight=risk_weights,
        )
        total = total + float(args.view_supervision_weight) * view_loss
        parts["view_supervision"] = float(view_loss.detach())

    if args.concept_loss_weight > 0 and "concept_logits" in output:
        raw_concept = F.binary_cross_entropy_with_logits(
            output["concept_logits"], batch["concept"], reduction="none", pos_weight=concept_weights
        )
        masked = raw_concept * batch["concept_mask"]
        concept_loss = masked.sum() / batch["concept_mask"].sum().clamp_min(1.0)
        total = total + float(args.concept_loss_weight) * concept_loss
        parts["concept"] = float(concept_loss.detach())

    if args.prototype_loss_weight > 0:
        prototype_loss = F.cross_entropy(output["prototype_logits"], batch["subtype"])
        total = total + float(args.prototype_loss_weight) * prototype_loss
        parts["prototype"] = float(prototype_loss.detach())

    b1_b2_mask = (batch["subtype"] == SUBTYPE_TO_INDEX["B1"]) | (
        batch["subtype"] == SUBTYPE_TO_INDEX["B2"]
    )
    boundary_mask = b1_b2_mask
    if args.expert_mode == "low_b2":
        boundary_mask = batch["subtype"] <= SUBTYPE_TO_INDEX["B2"]
    if args.boundary_loss_weight > 0 or args.expert_mode in {"boundary", "low_b2"}:
        if boundary_mask.any():
            boundary_targets = batch["label"][boundary_mask]
            boundary_loss = F.cross_entropy(output["boundary_logits"][boundary_mask], boundary_targets)
            weight = max(float(args.boundary_loss_weight), 0.35 if args.expert_mode in {"boundary", "low_b2"} else 0.0)
            total = total + weight * boundary_loss
            parts["boundary"] = float(boundary_loss.detach())

    if args.expert_mode in {"boundary", "low_b2"}:
        relevance_targets = b1_b2_mask.float()
        if args.expert_mode == "low_b2":
            relevance_targets = (batch["subtype"] <= SUBTYPE_TO_INDEX["B2"]).float()
        relevance_loss = F.binary_cross_entropy_with_logits(
            output["boundary_relevance_logits"], relevance_targets
        )
        relevance_weight = max(float(args.boundary_relevance_loss_weight), 0.20)
        total = total + relevance_weight * relevance_loss
        parts["boundary_relevance"] = float(relevance_loss.detach())

    if args.boundary_triplet_weight > 0:
        triplet = boundary_triplet_loss(output["embedding"], batch["subtype"], args.boundary_triplet_margin)
        total = total + float(args.boundary_triplet_weight) * triplet
        parts["triplet"] = float(triplet.detach())

    if args.expert_mode == "moe":
        expert_logits = output["expert_logits"]
        specialist_losses = []
        expert1_mask = torch.isin(
            batch["subtype"],
            torch.tensor(
                [SUBTYPE_TO_INDEX["AB"], SUBTYPE_TO_INDEX["B1"], SUBTYPE_TO_INDEX["B2"]],
                device=batch["subtype"].device,
            ),
        )
        expert2_mask = torch.isin(
            batch["subtype"],
            torch.tensor(
                [SUBTYPE_TO_INDEX["B1"], SUBTYPE_TO_INDEX["B2"], SUBTYPE_TO_INDEX["B3"], SUBTYPE_TO_INDEX["TC"]],
                device=batch["subtype"].device,
            ),
        )
        if expert1_mask.any():
            specialist_losses.append(F.cross_entropy(expert_logits[expert1_mask, 1], batch["label"][expert1_mask]))
        if expert2_mask.any():
            specialist_losses.append(F.cross_entropy(expert_logits[expert2_mask, 2], batch["label"][expert2_mask]))
        if specialist_losses:
            specialist_loss = torch.stack(specialist_losses).mean()
            total = total + float(args.moe_specialist_weight) * specialist_loss
            parts["specialist"] = float(specialist_loss.detach())
        mean_gate = output["expert_gate"].mean(dim=0)
        balance_loss = torch.sum((mean_gate - (1.0 / len(mean_gate))) ** 2)
        total = total + float(args.moe_balance_weight) * balance_loss
        parts["moe_balance"] = float(balance_loss.detach())
        if args.moe_gate_supervision_weight > 0:
            gate_targets = torch.zeros_like(batch["subtype"])
            gate_targets[(batch["subtype"] == SUBTYPE_TO_INDEX["B1"]) | (batch["subtype"] == SUBTYPE_TO_INDEX["B2"])] = 1
            gate_targets[(batch["subtype"] == SUBTYPE_TO_INDEX["B3"]) | (batch["subtype"] == SUBTYPE_TO_INDEX["TC"])] = 2
            gate_supervision = F.nll_loss(torch.log(output["expert_gate"].clamp_min(1e-7)), gate_targets)
            total = total + float(args.moe_gate_supervision_weight) * gate_supervision
            parts["moe_gate_supervision"] = float(gate_supervision.detach())

    if args.domain_adversarial_weight > 0:
        domain_loss = F.cross_entropy(output["domain_logits"], batch["group"])
        total = total + float(args.domain_adversarial_weight) * domain_loss
        parts["domain_adversarial"] = float(domain_loss.detach())

    if args.class_conditional_align_weight > 0:
        alignment_loss = class_conditional_alignment_loss(
            output["embedding"], batch["label"], batch["group"]
        )
        total = total + float(args.class_conditional_align_weight) * alignment_loss
        parts["class_conditional_align"] = float(alignment_loss.detach())

    if args.sentinel_loss_weight > 0:
        target = batch["label"].float()
        probability = torch.sigmoid(output["sentinel_logit"])
        positive_loss = -float(args.sentinel_positive_weight) * target * (
            (1.0 - probability).clamp_min(1e-7) ** float(args.sentinel_positive_gamma)
        ) * torch.log(probability.clamp_min(1e-7))
        negative_loss = -(1.0 - target) * (
            probability.clamp_min(1e-7) ** float(args.sentinel_negative_gamma)
        ) * torch.log((1.0 - probability).clamp_min(1e-7))
        sentinel_loss = (positive_loss + negative_loss).mean()
        total = total + float(args.sentinel_loss_weight) * sentinel_loss
        parts["sentinel"] = float(sentinel_loss.detach())

    parts["total"] = float(total.detach())
    return total, parts


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            device_batch = move_batch(batch, device)
            output = model(device_batch["feature"])
            probability = output["risk_log_prob"].exp()[:, 1]
            subtype_probability = torch.softmax(output["subtype_logits"], dim=1)
            for idx in range(len(probability)):
                row = {
                    "feature_row": int(device_batch["index"][idx].item()),
                    "label_idx": int(device_batch["label"][idx].item()),
                    "prob_high": float(probability[idx].item()),
                    "pred_idx": int(probability[idx].item() >= 0.5),
                    "pred_subtype_idx": int(torch.argmax(subtype_probability[idx]).item()),
                }
                for subtype_idx, subtype_name in enumerate(SUBTYPE_NAMES):
                    row[f"prob_subtype_{subtype_name}"] = float(subtype_probability[idx, subtype_idx].item())
                if "boundary_gate" in output:
                    row["boundary_gate"] = float(output["boundary_gate"][idx].item())
                if "expert_gate" in output:
                    for gate_idx in range(output["expert_gate"].shape[1]):
                        row[f"expert_gate_{gate_idx}"] = float(output["expert_gate"][idx, gate_idx].item())
                if "view_weights" in output:
                    for view_idx in range(output["view_weights"].shape[1]):
                        row[f"view_weight_{view_idx}"] = float(output["view_weights"][idx, view_idx].item())
                rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def sam_perturb(model: nn.Module, rho: float) -> list[tuple[nn.Parameter, torch.Tensor]]:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        return []
    grad_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(parameter.grad.detach(), ord=2) for parameter in parameters]),
        ord=2,
    )
    scale = float(rho) / float(grad_norm.item() + 1e-12)
    perturbations = []
    for parameter in parameters:
        perturbation = parameter.grad.detach() * scale
        parameter.add_(perturbation)
        perturbations.append((parameter, perturbation))
    return perturbations


@torch.no_grad()
def sam_restore(perturbations: list[tuple[nn.Parameter, torch.Tensor]]) -> None:
    for parameter, perturbation in perturbations:
        parameter.sub_(perturbation)


def train_fold(
    args: argparse.Namespace,
    fold_id: int,
    features: np.ndarray,
    metadata: pd.DataFrame,
    concept_labels: np.ndarray,
    concept_mask: np.ndarray,
    group_names: list[str],
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    set_seed(args.seed + fold_id)
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    val_fold = (fold_id % 5) + 1
    held_out_source = ""
    if args.split_mode == "source_lodo":
        held_out_source = group_names[fold_id - 1]
        source_values = metadata["source_dataset"].astype(str).to_numpy()
        held_out_mask = source_values == held_out_source
        val_mask = (~held_out_mask) & (metadata["master_fold_id"].to_numpy() == val_fold)
        train_mask = (~held_out_mask) & (~val_mask)
        test_indices = np.flatnonzero(held_out_mask)
        val_indices = np.flatnonzero(val_mask)
        train_indices = np.flatnonzero(train_mask)
    else:
        test_indices = np.flatnonzero(metadata["master_fold_id"].to_numpy() == fold_id)
        val_indices = np.flatnonzero(metadata["master_fold_id"].to_numpy() == val_fold)
        train_indices = np.flatnonzero(~metadata["master_fold_id"].isin([fold_id, val_fold]).to_numpy())
    if not len(train_indices) or not len(val_indices) or not len(test_indices):
        raise ValueError(f"Empty fold partition for fold={fold_id}")

    train_loader = make_loader(
        features,
        metadata,
        concept_labels,
        concept_mask,
        train_indices,
        args.batch_size,
        args.num_workers,
        True,
        args.subtype_balanced_sampler,
    )
    val_loader = make_loader(
        features, metadata, concept_labels, concept_mask, val_indices, args.batch_size, args.num_workers, False, False
    )
    test_loader = make_loader(
        features, metadata, concept_labels, concept_mask, test_indices, args.batch_size, args.num_workers, False, False
    )

    device = torch.device(args.device)
    _, num_views, _, feature_dim = features.shape
    model = DenseTask7Model(
        feature_dim=feature_dim,
        num_views=num_views,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        expert_mode=args.expert_mode,
        num_concepts=concept_labels.shape[1] if args.concept_loss_weight > 0 else 0,
        num_groups=len(group_names),
        prototype_temperature=args.prototype_temperature,
        boundary_fusion_alpha=args.boundary_fusion_alpha,
        domain_adversarial_lambda=args.domain_adversarial_lambda,
        risk_from_subtype_alpha=args.risk_from_subtype_alpha,
        sentinel_fusion_alpha=args.sentinel_fusion_alpha,
        mixstyle_probability=args.mixstyle_probability,
        mixstyle_alpha=args.mixstyle_alpha,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.sam_rho < 0:
        raise ValueError("sam_rho must be non-negative")
    max_epochs = int(args.max_epochs or args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    risk_weights = class_weights(metadata.iloc[train_indices]["label_idx"].to_numpy(), device) if args.class_weighting else None
    concept_weights = concept_pos_weights(concept_labels[train_indices], concept_mask[train_indices], device)
    group_dro_state = None
    if args.risk_objective == "group_dro":
        if args.sam_rho > 0:
            raise ValueError("SAM is not combined with stateful GroupDRO in this implementation")
        group_dro_state = GroupDROState(
            weights=torch.full((len(group_names),), 1.0 / len(group_names), device=device),
            eta=float(args.group_dro_eta),
        )

    best_metric = -math.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_parts: dict[str, list[float]] = {}
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["feature"])
            loss, parts = compute_loss(output, batch, args, risk_weights, concept_weights, group_dro_state)
            loss.backward()
            if args.sam_rho > 0:
                perturbations = sam_perturb(model, args.sam_rho)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["feature"])
                loss, parts = compute_loss(
                    output, batch, args, risk_weights, concept_weights, group_dro_state
                )
                loss.backward()
                sam_restore(perturbations)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            for key, value in parts.items():
                epoch_parts.setdefault(key, []).append(value)
        scheduler.step()
        val_predictions = evaluate(model, val_loader, device)
        val_metric = selection_value(
            val_predictions["label_idx"].to_numpy(), val_predictions["prob_high"].to_numpy()
        )
        history_row = {"epoch": epoch, "val_balanced_accuracy": val_metric, "lr": optimizer.param_groups[0]["lr"]}
        history_row.update({f"train_{key}": float(np.mean(values)) for key, values in epoch_parts.items()})
        if group_dro_state is not None:
            for idx, value in enumerate(group_dro_state.weights.detach().cpu().tolist()):
                history_row[f"group_weight_{idx}"] = value
        history.append(history_row)
        if val_metric > best_metric + 1e-12:
            best_metric = val_metric
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
        else:
            stale += 1
        print(
            f"fold={fold_id} epoch={epoch} val_bacc={val_metric:.4f} best={best_metric:.4f} "
            f"stale={stale} train_loss={history_row.get('train_total', float('nan')):.4f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    try:
        state = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(fold_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    val_predictions = evaluate(model, val_loader, device)
    test_predictions = evaluate(model, test_loader, device)
    test_metrics = metric_summary(test_predictions["label_idx"], test_predictions["prob_high"])
    summary = {
        "fold_id": fold_id,
        "val_fold_id": val_fold,
        "split_mode": args.split_mode,
        "held_out_source": held_out_source,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_metric,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    write_json(fold_dir / "fold_summary.json", summary)
    val_predictions.merge(metadata, on="feature_row", how="left", suffixes=("", "_meta")).to_csv(
        fold_dir / "val_predictions_with_metadata.csv", index=False, encoding="utf-8-sig"
    )
    test_predictions.merge(metadata, on="feature_row", how="left", suffixes=("", "_meta")).to_csv(
        fold_dir / "test_predictions_with_metadata.csv", index=False, encoding="utf-8-sig"
    )
    test_predictions.to_csv(fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    return summary, test_predictions


def summarize_oof(predictions: pd.DataFrame, metadata: pd.DataFrame, output_dir: Path) -> None:
    merged = predictions.merge(metadata, on="feature_row", how="left", suffixes=("", "_meta"))
    if "label_idx_meta" in merged.columns:
        if not np.array_equal(merged["label_idx"].to_numpy(), merged["label_idx_meta"].to_numpy()):
            raise ValueError("Prediction and metadata labels do not align.")
        merged = merged.drop(columns=["label_idx_meta"])
    merged.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    overall = metric_summary(merged["label_idx"], merged["prob_high"])
    metric_rows.append({"group_type": "overall", "group": "all", **overall})
    for group_column in ["domain", "source_dataset"]:
        for group_name, group in merged.groupby(group_column, dropna=False):
            metric_rows.append(
                {
                    "group_type": group_column,
                    "group": str(group_name),
                    **metric_summary(group["label_idx"], group["prob_high"]),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")

    subtype_rows = []
    for subtype, group in merged.groupby("task_l6_label", dropna=False):
        subtype_rows.append(
            {
                "subtype": str(subtype),
                "n": len(group),
                "risk_label": int(group["label_idx"].iloc[0]),
                "risk_accuracy": float((group["pred_idx"] == group["label_idx"]).mean()),
                "mean_prob_high": float(group["prob_high"].mean()),
                "subtype_head_accuracy": float(
                    (group["pred_subtype_idx"] == group["subtype_idx"]).mean()
                ),
            }
        )
    pd.DataFrame(subtype_rows).to_csv(output_dir / "oof_subtype_summary.csv", index=False, encoding="utf-8-sig")

    true_subtype = merged["subtype_idx"].to_numpy(dtype=int)
    predicted_subtype = merged["pred_subtype_idx"].to_numpy(dtype=int)
    subtype_probability = merged[[f"prob_subtype_{name}" for name in SUBTYPE_NAMES]].to_numpy(dtype=float)
    try:
        subtype_auc = float(
            roc_auc_score(true_subtype, subtype_probability, multi_class="ovr", average="macro")
        )
    except ValueError:
        subtype_auc = float("nan")
    sixclass_metrics = pd.DataFrame(
        [
            {
                "n": len(merged),
                "accuracy": float(accuracy_score(true_subtype, predicted_subtype)),
                "balanced_accuracy": float(balanced_accuracy_score(true_subtype, predicted_subtype)),
                "macro_f1": float(f1_score(true_subtype, predicted_subtype, average="macro", zero_division=0)),
                "macro_auc_ovr": subtype_auc,
            }
        ]
    )
    sixclass_metrics.to_csv(output_dir / "oof_sixclass_metrics.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    feature_bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_config = json.loads((feature_bank_dir / "feature_bank_config.json").read_text(encoding="utf-8"))
    if not feature_config.get("complete"):
        raise RuntimeError("Feature bank is not complete.")
    concept_columns = [item.strip() for item in args.concept_columns.split(",") if item.strip()]
    metadata, concept_labels, concept_mask, group_names = prepare_metadata(
        feature_bank_dir, args.split_csv, args.concept_csv, concept_columns
    )
    loaded = np.load(feature_bank_dir / "dense_features.float16.npy", mmap_mode="r")
    if args.load_features_to_ram:
        features = np.array(loaded, copy=True)
    else:
        features = loaded
    if len(features) != len(metadata):
        raise ValueError(f"Feature rows {len(features)} do not match metadata rows {len(metadata)}")

    run_config = vars(args).copy()
    run_config.update(
        {
            "feature_bank_config": feature_config,
            "concept_columns": concept_columns,
            "concept_covered_cases": int((concept_mask.sum(axis=1) > 0).sum()),
            "group_names": group_names,
            "feature_shape": list(features.shape),
        }
    )
    write_json(output_dir / "run_config.json", run_config)
    if args.split_mode == "source_lodo":
        folds = list(range(1, len(group_names) + 1))
    else:
        folds = sorted(metadata["master_fold_id"].unique().astype(int).tolist())
    if args.fold != "all":
        folds = [int(args.fold)]
    summaries = []
    predictions = []
    for fold_id in folds:
        summary, fold_predictions = train_fold(
            args,
            fold_id,
            features,
            metadata,
            concept_labels,
            concept_mask,
            group_names,
            output_dir,
        )
        summaries.append(summary)
        fold_predictions["fold_id"] = fold_id
        predictions.append(fold_predictions)
    pd.DataFrame(summaries).to_csv(output_dir / "cv_fold_summary.csv", index=False, encoding="utf-8-sig")
    if args.fold == "all":
        summarize_oof(pd.concat(predictions, ignore_index=True), metadata, output_dir)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
