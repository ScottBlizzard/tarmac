from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "thymic_baseline").exists():
    PROJECT_ROOT = Path(os.environ.get("THYMIC_PROJECT_ROOT", "/workspace/thymic_project"))
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_finetune_old_third_20260523 import (  # noqa: E402
    DEFAULT_INPUT,
    load_available_folds,
    make_image_df,
    move_to_device,
    prepare_fold_data,
)
from thymic_baseline.config import INPUT_VARIANTS, TaskConfig, get_task  # noqa: E402
from thymic_baseline.data import ThymicImageDataset  # noqa: E402
from thymic_baseline.metrics import summarize_prediction_frame  # noqa: E402
from thymic_baseline.train import (  # noqa: E402
    build_class_weights,
    build_dataloader,
    evaluate_model,
    format_metric_value,
    is_better_metric,
    metric_summary_text,
    metrics_to_frame,
    resolve_device,
    resolve_selection_metric,
    save_evaluation_outputs,
    write_json,
)

SUBTYPE_TO_ID = {name: index for index, name in enumerate(("A", "AB", "B1", "B2", "B3", "TC"))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 dense-token fine-tuning with low-rank attention adapters.")
    parser.add_argument("--registry-csv", default=str(DEFAULT_INPUT / "registry.csv"))
    parser.add_argument("--split-csv", default=str(DEFAULT_INPUT / "split.csv"))
    parser.add_argument("--task", default="task7_lowrisk_vs_highrisk_tc")
    parser.add_argument("--input-variant", default="whole_plus_crop", choices=INPUT_VARIANTS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="vit_large_patch16_dinov3_qkvb.lvd1689m")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), default="fivefold")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--selection-metric", default="balanced_accuracy")
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pooling", choices=("gated", "view_gated"), default="gated")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-last-blocks", type=int, default=2)
    parser.add_argument("--lora-targets", default="qkv,proj")
    parser.add_argument("--train-final-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--init-encoder-state-template",
        default="",
        help="Optional encoder-only state path; {fold} is replaced by the current fold id.",
    )
    parser.add_argument("--aug-profile", default="style_light")
    parser.add_argument(
        "--train-sampler",
        choices=(
            "none",
            "class_balanced",
            "domain_label_balanced",
            "source_label_balanced",
            "source_subtype_balanced",
        ),
        default="none",
    )
    parser.add_argument("--supervised-contrastive-weight", type=float, default=0.0)
    parser.add_argument(
        "--supervised-contrastive-mode",
        choices=(
            "binary",
            "cross_source",
            "subtype",
            "cross_source_subtype",
            "binary_b1b2",
            "cross_source_b1b2",
        ),
        default="binary",
    )
    parser.add_argument("--supervised-contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--boundary-negative-weight", type=float, default=3.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--audit-partitions-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class LoRAWeightParametrization(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        return original_weight + self.scale * (self.lora_b @ self.lora_a)


def last_vit_blocks(model: nn.Module, count: int) -> list[int]:
    block_ids = []
    for name, _ in model.named_modules():
        pieces = name.split(".")
        if len(pieces) >= 2 and pieces[0] == "blocks" and pieces[1].isdigit():
            block_ids.append(int(pieces[1]))
    unique = sorted(set(block_ids))
    if not unique:
        raise ValueError("LoRA dense fine-tuning currently requires a ViT-style blocks.N hierarchy.")
    return unique[-int(count) :]


def inject_lora(
    model: nn.Module,
    last_blocks: int,
    targets: set[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    selected_blocks = set(last_vit_blocks(model, last_blocks))
    replacements = []
    for name, module in list(model.named_modules()):
        pieces = name.split(".")
        if not isinstance(module, nn.Linear) or len(pieces) < 3:
            continue
        if pieces[0] != "blocks" or not pieces[1].isdigit() or int(pieces[1]) not in selected_blocks:
            continue
        if pieces[-1] not in targets:
            continue
        replacements.append((name, module))
    if not replacements:
        raise ValueError(f"No LoRA targets found for blocks={sorted(selected_blocks)} targets={sorted(targets)}")
    for _, module in replacements:
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
        parametrize.register_parametrization(
            module,
            "weight",
            LoRAWeightParametrization(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
            ),
        )
    return [name for name, _ in replacements]


def unwrap_feature_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ["x_norm_patchtokens", "last_hidden_state", "features", "x"]:
            if key in value and isinstance(value[key], torch.Tensor):
                return value[key]
        tensors = [item for item in value.values() if isinstance(item, torch.Tensor)]
        if tensors:
            return tensors[-1]
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if tensors:
            return tensors[-1]
    raise TypeError(f"Cannot unwrap dense feature tensor from {type(value)!r}")


def dense_tokens(encoder: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    dense = unwrap_feature_tensor(encoder.forward_features(inputs))
    feature_dim = int(getattr(encoder, "num_features", 0))
    if dense.ndim == 3:
        prefix = int(getattr(encoder, "num_prefix_tokens", 0))
        return dense[:, prefix:, :] if prefix else dense
    if dense.ndim == 4 and dense.shape[1] == feature_dim:
        return dense.flatten(2).transpose(1, 2)
    if dense.ndim == 4 and dense.shape[-1] == feature_dim:
        return dense.reshape(dense.shape[0], -1, dense.shape[-1])
    raise ValueError(f"Unsupported dense output shape {tuple(dense.shape)}")


class GatedPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.v = nn.Linear(hidden_dim, attention_dim)
        self.u = nn.Linear(hidden_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = self.w(torch.tanh(self.v(tokens)) * torch.sigmoid(self.u(tokens))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=-2)


class DenseRiskHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
        pooling: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.view_embeddings = nn.Parameter(torch.randn(num_views, hidden_dim) * 0.02)
        self.token_pool = GatedPool(hidden_dim, attention_dim)
        self.view_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim), nn.GELU(), nn.Linear(attention_dim, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, token_count, _ = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        if self.pooling == "gated":
            pooled = self.token_pool(tokens.reshape(batch_size, num_views * token_count, -1))
        else:
            view_repr = self.token_pool(tokens.reshape(batch_size * num_views, token_count, -1))
            view_repr = view_repr.reshape(batch_size, num_views, -1)
            context = view_repr.mean(dim=1, keepdim=True).expand_as(view_repr)
            view_scores = self.view_gate(torch.cat([view_repr, context], dim=-1)).squeeze(-1)
            view_weights = torch.softmax(view_scores, dim=1)
            pooled = torch.sum(view_repr * view_weights.unsqueeze(-1), dim=1)
        logits = self.head(pooled)
        if return_embedding:
            return logits, pooled
        return logits


class LoRADenseTask7Model(nn.Module):
    def __init__(self, args: argparse.Namespace, num_classes: int, fold_id: int | None = None) -> None:
        super().__init__()
        self.input_variant = args.input_variant
        self.encoder = timm.create_model(args.model_name, pretrained=True, num_classes=0, global_pool="")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        targets = {item.strip() for item in args.lora_targets.split(",") if item.strip()}
        self.lora_modules = inject_lora(
            self.encoder,
            last_blocks=args.lora_last_blocks,
            targets=targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        if args.train_final_norm:
            for name, parameter in self.encoder.named_parameters():
                if name.startswith("norm.") or name.startswith("fc_norm."):
                    parameter.requires_grad = True
        if args.init_encoder_state_template:
            state_path = Path(args.init_encoder_state_template.format(fold=fold_id))
            try:
                initial_state = torch.load(state_path, map_location="cpu", weights_only=True)
            except TypeError:
                initial_state = torch.load(state_path, map_location="cpu")
            _, unexpected = self.encoder.load_state_dict(initial_state, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected initial encoder keys from {state_path}: {unexpected[:10]}")
        num_views = 2 if args.input_variant == "whole_plus_crop" else 1
        self.dense_head = DenseRiskHead(
            feature_dim=int(self.encoder.num_features),
            num_views=num_views,
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
            pooling=args.pooling,
            num_classes=num_classes,
        )

    def forward(self, inputs: Any) -> torch.Tensor:
        logits, _ = self.forward_with_embedding(inputs)
        return logits

    def forward_with_embedding(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        views = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
        token_views = [dense_tokens(self.encoder, view) for view in views]
        output = self.dense_head(torch.stack(token_views, dim=1), return_embedding=True)
        assert isinstance(output, tuple)
        return output


def optimizer_for(model: LoRADenseTask7Model, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_parameters = [parameter for parameter in model.dense_head.parameters() if parameter.requires_grad]
    lora_parameters = [
        parameter
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and ("lora_a" in name or "lora_b" in name)
    ]
    norm_parameters = [
        parameter
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "lora_a" not in name and "lora_b" not in name
    ]
    groups = [{"params": head_parameters, "lr": args.head_lr}]
    if lora_parameters:
        groups.append({"params": lora_parameters, "lr": args.lora_lr})
    if norm_parameters:
        groups.append({"params": norm_parameters, "lr": args.lora_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}


def load_trainable_state(model: nn.Module, path: Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    absent_trainable = sorted(name for name in trainable_names if name in missing)
    if absent_trainable:
        raise ValueError(f"Trainable parameters missing from checkpoint: {absent_trainable[:10]}")


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    source_ids: torch.Tensor | None = None,
    hard_negative_mask: torch.Tensor | None = None,
    hard_negative_weight: float = 1.0,
) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    normalized = F.normalize(embeddings.float(), dim=1)
    similarity = normalized @ normalized.transpose(0, 1)
    similarity = similarity / max(float(temperature), 1e-6)
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    nonself = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & nonself
    if source_ids is not None:
        positives = positives & source_ids[:, None].ne(source_ids[None, :])
    positive_count = positives.sum(dim=1)
    valid = positive_count.gt(0)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    denominator_weights = torch.ones_like(similarity)
    if hard_negative_mask is not None:
        if hard_negative_mask.shape != similarity.shape:
            raise ValueError("hard_negative_mask must have the same shape as the similarity matrix")
        denominator_weights = torch.where(
            hard_negative_mask,
            torch.full_like(similarity, max(float(hard_negative_weight), 1.0)),
            denominator_weights,
        )
    denominator = (torch.exp(similarity) * nonself * denominator_weights).sum(dim=1).clamp_min(1e-12)
    log_probability = similarity - torch.log(denominator).unsqueeze(1)
    mean_positive_log_probability = (log_probability * positives).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_positive_log_probability[valid].mean()


def train_epoch(
    model: LoRADenseTask7Model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
    max_batches: int | None,
    contrastive_weight: float,
    contrastive_mode: str,
    contrastive_temperature: float,
    case_to_source_id: dict[str, int],
    case_to_subtype_id: dict[str, int],
    boundary_negative_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_contrastive_loss = 0.0
    total_items = 0
    for batch_idx, (inputs, labels, case_ids, _) in enumerate(loader, start=1):
        inputs = move_to_device(inputs, device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits, embeddings = model.forward_with_embedding(inputs)
            classification_loss = criterion(logits, labels)
            contrastive_loss = embeddings.sum() * 0.0
            if contrastive_weight > 0:
                subtype_ids = torch.tensor(
                    [case_to_subtype_id[str(case_id)] for case_id in case_ids],
                    dtype=torch.long,
                    device=device,
                )
                contrastive_labels = subtype_ids if "subtype" in contrastive_mode else labels
                source_ids = None
                if contrastive_mode.startswith("cross_source"):
                    source_ids = torch.tensor(
                        [case_to_source_id[str(case_id)] for case_id in case_ids],
                        dtype=torch.long,
                        device=device,
                    )
                hard_negative_mask = None
                if contrastive_mode.endswith("b1b2"):
                    b1_id = SUBTYPE_TO_ID["B1"]
                    b2_id = SUBTYPE_TO_ID["B2"]
                    hard_negative_mask = (
                        subtype_ids[:, None].eq(b1_id) & subtype_ids[None, :].eq(b2_id)
                    ) | (
                        subtype_ids[:, None].eq(b2_id) & subtype_ids[None, :].eq(b1_id)
                    )
                contrastive_loss = supervised_contrastive_loss(
                    embeddings,
                    contrastive_labels,
                    contrastive_temperature,
                    source_ids=source_ids,
                    hard_negative_mask=hard_negative_mask,
                    hard_negative_weight=boundary_negative_weight,
                )
            loss = classification_loss + float(contrastive_weight) * contrastive_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * int(labels.size(0))
        total_classification_loss += float(classification_loss.item()) * int(labels.size(0))
        total_contrastive_loss += float(contrastive_loss.item()) * int(labels.size(0))
        total_items += int(labels.size(0))
        if max_batches is not None and batch_idx >= max_batches:
            break
    denominator = max(total_items, 1)
    return {
        "loss": total_loss / denominator,
        "classification_loss": total_classification_loss / denominator,
        "contrastive_loss": total_contrastive_loss / denominator,
    }


def add_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "master_fold_id",
    ]
    available = [column for column in keep if column in metadata.columns]
    return predictions.merge(metadata[available].drop_duplicates("case_id"), on="case_id", how="left")


def domain_metrics(predictions: pd.DataFrame, task: TaskConfig) -> pd.DataFrame:
    rows = [metrics_to_frame("test_oof", "case", "mean", summarize_prediction_frame(predictions, task.class_names))]
    for domain, group in predictions.groupby("source_dataset", dropna=False):
        rows.append(
            metrics_to_frame(
                f"test_oof:{domain}", "case", "mean", summarize_prediction_frame(group, task.class_names)
            )
        )
    old_mask = ~predictions["source_dataset"].astype(str).str.startswith("third_batch", na=False)
    for name, mask in [("old", old_mask), ("third", ~old_mask)]:
        group = predictions[mask]
        if not group.empty:
            rows.append(
                metrics_to_frame(
                    f"test_oof:{name}", "case", "mean", summarize_prediction_frame(group, task.class_names)
                )
            )
    return pd.concat(rows, ignore_index=True)


def canonical_source_values(metadata: pd.DataFrame) -> pd.Series:
    source = metadata["source_dataset"].astype(str)
    return source.mask(source.str.startswith("third_batch", na=False), "third_batch")


def source_group_names(metadata: pd.DataFrame) -> list[str]:
    groups = sorted(canonical_source_values(metadata).unique().tolist())
    if len(groups) < 2:
        raise ValueError(f"source_lodo requires at least two source datasets, found {groups}")
    return groups


def prepare_run_data(
    args: argparse.Namespace,
    task: TaskConfig,
    fold_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if args.split_mode == "fivefold":
        fold_ids = load_available_folds(args.split_csv)
        train_df, val_df, test_df = prepare_fold_data(
            args.registry_csv,
            args.split_csv,
            task,
            fold_id,
            len(fold_ids),
        )
        return train_df, val_df, test_df, ""

    metadata = make_image_df(args.registry_csv, args.split_csv, task)
    groups = source_group_names(metadata)
    if fold_id < 1 or fold_id > len(groups):
        raise ValueError(f"source_lodo fold must be in 1..{len(groups)}, received {fold_id}")
    heldout_source = groups[fold_id - 1]
    val_fold = (fold_id % len(load_available_folds(args.split_csv))) + 1
    source = canonical_source_values(metadata)
    heldout_mask = source.eq(heldout_source)
    val_mask = (~heldout_mask) & metadata["master_fold_id"].eq(val_fold)
    train_mask = (~heldout_mask) & (~val_mask)
    train_df = metadata.loc[train_mask].reset_index(drop=True)
    val_df = metadata.loc[val_mask].reset_index(drop=True)
    test_df = metadata.loc[heldout_mask].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Empty source_lodo partition for fold={fold_id}, heldout={heldout_source}: "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )
    return train_df, val_df, test_df, heldout_source


def sampler_group_keys(image_df: pd.DataFrame, profile: str) -> pd.Series | None:
    if profile == "none":
        return None
    labels = image_df["label_idx"].astype(str)
    if profile == "class_balanced":
        return labels
    source = canonical_source_values(image_df)
    if profile == "domain_label_balanced":
        domain = source.where(~source.str.startswith("third_batch", na=False), "third")
        domain = domain.where(domain.eq("third"), "old")
        return domain + "::" + labels
    if profile == "source_label_balanced":
        return source + "::" + labels
    if profile == "source_subtype_balanced":
        subtype = image_df["task_l6_label"].fillna("unknown").astype(str)
        return source + "::" + subtype
    raise ValueError(f"Unsupported train sampler {profile!r}")


def build_train_dataloader(
    image_df: pd.DataFrame,
    args: argparse.Namespace,
    fold_id: int,
) -> DataLoader:
    previous_profile = os.environ.get("THYMIC_AUG_PROFILE")
    previous_sampler = os.environ.get("THYMIC_TRAIN_SAMPLER")
    os.environ["THYMIC_AUG_PROFILE"] = args.aug_profile
    os.environ.pop("THYMIC_TRAIN_SAMPLER", None)
    try:
        group_keys = sampler_group_keys(image_df, args.train_sampler)
        if group_keys is None:
            return build_dataloader(
                image_df,
                args.input_variant,
                args.image_size,
                True,
                args.batch_size,
                args.num_workers,
            )
        dataset = ThymicImageDataset(
            image_df=image_df,
            input_variant=args.input_variant,
            image_size=args.image_size,
            is_train=True,
        )
        counts = group_keys.value_counts()
        weights = group_keys.map(lambda key: 1.0 / float(counts[key])).to_numpy(dtype=float).copy()
        generator = torch.Generator()
        generator.manual_seed(args.seed + fold_id)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "shuffle": False,
            "sampler": sampler,
            "num_workers": args.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if args.num_workers > 0 and args.max_train_batches is None:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(**loader_kwargs)
    finally:
        if previous_profile is None:
            os.environ.pop("THYMIC_AUG_PROFILE", None)
        else:
            os.environ["THYMIC_AUG_PROFILE"] = previous_profile
        if previous_sampler is None:
            os.environ.pop("THYMIC_TRAIN_SAMPLER", None)
        else:
            os.environ["THYMIC_TRAIN_SAMPLER"] = previous_sampler


def run_fold(args: argparse.Namespace, task: TaskConfig, fold_id: int, output_dir: Path) -> dict[str, Any]:
    set_seed(args.seed + fold_id)
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df, heldout_source = prepare_run_data(args, task, fold_id)
    train_loader = build_train_dataloader(train_df, args, fold_id)
    train_canonical_source = canonical_source_values(train_df)
    source_to_id = {
        source: source_id
        for source_id, source in enumerate(sorted(train_canonical_source.unique().tolist()))
    }
    case_to_source_id = {
        str(case_id): source_to_id[str(source)]
        for case_id, source in zip(train_df["case_id"].astype(str), train_canonical_source)
    }
    unknown_subtypes = sorted(set(train_df["task_l6_label"].astype(str)) - set(SUBTYPE_TO_ID))
    if unknown_subtypes:
        raise ValueError(f"Unsupported Task6 subtype labels: {unknown_subtypes}")
    case_to_subtype_id = {
        str(case_id): SUBTYPE_TO_ID[str(subtype)]
        for case_id, subtype in zip(train_df["case_id"].astype(str), train_df["task_l6_label"].astype(str))
    }
    eval_workers = 0 if args.max_eval_batches is not None else args.num_workers
    val_loader = build_dataloader(val_df, args.input_variant, args.image_size, False, args.batch_size, eval_workers)
    test_loader = build_dataloader(test_df, args.input_variant, args.image_size, False, args.batch_size, eval_workers)

    device = resolve_device(args.device)
    model = LoRADenseTask7Model(args, task.num_classes, fold_id=fold_id).to(device)
    optimizer = optimizer_for(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    weights = build_class_weights(train_df, task.num_classes, device=device) if args.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
    selection_name = resolve_selection_metric(task, args.selection_metric)
    best_metric: float | None = None
    best_epoch = 0
    stale = 0
    history = []

    run_config = vars(args).copy()
    run_config.update(
        {
            "fold_id": fold_id,
            "heldout_source": heldout_source,
            "lora_modules": model.lora_modules,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "train_cases": int(train_df["case_id"].nunique()),
            "val_cases": int(val_df["case_id"].nunique()),
            "test_cases": int(test_df["case_id"].nunique()),
            "train_source_label_counts": {
                str(key): int(value)
                for key, value in (
                    train_canonical_source.astype(str) + "::" + train_df["label_idx"].astype(str)
                ).value_counts().sort_index().items()
            },
        }
    )
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"fold={fold_id} lora_modules={len(model.lora_modules)} "
        f"trainable={run_config['trainable_parameters']:,}/{run_config['total_parameters']:,}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_enabled,
            amp_dtype,
            args.grad_clip,
            args.max_train_batches,
            args.supervised_contrastive_weight,
            args.supervised_contrastive_mode,
            args.supervised_contrastive_temperature,
            case_to_source_id,
            case_to_subtype_id,
            args.boundary_negative_weight,
        )
        validation = evaluate_model(
            model,
            val_loader,
            task,
            device,
            aggregate_methods,
            criterion,
            args.max_eval_batches,
            f"lora fold{fold_id} val e{epoch}",
        )
        scheduler.step()
        selection = float(validation["case_metrics"]["mean"].get(selection_name, float("nan")))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_classification_loss": train_stats["classification_loss"],
                "train_contrastive_loss": train_stats["contrastive_loss"],
                "val_loss": float(validation["loss"]),
                "val_selection_metric": selection,
                "head_lr": float(optimizer.param_groups[0]["lr"]),
                "lora_lr": float(optimizer.param_groups[-1]["lr"]),
            }
        )
        if is_better_metric(selection, best_metric):
            best_metric = selection
            best_epoch = epoch
            stale = 0
            torch.save(trainable_state(model), fold_dir / "best_trainable_state.pt")
        else:
            stale += 1
        print(
            f"fold={fold_id} epoch={epoch} train_loss={train_stats['loss']:.4f} "
            f"train_contrastive={train_stats['contrastive_loss']:.4f} "
            f"val_{selection_name}={format_metric_value(selection)} stale={stale}",
            flush=True,
        )
        if stale >= args.patience:
            break

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    if best_epoch < 1:
        raise RuntimeError("LoRA fine-tuning did not create a valid checkpoint.")
    load_trainable_state(model, fold_dir / "best_trainable_state.pt", device)
    validation = evaluate_model(
        model, val_loader, task, device, aggregate_methods, criterion, args.max_eval_batches, f"lora fold{fold_id} val best"
    )
    test = evaluate_model(
        model, test_loader, task, device, aggregate_methods, criterion, args.max_eval_batches, f"lora fold{fold_id} test best"
    )
    save_evaluation_outputs(fold_dir, "val", validation, aggregate_methods)
    save_evaluation_outputs(fold_dir, "test", test, aggregate_methods)
    summary = {
        "fold_id": fold_id,
        "heldout_source": heldout_source,
        "best_epoch": best_epoch,
        "selection_metric": selection_name,
        "best_val_selection_metric": best_metric,
        "test_case_mean": test["case_metrics"]["mean"],
    }
    write_json(fold_dir / "fold_summary.json", summary)
    print(
        f"fold={fold_id} best_epoch={best_epoch} test: "
        f"{metric_summary_text(test['case_metrics']['mean'], [task.primary_metric, 'accuracy', 'balanced_accuracy'])}",
        flush=True,
    )
    return {
        "summary": {
            "fold_id": fold_id,
            "heldout_source": heldout_source,
            "best_epoch": best_epoch,
            "best_val_selection_metric": best_metric,
            "test_case_mean_primary_metric": float(test["case_metrics"]["mean"].get(task.primary_metric, float("nan"))),
            "test_case_mean_accuracy": float(test["case_metrics"]["mean"].get("accuracy", float("nan"))),
            "test_case_mean_balanced_accuracy": float(
                test["case_metrics"]["mean"].get("balanced_accuracy", float("nan"))
            ),
        },
        "test_images": test["image_predictions"],
        "test_cases": test["case_predictions"],
    }


def run_all(args: argparse.Namespace, task: TaskConfig, output_dir: Path) -> None:
    metadata = make_image_df(args.registry_csv, args.split_csv, task)
    if args.split_mode == "source_lodo":
        fold_ids = list(range(1, len(source_group_names(metadata)) + 1))
    else:
        fold_ids = load_available_folds(args.split_csv)
    summaries = []
    images = []
    cases: dict[str, list[pd.DataFrame]] = {
        item.strip(): [] for item in args.aggregate_methods.split(",") if item.strip()
    }
    for fold_id in fold_ids:
        result = run_fold(args, task, fold_id, output_dir)
        summaries.append(result["summary"])
        image_predictions = result["test_images"].copy()
        image_predictions["fold_id"] = fold_id
        images.append(image_predictions)
        for method, frame in result["test_cases"].items():
            fold_frame = frame.copy()
            fold_frame["fold_id"] = fold_id
            cases[method].append(fold_frame)
    pd.DataFrame(summaries).to_csv(output_dir / "cv_fold_summary.csv", index=False, encoding="utf-8-sig")
    image_frame = add_metadata(pd.concat(images, ignore_index=True), metadata)
    image_frame.to_csv(output_dir / "oof_image_predictions.csv", index=False, encoding="utf-8-sig")
    metric_frames = [
        metrics_to_frame("test_oof", "image", "none", summarize_prediction_frame(image_frame, task.class_names))
    ]
    for method, frames in cases.items():
        case_frame = add_metadata(pd.concat(frames, ignore_index=True), metadata)
        case_frame.to_csv(output_dir / f"oof_case_predictions_{method}.csv", index=False, encoding="utf-8-sig")
        metric_frames.append(
            metrics_to_frame("test_oof", "case", method, summarize_prediction_frame(case_frame, task.class_names))
        )
        if method == "mean":
            domain_metrics(case_frame, task).to_csv(
                output_dir / "oof_domain_metrics_mean.csv", index=False, encoding="utf-8-sig"
            )
    pd.concat(metric_frames, ignore_index=True).to_csv(output_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")


def audit_partitions(args: argparse.Namespace, task: TaskConfig, output_dir: Path) -> None:
    metadata = make_image_df(args.registry_csv, args.split_csv, task)
    if args.split_mode == "source_lodo":
        fold_ids = list(range(1, len(source_group_names(metadata)) + 1))
    else:
        fold_ids = load_available_folds(args.split_csv)
    if args.fold != "all":
        fold_ids = [int(args.fold)]
    rows = []
    for fold_id in fold_ids:
        train_df, val_df, test_df, heldout_source = prepare_run_data(args, task, fold_id)
        for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            counts = frame.groupby(["source_dataset", "label_idx"], dropna=False).size()
            for (source, label_idx), count in counts.items():
                rows.append(
                    {
                        "fold_id": fold_id,
                        "heldout_source": heldout_source,
                        "split": split_name,
                        "source_dataset": str(source),
                        "label_idx": int(label_idx),
                        "image_count": int(count),
                        "case_count": int(
                            frame.loc[
                                frame["source_dataset"].astype(str).eq(str(source))
                                & frame["label_idx"].eq(label_idx),
                                "case_id",
                            ].nunique()
                        ),
                    }
                )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "partition_audit.csv", index=False, encoding="utf-8-sig")
    print(audit.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))
    if args.audit_partitions_only:
        audit_partitions(args, task, output_dir)
        return
    if args.fold == "all":
        run_all(args, task, output_dir)
    else:
        run_fold(args, task, int(args.fold), output_dir)


if __name__ == "__main__":
    main()
