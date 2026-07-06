from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thymic_baseline.config import DEFAULT_IMAGE_SIZE, DEFAULT_RANDOM_SEED, get_task
from thymic_baseline.data import build_transform
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import (
    index_image_files,
    load_registry,
    load_split_assignments,
    merge_registry_with_splits,
    split_image_filenames,
    subset_by_fold,
)
from thymic_baseline.train import (
    build_class_weights,
    format_metric_value,
    is_better_metric,
    load_available_folds,
    metric_summary_text,
    resolve_device,
    save_evaluation_outputs,
    write_json,
)


SUPPORTED_VARIANTS = ("whole", "crop")
DIFFICULTY_CLASSES = ("easy", "medium", "hard_salvage_teacher", "hard_core")
GROSS_BUCKET_CLASSES = (
    "low_with_protective_gross",
    "high_with_invasive_or_complex_gross",
    "high_truth_with_bland_gross",
    "low_truth_with_highrisk_like_gross",
    "gross_trap_do_not_overtrust",
)
HEMONEC_CLASSES = ("none", "mild", "marked")
IRREGULARITY_CLASSES = ("low", "high")
CONFOUND_CLASSES = ("none", "A_AB", "B1", "B2", "TC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint Task6+Task7 DINOv2 fine-tuning.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--input-variant", default="whole", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-scope", default="last2_blocks", choices=("head_only", "last_block", "last2_blocks", "full"))
    parser.add_argument("--head-type", default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--aux-weight", type=float, default=0.5, help="Weight on Task6 auxiliary loss.")
    parser.add_argument("--task7-curriculum-csv", default=None, help="Optional case-level curriculum table for hard-core auxiliary heads.")
    parser.add_argument("--task7-gross-aux-csv", default=None, help="Optional case-level gross cue table for auxiliary heads.")
    parser.add_argument("--hard-core-aux-weight", type=float, default=0.0, help="Weight for binary hard_core auxiliary loss.")
    parser.add_argument("--difficulty-aux-weight", type=float, default=0.0, help="Weight for 4-way difficulty auxiliary loss.")
    parser.add_argument("--gross-bucket-aux-weight", type=float, default=0.0, help="Weight for gross bucket auxiliary loss.")
    parser.add_argument("--gross-cue-aux-weight", type=float, default=0.0, help="Total weight distributed over selected gross cue auxiliary losses.")
    parser.add_argument("--hard-core-highrisk-primary-weight", type=float, default=1.0, help="Primary Task7 loss multiplier for hard_core high-risk cases.")
    parser.add_argument("--hard-core-lowrisk-primary-weight", type=float, default=1.0, help="Primary Task7 loss multiplier for hard_core low-risk cases.")
    parser.add_argument("--hard-salvage-primary-weight", type=float, default=1.0, help="Primary Task7 loss multiplier for hard_salvage_teacher cases.")
    parser.add_argument("--medium-primary-weight", type=float, default=1.0, help="Primary Task7 loss multiplier for medium cases.")
    parser.add_argument(
        "--gross-cue-heads",
        default="round_smooth,multinodular,hemonec,irregularity,confound",
        help="Comma separated gross cue heads: pale_uniform,round_smooth,microcystic,multinodular,hemonec,irregularity,confound.",
    )
    parser.add_argument(
        "--selection-metric",
        default="auc",
        choices=("auc", "accuracy", "balanced_accuracy", "f1"),
        help="Task7 validation case metric used to pick best checkpoint.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Task67Dataset(Dataset):
    def __init__(
        self,
        image_df: pd.DataFrame,
        input_variant: str,
        image_size: int,
        is_train: bool,
        aux_columns: list[str] | None = None,
    ) -> None:
        self.image_df = image_df.reset_index(drop=True)
        self.input_variant = input_variant
        self.aux_columns = aux_columns or []
        self.transform = build_transform(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.image_df)

    def __getitem__(self, index: int):
        row = self.image_df.iloc[index]
        from PIL import Image  # local import to keep startup light
        from thymic_baseline.cropping import extract_specimen_crop

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")

        if self.input_variant == "whole":
            inputs = self.transform(image)
        elif self.input_variant == "crop":
            inputs = self.transform(extract_specimen_crop(image))
        else:
            raise ValueError(f"Unsupported input variant: {self.input_variant}")

        label6 = torch.tensor(int(row["label6_idx"]), dtype=torch.long)
        label7 = torch.tensor(int(row["label7_idx"]), dtype=torch.long)
        sample_weight = torch.tensor(float(row.get("primary_sample_weight", 1.0)), dtype=torch.float32)
        aux_targets = {
            column: torch.tensor(int(row[column]), dtype=torch.long)
            for column in self.aux_columns
            if column in row
        }
        return inputs, label6, label7, str(row["case_id"]), str(row["image_name"]), aux_targets, sample_weight


def build_multitask_registry(registry_csv: str, split_csv: str, images_root: str) -> pd.DataFrame:
    task6 = get_task("task6_sixclass")
    task7 = get_task("task7_lowrisk_vs_highrisk_tc")
    class_to_idx6 = {name: idx for idx, name in enumerate(task6.class_names)}
    class_to_idx7 = {name: idx for idx, name in enumerate(task7.class_names)}

    registry = load_registry(registry_csv)
    split_df = load_split_assignments(split_csv)
    merged = merge_registry_with_splits(registry, split_df)
    filtered = merged[
        merged[task6.label_column].isin(task6.class_names)
        & merged[task7.label_column].isin(task7.class_names)
    ].copy()
    if filtered.empty:
        raise ValueError("No rows remain after filtering for joint Task6+Task7.")

    image_index = index_image_files(images_root)
    rows: list[dict[str, Any]] = []
    for row in filtered.to_dict(orient="records"):
        for image_name in split_image_filenames(row["image_filenames"]):
            image_path = image_index.get(image_name)
            if image_path is None:
                raise FileNotFoundError(f"Image '{image_name}' not found under {images_root}.")
            rows.append(
                {
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "master_fold_id": int(row["master_fold_id"]),
                    "label6_name": row[task6.label_column],
                    "label6_idx": class_to_idx6[row[task6.label_column]],
                    "label7_name": row[task7.label_column],
                    "label7_idx": class_to_idx7[row[task7.label_column]],
                    "who_type_raw": row["who_type_raw"],
                    "source_case_folder": row["source_case_folder"],
                }
            )

    image_df = pd.DataFrame(rows)
    if image_df.empty:
        raise ValueError("No image rows were created for joint Task6+Task7.")
    return image_df.reset_index(drop=True)


def map_series_to_idx(series: pd.Series, classes: tuple[str, ...]) -> pd.Series:
    mapping = {name: idx for idx, name in enumerate(classes)}
    return series.astype(str).map(mapping).fillna(-100).astype(int)


def add_task7_auxiliary_columns(
    image_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, float]]:
    image_df = image_df.copy()
    image_df["primary_sample_weight"] = 1.0
    extra_head_dims: dict[str, int] = {}
    extra_head_weights: dict[str, float] = {}

    need_curriculum = (
        args.hard_core_aux_weight > 0
        or args.difficulty_aux_weight > 0
        or args.hard_core_highrisk_primary_weight != 1.0
        or args.hard_core_lowrisk_primary_weight != 1.0
        or args.hard_salvage_primary_weight != 1.0
        or args.medium_primary_weight != 1.0
    )
    if need_curriculum:
        if not args.task7_curriculum_csv:
            raise ValueError("--task7-curriculum-csv is required when curriculum auxiliary weights are positive.")
        curriculum_df = pd.read_csv(args.task7_curriculum_csv, dtype={"case_id": str})
        if "difficulty_fine" not in curriculum_df.columns:
            raise KeyError("Curriculum CSV missing column: difficulty_fine")
        curriculum_df = curriculum_df[["case_id", "difficulty_fine"]].drop_duplicates("case_id")
        image_df = image_df.merge(curriculum_df, on="case_id", how="left")
        hard_core_mask = image_df["difficulty_fine"].astype(str) == "hard_core"
        hard_salvage_mask = image_df["difficulty_fine"].astype(str) == "hard_salvage_teacher"
        medium_mask = image_df["difficulty_fine"].astype(str) == "medium"
        image_df.loc[hard_core_mask & image_df["label7_idx"].eq(1), "primary_sample_weight"] = float(args.hard_core_highrisk_primary_weight)
        image_df.loc[hard_core_mask & image_df["label7_idx"].eq(0), "primary_sample_weight"] = float(args.hard_core_lowrisk_primary_weight)
        image_df.loc[hard_salvage_mask, "primary_sample_weight"] = float(args.hard_salvage_primary_weight)
        image_df.loc[medium_mask, "primary_sample_weight"] = float(args.medium_primary_weight)
        image_df["aux_hard_core"] = (image_df["difficulty_fine"].astype(str) == "hard_core").astype(int)
        image_df["aux_difficulty"] = map_series_to_idx(image_df["difficulty_fine"], DIFFICULTY_CLASSES)
        if args.hard_core_aux_weight > 0:
            extra_head_dims["aux_hard_core"] = 2
            extra_head_weights["aux_hard_core"] = float(args.hard_core_aux_weight)
        if args.difficulty_aux_weight > 0:
            extra_head_dims["aux_difficulty"] = len(DIFFICULTY_CLASSES)
            extra_head_weights["aux_difficulty"] = float(args.difficulty_aux_weight)

    need_gross = args.gross_bucket_aux_weight > 0 or args.gross_cue_aux_weight > 0
    if need_gross:
        if not args.task7_gross_aux_csv:
            raise ValueError("--task7-gross-aux-csv is required when gross auxiliary weights are positive.")
        gross_df = pd.read_csv(args.task7_gross_aux_csv, dtype={"case_id": str})
        gross_keep = [
            "case_id",
            "antishortcut_bucket",
            "exp_manual_pale_uniform",
            "exp_manual_round_smooth",
            "exp_manual_microcystic",
            "exp_manual_multinodular",
            "exp_manual_hemonec",
            "exp_manual_irregularity",
            "exp_manual_confound_target",
        ]
        gross_keep = [col for col in gross_keep if col in gross_df.columns]
        gross_df = gross_df[gross_keep].drop_duplicates("case_id")
        image_df = image_df.merge(gross_df, on="case_id", how="left")
        if args.gross_bucket_aux_weight > 0:
            image_df["aux_gross_bucket"] = map_series_to_idx(image_df["antishortcut_bucket"], GROSS_BUCKET_CLASSES)
            extra_head_dims["aux_gross_bucket"] = len(GROSS_BUCKET_CLASSES)
            extra_head_weights["aux_gross_bucket"] = float(args.gross_bucket_aux_weight)

        cue_specs = {
            "pale_uniform": ("exp_manual_pale_uniform", "aux_gross_pale_uniform", ("no", "yes")),
            "round_smooth": ("exp_manual_round_smooth", "aux_gross_round_smooth", ("no", "yes")),
            "microcystic": ("exp_manual_microcystic", "aux_gross_microcystic", ("no", "yes")),
            "multinodular": ("exp_manual_multinodular", "aux_gross_multinodular", ("no", "yes")),
            "hemonec": ("exp_manual_hemonec", "aux_gross_hemonec", HEMONEC_CLASSES),
            "irregularity": ("exp_manual_irregularity", "aux_gross_irregularity", IRREGULARITY_CLASSES),
            "confound": ("exp_manual_confound_target", "aux_gross_confound", CONFOUND_CLASSES),
        }
        requested_cues = [item.strip() for item in args.gross_cue_heads.split(",") if item.strip()]
        active_cues = [name for name in requested_cues if name in cue_specs]
        per_cue_weight = float(args.gross_cue_aux_weight) / max(len(active_cues), 1)
        for cue_name in active_cues:
            source_col, aux_col, classes = cue_specs[cue_name]
            if source_col not in image_df.columns:
                continue
            image_df[aux_col] = map_series_to_idx(image_df[source_col], classes)
            extra_head_dims[aux_col] = len(classes)
            extra_head_weights[aux_col] = per_cue_weight

    for column in extra_head_dims:
        if column not in image_df.columns:
            image_df[column] = -100
        image_df[column] = image_df[column].fillna(-100).astype(int)
    return image_df, extra_head_dims, extra_head_weights


def build_dataloader(
    image_df: pd.DataFrame,
    input_variant: str,
    image_size: int,
    is_train: bool,
    batch_size: int,
    num_workers: int,
    aux_columns: list[str] | None = None,
) -> DataLoader:
    dataset = Task67Dataset(
        image_df=image_df,
        input_variant=input_variant,
        image_size=image_size,
        is_train=is_train,
        aux_columns=aux_columns,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": is_train,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


class DINOv2JointModel(nn.Module):
    def __init__(
        self,
        repo_dir: Path,
        model_name: str,
        num_classes6: int,
        num_classes7: int,
        dropout: float,
        head_type: str,
        hidden_dim: int,
        extra_head_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = torch.hub.load(str(repo_dir), model_name, source="local", pretrained=True)
        feature_dim = int(getattr(self.backbone, "embed_dim", 384))
        self.head6 = self._build_head(feature_dim, num_classes6, dropout, head_type, hidden_dim)
        self.head7 = self._build_head(feature_dim, num_classes7, dropout, head_type, hidden_dim)
        self.extra_heads = nn.ModuleDict(
            {
                name: self._build_head(feature_dim, int(num_classes), dropout, head_type, hidden_dim)
                for name, num_classes in (extra_head_dims or {}).items()
            }
        )

    @staticmethod
    def _build_head(feature_dim: int, num_classes: int, dropout: float, head_type: str, hidden_dim: int) -> nn.Module:
        if head_type == "linear":
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self.backbone.forward_features(inputs)
        cls_token = outputs["x_norm_clstoken"]
        extra_logits = {name: head(cls_token) for name, head in self.extra_heads.items()}
        return self.head6(cls_token), self.head7(cls_token), extra_logits


def load_init_checkpoint(model: DINOv2JointModel, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint}")
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded init checkpoint: {checkpoint} | missing_keys={len(missing)} | unexpected_keys={len(unexpected)}",
        flush=True,
    )


def configure_trainable_parameters(model: DINOv2JointModel, tune_scope: str) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False

    if tune_scope == "head_only":
        pass
    elif tune_scope == "last_block":
        for name, param in model.backbone.named_parameters():
            if name.startswith("blocks.11") or name.startswith("norm"):
                param.requires_grad = True
    elif tune_scope == "last2_blocks":
        for name, param in model.backbone.named_parameters():
            if name.startswith("blocks.10") or name.startswith("blocks.11") or name.startswith("norm"):
                param.requires_grad = True
    elif tune_scope == "full":
        for param in model.backbone.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported tune scope: {tune_scope}")

    for head in [model.head6, model.head7, *model.extra_heads.values()]:
        for param in head.parameters():
            param.requires_grad = True


def unpack_batch(batch):
    if len(batch) == 7:
        inputs, labels6, labels7, case_ids, image_names, aux_targets, sample_weights = batch
    elif len(batch) == 6:
        inputs, labels6, labels7, case_ids, image_names, aux_targets = batch
        sample_weights = None
    else:
        inputs, labels6, labels7, case_ids, image_names = batch
        aux_targets = {}
        sample_weights = None
    return inputs, labels6, labels7, case_ids, image_names, aux_targets, sample_weights


def build_optimizer(model: DINOv2JointModel, head_lr: float, backbone_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params = [param for name, param in model.named_parameters() if param.requires_grad and not name.startswith("backbone.")]
    backbone_params = [param for name, param in model.named_parameters() if param.requires_grad and name.startswith("backbone.")]
    param_groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_one_epoch(
    model: DINOv2JointModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion6: nn.Module,
    criterion7: nn.Module,
    device: torch.device,
    grad_clip: float,
    aux_weight: float,
    extra_criteria: dict[str, nn.Module] | None = None,
    extra_head_weights: dict[str, float] | None = None,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    extra_criteria = extra_criteria or {}
    extra_head_weights = extra_head_weights or {}
    for batch_idx, batch in enumerate(loader, start=1):
        inputs, labels6, labels7, _, _, aux_targets, sample_weights = unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        labels6 = labels6.to(device, non_blocking=True)
        labels7 = labels7.to(device, non_blocking=True)
        if sample_weights is None:
            sample_weights = torch.ones_like(labels7, dtype=torch.float32)
        sample_weights = sample_weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits6, logits7, extra_logits = model(inputs)
        loss6 = criterion6(logits6, labels6)
        loss7_per_item = F.cross_entropy(
            logits7,
            labels7,
            weight=criterion7.weight,
            label_smoothing=float(getattr(criterion7, "label_smoothing", 0.0)),
            reduction="none",
        )
        loss7 = (loss7_per_item * sample_weights).mean()
        loss = aux_weight * loss6 + loss7
        for name, logits in extra_logits.items():
            if name not in extra_criteria or name not in aux_targets:
                continue
            targets = aux_targets[name].to(device, non_blocking=True)
            if not bool((targets != -100).any()):
                continue
            loss = loss + float(extra_head_weights.get(name, 0.0)) * extra_criteria[name](logits, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = labels7.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if max_batches is not None and batch_idx >= max_batches:
            break
    return total_loss / max(total_items, 1)


def evaluate_head(
    model: DINOv2JointModel,
    loader: DataLoader,
    task_key: str,
    device: torch.device,
    aggregate_methods: list[str],
    criterion6: nn.Module,
    criterion7: nn.Module,
    max_batches: int | None = None,
) -> dict[str, Any]:
    task = get_task(task_key)
    prob_cols = [f"prob_{name}" for name in task.class_names]
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_items = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            inputs, labels6, labels7, case_ids, image_names, _, _ = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            labels6 = labels6.to(device, non_blocking=True)
            labels7 = labels7.to(device, non_blocking=True)
            logits6, logits7, _ = model(inputs)
            if task_key == "task6_sixclass":
                logits = logits6
                labels = labels6
                loss = criterion6(logits6, labels6)
            else:
                logits = logits7
                labels = labels7
                loss = criterion7(logits7, labels7)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            label_values = labels.cpu().numpy()
            for row_idx in range(batch_size):
                row = {
                    "case_id": str(case_ids[row_idx]),
                    "image_name": str(image_names[row_idx]),
                    "label_idx": int(label_values[row_idx]),
                    "pred_idx": int(preds[row_idx]),
                }
                for col_name, value in zip(prob_cols, probs[row_idx]):
                    row[col_name] = float(value)
                rows.append(row)
            if max_batches is not None and batch_idx >= max_batches:
                break

    prediction_df = pd.DataFrame(rows)
    image_metrics = summarize_prediction_frame(prediction_df, task.class_names)
    case_predictions: dict[str, pd.DataFrame] = {}
    case_metrics: dict[str, dict[str, float]] = {}
    for method in aggregate_methods:
        case_df = aggregate_case_predictions(prediction_df, task.class_names, method=method)
        case_predictions[method] = case_df
        case_metrics[method] = summarize_prediction_frame(case_df, task.class_names)
    return {
        "loss": total_loss / max(total_items, 1),
        "image_predictions": prediction_df,
        "image_metrics": image_metrics,
        "case_predictions": case_predictions,
        "case_metrics": case_metrics,
    }


def get_selection_metric_value(val_task7: dict[str, Any], selection_metric: str) -> float:
    case_mean = val_task7["case_metrics"]["mean"]
    return float(case_mean.get(selection_metric, float("nan")))


def run_single_fold(
    args: argparse.Namespace,
    fold_id: int,
    root_output_dir: Path,
    full_df: pd.DataFrame,
    extra_head_dims: dict[str, int],
    extra_head_weights: dict[str, float],
) -> dict[str, Any]:
    fold_dir = root_output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    available_folds = load_available_folds(args.split_csv)
    train_df, val_df, test_df = subset_by_fold(full_df, test_fold=fold_id, num_folds=len(available_folds))

    aux_columns = list(extra_head_dims)
    train_loader = build_dataloader(train_df, args.input_variant, args.image_size, True, args.batch_size, args.num_workers, aux_columns)
    val_loader = build_dataloader(val_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers, aux_columns)
    test_loader = build_dataloader(test_df, args.input_variant, args.image_size, False, args.batch_size, args.num_workers, aux_columns)

    device = resolve_device(args.device)
    model = DINOv2JointModel(
        repo_dir=Path(args.repo_dir),
        model_name=args.model_name,
        num_classes6=get_task("task6_sixclass").num_classes,
        num_classes7=get_task("task7_lowrisk_vs_highrisk_tc").num_classes,
        dropout=args.dropout,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        extra_head_dims=extra_head_dims,
    )
    load_init_checkpoint(model, args.init_checkpoint)
    configure_trainable_parameters(model, args.tune_scope)
    model = model.to(device)

    weights6 = build_class_weights(train_df.rename(columns={"label6_idx": "label_idx"}), get_task("task6_sixclass").num_classes, device=device) if args.class_weighting else None
    weights7 = build_class_weights(train_df.rename(columns={"label7_idx": "label_idx"}), get_task("task7_lowrisk_vs_highrisk_tc").num_classes, device=device) if args.class_weighting else None
    criterion6 = nn.CrossEntropyLoss(weight=weights6, label_smoothing=args.label_smoothing)
    criterion7 = nn.CrossEntropyLoss(weight=weights7, label_smoothing=args.label_smoothing)
    extra_criteria = {name: nn.CrossEntropyLoss(ignore_index=-100) for name in extra_head_dims}
    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

    best_selection_metric: float | None = None
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []

    run_config = {
        "task_primary": "task7_lowrisk_vs_highrisk_tc",
        "task_aux": "task6_sixclass",
        "input_variant": args.input_variant,
        "model_name": args.model_name,
        "tune_scope": args.tune_scope,
        "head_type": args.head_type,
        "aux_weight": float(args.aux_weight),
        "extra_head_dims": extra_head_dims,
        "extra_head_weights": extra_head_weights,
        "primary_sample_weight_summary": {
            "min": float(train_df["primary_sample_weight"].min()),
            "max": float(train_df["primary_sample_weight"].max()),
            "mean": float(train_df["primary_sample_weight"].mean()),
        },
        "selection_metric": args.selection_metric,
        "fold_id": fold_id,
        "train_cases": int(train_df["case_id"].nunique()),
        "val_cases": int(val_df["case_id"].nunique()),
        "test_cases": int(test_df["case_id"].nunique()),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "test_images": int(len(test_df)),
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== DINO Joint Fold {fold_id}/{len(available_folds)} | model={args.model_name} | tune_scope={args.tune_scope} | "
        f"head={args.head_type} | aux_weight={args.aux_weight} | selection_metric={args.selection_metric} ===",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion6=criterion6,
            criterion7=criterion7,
            device=device,
            grad_clip=args.grad_clip,
            aux_weight=args.aux_weight,
            extra_criteria=extra_criteria,
            extra_head_weights=extra_head_weights,
            max_batches=args.max_train_batches,
        )
        val_task6 = evaluate_head(model, val_loader, "task6_sixclass", device, aggregate_methods, criterion6, criterion7, args.max_eval_batches)
        val_task7 = evaluate_head(model, val_loader, "task7_lowrisk_vs_highrisk_tc", device, aggregate_methods, criterion6, criterion7, args.max_eval_batches)
        scheduler.step()

        val_auc7 = float(val_task7["case_metrics"]["mean"].get("auc", float("nan")))
        val_acc7 = float(val_task7["case_metrics"]["mean"].get("accuracy", float("nan")))
        val_bacc7 = float(val_task7["case_metrics"]["mean"].get("balanced_accuracy", float("nan")))
        val_f17 = float(val_task7["case_metrics"]["mean"].get("f1", float("nan")))
        val_f16 = float(val_task6["case_metrics"]["mean"].get("macro_f1", float("nan")))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc_task7": val_auc7,
                "val_acc_task7": val_acc7,
                "val_bacc_task7": val_bacc7,
                "val_f1_task7": val_f17,
                "val_macro_f1_task6": val_f16,
                "head_lr": float(optimizer.param_groups[0]["lr"]),
                "backbone_lr": float(optimizer.param_groups[-1]["lr"]),
            }
        )
        selection_value = get_selection_metric_value(val_task7, args.selection_metric)
        if is_better_metric(selection_value, best_selection_metric):
            best_selection_metric = selection_value
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
            best_flag = "yes"
        else:
            stale_epochs += 1
            best_flag = "no"

        print(
            f"[Fold {fold_id}] epoch={epoch} train_loss={train_loss:.4f} | "
            f"val_task7: {metric_summary_text(val_task7['case_metrics']['mean'], ['auc', 'accuracy', 'balanced_accuracy', 'f1'])} | "
            f"val_task6: {metric_summary_text(val_task6['case_metrics']['mean'], ['macro_f1', 'macro_auc'])} | "
            f"select={args.selection_metric}={format_metric_value(selection_value)} | "
            f"best_updated={best_flag} stale={stale_epochs}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch} (best_epoch={best_epoch}).", flush=True)
            break

    pd.DataFrame(history_rows).to_csv(fold_dir / "history.csv", index=False)
    if best_epoch < 1:
        raise RuntimeError("Joint training finished without a valid checkpoint.")

    try:
        best_state = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        best_state = torch.load(fold_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state)

    val_task6 = evaluate_head(model, val_loader, "task6_sixclass", device, aggregate_methods, criterion6, criterion7, args.max_eval_batches)
    val_task7 = evaluate_head(model, val_loader, "task7_lowrisk_vs_highrisk_tc", device, aggregate_methods, criterion6, criterion7, args.max_eval_batches)
    test_task6 = evaluate_head(model, test_loader, "task6_sixclass", device, aggregate_methods, criterion6, criterion7, args.max_eval_batches)
    test_task7 = evaluate_head(model, test_loader, "task7_lowrisk_vs_highrisk_tc", device, aggregate_methods, criterion6, criterion7, args.max_eval_batches)

    task6_dir = fold_dir / "task6_aux"
    task7_dir = fold_dir / "task7_primary"
    task6_dir.mkdir(exist_ok=True)
    task7_dir.mkdir(exist_ok=True)
    save_evaluation_outputs(task6_dir, "val", val_task6, aggregate_methods)
    save_evaluation_outputs(task6_dir, "test", test_task6, aggregate_methods)
    save_evaluation_outputs(task7_dir, "val", val_task7, aggregate_methods)
    save_evaluation_outputs(task7_dir, "test", test_task7, aggregate_methods)

    summary = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        f"best_val_{args.selection_metric}_task7": best_selection_metric,
        "val_task7_case_metrics_mean": val_task7["case_metrics"]["mean"],
        "test_task7_case_metrics_mean": test_task7["case_metrics"]["mean"],
        "val_task6_case_metrics_mean": val_task6["case_metrics"]["mean"],
        "test_task6_case_metrics_mean": test_task6["case_metrics"]["mean"],
        "aggregate_methods": aggregate_methods,
    }
    write_json(fold_dir / "fold_summary.json", summary)
    print(
        f"[Fold {fold_id}] finished | best_epoch={best_epoch} | "
        f"test_task7={metric_summary_text(test_task7['case_metrics']['mean'], ['auc', 'accuracy', 'balanced_accuracy', 'f1'])} | "
        f"test_task6={metric_summary_text(test_task6['case_metrics']['mean'], ['macro_f1', 'macro_auc'])}",
        flush=True,
    )
    return summary


def summarize_cv(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for summary in summaries:
        row = {"fold_id": summary["fold_id"], "best_epoch": summary["best_epoch"]}
        for key, value in summary["test_task7_case_metrics_mean"].items():
            row[f"task7_{key}"] = value
        for key, value in summary["test_task6_case_metrics_mean"].items():
            row[f"task6_{key}"] = value
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)
    df.to_csv(output_dir / "cv_fold_summary.csv", index=False)

    mean_row = {}
    for column in df.columns:
        if column in {"fold_id", "best_epoch"}:
            continue
        mean_row[column] = float(df[column].mean())
    write_json(output_dir / "cv_mean_metrics.json", mean_row)
    print(f"\n=== CV mean task7 === {metric_summary_text({k.replace('task7_', ''): v for k, v in mean_row.items() if k.startswith('task7_')}, ['auc', 'accuracy', 'balanced_accuracy', 'f1'])}", flush=True)
    print(f"=== CV mean task6 === {metric_summary_text({k.replace('task6_', ''): v for k, v in mean_row.items() if k.startswith('task6_')}, ['macro_f1', 'macro_auc'])}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = build_multitask_registry(args.registry_csv, args.split_csv, args.images_root)
    full_df, extra_head_dims, extra_head_weights = add_task7_auxiliary_columns(full_df, args)
    if extra_head_dims:
        full_df[["case_id", "image_name", "primary_sample_weight", *extra_head_dims.keys()]].to_csv(output_dir / "task7_auxiliary_image_labels.csv", index=False)
    elif "primary_sample_weight" in full_df.columns and float(full_df["primary_sample_weight"].max()) != 1.0:
        full_df[["case_id", "image_name", "primary_sample_weight"]].to_csv(output_dir / "task7_auxiliary_image_labels.csv", index=False)
    available_folds = load_available_folds(args.split_csv)
    if args.fold == "all":
        fold_ids = available_folds
    else:
        fold_ids = [int(args.fold)]
        if fold_ids[0] not in available_folds:
            raise ValueError(f"Fold {fold_ids[0]} not found in split assignments.")

    summaries = []
    for fold_id in fold_ids:
        summaries.append(run_single_fold(args, fold_id, output_dir, full_df, extra_head_dims, extra_head_weights))

    if len(summaries) > 1:
        summarize_cv(summaries, output_dir)


if __name__ == "__main__":
    main()
