from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from run_dinov2_frozen_probe import (
    aggregate_run_outputs,
    build_prediction_df,
    extract_dataset_features,
    format_metric_value,
    load_available_folds,
    load_dinov2_model,
    save_split_outputs,
    set_seed,
    trim_image_df,
    write_json,
)
from run_task56_dinov2_probe import TASK56, get_task, load_task56_image_df
from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


DEFAULT_DINO_IMAGE_SIZE = 518


DEFAULT_AUX_FIELDS = ("boundary_axis", "lowrisk", "bgroup", "tc")

FIXED_FIELD_CATEGORIES: dict[str, list[str]] = {
    "boundary_axis": ["A_AB", "B1_B2", "B2_B3", "B2_TC", "TC_B_group", "none"],
    "lowrisk": ["low", "medium", "high"],
    "bgroup": ["low", "medium", "high"],
    "tc": ["low", "medium", "high"],
    "manual_pale_uniform": ["no", "yes"],
    "manual_round_smooth": ["no", "yes"],
    "manual_microcystic": ["no", "yes"],
    "manual_multinodular": ["no", "yes"],
    "manual_hemonec": ["none", "mild", "marked"],
    "manual_irregularity": ["low", "high"],
    "manual_confound_target": ["none", "A_AB", "B1", "B2", "B3", "TC", "B_group"],
    "manual_view_limit": ["no", "yes"],
}

FIELD_TO_COLUMN: dict[str, str] = {
    "boundary_axis": "exp_round1_boundary_axis",
    "lowrisk": "exp_round1_lowrisk_impression",
    "bgroup": "exp_round1_bgroup_impression",
    "tc": "exp_round1_tc_impression",
    "manual_pale_uniform": "exp_manual_pale_uniform",
    "manual_round_smooth": "exp_manual_round_smooth",
    "manual_microcystic": "exp_manual_microcystic",
    "manual_multinodular": "exp_manual_multinodular",
    "manual_hemonec": "exp_manual_hemonec",
    "manual_irregularity": "exp_manual_irregularity",
    "manual_confound_target": "exp_manual_confound_target",
    "manual_view_limit": "exp_manual_view_limit",
}

BOUNDARY_AXIS_NORMALIZATION = {
    "A_AB": "A_AB",
    "B1_B2": "B1_B2",
    "B2_B3": "B2_B3",
    "B2_TC": "B2_TC",
    "TC_B_group": "TC_B_group",
    "none": "none",
    "A 与 AB 边界": "A_AB",
    "B1 与 B2 边界": "B1_B2",
    "B2 与 B3 边界": "B2_B3",
    "B2 与 TC 边界": "B2_TC",
    "TC 与 B 组边界": "TC_B_group",
    "无明显单一边界": "none",
}


@dataclass(frozen=True)
class AuxFieldSpec:
    key: str
    column: str
    categories: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.categories)


class MultiTaskProbeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_main_classes: int, aux_specs: list[AuxFieldSpec]) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(hidden_dim, num_main_classes)
        self.aux_heads = nn.ModuleDict({spec.key: nn.Linear(hidden_dim, spec.num_classes) for spec in aux_specs})

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shared = self.backbone(inputs)
        main_logits = self.main_head(shared)
        aux_logits = {name: head(shared) for name, head in self.aux_heads.items()}
        return main_logits, aux_logits


class ExperienceAuxProbe:
    def __init__(self, scaler: StandardScaler, model: MultiTaskProbeNet, device: torch.device) -> None:
        self.scaler = scaler
        self.model = model
        self.device = device

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(features).astype(np.float32, copy=False)
        inputs = torch.from_numpy(transformed).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(inputs)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task5/Task6/Task7 DINO probe with auxiliary experience labels.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASK56))
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", required=True, help="One or two comma-separated DINOv2 model names.")
    parser.add_argument("--feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--experience-label-csv", required=True, help="CSV of soft or strict experience-label candidates.")
    parser.add_argument("--aux-fields", default=",".join(DEFAULT_AUX_FIELDS))
    parser.add_argument("--aux-loss-weight", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=DEFAULT_DINO_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    return parser.parse_args()


def extract_concat_features(
    image_df: pd.DataFrame,
    model_names: list[str],
    models: list[torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    features_list: list[np.ndarray] = []
    labels_ref: np.ndarray | None = None
    case_ids_ref: list[str] | None = None
    image_names_ref: list[str] | None = None
    for model_name, model in zip(model_names, models):
        features, labels, case_ids, image_names = extract_dataset_features(
            image_df,
            input_variant=args.input_variant,
            feature_mode=args.feature_mode,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model=model,
            device=device,
        )
        if labels_ref is None:
            labels_ref = labels
            case_ids_ref = case_ids
            image_names_ref = image_names
        else:
            if not np.array_equal(labels_ref, labels):
                raise ValueError(f"Label mismatch while concatenating features for {model_name}.")
            if case_ids_ref != case_ids or image_names_ref != image_names:
                raise ValueError(f"Sample order mismatch while concatenating features for {model_name}.")
        features_list.append(features)
    assert labels_ref is not None and case_ids_ref is not None and image_names_ref is not None
    return np.concatenate(features_list, axis=1), labels_ref, case_ids_ref, image_names_ref


def extract_features_for_split(
    image_df: pd.DataFrame,
    model_names: list[str],
    models: list[torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if len(model_names) == 1:
        return extract_dataset_features(
            image_df,
            input_variant=args.input_variant,
            feature_mode=args.feature_mode,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model=models[0],
            device=device,
        )
    return extract_concat_features(image_df, model_names, models, args, device)


def compute_selection_metric(task, prediction_df: pd.DataFrame) -> float:
    case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean")
    metrics = summarize_prediction_frame(case_df, task.class_names)
    return float(metrics.get(task.primary_metric, float("nan")))


def resolve_aux_specs(experience_df: pd.DataFrame, aux_fields: list[str]) -> list[AuxFieldSpec]:
    specs: list[AuxFieldSpec] = []
    for key in aux_fields:
        if key not in FIELD_TO_COLUMN:
            raise KeyError(f"Unknown aux field: {key}. Available: {', '.join(FIELD_TO_COLUMN)}")
        column = FIELD_TO_COLUMN[key]
        if column not in experience_df.columns:
            raise KeyError(f"Experience label CSV missing column: {column}")
        categories = FIXED_FIELD_CATEGORIES[key]
        specs.append(AuxFieldSpec(key=key, column=column, categories=tuple(categories)))
    return specs


def build_aux_targets(
    case_ids: list[str],
    image_names: list[str],
    experience_df: pd.DataFrame,
    aux_specs: list[AuxFieldSpec],
) -> dict[str, np.ndarray]:
    exp = experience_df.copy()
    exp["case_id"] = exp["case_id"].astype(str)
    exp["image_name"] = exp["image_name"].astype(str)
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for _, row in exp.iterrows():
        mapping[(row["case_id"], row["image_name"])] = row.to_dict()

    outputs: dict[str, np.ndarray] = {}
    for spec in aux_specs:
        cat_to_idx = {cat: idx for idx, cat in enumerate(spec.categories)}
        arr = np.full((len(case_ids),), -100, dtype=np.int64)
        for idx, key in enumerate(zip(case_ids, image_names)):
            payload = mapping.get((str(key[0]), str(key[1])))
            if payload is None:
                continue
            value = str(payload.get(spec.column, "")).strip()
            if spec.key == "boundary_axis":
                value = BOUNDARY_AXIS_NORMALIZATION.get(value, value)
            if value in cat_to_idx:
                arr[idx] = cat_to_idx[value]
        outputs[spec.key] = arr
    return outputs


def fit_aux_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_aux_targets: dict[str, np.ndarray],
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
    aux_specs: list[AuxFieldSpec],
) -> tuple[ExperienceAuxProbe, int, float]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features).astype(np.float32, copy=False)
    val_x = scaler.transform(val_features).astype(np.float32, copy=False)

    device = resolve_device(args.device)
    model = MultiTaskProbeNet(
        input_dim=train_x.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_main_classes=len(task.class_names),
        aux_specs=aux_specs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_counts = np.bincount(train_labels, minlength=len(task.class_names)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    main_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    aux_criteria = {spec.key: nn.CrossEntropyLoss(ignore_index=-100) for spec in aux_specs}

    train_inputs = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_labels.astype(np.int64, copy=False))
    aux_tensors = {k: torch.from_numpy(v.astype(np.int64, copy=False)) for k, v in train_aux_targets.items()}
    val_inputs = torch.from_numpy(val_x)

    batch_size = min(64, len(train_inputs))
    rng = np.random.default_rng(args.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_metric: float | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = rng.permutation(len(train_inputs))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_inputs = train_inputs[batch_idx].to(device)
            batch_targets = train_targets[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            main_logits, aux_logits = model(batch_inputs)
            loss = main_criterion(main_logits, batch_targets)

            aux_losses: list[torch.Tensor] = []
            for spec in aux_specs:
                batch_aux = aux_tensors[spec.key][batch_idx].to(device)
                if (batch_aux != -100).any():
                    aux_losses.append(aux_criteria[spec.key](aux_logits[spec.key], batch_aux))
            if aux_losses:
                loss = loss + args.aux_loss_weight * torch.stack(aux_losses).mean()

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_inputs.to(device))
            val_probs = F.softmax(val_logits, dim=1).cpu().numpy().astype(np.float64)
        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=val_labels,
            case_ids=val_case_ids,
            image_names=val_image_names,
            class_names=task.class_names,
        )
        metric_value = compute_selection_metric(task, val_predictions)
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_metric = float(metric_value)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None or best_metric is None:
        raise RuntimeError("Failed to fit experience-aux probe.")
    model.load_state_dict(best_state)
    return ExperienceAuxProbe(scaler=scaler, model=model, device=device), int(best_epoch), float(best_metric)


def run_single_fold(
    args: argparse.Namespace,
    task,
    image_df: pd.DataFrame,
    models: list[torch.nn.Module],
    model_names: list[str],
    device: torch.device,
    fold_id: int,
    fold_count: int,
    experience_df: pd.DataFrame,
    aux_specs: list[AuxFieldSpec],
) -> dict[str, Any]:
    fold_dir = Path(args.output_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=fold_count)
    train_df = trim_image_df(train_df, args.max_train_images)
    val_df = trim_image_df(val_df, args.max_val_images)
    test_df = trim_image_df(test_df, args.max_test_images)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

    train_features, train_labels, train_case_ids, train_image_names = extract_features_for_split(train_df, model_names, models, args, device)
    val_features, val_labels, val_case_ids, val_image_names = extract_features_for_split(val_df, model_names, models, args, device)
    test_features, test_labels, test_case_ids, test_image_names = extract_features_for_split(test_df, model_names, models, args, device)

    train_aux_targets = build_aux_targets(train_case_ids, train_image_names, experience_df, aux_specs)
    aux_coverage = {
        spec.key: int((train_aux_targets[spec.key] != -100).sum())
        for spec in aux_specs
    }

    run_config = {
        "task": task.key,
        "input_variant": args.input_variant,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "model_names": model_names,
        "feature_mode": args.feature_mode,
        "aux_fields": [spec.key for spec in aux_specs],
        "aux_loss_weight": float(args.aux_loss_weight),
        "fold_id": fold_id,
        "device": str(device),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "test_images": int(len(test_df)),
        "train_cases": int(train_df["case_id"].nunique()),
        "val_cases": int(val_df["case_id"].nunique()),
        "test_cases": int(test_df["case_id"].nunique()),
        "aux_train_coverage": aux_coverage,
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== Task56 ExperienceAux Fold {fold_id}/{fold_count} | task={task.key} | models={model_names} ===",
        flush=True,
    )
    print(
        f"train_cases={run_config['train_cases']}, val_cases={run_config['val_cases']}, test_cases={run_config['test_cases']}, "
        f"aux_coverage={aux_coverage}",
        flush=True,
    )

    probe, best_epoch, best_val_metric = fit_aux_probe(
        args=args,
        train_features=train_features,
        train_labels=train_labels,
        train_aux_targets=train_aux_targets,
        val_features=val_features,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
        aux_specs=aux_specs,
    )

    val_predictions = build_prediction_df(
        probs=probe.predict_proba(val_features),
        labels=val_labels,
        case_ids=val_case_ids,
        image_names=val_image_names,
        class_names=task.class_names,
    )
    test_predictions = build_prediction_df(
        probs=probe.predict_proba(test_features),
        labels=test_labels,
        case_ids=test_case_ids,
        image_names=test_image_names,
        class_names=task.class_names,
    )

    val_case_metrics = save_split_outputs(fold_dir, "val", val_predictions, task, aggregate_methods)
    test_case_metrics = save_split_outputs(fold_dir, "test", test_predictions, task, aggregate_methods)
    fold_summary = {
        "best_c": float("nan"),
        "best_epoch": int(best_epoch),
        "best_val_primary_metric": float(best_val_metric),
        "val_case_mean": val_case_metrics["mean"],
        "test_case_mean": test_case_metrics["mean"],
        "aux_train_coverage": aux_coverage,
    }
    write_json(fold_dir / "fold_summary.json", fold_summary)
    print(
        f"[Fold {fold_id}] best_epoch={best_epoch} | "
        f"val_{task.primary_metric}={format_metric_value(val_case_metrics['mean'].get(task.primary_metric))} | "
        f"test_{task.primary_metric}={format_metric_value(test_case_metrics['mean'].get(task.primary_metric))} | "
        f"test_acc={format_metric_value(test_case_metrics['mean'].get('accuracy'))}",
        flush=True,
    )
    return fold_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    task = get_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if not model_names or len(model_names) > 2:
        raise ValueError("--model-names must contain one or two model names.")

    experience_df = pd.read_csv(args.experience_label_csv, dtype={"case_id": str})
    aux_fields = [item.strip() for item in args.aux_fields.split(",") if item.strip()]
    aux_specs = resolve_aux_specs(experience_df, aux_fields)

    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    repo_dir = Path(args.repo_dir)
    models = [load_dinov2_model(repo_dir=repo_dir, model_name=name, device=device) for name in model_names]

    print(
        f"Starting Task56 experience-aux probe: task={task.key}, models={model_names}, "
        f"aux_fields={aux_fields}, aux_loss_weight={args.aux_loss_weight}, device={device}, folds={fold_ids}",
        flush=True,
    )

    for fold_id in fold_ids:
        run_single_fold(
            args=args,
            task=task,
            image_df=image_df,
            models=models,
            model_names=model_names,
            device=device,
            fold_id=fold_id,
            fold_count=len(available_folds),
            experience_df=experience_df,
            aux_specs=aux_specs,
        )

    if len(fold_ids) > 1:
        aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
        aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
