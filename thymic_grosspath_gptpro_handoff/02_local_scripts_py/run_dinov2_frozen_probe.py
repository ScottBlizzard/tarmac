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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS, TaskConfig, get_task
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import (
    expand_registry_to_images,
    filter_registry_for_task,
    load_registry,
    load_split_assignments,
    merge_registry_with_splits,
    subset_by_fold,
)
from thymic_baseline.train import build_dataloader, format_metric_value, metrics_to_frame, resolve_device


DEFAULT_IMAGE_SIZE = 518
FEATURE_MODES = ("cls", "patch_mean", "cls_patchmean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen DINOv2 feature probe on thymic tasks.")
    parser.add_argument("--registry-csv", required=True, help="Case-level registry CSV.")
    parser.add_argument("--split-csv", required=True, help="Case-level split assignment CSV.")
    parser.add_argument("--images-root", required=True, help="Root directory containing all image files.")
    parser.add_argument("--task", required=True, help="Task key defined in thymic_baseline.config.")
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all", help="Fold id 1-5, or 'all' to run 5-fold CV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports and cached predictions.")
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2", help="Local path to the cloned DINOv2 repo.")
    parser.add_argument("--model-name", default="dinov2_vits14", help="torch.hub model entry, e.g. dinov2_vits14.")
    parser.add_argument("--feature-mode", default="cls", choices=FEATURE_MODES, help="Frozen feature pooling strategy.")
    parser.add_argument("--probe-type", default="logreg", choices=("logreg", "mlp", "lda"), help="Probe head type.")
    parser.add_argument("--fit-level", default="image", choices=("image", "case"), help="Fit probe on image-level or case-aggregated features.")
    parser.add_argument("--case-feature-agg", default="mean", choices=("mean",), help="Aggregation for case-level frozen features.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated logistic-regression C values.",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=512, help="Hidden size for MLP probe.")
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout for MLP probe.")
    parser.add_argument("--mlp-lr", type=float, default=1e-3, help="Learning rate for MLP probe.")
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4, help="Weight decay for MLP probe.")
    parser.add_argument("--mlp-epochs", type=int, default=60, help="Maximum epochs for MLP probe.")
    parser.add_argument("--mlp-patience", type=int, default=10, help="Early-stop patience for MLP probe.")
    parser.add_argument(
        "--lda-shrinkage",
        default="auto",
        help="Shrinkage for LDA probe. Use 'auto' or a float in [0,1].",
    )
    parser.add_argument(
        "--aggregate-methods",
        default="mean,max_prob,majority_vote",
        help="Comma-separated aggregation methods for case-level evaluation.",
    )
    parser.add_argument("--max-train-images", type=int, default=None, help="Optional image-level smoke limit.")
    parser.add_argument("--max-val-images", type=int, default=None, help="Optional image-level smoke limit.")
    parser.add_argument("--max-test-images", type=int, default=None, help="Optional image-level smoke limit.")
    parser.add_argument(
        "--logreg-class-weight",
        default="balanced",
        choices=("balanced", "none"),
        help="Class-weight mode for logistic regression probe.",
    )
    parser.add_argument(
        "--subtype-weight-map",
        default="",
        help="Optional subtype sample-weight map, e.g. 'thymoma_B2:1.5,thymic_carcinoma:1.4'.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task_image_df(registry_csv: str, split_csv: str, images_root: str, task: TaskConfig) -> pd.DataFrame:
    registry = load_registry(registry_csv)
    split_df = load_split_assignments(split_csv)
    merged = merge_registry_with_splits(registry, split_df)
    filtered = filter_registry_for_task(merged, task)
    return expand_registry_to_images(filtered, task=task, images_root=images_root)


def load_available_folds(split_csv: str) -> list[int]:
    split_df = load_split_assignments(split_csv)
    fold_ids = sorted(int(item) for item in split_df["master_fold_id"].dropna().unique().tolist())
    if not fold_ids:
        raise ValueError("No fold ids found in split CSV.")
    return fold_ids


def load_dinov2_model(repo_dir: Path, model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(str(repo_dir), model_name, source="local", pretrained=True)
    model.eval().to(device)
    return model


def trim_image_df(image_df: pd.DataFrame, max_images: int | None) -> pd.DataFrame:
    if max_images is None or len(image_df) <= max_images:
        return image_df.reset_index(drop=True)
    return image_df.iloc[:max_images].reset_index(drop=True)


def extract_feature_tensor(outputs: dict[str, torch.Tensor], feature_mode: str) -> torch.Tensor:
    cls_token = outputs["x_norm_clstoken"]
    patch_tokens = outputs["x_norm_patchtokens"]
    if feature_mode == "cls":
        return cls_token
    if feature_mode == "patch_mean":
        return patch_tokens.mean(dim=1)
    if feature_mode == "cls_patchmean":
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def extract_dino_features(model: torch.nn.Module, batch_inputs: Any, feature_mode: str) -> torch.Tensor:
    if isinstance(batch_inputs, torch.Tensor):
        outputs = model.forward_features(batch_inputs)
        return extract_feature_tensor(outputs, feature_mode)
    if isinstance(batch_inputs, (tuple, list)):
        parts = [extract_dino_features(model, item, feature_mode) for item in batch_inputs]
        return torch.cat(parts, dim=1)
    raise TypeError(f"Unsupported batch input type: {type(batch_inputs)!r}")


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, tuple):
        return tuple(move_to_device(item, device) for item in inputs)
    if isinstance(inputs, list):
        return [move_to_device(item, device) for item in inputs]
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def extract_dataset_features(
    image_df: pd.DataFrame,
    input_variant: str,
    feature_mode: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    loader: DataLoader = build_dataloader(
        image_df=image_df,
        input_variant=input_variant,
        image_size=image_size,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    case_ids: list[str] = []
    image_names: list[str] = []

    with torch.no_grad():
        for inputs, batch_labels, batch_case_ids, batch_image_names in loader:
            inputs = move_to_device(inputs, device)
            batch_features = extract_dino_features(model, inputs, feature_mode)
            features.append(batch_features.detach().cpu().numpy().astype(np.float32))
            labels.append(batch_labels.numpy().astype(np.int64))
            case_ids.extend(str(item) for item in batch_case_ids)
            image_names.extend(str(item) for item in batch_image_names)

    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        case_ids,
        image_names,
    )


def aggregate_features_to_cases(
    features: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    image_names: list[str],
    agg: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if agg != "mean":
        raise ValueError(f"Unsupported case feature aggregation: {agg}")
    groups: dict[str, dict[str, Any]] = {}
    for idx, case_id in enumerate(case_ids):
        if case_id not in groups:
            groups[case_id] = {
                "features": [],
                "label": int(labels[idx]),
                "image_names": [],
            }
        groups[case_id]["features"].append(features[idx])
        groups[case_id]["image_names"].append(str(image_names[idx]))
    out_features: list[np.ndarray] = []
    out_labels: list[int] = []
    out_case_ids: list[str] = []
    out_image_names: list[str] = []
    for case_id in sorted(groups):
        group = groups[case_id]
        out_features.append(np.mean(np.stack(group["features"], axis=0), axis=0))
        out_labels.append(group["label"])
        out_case_ids.append(case_id)
        out_image_names.append("|".join(group["image_names"]))
    return (
        np.stack(out_features, axis=0).astype(np.float32),
        np.asarray(out_labels, dtype=np.int64),
        out_case_ids,
        out_image_names,
    )


def parse_subtype_weight_map(raw: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    if not raw.strip():
        return mapping
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid subtype-weight item: {item!r}")
        key, value = item.split(":", 1)
        mapping[key.strip()] = float(value.strip())
    return mapping


def build_sample_weights(subtypes: list[str], weight_map: dict[str, float]) -> np.ndarray | None:
    if not weight_map:
        return None
    return np.asarray([float(weight_map.get(str(item), 1.0)) for item in subtypes], dtype=np.float64)


def aggregate_weights_to_cases(weights: np.ndarray | None, case_ids: list[str]) -> np.ndarray | None:
    if weights is None:
        return None
    groups: dict[str, list[float]] = {}
    for weight, case_id in zip(weights, case_ids):
        groups.setdefault(str(case_id), []).append(float(weight))
    return np.asarray([float(np.mean(groups[case_id])) for case_id in sorted(groups)], dtype=np.float64)


def build_prediction_df(
    probs: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    image_names: list[str],
    class_names: tuple[str, ...],
) -> pd.DataFrame:
    pred_idx = probs.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    prob_cols = [f"prob_{name}" for name in class_names]
    for idx in range(len(labels)):
        row = {
            "case_id": str(case_ids[idx]),
            "image_name": str(image_names[idx]),
            "label_idx": int(labels[idx]),
            "pred_idx": int(pred_idx[idx]),
        }
        for col_name, value in zip(prob_cols, probs[idx]):
            row[col_name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_selection_metric(
    task: TaskConfig,
    prediction_df: pd.DataFrame,
    selection_metric: str | None = None,
) -> float:
    case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean")
    metrics = summarize_prediction_frame(case_df, task.class_names)
    metric_key = task.primary_metric if selection_metric in (None, "", "primary") else selection_metric
    return float(metrics.get(metric_key, float("nan")))


class StandardizedMLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class TorchMLPProbe:
    def __init__(self, scaler: StandardScaler, model: StandardizedMLPProbe, device: torch.device) -> None:
        self.scaler = scaler
        self.model = model
        self.device = device

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(features).astype(np.float32, copy=False)
        inputs = torch.from_numpy(transformed).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float64)


def fit_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_sample_weights: np.ndarray | None,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task: TaskConfig,
    c_grid: list[float],
    seed: int,
) -> tuple[Any, float, float]:
    if args.probe_type == "mlp":
        return fit_mlp_probe(
            args=args,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            val_case_ids=val_case_ids,
            val_image_names=val_image_names,
            task=task,
            seed=seed,
        )
    if args.probe_type == "lda":
        return fit_lda_probe(
            args=args,
            train_features=train_features,
            train_labels=train_labels,
            train_sample_weights=train_sample_weights,
            val_features=val_features,
            val_labels=val_labels,
            val_case_ids=val_case_ids,
            val_image_names=val_image_names,
            task=task,
        )
    return fit_logreg_probe(
        args=args,
        train_features=train_features,
        train_labels=train_labels,
        train_sample_weights=train_sample_weights,
        val_features=val_features,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
        c_grid=c_grid,
        seed=seed,
    )


def fit_logreg_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_sample_weights: np.ndarray | None,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task: TaskConfig,
    c_grid: list[float],
    seed: int,
) -> tuple[Pipeline, float, float]:
    best_model: Pipeline | None = None
    best_c = float("nan")
    best_metric: float | None = None

    for c_value in c_grid:
        classifier = LogisticRegression(
            C=c_value,
            max_iter=4000,
            solver="lbfgs",
            class_weight=None if args.logreg_class_weight == "none" else "balanced",
            random_state=seed,
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ]
        )
        fit_kwargs: dict[str, Any] = {}
        if train_sample_weights is not None:
            fit_kwargs["clf__sample_weight"] = train_sample_weights
        model.fit(train_features, train_labels, **fit_kwargs)
        val_probs = model.predict_proba(val_features)
        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=val_labels,
            case_ids=val_case_ids,
            image_names=val_image_names,
            class_names=task.class_names,
        )
        metric_value = compute_selection_metric(
            task,
            val_predictions,
            getattr(args, "selection_metric", None),
        )
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_model = model
            best_c = float(c_value)
            best_metric = float(metric_value)

    if best_model is None or best_metric is None:
        raise RuntimeError("Failed to fit any valid probe model.")
    return best_model, best_c, best_metric


def fit_mlp_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task: TaskConfig,
    seed: int,
) -> tuple[TorchMLPProbe, float, float]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features).astype(np.float32, copy=False)
    val_x = scaler.transform(val_features).astype(np.float32, copy=False)

    train_inputs = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_labels.astype(np.int64, copy=False))
    val_inputs = torch.from_numpy(val_x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandardizedMLPProbe(
        input_dim=train_x.shape[1],
        hidden_dim=args.mlp_hidden_dim,
        num_classes=len(task.class_names),
        dropout=args.mlp_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)

    class_counts = np.bincount(train_labels, minlength=len(task.class_names)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

    train_batch_size = min(64, len(train_inputs))
    best_state: dict[str, torch.Tensor] | None = None
    best_metric: float | None = None
    best_epoch = 0
    stale_epochs = 0

    rng = np.random.default_rng(seed)
    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        indices = rng.permutation(len(train_inputs))
        for start in range(0, len(indices), train_batch_size):
            batch_idx = indices[start : start + train_batch_size]
            batch_inputs = train_inputs[batch_idx].to(device)
            batch_targets = train_targets[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_inputs.to(device))
            val_probs = F.softmax(val_logits, dim=1).cpu().numpy().astype(np.float64)
        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=val_labels,
            case_ids=val_case_ids,
            image_names=val_image_names,
            class_names=task.class_names,
        )
        metric_value = compute_selection_metric(
            task,
            val_predictions,
            getattr(args, "selection_metric", None),
        )
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_metric = float(metric_value)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.mlp_patience:
                break

    if best_state is None or best_metric is None:
        raise RuntimeError("Failed to fit a valid MLP probe.")
    model.load_state_dict(best_state)
    probe = TorchMLPProbe(scaler=scaler, model=model, device=device)
    return probe, float(best_epoch), float(best_metric)


def fit_lda_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_sample_weights: np.ndarray | None,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task: TaskConfig,
) -> tuple[Pipeline, float, float]:
    if train_sample_weights is not None:
        raise ValueError("LDA probe does not support subtype sample weights.")
    shrinkage_value: str | float
    if args.lda_shrinkage == "auto":
        shrinkage_value = "auto"
    else:
        shrinkage_value = float(args.lda_shrinkage)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage_value)),
        ]
    )
    model.fit(train_features, train_labels)
    val_probs = model.predict_proba(val_features)
    val_predictions = build_prediction_df(
        probs=val_probs,
        labels=val_labels,
        case_ids=val_case_ids,
        image_names=val_image_names,
        class_names=task.class_names,
    )
    metric_value = compute_selection_metric(
        task,
        val_predictions,
        getattr(args, "selection_metric", None),
    )
    return model, float("nan"), float(metric_value)


def save_split_outputs(
    output_dir: Path,
    split_name: str,
    prediction_df: pd.DataFrame,
    task: TaskConfig,
    aggregate_methods: list[str],
) -> dict[str, dict[str, float]]:
    metrics_frames = [metrics_to_frame(split_name, "image", "none", summarize_prediction_frame(prediction_df, task.class_names))]
    prediction_df.to_csv(output_dir / f"{split_name}_image_predictions.csv", index=False)

    case_metrics: dict[str, dict[str, float]] = {}
    for method in aggregate_methods:
        case_df = aggregate_case_predictions(prediction_df, task.class_names, method=method)
        case_df.to_csv(output_dir / f"{split_name}_case_predictions_{method}.csv", index=False)
        metrics = summarize_prediction_frame(case_df, task.class_names)
        case_metrics[method] = metrics
        metrics_frames.append(metrics_to_frame(split_name, "case", method, metrics))

    pd.concat(metrics_frames, ignore_index=True).to_csv(output_dir / f"{split_name}_metrics.csv", index=False)
    return case_metrics


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_run_outputs(root_output_dir: Path, task: TaskConfig, fold_ids: list[int], aggregate_methods: list[str]) -> None:
    fold_rows: list[dict[str, Any]] = []
    all_test_images: list[pd.DataFrame] = []
    all_test_cases: dict[str, list[pd.DataFrame]] = {method: [] for method in aggregate_methods}

    for fold_id in fold_ids:
        fold_dir = root_output_dir / f"fold_{fold_id}"
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "best_c": float(summary["best_c"]),
                "best_val_primary_metric": float(summary["best_val_primary_metric"]),
                "test_case_mean_primary_metric": float(summary["test_case_mean"].get(task.primary_metric, float("nan"))),
                "test_case_mean_accuracy": float(summary["test_case_mean"].get("accuracy", float("nan"))),
                "test_case_mean_balanced_accuracy": float(summary["test_case_mean"].get("balanced_accuracy", float("nan"))),
            }
        )

        image_df = pd.read_csv(fold_dir / "test_image_predictions.csv")
        image_df["fold_id"] = fold_id
        all_test_images.append(image_df)

        for method in aggregate_methods:
            case_df = pd.read_csv(fold_dir / f"test_case_predictions_{method}.csv")
            case_df["fold_id"] = fold_id
            all_test_cases[method].append(case_df)

    pd.DataFrame(fold_rows).to_csv(root_output_dir / "cv_fold_summary.csv", index=False)

    image_predictions_df = pd.concat(all_test_images, ignore_index=True)
    image_predictions_df.to_csv(root_output_dir / "oof_image_predictions.csv", index=False)
    metric_frames = [metrics_to_frame("test_oof", "image", "none", summarize_prediction_frame(image_predictions_df, task.class_names))]

    for method, frames in all_test_cases.items():
        case_predictions_df = pd.concat(frames, ignore_index=True)
        case_predictions_df.to_csv(root_output_dir / f"oof_case_predictions_{method}.csv", index=False)
        metric_frames.append(
            metrics_to_frame("test_oof", "case", method, summarize_prediction_frame(case_predictions_df, task.class_names))
        )

    pd.concat(metric_frames, ignore_index=True).to_csv(root_output_dir / "oof_metrics.csv", index=False)


def run_single_fold(
    args: argparse.Namespace,
    task: TaskConfig,
    image_df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    fold_id: int,
    fold_count: int,
) -> dict[str, Any]:
    fold_dir = Path(args.output_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    subtype_weight_map = parse_subtype_weight_map(args.subtype_weight_map)
    train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=fold_count)
    train_df = trim_image_df(train_df, args.max_train_images)
    val_df = trim_image_df(val_df, args.max_val_images)
    test_df = trim_image_df(test_df, args.max_test_images)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

    run_config = {
        "task": task.key,
        "input_variant": args.input_variant,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "model_name": args.model_name,
        "feature_mode": args.feature_mode,
        "probe_type": args.probe_type,
        "fit_level": args.fit_level,
        "case_feature_agg": args.case_feature_agg,
        "logreg_class_weight": args.logreg_class_weight,
        "subtype_weight_map": subtype_weight_map,
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
        "c_grid": [float(item) for item in args.c_grid.split(",") if item.strip()],
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== DINOv2 Fold {fold_id}/{fold_count} | task={task.key} | variant={args.input_variant} | model={args.model_name} ===",
        flush=True,
    )
    print(
        f"feature_mode={args.feature_mode}, train_cases={run_config['train_cases']}, val_cases={run_config['val_cases']}, "
        f"test_cases={run_config['test_cases']}, train_images={run_config['train_images']}, "
        f"val_images={run_config['val_images']}, test_images={run_config['test_images']}",
        flush=True,
    )

    train_features, train_labels, _, _ = extract_dataset_features(
        train_df,
        input_variant=args.input_variant,
        feature_mode=args.feature_mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )
    val_features, val_labels, val_case_ids, val_image_names = extract_dataset_features(
        val_df,
        input_variant=args.input_variant,
        feature_mode=args.feature_mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )
    test_features, test_labels, test_case_ids, test_image_names = extract_dataset_features(
        test_df,
        input_variant=args.input_variant,
        feature_mode=args.feature_mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=model,
        device=device,
    )
    train_sample_weights = build_sample_weights(train_df["who_type_raw"].astype(str).tolist(), subtype_weight_map)

    if args.fit_level == "case":
        train_features, train_labels, _, _ = aggregate_features_to_cases(
            train_features, train_labels, train_df["case_id"].astype(str).tolist(), train_df["image_name"].astype(str).tolist(), args.case_feature_agg
        )
        val_features, val_labels, val_case_ids, val_image_names = aggregate_features_to_cases(
            val_features, val_labels, val_case_ids, val_image_names, args.case_feature_agg
        )
        test_features, test_labels, test_case_ids, test_image_names = aggregate_features_to_cases(
            test_features, test_labels, test_case_ids, test_image_names, args.case_feature_agg
        )
        train_sample_weights = aggregate_weights_to_cases(
            train_sample_weights,
            train_df["case_id"].astype(str).tolist(),
        )

    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
    probe, best_c, best_val_metric = fit_probe(
        args=args,
        train_features=train_features,
        train_labels=train_labels,
        train_sample_weights=train_sample_weights,
        val_features=val_features,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
        c_grid=c_grid,
        seed=args.seed,
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
        "best_c": float(best_c),
        "best_val_primary_metric": float(best_val_metric),
        "val_case_mean": val_case_metrics["mean"],
        "test_case_mean": test_case_metrics["mean"],
    }
    write_json(fold_dir / "fold_summary.json", fold_summary)
    print(
        f"[Fold {fold_id}] best_c={best_c:.4g} | "
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

    image_df = load_task_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    repo_dir = Path(args.repo_dir)
    model = load_dinov2_model(repo_dir=repo_dir, model_name=args.model_name, device=device)

    print(
        f"Starting DINOv2 frozen probe: task={task.key}, variant={args.input_variant}, "
        f"model={args.model_name}, feature_mode={args.feature_mode}, device={device}, folds={fold_ids}",
        flush=True,
    )
    for fold_id in fold_ids:
        run_single_fold(
            args=args,
            task=task,
            image_df=image_df,
            model=model,
            device=device,
            fold_id=fold_id,
            fold_count=len(available_folds),
        )

    if len(fold_ids) > 1:
        aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
        aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
