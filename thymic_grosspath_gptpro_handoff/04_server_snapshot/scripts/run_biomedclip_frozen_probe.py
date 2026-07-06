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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BIOMEDCOOP_ROOT = PROJECT_ROOT / "third_party" / "round3" / "BiomedCoOp"
if str(BIOMEDCOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMEDCOOP_ROOT))

from open_clip.src.open_clip import create_model_from_pretrained
from open_clip.src.open_clip.model import _build_vision_tower
from open_clip.src.open_clip.transform import PreprocessCfg, image_transform_v2
from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS, TaskConfig, get_task
from thymic_baseline.cropping import extract_specimen_crop
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import (
    expand_registry_to_images,
    filter_registry_for_task,
    load_registry,
    load_split_assignments,
    merge_registry_with_splits,
    subset_by_fold,
)
from thymic_baseline.train import format_metric_value, metrics_to_frame, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen BiomedCLIP feature probe on thymic tasks.")
    parser.add_argument("--registry-csv", required=True, help="Case-level registry CSV.")
    parser.add_argument("--split-csv", required=True, help="Case-level split assignment CSV.")
    parser.add_argument("--images-root", required=True, help="Root directory containing all image files.")
    parser.add_argument("--task", required=True, help="Task key defined in thymic_baseline.config.")
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all", help="Fold id 1-5, or 'all' to run 5-fold CV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports and cached predictions.")
    parser.add_argument(
        "--model-name",
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        help="BiomedCLIP model identifier for open_clip.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated logistic-regression C values.",
    )
    parser.add_argument(
        "--aggregate-methods",
        default="mean,max_prob,majority_vote",
        help="Comma-separated aggregation methods for case-level evaluation.",
    )
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
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


class BiomedClipImageDataset(Dataset):
    def __init__(self, image_df: pd.DataFrame, input_variant: str, preprocess) -> None:
        self.image_df = image_df.reset_index(drop=True)
        self.input_variant = input_variant
        self.preprocess = preprocess
        self._crop_cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.image_df)

    def _load_image(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def _get_specimen_crop(self, image_path: str, image: Image.Image) -> Image.Image:
        cached = self._crop_cache.get(image_path)
        if cached is None:
            cached = extract_specimen_crop(image)
            self._crop_cache[image_path] = cached
        return cached.copy()

    def __getitem__(self, index: int):
        row = self.image_df.iloc[index]
        image = self._load_image(row["image_path"])

        if self.input_variant == "whole":
            inputs: Any = self.preprocess(image)
        elif self.input_variant == "crop":
            inputs = self.preprocess(self._get_specimen_crop(row["image_path"], image))
        elif self.input_variant == "whole_plus_crop":
            inputs = (
                self.preprocess(image),
                self.preprocess(self._get_specimen_crop(row["image_path"], image)),
            )
        else:
            raise ValueError(f"Unsupported input variant: {self.input_variant}")

        return inputs, int(row["label_idx"]), str(row["case_id"]), str(row["image_name"])


class VisionOnlyWrapper(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image)


def build_collate_fn(input_variant: str):
    def collate(batch):
        inputs, labels, case_ids, image_names = zip(*batch)
        labels_arr = np.asarray(labels, dtype=np.int64)
        if input_variant == "whole_plus_crop":
            whole = torch.stack([item[0] for item in inputs], dim=0)
            crop = torch.stack([item[1] for item in inputs], dim=0)
            pixel_inputs: Any = (whole, crop)
        else:
            pixel_inputs = torch.stack(list(inputs), dim=0)
        return pixel_inputs, labels_arr, list(case_ids), list(image_names)

    return collate


def build_dataloader(
    image_df: pd.DataFrame,
    input_variant: str,
    batch_size: int,
    num_workers: int,
    preprocess,
) -> DataLoader:
    dataset = BiomedClipImageDataset(image_df=image_df, input_variant=input_variant, preprocess=preprocess)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": build_collate_fn(input_variant),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def load_biomedclip_model(model_name: str, device: torch.device):
    model_path = Path(model_name)
    if model_path.exists():
        config = json.loads((model_path / "open_clip_config.json").read_text(encoding="utf-8"))
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]
        visual = _build_vision_tower(
            embed_dim=int(model_cfg["embed_dim"]),
            vision_cfg=model_cfg["vision_cfg"],
            quick_gelu=False,
            cast_dtype=None,
        )
        state_dict = torch.load(model_path / "open_clip_pytorch_model.bin", map_location="cpu")
        visual_state = {key[len("visual."):]: value for key, value in state_dict.items() if key.startswith("visual.")}
        visual.load_state_dict(visual_state, strict=True)
        preprocess = image_transform_v2(PreprocessCfg(**preprocess_cfg), is_train=False)
        model = VisionOnlyWrapper(visual).to(device).eval()
        return model, preprocess

    model, preprocess = create_model_from_pretrained(model_name)
    model = model.to(device).eval()
    return model, preprocess


def trim_image_df(image_df: pd.DataFrame, max_images: int | None) -> pd.DataFrame:
    if max_images is None or len(image_df) <= max_images:
        return image_df.reset_index(drop=True)
    return image_df.iloc[:max_images].reset_index(drop=True)


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, tuple):
        return tuple(move_to_device(item, device) for item in inputs)
    if isinstance(inputs, list):
        return [move_to_device(item, device) for item in inputs]
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def extract_image_features(model, batch_inputs: Any) -> torch.Tensor:
    if isinstance(batch_inputs, torch.Tensor):
        return model.encode_image(batch_inputs)
    if isinstance(batch_inputs, (tuple, list)):
        parts = [extract_image_features(model, item) for item in batch_inputs]
        return torch.cat(parts, dim=1)
    raise TypeError(f"Unsupported batch input type: {type(batch_inputs)!r}")


def extract_dataset_features(
    image_df: pd.DataFrame,
    input_variant: str,
    batch_size: int,
    num_workers: int,
    preprocess,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    loader = build_dataloader(
        image_df=image_df,
        input_variant=input_variant,
        batch_size=batch_size,
        num_workers=num_workers,
        preprocess=preprocess,
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    case_ids: list[str] = []
    image_names: list[str] = []

    with torch.no_grad():
        for inputs, batch_labels, batch_case_ids, batch_image_names in loader:
            inputs = move_to_device(inputs, device)
            batch_features = extract_image_features(model, inputs)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            features.append(batch_features.detach().cpu().numpy().astype(np.float32))
            labels.append(batch_labels.astype(np.int64))
            case_ids.extend(str(item) for item in batch_case_ids)
            image_names.extend(str(item) for item in batch_image_names)

    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        case_ids,
        image_names,
    )


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


def compute_selection_metric(task: TaskConfig, prediction_df: pd.DataFrame) -> float:
    case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean")
    metrics = summarize_prediction_frame(case_df, task.class_names)
    return float(metrics.get(task.primary_metric, float("nan")))


def fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
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
            class_weight="balanced",
            random_state=seed,
        )
        model = Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
        model.fit(train_features, train_labels)
        val_probs = model.predict_proba(val_features)
        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=val_labels,
            case_ids=val_case_ids,
            image_names=val_image_names,
            class_names=task.class_names,
        )
        metric_value = compute_selection_metric(task, val_predictions)
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_model = model
            best_c = float(c_value)
            best_metric = float(metric_value)

    if best_model is None or best_metric is None:
        raise RuntimeError("Failed to fit any valid probe model.")
    return best_model, best_c, best_metric


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
        metric_frames.append(metrics_to_frame("test_oof", "case", method, summarize_prediction_frame(case_predictions_df, task.class_names)))

    pd.concat(metric_frames, ignore_index=True).to_csv(root_output_dir / "oof_metrics.csv", index=False)


def run_single_fold(
    args: argparse.Namespace,
    task: TaskConfig,
    image_df: pd.DataFrame,
    model,
    preprocess,
    device: torch.device,
    fold_id: int,
    fold_count: int,
) -> dict[str, Any]:
    fold_dir = Path(args.output_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=fold_count)
    train_df = trim_image_df(train_df, args.max_train_images)
    val_df = trim_image_df(val_df, args.max_val_images)
    test_df = trim_image_df(test_df, args.max_test_images)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

    run_config = {
        "task": task.key,
        "input_variant": args.input_variant,
        "model_name": args.model_name,
        "fold_id": fold_id,
        "device": str(device),
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
        f"\n=== BiomedCLIP Fold {fold_id}/{fold_count} | task={task.key} | variant={args.input_variant} ===",
        flush=True,
    )
    print(
        f"train_cases={run_config['train_cases']}, val_cases={run_config['val_cases']}, "
        f"test_cases={run_config['test_cases']}, train_images={run_config['train_images']}, "
        f"val_images={run_config['val_images']}, test_images={run_config['test_images']}",
        flush=True,
    )

    train_features, train_labels, _, _ = extract_dataset_features(
        train_df,
        input_variant=args.input_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=preprocess,
        model=model,
        device=device,
    )
    val_features, val_labels, val_case_ids, val_image_names = extract_dataset_features(
        val_df,
        input_variant=args.input_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=preprocess,
        model=model,
        device=device,
    )
    test_features, test_labels, test_case_ids, test_image_names = extract_dataset_features(
        test_df,
        input_variant=args.input_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=preprocess,
        model=model,
        device=device,
    )

    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
    probe, best_c, best_val_metric = fit_probe(
        train_features=train_features,
        train_labels=train_labels,
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
    model, preprocess = load_biomedclip_model(model_name=args.model_name, device=device)

    print(
        f"Starting BiomedCLIP frozen probe: task={task.key}, variant={args.input_variant}, "
        f"model_name={args.model_name}, device={device}, folds={fold_ids}",
        flush=True,
    )
    for fold_id in fold_ids:
        run_single_fold(
            args=args,
            task=task,
            image_df=image_df,
            model=model,
            preprocess=preprocess,
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
