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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_dinov2_frozen_probe import (  # type: ignore
    aggregate_features_to_cases as aggregate_dino_features_to_cases,
    build_prediction_df,
    extract_dataset_features as extract_dino_dataset_features,
    load_available_folds,
    load_dinov2_model,
    load_task_image_df,
    trim_image_df,
    write_json,
)
from run_plip_caseprobe import (  # type: ignore
    aggregate_features_to_cases as aggregate_plip_features_to_cases,
    extract_dataset_features as extract_plip_dataset_features,
    load_plip_model,
)
from thymic_baseline.config import DEFAULT_RANDOM_SEED, get_task
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import format_metric_value, metrics_to_frame, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task4 DINO+PLIP case-level fusion probe.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", default="task4_who_5class")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--dino-feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--plip-model-dir", default="third_party/round3/local_models/plip")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--probe-type", default="mlp", choices=("logreg", "mlp"))
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-dropout", type=float, default=0.4)
    parser.add_argument("--mlp-lr", type=float, default=5e-4)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-3)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-patience", type=int, default=12)
    parser.add_argument("--aggregate-methods", default="mean")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_selection_metric(task, prediction_df: pd.DataFrame) -> float:
    case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean")
    metrics = summarize_prediction_frame(case_df, task.class_names)
    return float(metrics.get(task.primary_metric, float("nan")))


def align_case_features(
    feature_blocks: list[tuple[np.ndarray, np.ndarray, list[str], list[str]]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    base_features, base_labels, base_case_ids, base_image_names = feature_blocks[0]
    combined = [base_features]
    for features, labels, case_ids, image_names in feature_blocks[1:]:
        if case_ids != base_case_ids:
            raise ValueError("Case ids are not aligned across feature blocks.")
        if image_names != base_image_names:
            raise ValueError("Case image-name traces are not aligned across feature blocks.")
        if not np.array_equal(labels, base_labels):
            raise ValueError("Case labels are not aligned across feature blocks.")
        combined.append(features)
    return np.concatenate(combined, axis=1), base_labels, base_case_ids, base_image_names


class StandardizedMLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
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


def fit_logreg_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
    c_grid: list[float],
    seed: int,
) -> tuple[Pipeline, float, float]:
    best_model: Pipeline | None = None
    best_c = float("nan")
    best_metric: float | None = None
    for c_value in c_grid:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        max_iter=4000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(train_features, train_labels)
        val_probs = model.predict_proba(val_features)
        val_predictions = build_prediction_df(val_probs, val_labels, val_case_ids, val_image_names, task.class_names)
        metric_value = compute_selection_metric(task, val_predictions)
        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_model = model
            best_c = float(c_value)
            best_metric = float(metric_value)
    if best_model is None or best_metric is None:
        raise RuntimeError("Failed to fit logistic fusion probe.")
    return best_model, best_c, best_metric


def fit_mlp_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
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

    train_batch_size = min(32, len(train_inputs))
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
        metric_value = compute_selection_metric(task, val_predictions)
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
        raise RuntimeError("Failed to fit MLP fusion probe.")
    model.load_state_dict(best_state)
    probe = TorchMLPProbe(scaler=scaler, model=model, device=device)
    return probe, float(best_epoch), float(best_metric)


def fit_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
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
    return fit_logreg_probe(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
        c_grid=c_grid,
        seed=seed,
    )


def save_mode_metrics(
    fold_dir: Path,
    split_name: str,
    mode_to_predictions: dict[str, pd.DataFrame],
    task,
) -> dict[str, dict[str, float]]:
    rows: list[pd.DataFrame] = []
    metrics_by_mode: dict[str, dict[str, float]] = {}
    for mode_name, prediction_df in mode_to_predictions.items():
        case_df = aggregate_case_predictions(prediction_df, task.class_names, method="mean")
        metrics = summarize_prediction_frame(case_df, task.class_names)
        metrics_by_mode[mode_name] = metrics
        rows.append(metrics_to_frame(split_name, "case", mode_name, metrics))
    pd.concat(rows, ignore_index=True).to_csv(fold_dir / f"{split_name}_metrics_by_mode.csv", index=False)
    return metrics_by_mode


def aggregate_run_outputs(root_output_dir: Path, task, fold_ids: list[int]) -> None:
    all_mode_metrics: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        fold_dir = root_output_dir / f"fold_{fold_id}"
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "best_stat": float(summary["best_stat"]),
                "best_val_primary_metric": float(summary["best_val_primary_metric"]),
                "test_case_mean_primary_metric": float(summary["test_case_by_mode"]["fusion_case"].get(task.primary_metric, float("nan"))),
                "test_case_mean_accuracy": float(summary["test_case_by_mode"]["fusion_case"].get("accuracy", float("nan"))),
                "test_case_mean_balanced_accuracy": float(summary["test_case_by_mode"]["fusion_case"].get("balanced_accuracy", float("nan"))),
            }
        )
        mode_df = pd.read_csv(fold_dir / "test_metrics_by_mode.csv")
        mode_df["fold_id"] = fold_id
        all_mode_metrics.append(mode_df)
    pd.DataFrame(fold_rows).to_csv(root_output_dir / "cv_fold_summary.csv", index=False)
    metrics_df = pd.concat(all_mode_metrics, ignore_index=True)
    metrics_df.to_csv(root_output_dir / "oof_metrics_by_mode.csv", index=False)
    fusion_rows = metrics_df[metrics_df["aggregation"] == "fusion_case"].copy()
    if not fusion_rows.empty:
        summary = {
            "split": "test_oof",
            "level": "case",
            "aggregation": "fusion_case",
        }
        for col in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"]:
            summary[col] = float(fusion_rows[col].mean())
        pd.DataFrame([summary]).to_csv(root_output_dir / "oof_metrics.csv", index=False)


def run_single_fold(
    args: argparse.Namespace,
    task,
    image_df: pd.DataFrame,
    dino_model: torch.nn.Module,
    plip_model,
    plip_processor,
    device: torch.device,
    fold_id: int,
    available_folds: list[int],
) -> dict[str, Any]:
    fold_dir = Path(args.output_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
    train_df = trim_image_df(train_df, args.max_train_images)
    val_df = trim_image_df(val_df, args.max_val_images)
    test_df = trim_image_df(test_df, args.max_test_images)
    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]

    run_config = {
        "task": task.key,
        "probe_type": args.probe_type,
        "dino_model_name": args.dino_model_name,
        "dino_feature_mode": args.dino_feature_mode,
        "plip_model_dir": str(Path(args.plip_model_dir).resolve()),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "fold_id": int(fold_id),
    }
    write_json(fold_dir / "run_config.json", run_config)

    print(
        f"\n=== DINO+PLIP Fusion Fold {fold_id}/{len(available_folds)} | task={task.key} | probe={args.probe_type} ===",
        flush=True,
    )
    print(
        f"train_cases={train_df['case_id'].nunique()}, val_cases={val_df['case_id'].nunique()}, test_cases={test_df['case_id'].nunique()}, "
        f"train_images={len(train_df)}, val_images={len(val_df)}, test_images={len(test_df)}",
        flush=True,
    )

    def extract_split_features(df: pd.DataFrame):
        dino_case = aggregate_dino_features_to_cases(
            *extract_dino_dataset_features(
                image_df=df,
                input_variant="whole",
                feature_mode=args.dino_feature_mode,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                model=dino_model,
                device=device,
            ),
            agg="mean",
        )
        plip_case = aggregate_plip_features_to_cases(
            *extract_plip_dataset_features(
                image_df=df,
                input_variant="whole",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                processor=plip_processor,
                model=plip_model,
                device=device,
            ),
            agg="mean",
        )
        return dino_case, plip_case

    train_dino_case, train_plip_case = extract_split_features(train_df)
    val_dino_case, val_plip_case = extract_split_features(val_df)
    test_dino_case, test_plip_case = extract_split_features(test_df)

    train_combo, train_labels, train_case_ids, train_image_names = align_case_features([train_dino_case, train_plip_case])
    val_combo, val_labels, val_case_ids, val_image_names = align_case_features([val_dino_case, val_plip_case])
    test_combo, test_labels, test_case_ids, test_image_names = align_case_features([test_dino_case, test_plip_case])

    combo_model, best_stat, best_val_metric = fit_probe(
        args=args,
        train_features=train_combo,
        train_labels=train_labels,
        val_features=val_combo,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
        c_grid=c_grid,
        seed=args.seed,
    )

    dino_only_model, _, _ = fit_logreg_probe(
        train_dino_case[0],
        train_dino_case[1],
        val_dino_case[0],
        val_dino_case[1],
        val_dino_case[2],
        val_dino_case[3],
        task,
        c_grid,
        args.seed,
    )
    plip_only_model, _, _ = fit_logreg_probe(
        train_plip_case[0],
        train_plip_case[1],
        val_plip_case[0],
        val_plip_case[1],
        val_plip_case[2],
        val_plip_case[3],
        task,
        c_grid,
        args.seed,
    )

    val_combo_probs = combo_model.predict_proba(val_combo)
    test_combo_probs = combo_model.predict_proba(test_combo)
    val_dino_probs = dino_only_model.predict_proba(val_dino_case[0])
    test_dino_probs = dino_only_model.predict_proba(test_dino_case[0])
    val_plip_probs = plip_only_model.predict_proba(val_plip_case[0])
    test_plip_probs = plip_only_model.predict_proba(test_plip_case[0])

    val_predictions = build_prediction_df(val_combo_probs, val_labels, val_case_ids, val_image_names, task.class_names)
    test_predictions = build_prediction_df(test_combo_probs, test_labels, test_case_ids, test_image_names, task.class_names)

    val_mode_metrics = save_mode_metrics(
        fold_dir,
        "val",
        {
            "dino_case": build_prediction_df(val_dino_probs, val_dino_case[1], val_dino_case[2], val_dino_case[3], task.class_names),
            "plip_case": build_prediction_df(val_plip_probs, val_plip_case[1], val_plip_case[2], val_plip_case[3], task.class_names),
            "fusion_case": val_predictions,
        },
        task,
    )
    test_mode_metrics = save_mode_metrics(
        fold_dir,
        "test",
        {
            "dino_case": build_prediction_df(test_dino_probs, test_dino_case[1], test_dino_case[2], test_dino_case[3], task.class_names),
            "plip_case": build_prediction_df(test_plip_probs, test_plip_case[1], test_plip_case[2], test_plip_case[3], task.class_names),
            "fusion_case": test_predictions,
        },
        task,
    )

    fold_summary = {
        "best_stat": float(best_stat),
        "best_val_primary_metric": float(best_val_metric),
        "val_case_by_mode": val_mode_metrics,
        "test_case_by_mode": test_mode_metrics,
    }
    write_json(fold_dir / "fold_summary.json", fold_summary)
    print(
        f"[Fold {fold_id}] best_stat={best_stat:.4g} | "
        f"val_{task.primary_metric}={format_metric_value(val_mode_metrics['fusion_case'].get(task.primary_metric, float('nan')))} | "
        f"test_{task.primary_metric}={format_metric_value(test_mode_metrics['fusion_case'].get(task.primary_metric, float('nan')))} | "
        f"test_acc={format_metric_value(test_mode_metrics['fusion_case'].get('accuracy', float('nan')))}",
        flush=True,
    )
    return fold_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task = get_task(args.task)
    if task.key != "task4_who_5class":
        raise ValueError("This script currently targets task4_who_5class.")
    image_df = load_task_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    print(
        f"Starting DINO+PLIP case fusion: task={task.key}, dino={args.dino_model_name}, probe={args.probe_type}, device={device}, folds={fold_ids}",
        flush=True,
    )
    dino_model = load_dinov2_model(Path(args.dino_repo_dir), args.dino_model_name, device)
    plip_model, plip_processor = load_plip_model(Path(args.plip_model_dir), device)
    for fold_id in fold_ids:
        run_single_fold(args, task, image_df, dino_model, plip_model, plip_processor, device, fold_id, available_folds)
    if len(fold_ids) > 1:
        aggregate_run_outputs(output_dir, task, fold_ids)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
