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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
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
from thymic_baseline.config import DEFAULT_IMAGE_SIZE, DEFAULT_RANDOM_SEED, INPUT_VARIANTS
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


class TextAuxProbeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_main_classes: int, text_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(hidden_dim, num_main_classes)
        self.text_head = nn.Linear(hidden_dim, text_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.backbone(inputs)
        return self.main_head(shared), self.text_head(shared)


class TextAuxProbe:
    def __init__(self, scaler: StandardScaler, model: TextAuxProbeNet, device: torch.device) -> None:
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


def aggregate_text_aux_outputs(output_dir: Path, task, fold_ids: list[int]) -> None:
    fold_rows: list[dict[str, float | int]] = []
    oof_frames: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        fold_dir = output_dir / f"fold_{fold_id}"
        summary_path = fold_dir / "fold_summary.json"
        if not summary_path.exists():
            continue
        summary = pd.read_json(summary_path, typ="series")
        test_metrics = summary["test_case_mean"]
        fold_rows.append(
            {
                "fold_id": fold_id,
                "best_epoch": int(summary["best_epoch"]),
                "val_primary_metric": float(summary["best_val_primary_metric"]),
                "test_accuracy": float(test_metrics.get("accuracy", float("nan"))),
                "test_balanced_accuracy": float(test_metrics.get("balanced_accuracy", float("nan"))),
                "test_macro_f1": float(test_metrics.get("macro_f1", float("nan"))),
                "test_macro_auc": float(test_metrics.get("macro_auc", float("nan"))),
            }
        )
        case_path = fold_dir / "test_case_predictions_mean.csv"
        if case_path.exists():
            oof_frames.append(pd.read_csv(case_path))

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(output_dir / "cv_fold_summary.csv", index=False, encoding="utf-8-sig")

    if oof_frames:
        oof = pd.concat(oof_frames, ignore_index=True)
        oof.to_csv(output_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
        metrics = summarize_prediction_frame(oof, task.class_names)
        pd.DataFrame(
            [
                {
                    "split": "test_oof",
                    "level": "case",
                    "aggregation": "mean",
                    **metrics,
                }
            ]
        ).to_csv(output_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task6 DINO probe with text-derived auxiliary supervision from revised clue fields.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASK56))
    parser.add_argument("--input-variant", default="whole", choices=INPUT_VARIANTS)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--model-names", required=True, help="One or two comma-separated DINOv2 model names.")
    parser.add_argument("--feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--experience-label-csv", required=True)
    parser.add_argument("--text-columns", default="exp_round2_key_discriminative_clues,exp_round2_confounding_clues")
    parser.add_argument("--text-loss-weight", type=float, default=0.35)
    parser.add_argument("--text-max-features", type=int, default=256)
    parser.add_argument("--text-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
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


def build_text_target_bundle(
    case_ids: list[str],
    image_names: list[str],
    experience_df: pd.DataFrame,
    text_columns: list[str],
    max_features: int,
    text_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    exp = experience_df.copy()
    exp["case_id"] = exp["case_id"].astype(str)
    exp["image_name"] = exp["image_name"].astype(str)
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in exp.iterrows():
        mapping[(row["case_id"], row["image_name"])] = row.to_dict()

    texts: list[str] = []
    sample_indices: list[int] = []
    for idx, key in enumerate(zip(case_ids, image_names)):
        payload = mapping.get((str(key[0]), str(key[1])))
        if payload is None:
            continue
        parts = []
        for col in text_columns:
            val = str(payload.get(col, "")).strip()
            if val and val.lower() != "nan":
                parts.append(val)
        if parts:
            texts.append(" [SEP] ".join(parts))
            sample_indices.append(idx)

    targets = np.zeros((len(case_ids), max(1, text_dim)), dtype=np.float32)
    mask = np.zeros((len(case_ids),), dtype=bool)
    if not texts:
        return targets, mask, {"coverage": 0, "text_dim": max(1, text_dim), "svd_dim": 0}

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts)
    n_samples, n_features = tfidf.shape
    max_rank = max(1, min(n_samples - 1, n_features - 1, text_dim))
    if max_rank >= 2:
        reducer = TruncatedSVD(n_components=max_rank, random_state=0)
        dense = reducer.fit_transform(tfidf).astype(np.float32, copy=False)
        svd_dim = max_rank
    else:
        dense = tfidf.toarray().astype(np.float32, copy=False)
        if dense.shape[1] > text_dim:
            dense = dense[:, :text_dim]
        svd_dim = dense.shape[1]

    if dense.shape[1] < targets.shape[1]:
        padded = np.zeros((dense.shape[0], targets.shape[1]), dtype=np.float32)
        padded[:, : dense.shape[1]] = dense
        dense = padded

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.clip(norms, 1e-6, None)

    for pos, sample_idx in enumerate(sample_indices):
        targets[sample_idx] = dense[pos]
        mask[sample_idx] = True

    meta = {
        "coverage": int(mask.sum()),
        "text_dim": int(targets.shape[1]),
        "svd_dim": int(svd_dim),
        "vocab_size": int(len(vectorizer.vocabulary_)),
    }
    return targets, mask, meta


def fit_text_aux_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_text_targets: np.ndarray,
    train_text_mask: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
) -> tuple[TextAuxProbe, int, float]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features).astype(np.float32, copy=False)
    val_x = scaler.transform(val_features).astype(np.float32, copy=False)

    device = resolve_device(args.device)
    model = TextAuxProbeNet(
        input_dim=train_x.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_main_classes=len(task.class_names),
        text_dim=train_text_targets.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_counts = np.bincount(train_labels, minlength=len(task.class_names)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    main_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

    train_inputs = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_labels.astype(np.int64, copy=False))
    aux_targets = torch.from_numpy(train_text_targets.astype(np.float32, copy=False))
    aux_mask = torch.from_numpy(train_text_mask.astype(bool, copy=False))
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
            main_logits, text_pred = model(batch_inputs)
            loss = main_criterion(main_logits, batch_targets)
            batch_mask = aux_mask[batch_idx].to(device)
            if batch_mask.any():
                target = aux_targets[batch_idx].to(device)
                text_loss = F.mse_loss(text_pred[batch_mask], target[batch_mask])
                loss = loss + args.text_loss_weight * text_loss
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
        raise RuntimeError("Failed to fit text-aux probe.")
    model.load_state_dict(best_state)
    return TextAuxProbe(scaler=scaler, model=model, device=device), int(best_epoch), float(best_metric)


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
    text_columns: list[str],
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

    train_text_targets, train_text_mask, aux_meta = build_text_target_bundle(
        train_case_ids,
        train_image_names,
        experience_df,
        text_columns=text_columns,
        max_features=args.text_max_features,
        text_dim=args.text_dim,
    )

    run_config = {
        "task": task.key,
        "input_variant": args.input_variant,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "model_names": model_names,
        "feature_mode": args.feature_mode,
        "text_columns": text_columns,
        "text_loss_weight": float(args.text_loss_weight),
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
        "text_aux_meta": aux_meta,
    }
    write_json(fold_dir / "run_config.json", run_config)
    print(
        f"\n=== Task56 TextAux Fold {fold_id}/{fold_count} | task={task.key} | models={model_names} ===",
        flush=True,
    )
    print(
        f"train_cases={run_config['train_cases']}, val_cases={run_config['val_cases']}, test_cases={run_config['test_cases']}, "
        f"text_aux_coverage={aux_meta['coverage']}, text_dim={aux_meta['text_dim']}",
        flush=True,
    )

    probe, best_epoch, best_val_metric = fit_text_aux_probe(
        args=args,
        train_features=train_features,
        train_labels=train_labels,
        train_text_targets=train_text_targets,
        train_text_mask=train_text_mask,
        val_features=val_features,
        val_labels=val_labels,
        val_case_ids=val_case_ids,
        val_image_names=val_image_names,
        task=task,
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
        "best_epoch": int(best_epoch),
        "best_val_primary_metric": float(best_val_metric),
        "val_case_mean": val_case_metrics["mean"],
        "test_case_mean": test_case_metrics["mean"],
        "text_aux_meta": aux_meta,
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
    text_columns = [item.strip() for item in args.text_columns.split(",") if item.strip()]
    for col in text_columns:
        if col not in experience_df.columns:
            raise KeyError(f"Experience label CSV missing column: {col}")

    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    device = resolve_device(args.device)
    repo_dir = Path(args.repo_dir)
    models = [load_dinov2_model(repo_dir=repo_dir, model_name=name, device=device) for name in model_names]

    print(
        f"Starting Task56 text-aux probe: task={task.key}, models={model_names}, "
        f"text_columns={text_columns}, text_loss_weight={args.text_loss_weight}, device={device}, folds={fold_ids}",
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
            text_columns=text_columns,
        )

    if len(fold_ids) > 1:
        aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
        try:
            aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
        except Exception as exc:
            print(f"aggregate_run_outputs failed ({exc}); falling back to text-aux manual aggregation.", flush=True)
            aggregate_text_aux_outputs(output_dir, task, fold_ids)
        print(f"Aggregated {len(fold_ids)} folds into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
