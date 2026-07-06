from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
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
from run_task56_dinov2_probe import get_task, load_task56_image_df
from thymic_baseline.config import DEFAULT_RANDOM_SEED
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


DEFAULT_DINO_IMAGE_SIZE = 518
TASK7_KEY = "task7_lowhigh_tc"
VIEW_TYPES = ("cut_surface", "outer_surface", "mixed", "unclear")
VIEW_MODES = ("cut_surface", "outer_surface", "cut_heavy", "outer_heavy", "balanced_mixed", "unclear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task7 formal view-specialist MLP on frozen DINO concat features.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", default=TASK7_KEY, choices=(TASK7_KEY,))
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", required=True, help="Comma-separated two DINO model names.")
    parser.add_argument("--feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--viewtype-seed-csv", required=True)
    parser.add_argument("--full-template-csv", required=True)
    parser.add_argument("--image-size", type=int, default=DEFAULT_DINO_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--global-loss-weight", type=float, default=0.4)
    parser.add_argument("--cut-loss-weight", type=float, default=0.4)
    parser.add_argument("--outer-loss-weight", type=float, default=0.4)
    parser.add_argument("--global-logit-bias", type=float, default=0.15)
    parser.add_argument("--mixed-margin", type=float, default=0.08)
    parser.add_argument("--balanced-train-weight", type=float, default=0.45)
    parser.add_argument("--cut-global-weight", type=float, default=0.15)
    parser.add_argument("--heavy-global-weight", type=float, default=0.25)
    parser.add_argument("--balanced-global-weight", type=float, default=0.30)
    return parser.parse_args()


class MultiHeadViewNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_head = nn.Linear(hidden_dim, 2)
        self.cut_head = nn.Linear(hidden_dim, 2)
        self.outer_head = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(inputs)
        return self.global_head(hidden), self.cut_head(hidden), self.outer_head(hidden)


class ViewSpecialistProbe:
    def __init__(
        self,
        scaler: StandardScaler,
        model: MultiHeadViewNet,
        device: torch.device,
        mixed_margin: float,
        cut_global_weight: float,
        heavy_global_weight: float,
        balanced_global_weight: float,
    ) -> None:
        self.scaler = scaler
        self.model = model
        self.device = device
        self.mixed_margin = mixed_margin
        self.cut_global_weight = cut_global_weight
        self.heavy_global_weight = heavy_global_weight
        self.balanced_global_weight = balanced_global_weight

    def predict_proba(self, features: np.ndarray, view_probs: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(features).astype(np.float32, copy=False)
        inputs = torch.from_numpy(transformed).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits_global, logits_cut, logits_outer = self.model(inputs)
            blend_logits = blend_logits_with_view_probs(
                logits_global,
                logits_cut,
                logits_outer,
                torch.from_numpy(view_probs.astype(np.float32, copy=False)).to(self.device),
                mixed_margin=self.mixed_margin,
                cut_global_weight=self.cut_global_weight,
                heavy_global_weight=self.heavy_global_weight,
                balanced_global_weight=self.balanced_global_weight,
            )
            probs = F.softmax(blend_logits, dim=1)
        return probs.cpu().numpy().astype(np.float64)


def fit_viewtype_classifier(features: np.ndarray, view_labels: np.ndarray) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(features, view_labels)
    return clf


def extract_concat_features(
    image_df: pd.DataFrame,
    model_names: list[str],
    models: list[torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    features_list: list[np.ndarray] = []
    labels_ref = None
    case_ids_ref = None
    image_names_ref = None
    for model_name, model in zip(model_names, models):
        features, labels, case_ids, image_names = extract_dataset_features(
            image_df,
            input_variant="whole",
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


def build_feature_frame(
    image_df: pd.DataFrame,
    full_features: np.ndarray,
    full_labels: np.ndarray,
    full_case_ids: list[str],
    full_image_names: list[str],
) -> pd.DataFrame:
    feat_df = pd.DataFrame(
        {
            "training_case_id": image_df["training_case_id"].astype(str).tolist(),
            "case_id": full_case_ids,
            "image_name": full_image_names,
            "label_idx": full_labels.tolist(),
        }
    )
    feat_df["feature_row"] = np.arange(len(feat_df))
    return feat_df


def add_view_modes(
    feat_df: pd.DataFrame,
    mixed_margin: float,
) -> pd.DataFrame:
    out = feat_df.copy()
    view_mode: list[str] = []
    for row in out.itertuples(index=False):
        pred_view_type = getattr(row, "pred_view_type")
        cut_prob = float(getattr(row, "view_prob_cut_surface"))
        outer_prob = float(getattr(row, "view_prob_outer_surface"))
        if pred_view_type == "mixed":
            diff = cut_prob - outer_prob
            if diff >= mixed_margin:
                view_mode.append("cut_heavy")
            elif diff <= -mixed_margin:
                view_mode.append("outer_heavy")
            else:
                view_mode.append("balanced_mixed")
        else:
            view_mode.append(str(pred_view_type))
    out["pred_view_mode"] = view_mode
    return out


def build_view_probs(
    feat_df: pd.DataFrame,
    full_features: np.ndarray,
    view_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, Pipeline]:
    labeled = feat_df.merge(view_seed, on="training_case_id", how="inner")
    view_to_idx = {name: idx for idx, name in enumerate(VIEW_TYPES)}
    labeled = labeled[labeled["view_type_seed"].isin(view_to_idx)].copy()
    labeled["view_idx"] = labeled["view_type_seed"].map(view_to_idx).astype(int)
    if len(labeled) < 20:
        raise ValueError("Not enough view-type seed labels.")

    view_clf = fit_viewtype_classifier(
        full_features[labeled["feature_row"].to_numpy()],
        labeled["view_idx"].to_numpy(),
    )
    view_pred_prob = view_clf.predict_proba(full_features)
    idx_to_view = {idx: name for name, idx in view_to_idx.items()}

    out = feat_df.copy()
    for idx, name in idx_to_view.items():
        out[f"view_prob_{name}"] = view_pred_prob[:, idx]
    pred_idx = view_pred_prob.argmax(axis=1)
    out["pred_view_type"] = [idx_to_view[int(x)] for x in pred_idx]
    out["pred_view_confidence"] = view_pred_prob.max(axis=1)
    return out, view_clf


def specialist_sample_weights(
    view_probs: torch.Tensor,
    mixed_margin: float,
    balanced_train_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_idx = view_probs.argmax(dim=1)
    cut_prob = view_probs[:, 0]
    outer_prob = view_probs[:, 1]
    mixed_prob = view_probs[:, 2]
    mixed_mask = pred_idx == 2
    cut_mask = pred_idx == 0
    outer_mask = pred_idx == 1

    diff = cut_prob - outer_prob
    cut_heavy_mask = mixed_mask & (diff >= mixed_margin)
    outer_heavy_mask = mixed_mask & (diff <= -mixed_margin)
    balanced_mask = mixed_mask & ~(cut_heavy_mask | outer_heavy_mask)

    cut_outer_sum = (cut_prob + outer_prob).clamp_min(1e-6)
    balanced_cut_share = cut_prob / cut_outer_sum
    balanced_outer_share = outer_prob / cut_outer_sum

    cut_weight = torch.zeros_like(cut_prob)
    outer_weight = torch.zeros_like(outer_prob)

    cut_weight = torch.where(cut_mask | cut_heavy_mask, torch.ones_like(cut_weight), cut_weight)
    outer_weight = torch.where(outer_mask | outer_heavy_mask, torch.ones_like(outer_weight), outer_weight)

    cut_weight = torch.where(
        balanced_mask,
        balanced_train_weight * balanced_cut_share + 0.15 * mixed_prob,
        cut_weight,
    )
    outer_weight = torch.where(
        balanced_mask,
        balanced_train_weight * balanced_outer_share + 0.15 * mixed_prob,
        outer_weight,
    )
    return cut_weight, outer_weight


def blend_logits_with_view_probs(
    logits_global: torch.Tensor,
    logits_cut: torch.Tensor,
    logits_outer: torch.Tensor,
    view_probs: torch.Tensor,
    mixed_margin: float,
    cut_global_weight: float,
    heavy_global_weight: float,
    balanced_global_weight: float,
) -> torch.Tensor:
    pred_idx = view_probs.argmax(dim=1)
    cut_prob = view_probs[:, 0]
    outer_prob = view_probs[:, 1]
    mixed_mask = pred_idx == 2
    cut_mask = pred_idx == 0
    outer_mask = pred_idx == 1
    unclear_mask = pred_idx == 3

    diff = cut_prob - outer_prob
    cut_heavy_mask = mixed_mask & (diff >= mixed_margin)
    outer_heavy_mask = mixed_mask & (diff <= -mixed_margin)
    balanced_mask = mixed_mask & ~(cut_heavy_mask | outer_heavy_mask)

    cut_outer_sum = (cut_prob + outer_prob).clamp_min(1e-6)
    balanced_cut_share = cut_prob / cut_outer_sum
    balanced_outer_share = outer_prob / cut_outer_sum

    cut_weight = torch.zeros_like(cut_prob)
    outer_weight = torch.zeros_like(outer_prob)
    global_weight = torch.ones_like(cut_prob)

    cut_weight = torch.where(cut_mask, torch.full_like(cut_weight, 1.0 - cut_global_weight), cut_weight)
    global_weight = torch.where(cut_mask, torch.full_like(global_weight, cut_global_weight), global_weight)

    outer_weight = torch.where(outer_mask, torch.full_like(outer_weight, 1.0 - cut_global_weight), outer_weight)
    global_weight = torch.where(outer_mask, torch.full_like(global_weight, cut_global_weight), global_weight)

    cut_weight = torch.where(cut_heavy_mask, torch.full_like(cut_weight, 1.0 - heavy_global_weight), cut_weight)
    global_weight = torch.where(cut_heavy_mask, torch.full_like(global_weight, heavy_global_weight), global_weight)

    outer_weight = torch.where(outer_heavy_mask, torch.full_like(outer_weight, 1.0 - heavy_global_weight), outer_weight)
    global_weight = torch.where(outer_heavy_mask, torch.full_like(global_weight, heavy_global_weight), global_weight)

    cut_weight = torch.where(
        balanced_mask,
        (1.0 - balanced_global_weight) * balanced_cut_share,
        cut_weight,
    )
    outer_weight = torch.where(
        balanced_mask,
        (1.0 - balanced_global_weight) * balanced_outer_share,
        outer_weight,
    )
    global_weight = torch.where(
        balanced_mask,
        torch.full_like(global_weight, balanced_global_weight),
        global_weight,
    )

    cut_weight = torch.where(unclear_mask, torch.zeros_like(cut_weight), cut_weight)
    outer_weight = torch.where(unclear_mask, torch.zeros_like(outer_weight), outer_weight)
    global_weight = torch.where(unclear_mask, torch.ones_like(global_weight), global_weight)

    total = cut_weight + outer_weight + global_weight + 1e-8
    cut_weight = (cut_weight / total).unsqueeze(1)
    outer_weight = (outer_weight / total).unsqueeze(1)
    global_weight = (global_weight / total).unsqueeze(1)
    return global_weight * logits_global + cut_weight * logits_cut + outer_weight * logits_outer


def build_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.from_numpy(weights)


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if sample_weights is not None:
        losses = losses * sample_weights
        denom = sample_weights.sum().clamp_min(1e-6)
        return losses.sum() / denom
    return losses.mean()


def fit_viewspecialist_probe(
    args: argparse.Namespace,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_view_probs: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_view_probs: np.ndarray,
    val_case_ids: list[str],
    val_image_names: list[str],
    task,
) -> tuple[ViewSpecialistProbe, int, float]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features).astype(np.float32, copy=False)
    val_x = scaler.transform(val_features).astype(np.float32, copy=False)

    device = resolve_device(args.device)
    model = MultiHeadViewNet(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = build_class_weights(train_labels, len(task.class_names)).to(device)

    train_inputs = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_labels.astype(np.int64, copy=False))
    train_view = torch.from_numpy(train_view_probs.astype(np.float32, copy=False))
    val_inputs = torch.from_numpy(val_x)
    val_view = torch.from_numpy(val_view_probs.astype(np.float32, copy=False))

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
            batch_view = train_view[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits_global, logits_cut, logits_outer = model(batch_inputs)
            final_logits = blend_logits_with_view_probs(
                logits_global,
                logits_cut,
                logits_outer,
                batch_view,
                mixed_margin=args.mixed_margin,
                cut_global_weight=args.cut_global_weight,
                heavy_global_weight=args.heavy_global_weight,
                balanced_global_weight=args.balanced_global_weight,
            )
            cut_sample_weight, outer_sample_weight = specialist_sample_weights(
                batch_view,
                mixed_margin=args.mixed_margin,
                balanced_train_weight=args.balanced_train_weight,
            )
            loss = weighted_ce(final_logits, batch_targets, class_weights)
            loss = loss + args.global_loss_weight * weighted_ce(logits_global, batch_targets, class_weights)
            loss = loss + args.cut_loss_weight * weighted_ce(logits_cut, batch_targets, class_weights, cut_sample_weight)
            loss = loss + args.outer_loss_weight * weighted_ce(logits_outer, batch_targets, class_weights, outer_sample_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_global, logits_cut, logits_outer = model(val_inputs.to(device))
            val_logits = blend_logits_with_view_probs(
                logits_global,
                logits_cut,
                logits_outer,
                val_view.to(device),
                mixed_margin=args.mixed_margin,
                cut_global_weight=args.cut_global_weight,
                heavy_global_weight=args.heavy_global_weight,
                balanced_global_weight=args.balanced_global_weight,
            )
            val_probs = F.softmax(val_logits, dim=1).cpu().numpy().astype(np.float64)
        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=val_labels,
            case_ids=val_case_ids,
            image_names=val_image_names,
            class_names=task.class_names,
        )
        case_df = aggregate_case_predictions(val_predictions, task.class_names, method="mean")
        metric_value = float(summarize_prediction_frame(case_df, task.class_names).get(task.primary_metric, float("nan")))

        if best_metric is None or (not math.isnan(metric_value) and metric_value > best_metric):
            best_metric = float(metric_value)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None or best_metric is None:
        raise RuntimeError("Failed to fit view-specialist probe.")
    model.load_state_dict(best_state)
    return (
        ViewSpecialistProbe(
            scaler=scaler,
            model=model,
            device=device,
            mixed_margin=args.mixed_margin,
            cut_global_weight=args.cut_global_weight,
            heavy_global_weight=args.heavy_global_weight,
            balanced_global_weight=args.balanced_global_weight,
        ),
        int(best_epoch),
        float(best_metric),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if len(model_names) != 2:
        raise ValueError("--model-names must contain exactly two model names.")

    full_template = pd.read_csv(args.full_template_csv, encoding="utf-8-sig")
    full_template["training_image_name"] = full_template["training_image_name"].astype(str)
    full_template["training_case_id"] = full_template["training_case_id"].astype(str)
    view_seed = pd.read_csv(args.viewtype_seed_csv, encoding="utf-8-sig")
    view_seed["training_case_id"] = view_seed["training_case_id"].astype(str)

    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task).copy()
    image_df["training_image_name"] = image_df["image_name"].astype(str)
    template_map = full_template[["training_image_name", "training_case_id"]].drop_duplicates()
    image_df = image_df.merge(template_map, on="training_image_name", how="left")
    if image_df["training_case_id"].isna().any():
        missing = image_df.loc[image_df["training_case_id"].isna(), "training_image_name"].head(10).tolist()
        raise ValueError(f"Missing training_case_id mapping for images: {missing}")
    image_df["training_case_id"] = image_df["training_case_id"].astype(str)

    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    repo_dir = Path(args.repo_dir)
    models = [load_dinov2_model(repo_dir=repo_dir, model_name=name, device=device) for name in model_names]

    all_df = trim_image_df(image_df, None)
    full_features, full_labels, full_case_ids, full_image_names = extract_concat_features(all_df, model_names, models, args, device)
    feat_df = build_feature_frame(all_df, full_features, full_labels, full_case_ids, full_image_names)
    feat_df, _ = build_view_probs(feat_df, full_features, view_seed)
    feat_df = add_view_modes(feat_df, mixed_margin=args.mixed_margin)
    feat_df.to_csv(output_dir / "predicted_view_types.csv", index=False, encoding="utf-8-sig")
    (output_dir / "predicted_view_counts.json").write_text(
        json.dumps(
            {
                "pred_view_type": feat_df["pred_view_type"].value_counts().to_dict(),
                "pred_view_mode": feat_df["pred_view_mode"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    oof_case_frames: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, Any]] = []

    for fold_id in fold_ids:
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
        train_df = trim_image_df(train_df, args.max_train_images)
        val_df = trim_image_df(val_df, args.max_val_images)
        test_df = trim_image_df(test_df, args.max_test_images)

        train_case_ids = set(train_df["training_case_id"].astype(str))
        val_case_ids = set(val_df["training_case_id"].astype(str))
        test_case_ids = set(test_df["training_case_id"].astype(str))

        train_rows = feat_df["training_case_id"].isin(train_case_ids)
        val_rows = feat_df["training_case_id"].isin(val_case_ids)
        test_rows = feat_df["training_case_id"].isin(test_case_ids)

        x_train = full_features[train_rows.to_numpy()]
        y_train = feat_df.loc[train_rows, "label_idx"].to_numpy(dtype=int)
        x_val = full_features[val_rows.to_numpy()]
        y_val = feat_df.loc[val_rows, "label_idx"].to_numpy(dtype=int)
        x_test = full_features[test_rows.to_numpy()]
        y_test = feat_df.loc[test_rows, "label_idx"].to_numpy(dtype=int)

        view_cols = [f"view_prob_{name}" for name in VIEW_TYPES]
        train_view_probs = feat_df.loc[train_rows, view_cols].to_numpy(dtype=np.float32)
        val_view_probs = feat_df.loc[val_rows, view_cols].to_numpy(dtype=np.float32)
        test_view_probs = feat_df.loc[test_rows, view_cols].to_numpy(dtype=np.float32)

        probe, best_epoch, best_val_metric = fit_viewspecialist_probe(
            args=args,
            train_features=x_train,
            train_labels=y_train,
            train_view_probs=train_view_probs,
            val_features=x_val,
            val_labels=y_val,
            val_view_probs=val_view_probs,
            val_case_ids=feat_df.loc[val_rows, "case_id"].astype(str).tolist(),
            val_image_names=feat_df.loc[val_rows, "image_name"].astype(str).tolist(),
            task=task,
        )

        val_predictions = build_prediction_df(
            probs=probe.predict_proba(x_val, val_view_probs),
            labels=y_val,
            case_ids=feat_df.loc[val_rows, "case_id"].astype(str).tolist(),
            image_names=feat_df.loc[val_rows, "image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )
        test_predictions = build_prediction_df(
            probs=probe.predict_proba(x_test, test_view_probs),
            labels=y_test,
            case_ids=feat_df.loc[test_rows, "case_id"].astype(str).tolist(),
            image_names=feat_df.loc[test_rows, "image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )

        val_metrics = save_split_outputs(fold_dir, "val", val_predictions, task, aggregate_methods)
        test_metrics = save_split_outputs(fold_dir, "test", test_predictions, task, aggregate_methods)
        write_json(
            fold_dir / "fold_summary.json",
            {
                "task": task.key,
                "fold_id": fold_id,
                "train_images": int(train_rows.sum()),
                "val_images": int(val_rows.sum()),
                "test_images": int(test_rows.sum()),
                "best_epoch": int(best_epoch),
                "best_val_primary_metric": float(best_val_metric),
                "pred_view_counts_train": feat_df.loc[train_rows, "pred_view_type"].value_counts().to_dict(),
                "pred_view_counts_test": feat_df.loc[test_rows, "pred_view_type"].value_counts().to_dict(),
                "pred_view_mode_counts_train": feat_df.loc[train_rows, "pred_view_mode"].value_counts().to_dict(),
                "pred_view_mode_counts_test": feat_df.loc[test_rows, "pred_view_mode"].value_counts().to_dict(),
                "val_case_mean": val_metrics["mean"],
                "test_case_mean": test_metrics["mean"],
            },
        )
        fold_case = pd.read_csv(fold_dir / "test_case_predictions_mean.csv")
        fold_case["fold_id"] = fold_id
        oof_case_frames.append(fold_case)
        fold_summary_rows.append({"fold_id": fold_id, **test_metrics["mean"]})
        print(
            f"[Fold {fold_id}] "
            f"best_epoch={best_epoch} | "
            f"test_auc={format_metric_value(test_metrics['mean'].get('auc'))} | "
            f"test_acc={format_metric_value(test_metrics['mean'].get('accuracy'))} | "
            f"test_bacc={format_metric_value(test_metrics['mean'].get('balanced_accuracy'))} | "
            f"test_f1={format_metric_value(test_metrics['mean'].get('f1'))}",
            flush=True,
        )

    oof_case = pd.concat(oof_case_frames, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    y_true = oof_case["label_idx"].astype(int).to_numpy()
    y_pred = oof_case["pred_idx"].astype(int).to_numpy()
    prob_high = oof_case["prob_high_risk_group"].astype(float).to_numpy()
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(
            0.5
            * (
                (tp / (tp + fn) if (tp + fn) else float("nan"))
                + (tn / (tn + fp) if (tn + fp) else float("nan"))
            )
        ),
        "auc": float(roc_auc_score(y_true, prob_high)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn) if (tp + fn) else float("nan")),
        "specificity": float(tn / (tn + fp) if (tn + fp) else float("nan")),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    pd.DataFrame(fold_summary_rows).to_csv(output_dir / "cv_fold_summary_manual.csv", index=False)
    oof_case.to_csv(output_dir / "oof_case_predictions_mean_manual.csv", index=False)
    (output_dir / "oof_metrics_manual.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if len(fold_ids) > 1:
            aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
    except Exception as exc:  # pragma: no cover
        print(f"[Warn] aggregate_run_outputs failed: {exc}", flush=True)
    print("Overall metrics:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

