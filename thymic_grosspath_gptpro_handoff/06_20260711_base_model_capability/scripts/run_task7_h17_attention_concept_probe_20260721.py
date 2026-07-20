from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_task7_h17_historical_concept_probe_20260721 import (
    CONCEPTS,
    SEED,
    SOURCES,
    binary_metrics,
    fold_indices,
    load_inputs,
    safe_binary_metrics,
    write_csv,
    write_json,
)


ROLE_NAMES = (
    "specimen",
    "core",
    "outer_ring",
    "largest_component",
    "largest_1",
    "largest_2",
    "largest_3",
    "darkest_1",
    "darkest_2",
    "darkest_3",
    "palest_1",
    "palest_2",
    "high_gradient_1",
    "high_gradient_2",
    "maroon_1",
    "maroon_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered H17-0 frozen-region attention probe."
    )
    parser.add_argument("--region-assets-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--concept-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--role-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenRegionAttentionProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        role_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.region_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), role_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + role_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, regions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.region_projection(regions)
        role_ids = torch.arange(
            len(ROLE_NAMES), device=regions.device, dtype=torch.long
        )
        role = self.role_embedding(role_ids)[None].expand(regions.shape[0], -1, -1)
        attention_logits = self.attention(torch.cat([projected, role], dim=-1))
        attention = torch.softmax(attention_logits.squeeze(-1), dim=1)
        pooled = torch.sum(projected * attention[..., None], dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, attention


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=2).float()
    class_weights = labels.numel() / (2.0 * counts.clamp_min(1.0))
    sample_weights = class_weights.gather(0, labels.long())
    losses = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        reduction="none",
    )
    return (losses * sample_weights).mean()


@torch.no_grad()
def predict(
    model: nn.Module,
    features: torch.Tensor,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, attention = model(features[torch.as_tensor(indices, device=features.device)])
    return (
        torch.sigmoid(logits).cpu().numpy(),
        attention.cpu().numpy(),
    )


def train_fold(
    fold: int,
    features: torch.Tensor,
    metadata: pd.DataFrame,
    concepts: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels_all = concepts[CONCEPTS[0]].to_numpy(dtype=int)
    known_all = labels_all >= 0
    train, validation, test = fold_indices(metadata, fold)
    train_known = train[known_all[train]]
    validation_known = validation[known_all[validation]]
    if np.unique(labels_all[train_known]).size < 2:
        raise ValueError(f"Fold {fold} train contains one concept class")
    if np.unique(labels_all[validation_known]).size < 2:
        raise ValueError(f"Fold {fold} validation contains one concept class")

    fold_seed = args.seed + 1000 * fold
    set_seed(fold_seed)
    model = FrozenRegionAttentionProbe(
        feature_dim=int(features.shape[-1]),
        hidden_dim=args.hidden_dim,
        role_dim=args.role_dim,
        dropout=args.dropout,
    ).to(features.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_index = torch.as_tensor(train_known, device=features.device)
    train_labels = torch.as_tensor(
        labels_all[train_known],
        dtype=torch.long,
        device=features.device,
    )

    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features[train_index])
        loss = weighted_bce(logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        validation_probability, _ = predict(model, features, validation_known)
        validation_metrics = binary_metrics(
            labels_all[validation_known],
            validation_probability,
        )
        key = (
            float(validation_metrics["balanced_accuracy"]),
            float(validation_metrics["auc"]),
            -float(epoch),
        )
        improved = best_key is None or key > best_key
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "improved": improved,
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
        )
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= 30 and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not create a checkpoint")
    model.load_state_dict(best_state)
    probability, attention = predict(model, features, test)
    test_known = known_all[test]
    test_metrics = binary_metrics(
        labels_all[test][test_known],
        probability[test_known],
    )

    prediction_frame = metadata.iloc[test][
        ["feature_row", "case_id", "source_dataset", "master_fold_id"]
    ].copy()
    prediction_frame["concept"] = CONCEPTS[0]
    prediction_frame["state"] = labels_all[test]
    prediction_frame["known"] = prediction_frame["state"].ge(0)
    prediction_frame["probability"] = probability
    prediction_frame["prediction"] = (probability >= 0.5).astype(int)
    prediction_frame["correct"] = np.where(
        prediction_frame["known"],
        prediction_frame["prediction"].eq(prediction_frame["state"]),
        False,
    )

    attention_frame = prediction_frame[
        ["feature_row", "case_id", "source_dataset", "master_fold_id", "known", "state"]
    ].copy()
    for role_index, role_name in enumerate(ROLE_NAMES):
        attention_frame[f"attention_{role_name}"] = attention[:, role_index]
    attention_frame["attention_canonical_0_2"] = attention[:, :3].sum(axis=1)
    attention_frame["attention_outer_ring"] = attention[:, 2]
    attention_frame["attention_chromatic_7_15"] = attention[:, 7:].sum(axis=1)

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_csv(fold_dir / "training_history.csv", pd.DataFrame(history))
    checkpoint = {
        "model_state": {key: value.cpu() for key, value in best_state.items()},
        "fold": fold,
        "best_epoch": best_epoch,
        "seed": fold_seed,
        "risk_labels_read": False,
        "subtype_labels_read": False,
    }
    temporary = fold_dir / "best_probe.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, fold_dir / "best_probe.pt")
    summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "train_known": int(len(train_known)),
        "validation_known": int(len(validation_known)),
        "test_known": int(test_known.sum()),
        "test_metrics": test_metrics,
    }
    write_json(fold_dir / "fold_summary.json", summary)
    return prediction_frame, attention_frame, summary


def summarize(
    predictions: pd.DataFrame,
    attention: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    known = predictions.loc[predictions["known"]].copy()
    overall = binary_metrics(
        known["state"].to_numpy(dtype=int),
        known["probability"].to_numpy(dtype=float),
    )
    fold_rows: list[dict[str, Any]] = []
    for fold, group in known.groupby("master_fold_id", sort=True):
        fold_rows.append(
            {
                "fold": int(fold),
                **safe_binary_metrics(
                    group["state"].to_numpy(dtype=int),
                    group["probability"].to_numpy(dtype=float),
                ),
            }
        )
    source_rows: list[dict[str, Any]] = []
    for source, group in known.groupby("source_dataset", sort=False):
        source_rows.append(
            {
                "source_dataset": str(source),
                **safe_binary_metrics(
                    group["state"].to_numpy(dtype=int),
                    group["probability"].to_numpy(dtype=float),
                ),
            }
        )

    known_attention = attention.loc[attention["known"]].copy()
    attention_summary = {
        role_name: float(known_attention[f"attention_{role_name}"].mean())
        for role_name in ROLE_NAMES
    }
    attention_summary.update(
        {
            "canonical_0_2_mean": float(
                known_attention["attention_canonical_0_2"].mean()
            ),
            "outer_ring_mean": float(
                known_attention["attention_outer_ring"].mean()
            ),
            "chromatic_7_15_mean": float(
                known_attention["attention_chromatic_7_15"].mean()
            ),
        }
    )
    overall["attention"] = attention_summary
    return overall, pd.DataFrame(fold_rows), pd.DataFrame(source_rows)


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError(f"H17 attention probe seed is locked to {SEED}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    set_seed(args.seed)

    region_assets_dir = Path(args.region_assets_dir)
    metadata_path = Path(args.metadata_csv)
    concept_path = Path(args.concept_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "RUN.status"
    status_path.write_text("running_h17_0_attention\n", encoding="utf-8")

    metadata, concepts, region_features, _, _ = load_inputs(
        region_assets_dir,
        metadata_path,
        concept_path,
    )
    features = torch.as_tensor(
        np.asarray(region_features, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    features = F.normalize(features, dim=-1)

    prediction_frames: list[pd.DataFrame] = []
    attention_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for fold in range(1, 6):
        predictions, attention, summary = train_fold(
            fold,
            features,
            metadata,
            concepts,
            args,
            output_dir,
        )
        prediction_frames.append(predictions)
        attention_frames.append(attention)
        summaries.append(summary)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    attention = pd.concat(attention_frames, ignore_index=True)
    overall, fold_metrics, source_metrics = summarize(predictions, attention)
    fold_auc = fold_metrics["auc"].to_numpy(dtype=float)
    source_auc = source_metrics["auc"].to_numpy(dtype=float)
    checks = {
        "auc_ge_0_75": bool(overall["auc"] >= 0.75),
        "balanced_accuracy_ge_0_68": bool(overall["balanced_accuracy"] >= 0.68),
        "folds_auc_gt_0_70_ge_4": bool((fold_auc > 0.70).sum() >= 4),
        "all_sources_auc_ge_0_65": bool(
            len(source_auc) == 3 and (source_auc >= 0.65).all()
        ),
        "canonical_attention_above_uniform": bool(
            overall["attention"]["canonical_0_2_mean"] > 3.0 / len(ROLE_NAMES)
        ),
    }
    passed = all(checks.values())
    decision = {
        "status": (
            "PASS_H17_0_ATTENTION_PROCEED_60_CASE_MORPHOLOGY_PILOT"
            if passed
            else "FAIL_H17_0_ATTENTION_RUN_20_CASE_VISIBILITY_MINIPILOT"
        ),
        "passed": passed,
        "overall": overall,
        "fold_auc": [float(value) for value in fold_auc],
        "source_auc": {
            str(row["source_dataset"]): float(row["auc"])
            for _, row in source_metrics.iterrows()
        },
        "checks": checks,
        "c1_c2_exact_complements": True,
        "independent_historical_concept_dimensions": 1,
        "risk_labels_read": False,
        "subtype_labels_read": False,
        "strict_external_read": False,
    }
    config = {
        "experiment": "H17_0_FROZEN_REGION_ATTENTION_PROBE_20260721",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "hidden_dim": args.hidden_dim,
        "role_dim": args.role_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "roles": list(ROLE_NAMES),
        "frozen_region_features": True,
        "risk_labels_read": False,
        "subtype_labels_read": False,
        "strict_external_read": False,
    }
    write_csv(output_dir / "oof_attention_concept_predictions_server_only.csv", predictions)
    write_csv(output_dir / "oof_region_attention_server_only.csv", attention)
    write_csv(output_dir / "fold_metrics.csv", fold_metrics)
    write_csv(output_dir / "source_metrics.csv", source_metrics)
    write_json(output_dir / "fold_summaries.json", summaries)
    write_json(output_dir / "h17_0_attention_config.json", config)
    write_json(output_dir / "h17_0_attention_gate.json", decision)
    status_path.write_text(decision["status"] + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
