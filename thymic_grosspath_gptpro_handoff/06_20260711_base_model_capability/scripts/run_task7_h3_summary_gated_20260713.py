from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_task7_spatial_relational_20260713 import (
    EXPECTED_SOURCES,
    GatedTokenPool,
    fold_partitions,
    load_metadata,
    make_loader,
    metric_record,
    predict,
    set_seed,
    source_risk_sampler,
    summarize_predictions,
    train_model,
    write_json,
)


EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
EXPECTED_STATISTICS = ("mean", "std", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked H3A summary-token gated representation screen."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--split-mode", choices=("fivefold", "source_lodo"), required=True)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class SummaryGatedHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        num_statistics: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, hidden_dim))
        self.statistic_embeddings = nn.Parameter(torch.zeros(num_statistics, hidden_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)
        nn.init.normal_(self.statistic_embeddings, std=0.02)
        self.pool = GatedTokenPool(hidden_dim, attention_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, features: torch.Tensor, view_bounds: torch.Tensor) -> torch.Tensor:
        del view_bounds
        batch_size, num_views, num_statistics, _ = features.shape
        tokens = self.project(self.input_norm(features))
        tokens = tokens + self.view_embeddings[:num_views].view(1, num_views, 1, -1)
        tokens = tokens + self.statistic_embeddings[:num_statistics].view(
            1, 1, num_statistics, -1
        )
        pooled = self.pool(tokens.reshape(batch_size, num_views * num_statistics, -1))
        return self.classifier(self.output_norm(pooled))


def prediction_frame(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    fold_id: int,
    held_source: str,
    candidate: str,
    split_role: str,
) -> pd.DataFrame:
    order = np.argsort(indices)
    indices = indices[order]
    labels = labels[order]
    probability = probability[order]
    result = metadata.iloc[indices][
        ["feature_row", "case_id", "source_dataset", "task_l6_label", "label_idx"]
    ].copy()
    if not np.array_equal(labels, result["label_idx"].to_numpy(dtype=int)):
        raise RuntimeError(f"{split_role} labels do not align with metadata")
    result["fold_id"] = fold_id
    result["held_out_source"] = held_source
    result["candidate"] = candidate
    result["split_role"] = split_role
    result["prob_high"] = probability
    result["pred_idx"] = (probability >= 0.5).astype(int)
    result["correct"] = result["pred_idx"].eq(result["label_idx"])
    return result


def run_fold(
    fold_id: int,
    features: np.ndarray,
    view_bounds: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = output_dir / f"fold_{fold_id}"
    test_path = fold_dir / "test_predictions.csv"
    val_path = fold_dir / "validation_predictions.csv"
    summary_path = fold_dir / "fold_summary.json"
    if test_path.exists() and val_path.exists() and summary_path.exists():
        print(f"[resume] {args.candidate} {args.split_mode} fold {fold_id}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_csv(
            test_path, dtype={"case_id": str}
        )

    train_indices, val_indices, test_indices, held_source = fold_partitions(
        metadata, args.split_mode, fold_id
    )
    fold_seed = args.seed + 1000 * fold_id
    set_seed(fold_seed)
    device = torch.device(args.device)
    model = SummaryGatedHead(
        feature_dim=int(features.shape[-1]),
        num_views=int(features.shape[1]),
        num_statistics=int(features.shape[2]),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    sampler = source_risk_sampler(metadata, train_indices, fold_seed + 1)
    train_loader = make_loader(
        features,
        view_bounds,
        metadata,
        train_indices,
        None,
        args.batch_size,
        args.num_workers,
        sampler,
    )
    val_loader = make_loader(
        features,
        view_bounds,
        metadata,
        val_indices,
        None,
        args.batch_size,
        args.num_workers,
        None,
    )
    test_loader = make_loader(
        features,
        view_bounds,
        metadata,
        test_indices,
        None,
        args.batch_size,
        args.num_workers,
        None,
    )
    trained = train_model(model, train_loader, val_loader, fold_dir, args, device)
    val_index, val_label, val_probability = predict(trained.model, val_loader, device)
    test_index, test_label, test_probability = predict(trained.model, test_loader, device)
    val_frame = prediction_frame(
        metadata,
        val_index,
        val_label,
        val_probability,
        fold_id,
        held_source,
        args.candidate,
        "validation",
    )
    test_frame = prediction_frame(
        metadata,
        test_index,
        test_label,
        test_probability,
        fold_id,
        held_source,
        args.candidate,
        "test",
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    val_frame.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_frame.to_csv(test_path, index=False, encoding="utf-8-sig")
    summary = {
        "fold_id": fold_id,
        "split_mode": args.split_mode,
        "held_out_source": held_source,
        "candidate": args.candidate,
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "best_epoch": trained.best_epoch,
        "best_val_bacc": trained.best_val_bacc,
        "validation_metrics": metric_record(val_label, val_probability),
        "test_metrics": metric_record(test_label, test_probability),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    write_json(summary_path, summary)
    del trained, model, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, test_frame


def main() -> None:
    args = parse_args()
    args.variant = args.candidate
    if args.seed not in (20260713, 20260714):
        raise ValueError("Only the preregistered primary and confirmation seeds are allowed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    feature_bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_config_path = feature_bank_dir / "representation_bank_config.json"
    bank_config = json.loads(bank_config_path.read_text(encoding="utf-8"))
    if bank_config.get("complete") is not True:
        raise ValueError("Representation bank is not complete")
    if tuple(bank_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError("Representation bank view order differs from H3 lock")
    if tuple(bank_config.get("statistics", [])) != EXPECTED_STATISTICS:
        raise ValueError("Representation statistic order differs from H3 lock")
    metadata = load_metadata(feature_bank_dir, Path(args.split_csv))
    features = np.load(feature_bank_dir / "summary_features.float16.npy", mmap_mode="r")
    expected_prefix = (591, len(EXPECTED_VIEWS), len(EXPECTED_STATISTICS))
    if features.shape[:3] != expected_prefix or features.dtype != np.float16:
        raise ValueError(f"Unexpected H3 feature bank: {features.shape} {features.dtype}")
    if not np.isfinite(np.asarray(features)).all():
        raise ValueError("H3 representation bank contains non-finite values")
    view_bounds = np.zeros((len(metadata), len(EXPECTED_VIEWS), 4), dtype=np.float32)
    run_config = vars(args).copy()
    run_config.update(
        {
            "feature_shape": list(features.shape),
            "bank_config_sha256": sha256_file(bank_config_path),
            "source_counts": metadata["source_dataset"].value_counts().sort_index().to_dict(),
            "subtype_counts": metadata["task_l6_label"].value_counts().to_dict(),
            "expected_sources": list(EXPECTED_SOURCES),
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in SummaryGatedHead(
                        features.shape[-1],
                        features.shape[1],
                        features.shape[2],
                        args.hidden_dim,
                        args.attention_dim,
                        args.dropout,
                    ).parameters()
                )
            ),
        }
    )
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise ValueError("Existing output directory has a different locked configuration")
    else:
        write_json(config_path, run_config)

    folds = [1, 2, 3] if args.split_mode == "source_lodo" else [1, 2, 3, 4, 5]
    if args.fold != "all":
        requested = int(args.fold)
        if requested not in folds:
            raise ValueError(f"Invalid fold {requested} for {args.split_mode}")
        folds = [requested]
    summaries = []
    prediction_frames = []
    for fold_id in folds:
        summary, frame = run_fold(
            fold_id,
            features,
            view_bounds,
            metadata,
            output_dir,
            args,
        )
        summaries.append(summary)
        prediction_frames.append(frame)
    pd.DataFrame(summaries).to_json(
        output_dir / "fold_summaries.json", orient="records", indent=2
    )
    if len(folds) == (3 if args.split_mode == "source_lodo" else 5):
        summarize_predictions(pd.concat(prediction_frames, ignore_index=True), output_dir)
        (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    else:
        (output_dir / "RUN.status").write_text("partial\n", encoding="utf-8")


if __name__ == "__main__":
    main()
