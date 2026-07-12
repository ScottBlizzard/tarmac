from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_task7_anatomy_roi_e1_20260712 as e1  # noqa: E402

d1 = e1.d1
a1 = e1.a1


VARIANTS = ("multibag_mean", "multibag_consistency")


@dataclass(frozen=True)
class CaseMultiBagPlan:
    bags: tuple[tuple[a1.TileWindow, ...], ...]
    centers: tuple[tuple[e1.ScoredCenter, ...], ...]
    anatomy_centers: tuple[e1.ScoredCenter, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 F1 label-free multi-random-bag consistency classifier."
    )
    parser.add_argument("--registry-csv", default=str(a1.DEFAULT_REGISTRY))
    parser.add_argument("--split-csv", default=str(a1.DEFAULT_SPLIT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", default="old_data,third_batch")
    parser.add_argument("--model-name", default="vit_large_patch16_siglip_512.v2_webli")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--split-modes", default="fivefold,source_lodo")
    parser.add_argument("--bag-count", type=int, default=3)
    parser.add_argument("--roi-scales", default="0.40,0.25")
    parser.add_argument("--detail-scale", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--minimum-interior-tissue", type=float, default=0.82)
    parser.add_argument("--minimum-interface-tissue", type=float, default=0.25)
    parser.add_argument("--maximum-interface-tissue", type=float, default=0.90)
    parser.add_argument("--minimum-center-distance", type=float, default=0.20)
    parser.add_argument("--random-match-pool", type=int, default=8)
    parser.add_argument("--random-candidates", type=int, default=512)
    parser.add_argument("--random-coverage-slack", type=float, default=0.01)
    parser.add_argument("--neutral-background-value", type=int, default=127)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def bag_seed(args: argparse.Namespace, case_id: str, bag_index: int) -> int:
    if bag_index == 0:
        return d1.stable_seed(args.seed, case_id, "matched_random_roi")
    return d1.stable_seed(args.seed, case_id, "f1_multibag", bag_index)


def build_multibag_plans(
    args: argparse.Namespace, records: pd.DataFrame, output_dir: Path
) -> list[CaseMultiBagPlan]:
    if args.bag_count < 2:
        raise ValueError("F1 requires at least two independently sampled ROI bags")
    scales = [float(item) for item in args.roi_scales.split(",") if item.strip()]
    plans: list[CaseMultiBagPlan] = []
    rows: list[dict[str, Any]] = []
    progress = tqdm(records.iterrows(), total=len(records), desc="F1 random-bag plans")
    for index, row in progress:
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        specimen, mask = d1.specimen_and_mask(image)
        candidates = e1.score_candidates(specimen, mask, args.detail_scale, args.grid_size)
        anatomy_centers = e1.choose_anatomy_centers(candidates, args)
        bag_windows: list[tuple[a1.TileWindow, ...]] = []
        bag_centers: list[tuple[e1.ScoredCenter, ...]] = []
        for bag_index in range(args.bag_count):
            centers = e1.choose_matched_random_centers(
                mask,
                anatomy_centers,
                args,
                bag_seed(args, str(row["case_id"]), bag_index),
            )
            windows = tuple(
                d1.centered_window(mask, center.center_x, center.center_y, scale)
                for center in centers
                for scale in scales
            )
            bag_centers.append(centers)
            bag_windows.append(windows)
            for center_index, (center, target) in enumerate(
                zip(centers, anatomy_centers)
            ):
                for scale_index, scale in enumerate(scales):
                    window = windows[center_index * len(scales) + scale_index]
                    rows.append(
                        {
                            "feature_row": int(index),
                            "case_id": str(row["case_id"]),
                            "bag_index": bag_index,
                            "role": center.role,
                            "roi_index": center_index * len(scales) + scale_index,
                            "center_x": center.center_x,
                            "center_y": center.center_y,
                            "scale": scale,
                            "tissue_coverage": window.tissue_coverage,
                            "detail_tissue_coverage": center.tissue_coverage,
                            "target_center_x": target.center_x,
                            "target_center_y": target.center_y,
                            "target_detail_tissue_coverage": target.tissue_coverage,
                            "coverage_gap": abs(
                                center.tissue_coverage - target.tissue_coverage
                            ),
                            "random_fallback": center.fallback,
                            "anatomy_fallback": target.fallback,
                        }
                    )
        plans.append(
            CaseMultiBagPlan(
                tuple(bag_windows), tuple(bag_centers), tuple(anatomy_centers)
            )
        )
    plan_path = output_dir / "f1_random_bag_plan.csv"
    if plan_path.exists():
        plan_path.unlink()
    pd.DataFrame(rows).to_csv(plan_path, index=False, encoding="utf-8-sig")
    return plans


class MultiBagRoiViewDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        plans: list[CaseMultiBagPlan],
        transform: transforms.Compose,
        neutral_value: int,
    ) -> None:
        self.records = records
        self.plans = plans
        self.transform = transform
        self.neutral_value = neutral_value

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        specimen, mask = d1.specimen_and_mask(image)
        specimen = e1.neutralize_specimen(specimen, mask, self.neutral_value)
        views = [
            self.transform(a1.crop_relative(specimen, window))
            for bag in self.plans[index].bags
            for window in bag
        ]
        return index, torch.stack(views, dim=0)


def extract_multibag_features(
    args: argparse.Namespace,
    records: pd.DataFrame,
    plans: list[CaseMultiBagPlan],
    output_dir: Path,
) -> np.ndarray:
    device = torch.device(args.device)
    model = d1.create_backbone(args.model_name, "dense").eval().to(device)
    image_size = d1.model_image_size(model, args.image_size)
    mean, std = d1.model_normalization(model)
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            dummy_tokens = d1.extract_feature_tokens(model, dummy, "dense")
    token_count, feature_dim = map(int, dummy_tokens.shape[1:])
    views_per_bag = len(plans[0].bags[0])
    shape = (
        len(records),
        args.bag_count,
        views_per_bag,
        token_count,
        feature_dim,
    )
    estimated_gib = float(np.prod(shape) * np.dtype(np.float16).itemsize / (1024**3))
    print(f"[f1-roi-ram-bank] shape={shape} estimated_gib={estimated_gib:.2f}", flush=True)
    features = np.empty(shape, dtype=np.float16)
    loader_kwargs: dict[str, Any] = {
        "dataset": MultiBagRoiViewDataset(
            records, plans, transform, args.neutral_background_value
        ),
        "batch_size": args.extract_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    progress = tqdm(loader, desc="F1 random-bag ROI tokens", file=sys.stdout)
    for row_indices, view_batch in progress:
        batch_size, num_views, channels, height, width = view_batch.shape
        inputs = view_batch.reshape(
            batch_size * num_views, channels, height, width
        ).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            tokens = d1.extract_feature_tokens(model, inputs, "dense")
        tokens = tokens.reshape(
            batch_size,
            args.bag_count,
            views_per_bag,
            token_count,
            feature_dim,
        )
        indices = row_indices.numpy().astype(int)
        features[indices] = tokens.detach().cpu().numpy().astype(np.float16, copy=False)
        progress.set_postfix(cases=int(indices[-1]) + 1, total=len(records))
    progress.close()
    d1.write_json(
        output_dir / "f1_roi_ram_feature_config.json",
        {
            "model_name": args.model_name,
            "image_size": image_size,
            "feature_shape": list(shape),
            "estimated_ram_gib": estimated_gib,
            "disk_feature_bank_written": False,
        },
    )
    del model, dummy_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


class MultiBagFeatureDataset(Dataset):
    def __init__(
        self,
        base_features: np.ndarray,
        roi_features: np.ndarray,
        records: pd.DataFrame,
        indices: np.ndarray,
    ) -> None:
        self.base_features = base_features
        self.roi_features = roi_features
        self.records = records
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "base": torch.from_numpy(
                np.asarray(self.base_features[index], dtype=np.float16)
            ),
            "roi": torch.from_numpy(
                np.asarray(self.roi_features[index], dtype=np.float16)
            ),
            "label": torch.tensor(
                int(self.records.iloc[index]["label_idx"]), dtype=torch.long
            ),
        }


def make_loader(
    base_features: np.ndarray,
    roi_features: np.ndarray,
    records: pd.DataFrame,
    indices: np.ndarray,
    batch_size: int,
    train: bool,
    seed: int,
) -> DataLoader:
    dataset = MultiBagFeatureDataset(base_features, roi_features, records, indices)
    sampler = d1.balanced_sampler(records, indices, seed) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def forward_multibag(
    model: d1.GatedDirectHead, base: torch.Tensor, roi: torch.Tensor
) -> torch.Tensor:
    batch_size, bag_count, roi_views, token_count, feature_dim = roi.shape
    base_views = base.shape[1]
    base_expanded = base[:, None].expand(
        batch_size, bag_count, base_views, token_count, feature_dim
    )
    feature = torch.cat([base_expanded, roi], dim=2)
    logits, _ = model(
        feature.reshape(
            batch_size * bag_count,
            base_views + roi_views,
            token_count,
            feature_dim,
        )
    )
    return logits.reshape(batch_size, bag_count, 2)


def jensen_shannon_consistency(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1).clamp_min(1e-7)
    mean_probability = probabilities.mean(dim=1, keepdim=True).clamp_min(1e-7)
    divergence = probabilities * (
        probabilities.log() - mean_probability.log()
    )
    return divergence.sum(dim=-1).mean()


def evaluate_head(
    model: d1.GatedDirectHead,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            base = batch["base"].to(device, non_blocking=True)
            roi = batch["roi"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                bag_logits = forward_multibag(model, base, roi)
            mean_logits = bag_logits.float().mean(dim=1)
            probability = torch.softmax(mean_logits, dim=1)[:, 1].cpu().numpy()
            bag_probability = torch.softmax(bag_logits.float(), dim=2)[:, :, 1]
            probability_std = bag_probability.std(dim=1, correction=0).cpu().numpy()
            for index, label, prob, prob_std in zip(
                batch["index"].numpy().astype(int),
                batch["label"].numpy().astype(int),
                probability,
                probability_std,
            ):
                rows.append(
                    {
                        "feature_row": int(index),
                        "label_idx": int(label),
                        "prob_high": float(prob),
                        "bag_prob_std": float(prob_std),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["pred_idx"] = (frame["prob_high"] >= 0.5).astype(int)
    return frame


def train_head(
    args: argparse.Namespace,
    base_features: np.ndarray,
    roi_features: np.ndarray,
    records: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    variant: str,
    seed: int,
    output_dir: Path,
) -> tuple[d1.GatedDirectHead, dict[str, Any], pd.DataFrame]:
    a1.set_seed(seed)
    device = torch.device(args.device)
    num_views = base_features.shape[1] + roi_features.shape[2]
    model = d1.GatedDirectHead(
        feature_dim=base_features.shape[-1],
        num_views=num_views,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    ).to(device)
    train_loader = make_loader(
        base_features,
        roi_features,
        records,
        train_indices,
        args.head_batch_size,
        True,
        seed,
    )
    val_loader = make_loader(
        base_features,
        roi_features,
        records,
        val_indices,
        args.head_batch_size,
        False,
        seed,
    )
    test_loader = make_loader(
        base_features,
        roi_features,
        records,
        test_indices,
        args.head_batch_size,
        False,
        seed,
    )
    consistency_weight = (
        0.0 if variant == "multibag_mean" else args.consistency_weight
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.amp and device.type == "cuda"
    )
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        supervised_losses = []
        consistency_losses = []
        for batch in train_loader:
            base = batch["base"].to(device, non_blocking=True)
            roi = batch["roi"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                bag_logits = forward_multibag(model, base, roi)
                mean_logits = bag_logits.mean(dim=1)
                supervised = F.cross_entropy(mean_logits, label)
                consistency = jensen_shannon_consistency(bag_logits)
                loss = supervised + consistency_weight * consistency
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            consistency_losses.append(float(consistency.detach()))
        scheduler.step()
        validation = evaluate_head(model, val_loader, device, args.amp)
        val_bacc = float(
            balanced_accuracy_score(validation["label_idx"], validation["pred_idx"])
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "supervised_loss": float(np.mean(supervised_losses)),
                "consistency_loss": float(np.mean(consistency_losses)),
                "val_balanced_accuracy": val_bacc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_bacc > best_metric + 1e-12:
            best_metric = val_bacc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No best F1 head state was retained")
    model.load_state_dict(best_state)
    predictions = evaluate_head(model, test_loader, device, args.amp)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    predictions.to_csv(
        output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    torch.save(best_state, output_dir / "best_head.pt")
    summary = {
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_metric,
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "test_n": int(len(test_indices)),
        "num_views_per_bag": num_views,
        "bag_count": args.bag_count,
        "consistency_weight": consistency_weight,
    }
    d1.write_json(output_dir / "fold_summary.json", summary)
    return model, summary, predictions


def main() -> None:
    args = parse_args()
    a1.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = a1.load_records(args)
    metadata_path = output_dir / "metadata.csv"
    if metadata_path.exists():
        metadata_path.unlink()
    records.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    d1.write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "records": len(records),
            "variants": VARIANTS,
            "base_views": d1.BASE_VIEW_NAMES,
            "first_bag_reproduces": "E1 matched_random_roi plan",
            "training_target": "direct binary Task7 at 100% coverage",
            "bag_aggregation": "shared visual head followed by mean logits",
            "prohibited_inputs": [
                "source",
                "batch",
                "stage1_probability",
                "confidence",
                "difficulty",
                "true_subtype",
                "error_label",
            ],
            "disk_feature_bank_written": False,
        },
    )
    base_features, _, _ = d1.extract_base_features(args, records, output_dir)
    plans = build_multibag_plans(args, records, output_dir)
    roi_features = extract_multibag_features(args, records, plans, output_dir)
    if args.extract_only:
        print("[done] F1 base and multibag extraction smoke completed", flush=True)
        return

    all_summaries: list[dict[str, Any]] = []
    split_modes = [item.strip() for item in args.split_modes.split(",") if item.strip()]
    for split_mode in split_modes:
        if split_mode == "fivefold":
            fold_values: list[int | str] = sorted(
                records["master_fold_id"].unique().astype(int).tolist()
            )
        elif split_mode == "source_lodo":
            fold_values = sorted(records["source_dataset"].astype(str).unique().tolist())
        else:
            raise ValueError(f"Unsupported split mode: {split_mode}")
        if args.max_folds is not None:
            fold_values = fold_values[: args.max_folds]
        for fold_position, fold_value in enumerate(fold_values):
            fold_root = output_dir / split_mode / f"fold_{fold_position + 1}"
            complete = all(
                (fold_root / variant / "test_predictions.csv").is_file()
                for variant in VARIANTS
            )
            if complete:
                print(f"[skip] complete {split_mode} {fold_value}", flush=True)
                continue
            if fold_root.exists():
                shutil.rmtree(fold_root)
            train_indices, val_indices, test_indices, held_source = a1.fold_indices(
                records, split_mode, fold_position, fold_value
            )
            if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
                raise ValueError(f"Empty partition for {split_mode} {fold_value}")
            diagnostic_seed = d1.stable_seed(
                args.seed, split_mode, fold_value, "diagnostic"
            )
            for variant in VARIANTS:
                model, head_summary, predictions = train_head(
                    args,
                    base_features,
                    roi_features,
                    records,
                    train_indices,
                    val_indices,
                    test_indices,
                    variant,
                    diagnostic_seed,
                    fold_root / variant,
                )
                head_summary.update(
                    {
                        "variant": variant,
                        "split_mode": split_mode,
                        "fold_value": str(fold_value),
                        "held_source": held_source,
                    }
                )
                d1.write_json(fold_root / variant / "fold_summary.json", head_summary)
                del model, predictions
                if args.device == "cuda":
                    torch.cuda.empty_cache()

        for variant in VARIANTS:
            frames = []
            for fold_position, _ in enumerate(fold_values):
                path = (
                    output_dir
                    / split_mode
                    / f"fold_{fold_position + 1}"
                    / variant
                    / "test_predictions.csv"
                )
                if path.is_file():
                    frames.append(pd.read_csv(path, encoding="utf-8-sig"))
            if not frames:
                continue
            predictions = pd.concat(frames, ignore_index=True)
            variant_dir = output_dir / split_mode / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            overall = d1.summarize_variant(records, predictions, variant_dir)
            all_summaries.append(
                {"split_mode": split_mode, "variant": variant, **overall}
            )

    summary_path = output_dir / "F1_MULTIBAG_SUMMARY.csv"
    if summary_path.exists():
        summary_path.unlink()
    pd.DataFrame(all_summaries).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    d1.write_json(output_dir / "F1_MULTIBAG_COMPLETE.json", {"complete": True})


if __name__ == "__main__":
    main()
