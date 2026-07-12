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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_task7_attention_roi_d1_20260712 as d1  # noqa: E402

a1 = d1.a1


VARIANTS = ("base_only", "physician_roi", "matched_random_roi")


@dataclass(frozen=True)
class CaseOraclePlan:
    physician: tuple[a1.TileWindow, ...]
    random: tuple[a1.TileWindow, ...]
    reader_has_roi: tuple[bool, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 G1 two-reader physician ROI oracle direct-model evaluation."
    )
    parser.add_argument("--locked-annotations", required=True)
    parser.add_argument("--secure-key", required=True)
    parser.add_argument("--split-csv", default=str(a1.DEFAULT_SPLIT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="vit_large_patch16_siglip_512.v2_webli")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--split-modes", default="fivefold,source_lodo")
    parser.add_argument("--context-expansion", type=float, default=1.50)
    parser.add_argument("--random-candidates", type=int, default=512)
    parser.add_argument("--random-match-pool", type=int, default=8)
    parser.add_argument("--minimum-random-distance", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def load_oracle_records(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.read_csv(
        args.locked_annotations,
        dtype={"oracle_id": str, "reader_id": str},
        encoding="utf-8-sig",
    )
    secure = pd.read_csv(
        args.secure_key,
        dtype={"oracle_id": str, "case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    if len(annotations) != 240 or annotations.duplicated(["oracle_id", "reader_id"]).any():
        raise ValueError("Locked annotations must contain 240 unique oracle-reader rows")
    if set(annotations["reader_id"]) != {"reader_1", "reader_2"}:
        raise ValueError("Locked annotations must contain reader_1 and reader_2")
    if len(secure) != 120 or secure["oracle_id"].duplicated().any():
        raise ValueError("Secure key must contain 120 unique oracle IDs")
    if set(secure["oracle_id"]) != set(annotations["oracle_id"]):
        raise ValueError("Locked annotation and secure-key oracle IDs differ")
    if set(annotations["annotation_status"].astype(str).str.lower()) != {"complete"}:
        raise ValueError("Every annotation row must be complete")

    locked_split = pd.read_csv(
        args.split_csv, dtype={"case_id": str}, encoding="utf-8-sig"
    )[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    records = secure.merge(locked_split, on="case_id", how="left", validate="one_to_one")
    if records["master_fold_id"].isna().any():
        raise ValueError("Secure oracle cases are missing locked master folds")
    records["master_fold_id"] = records["master_fold_id"].astype(int)
    records["label_idx"] = records["label_idx"].astype(int)
    records = records.sort_values("oracle_id").reset_index(drop=True)
    if args.max_cases is not None:
        records = records.head(int(args.max_cases)).copy()
        annotations = annotations[annotations["oracle_id"].isin(records["oracle_id"])].copy()
    missing_images = [path for path in records["image_path"].astype(str) if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing oracle images: {missing_images[:5]}")
    records.insert(0, "feature_row", np.arange(len(records), dtype=int))
    return records, annotations


def annotation_box(row: pd.Series, prefix: str) -> tuple[float, float, float, float] | None:
    columns = [
        f"{prefix}_x1_norm",
        f"{prefix}_y1_norm",
        f"{prefix}_x2_norm",
        f"{prefix}_y2_norm",
    ]
    values = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).all():
        return None
    if np.isnan(values).any():
        raise ValueError(f"Partial locked box for {row['oracle_id']} {row['reader_id']} {prefix}")
    x1, y1, x2, y2 = values.tolist()
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Invalid locked box for {row['oracle_id']} {row['reader_id']} {prefix}")
    return x1, y1, x2, y2


def expand_bounds(
    bounds: tuple[float, float, float, float], factor: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bounds
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = min(1.0, (x2 - x1) * factor)
    height = min(1.0, (y2 - y1) * factor)
    left = min(1.0 - width, max(0.0, center_x - width / 2.0))
    top = min(1.0 - height, max(0.0, center_y - height / 2.0))
    return left, top, left + width, top + height


def make_window(
    mask: np.ndarray, bounds: tuple[float, float, float, float]
) -> a1.TileWindow:
    return a1.TileWindow(*bounds, a1.normalized_window_coverage(mask, bounds))


def tissue_mask_for_image(image: Image.Image) -> np.ndarray:
    thumbnail = image.copy()
    thumbnail.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    mask = a1.detect_specimen_mask(np.asarray(thumbnail, dtype=np.uint8)) > 0
    if mask.size == 0:
        return np.ones((8, 8), dtype=bool)
    return mask


def matched_random_bounds(
    mask: np.ndarray,
    target: tuple[float, float, float, float],
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = target
    width = x2 - x1
    height = y2 - y1
    target_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    target_coverage = a1.normalized_window_coverage(mask, target)
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(args.random_candidates):
        center_x = float(rng.uniform(width / 2.0, 1.0 - width / 2.0))
        center_y = float(rng.uniform(height / 2.0, 1.0 - height / 2.0))
        if math.dist((center_x, center_y), target_center) < args.minimum_random_distance:
            continue
        bounds = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
        coverage = a1.normalized_window_coverage(mask, bounds)
        candidates.append((abs(coverage - target_coverage), bounds))
    if not candidates:
        return target
    candidates.sort(key=lambda item: (item[0], *item[1]))
    short = candidates[: min(args.random_match_pool, len(candidates))]
    return short[int(rng.integers(0, len(short)))][1]


def build_oracle_plans(
    args: argparse.Namespace,
    records: pd.DataFrame,
    annotations: pd.DataFrame,
    output_dir: Path,
) -> list[CaseOraclePlan]:
    annotation_lookup = {
        (str(row["oracle_id"]), str(row["reader_id"])): row
        for _, row in annotations.iterrows()
    }
    plans = []
    rows = []
    for index, record in tqdm(records.iterrows(), total=len(records), desc="G1 ROI plans"):
        with Image.open(str(record["image_path"])) as source:
            image = source.convert("RGB")
        mask = tissue_mask_for_image(image)
        physician_windows = []
        random_windows = []
        has_roi = []
        for reader_index, reader_id in enumerate(("reader_1", "reader_2")):
            annotation = annotation_lookup[(str(record["oracle_id"]), reader_id)]
            roi = annotation_box(annotation, "roi1")
            reader_has_roi = (
                str(annotation["no_visually_diagnostic_roi"]).strip().lower() == "no"
                and roi is not None
            )
            has_roi.append(reader_has_roi)
            if not reader_has_roi:
                detail_bounds = (0.0, 0.0, 1.0, 1.0)
                context_bounds = detail_bounds
                random_detail = detail_bounds
                random_context = context_bounds
            else:
                detail_bounds = roi
                context_bounds = expand_bounds(detail_bounds, args.context_expansion)
                random_detail = matched_random_bounds(
                    mask,
                    detail_bounds,
                    args,
                    d1.stable_seed(args.seed, record["oracle_id"], reader_id, "random"),
                )
                random_context = expand_bounds(random_detail, args.context_expansion)
            manual_pair = [make_window(mask, context_bounds), make_window(mask, detail_bounds)]
            random_pair = [make_window(mask, random_context), make_window(mask, random_detail)]
            physician_windows.extend(manual_pair)
            random_windows.extend(random_pair)
            for scale_name, manual_window, random_window in zip(
                ("context", "detail"), manual_pair, random_pair
            ):
                rows.append(
                    {
                        "feature_row": int(index),
                        "oracle_id": str(record["oracle_id"]),
                        "case_id": str(record["case_id"]),
                        "reader_id": reader_id,
                        "reader_has_roi": reader_has_roi,
                        "scale_name": scale_name,
                        "manual_x1": manual_window.x1,
                        "manual_y1": manual_window.y1,
                        "manual_x2": manual_window.x2,
                        "manual_y2": manual_window.y2,
                        "manual_tissue_coverage": manual_window.tissue_coverage,
                        "random_x1": random_window.x1,
                        "random_y1": random_window.y1,
                        "random_x2": random_window.x2,
                        "random_y2": random_window.y2,
                        "random_tissue_coverage": random_window.tissue_coverage,
                    }
                )
        plans.append(
            CaseOraclePlan(
                tuple(physician_windows), tuple(random_windows), tuple(has_roi)
            )
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "g1_physician_roi_plan.csv", index=False, encoding="utf-8-sig"
    )
    return plans


class OracleRoiDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        plans: list[CaseOraclePlan],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.plans = plans
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        with Image.open(str(self.records.iloc[index]["image_path"])) as source:
            image = source.convert("RGB")
        plan = self.plans[index]
        manual = [self.transform(a1.crop_relative(image, window)) for window in plan.physician]
        random = [self.transform(a1.crop_relative(image, window)) for window in plan.random]
        return index, torch.stack(manual + random, dim=0)


def extract_oracle_features(
    args: argparse.Namespace,
    records: pd.DataFrame,
    plans: list[CaseOraclePlan],
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    model = d1.create_backbone(args.model_name, "dense").eval().to(device)
    image_size = d1.model_image_size(model, args.image_size)
    mean, std = d1.model_normalization(model)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
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
    roi_views = len(plans[0].physician)
    manual_features = np.empty(
        (len(records), roi_views, token_count, feature_dim), dtype=np.float16
    )
    random_features = np.empty_like(manual_features)
    loader_kwargs: dict[str, Any] = {
        "dataset": OracleRoiDataset(records, plans, transform),
        "batch_size": args.extract_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    for row_indices, view_batch in tqdm(loader, desc="G1 manual/random ROI tokens"):
        batch_size, num_views, channels, height, width = view_batch.shape
        inputs = view_batch.reshape(batch_size * num_views, channels, height, width).to(
            device, non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            tokens = d1.extract_feature_tokens(model, inputs, "dense")
        tokens = tokens.reshape(batch_size, num_views, token_count, feature_dim)
        indices = row_indices.numpy().astype(int)
        array = tokens.detach().cpu().numpy().astype(np.float16, copy=False)
        manual_features[indices] = array[:, :roi_views]
        random_features[indices] = array[:, roi_views:]
    del model, dummy_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manual_features, random_features


def main() -> None:
    args = parse_args()
    a1.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, annotations = load_oracle_records(args)
    records.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    d1.write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "records": len(records),
            "variants": VARIANTS,
            "physician_views": "reader1/reader2 ROI1 at exact and 1.5x context",
            "missing_roi_fallback": "whole image duplicated at both ROI scales",
            "primary_conditional_subset": "both readers provide ROI1",
            "prohibited_inputs": ["C1 probability", "confidence", "source", "difficulty", "error"],
            "disk_feature_bank_written": False,
        },
    )
    base_features, _, _ = d1.extract_base_features(args, records, output_dir)
    plans = build_oracle_plans(args, records, annotations, output_dir)
    manual_features, random_features = extract_oracle_features(args, records, plans)
    if args.extract_only:
        print("[done] G1 extraction smoke completed", flush=True)
        return

    summaries = []
    for split_mode in [item.strip() for item in args.split_modes.split(",") if item.strip()]:
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
                continue
            if fold_root.exists():
                shutil.rmtree(fold_root)
            train_indices, val_indices, test_indices, held_source = a1.fold_indices(
                records, split_mode, fold_position, fold_value
            )
            if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
                raise ValueError(f"Empty G1 partition for {split_mode} {fold_value}")
            variant_features = {
                "base_only": None,
                "physician_roi": manual_features,
                "matched_random_roi": random_features,
            }
            seed = d1.stable_seed(args.seed, split_mode, fold_value, "g1_diagnostic")
            for variant, roi_features in variant_features.items():
                model, fold_summary, predictions = d1.train_head(
                    args,
                    base_features,
                    records,
                    train_indices,
                    val_indices,
                    test_indices,
                    roi_features,
                    args.epochs,
                    args.patience,
                    seed,
                    fold_root / variant,
                )
                fold_summary.update(
                    {
                        "variant": variant,
                        "split_mode": split_mode,
                        "fold_value": str(fold_value),
                        "held_source": held_source,
                    }
                )
                d1.write_json(fold_root / variant / "fold_summary.json", fold_summary)
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
            variant_dir = output_dir / split_mode / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            overall = d1.summarize_variant(
                records, pd.concat(frames, ignore_index=True), variant_dir
            )
            summaries.append({"split_mode": split_mode, "variant": variant, **overall})
    pd.DataFrame(summaries).to_csv(
        output_dir / "G1_PHYSICIAN_ROI_SUMMARY.csv", index=False, encoding="utf-8-sig"
    )
    d1.write_json(output_dir / "G1_PHYSICIAN_ROI_COMPLETE.json", {"complete": True})


if __name__ == "__main__":
    main()
