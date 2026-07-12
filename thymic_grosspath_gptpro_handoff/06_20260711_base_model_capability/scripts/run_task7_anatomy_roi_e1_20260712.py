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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_task7_attention_roi_d1_20260712 as d1  # noqa: E402

a1 = d1.a1


VARIANTS = ("base_only", "anatomy_roi", "matched_random_roi")
ROI_ROLES = ("interior_heterogeneity", "capsule_interface")


@dataclass(frozen=True)
class ScoredCenter:
    role: str
    center_x: float
    center_y: float
    tissue_coverage: float
    anatomy_score: float
    fallback: bool


@dataclass(frozen=True)
class CaseRoiPlan:
    anatomy: tuple[a1.TileWindow, ...]
    random: tuple[a1.TileWindow, ...]
    anatomy_centers: tuple[ScoredCenter, ...]
    random_centers: tuple[ScoredCenter, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label-free anatomy ROI direct-model experiment with matched random controls."
    )
    parser.add_argument("--registry-csv", default=str(a1.DEFAULT_REGISTRY))
    parser.add_argument("--split-csv", default=str(a1.DEFAULT_SPLIT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", default="old_data,third_batch")
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


def window_slices(
    window: a1.TileWindow, shape: tuple[int, int]
) -> tuple[slice, slice]:
    height, width = shape
    left = min(width - 1, max(0, int(math.floor(window.x1 * width))))
    top = min(height - 1, max(0, int(math.floor(window.y1 * height))))
    right = min(width, max(left + 1, int(math.ceil(window.x2 * width))))
    bottom = min(height, max(top + 1, int(math.ceil(window.y2 * height))))
    return slice(top, bottom), slice(left, right)


def boundary_band(mask: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    radius = max(1, int(round(min(mask.shape) * 0.015)))
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - tensor, kernel_size=kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)[0, 0].numpy()


def robust_range(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    q10, q90 = np.quantile(values, [0.10, 0.90])
    return float(q90 - q10)


def score_candidates(
    specimen: Image.Image,
    mask: np.ndarray,
    detail_scale: float,
    grid_size: int,
) -> list[dict[str, Any]]:
    height, width = mask.shape
    analysis_rgb = np.asarray(
        specimen.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    gray = (
        0.299 * analysis_rgb[..., 0]
        + 0.587 * analysis_rgb[..., 1]
        + 0.114 * analysis_rgb[..., 2]
    )
    saturation = analysis_rgb.max(axis=2) - analysis_rgb.min(axis=2)
    grad_y, grad_x = np.gradient(gray)
    gradient = np.hypot(grad_x, grad_y)
    boundary = boundary_band(mask)
    rows: list[dict[str, Any]] = []
    for window in a1.candidate_windows(mask, detail_scale, grid_size):
        ys, xs = window_slices(window, mask.shape)
        local_tissue = mask[ys, xs]
        if local_tissue.any():
            gray_values = gray[ys, xs][local_tissue]
            saturation_values = saturation[ys, xs][local_tissue]
            gradient_values = gradient[ys, xs][local_tissue]
            heterogeneity = (
                robust_range(gray_values)
                + 0.50 * robust_range(saturation_values)
                + 0.50 * float(gradient_values.mean())
            )
        else:
            heterogeneity = 0.0
        boundary_density = float(boundary[ys, xs].mean())
        interface_gradient = float(gradient[ys, xs].mean())
        rows.append(
            {
                "window": window,
                "heterogeneity_score": float(heterogeneity),
                "interface_score": float(2.0 * boundary_density + interface_gradient),
                "boundary_density": boundary_density,
            }
        )
    return rows


def choose_anatomy_centers(
    candidates: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[ScoredCenter, ScoredCenter]:
    interior_pool = [
        item
        for item in candidates
        if item["window"].tissue_coverage >= args.minimum_interior_tissue
    ]
    interior_fallback = not interior_pool
    if not interior_pool:
        interior_pool = candidates
    interior_item = max(
        interior_pool,
        key=lambda item: (
            item["heterogeneity_score"],
            item["window"].tissue_coverage,
            -abs(item["window"].center_x - 0.5) - abs(item["window"].center_y - 0.5),
        ),
    )
    interior_window = interior_item["window"]
    interior = ScoredCenter(
        ROI_ROLES[0],
        interior_window.center_x,
        interior_window.center_y,
        interior_window.tissue_coverage,
        interior_item["heterogeneity_score"],
        interior_fallback,
    )

    interface_pool = [
        item
        for item in candidates
        if args.minimum_interface_tissue
        <= item["window"].tissue_coverage
        <= args.maximum_interface_tissue
        and math.dist(
            (item["window"].center_x, item["window"].center_y),
            (interior.center_x, interior.center_y),
        )
        >= args.minimum_center_distance
        and item["boundary_density"] > 0.0
    ]
    interface_fallback = not interface_pool
    if not interface_pool:
        interface_pool = [
            item
            for item in candidates
            if math.dist(
                (item["window"].center_x, item["window"].center_y),
                (interior.center_x, interior.center_y),
            )
            >= args.minimum_center_distance
        ]
    if not interface_pool:
        interface_pool = candidates
    interface_item = max(
        interface_pool,
        key=lambda item: (
            item["interface_score"],
            -abs(item["window"].tissue_coverage - 0.60),
            -item["window"].center_y,
            -item["window"].center_x,
        ),
    )
    interface_window = interface_item["window"]
    interface = ScoredCenter(
        ROI_ROLES[1],
        interface_window.center_x,
        interface_window.center_y,
        interface_window.tissue_coverage,
        interface_item["interface_score"],
        interface_fallback,
    )
    return interior, interface


def choose_matched_random_centers(
    mask: np.ndarray,
    anatomy_centers: tuple[ScoredCenter, ScoredCenter],
    args: argparse.Namespace,
    seed: int,
) -> tuple[ScoredCenter, ScoredCenter]:
    rng = np.random.default_rng(seed)
    half = args.detail_scale / 2.0
    random_windows = [
        d1.centered_window(
            mask,
            float(rng.uniform(half, 1.0 - half)),
            float(rng.uniform(half, 1.0 - half)),
            args.detail_scale,
        )
        for _ in range(args.random_candidates)
    ]
    selected: list[ScoredCenter] = []
    selected_by_role: dict[str, ScoredCenter] = {}
    # The interface coverage is geometrically harder to match, so reserve it first.
    for target in (anatomy_centers[1], anatomy_centers[0]):
        role_pool: list[a1.TileWindow] = []
        for window in random_windows:
            if target.role == ROI_ROLES[0] and window.tissue_coverage < args.minimum_interior_tissue:
                continue
            if target.role == ROI_ROLES[1] and not (
                args.minimum_interface_tissue
                <= window.tissue_coverage
                <= args.maximum_interface_tissue
            ):
                continue
            if any(
                math.dist((window.center_x, window.center_y), (center.center_x, center.center_y))
                < args.minimum_center_distance
                for center in anatomy_centers
            ):
                continue
            if any(
                math.dist((window.center_x, window.center_y), (center.center_x, center.center_y))
                < args.minimum_center_distance
                for center in selected
            ):
                continue
            role_pool.append(window)
        fallback = not role_pool
        if not role_pool:
            role_pool = [
                window
                for window in random_windows
                if all(
                    math.dist(
                        (window.center_x, window.center_y),
                        (center.center_x, center.center_y),
                    )
                    >= args.minimum_center_distance
                    for center in anatomy_centers
                )
                and all(
                    math.dist(
                        (window.center_x, window.center_y),
                        (center.center_x, center.center_y),
                    )
                    >= args.minimum_center_distance
                    for center in selected
                )
            ]
        if not role_pool:
            relaxed_distance = 0.75 * args.minimum_center_distance
            role_pool = [
                window
                for window in random_windows
                if all(
                    math.dist(
                        (window.center_x, window.center_y),
                        (center.center_x, center.center_y),
                    )
                    >= relaxed_distance
                    for center in anatomy_centers
                )
                and all(
                    math.dist(
                        (window.center_x, window.center_y),
                        (center.center_x, center.center_y),
                    )
                    >= relaxed_distance
                    for center in selected
                )
            ]
        if not role_pool:
            role_pool = random_windows
        role_pool.sort(
            key=lambda window: (
                abs(window.tissue_coverage - target.tissue_coverage),
                window.center_y,
                window.center_x,
            )
        )
        best_gap = abs(role_pool[0].tissue_coverage - target.tissue_coverage)
        coverage_matched = [
            window
            for window in role_pool
            if abs(window.tissue_coverage - target.tissue_coverage)
            <= best_gap + args.random_coverage_slack
        ]
        short = coverage_matched[:
            max(1, min(args.random_match_pool, len(coverage_matched)))
        ]
        window = short[int(rng.integers(0, len(short)))]
        chosen = ScoredCenter(
            target.role,
            window.center_x,
            window.center_y,
            window.tissue_coverage,
            float("nan"),
            fallback,
        )
        selected.append(chosen)
        selected_by_role[target.role] = chosen
    return selected_by_role[ROI_ROLES[0]], selected_by_role[ROI_ROLES[1]]


def make_case_plan(
    image: Image.Image, args: argparse.Namespace, seed: int
) -> tuple[CaseRoiPlan, Image.Image, np.ndarray]:
    specimen, mask = d1.specimen_and_mask(image)
    candidates = score_candidates(specimen, mask, args.detail_scale, args.grid_size)
    anatomy_centers = choose_anatomy_centers(candidates, args)
    random_centers = choose_matched_random_centers(mask, anatomy_centers, args, seed)
    scales = [float(item) for item in args.roi_scales.split(",") if item.strip()]
    anatomy_windows = tuple(
        d1.centered_window(mask, center.center_x, center.center_y, scale)
        for center in anatomy_centers
        for scale in scales
    )
    random_windows = tuple(
        d1.centered_window(mask, center.center_x, center.center_y, scale)
        for center in random_centers
        for scale in scales
    )
    return (
        CaseRoiPlan(anatomy_windows, random_windows, anatomy_centers, random_centers),
        specimen,
        mask,
    )


def build_roi_plans(
    args: argparse.Namespace, records: pd.DataFrame, output_dir: Path
) -> list[CaseRoiPlan]:
    plans: list[CaseRoiPlan] = []
    rows: list[dict[str, Any]] = []
    scales = [float(item) for item in args.roi_scales.split(",") if item.strip()]
    progress = tqdm(records.iterrows(), total=len(records), desc="E1 label-free ROI plans")
    for index, row in progress:
        with Image.open(str(row["image_path"])) as source:
            image = source.convert("RGB")
        plan, _, _ = make_case_plan(
            image,
            args,
            d1.stable_seed(args.seed, row["case_id"], "matched_random_roi"),
        )
        plans.append(plan)
        for route, centers, windows in (
            ("anatomy", plan.anatomy_centers, plan.anatomy),
            ("random", plan.random_centers, plan.random),
        ):
            for center_index, center in enumerate(centers):
                for scale_index, scale in enumerate(scales):
                    window = windows[center_index * len(scales) + scale_index]
                    rows.append(
                        {
                            "feature_row": int(index),
                            "case_id": str(row["case_id"]),
                            "route": route,
                            "role": center.role,
                            "roi_index": center_index * len(scales) + scale_index,
                            "center_x": center.center_x,
                            "center_y": center.center_y,
                            "scale": scale,
                            "tissue_coverage": window.tissue_coverage,
                            "detail_tissue_coverage": center.tissue_coverage,
                            "anatomy_score": center.anatomy_score,
                            "fallback": center.fallback,
                        }
                    )
    plan_path = output_dir / "anatomy_roi_plan.csv"
    if plan_path.exists():
        plan_path.unlink()
    pd.DataFrame(rows).to_csv(plan_path, index=False, encoding="utf-8-sig")
    return plans


def neutralize_specimen(
    specimen: Image.Image, mask: np.ndarray, neutral_value: int
) -> Image.Image:
    native_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            specimen.size, Image.Resampling.NEAREST
        )
    ) > 0
    rgb = np.asarray(specimen, dtype=np.uint8).copy()
    rgb[~native_mask] = np.uint8(neutral_value)
    return Image.fromarray(rgb, mode="RGB")


class RoiViewDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        plans: list[CaseRoiPlan],
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
        specimen = neutralize_specimen(specimen, mask, self.neutral_value)
        plan = self.plans[index]
        anatomy_views = [
            self.transform(a1.crop_relative(specimen, window)) for window in plan.anatomy
        ]
        random_views = [
            self.transform(a1.crop_relative(specimen, window)) for window in plan.random
        ]
        return index, torch.stack(anatomy_views + random_views, dim=0)


def extract_roi_features(
    args: argparse.Namespace,
    records: pd.DataFrame,
    plans: list[CaseRoiPlan],
) -> tuple[np.ndarray, np.ndarray]:
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
    roi_views = len(plans[0].anatomy)
    anatomy_features = np.empty(
        (len(records), roi_views, token_count, feature_dim), dtype=np.float16
    )
    random_features = np.empty_like(anatomy_features)
    loader_kwargs: dict[str, Any] = {
        "dataset": RoiViewDataset(
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
    progress = tqdm(loader, desc="E1 anatomy/random ROI tokens", file=sys.stdout)
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
        tokens = tokens.reshape(batch_size, num_views, token_count, feature_dim)
        indices = row_indices.numpy().astype(int)
        array = tokens.detach().cpu().numpy().astype(np.float16, copy=False)
        anatomy_features[indices] = array[:, :roi_views]
        random_features[indices] = array[:, roi_views:]
        progress.set_postfix(cases=int(indices[-1]) + 1, total=len(records))
    progress.close()
    del model, dummy_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return anatomy_features, random_features


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
            "base_views": d1.BASE_VIEW_NAMES,
            "variants": VARIANTS,
            "roi_roles": ROI_ROLES,
            "selector_input": "raw RGB pixels and deterministic specimen mask only",
            "diagnostic_input": "base and optional raw-pixel ROI tokens only",
            "prohibited_inputs": [
                "label",
                "stage1_probability",
                "confidence",
                "difficulty",
                "source",
                "batch",
                "true_subtype",
                "error_label",
            ],
            "selection_policy": (
                "One fixed label-free plan for every split; interior heterogeneity and "
                "capsule/interface centers; tissue-matched random control mandatory."
            ),
            "background_policy": (
                f"Outside-specimen RGB set to fixed {args.neutral_background_value} "
                "for anatomy and random ROI views."
            ),
            "disk_feature_bank_written": False,
        },
    )
    base_features, _, _ = d1.extract_base_features(args, records, output_dir)
    plans = build_roi_plans(args, records, output_dir)
    anatomy_features, random_features = extract_roi_features(args, records, plans)
    if args.extract_only:
        print("[done] E1 base and ROI extraction smoke completed", flush=True)
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
            variant_features = {
                "base_only": None,
                "anatomy_roi": anatomy_features,
                "matched_random_roi": random_features,
            }
            diagnostic_seed = d1.stable_seed(
                args.seed, split_mode, fold_value, "diagnostic"
            )
            for variant, roi_features in variant_features.items():
                model, head_summary, predictions = d1.train_head(
                    args,
                    base_features,
                    records,
                    train_indices,
                    val_indices,
                    test_indices,
                    roi_features,
                    args.epochs,
                    args.patience,
                    diagnostic_seed,
                    fold_root / variant,
                )
                head_summary.update(
                    {
                        "variant": variant,
                        "split_mode": split_mode,
                        "fold_value": str(fold_value),
                        "held_source": held_source,
                        "selector_is_label_free": True,
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

    summary_path = output_dir / "E1_ANATOMY_ROI_SUMMARY.csv"
    if summary_path.exists():
        summary_path.unlink()
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    d1.write_json(output_dir / "E1_ANATOMY_ROI_COMPLETE.json", {"complete": True})


if __name__ == "__main__":
    main()
