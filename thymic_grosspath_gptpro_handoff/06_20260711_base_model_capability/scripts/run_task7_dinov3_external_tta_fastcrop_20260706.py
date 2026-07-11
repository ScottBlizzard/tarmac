#!/usr/bin/env python3
"""Locked external TTA evaluation with cached fast specimen crops.

The original external TTA script calls extract_specimen_crop on the full
resolution PIL image for every fold and every TTA view. That is prohibitively
slow for whole_plus_crop models. This script keeps the model and evaluation
logic unchanged, but detects the crop bbox on a downsampled image once per case
and reuses a cached crop for all folds/views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_oof_tta_20260524 import (  # noqa: E402
    apply_view,
    load_model,
    move_to_device,
    normalize_transform,
    read_json,
)
from thymic_baseline.cropping import detect_specimen_bbox, expand_bbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast locked external TTA evaluation for Task7 DINOv3 fold models.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--external-registry-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", default="orig,hflip,vflip,hvflip")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detect-max-dim", type=int, default=384)
    parser.add_argument("--crop-cache-dir", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def cache_key(case_id: str, image_path: str) -> str:
    digest = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:12]
    safe_case = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)[:80]
    return f"{safe_case}__{digest}.jpg"


def fast_specimen_crop_from_path(path: str, margin_ratio: float = 0.12, detect_max_dim: int = 384) -> Image.Image:
    with Image.open(path) as low:
        orig_w, orig_h = low.size
        low.draft("RGB", (detect_max_dim, detect_max_dim))
        low.thumbnail((detect_max_dim, detect_max_dim), Image.Resampling.BILINEAR)
        small = low.convert("RGB")
        small_arr = np.asarray(small)
        small_w, small_h = small.size
    bbox = detect_specimen_bbox(small_arr)
    bbox = expand_bbox(bbox, small_arr.shape[:2], margin_ratio=margin_ratio)
    sx = orig_w / max(small_w, 1)
    sy = orig_h / max(small_h, 1)
    x1, y1, x2, y2 = bbox
    full_box = (
        max(0, int(round(x1 * sx))),
        max(0, int(round(y1 * sy))),
        min(orig_w, int(round(x2 * sx))),
        min(orig_h, int(round(y2 * sy))),
    )
    with Image.open(path) as full:
        return full.convert("RGB").crop(full_box)


def prepare_crop_cache(
    frame: pd.DataFrame,
    cache_dir: Path,
    detect_max_dim: int,
    jpeg_quality: int,
) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    iterator = frame[["case_id", "image_path"]].drop_duplicates().itertuples(index=False)
    for case_id, image_path in tqdm(list(iterator), desc="prepare crop cache", dynamic_ncols=True):
        key = cache_key(str(case_id), str(image_path))
        out = cache_dir / key
        mapping[str(image_path)] = str(out)
        if out.exists() and out.stat().st_size > 0:
            continue
        tmp = out.with_suffix(".tmp.jpg")
        crop = fast_specimen_crop_from_path(str(image_path), detect_max_dim=detect_max_dim)
        crop.save(tmp, format="JPEG", quality=jpeg_quality, subsampling=0)
        tmp.replace(out)
    return mapping


class FastTTADataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        input_variant: str,
        image_size: int,
        view: str,
        crop_cache: dict[str, str] | None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.input_variant = input_variant
        self.view = view
        self.crop_cache = crop_cache or {}
        self.transform = normalize_transform(image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def _load_whole(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _load_crop(self, path: str) -> Image.Image:
        cached = self.crop_cache.get(path)
        if cached:
            with Image.open(cached) as image:
                return image.convert("RGB")
        return fast_specimen_crop_from_path(path)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        whole = apply_view(self._load_whole(path), self.view)
        if self.input_variant == "whole":
            inputs = self.transform(whole)
        elif self.input_variant == "crop":
            inputs = self.transform(apply_view(self._load_crop(path), self.view))
        elif self.input_variant == "whole_plus_crop":
            inputs = (
                self.transform(whole),
                self.transform(apply_view(self._load_crop(path), self.view)),
            )
        else:
            raise ValueError(f"Unsupported input variant: {self.input_variant}")
        image_name = str(row["image_name"]) if "image_name" in row else Path(path).name
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return inputs, label, str(row["case_id"]), image_name


def predict_view_fast(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    input_variant: str,
    image_size: int,
    view: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    crop_cache: dict[str, str] | None,
) -> tuple[list[str], list[str], list[int], np.ndarray]:
    dataset = FastTTADataset(frame, input_variant, image_size, view, crop_cache)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    case_ids: list[str] = []
    image_names: list[str] = []
    labels: list[int] = []
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, batch_labels, batch_case_ids, batch_image_names in tqdm(loader, desc=f"tta {view}", leave=False, dynamic_ncols=True):
            logits = model(move_to_device(inputs, device))
            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(batch_probs)
            labels.extend(int(x) for x in batch_labels.cpu().numpy().tolist())
            case_ids.extend(str(x) for x in batch_case_ids)
            image_names.extend(str(x) for x in batch_image_names)
    return case_ids, image_names, labels, np.concatenate(probs, axis=0)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = read_json(run_dir / "args.json")
    external = pd.read_csv(args.external_registry_csv, dtype={"case_id": str}).reset_index(drop=True)
    if "image_name" not in external.columns:
        external["image_name"] = external["image_path"].map(lambda p: Path(str(p)).name)
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    device = resolve_device(args.device)

    input_variant = str(run_args["input_variant"])
    crop_cache = None
    if input_variant in {"crop", "whole_plus_crop"}:
        cache_dir = Path(args.crop_cache_dir) if args.crop_cache_dir else out_dir / "fast_crop_cache"
        crop_cache = prepare_crop_cache(
            external,
            cache_dir=cache_dir,
            detect_max_dim=int(args.detect_max_dim),
            jpeg_quality=int(args.jpeg_quality),
        )

    fold_probs: list[np.ndarray] = []
    base_case_ids: list[str] | None = None
    base_image_names: list[str] | None = None
    base_labels: list[int] | None = None
    checkpoints = sorted(run_dir.glob("fold_*/best_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found under {run_dir}")
    for checkpoint in checkpoints:
        fold_id = checkpoint.parent.name
        print(f"[{fold_id}] loading model", flush=True)
        model = load_model(run_args, checkpoint, device)
        view_probs: list[np.ndarray] = []
        for view in views:
            case_ids, image_names, labels, probs = predict_view_fast(
                model=model,
                frame=external,
                input_variant=input_variant,
                image_size=int(run_args["image_size"]),
                view=view,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                crop_cache=crop_cache,
            )
            if base_case_ids is None:
                base_case_ids = case_ids
                base_image_names = image_names
                base_labels = labels
            elif base_case_ids != case_ids:
                raise RuntimeError(f"Case order mismatch for {fold_id} view {view}")
            view_probs.append(probs)
        fold_probs.append(np.mean(np.stack(view_probs, axis=0), axis=0))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert base_case_ids is not None and base_image_names is not None and base_labels is not None
    mean_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
    pred_idx = mean_probs.argmax(axis=1)
    out = external.copy()
    out["prob_low_risk_group"] = mean_probs[:, 0]
    out["prob_high_risk_group"] = mean_probs[:, 1]
    out["pred_idx"] = pred_idx.astype(int)
    out["tta_views"] = ",".join(views)
    out["n_folds"] = len(checkpoints)
    out["correct"] = (out["pred_idx"].astype(int) == out["label_idx"].astype(int)).astype(int)
    out.to_csv(out_dir / "external_tta_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "args.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "external_registry_csv": str(args.external_registry_csv),
                "views": views,
                "fast_crop": True,
                "detect_max_dim": int(args.detect_max_dim),
                "crop_cache_dir": str(Path(args.crop_cache_dir) if args.crop_cache_dir else out_dir / "fast_crop_cache"),
                "source_args": run_args,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
