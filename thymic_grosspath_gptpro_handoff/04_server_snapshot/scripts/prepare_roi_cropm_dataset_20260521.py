from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from thymic_baseline.cropping import detect_specimen_bbox, expand_bbox


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create expanded ROI crop image mirror.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--margin", type=float, default=0.40)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def crop_one(src: Path, dst: Path, margin: float, quality: int, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        rgb = np.array(im.convert("RGB"))
    bbox = detect_specimen_bbox(rgb)
    bbox = expand_bbox(bbox, rgb.shape[:2], margin_ratio=margin)
    x1, y1, x2, y2 = bbox
    crop = Image.fromarray(rgb[y1:y2, x1:x2, :])
    crop.save(dst, quality=quality)
    return True


def main() -> None:
    args = parse_args()
    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    paths = sorted(p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    made = 0
    for idx, src in enumerate(paths, 1):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        if crop_one(src, dst, args.margin, args.quality, args.overwrite):
            made += 1
        if idx % 100 == 0:
            print(f"processed {idx}/{len(paths)} made={made}", flush=True)
    print(f"done input={in_root} output={out_root} total={len(paths)} made={made}", flush=True)


if __name__ == "__main__":
    main()
