from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror an image dataset after blue-board background normalization.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--fit-target", action="store_true", help="Estimate target background color from input-root.")
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--gain-min", type=float, default=0.65)
    parser.add_argument("--gain-max", type=float, default=1.55)
    parser.add_argument("--border-frac", type=float, default=0.08)
    parser.add_argument("--max-fit-images", type=int, default=1200)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_small_rgb(path: Path, max_side: int = 1024) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def border_pixels(rgb: np.ndarray, border_frac: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = max(4, int(min(h, w) * border_frac))
    return np.concatenate(
        [
            rgb[:m, :, :].reshape(-1, 3),
            rgb[-m:, :, :].reshape(-1, 3),
            rgb[:, :m, :].reshape(-1, 3),
            rgb[:, -m:, :].reshape(-1, 3),
        ],
        axis=0,
    )


def estimate_background_rgb(rgb: np.ndarray, border_frac: float) -> np.ndarray:
    pix = border_pixels(rgb, border_frac)
    gray = 0.299 * pix[:, 0] + 0.587 * pix[:, 1] + 0.114 * pix[:, 2]
    maxc = pix.max(axis=1)
    minc = pix.min(axis=1)
    sat = np.where(maxc > 1.0, (maxc - minc) / np.maximum(maxc, 1.0), 0.0)

    # Keep the board-like border majority while dropping black frames, glare, and tissue-touching borders.
    mask = (gray > 55.0) & (gray < 245.0) & (sat < 0.55)
    if int(mask.sum()) < max(100, int(0.08 * len(pix))):
        mask = (gray > 45.0) & (gray < 250.0)
    if int(mask.sum()) < max(50, int(0.03 * len(pix))):
        mask = np.ones(len(pix), dtype=bool)
    return np.median(pix[mask], axis=0).astype(np.float32)


def fit_target(paths: list[Path], border_frac: float, max_fit_images: int) -> dict[str, object]:
    if not paths:
        raise FileNotFoundError("No images found for target fitting.")
    selected = paths[:max_fit_images]
    bgs = []
    for idx, path in enumerate(selected, 1):
        try:
            bgs.append(estimate_background_rgb(load_small_rgb(path), border_frac))
        except Exception as exc:
            print(f"[warn] fit skipped {path}: {exc}", flush=True)
        if idx % 200 == 0:
            print(f"[fit] {idx}/{len(selected)}", flush=True)
    if not bgs:
        raise RuntimeError("Failed to estimate any background colors.")
    target_rgb = np.median(np.stack(bgs, axis=0), axis=0)
    return {
        "target_rgb": [float(x) for x in target_rgb],
        "fit_images": int(len(bgs)),
        "border_frac": float(border_frac),
    }


def normalize_one(
    src: Path,
    dst: Path,
    target_rgb: np.ndarray,
    strength: float,
    gain_min: float,
    gain_max: float,
    border_frac: float,
    quality: int,
    overwrite: bool,
) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    bg = estimate_background_rgb(rgb, border_frac)
    gain = np.clip(target_rgb / np.maximum(bg, 1.0), gain_min, gain_max)
    corrected = np.clip(rgb * gain.reshape(1, 1, 3), 0.0, 255.0)
    out = np.clip((1.0 - strength) * rgb + strength * corrected, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(out).save(dst, quality=quality)
    return True


def main() -> None:
    args = parse_args()
    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    target_path = Path(args.target_json)
    paths = image_paths(in_root)
    if args.fit_target or not target_path.exists():
        target = fit_target(paths, args.border_frac, args.max_fit_images)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(target, indent=2), encoding="utf-8")
        print(f"[target] {target_path} rgb={target['target_rgb']}", flush=True)
    else:
        target = json.loads(target_path.read_text(encoding="utf-8"))
        print(f"[target] loaded {target_path} rgb={target['target_rgb']}", flush=True)

    target_rgb = np.asarray(target["target_rgb"], dtype=np.float32)
    made = 0
    for idx, src in enumerate(paths, 1):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        if normalize_one(
            src,
            dst,
            target_rgb,
            args.strength,
            args.gain_min,
            args.gain_max,
            args.border_frac,
            args.quality,
            args.overwrite,
        ):
            made += 1
        if idx % 100 == 0:
            print(f"[norm] processed {idx}/{len(paths)} made={made}", flush=True)
    print(f"[done] input={in_root} output={out_root} total={len(paths)} made={made}", flush=True)


if __name__ == "__main__":
    main()
