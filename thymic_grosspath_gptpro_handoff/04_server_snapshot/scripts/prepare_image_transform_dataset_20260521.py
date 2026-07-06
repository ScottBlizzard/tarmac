from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror an image dataset with a deterministic transform.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--op", choices=("flip_lr", "flip_tb", "rot180"), default="flip_lr")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def transform(im: Image.Image, op: str) -> Image.Image:
    if op == "flip_lr":
        return ImageOps.mirror(im)
    if op == "flip_tb":
        return ImageOps.flip(im)
    if op == "rot180":
        return im.rotate(180, expand=True)
    raise ValueError(op)


def main() -> None:
    args = parse_args()
    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    paths = sorted(p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    made = 0
    for idx, src in enumerate(paths, 1):
        dst = out_root / src.relative_to(in_root)
        if dst.exists() and not args.overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            out = transform(im.convert("RGB"), args.op)
            out.save(dst, quality=args.quality)
        made += 1
        if idx % 100 == 0:
            print(f"[{args.op}] processed {idx}/{len(paths)} made={made}", flush=True)
    print(f"[done] op={args.op} input={in_root} output={out_root} total={len(paths)} made={made}", flush=True)


if __name__ == "__main__":
    main()
