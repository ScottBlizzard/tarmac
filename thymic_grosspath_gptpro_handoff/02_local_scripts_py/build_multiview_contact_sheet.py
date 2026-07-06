from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    copy = img.convert("RGB")
    copy.thumbnail((box_w, box_h))
    canvas = Image.new("RGB", (box_w, box_h), "white")
    x = (box_w - copy.width) // 2
    y = (box_h - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cell-width", type=int, default=520)
    parser.add_argument("--cell-height", type=int, default=340)
    args = parser.parse_args()

    pairs = []
    for line in Path(args.pairs_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"Expected 4 tab-separated fields, got: {line}")
        pairs.append(parts)

    margin = 20
    label_h = 48
    rows = len(pairs)
    width = margin * 3 + args.cell_width * 2
    height = margin * (rows + 1) + rows * (args.cell_height + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (case_id, img1_path, img2_path, note) in enumerate(pairs):
        y0 = margin + idx * (args.cell_height + label_h + margin)
        draw.text((margin, y0), f"{case_id}  {note}", fill="black")
        img1 = fit_image(Image.open(img1_path), args.cell_width, args.cell_height)
        img2 = fit_image(Image.open(img2_path), args.cell_width, args.cell_height)
        canvas.paste(img1, (margin, y0 + label_h))
        canvas.paste(img2, (margin * 2 + args.cell_width, y0 + label_h))
        draw.text((margin, y0 + label_h + args.cell_height + 4), "图1", fill="black")
        draw.text((margin * 2 + args.cell_width, y0 + label_h + args.cell_height + 4), "图2", fill="black")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
