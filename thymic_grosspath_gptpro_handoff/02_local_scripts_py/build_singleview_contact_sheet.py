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
    parser.add_argument("--items-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=420)
    parser.add_argument("--cell-height", type=int, default=300)
    args = parser.parse_args()

    items = []
    for line in Path(args.items_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Expected 3 tab-separated fields, got: {line}")
        items.append(parts)

    rows = (len(items) + args.cols - 1) // args.cols
    margin = 18
    label_h = 42
    width = margin * (args.cols + 1) + args.cols * args.cell_width
    height = margin * (rows + 1) + rows * (args.cell_height + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (case_id, image_path, note) in enumerate(items):
        row = idx // args.cols
        col = idx % args.cols
        x0 = margin + col * (args.cell_width + margin)
        y0 = margin + row * (args.cell_height + label_h + margin)
        draw.text((x0, y0), f"{case_id}  {note}", fill="black")
        img = fit_image(Image.open(image_path), args.cell_width, args.cell_height)
        canvas.paste(img, (x0, y0 + label_h))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
