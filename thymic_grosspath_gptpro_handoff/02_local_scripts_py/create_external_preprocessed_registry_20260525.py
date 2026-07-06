from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thymic_baseline.cropping import extract_specimen_crop  # noqa: E402


BG_COLORS = {
    "crop_pad_blue": (72, 174, 218),
    "crop_pad_gray": (214, 214, 208),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create label-agnostic preprocessed external registry images.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(BG_COLORS))
    parser.add_argument("--canvas-scale", type=float, default=1.18)
    parser.add_argument("--quality", type=int, default=95)
    return parser.parse_args()


def pad_crop(crop: Image.Image, mode: str, canvas_scale: float) -> Image.Image:
    crop = crop.convert("RGB")
    w, h = crop.size
    side = int(max(w, h) * canvas_scale)
    side = max(side, w, h, 64)
    canvas = Image.new("RGB", (side, side), BG_COLORS[mode])
    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(crop, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_csv, dtype={"case_id": str})
    rows = []
    for row in frame.to_dict(orient="records"):
        src = Path(str(row["image_path"]))
        out_name = f"{row['case_id']}_{src.stem}_{args.mode}.jpg"
        out_path = image_dir / out_name
        try:
            with Image.open(src) as image:
                crop = extract_specimen_crop(image.convert("RGB"))
                processed = pad_crop(crop, args.mode, args.canvas_scale)
                processed.save(out_path, quality=args.quality)
            row["original_image_path"] = str(src)
            row["image_path"] = str(out_path)
            row["image_name"] = out_name
            row["preprocess_mode"] = args.mode
            row["preprocess_canvas_scale"] = float(args.canvas_scale)
        except Exception as exc:  # noqa: BLE001
            row["preprocess_error"] = repr(exc)
            row["preprocess_mode"] = f"{args.mode}_failed"
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = output_dir / f"external_folder_task7_registry_{args.mode}.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[done] {out_csv} n={len(out)} failed={out.get('preprocess_error', pd.Series(dtype=str)).notna().sum()}")


if __name__ == "__main__":
    main()
