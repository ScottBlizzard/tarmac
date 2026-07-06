from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def resolve_image_path(images_root: Path, source_case_folder: str, image_name: str, case_id: int) -> Path:
    folder = images_root / source_case_folder
    direct = folder / image_name
    if direct.exists():
        return direct

    nested = folder / str(case_id) / image_name
    if nested.exists():
        return nested

    stem = Path(image_name).stem.lower()
    suffix = Path(image_name).suffix.lower()
    candidates = list(folder.rglob("*"))
    files = [p for p in candidates if p.is_file()]

    exact_stem = [p for p in files if p.stem.lower() == stem]
    if exact_stem:
        return sorted(exact_stem)[0]

    exact_name = [p for p in files if p.name.lower() == image_name.lower()]
    if exact_name:
        return sorted(exact_name)[0]

    fuzzy = [
        p
        for p in files
        if stem in p.stem.lower()
        and (not suffix or p.suffix.lower() == suffix)
    ]
    if fuzzy:
        return sorted(fuzzy)[0]

    raise FileNotFoundError(f"Could not resolve image path for case_id={case_id}, folder={source_case_folder}, image_name={image_name}")


def tissue_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s > 25) | (v < 235)).astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def border_touch_level(mask: np.ndarray) -> tuple[str, int]:
    h, w = mask.shape
    bands = {
        "top": mask[:10, :],
        "bottom": mask[h - 10 :, :],
        "left": mask[:, :10],
        "right": mask[:, w - 10 :],
    }
    touched = 0
    for band in bands.values():
        if (band > 0).mean() > 0.08:
            touched += 1
    if touched >= 3:
        return "high", touched
    if touched >= 1:
        return "medium", touched
    return "low", touched


def quality_bucket(score: float) -> str:
    if score >= 75:
        return "good"
    if score >= 55:
        return "fair"
    return "poor"


def image_metrics(path: Path) -> dict[str, float | str]:
    img = load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = tissue_mask(img)
    non_mask = mask == 0

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    glare_ratio = float(((hsv[:, :, 2] > 245) & (hsv[:, :, 1] < 35)).mean())
    dark_ratio = float((gray < 45).mean())
    bright_ratio = float((gray > 220).mean())
    area_ratio = float((mask > 0).mean())
    touch_label, touch_count = border_touch_level(mask)

    edges = cv2.Canny(gray, 80, 160)
    bg_edge_density = float((edges[non_mask] > 0).mean()) if non_mask.any() else 0.0

    if blur_var < 20:
        clarity = "poor"
    elif blur_var < 45:
        clarity = "fair"
    else:
        clarity = "good"

    if brightness < 105:
        exposure = "dark"
    elif brightness > 175:
        exposure = "bright"
    else:
        exposure = "normal"

    if glare_ratio >= 0.12:
        glare = "obvious"
    elif glare_ratio >= 0.04:
        glare = "mild"
    else:
        glare = "none"

    if bg_edge_density >= 0.08:
        background = "high"
    elif bg_edge_density >= 0.03:
        background = "medium"
    else:
        background = "low"

    if area_ratio < 0.28:
        specimen_ratio = "small"
    elif area_ratio < 0.58:
        specimen_ratio = "medium"
    else:
        specimen_ratio = "large"

    score_map = {
        "clarity": {"good": 2, "fair": 1, "poor": 0}[clarity],
        "exposure": {"normal": 2, "dark": 1, "bright": 1}[exposure],
        "glare": {"none": 2, "mild": 1, "obvious": 0}[glare],
        "background": {"low": 2, "medium": 1, "high": 0}[background],
    }
    quality_score = round(sum(score_map.values()) / 8 * 100, 1)
    overall_quality = quality_bucket(quality_score)

    return {
        "blur_var": round(blur_var, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "glare_ratio": round(glare_ratio, 4),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "tissue_area_ratio": round(area_ratio, 4),
        "bg_edge_density": round(bg_edge_density, 4),
        "border_touch_count": touch_count,
        "border_touch_level": touch_label,
        "image_clarity": clarity,
        "exposure": exposure,
        "glare_or_reflection": glare,
        "background_clutter": background,
        "specimen_ratio": specimen_ratio,
        "quality_score": quality_score,
        "overall_quality": overall_quality,
    }


def aggregate_case(rows: list[dict[str, object]]) -> dict[str, object]:
    blur_vals = np.array([float(r["blur_var"]) for r in rows])
    bright_vals = np.array([float(r["brightness"]) for r in rows])
    glare_vals = np.array([float(r["glare_ratio"]) for r in rows])
    area_vals = np.array([float(r["tissue_area_ratio"]) for r in rows])
    bg_vals = np.array([float(r["bg_edge_density"]) for r in rows])
    touch_vals = np.array([int(r["border_touch_count"]) for r in rows])

    if len(rows) == 1:
        multiview_consistency = "single"
    else:
        blur_cv = float(np.std(blur_vals) / (np.mean(blur_vals) + 1e-6))
        bright_std = float(np.std(bright_vals))
        area_std = float(np.std(area_vals))
        if blur_cv > 0.55 or bright_std > 22 or area_std > 0.12:
            multiview_consistency = "inconsistent"
        elif blur_cv > 0.28 or bright_std > 12 or area_std > 0.06:
            multiview_consistency = "slightly_inconsistent"
        else:
            multiview_consistency = "consistent"

    case_quality_score = round(np.mean([float(r["quality_score"]) for r in rows]), 1)
    case_quality = quality_bucket(case_quality_score)

    return {
        "case_image_count": len(rows),
        "case_blur_var_mean": round(float(blur_vals.mean()), 2),
        "case_brightness_mean": round(float(bright_vals.mean()), 2),
        "case_glare_ratio_max": round(float(glare_vals.max()), 4),
        "case_tissue_area_ratio_mean": round(float(area_vals.mean()), 4),
        "case_bg_edge_density_mean": round(float(bg_vals.mean()), 4),
        "case_border_touch_level": "high" if int(touch_vals.max()) >= 3 else "medium" if int(touch_vals.max()) >= 1 else "low",
        "case_multiview_consistency": multiview_consistency,
        "case_quality_score": case_quality_score,
        "case_overall_quality": case_quality,
    }


def expand_registry(registry_csv: Path, images_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    df = pd.read_csv(registry_csv)
    image_rows: list[dict[str, object]] = []
    case_groups: dict[int, list[dict[str, object]]] = defaultdict(list)

    for _, row in df.iterrows():
        case_id = int(row["case_id"])
        image_names = [x.strip() for x in str(row["image_filenames"]).split(";") if x.strip()]
        folder = str(row["source_case_folder"])
        for image_name in image_names:
            image_path = resolve_image_path(images_root, folder, image_name, case_id)
            metrics = image_metrics(image_path)
            record = {
                "case_id": case_id,
                "source_case_folder": folder,
                "who_type_raw": row["who_type_raw"],
                "task_l3_label": row.get("task_l3_label", ""),
                "task_l4_label": row.get("task_l4_label", ""),
                "rare_subtype_role": row.get("rare_subtype_role", ""),
                "image_name": image_name,
                "image_relpath": str(image_path.relative_to(images_root)),
                **metrics,
            }
            image_rows.append(record)
            case_groups[case_id].append(record)

    row_map = {int(r["case_id"]): r for _, r in df.iterrows()}
    case_rows: list[dict[str, object]] = []
    for case_id, rows in case_groups.items():
        meta = row_map[case_id]
        agg = aggregate_case(rows)
        case_rows.append(
            {
                "case_id": case_id,
                "source_case_folder": meta["source_case_folder"],
                "who_type_raw": meta["who_type_raw"],
                "task_l3_label": meta.get("task_l3_label", ""),
                "task_l4_label": meta.get("task_l4_label", ""),
                "image_filenames": meta["image_filenames"],
                **agg,
            }
        )
    return image_rows, case_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate objective image-quality labels for the thymic gross-image dataset.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-image-csv", required=True)
    parser.add_argument("--output-case-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_rows, case_rows = expand_registry(Path(args.registry_csv), Path(args.images_root))
    write_csv(Path(args.output_image_csv), image_rows)
    write_csv(Path(args.output_case_csv), case_rows)
    print(args.output_image_csv)
    print(args.output_case_csv)
    print(f"images={len(image_rows)} cases={len(case_rows)}")


if __name__ == "__main__":
    main()
