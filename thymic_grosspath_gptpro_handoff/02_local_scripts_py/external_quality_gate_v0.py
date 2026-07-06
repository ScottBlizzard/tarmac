from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


EXTERNAL_DIR_NAME = "\u80f8\u817a\u7624+\u764c"
LOW_LABEL = "low_risk_group"
HIGH_LABEL = "high_risk_group"


def find_external_dir(root: Path) -> Path:
    for p in root.iterdir():
        if p.is_dir() and p.name == EXTERNAL_DIR_NAME:
            return p
    raise FileNotFoundError(f"Cannot find external image dir: {EXTERNAL_DIR_NAME}")


def case_id_from_name(name: str) -> str:
    stem = Path(name).stem
    if "--" in stem:
        return stem.split("--")[-1].strip()
    return stem.strip()


def load_rgb(path: Path, max_side: int = 1400) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        orig_w, orig_h = im.size
        scale = min(1.0, max_side / max(orig_w, orig_h))
        if scale < 1.0:
            im = im.resize((max(1, int(orig_w * scale)), max(1, int(orig_h * scale))), Image.Resampling.LANCZOS)
        return np.asarray(im), orig_w, orig_h


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return labels == idx


def foreground_mask(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    h, w = rgb.shape[:2]
    border = max(4, int(min(h, w) * 0.04))
    border_pixels = np.concatenate(
        [
            rgb[:border].reshape(-1, 3),
            rgb[-border:].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Main foreground estimate: tissue differs from the border background and
    # usually has more color/texture than a white/blue/gray board.
    mask = ((dist > 34) & (val < 248)) | ((sat > 35) & (dist > 22) & (val < 252))
    mask &= val > 18
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
    if mask.mean() > 0.015:
        mask = largest_component(mask)

    stats = {
        "bg_r": float(bg[0]),
        "bg_g": float(bg[1]),
        "bg_b": float(bg[2]),
        "dist_p90": float(np.percentile(dist, 90)),
        "sat_mean": float(np.mean(sat)),
    }
    return mask, stats


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def image_features(path: Path) -> dict[str, object]:
    rgb, orig_w, orig_h = load_rgb(path)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    mask, bg_stats = foreground_mask(rgb)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        x0 = y0 = 0
        x1, y1 = w, h
        bbox_area_ratio = 1.0
        fg_ratio = 0.0
        touch_edges = 0
    else:
        x0, y0, x1, y1 = bbox
        bbox_area_ratio = ((x1 - x0) * (y1 - y0)) / float(w * h)
        fg_ratio = float(mask.mean())
        margin = max(3, int(min(h, w) * 0.025))
        touch_edges = int(x0 <= margin) + int(y0 <= margin) + int((w - x1) <= margin) + int((h - y1) <= margin)

    crop_gray = gray[y0:y1, x0:x1]
    crop_val = val[y0:y1, x0:x1]
    crop_sat = sat[y0:y1, x0:x1]
    if crop_gray.size == 0:
        crop_gray = gray
        crop_val = val
        crop_sat = sat

    lap_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
    tenengrad = float(np.mean(cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3) ** 2 + cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3) ** 2))
    brightness_mean = float(np.mean(crop_val))
    contrast = float(np.std(crop_gray))
    dark_ratio = float(np.mean(crop_val < 35))
    bright_ratio = float(np.mean(crop_val > 245))
    glare_ratio = float(np.mean((crop_val > 245) & (crop_sat < 38)))
    saturation_mean = float(np.mean(crop_sat))
    file_mb = path.stat().st_size / (1024 * 1024)
    megapixels = (orig_w * orig_h) / 1_000_000

    return {
        "image_name": path.name,
        "original_case_id": case_id_from_name(path.name),
        "source_folder_local": path.parent.name,
        "local_path": str(path),
        "width": orig_w,
        "height": orig_h,
        "megapixels": megapixels,
        "file_mb": file_mb,
        "fg_ratio": fg_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "touch_edges": touch_edges,
        "lap_var": lap_var,
        "tenengrad": tenengrad,
        "brightness_mean": brightness_mean,
        "contrast": contrast,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "glare_ratio": glare_ratio,
        "saturation_mean": saturation_mean,
        **bg_stats,
    }


def quality_decision(row: pd.Series) -> tuple[str, float, str]:
    reasons: list[str] = []
    penalty = 0.0
    hard_unreadable = False

    if row.megapixels < 0.15 or min(row.width, row.height) < 360:
        reasons.append("极低分辨率")
        penalty += 40
        hard_unreadable = True
    elif row.megapixels < 0.45 or min(row.width, row.height) < 650:
        reasons.append("分辨率偏低")
        penalty += 10

    if row.bbox_area_ratio < 0.055 or row.fg_ratio < 0.012:
        reasons.append("主体过小，判读信息不足")
        penalty += 42
        hard_unreadable = True
    elif row.bbox_area_ratio < 0.10 or row.fg_ratio < 0.025:
        reasons.append("主体占比过小/背景过多")
        penalty += 18
    elif row.bbox_area_ratio < 0.16 or row.fg_ratio < 0.045:
        reasons.append("主体偏小")
        penalty += 8

    if row.lap_var < 22:
        reasons.append("严重模糊")
        penalty += 38
        hard_unreadable = True
    elif row.lap_var < 35:
        reasons.append("清晰度不足，建议复核")
        penalty += 14
    elif row.lap_var < 65:
        reasons.append("清晰度一般")
        penalty += 5

    if row.contrast < 24:
        reasons.append("对比度低")
        penalty += 8
    if row.bright_ratio > 0.12 or row.glare_ratio > 0.08:
        reasons.append("过曝/反光明显")
        penalty += 8
    elif row.bright_ratio > 0.06 or row.glare_ratio > 0.035:
        reasons.append("局部反光或偏亮")
        penalty += 4
    if row.dark_ratio > 0.12 and row.brightness_mean < 95:
        reasons.append("欠曝/暗区偏多")
        penalty += 8
    if row.touch_edges >= 3 and row.bbox_area_ratio > 0.18:
        reasons.append("主体贴边明显")
        penalty += 4
    if row.fg_ratio > 0.80:
        reasons.append("背景/主体分离不稳定")
        penalty += 6

    score = max(0.0, 100.0 - penalty)
    if hard_unreadable or score < 55:
        status = "reject"
    elif score < 82 or reasons:
        status = "borderline"
    else:
        status = "pass"

    if not reasons:
        reasons.append("质量初筛未见明显问题")
    return status, score, "；".join(reasons)


def binary_metrics(df: pd.DataFrame, pred_col: str, label_col: str = "label_idx") -> dict[str, object]:
    if df.empty:
        return {"n": 0}
    y = df[label_col].astype(int).to_numpy()
    pred = df[pred_col].astype(int).to_numpy()
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    acc = float((tp + tn) / len(df))
    sens = float(tp / (tp + fn)) if tp + fn else math.nan
    spec = float(tn / (tn + fp)) if tn + fp else math.nan
    bacc = float(np.nanmean([sens, spec]))
    precision = float(tp / (tp + fp)) if tp + fp else math.nan
    f1 = float(2 * precision * sens / (precision + sens)) if precision + sens else math.nan
    return {
        "n": int(len(df)),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "sensitivity_high": sens,
        "specificity_low": spec,
        "precision_high": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    external_dir = find_external_dir(args.root)
    image_paths = sorted([p for p in external_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    rows = []
    for path in image_paths:
        try:
            rows.append(image_features(path))
        except Exception as exc:
            rows.append({
                "image_name": path.name,
                "original_case_id": case_id_from_name(path.name),
                "source_folder_local": path.parent.name,
                "local_path": str(path),
                "quality_status": "reject",
                "quality_score": 0,
                "quality_reasons": f"读取失败：{exc}",
            })

    q = pd.DataFrame(rows)
    decisions = q.apply(quality_decision, axis=1, result_type="expand")
    q["quality_status"] = decisions[0]
    q["quality_score"] = decisions[1]
    q["quality_reasons"] = decisions[2]
    q["quality_pass_for_model"] = q["quality_status"].isin(["pass", "borderline"]).astype(int)

    pred = pd.read_csv(args.pred)
    pred["original_case_id"] = pred["original_case_id"].astype(str)
    q["original_case_id"] = q["original_case_id"].astype(str)
    for c in ["label_idx", "locked162_blend_pred_idx", "strict_task7_eval"]:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")
    merged = pred.merge(q, on=["image_name", "original_case_id"], how="left")

    eval_rows = []
    pred_col = "locked162_blend_pred_idx"
    for subset_name, sub in [
        ("all_external", merged),
        ("strict_external", merged[merged["strict_task7_eval"] == 1]),
        ("quality_pass_all", merged[merged["quality_status"].isin(["pass", "borderline"])]),
        ("quality_pass_strict", merged[(merged["strict_task7_eval"] == 1) & (merged["quality_status"].isin(["pass", "borderline"]))]),
        ("quality_pass_only_strict", merged[(merged["strict_task7_eval"] == 1) & (merged["quality_status"] == "pass")]),
        ("borderline_strict", merged[(merged["strict_task7_eval"] == 1) & (merged["quality_status"] == "borderline")]),
        ("reject_strict", merged[(merged["strict_task7_eval"] == 1) & (merged["quality_status"] == "reject")]),
    ]:
        m = binary_metrics(sub, pred_col=pred_col)
        m["subset"] = subset_name
        m["coverage"] = len(sub) / len(merged) if len(merged) else 0
        m["strict_coverage"] = len(sub) / max(1, int((merged["strict_task7_eval"] == 1).sum()))
        eval_rows.append(m)

    quality_counts = q["quality_status"].value_counts().to_dict()
    reason_counts: dict[str, int] = {}
    for reason_text in q["quality_reasons"].fillna(""):
        for reason in reason_text.split("；"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    q.to_csv(args.outdir / "external_quality_features_v0.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(args.outdir / "external_quality_predictions_merged_v0.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eval_rows).to_csv(args.outdir / "external_quality_gate_eval_v0.csv", index=False, encoding="utf-8-sig")
    with (args.outdir / "external_quality_gate_summary_v0.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image_count": len(q),
                "quality_counts": quality_counts,
                "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                "prediction_file": str(args.pred),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(pd.DataFrame(eval_rows).to_string(index=False))
    print(json.dumps({"quality_counts": quality_counts, "reason_counts": reason_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
