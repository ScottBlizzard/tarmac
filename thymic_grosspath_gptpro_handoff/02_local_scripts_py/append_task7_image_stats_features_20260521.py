from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append simple image-stat features to cached Task7 DINO features.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-feature-dir", required=True)
    parser.add_argument("--third-feature-dir", required=True)
    parser.add_argument("--old-registry-csv", required=True)
    parser.add_argument("--third-registry-csv", required=True)
    parser.add_argument("--old-output-dir", required=True)
    parser.add_argument("--third-output-dir", required=True)
    parser.add_argument("--stats-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})


def load_old_feature(root: Path, feature_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / feature_dir
    table = read_csv(d / "case_dino_concat_feature_table.csv")
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, feat


def load_third_feature(root: Path, feature_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / feature_dir
    table = read_csv(d / "third_batch_dino_concat_feature_table.csv")
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, feat


def align_feature(order: pd.Series, table: pd.DataFrame, feat: np.ndarray) -> np.ndarray:
    idx = order.to_frame("case_id").merge(table[["case_id", "feature_idx"]], on="case_id", how="left")["feature_idx"]
    if idx.isna().any():
        missing = order[idx.isna()].head(10).tolist()
        raise KeyError(f"Missing features: {missing}")
    return feat[idx.astype(int).to_numpy()].astype(np.float32)


def resolve_path(path_value: object, root: Path) -> Path:
    p = Path(str(path_value))
    if p.exists():
        return p
    q = root / str(path_value)
    if q.exists():
        return q
    raise FileNotFoundError(str(path_value))


def image_stats(path: Path) -> dict[str, float]:
    with Image.open(path) as im:
        w, h = im.size
        im = im.convert("RGB")
        im.thumbnail((768, 768), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hh, ww = gray.shape
    m = max(4, int(min(hh, ww) * 0.06))
    border = np.concatenate(
        [
            arr[:m, :, :].reshape(-1, 3),
            arr[-m:, :, :].reshape(-1, 3),
            arr[:, :m, :].reshape(-1, 3),
            arr[:, -m:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(arr - bg.reshape(1, 1, 3), axis=2)
    fg = diff > max(25.0, float(np.std(border)) * 1.15)
    fg[:m, :] = False
    fg[-m:, :] = False
    fg[:, :m] = False
    fg[:, -m:] = False
    if fg.any():
        ys, xs = np.where(fg)
        bbox_area = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(hh * ww)
    else:
        bbox_area = 0.0

    rgb = arr.reshape(-1, 3)
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    sat = np.where(maxc > 1.0, (maxc - minc) / np.maximum(maxc, 1.0), 0.0)
    border_gray = 0.299 * border[:, 0] + 0.587 * border[:, 1] + 0.114 * border[:, 2]
    border_max = border.max(axis=1)
    border_min = border.min(axis=1)
    border_sat = np.where(border_max > 1.0, (border_max - border_min) / np.maximum(border_max, 1.0), 0.0)
    red_like = ((arr[:, :, 0] > arr[:, :, 1] * 1.10) & (arr[:, :, 0] > arr[:, :, 2] * 1.08) & (gray < 190)).mean()
    blue_bg = ((border[:, 2] > border[:, 0] + 8) & (border[:, 2] > border[:, 1] + 3)).mean()
    return {
        "log_megapixels": math.log1p(w * h / 1e6),
        "aspect": float(w / h) if h else 1.0,
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "saturation_mean": float(sat.mean()),
        "border_brightness": float(border_gray.mean()),
        "border_saturation": float(border_sat.mean()),
        "border_blue_ratio": float(blue_bg),
        "subject_area_proxy": float(fg.mean()),
        "subject_bbox_area_proxy": float(bbox_area),
        "red_tissue_ratio": float(red_like),
    }


def stats_frame(order: pd.Series, registry: pd.DataFrame, path_col: str, root: Path) -> pd.DataFrame:
    reg = registry[["case_id", path_col]].copy()
    rows = []
    for case_id in order.astype(str).tolist():
        hit = reg.loc[reg["case_id"].astype(str) == case_id]
        if hit.empty:
            raise KeyError(f"Missing registry row for {case_id}")
        path = resolve_path(hit.iloc[0][path_col], root)
        row = image_stats(path)
        row["case_id"] = case_id
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    old_table, old_dino_all = load_old_feature(root, args.old_feature_dir)
    third_table, third_dino_all = load_third_feature(root, args.third_feature_dir)
    old_order = old_table["case_id"].astype(str).reset_index(drop=True)
    third_order = third_table["case_id"].astype(str).reset_index(drop=True)

    old_registry = read_csv(root / args.old_registry_csv)
    third_registry = read_csv(root / args.third_registry_csv)
    old_stats = stats_frame(old_order, old_registry, "training_image_path", root)
    third_stats = stats_frame(third_order, third_registry, "image_path", root)
    stat_cols = [c for c in old_stats.columns if c != "case_id"]
    scaler = StandardScaler()
    old_stat_x = scaler.fit_transform(old_stats[stat_cols].astype(float)).astype(np.float32)
    third_stat_x = scaler.transform(third_stats[stat_cols].astype(float)).astype(np.float32)

    if args.stats_only:
        old_x = old_stat_x
        third_x = third_stat_x
    else:
        old_x = np.concatenate([align_feature(old_order, old_table, old_dino_all), old_stat_x], axis=1).astype(np.float32)
        third_x = np.concatenate([align_feature(third_order, third_table, third_dino_all), third_stat_x], axis=1).astype(np.float32)

    old_out = root / args.old_output_dir
    third_out = root / args.third_output_dir
    old_out.mkdir(parents=True, exist_ok=True)
    third_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": old_order, "feature_idx": np.arange(len(old_order), dtype=int)}).to_csv(
        old_out / "case_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"case_id": third_order, "feature_idx": np.arange(len(third_order), dtype=int)}).to_csv(
        third_out / "third_batch_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig"
    )
    np.save(old_out / "case_dino_concat_features.npy", old_x)
    np.save(third_out / "third_batch_dino_concat_features.npy", third_x)
    old_stats.to_csv(old_out / "old_image_stats.csv", index=False, encoding="utf-8-sig")
    third_stats.to_csv(third_out / "third_image_stats.csv", index=False, encoding="utf-8-sig")
    print(f"[done] old={old_x.shape} third={third_x.shape} stats={len(stat_cols)} stats_only={args.stats_only}", flush=True)


if __name__ == "__main__":
    main()
