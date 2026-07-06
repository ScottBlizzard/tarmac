from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "third_batch_shift_20260521"
OUT.mkdir(parents=True, exist_ok=True)

OLD_REGISTRY = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "frozen_inputs" / "combined_task567_registry_with_gross_findings_20260520.csv"
OLD_VIEWTYPE = OUT / "old_viewtype_full_template.csv"
THIRD_REGISTRY = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "01_third_batch_64style_image_only_20260521" / "third_batch_task7_registry.csv"
THIRD_DIR = ROOT / "第三批306例"
THIRD_RAWTOP_BEST = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "02_third_batch_rawtop_stack_20260521" / "third_batch_rawtop_stack_best_case_predictions.csv"
THIRD_64 = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "01_third_batch_64style_image_only_20260521" / "third_batch_external_case_predictions.csv"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _resolve_old_path(path_value: str, source_case_folder: str | None = None, selected_image_name: str | None = None) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    p = Path(path_value)
    if p.exists():
        return p
    if source_case_folder and selected_image_name:
        local = ROOT / "artifacts" / "image_review" / "thymic_gross_images" / str(source_case_folder) / str(selected_image_name)
        if local.exists():
            return local
    return None


def _resolve_third_path(row: pd.Series) -> Path | None:
    p = THIRD_DIR / str(row["source_folder"]) / str(row["image_name"])
    return p if p.exists() else None


def _image_stats(path: Path) -> dict:
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        im.draft("RGB", (768, 768))
        im = im.convert("RGB")
        small = im.copy()
        small.thumbnail((768, 768), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
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
    border_gray = 0.299 * border[:, 0] + 0.587 * border[:, 1] + 0.114 * border[:, 2]
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(arr - bg.reshape(1, 1, 3), axis=2)
    fg = diff > max(28.0, float(np.std(border_gray)) * 1.4)
    # Avoid counting the outer frame itself as foreground.
    fg[:m, :] = False
    fg[-m:, :] = False
    fg[:, :m] = False
    fg[:, -m:] = False
    rgb = arr.reshape(-1, 3)
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    sat = np.where(maxc > 0, (maxc - minc) / maxc, 0.0)
    border_max = border.max(axis=1)
    border_min = border.min(axis=1)
    border_sat = np.where(border_max > 0, (border_max - border_min) / border_max, 0.0)
    red_like = ((arr[:, :, 0] > arr[:, :, 1] * 1.10) & (arr[:, :, 0] > arr[:, :, 2] * 1.08) & (gray < 185)).mean()
    pale_bg = ((border_gray > 185) & (border_sat < 0.13)).mean()
    blue_bg = ((border[:, 2] > border[:, 0] + 8) & (border[:, 2] > border[:, 1] + 3)).mean()
    return {
        "width": orig_w,
        "height": orig_h,
        "megapixels": orig_w * orig_h / 1e6,
        "aspect": orig_w / orig_h if orig_h else math.nan,
        "file_kb": path.stat().st_size / 1024,
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "saturation_mean": float(sat.mean()),
        "border_brightness": float(border_gray.mean()),
        "border_saturation": float(border_sat.mean()),
        "border_blue_ratio": float(blue_bg),
        "border_pale_ratio": float(pale_bg),
        "subject_area_proxy": float(fg.mean()),
        "red_tissue_ratio": float(red_like),
    }


def _summarize_numeric(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group_col):
        row = {group_col: name, "n": len(g)}
        for c in cols:
            row[f"{c}_median"] = float(g[c].median())
            row[f"{c}_mean"] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _try_font(size: int = 22) -> ImageFont.ImageFont:
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def _make_contact_sheet(rows: list[dict], out_path: Path, title: str, cols: int = 5, tile: int = 260) -> None:
    font = _try_font(18)
    title_font = _try_font(24)
    pad = 12
    label_h = 58
    rows_n = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * tile + (cols + 1) * pad, rows_n * (tile + label_h) + (rows_n + 1) * pad + 44), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, pad), title, fill=(0, 0, 0), font=title_font)
    y0 = pad + 44
    for i, item in enumerate(rows):
        x = pad + (i % cols) * (tile + pad)
        y = y0 + (i // cols) * (tile + label_h + pad)
        try:
            with Image.open(item["path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
                bg = Image.new("RGB", (tile, tile), (245, 245, 245))
                bg.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
                sheet.paste(bg, (x, y))
        except Exception as exc:
            draw.rectangle((x, y, x + tile, y + tile), fill=(240, 220, 220))
            draw.text((x + 6, y + 6), f"open failed\n{exc}", fill=(130, 0, 0), font=font)
        label = item["label"]
        draw.text((x, y + tile + 4), label, fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def _sample_rows(df: pd.DataFrame, label_col: str, path_col: str, per_group: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for label, g in df.groupby(label_col, sort=True):
        items = g.to_dict("records")
        rng.shuffle(items)
        for r in items[:per_group]:
            pred_part = ""
            if "pred_rawtop" in r:
                pred_part = f"\nrawtop:{r.get('pred_rawtop')} pH={float(r.get('prob_high_rawtop', 0)):.2f} ok={int(r.get('correct_rawtop', 0))}"
            rows.append(
                {
                    "path": Path(r[path_col]),
                    "label": f"{r.get('original_case_id', r.get('case_id'))} {r[label_col]}{pred_part}",
                }
            )
    return rows


def main() -> None:
    old_reg = _safe_read_csv(OLD_REGISTRY)
    old_view = _safe_read_csv(OLD_VIEWTYPE)
    third = _safe_read_csv(THIRD_REGISTRY)

    old_view["local_path_resolved"] = old_view.apply(
        lambda r: str(_resolve_old_path(r.get("local_review_path", ""), r.get("source_case_folder", ""), r.get("selected_image_name", "")) or ""),
        axis=1,
    )
    old_view = old_view[old_view["local_path_resolved"].map(lambda x: bool(x) and Path(x).exists())].copy()
    old_view["task_l6_label"] = old_view["who_type_raw"].map(
        {
            "thymoma_A": "A",
            "thymoma_AB": "AB",
            "thymoma_B1": "B1",
            "thymoma_B2": "B2",
            "thymoma_B3": "B3",
            "thymic_carcinoma": "TC",
        }
    )
    old_view["task_l7_label"] = np.where(old_view["task_l6_label"].isin(["A", "AB", "B1"]), "low_risk_group", "high_risk_group")

    third["local_path_resolved"] = third.apply(lambda r: str(_resolve_third_path(r) or ""), axis=1)
    third = third[third["local_path_resolved"].map(lambda x: bool(x) and Path(x).exists())].copy()

    if THIRD_RAWTOP_BEST.exists():
        pred = _safe_read_csv(THIRD_RAWTOP_BEST)
        pred = pred[["case_id", "prob_high_risk_group", "pred_idx", "correct"]].rename(
            columns={
                "prob_high_risk_group": "prob_high_rawtop",
                "pred_idx": "pred_idx_rawtop",
                "correct": "correct_rawtop",
            }
        )
        third = third.merge(pred, on="case_id", how="left")
        third["pred_rawtop"] = third["pred_idx_rawtop"].map({0: "low", 1: "high"})

    stat_rows = []
    for dataset_name, df, path_col in [
        ("old_train", old_view, "local_path_resolved"),
        ("third_batch", third, "local_path_resolved"),
    ]:
        for _, row in df.iterrows():
            path = Path(row[path_col])
            try:
                s = _image_stats(path)
            except Exception as exc:
                continue
            s.update(
                {
                    "dataset": dataset_name,
                    "case_id": row.get("case_id", ""),
                    "original_case_id": row.get("original_case_id", row.get("case_id", "")),
                    "task_l6_label": row.get("task_l6_label", ""),
                    "task_l7_label": row.get("task_l7_label", ""),
                    "view_type_round1": row.get("view_type_round1", ""),
                    "path": str(path),
                }
            )
            if dataset_name == "third_batch":
                s["source_folder"] = row.get("source_folder", "")
                s["prob_high_rawtop"] = row.get("prob_high_rawtop", np.nan)
                s["pred_idx_rawtop"] = row.get("pred_idx_rawtop", np.nan)
                s["correct_rawtop"] = row.get("correct_rawtop", np.nan)
            stat_rows.append(s)
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(OUT / "old_vs_third_image_quality_stats.csv", index=False, encoding="utf-8-sig")

    # Class/view composition.
    comp_rows = []
    for dataset_name, df in [("old_train", old_reg), ("third_batch", third)]:
        for col in ["task_l6_label", "task_l7_label"]:
            vc = df[col].value_counts(dropna=False)
            for k, v in vc.items():
                comp_rows.append({"dataset": dataset_name, "dimension": col, "label": k, "count": int(v), "pct": float(v / len(df))})
    pd.DataFrame(comp_rows).to_csv(OUT / "old_vs_third_label_distribution.csv", index=False, encoding="utf-8-sig")

    if "view_type_round1" in old_view.columns:
        old_view["view_type_round1"].value_counts(dropna=False).rename_axis("view_type").reset_index(name="count").to_csv(
            OUT / "old_viewtype_counts_from_template.csv", index=False, encoding="utf-8-sig"
        )

    num_cols = [
        "megapixels",
        "brightness_mean",
        "contrast_std",
        "saturation_mean",
        "border_brightness",
        "border_saturation",
        "border_blue_ratio",
        "border_pale_ratio",
        "subject_area_proxy",
        "red_tissue_ratio",
    ]
    summary_dataset = _summarize_numeric(stats, "dataset", num_cols)
    summary_dataset.to_csv(OUT / "old_vs_third_quality_summary_by_dataset.csv", index=False, encoding="utf-8-sig")
    summary_l6 = _summarize_numeric(stats, "task_l6_label", num_cols)
    summary_l6.to_csv(OUT / "old_vs_third_quality_summary_by_l6_mixed.csv", index=False, encoding="utf-8-sig")

    # Third error tables grouped by subtype.
    if "correct_rawtop" in third.columns:
        err = third.copy()
        err["rawtop_correct"] = err["correct_rawtop"].fillna(-1).astype(int)
        by_subtype = (
            err.groupby("task_l6_label")
            .agg(n=("case_id", "size"), acc=("rawtop_correct", "mean"), mean_p_high=("prob_high_rawtop", "mean"))
            .reset_index()
        )
        by_subtype.to_csv(OUT / "third_rawtop_error_by_subtype.csv", index=False, encoding="utf-8-sig")
        third.to_csv(OUT / "third_with_rawtop_predictions_and_paths.csv", index=False, encoding="utf-8-sig")

    # Contact sheets: balanced old by six types, third by four folders, and third rawtop errors.
    old_for_sheet = old_view.rename(columns={"local_path_resolved": "path_sheet"})
    third_for_sheet = third.rename(columns={"local_path_resolved": "path_sheet"})
    _make_contact_sheet(
        _sample_rows(old_for_sheet, "task_l6_label", "path_sheet", per_group=5, seed=11),
        OUT / "contact_old_train_by_l6.jpg",
        "旧训练集抽样：六分类各5例",
        cols=5,
    )
    _make_contact_sheet(
        _sample_rows(third_for_sheet, "task_l6_label", "path_sheet", per_group=8, seed=13),
        OUT / "contact_third_batch_by_l6.jpg",
        "第三批抽样：四类各最多8例",
        cols=4,
    )
    if "correct_rawtop" in third_for_sheet.columns:
        wrong = third_for_sheet[third_for_sheet["correct_rawtop"] == 0].copy()
        right = third_for_sheet[third_for_sheet["correct_rawtop"] == 1].copy()
        _make_contact_sheet(
            _sample_rows(wrong, "task_l6_label", "path_sheet", per_group=8, seed=19),
            OUT / "contact_third_rawtop_wrong_by_l6.jpg",
            "第三批 rawtop 错例抽样：按真实类别",
            cols=4,
        )
        _make_contact_sheet(
            _sample_rows(right, "task_l6_label", "path_sheet", per_group=6, seed=23),
            OUT / "contact_third_rawtop_right_by_l6.jpg",
            "第三批 rawtop 对例抽样：按真实类别",
            cols=4,
        )

    # Write a compact markdown note for quick review.
    with (OUT / "third_batch_shift_quick_summary.md").open("w", encoding="utf-8") as f:
        f.write("# 第三批与旧训练集差异快速核查\n\n")
        f.write("## 类别构成\n\n")
        f.write(pd.DataFrame(comp_rows).to_string(index=False))
        f.write("\n\n## 图像统计：按数据集\n\n")
        f.write(summary_dataset.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if (OUT / "third_rawtop_error_by_subtype.csv").exists():
            f.write("\n\n## 第三批 rawtop 外部测试：按亚型\n\n")
            f.write(pd.read_csv(OUT / "third_rawtop_error_by_subtype.csv").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    print(f"wrote outputs to {OUT}")
    print(summary_dataset.to_string(index=False))
    if (OUT / "third_rawtop_error_by_subtype.csv").exists():
        print(pd.read_csv(OUT / "third_rawtop_error_by_subtype.csv").to_string(index=False))


if __name__ == "__main__":
    main()
