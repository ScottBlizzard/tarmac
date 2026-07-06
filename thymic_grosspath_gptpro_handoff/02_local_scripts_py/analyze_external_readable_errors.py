from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


OUTDIR = Path("outputs/external_quality_gate_20260525")


def metrics(df: pd.DataFrame, pred_col: str = "locked162_blend_pred_idx") -> dict[str, object]:
    y = df["label_idx"].astype(int)
    p = pd.to_numeric(df[pred_col], errors="coerce").astype(int)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    n = len(df)
    acc = (tn + tp) / n if n else np.nan
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    bacc = np.nanmean([sens, spec])
    return {
        "n": n,
        "acc": acc,
        "bacc": bacc,
        "sensitivity_high": sens,
        "specificity_low": spec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def make_gallery(df: pd.DataFrame, name: str, max_n: int = 40) -> None:
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 17)
        small = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 14)
    except Exception:
        font = small = None
    cols = 4
    cell_w, cell_h = 300, 250
    rows = max(1, int(np.ceil(min(max_n, len(df)) / cols)))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    drawn = 0
    for _, r in df.head(max_n).iterrows():
        if not isinstance(r.get("local_path"), str):
            continue
        p = Path(r["local_path"])
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((280, 175), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_w, cell_h), "white")
        canvas.paste(im, ((cell_w - im.width) // 2, 5))
        draw = ImageDraw.Draw(canvas)
        pred = "H" if int(r["locked162_blend_pred_idx"]) == 1 else "L"
        truth = "H" if int(r["label_idx"]) == 1 else "L"
        status = "OK" if int(r["locked162_blend_correct"]) == 1 else "ERR"
        line1 = f"{r['original_case_id']} {r['task_l6_label']} T={truth} P={pred} {status}"
        prob = float(r["locked162_blend_prob_high"])
        line2 = f"pH={prob:.3f} q={r['gpt_quality_label_v1']}"
        line3 = str(r["gpt_quality_reason_v1"])[:32]
        draw.text((6, 184), line1, fill="black", font=font)
        draw.text((6, 207), line2, fill=(40, 40, 40), font=small)
        draw.text((6, 227), line3, fill=(70, 70, 70), font=small)
        sheet.paste(canvas, ((drawn % cols) * cell_w, (drawn // cols) * cell_h))
        drawn += 1
        if drawn >= max_n:
            break
    sheet.save(OUTDIR / name, quality=92)


def main() -> None:
    df = pd.read_csv(OUTDIR / "external_gpt_quality_labels_v1.csv")
    df["original_case_id"] = df["original_case_id"].astype(str)
    strict = df[df["strict_task7_eval"] == 1].copy()
    auto = strict[strict["gpt_quality_label_v1"] == "readable_auto"].copy()
    auto["error_type"] = np.select(
        [
            (auto["label_idx"].astype(int) == 1) & (auto["locked162_blend_pred_idx"].astype(int) == 0),
            (auto["label_idx"].astype(int) == 0) & (auto["locked162_blend_pred_idx"].astype(int) == 1),
            auto["locked162_blend_correct"].astype(int) == 1,
        ],
        ["FN_high_to_low", "FP_low_to_high", "correct"],
        default="unknown",
    )
    auto["prob_high"] = pd.to_numeric(auto["locked162_blend_prob_high"], errors="coerce")
    auto["confidence_margin"] = (auto["prob_high"] - 0.5).abs()

    summary_rows = []
    for name, sub in [
        ("strict_all", strict),
        ("readable_auto", auto),
        ("readable_auto_correct", auto[auto["error_type"] == "correct"]),
        ("readable_auto_FN_high_to_low", auto[auto["error_type"] == "FN_high_to_low"]),
        ("readable_auto_FP_low_to_high", auto[auto["error_type"] == "FP_low_to_high"]),
    ]:
        row = metrics(sub)
        row["subset"] = name
        summary_rows.append(row)

    subtype = (
        auto.groupby(["task_l6_label", "label_idx", "error_type"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["task_l6_label", "error_type"])
    )
    subtype_pivot = subtype.pivot_table(
        index=["task_l6_label", "label_idx"],
        columns="error_type",
        values="n",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for col in ["correct", "FN_high_to_low", "FP_low_to_high"]:
        if col not in subtype_pivot.columns:
            subtype_pivot[col] = 0
    subtype_pivot["total"] = subtype_pivot[["correct", "FN_high_to_low", "FP_low_to_high"]].sum(axis=1)
    subtype_pivot["acc"] = subtype_pivot["correct"] / subtype_pivot["total"].replace(0, np.nan)

    feature_cols = [
        "megapixels",
        "fg_ratio",
        "bbox_area_ratio",
        "touch_edges",
        "lap_var",
        "brightness_mean",
        "contrast",
        "bright_ratio",
        "glare_ratio",
        "saturation_mean",
        "confidence_margin",
        "prob_high",
    ]
    feat_summary = (
        auto.groupby("error_type")[feature_cols]
        .agg(["count", "mean", "median"])
        .reset_index()
    )

    auto.to_csv(OUTDIR / "external_readable_auto_case_error_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "external_readable_auto_error_summary.csv", index=False, encoding="utf-8-sig")
    subtype_pivot.to_csv(OUTDIR / "external_readable_auto_subtype_error_pivot.csv", index=False, encoding="utf-8-sig")
    feat_summary.to_csv(OUTDIR / "external_readable_auto_feature_error_summary.csv", index=False, encoding="utf-8-sig")

    make_gallery(auto[auto["error_type"] == "FN_high_to_low"].sort_values("prob_high"), "external_readable_auto_FN_gallery.jpg")
    make_gallery(auto[auto["error_type"] == "FP_low_to_high"].sort_values("prob_high", ascending=False), "external_readable_auto_FP_gallery.jpg")
    make_gallery(auto[auto["error_type"] == "correct"].sort_values("confidence_margin", ascending=False), "external_readable_auto_correct_confident_gallery.jpg")

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nSubtype pivot:")
    print(subtype_pivot.to_string(index=False))
    print("\nFeature summary:")
    print(feat_summary.to_string(index=False))


if __name__ == "__main__":
    main()
