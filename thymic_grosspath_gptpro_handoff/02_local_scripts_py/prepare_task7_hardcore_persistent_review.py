from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "task7_curriculum_runs" / "49_hardcore_persistent_review"
PRED12 = ROOT / "outputs" / "task7_curriculum_runs" / "12_stage2_salvage_foldwise_blend_noncore" / "oof_case_predictions_mean.csv"
PRED36 = ROOT / "outputs" / "task7_curriculum_runs" / "36_stage3_balcore_foldwise_blend_noncore" / "oof_case_predictions_mean.csv"
REGISTRY = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "frozen_inputs" / "combined_task567_registry.csv"
WEAKPLUS = ROOT / "reports" / "ThymicGross" / "experience_labeling" / "task7_antishortcut_aux_labels_weakplus_20260518.csv"
SOURCE_DIRS = [
    ROOT / "artifacts" / "task7_full_selected_images" / "selected_images",
    ROOT / "artifacts" / "task7_view_review_batch2",
    ROOT / "artifacts" / "image_review" / "thymic_gross_images",
    ROOT / "outputs" / "task7_curriculum_runs" / "17_salvage_failed_analysis" / "images",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task7 hard-core persistent wrong review package.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    return parser.parse_args()


def find_image(filename: str, fallback_names: list[str] | None = None) -> Path | None:
    candidates = [filename]
    if fallback_names:
        candidates.extend([name for name in fallback_names if name])
    for source in SOURCE_DIRS:
        if not source.exists():
            continue
        for candidate in candidates:
            direct = source / candidate
            if direct.exists():
                return direct
            matches = list(source.rglob(candidate))
            if matches:
                return matches[0]
    return None


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\msyh.ttc")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def summarize_exp(row: pd.Series) -> str:
    clues: list[str] = []
    if row.get("exp_manual_hemonec", "") in {"mild", "marked"}:
        clues.append("出血/坏死样")
    if row.get("exp_manual_irregularity", "") == "high":
        clues.append("不规则")
    if row.get("exp_manual_multinodular", "") == "yes":
        clues.append("多结节")
    if row.get("exp_manual_pale_uniform", "") == "yes":
        clues.append("淡白均一")
    if row.get("exp_manual_round_smooth", "") == "yes":
        clues.append("圆钝/光滑")
    if row.get("exp_manual_view_limit", "") == "yes":
        clues.append("视野受限")
    target = row.get("exp_manual_confound_target", "")
    if target and target != "none":
        clues.append(f"外观偏{target}")
    return "；".join(clues) if clues else "未见现成经验标签"


def initial_review(row: pd.Series) -> str:
    err = row["error_direction"]
    summary = row["experience_clues"]
    if summary == "未见现成经验标签":
        return "需医生重点复核；现有经验标签未覆盖，先按持续错判病例保留。"
    if err == "低危误升高危":
        return f"模型可能被高危样干扰带偏：{summary}。需医生判断这些线索是否只是取材/出血干扰。"
    return f"模型可能被低危样外观带偏：{summary}。需医生判断是否存在被图像忽略的高危区域。"


def build_contact_sheet(df: pd.DataFrame, image_dir: Path, out_path: Path, title: str) -> None:
    cols = 3
    thumb_w, thumb_h = 360, 250
    pad = 18
    text_h = 112
    title_h = 50
    rows = (len(df) + cols - 1) // cols
    sheet_w = cols * (thumb_w + pad) + pad
    sheet_h = title_h + rows * (thumb_h + text_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(24)
    body_font = load_font(17)
    small_font = load_font(15)
    draw.text((pad, 12), title, fill=(20, 45, 70), font=title_font)
    for idx, row in enumerate(df.itertuples(index=False)):
        x = pad + (idx % cols) * (thumb_w + pad)
        y = title_h + (idx // cols) * (thumb_h + text_h + pad)
        path = image_dir / row.image_file
        try:
            im = Image.open(path).convert("RGB")
            im = ImageOps.contain(im, (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
            bg = Image.new("RGB", (thumb_w, thumb_h), (245, 247, 250))
            bg.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            sheet.paste(bg, (x, y))
        except Exception:
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(180, 40, 40), width=2)
            draw.text((x + 8, y + 8), "image missing", fill=(180, 40, 40), font=body_font)
        text_y = y + thumb_h + 6
        header = f"{row.original_case_id} {row.task_l6_label} {row.error_direction} p高={row.prob_high_risk_group:.3f}"
        draw.text((x, text_y), header, fill=(0, 0, 0), font=body_font)
        text_y += 24
        for line in wrap_text(str(row.experience_clues), small_font, thumb_w, draw)[:3]:
            draw.text((x, text_y), line, fill=(40, 40, 40), font=small_font)
            text_y += 20
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PRED12)
    pred36 = pd.read_csv(PRED36)[["case_id", "prob_high_risk_group", "pred_idx"]].rename(
        columns={"prob_high_risk_group": "prob_high_risk_group_balcore_fusion", "pred_idx": "pred_idx_balcore_fusion"}
    )
    registry = pd.read_csv(REGISTRY, dtype=str).fillna("")
    weak = pd.read_csv(WEAKPLUS, dtype=str).fillna("") if WEAKPLUS.exists() else pd.DataFrame()

    pred["stage2_pred"] = (pred["prob_stage2_easy_medium"] >= 0.5).astype(int)
    pred["stage3_pred"] = (pred["prob_stage3_salvage_hard"] >= 0.5).astype(int)
    pred["blend_correct"] = pred["label_idx"] == pred["pred_idx"]
    pred["stage2_correct"] = pred["label_idx"] == pred["stage2_pred"]
    pred["stage3_correct"] = pred["label_idx"] == pred["stage3_pred"]
    df = pred[
        pred["difficulty_fine"].eq("hard_core")
        & (~pred["stage2_correct"])
        & (~pred["stage3_correct"])
        & (~pred["blend_correct"])
    ].copy()
    df = df.merge(
        registry[
            [
                "case_id",
                "original_case_id",
                "who_type_raw",
                "task_l6_label",
                "task_l7_label",
                "selected_original_image_name",
                "training_image_path",
                "selection_rule",
                "original_image_count",
            ]
        ],
        on="case_id",
        how="left",
    )
    if not weak.empty:
        df = df.merge(weak, on="case_id", how="left")
    df = df.merge(pred36, on="case_id", how="left")
    df["true_risk"] = df["label_idx"].map({0: "低危", 1: "高危"})
    df["model_pred"] = df["pred_idx"].map({0: "低危", 1: "高危"})
    df["error_direction"] = df.apply(
        lambda row: "低危误升高危" if int(row["label_idx"]) == 0 else "高危漏判低危", axis=1
    )
    df["wrong_confidence"] = df.apply(
        lambda row: row["prob_high_risk_group"] if int(row["label_idx"]) == 0 else 1 - row["prob_high_risk_group"],
        axis=1,
    )
    df["image_file"] = df["training_image_path"].map(lambda value: Path(str(value)).name)
    copied = []
    for row in df.itertuples(index=False):
        filename = str(row.image_file)
        source = find_image(filename, [str(row.selected_original_image_name)])
        copied.append(source is not None)
        if source is not None:
            shutil.copy2(source, image_dir / filename)
    df["image_found"] = copied
    df["experience_clues"] = df.apply(summarize_exp, axis=1)
    df["our_review_initial"] = df.apply(initial_review, axis=1)
    df = df.sort_values(["error_direction", "wrong_confidence"], ascending=[True, False]).reset_index(drop=True)

    keep_cols = [
        "case_id",
        "original_case_id",
        "who_type_raw",
        "task_l6_label",
        "true_risk",
        "model_pred",
        "error_direction",
        "prob_high_risk_group",
        "prob_stage2_easy_medium",
        "prob_stage3_salvage_hard",
        "prob_high_risk_group_balcore_fusion",
        "wrong_confidence",
        "original_image_count",
        "selected_original_image_name",
        "image_file",
        "image_found",
        "experience_clues",
        "our_review_initial",
    ]
    df[keep_cols].to_csv(out_dir / "hardcore_persistent_wrong_43_review_table.csv", index=False, encoding="utf-8-sig")

    for direction, sub in df.groupby("error_direction", sort=False):
        safe = "low_to_high" if direction == "低危误升高危" else "high_to_low"
        build_contact_sheet(
            sub,
            image_dir,
            out_dir / f"hardcore_persistent_{safe}_contact_sheet.jpg",
            f"Task7 核心 hard 持续错判：{direction}（n={len(sub)}）",
        )

    print(f"persistent wrong rows={len(df)}")
    print(df["error_direction"].value_counts().to_string())
    print(f"images found={int(df['image_found'].sum())}/{len(df)}")
    print(out_dir)


if __name__ == "__main__":
    main()
