from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "汇报" / "task56_case_review_assets"
IMAGE_DIR = ASSET_DIR / "images"
CASE_LIST_CSV = ASSET_DIR / "case_list.csv"
MANIFEST_CSV = ASSET_DIR / "manifest.csv"
OUTPUT_CSV = ASSET_DIR / "Task56医生复核与质量标注模板_2026-05-09.csv"
OUTPUT_MD = ROOT / "汇报" / "Task56模型问题分析与医生复核要点_2026-05-09.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_image(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


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
        return "高", touched
    if touched >= 1:
        return "中", touched
    return "低", touched


def label_by_thresholds(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value < low:
        return low_label
    if value < high:
        return mid_label
    return high_label


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
    if non_mask.any():
        bg_edge_density = float((edges[non_mask] > 0).mean())
    else:
        bg_edge_density = 0.0

    return {
        "blur_var": blur_var,
        "brightness": brightness,
        "contrast": contrast,
        "glare_ratio": glare_ratio,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "area_ratio": area_ratio,
        "border_touch_count": touch_count,
        "border_touch_label": touch_label,
        "bg_edge_density": bg_edge_density,
    }


def assign_case_labels(metrics_list: list[dict[str, float | str]]) -> dict[str, str | float]:
    blur_vals = np.array([m["blur_var"] for m in metrics_list], dtype=float)
    bright_vals = np.array([m["brightness"] for m in metrics_list], dtype=float)
    glare_vals = np.array([m["glare_ratio"] for m in metrics_list], dtype=float)
    area_vals = np.array([m["area_ratio"] for m in metrics_list], dtype=float)
    bg_vals = np.array([m["bg_edge_density"] for m in metrics_list], dtype=float)
    touch_counts = np.array([m["border_touch_count"] for m in metrics_list], dtype=float)

    blur_mean = float(blur_vals.mean())
    bright_mean = float(bright_vals.mean())
    glare_max = float(glare_vals.max())
    area_mean = float(area_vals.mean())
    bg_mean = float(bg_vals.mean())
    touch_max = int(touch_counts.max())

    clarity_label = "差" if blur_mean < 20 else "一般" if blur_mean < 45 else "好"
    if bright_mean < 105:
        exposure_label = "偏暗"
    elif bright_mean > 175:
        exposure_label = "偏亮"
    else:
        exposure_label = "正常"

    if glare_max >= 0.12:
        glare_label = "明显"
    elif glare_max >= 0.04:
        glare_label = "轻度"
    else:
        glare_label = "无"

    if area_mean < 0.28:
        specimen_ratio_label = "小"
    elif area_mean < 0.58:
        specimen_ratio_label = "中"
    else:
        specimen_ratio_label = "大"

    if bg_mean >= 0.08:
        background_label = "高"
    elif bg_mean >= 0.03:
        background_label = "中"
    else:
        background_label = "低"

    if touch_max >= 3:
        border_touch_label = "高"
    elif touch_max >= 1:
        border_touch_label = "中"
    else:
        border_touch_label = "低"

    if len(metrics_list) == 1:
        multiview_consistency = "单图"
    else:
        blur_cv = float(np.std(blur_vals) / (np.mean(blur_vals) + 1e-6))
        bright_std = float(np.std(bright_vals))
        area_std = float(np.std(area_vals))
        if blur_cv > 0.55 or bright_std > 22 or area_std > 0.12:
            multiview_consistency = "不一致"
        elif blur_cv > 0.28 or bright_std > 12 or area_std > 0.06:
            multiview_consistency = "轻度不一致"
        else:
            multiview_consistency = "一致"

    score_map = {
        "clarity": {"好": 2, "一般": 1, "差": 0}[clarity_label],
        "exposure": {"正常": 2, "偏暗": 1, "偏亮": 1}[exposure_label],
        "glare": {"无": 2, "轻度": 1, "明显": 0}[glare_label],
        "background": {"低": 2, "中": 1, "高": 0}[background_label],
        "consistency": {"一致": 2, "轻度不一致": 1, "不一致": 0, "单图": 1.5}[multiview_consistency],
    }
    raw_score = sum(score_map.values())
    quality_score = round(raw_score / 9.5 * 100, 1)
    if quality_score >= 75:
        overall_quality = "较好"
    elif quality_score >= 55:
        overall_quality = "一般"
    else:
        overall_quality = "较差"

    return {
        "ai_blur_var_mean": round(blur_mean, 2),
        "ai_brightness_mean": round(bright_mean, 2),
        "ai_glare_ratio_max": round(glare_max, 4),
        "ai_tissue_area_ratio_mean": round(area_mean, 4),
        "ai_bg_edge_density_mean": round(bg_mean, 4),
        "ai_border_touch_level": border_touch_label,
        "ai_image_clarity": clarity_label,
        "ai_exposure": exposure_label,
        "ai_glare_or_reflection": glare_label,
        "ai_background_clutter": background_label,
        "ai_specimen_ratio": specimen_ratio_label,
        "ai_multiview_consistency": multiview_consistency,
        "ai_quality_score": quality_score,
        "ai_overall_quality": overall_quality,
    }


def issue_hypothesis(task: str, true_name: str, pred_name: str, quality: dict[str, str | float]) -> tuple[str, str]:
    pair = frozenset([true_name, pred_name])
    task5_boundary = {frozenset(["TC", "B123"])}
    task6_boundary = {
        frozenset(["A", "AB"]),
        frozenset(["B1", "B2"]),
        frozenset(["B2", "B3"]),
        frozenset(["B2", "TC"]),
        frozenset(["B1", "A"]),
    }

    severe_blur = float(quality["ai_blur_var_mean"]) < 20
    quality_flag = (
        quality["ai_overall_quality"] == "较差"
        or quality["ai_glare_or_reflection"] == "明显"
        or quality["ai_background_clutter"] == "高"
        or severe_blur
        or quality["ai_multiview_consistency"] == "不一致"
    )

    if task == "Task5":
        if pair in task5_boundary and quality_flag:
            return "边界+质量双因素", "优先请医生判断：这例是真正位于 TC 与 B1-3 边界，还是图像质量/取材造成外观偏移。"
        if pair in task5_boundary:
            return "更像真实视觉边界", "优先请医生判断：这例从肉眼看是否确实更接近 B1-3，而不是典型 TC。"
        if quality_flag:
            return "更像图像质量/拍摄因素", "优先请医生判断：排除拍摄角度、反光、裁切或多图不一致导致的误判。"
        return "建议标签/取材复核", "如果医生也认为外观并不支持当前标签，建议回看取材与标签对应关系。"

    if pair in task6_boundary and quality_flag:
        return "边界+质量双因素", "优先请医生判断：这是典型边界类，同时图像质量因素可能放大了误判。"
    if pair in task6_boundary:
        return "更像真实视觉边界", "优先请医生判断：这例是否本来就处于 A/AB 或 B2/B3/TC 连续谱边界。"
    if quality_flag:
        return "更像图像质量/拍摄因素", "优先请医生判断：先排除质量、反光、构图和取材视角问题。"
    return "建议标签/取材复核", "模型高置信且不属于常见边界对，建议优先复核标签与标本对应关系。"


def merge_case_rows(case_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    manifest_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in manifest_rows:
        manifest_map[(row["task"], row["case_id"])].append(row["export_name"])

    merged = []
    for row in case_rows:
        export_names = manifest_map[(row["task"], row["case_id"])]
        metrics_list = [image_metrics(IMAGE_DIR / name) for name in export_names]
        quality = assign_case_labels(metrics_list)
        hypothesis, review_point = issue_hypothesis(row["task"], row["true_name"], row["pred_name"], quality)
        merged.append(
            {
                **row,
                "export_image_names": ";".join(export_names),
                **{k: str(v) for k, v in quality.items()},
                "ai_issue_hypothesis": hypothesis,
                "doctor_review_focus": review_point,
                "doctor_visual_boundary_blurry": "",
                "doctor_sampling_issue": "",
                "doctor_photo_issue": "",
                "doctor_label_recheck_needed": "",
                "doctor_case_difficulty": "",
                "doctor_comment": "",
            }
        )
    return merged


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "task",
        "case_id",
        "who_type_raw",
        "true_name",
        "pred_name",
        "fold_id",
        "pred_prob",
        "true_prob",
        "margin",
        "export_image_names",
        "ai_blur_var_mean",
        "ai_brightness_mean",
        "ai_glare_ratio_max",
        "ai_tissue_area_ratio_mean",
        "ai_bg_edge_density_mean",
        "ai_image_clarity",
        "ai_exposure",
        "ai_glare_or_reflection",
        "ai_background_clutter",
        "ai_specimen_ratio",
        "ai_border_touch_level",
        "ai_multiview_consistency",
        "ai_quality_score",
        "ai_overall_quality",
        "ai_issue_hypothesis",
        "doctor_review_focus",
        "doctor_visual_boundary_blurry",
        "doctor_sampling_issue",
        "doctor_photo_issue",
        "doctor_label_recheck_needed",
        "doctor_case_difficulty",
        "doctor_comment",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]]) -> None:
    task_rows = defaultdict(list)
    issue_counter = Counter()
    quality_counter = Counter()
    exposure_counter = Counter()
    glare_counter = Counter()
    for row in rows:
        task_rows[row["task"]].append(row)
        issue_counter[row["ai_issue_hypothesis"]] += 1
        quality_counter[row["ai_overall_quality"]] += 1
        exposure_counter[row["ai_exposure"]] += 1
        glare_counter[row["ai_glare_or_reflection"]] += 1

    lines = [
        "# Task5 / Task6 模型问题分析与医生复核要点",
        "",
        "日期：2026-05-09",
        "",
        "这份材料的目标不是替代医生判断，而是先把我们自己能做的工作做完：",
        "",
        "- 先用客观图像指标给 16 例高置信错例做一轮机器初筛",
        "- 把更像图像质量问题、真实视觉边界问题、还是值得标签/取材复核的问题先分开",
        "- 让医生后续只回答真正需要临床判断的问题",
        "",
        "本轮机器初筛基于以下客观指标：",
        "",
        "- 清晰度：以拉普拉斯方差近似反映模糊程度",
        "- 曝光：以平均亮度衡量整体偏暗或偏亮",
        "- 反光：以高亮低饱和像素比例衡量镜面反射/高光",
        "- 背景干扰：以背景边缘密度近似衡量杂乱程度",
        "- 标本占画面比例：近似反映主体是否过小",
        "- 贴边程度：近似反映是否存在明显近景裁切",
        "- 多图一致性：多图病例内部的亮度、清晰度和主体占比差异",
        "",
        "需要强调的是：这些标签是客观机器先验，不等于病理判断，也不等于最终临床结论。",
        "",
        "## 1. 总体筛查结论",
        "",
        f"- 共分析高置信错例 {len(rows)} 例，其中 Task5 {len(task_rows['Task5'])} 例，Task6 {len(task_rows['Task6'])} 例。",
        f"- 机器初筛的整体图像质量分布：较好 {quality_counter['较好']} 例，一般 {quality_counter['一般']} 例，较差 {quality_counter['较差']} 例。",
        f"- 曝光分布：正常 {exposure_counter['正常']} 例，偏暗 {exposure_counter['偏暗']} 例，偏亮 {exposure_counter['偏亮']} 例。",
        f"- 反光分布：无 {glare_counter['无']} 例，轻度 {glare_counter['轻度']} 例，明显 {glare_counter['明显']} 例。",
        "",
        "按问题归因初筛，当前 16 例可先分成以下几类：",
        "",
    ]
    for key in ["更像真实视觉边界", "边界+质量双因素", "更像图像质量/拍摄因素", "建议标签/取材复核"]:
        if issue_counter[key]:
            lines.append(f"- {key}：{issue_counter[key]} 例")
    lines += [
        "",
        "这说明目前高置信错例并不都是单纯图像质量差。相当一部分病例即使客观质量不差，依然落在我们已知的难边界带上，这更像是真实的视觉连续谱问题；只有一部分病例明显带有反光、贴边、背景干扰或多图不一致等质量因素。",
        "",
        "## 2. Task5 初筛结论",
        "",
        "Task5 的高置信错例全部来自真实为 TC、但被模型错分到 A/AB 或 B1-3 的病例。",
        "",
        "这批病例的机器初筛重点是回答两个问题：",
        "",
        "- 它们是不是真的视觉上更接近 B1-3，而不是典型 TC",
        "- 还是说图像质量、构图或取材问题把模型带偏了",
        "",
        "从当前初筛看，Task5 的主要矛盾仍然是 TC 与 B1-3 的边界，而不是全局质量普遍差。也就是说，Task5 的短板更像“真正难分”，而不是“图拍坏了”。",
        "",
        "## 3. Task6 初筛结论",
        "",
        "Task6 的高置信错例主要落在以下几条边界带：",
        "",
        "- A vs AB",
        "- B1 vs B2",
        "- B2 vs B3",
        "- B2 vs TC",
        "",
        "这与前面 confusion matrix 的统计是一致的，说明高置信错例不是随机出现，而是稳定集中在已知最难的边界类上。",
        "",
        "从机器初筛看，Task6 的错例里同时存在两类情况：",
        "",
        "- 一类图像质量并不差，但仍然高置信错，说明更像真实视觉边界问题",
        "- 另一类同时伴随明显反光、贴边或多图不一致，这类病例更适合先从图像因素排查",
        "",
        "## 4. 医生最值得优先看的问题",
        "",
        "基于这轮机器初筛，医生最值得优先判断的是下面几类问题：",
        "",
        "- 这例是不是肉眼上本来就位于 A/AB、B1/B2、B2/B3 或 B2/TC 连续谱边界",
        "- 这例的 gross image 与病理标签是否可能存在取材不完全一致",
        "- 这例是否属于医生自己也会犹豫的病例",
        "- 这例是否值得二次标签复核",
        "",
        "相对来说，清晰度、曝光、反光、贴边和多图一致性这些客观因素，我们已经先补了一版机器标签，医生不需要从零开始判断。",
        "",
        "## 5. 每例机器初筛摘要",
        "",
        "| 任务 | case_id | 真值 | 预测 | 质量 | 机器初筛判断 | 医生优先判断点 |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['case_id']} | {row['true_name']} | {row['pred_name']} | {row['ai_overall_quality']} | {row['ai_issue_hypothesis']} | {row['doctor_review_focus']} |"
        )

    lines += [
        "",
        "## 6. 文件说明",
        "",
        "机器预填后的完整模板在：",
        "",
        f"- `{OUTPUT_CSV.relative_to(ROOT)}`",
        "",
        "这个模板已经包含：",
        "",
        "- 每例任务、真值、预测、概率差值",
        "- 对应导出图片文件名",
        "- 机器预填的客观图像质量标签",
        "- 机器给出的初步问题归因",
        "- 留给医生填写的临床判断列",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    case_rows = read_csv(CASE_LIST_CSV)
    manifest_rows = read_csv(MANIFEST_CSV)
    merged = merge_case_rows(case_rows, manifest_rows)
    write_csv(merged)
    write_markdown(merged)
    print(OUTPUT_CSV)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
