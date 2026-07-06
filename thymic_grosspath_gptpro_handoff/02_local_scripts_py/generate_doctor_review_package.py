from __future__ import annotations

import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "汇报"
THYMIC_REPORT_DIR = ROOT / "reports" / "ThymicGross"
ASSET_DIR = REPORT_DIR / "task56_case_review_assets"
IMAGE_DIR = ASSET_DIR / "images"
PDF_CACHE_DIR = ASSET_DIR / "pdf_cache"
CASE_CSV = ASSET_DIR / "case_list.csv"
MANIFEST_CSV = ASSET_DIR / "manifest.csv"
TEMPLATE_CSV = ASSET_DIR / "Task56医生复核与质量标注模板_2026-05-09.csv"
CASE_QUALITY_CSV = ROOT / "artifacts" / "image_review" / "full_case_quality_labels.csv"
IMAGE_QUALITY_CSV = ROOT / "artifacts" / "image_review" / "full_image_quality_labels.csv"
EXP_CSV = THYMIC_REPORT_DIR / "experience_labeling" / "task56_experience_label_core_round2_merged.csv"

CASE_REVIEW_MD = REPORT_DIR / "Task56高置信错例医生回看清单_2026-05-07.md"
CASE_REVIEW_PDF = REPORT_DIR / "Task56高置信错例医生回看清单_2026-05-07.pdf"
QUALITY_MD = REPORT_DIR / "Task56全量图像质量初筛说明_2026-05-10.md"
QUALITY_PDF = REPORT_DIR / "Task56全量图像质量初筛说明_2026-05-10.pdf"
QUALITY_CASE_EXPORT = REPORT_DIR / "Task56全量图像质量初筛标签_病例级_2026-05-10.csv"
QUALITY_IMAGE_EXPORT = REPORT_DIR / "Task56全量图像质量初筛标签_图像级_2026-05-10.csv"
EXP_MD = REPORT_DIR / "Task56经验标签核心集说明_2026-05-10.md"
EXP_PDF = REPORT_DIR / "Task56经验标签核心集说明_2026-05-10.pdf"
EXP_EXPORT = REPORT_DIR / "Task56经验标签核心集_医生版_2026-05-10.csv"
INDEX_MD = REPORT_DIR / "Task56医生复核材料目录_2026-05-10.md"
INDEX_PDF = REPORT_DIR / "Task56医生复核材料目录_2026-05-10.pdf"

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiDoctorReviewPkg"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=18.5,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d"),
        spaceAfter=4 * mm,
    )
    h1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=12.5,
        leading=16.5,
        textColor=colors.HexColor("#16324f"),
        spaceBefore=3 * mm,
        spaceAfter=1.5 * mm,
    )
    h2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=10.8,
        leading=14.5,
        textColor=colors.HexColor("#244a73"),
        spaceBefore=2 * mm,
        spaceAfter=1.2 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9.2,
        leading=13.4,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#222222"),
        spaceAfter=1.4 * mm,
    )
    note = ParagraphStyle(
        "NoteCN",
        parent=body,
        backColor=colors.HexColor("#f6f8fb"),
        borderColor=colors.HexColor("#d6e0ec"),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=1 * mm,
        spaceAfter=1.8 * mm,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=8 * mm,
        firstLineIndent=-4.5 * mm,
    )
    table_header = ParagraphStyle(
        "TableHeaderCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.0,
        leading=9.8,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_body = ParagraphStyle(
        "TableBodyCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=7.9,
        leading=9.6,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    caption = ParagraphStyle(
        "CaptionCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.3,
        leading=10.0,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#465a6d"),
        spaceBefore=1 * mm,
        spaceAfter=2 * mm,
    )
    return title, h1, h2, body, note, bullet, table_header, table_body, caption


def clean(text: object) -> str:
    return html.escape(str(text)).replace("`", "").replace("**", "")


def build_table(data, widths, header_style, body_style):
    wrapped = []
    for i, row in enumerate(data):
        style = header_style if i == 0 else body_style
        wrapped.append([Paragraph(clean(cell), style) for cell in row])
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
            ]
        )
    )
    return table


def parse_markdown(md_text: str):
    blocks: list[tuple[str, object]] = []
    lines = md_text.splitlines()
    paragraph: list[str] = []
    i = 0

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            blocks.append(("p", " ".join(paragraph).strip()))
            paragraph = []

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flush_paragraph()
            i += 1
            continue
        if line.startswith("# "):
            flush_paragraph()
            blocks.append(("title", line[2:].strip()))
            i += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            blocks.append(("h1", line[3:].strip()))
            i += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            blocks.append(("h2", line[4:].strip()))
            i += 1
            continue
        if line.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", line[2:].strip()))
            i += 1
            continue
        if line.startswith("|"):
            flush_paragraph()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw = lines[i].strip()
                if set(raw.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                    i += 1
                    continue
                rows.append([cell.strip() for cell in raw.strip("|").split("|")])
                i += 1
            blocks.append(("table", rows))
            continue
        paragraph.append(line.strip())
        i += 1

    flush_paragraph()
    return blocks


def generic_table_widths(ncols: int) -> list[float]:
    inner = A4[0] - 28 * mm
    if ncols == 4:
        ratios = [0.18, 0.22, 0.24, 0.36]
    elif ncols == 5:
        ratios = [0.16, 0.22, 0.16, 0.16, 0.30]
    elif ncols == 6:
        ratios = [0.15, 0.20, 0.17, 0.15, 0.13, 0.20]
    else:
        ratios = [1 / ncols] * ncols
    return [inner * r for r in ratios]


def render_markdown_pdf(md_path: Path, pdf_path: Path) -> None:
    register_font()
    title, h1, h2, body, note, bullet, table_header, table_body, caption = build_styles()
    blocks = parse_markdown(md_path.read_text(encoding="utf-8"))
    story = []
    for kind, payload in blocks:
        if kind == "title":
            story.append(Paragraph(clean(payload), title))
        elif kind == "h1":
            story.append(Paragraph(clean(payload), h1))
        elif kind == "h2":
            story.append(Paragraph(clean(payload), h2))
        elif kind == "p":
            story.append(Paragraph(clean(payload), body))
        elif kind == "bullet":
            story.append(Paragraph(f"• {clean(payload)}", bullet))
        elif kind == "table":
            rows = payload
            ncols = max(len(r) for r in rows)
            rows = [r + [""] * (ncols - len(r)) for r in rows]
            story.append(build_table(rows, generic_table_widths(ncols), table_header, table_body))
            story.append(Spacer(1, 2 * mm))
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=md_path.stem,
        author="Thymic Gross Team",
    )
    doc.build(story)


def get_pdf_cache_image(path: Path) -> Path:
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PDF_CACHE_DIR / path.name
    if cache_path.exists():
        return cache_path
    with PILImage.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((1800, 1800))
        img.save(cache_path, format="JPEG", quality=82, optimize=True)
    return cache_path


def image_flowable(path: Path, max_w_mm: float, max_h_mm: float) -> Image:
    cache_path = get_pdf_cache_image(path)
    reader = ImageReader(str(cache_path))
    w, h = reader.getSize()
    max_w = max_w_mm * mm
    max_h = max_h_mm * mm
    scale = min(max_w / w, max_h / h)
    return Image(str(cache_path), width=w * scale, height=h * scale)


def build_image_table(paths: list[Path], caption_style: ParagraphStyle):
    cells = []
    per_width = 78 if len(paths) > 1 else 120
    for idx, path in enumerate(paths, start=1):
        img = image_flowable(path, per_width, 78)
        cap = Paragraph(clean(f"图 {idx}: {path.name}"), caption_style)
        cells.append([img, cap])
    if len(cells) == 1:
        rows = [[cells[0][0]], [cells[0][1]]]
        return Table(rows, colWidths=[120 * mm], style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    img_row = [cell[0] for cell in cells]
    cap_row = [cell[1] for cell in cells]
    table = Table([img_row, cap_row], colWidths=[78 * mm] * len(cells))
    table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def load_case_review_rows():
    case_df = pd.read_csv(CASE_CSV, encoding="utf-8-sig")
    manifest_df = pd.read_csv(MANIFEST_CSV, encoding="utf-8-sig")
    template_df = pd.read_csv(TEMPLATE_CSV, encoding="utf-8-sig")
    exp_df = pd.read_csv(EXP_CSV)

    manifest_map: dict[tuple[str, int], list[str]] = defaultdict(list)
    for _, row in manifest_df.iterrows():
        manifest_map[(row["task"], int(row["case_id"]))].append(str(row["export_name"]))

    template_map = {(row["task"], int(row["case_id"])): row for _, row in template_df.iterrows()}

    exp_group = defaultdict(list)
    for _, row in exp_df.iterrows():
        exp_group[int(row["case_id"])].append(row)

    rows = []
    for _, row in case_df.iterrows():
        key = (row["task"], int(row["case_id"]))
        temp = template_map[key]
        exp_rows = exp_group[int(row["case_id"])]
        key_clues = unique_texts([r["exp_round2_key_discriminative_clues"] for r in exp_rows])
        conf_clues = unique_texts([r["exp_round2_confounding_clues"] for r in exp_rows])
        stabilities = [str(r["exp_round2_label_stability"]) for r in exp_rows if pd.notna(r["exp_round2_label_stability"])]
        boundary_axes = [str(r["exp_round1_boundary_axis"]) for r in exp_rows if pd.notna(r["exp_round1_boundary_axis"]) and str(r["exp_round1_boundary_axis"]) != "none"]
        revised_task5 = unique_texts([r["exp_round2_revised_task5_guess"] for r in exp_rows if pd.notna(r["exp_round2_revised_task5_guess"])])
        revised_task6 = unique_texts([r["exp_round2_revised_task6_guess"] for r in exp_rows if pd.notna(r["exp_round2_revised_task6_guess"])])
        visual_summary = unique_texts([r["ai_visual_summary"] for r in exp_rows if pd.notna(r["ai_visual_summary"])], limit=2)
        blind_task5 = unique_texts([r["ai_blind_guess_task5"] for r in exp_rows if pd.notna(r["ai_blind_guess_task5"])])
        blind_task6 = unique_texts([r["ai_blind_guess_task6"] for r in exp_rows if pd.notna(r["ai_blind_guess_task6"])])

        rows.append(
            {
                "task": row["task"],
                "case_id": int(row["case_id"]),
                "who_type_raw": row["who_type_raw"],
                "true_name": row["true_name"],
                "pred_name": row["pred_name"],
                "fold_id": int(row["fold_id"]),
                "pred_prob": float(row["pred_prob"]),
                "true_prob": float(row["true_prob"]),
                "margin": float(row["margin"]),
                "export_images": sorted(manifest_map[key]),
                "ai_overall_quality": temp["ai_overall_quality"],
                "ai_image_clarity": temp["ai_image_clarity"],
                "ai_exposure": temp["ai_exposure"],
                "ai_background_clutter": temp["ai_background_clutter"],
                "ai_multiview_consistency": temp["ai_multiview_consistency"],
                "ai_issue_hypothesis": temp["ai_issue_hypothesis"],
                "doctor_review_focus": temp["doctor_review_focus"],
                "key_clues": key_clues,
                "confounding_clues": conf_clues,
                "stability_summary": summarize_stability(stabilities),
                "boundary_axes": "、".join(boundary_axes) if boundary_axes else "未见明确单一边界轴",
                "revised_task5": " / ".join(revised_task5) if revised_task5 else "",
                "revised_task6": " / ".join(revised_task6) if revised_task6 else "",
                "visual_summary": visual_summary,
                "blind_task5": " / ".join(blind_task5) if blind_task5 else "",
                "blind_task6": " / ".join(blind_task6) if blind_task6 else "",
            }
        )
    return rows


def unique_texts(values, limit: int | None = None) -> list[str]:
    out = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text == "nan":
            continue
        if text not in out:
            out.append(text)
    return out[:limit] if limit is not None else out


def summarize_stability(values: list[str]) -> str:
    if not values:
        return "未标注"
    s = set(values)
    if "conflicting_clues" in s:
        return "存在明显冲突线索"
    if "uncertain" in s and "stable" in s:
        return "部分视图稳定，部分视图仍犹豫"
    if "uncertain" in s:
        return "主要属于边界犹豫型"
    return "整体较稳定"


def build_case_reason(row: dict) -> str:
    parts = [
        f"图像整体质量{row['ai_overall_quality']}，清晰度{row['ai_image_clarity']}，曝光{row['ai_exposure']}，背景干扰{row['ai_background_clutter']}。"
    ]
    parts.append(f"按目前这轮整理，我们更倾向把它放在“{row['ai_issue_hypothesis']}”这一类。")
    if row["key_clues"]:
        parts.append(f"支持当前判断的线索主要有：{'; '.join(row['key_clues'])}。")
    if row["confounding_clues"]:
        parts.append(f"容易把判断带偏的地方主要有：{'; '.join(row['confounding_clues'])}。")
    if row["boundary_axes"] and row["boundary_axes"] != "未见明确单一边界轴":
        parts.append(f"从经验标签看，这例主要卡在“{row['boundary_axes']}”这条边界上。")
    return " ".join(parts)


def build_doctor_questions(row: dict) -> list[str]:
    issue = row["ai_issue_hypothesis"]
    questions = []
    if "标签/取材复核" in issue:
        questions.append("这例从肉眼看是否支持当前病理标签？是否存在取材与病理标签不完全对应的可能？")
        questions.append("如果医生也觉得外观与当前标签不完全匹配，是否建议列入二次标签复核？")
    elif "图像质量/拍摄因素" in issue:
        questions.append("这例是否存在拍摄视角、切面不足、碎片化、出血遮挡或构图问题，导致外观被放大或误导？")
        questions.append("如果忽略这些拍摄和取材因素，这例在临床上更接近哪一类？")
    elif "边界+质量双因素" in issue:
        questions.append("这例更像真实边界病例，还是更像图像/取材因素叠加后造成的误判？")
        questions.append("如果从临床经验看，它更接近哪一侧？是否值得同时记录为边界病例？")
    else:
        questions.append("这例从肉眼看是否本来就位于已知连续谱边界上？如果是，更接近哪一侧？")
        questions.append("这类边界在临床上是否也常见？是否属于医生自己也会犹豫的病例？")
    return questions


def write_case_review_markdown(rows: list[dict]) -> None:
    counts = Counter(row["ai_issue_hypothesis"] for row in rows)
    lines = []
    lines.append("# Task5 与 Task6 典型边界错例医生回看清单")
    lines.append("日期：2026-05-10")
    lines.append("")
    lines.append("## 1. 这份清单是做什么的")
    lines.append("这里放的是目前最值得优先回看的 16 例错例。每一例前面都先写清我们已经看到的情况：图像质量大致如何，哪些线索支持原标签，哪些地方容易让判断跑偏，以及最想请老师帮忙确认的问题。")
    lines.append("")
    lines.append("## 2. 我们已经先做了哪些分析")
    lines.append("- 先把 16 例都过了一遍基础质控。")
    lines.append("- 对进入经验标签核心集的图补了视觉线索、干扰线索和主要边界轴。")
    lines.append("- 先按“更像真实边界”“边界和质量双因素”“更像拍摄因素”“建议标签或取材复核”分开整理。")
    lines.append("")
    lines.append("当前 16 例的机器初筛归因分布为：")
    for name, count in counts.items():
        lines.append(f"- {name}：{count} 例")
    lines.append("")
    lines.append("## 3. 医生回看时最值得回答的问题")
    lines.append("- 这例是不是肉眼上本来就位于 A/AB、B1/B2、B2/B3 或 B2/TC 连续谱边界？")
    lines.append("- 这例的 gross image 与病理标签是否可能存在取材或对应关系问题？")
    lines.append("- 这例是否属于医生自己也会犹豫的病例？")
    lines.append("- 这例如果要再补标签，最值得补的是“边界病例”还是“建议复核标签”？")
    lines.append("")
    lines.append("## 4. 逐例清单")
    lines.append("")
    for idx, row in enumerate(rows, start=1):
        lines.append(f"### {idx}. {row['task']} | case_id={row['case_id']}")
        lines.append(f"- 原始病理标签：{row['who_type_raw']}")
        lines.append(f"- 当前任务真值：{row['true_name']}")
        lines.append(f"- 当前任务预测：{row['pred_name']}")
        lines.append(f"- 预测类别概率：{row['pred_prob']:.4f}")
        lines.append(f"- 真实类别概率：{row['true_prob']:.4f}")
        lines.append(f"- 置信差值：{row['margin']:.4f}")
        lines.append(f"- 质量初筛：整体 {row['ai_overall_quality']}，清晰度 {row['ai_image_clarity']}，曝光 {row['ai_exposure']}，背景干扰 {row['ai_background_clutter']}，多图一致性 {row['ai_multiview_consistency']}")
        lines.append(f"- 我们的先验判断：{row['ai_issue_hypothesis']}")
        if row["blind_task5"] or row["blind_task6"]:
            lines.append(f"- 纯视觉直觉印象：Task5 倾向 {row['blind_task5'] or '未标'}；Task6 倾向 {row['blind_task6'] or '未标'}")
        if row["revised_task5"] or row["revised_task6"]:
            lines.append(f"- 经验回看修正：Task5 倾向 {row['revised_task5'] or '未标'}；Task6 倾向 {row['revised_task6'] or '未标'}")
        lines.append(f"- 稳定性判断：{row['stability_summary']}")
        lines.append(f"- 我们目前的看法：{build_case_reason(row)}")
        lines.append("- 希望医生重点判断：")
        for q in build_doctor_questions(row):
            lines.append(f"  - {q}")
        for export_name in row["export_images"]:
            rel = f"task56_case_review_assets/images/{export_name}"
            lines.append(f"![{row['case_id']}]({rel})")
        lines.append("")
    CASE_REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")


def build_case_review_pdf(rows: list[dict]) -> None:
    register_font()
    title, h1, h2, body, note, bullet, table_header, table_body, caption = build_styles()
    story = []
    story.append(Paragraph("Task5 与 Task6 典型边界错例医生回看清单", title))
    story.append(Paragraph("日期：2026-05-10", body))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("1. 这份清单是做什么的", h1))
    story.append(
        Paragraph(
            "这里放的是目前最值得优先回看的 16 例错例。每一例前面都先写清我们已经看到的情况：图像质量大致如何，哪些线索支持原标签，哪些地方容易让判断跑偏，以及最想请老师帮忙确认的问题。",
            body,
        )
    )
    story.append(Paragraph("2. 我们已经先做了哪些分析", h1))
    story.append(Paragraph("• 先把 16 例都过了一遍基础质控。", body))
    story.append(Paragraph("• 对进入经验标签核心集的图补了视觉线索、干扰线索和主要边界轴。", body))
    story.append(Paragraph("• 先按“更像真实边界”“边界和质量双因素”“更像拍摄因素”“建议标签或取材复核”分开整理。", body))
    issue_counts = Counter(row["ai_issue_hypothesis"] for row in rows)
    story.append(
        build_table(
            [["当前机器初筛归因", "病例数"]] + [[k, str(v)] for k, v in issue_counts.items()],
            [105 * mm, 45 * mm],
            table_header,
            table_body,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("3. 医生回看时最值得回答的问题", h1))
    story.append(Paragraph("• 这例是不是肉眼上本来就位于 A/AB、B1/B2、B2/B3 或 B2/TC 连续谱边界？", body))
    story.append(Paragraph("• 这例的 gross image 与病理标签是否可能存在取材或对应关系问题？", body))
    story.append(Paragraph("• 这例是否属于医生自己也会犹豫的病例？", body))
    story.append(Paragraph("• 这例如果要再补标签，最值得补的是边界病例，还是建议复核标签？", body))
    story.append(Paragraph("4. 逐例清单", h1))

    for idx, row in enumerate(rows, start=1):
        meta_table = build_table(
            [
                ["字段", "内容"],
                ["任务 / case_id", f"{row['task']} / {row['case_id']}"],
                ["原始病理标签", row["who_type_raw"]],
                ["当前任务真值", row["true_name"]],
                ["当前任务预测", row["pred_name"]],
                ["预测类别概率", f"{row['pred_prob']:.4f}"],
                ["真实类别概率", f"{row['true_prob']:.4f}"],
                ["置信差值", f"{row['margin']:.4f}"],
                ["质量初筛", f"整体{row['ai_overall_quality']}；清晰度{row['ai_image_clarity']}；曝光{row['ai_exposure']}；背景{row['ai_background_clutter']}；多图{row['ai_multiview_consistency']}"],
                ["我们的先验判断", row["ai_issue_hypothesis"]],
                ["经验边界轴", row["boundary_axes"]],
                ["稳定性判断", row["stability_summary"]],
            ],
            [42 * mm, 126 * mm],
            table_header,
            table_body,
        )
        image_paths = [IMAGE_DIR / name for name in row["export_images"] if (IMAGE_DIR / name).exists()]
        question_text = "<br/>".join([f"• {clean(q)}" for q in build_doctor_questions(row)])
        detail_note = (
            f"<b>我们目前的看法：</b>{clean(build_case_reason(row))}<br/><br/>"
            f"<b>请医生重点判断：</b><br/>{question_text}"
        )
        block = [
            Paragraph(f"{idx}. {row['task']} | case_id={row['case_id']}", h2),
            meta_table,
            Paragraph(detail_note, note),
        ]
        if image_paths:
            block.append(build_image_table(image_paths, caption))
        block.append(Spacer(1, 4 * mm))
        story.append(KeepTogether(block))

    doc = SimpleDocTemplate(
        str(CASE_REVIEW_PDF),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Task5 与 Task6 典型边界错例医生回看清单",
        author="Thymic Gross Team",
    )
    doc.build(story)


def export_quality_files() -> None:
    case_df = pd.read_csv(CASE_QUALITY_CSV)
    img_df = pd.read_csv(IMAGE_QUALITY_CSV)
    value_map = {
        "good": "较好",
        "fair": "一般",
        "poor": "较差",
        "single": "单图",
        "consistent": "一致",
        "slightly_inconsistent": "轻度不一致",
        "normal": "正常",
        "bright": "偏亮",
        "dark": "偏暗",
        "none": "无",
        "low": "低",
        "medium": "中",
        "high": "高",
        "large": "大",
        "low_risk": "A/AB",
        "high_risk": "B1-3",
    }
    for col in ["task_l3_label", "case_multiview_consistency", "case_border_touch_level", "case_overall_quality"]:
        case_df[col] = case_df[col].map(lambda x: value_map.get(x, x))
    for col in [
        "task_l3_label",
        "image_clarity",
        "exposure",
        "glare_or_reflection",
        "background_clutter",
        "specimen_ratio",
        "border_touch_level",
        "overall_quality",
    ]:
        img_df[col] = img_df[col].map(lambda x: value_map.get(x, x))
    case_export = case_df[
        [
            "case_id",
            "source_case_folder",
            "who_type_raw",
            "task_l3_label",
            "task_l4_label",
            "image_filenames",
            "case_image_count",
            "case_multiview_consistency",
            "case_border_touch_level",
            "case_quality_score",
            "case_overall_quality",
        ]
    ].rename(
        columns={
            "case_id": "病例ID",
            "source_case_folder": "来源文件夹",
            "who_type_raw": "原始病理标签",
            "task_l3_label": "Task5组别",
            "task_l4_label": "Task6组别",
            "image_filenames": "图像文件",
            "case_image_count": "图像数",
            "case_multiview_consistency": "多图一致性",
            "case_border_touch_level": "贴边程度",
            "case_quality_score": "综合质量分",
            "case_overall_quality": "综合质量等级",
        }
    )
    img_export = img_df[
        [
            "case_id",
            "source_case_folder",
            "who_type_raw",
            "task_l3_label",
            "task_l4_label",
            "image_name",
            "image_clarity",
            "exposure",
            "glare_or_reflection",
            "background_clutter",
            "specimen_ratio",
            "border_touch_level",
            "quality_score",
            "overall_quality",
        ]
    ].rename(
        columns={
            "case_id": "病例ID",
            "source_case_folder": "来源文件夹",
            "who_type_raw": "原始病理标签",
            "task_l3_label": "Task5组别",
            "task_l4_label": "Task6组别",
            "image_name": "图像文件",
            "image_clarity": "清晰度",
            "exposure": "曝光",
            "glare_or_reflection": "反光",
            "background_clutter": "背景干扰",
            "specimen_ratio": "主体占比",
            "border_touch_level": "贴边程度",
            "quality_score": "综合质量分",
            "overall_quality": "综合质量等级",
        }
    )
    case_export.to_csv(QUALITY_CASE_EXPORT, index=False, encoding="utf-8-sig")
    img_export.to_csv(QUALITY_IMAGE_EXPORT, index=False, encoding="utf-8-sig")

    good_case = int((case_df["case_overall_quality"] == "较好").sum())
    fair_case = int((case_df["case_overall_quality"] == "一般").sum())
    poor_case = int((case_df["case_overall_quality"] == "较差").sum())
    good_img = int((img_df["overall_quality"] == "较好").sum())
    fair_img = int((img_df["overall_quality"] == "一般").sum())
    poor_img = int((img_df["overall_quality"] == "较差").sum())

    lines = []
    lines.append("# Task56 全量图像质量初筛说明")
    lines.append("日期：2026-05-10")
    lines.append("")
    lines.append("## 1. 这份说明主要想回答什么")
    lines.append("- 病例级质量标签文件：`Task56全量图像质量初筛标签_病例级_2026-05-10.csv`")
    lines.append("- 图像级质量标签文件：`Task56全量图像质量初筛标签_图像级_2026-05-10.csv`")
    lines.append("")
    lines.append("这一部分主要想看一个问题：现在遇到的困难，主要是不是由拍摄质量造成。这里的标签只反映图像本身，不涉及病理判断。")
    lines.append("")
    lines.append("## 2. 整体结果")
    lines.append(f"- 图像级：较好 = {good_img}，一般 = {fair_img}，较差 = {poor_img}")
    lines.append(f"- 病例级：较好 = {good_case}，一般 = {fair_case}，较差 = {poor_case}")
    lines.append("- 曝光基本正常，明显反光几乎没有。")
    lines.append("- 当前错误并没有明显集中在低质量图上。")
    lines.append("")
    lines.append("## 3. 这说明什么")
    lines.append("目前看，Task5 和 Task6 的主要困难并不是“大面积拍摄质量差”，而是类别本身边界接近，尤其是 TC 与 B 组、A 与 AB、以及 B1/B2/B3/TC 这一带。")
    lines.append("")
    lines.append("## 4. 医生怎么看这份标签")
    lines.append("- 这份标签不用医生逐行重打。")
    lines.append("- 更适合把它当成回看病例时的背景信息：如果一例图质量并不差却还是高置信度错，通常更值得从边界病例或取材对应关系去理解。")
    lines.append("- 后面如果继续扩库，这套字段也可以直接沿用，先筛掉拍摄和构图层面的明显问题。")
    QUALITY_MD.write_text("\n".join(lines), encoding="utf-8")
    render_markdown_pdf(QUALITY_MD, QUALITY_PDF)


def export_experience_files() -> None:
    exp_df = pd.read_csv(EXP_CSV)
    value_map = {
        "good": "较好",
        "fair": "一般",
        "poor": "较差",
        "high": "高",
        "medium": "中",
        "low": "低",
        "stable": "稳定",
        "uncertain": "犹豫",
        "conflicting_clues": "线索冲突",
        "A_AB": "A 与 AB 边界",
        "B1_B2": "B1 与 B2 边界",
        "B2_B3": "B2 与 B3 边界",
        "B2_TC": "B2 与 TC 边界",
        "TC_B_group": "TC 与 B 组边界",
        "none": "无明显单一边界",
        "A_high_conf_wrong": "高置信错例",
        "B_doctor_review": "优先医生复核",
        "C_boundary": "边界病例",
        "D_prototype": "稳定原型",
        "high_conf_wrong_Task5": "Task5高置信错例",
        "high_conf_wrong_Task6": "Task6高置信错例",
        "doctor_review_focus": "医生优先复核",
        "boundary_case_sample": "边界病例样本",
        "prototype_case_sample": "稳定原型样本",
    }
    exp_df["selection_tier"] = exp_df["selection_tier"].map(lambda x: value_map.get(x, x))
    exp_df["selection_reason"] = exp_df["selection_reason"].map(lambda x: value_map.get(x, x))
    exp_df["overall_quality"] = exp_df["overall_quality"].map(lambda x: value_map.get(x, x))
    exp_df["ai_guess_confidence"] = exp_df["ai_guess_confidence"].map(lambda x: value_map.get(x, x))
    exp_df["exp_round2_label_stability"] = exp_df["exp_round2_label_stability"].map(lambda x: value_map.get(x, x))
    exp_df["exp_round1_boundary_axis"] = exp_df["exp_round1_boundary_axis"].map(lambda x: value_map.get(x, x))
    exp_export = exp_df[
        [
            "selection_tier",
            "selection_reason",
            "case_id",
            "task5_label",
            "task6_label",
            "who_type_raw",
            "image_name",
            "overall_quality",
            "ai_blind_guess_task5",
            "ai_blind_guess_task6",
            "ai_guess_confidence",
            "exp_round1_boundary_axis",
            "exp_round2_revised_task5_guess",
            "exp_round2_revised_task6_guess",
            "exp_round2_key_discriminative_clues",
            "exp_round2_confounding_clues",
            "exp_round2_label_stability",
        ]
    ].rename(
        columns={
            "selection_tier": "入选层级",
            "selection_reason": "入选原因",
            "case_id": "病例ID",
            "task5_label": "Task5真值",
            "task6_label": "Task6真值",
            "who_type_raw": "原始病理标签",
            "image_name": "图像文件",
            "overall_quality": "整体质量",
            "ai_blind_guess_task5": "盲看Task5判断",
            "ai_blind_guess_task6": "盲看Task6判断",
            "ai_guess_confidence": "盲看置信度",
            "exp_round1_boundary_axis": "主要边界轴",
            "exp_round2_revised_task5_guess": "回看后Task5判断",
            "exp_round2_revised_task6_guess": "回看后Task6判断",
            "exp_round2_key_discriminative_clues": "主要支持线索",
            "exp_round2_confounding_clues": "主要干扰线索",
            "exp_round2_label_stability": "稳定性",
        }
    )
    exp_export.to_csv(EXP_EXPORT, index=False, encoding="utf-8-sig")

    task5_agree = (exp_df["exp_round2_revised_task5_guess"] == exp_df["task5_label"]).sum()
    task6_agree = (exp_df["exp_round2_revised_task6_guess"] == exp_df["task6_label"]).sum()
    total = len(exp_df)
    stability_counts = exp_df["exp_round2_label_stability"].fillna("未标注").value_counts().to_dict()

    lines = []
    lines.append("# Task56 经验标签核心集说明")
    lines.append("日期：2026-05-10")
    lines.append("")
    lines.append("## 1. 这份材料是什么")
    lines.append("这一部分整理的是我们反复看图后记下来的视觉线索，主要想说明一张图为什么会更像 A/AB、B 组或者 TC。")
    lines.append("")
    lines.append("对应文件：`Task56经验标签核心集_医生版_2026-05-10.csv`")
    lines.append("")
    lines.append("## 2. 当前核心结果")
    lines.append(f"- 核心图像：{total} 张")
    lines.append(f"- Task5 经验回看与真值一致：{task5_agree}/{total} = {task5_agree/total:.4f}")
    lines.append(f"- Task6 经验回看与真值一致：{task6_agree}/{total} = {task6_agree/total:.4f}")
    lines.append("")
    lines.append("稳定性分布：")
    for key, value in stability_counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 3. 这些标签记录了什么")
    lines.append("- 纯看图时最直观的印象")
    lines.append("- 回看后认为更像哪一类")
    lines.append("- 哪些线索支持当前判断")
    lines.append("- 哪些线索会把人或模型带偏")
    lines.append("- 这张图整体是稳定原型，还是典型边界病例")
    lines.append("")
    lines.append("## 4. 医生怎么使用")
    lines.append("- “稳定”：更像相对稳定的原型图，可以帮助确认某一类的典型外观。")
    lines.append("- “犹豫”：更像边界病例，最值得结合病理和取材讨论。")
    lines.append("- “线索冲突”：图上存在相互冲突的线索，最值得重点回看。")
    lines.append("")
    lines.append("## 5. 这套标签现在有什么用")
    lines.append("这套经验标签现在最主要的作用，是先把我们看图时反复遇到的边界线索整理出来。后面如果医生愿意参与，只需要重点看“犹豫”和“线索冲突”这些图，就能把时间更多放在最关键的边界病例上。")
    EXP_MD.write_text("\n".join(lines), encoding="utf-8")
    render_markdown_pdf(EXP_MD, EXP_PDF)


def export_index_file() -> None:
    lines = []
    lines.append("# Task56 医生复核材料目录")
    lines.append("日期：2026-05-10")
    lines.append("")
    lines.append("这套材料建议按下面顺序看。")
    lines.append("")
    lines.append("## 1. 先看典型边界病例")
    lines.append("- `Task56高置信错例医生回看清单_2026-05-07.pdf`")
    lines.append("- 这是最值得优先看的主文件。每一例都已经补了我们的初步判断、为什么会错、以及希望医生重点回答的问题。")
    lines.append("")
    lines.append("## 2. 再看质量初筛")
    lines.append("- `Task56全量图像质量初筛说明_2026-05-10.pdf`")
    lines.append("- `Task56全量图像质量初筛标签_病例级_2026-05-10.csv`")
    lines.append("- `Task56全量图像质量初筛标签_图像级_2026-05-10.csv`")
    lines.append("- 这部分主要用来说明：当前困难并不是大面积图像质量差。")
    lines.append("")
    lines.append("## 3. 最后看经验标签")
    lines.append("- `Task56经验标签核心集说明_2026-05-10.pdf`")
    lines.append("- `Task56经验标签核心集_医生版_2026-05-10.csv`")
    lines.append("- 这部分主要用来说明：我们已经开始把“为什么像某一类”的视觉经验整理成可复核的字段。")
    lines.append("")
    lines.append("## 4. 如果医生时间有限，最值得优先反馈什么")
    lines.append("- 哪些错例从肉眼看本来就属于边界病例")
    lines.append("- 哪些错例可能存在取材与标签不完全对应")
    lines.append("- 哪些错例如果要重新补标签，最值得补“边界病例”或“建议复核标签”")
    INDEX_MD.write_text("\n".join(lines), encoding="utf-8")
    render_markdown_pdf(INDEX_MD, INDEX_PDF)


def main() -> None:
    rows = load_case_review_rows()
    write_case_review_markdown(rows)
    build_case_review_pdf(rows)
    export_quality_files()
    export_experience_files()
    export_index_file()
    print(CASE_REVIEW_MD)
    print(CASE_REVIEW_PDF)
    print(QUALITY_MD)
    print(QUALITY_PDF)
    print(EXP_MD)
    print(EXP_PDF)
    print(INDEX_MD)
    print(INDEX_PDF)


if __name__ == "__main__":
    main()
