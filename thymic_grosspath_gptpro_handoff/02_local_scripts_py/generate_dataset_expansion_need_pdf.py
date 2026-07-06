from __future__ import annotations

import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "汇报" / "胸腺影像分析阶段性总结与扩库建议_2026-05-07.md"
OUTPUT_PDF = ROOT / "汇报" / "胸腺影像分析阶段性总结与扩库建议_2026-05-07.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiDatasetNeedV2"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=20.5,
        leading=26.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d"),
        spaceAfter=6 * mm,
    )
    h1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=14.8,
        leading=20.0,
        textColor=colors.HexColor("#153a5b"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    h2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=12.2,
        leading=16.2,
        textColor=colors.HexColor("#244a73"),
        spaceBefore=3 * mm,
        spaceAfter=1.5 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.2,
        leading=15.4,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#222222"),
        spaceAfter=1.3 * mm,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=9 * mm,
        firstLineIndent=-5 * mm,
    )
    table_header = ParagraphStyle(
        "TableHeaderCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.8,
        leading=11.0,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_body = ParagraphStyle(
        "TableBodyCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.6,
        leading=10.8,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    return title, h1, h2, body, bullet, table_header, table_body


def clean_inline(text: str) -> str:
    return html.escape(text).replace("`", "").replace("**", "")


def parse_markdown_lines(lines: list[str]):
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            blocks.append(("p", " ".join(paragraph).strip()))
            paragraph = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_paragraph()
            continue
        if line.startswith("# "):
            flush_paragraph()
            blocks.append(("title", line[2:].strip()))
            continue
        if line.startswith("## "):
            flush_paragraph()
            blocks.append(("h1", line[3:].strip()))
            continue
        if line.startswith("### "):
            flush_paragraph()
            blocks.append(("h2", line[4:].strip()))
            continue
        if line.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", line[2:].strip()))
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    return blocks


def build_table(data: list[list[str]], col_widths: list[float], header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    wrapped = []
    for row_idx, row in enumerate(data):
        style = header_style if row_idx == 0 else body_style
        wrapped.append([Paragraph(clean_inline(cell), style) for cell in row])
    table = Table(wrapped, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#b9c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ]
        )
    )
    return table


def build_result_table(header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    data = [
        ["任务", "当前较优配置", "关键结果"],
        ["Task1", "DINOv2 vitl14 + whole", "AUC 0.8707；若更看重平衡性，vits14 + whole 的 BACC=0.7607 更稳"],
        ["Task2", "DINOv2 vits14 + whole", "AUC 0.8638；BACC 0.7875"],
        ["Task3", "DINOv2 vitl14 + whole + fold-wise threshold", "AUC 0.7621；BACC 0.7167"],
        ["Task4", "DINOv2 vits14 + vitb14", "F1 最好：Macro-F1 0.5236；Macro-AUC 0.7533"],
        ["Task4", "DINOv2 vits14 + vitb14 + PLIP", "AUC 最好：Macro-F1 0.5226；Macro-AUC 0.7656"],
        ["Task5", "DINOv2 vitl14 + whole", "F1 最好：Macro-F1 0.6614；Macro-AUC 0.7959"],
        ["Task5", "DINOv2 vitl14 + vitb14 特征拼接", "AUC 最好：Macro-F1 0.6573；Macro-AUC 0.8229"],
        ["Task6", "DINOv2 (vits14 + vitb14) 拼接 + Task5 层级先验", "Macro-F1 0.5127；Macro-AUC 0.8095"],
    ]
    return build_table(data, [20 * mm, 61 * mm, 89 * mm], header_style, body_style)


def build_method_table(header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    data = [
        ["方法方向", "当前判断", "原因"],
        ["DINOv2 whole 强表征", "保留并作为主线", "这是最明确、最稳定的提升来源"],
        ["双 backbone 特征拼接", "保留", "对 Task5 AUC、Task6 F1 都带来实质提升"],
        ["少量概率融合", "只在特定任务保留", "Task4 有效，但 Task6 的简单概率融合无效"],
        ["Task5 作为 Task6 层级先验", "保留", "能减少跨粗分组的大错，并把 Task6 F1 推到 0.5127"],
        ["PLIP 单模", "降级为对照", "能跑通，但明显弱于 DINOv2"],
        ["crop / whole+crop", "暂不继续", "没有稳定打过当前 best whole 主线"],
        ["SuRImage Fusion", "停止扩展", "在当前数据规模下不稳定，整体不如强表征主线"],
        ["CONDOR / ordinal loss / 轻量微调", "停止扩展", "当前都没有形成可持续优势"],
    ]
    return build_table(data, [34 * mm, 25 * mm, 109 * mm], header_style, body_style)


def build_metric_table(header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    data = [
        ["指标", "通俗理解", "当前项目里怎样算可观"],
        ["AUC", "模型总体排序能力，不依赖固定阈值", "二分类里 0.75-0.8 说明有明显信号，0.8+ 已经比较强"],
        ["Macro-F1", "多分类里每个类别单独算，再平均", "Task5 三分类到 0.60 左右已比较像样，0.65+ 已较好；Task6 六分类到 0.50+ 已有研究价值"],
        ["Balanced Accuracy", "各类召回率的平均值", "适合看模型是否只偏向多数类；Task3 的 0.7167 说明风险分层已较稳"],
    ]
    return build_table(data, [30 * mm, 62 * mm, 78 * mm], header_style, body_style)


def build_error_table(header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    data = [
        ["任务", "当前主要短板", "最值得医生关注的边界"],
        ["Task5", "TC recall 只有 0.50，主要与 B1-3 互相混淆", "TC vs B1-3"],
        ["Task6", "A recall 0.30，B2 recall 0.35，是最难两类", "A vs AB；B1 vs B2；B2 vs B3；B2 vs TC"],
        ["Task6 增强版", "总错例从 61 降到 58，跨粗分组大错从 39 降到 34", "说明层级先验在帮忙，但边界病例仍不足"],
    ]
    return build_table(data, [24 * mm, 69 * mm, 77 * mm], header_style, body_style)


def main() -> None:
    register_font()
    title_style, h1_style, h2_style, body_style, bullet_style, table_header_style, table_body_style = build_styles()
    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    blocks = parse_markdown_lines(lines)

    story = []
    for block_type, text in blocks:
        safe = clean_inline(text)
        if block_type == "title":
            story.append(Paragraph(safe, title_style))
            continue
        if block_type == "h1":
            story.append(Paragraph(safe, h1_style))
            if safe.startswith("3. 当前任务体系和目前最好结果"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_result_table(table_header_style, table_body_style))
                story.append(Spacer(1, 3 * mm))
            if safe.startswith("4. 我们试了哪些方法"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_method_table(table_header_style, table_body_style))
                story.append(Spacer(1, 3 * mm))
            if safe.startswith("6. 怎么看这些数字"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_metric_table(table_header_style, table_body_style))
                story.append(Spacer(1, 3 * mm))
            if safe.startswith("8. 当前错误分析说明了什么"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_error_table(table_header_style, table_body_style))
                story.append(Spacer(1, 3 * mm))
            continue
        if block_type == "h2":
            story.append(Paragraph(safe, h2_style))
            continue
        if block_type == "bullet":
            story.append(Paragraph(f"• {safe}", bullet_style))
            continue
        story.append(Paragraph(safe, body_style))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="胸腺影像分析阶段性总结与扩库建议",
        author="OpenAI Codex",
    )
    doc.build(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
