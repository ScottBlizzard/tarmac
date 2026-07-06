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
SOURCE_MD = ROOT / "reports" / "ThymicGross" / "2026-05-07_导师汇报_阶段进展与后续计划.md"
OUTPUT_PDF = ROOT / "reports" / "ThymicGross" / "2026-05-07_导师汇报_阶段进展与后续计划.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiAdvisor"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=21,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d"),
        spaceAfter=7 * mm,
    )
    h1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=15.5,
        leading=21,
        textColor=colors.HexColor("#153a5b"),
        spaceBefore=4.5 * mm,
        spaceAfter=2 * mm,
    )
    h2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=12.5,
        leading=17,
        textColor=colors.HexColor("#244a73"),
        spaceBefore=3 * mm,
        spaceAfter=1.5 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.3,
        leading=15.5,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#222222"),
        spaceAfter=1.4 * mm,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=9 * mm,
        firstLineIndent=-5 * mm,
    )
    note = ParagraphStyle(
        "NoteCN",
        parent=body,
        backColor=colors.HexColor("#f5f7fb"),
        borderColor=colors.HexColor("#d7e1ee"),
        borderWidth=0.6,
        borderPadding=6,
        spaceBefore=2 * mm,
        spaceAfter=3 * mm,
    )
    table_header = ParagraphStyle(
        "TableHeaderCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.9,
        leading=11.2,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_body = ParagraphStyle(
        "TableBodyCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.7,
        leading=11.0,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    return title, h1, h2, body, bullet, note, table_header, table_body


def clean_inline(text: str) -> str:
    return html.escape(text).replace("`", "")


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
        if line.startswith("> "):
            flush_paragraph()
            blocks.append(("note", line[2:].strip()))
            continue
        if line.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", line[2:].strip()))
            continue
        if line[:2].isdigit() and line[2:4] == ". ":
            flush_paragraph()
            blocks.append(("bullet", line.strip()))
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    return blocks


def build_table(
    data: list[list[str]],
    col_widths: list[float],
    header_style: ParagraphStyle,
    body_style: ParagraphStyle,
) -> Table:
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


def build_plan_table(header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    data = [
        ["上周计划事项", "状态", "当前说明"],
        ["固定 160 例数据集、4 个主任务与 5-fold patient-level 评估协议", "√", "已完成，并作为后续全部实验统一口径"],
        ["依次跑通任务 1-3 的主要实验并输出 AUC/ACC/BACC/Sens/Spec", "√", "核心任务已完成；Task3 已完成 whole/crop/dual 对照，Task1/2 已得到可汇报结果"],
        ["进行阈值分析，重点改善 specificity", "√", "已完成阈值扫描与 fold-wise 阈值分析，明确其作用边界"],
        ["评估类别不平衡采样或 loss 改造对 balanced accuracy 的影响", "√", "已完成 label smoothing、weighted/loss 改造、checkpoint sweep 等验证"],
        ["在原计划基础上补充更强模型探索", "√", "已额外完成 SuRImage 严格复现、旧主线优化和 DINOv2 强表征验证"],
    ]
    return build_table(data, [50 * mm, 12 * mm, 103 * mm], header_style, body_style)


def build_repro_table(header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    data = [
        ["阶段", "任务/设置", "关键结果"],
        ["SuRImage 严格复现", "Stage1 coarse（低危 vs 高危）", "AUC 0.7571，BACC 0.6875"],
        ["SuRImage 严格复现", "Stage1 fine（WHO 五分类）", "Macro-F1 0.3515，Macro-AUC 0.6538"],
        ["SuRImage 严格复现", "Fusion fine（WHO 五分类）", "Macro-F1 0.2396，性能未优于 Stage1"],
        ["旧主线优化后", "Stage1 + merged crop", "Macro-F1 约 0.4523-0.456，Macro-AUC 约 0.705-0.720"],
    ]
    return build_table(data, [32 * mm, 58 * mm, 75 * mm], header_style, body_style)


def build_current_result_table(header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    data = [
        ["当前主结果", "设置", "关键数字"],
        ["Task1：良性增生 vs TET", "DINOv2 vits14 + whole", "AUC 0.8161，BACC 0.7607"],
        ["Task2：胸腺瘤 vs 胸腺癌", "DINOv2 vits14 + whole", "AUC 0.8638，BACC 0.7875"],
        ["Task3：低危 vs 高危", "DINOv2 vits14 + whole", "AUC 0.7250，BACC 0.6458"],
        ["Task4：WHO 五分类", "DINOv2 vits14 + whole", "Macro-F1 0.5164，Macro-AUC 0.7413"],
        ["Task4：旧主线稳定结果", "SuRImage 优化线", "Macro-F1 约 0.4523，Macro-AUC 约 0.7202"],
        ["Task4：PLIP 快速对照", "PLIP frozen probe + whole", "Macro-F1 0.4050，Macro-AUC 0.7094"],
    ]
    return build_table(data, [39 * mm, 56 * mm, 70 * mm], header_style, body_style)


def main() -> None:
    register_font()
    (
        title_style,
        h1_style,
        h2_style,
        body_style,
        bullet_style,
        note_style,
        table_header_style,
        table_body_style,
    ) = build_styles()
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
            if safe.startswith("2. 上周计划完成情况"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_plan_table(table_header_style, table_body_style))
                story.append(Spacer(1, 3 * mm))
            if safe.startswith("3. 同伴阶段复现结果概览"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_repro_table(table_header_style, table_body_style))
                story.append(Spacer(1, 3 * mm))
            if safe.startswith("5. 当前关键结果概览"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_current_result_table(table_header_style, table_body_style))
                story.append(Spacer(1, 3 * mm))
            continue
        if block_type == "h2":
            story.append(Paragraph(safe, h2_style))
            continue
        if block_type == "bullet":
            story.append(Paragraph(f"• {safe}", bullet_style))
            continue
        if block_type == "note":
            story.append(Paragraph(safe, note_style))
            continue
        story.append(Paragraph(safe, body_style))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="胸腺影像分析阶段进展与后续计划",
        author="OpenAI Codex",
    )
    doc.build(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
