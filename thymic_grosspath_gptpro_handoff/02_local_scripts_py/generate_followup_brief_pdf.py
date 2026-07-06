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
SOURCE_MD = ROOT / "汇报" / "胸腺影像分析后续进展简报_2026-05-07.md"
OUTPUT_PDF = ROOT / "汇报" / "胸腺影像分析后续进展简报_2026-05-07.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiFollowup"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=20,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d"),
        spaceAfter=6 * mm,
    )
    h1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=14.8,
        leading=20,
        textColor=colors.HexColor("#153a5b"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    h2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=12.2,
        leading=16.5,
        textColor=colors.HexColor("#244a73"),
        spaceBefore=3 * mm,
        spaceAfter=1.5 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.2,
        leading=15.3,
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
    return title, h1, h2, body, bullet, table_header, table_body


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
        ["Task1", "DINOv2 vitl14 + whole", "AUC 0.8707；若更看重平衡性，则 vits14 + whole 的 BACC=0.7607 更稳"],
        ["Task2", "DINOv2 vits14 + whole", "AUC 0.8638，BACC 0.7875"],
        ["Task3", "DINOv2 vitl14 + whole + fold-wise threshold", "AUC 0.7621，BACC 0.7167"],
        ["Task4（最好 F1）", "DINOv2 vits14 + vitb14 融合", "Macro-F1 0.5236，Macro-AUC 0.7533"],
        ["Task4（最好 AUC）", "DINOv2 vits14 + vitb14 + PLIP 融合", "Macro-F1 0.5226，Macro-AUC 0.7656"],
    ]
    return build_table(data, [26 * mm, 58 * mm, 86 * mm], header_style, body_style)


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
            if safe.startswith("3. 新增关键结果"):
                story.append(Spacer(1, 1.5 * mm))
                story.append(build_result_table(table_header_style, table_body_style))
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
        title="胸腺影像分析后续进展简报",
        author="OpenAI Codex",
    )
    doc.build(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
