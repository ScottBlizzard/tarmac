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
SOURCE_MD = ROOT / "reports" / "ThymicGross" / "2026-05-07_交接后阶段进展总览.md"
OUTPUT_PDF = ROOT / "reports" / "ThymicGross" / "2026-05-07_交接后阶段进展总览.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiLocal"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d"),
        spaceAfter=8 * mm,
    )
    h1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=16,
        leading=21,
        textColor=colors.HexColor("#16324f"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm,
    )
    h2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#244a73"),
        spaceBefore=3.5 * mm,
        spaceAfter=1.5 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.5,
        leading=16,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#222222"),
        spaceAfter=1.5 * mm,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=10 * mm,
        firstLineIndent=-5 * mm,
    )
    note = ParagraphStyle(
        "NoteCN",
        parent=body,
        backColor=colors.HexColor("#f5f7fb"),
        borderColor=colors.HexColor("#d7e1ee"),
        borderWidth=0.6,
        borderPadding=6,
        borderRadius=None,
        spaceBefore=2 * mm,
        spaceAfter=3 * mm,
    )
    return title, h1, h2, body, bullet, note


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
        if line[:3].isdigit() and line[3:5] == ". ":
            flush_paragraph()
            blocks.append(("bullet", line.strip()))
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    return blocks


def build_metric_table() -> Table:
    data = [
        ["阶段", "任务/设置", "关键结果"],
        ["同伴严格复现", "SuRImage Stage1 fine", "Macro-F1 0.3515 / Macro-AUC 0.6538"],
        ["同伴优化后旧主线", "Stage1 + merged crop", "Macro-F1 ≈ 0.456 / Macro-AUC ≈ 0.705"],
        ["我们接手后的新主线", "DINOv2 Task4 whole", "Macro-F1 0.5164 / Macro-AUC 0.7413"],
        ["我们接手后的新主线", "DINOv2 Task3 whole", "AUC 0.7250 / BACC 0.6458"],
    ]
    table = Table(data, colWidths=[32 * mm, 58 * mm, 80 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEADING", (0, 0), (-1, -1), 13),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b9c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def main() -> None:
    register_font()
    title_style, h1_style, h2_style, body_style, bullet_style, note_style = build_styles()
    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    blocks = parse_markdown_lines(lines)

    story = []
    inserted_table = False

    for block_type, text in blocks:
        safe = clean_inline(text)
        if block_type == "title":
            story.append(Paragraph(safe, title_style))
            continue
        if block_type == "h1":
            story.append(Paragraph(safe, h1_style))
            if safe.startswith("5. 当前最重要的实验结果") and not inserted_table:
                story.append(Spacer(1, 2 * mm))
                story.append(build_metric_table())
                story.append(Spacer(1, 3 * mm))
                inserted_table = True
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
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="交接后阶段进展总览",
        author="OpenAI Codex",
    )
    doc.build(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
