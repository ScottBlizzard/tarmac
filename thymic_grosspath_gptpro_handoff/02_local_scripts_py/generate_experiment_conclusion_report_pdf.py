from __future__ import annotations

import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "汇报" / "胸腺影像分析实验结果与阶段结论_2026-05-09.md"
OUTPUT_PDF = ROOT / "汇报" / "胸腺影像分析实验结果与阶段结论_2026-05-09.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiExpConclusion"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=18,
        leading=23,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d"),
        spaceAfter=3 * mm,
    )
    h1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#16324f"),
        spaceBefore=2.5 * mm,
        spaceAfter=1.5 * mm,
    )
    h2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=10.8,
        leading=14,
        textColor=colors.HexColor("#244a73"),
        spaceBefore=2 * mm,
        spaceAfter=1.2 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9.0,
        leading=13.2,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#222222"),
        spaceAfter=1.6 * mm,
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
        fontSize=7.8,
        leading=9.5,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_body = ParagraphStyle(
        "TableBodyCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=7.6,
        leading=9.3,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    return title, h1, h2, body, bullet, table_header, table_body


def clean_inline(text: str) -> str:
    text = text.replace("`", "")
    return html.escape(text)


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
                parts = [cell.strip() for cell in raw.strip("|").split("|")]
                rows.append(parts)
                i += 1
            blocks.append(("table", rows))
            continue
        paragraph.append(line.strip())
        i += 1

    flush_paragraph()
    return blocks


def table_widths(ncols: int, page_width: float) -> list[float]:
    inner = page_width - 20 * mm
    if ncols == 4:
        ratios = [0.18, 0.23, 0.24, 0.35]
    elif ncols == 5:
        ratios = [0.17, 0.28, 0.12, 0.12, 0.31]
    elif ncols == 6:
        ratios = [0.11, 0.19, 0.22, 0.10, 0.12, 0.26]
    else:
        ratios = [1 / ncols] * ncols
    return [inner * r for r in ratios]


def build_table(rows: list[list[str]], page_width: float, header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    ncols = max(len(r) for r in rows)
    normalized = [r + [""] * (ncols - len(r)) for r in rows]
    wrapped = []
    for idx, row in enumerate(normalized):
        style = header_style if idx == 0 else body_style
        wrapped.append([Paragraph(clean_inline(cell), style) for cell in row])
    table = Table(wrapped, colWidths=table_widths(ncols, page_width), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
            ]
        )
    )
    return table


def main() -> None:
    register_font()
    title_style, h1_style, h2_style, body_style, bullet_style, table_header_style, table_body_style = build_styles()
    blocks = parse_markdown(SOURCE_MD.read_text(encoding="utf-8"))

    page_width, _ = landscape(A4)
    story = []
    for kind, payload in blocks:
        if kind == "title":
            story.append(Paragraph(clean_inline(str(payload)), title_style))
        elif kind == "h1":
            story.append(Paragraph(clean_inline(str(payload)), h1_style))
        elif kind == "h2":
            story.append(Paragraph(clean_inline(str(payload)), h2_style))
        elif kind == "p":
            story.append(Paragraph(clean_inline(str(payload)), body_style))
        elif kind == "bullet":
            story.append(Paragraph(f"• {clean_inline(str(payload))}", bullet_style))
        elif kind == "table":
            story.append(build_table(payload, page_width, table_header_style, table_body_style))
            story.append(Spacer(1, 2.2 * mm))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title="胸腺影像分析实验结果与阶段结论",
        author="OpenAI Codex",
    )
    doc.build(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
