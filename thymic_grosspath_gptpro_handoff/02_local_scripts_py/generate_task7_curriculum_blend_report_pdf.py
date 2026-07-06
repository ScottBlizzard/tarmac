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
SOURCE_MD = ROOT / "汇报" / "2026-05-18_Task7方案B课程学习融合阶段结果.md"
OUTPUT_PDF = ROOT / "汇报" / "2026-05-18_Task7方案B课程学习融合阶段结果.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7CurriculumBlend"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def clean_inline(text: str) -> str:
    return html.escape(text.replace("`", "").replace("**", ""))


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName=FONT_NAME,
            fontSize=17,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#16324f"),
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "HeadingCN",
            parent=styles["Heading1"],
            fontName=FONT_NAME,
            fontSize=12.2,
            leading=16,
            textColor=colors.HexColor("#1b3a57"),
            spaceBefore=2.4 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.2,
            leading=13.4,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            spaceAfter=1.5 * mm,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.0,
            leading=13.2,
            leftIndent=7.5 * mm,
            firstLineIndent=-4.5 * mm,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            spaceAfter=1.0 * mm,
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.2,
            leading=9.0,
            wordWrap="CJK",
            textColor=colors.HexColor("#16324f"),
        ),
        "table_body": ParagraphStyle(
            "TableBodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.1,
            leading=8.8,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
        ),
    }


def parse_markdown(text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    paragraph: list[str] = []
    idx = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(("p", " ".join(paragraph).strip()))
            paragraph = []

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            idx += 1
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            blocks.append(("title", stripped[2:].strip()))
            idx += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append(("h1", stripped[3:].strip()))
            idx += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", stripped[2:].strip()))
            idx += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            rows: list[list[str]] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                raw = lines[idx].strip()
                marker = raw.replace("|", "").replace("-", "").replace(":", "").strip()
                if marker:
                    rows.append([cell.strip() for cell in raw.strip("|").split("|")])
                idx += 1
            blocks.append(("table", rows))
            continue
        paragraph.append(stripped)
        idx += 1
    flush_paragraph()
    return blocks


def col_widths(ncols: int, page_width: float) -> list[float]:
    usable = page_width - 4 * mm
    presets = {
        4: [0.34, 0.22, 0.22, 0.22],
        5: [0.32, 0.17, 0.17, 0.17, 0.17],
        6: [0.25, 0.12, 0.14, 0.16, 0.13, 0.20],
    }
    ratios = presets.get(ncols, [1.0 / ncols] * ncols)
    return [usable * ratio for ratio in ratios]


def build_table(rows: list[list[str]], width: float, styles: dict[str, ParagraphStyle]) -> Table:
    ncols = max(len(row) for row in rows)
    normalized = [row + [""] * (ncols - len(row)) for row in rows]
    wrapped = []
    for row_idx, row in enumerate(normalized):
        style = styles["table_head"] if row_idx == 0 else styles["table_body"]
        wrapped.append([Paragraph(clean_inline(cell), style) for cell in row])
    table = Table(wrapped, colWidths=col_widths(ncols, width), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f2")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.7),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def main() -> None:
    register_font()
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    story = []
    for kind, payload in parse_markdown(SOURCE_MD.read_text(encoding="utf-8")):
        if kind == "title":
            story.append(Paragraph(clean_inline(str(payload)), styles["title"]))
            story.append(Spacer(1, 1.5 * mm))
        elif kind == "h1":
            story.append(Paragraph(clean_inline(str(payload)), styles["h1"]))
        elif kind == "p":
            story.append(Paragraph(clean_inline(str(payload)), styles["body"]))
        elif kind == "bullet":
            story.append(Paragraph("• " + clean_inline(str(payload)), styles["bullet"]))
        elif kind == "table":
            story.append(build_table(payload, doc.width, styles))
            story.append(Spacer(1, 2.0 * mm))
    doc.build(story)
    print(f"Generated PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
