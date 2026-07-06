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
SOURCE_MD = ROOT / "汇报" / "2026-05-18_Task7阶段修补与当前结果汇报.md"
OUTPUT_PDF = ROOT / "汇报" / "2026-05-18_Task7阶段修补与当前结果汇报.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7StageReport"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def read_source_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName=FONT_NAME,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2d3d"),
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            parent=styles["Heading1"],
            fontName=FONT_NAME,
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=2.2 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.2,
            leading=13.2,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            spaceAfter=1.4 * mm,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.0,
            leading=13.0,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            leftIndent=7.2 * mm,
            firstLineIndent=-4.2 * mm,
            spaceAfter=1.1 * mm,
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.8,
            leading=9.5,
            wordWrap="CJK",
            textColor=colors.HexColor("#16324f"),
        ),
        "table_body": ParagraphStyle(
            "TableBodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.6,
            leading=9.3,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
        ),
    }


def clean_inline(text: str) -> str:
    return html.escape(text.replace("`", "").replace("**", ""))


def parse_markdown(md_text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = md_text.splitlines()
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
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
        if line.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", line[2:].strip()))
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush_paragraph()
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw = lines[i].strip()
                stripped = raw.replace("|", "").replace("-", "").replace(":", "").strip()
                if not stripped:
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


def get_col_widths(ncols: int, page_width: float) -> list[float]:
    usable = page_width - 24 * mm
    presets = {
        2: [0.4, 0.6],
        3: [0.22, 0.18, 0.60],
        4: [0.30, 0.175, 0.175, 0.35],
        5: [0.23, 0.15, 0.15, 0.15, 0.32],
        6: [0.22, 0.13, 0.13, 0.13, 0.13, 0.26],
    }
    ratios = presets.get(ncols, [1 / ncols] * ncols)
    return [usable * ratio for ratio in ratios]


def build_table(rows: list[list[str]], page_width: float, styles: dict[str, ParagraphStyle]) -> Table:
    ncols = max(len(row) for row in rows)
    normalized = [row + [""] * (ncols - len(row)) for row in rows]
    wrapped = []
    for idx, row in enumerate(normalized):
        style = styles["table_head"] if idx == 0 else styles["table_body"]
        wrapped.append([Paragraph(clean_inline(cell), style) for cell in row])
    table = Table(wrapped, colWidths=get_col_widths(ncols, page_width), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                ("TOPPADDING", (0, 0), (-1, -1), 2.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
            ]
        )
    )
    return table


def main() -> None:
    register_font()
    styles = build_styles()
    blocks = parse_markdown(read_source_text(SOURCE_MD))
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    story = []
    for kind, payload in blocks:
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
            story.append(Spacer(1, 2.2 * mm))

    doc.build(story)
    print(f"Generated PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
