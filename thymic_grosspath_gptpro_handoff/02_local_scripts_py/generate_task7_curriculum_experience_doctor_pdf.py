from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "汇报" / "2026-05-18_Task7课程学习与经验标签阶段汇报_医生版.md"
OUTPUT_PDF = ROOT / "汇报" / "2026-05-18_Task7课程学习与经验标签阶段汇报_医生版.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7CurriculumExperience"


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
            textColor=colors.HexColor("#17324d"),
            spaceAfter=3.0 * mm,
        ),
        "h1": ParagraphStyle(
            "HeadingCN",
            parent=styles["Heading1"],
            fontName=FONT_NAME,
            fontSize=12.2,
            leading=16,
            textColor=colors.HexColor("#183b59"),
            spaceBefore=2.6 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.0,
            leading=13.2,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            spaceAfter=1.35 * mm,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.8,
            leading=12.8,
            leftIndent=7.0 * mm,
            firstLineIndent=-4.2 * mm,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            spaceAfter=1.0 * mm,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.0,
            leading=10.0,
            alignment=TA_CENTER,
            wordWrap="CJK",
            textColor=colors.HexColor("#4a5568"),
            spaceBefore=1.0 * mm,
            spaceAfter=2.0 * mm,
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.7,
            leading=8.1,
            wordWrap="CJK",
            textColor=colors.HexColor("#17324d"),
        ),
        "table_body": ParagraphStyle(
            "TableBodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.6,
            leading=8.0,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
        ),
    }


def parse_markdown(text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    paragraph: list[str] = []
    idx = 0
    image_re = re.compile(r"!\[(?P<caption>.*?)\]\((?P<path>.*?)\)")

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
        image_match = image_re.fullmatch(stripped)
        if image_match:
            flush_paragraph()
            blocks.append(("image", (image_match.group("caption"), image_match.group("path"))))
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
            if rows:
                blocks.append(("table", rows))
            continue
        paragraph.append(stripped)
        idx += 1
    flush_paragraph()
    return blocks


def col_widths(ncols: int, page_width: float) -> list[float]:
    usable = page_width - 4 * mm
    presets = {
        2: [0.30, 0.70],
        3: [0.28, 0.20, 0.52],
        4: [0.23, 0.14, 0.20, 0.43],
        5: [0.30, 0.15, 0.15, 0.15, 0.25],
        6: [0.25, 0.12, 0.13, 0.15, 0.13, 0.22],
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
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("GRID", (0, 0), (-1, -1), 0.32, colors.HexColor("#b6c7d6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ]
        )
    )
    return table


def resolve_image(path_text: str) -> Path:
    raw = path_text.strip()
    path = Path(raw)
    if path.exists():
        return path
    candidate = ROOT / raw
    return candidate


def build_image(path: Path, width: float) -> Image:
    image = Image(str(path))
    max_width = width
    max_height = 95 * mm
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    image.hAlign = "CENTER"
    return image


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawCentredString(A4[0] / 2, 7 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def main() -> None:
    register_font()
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
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
            story.append(Spacer(1, 1.9 * mm))
        elif kind == "image":
            caption, path_text = payload
            path = resolve_image(str(path_text))
            if path.exists():
                story.append(build_image(path, doc.width))
                story.append(Paragraph(clean_inline(str(caption)), styles["caption"]))
            else:
                story.append(Paragraph(f"图像文件未找到：{clean_inline(str(path_text))}", styles["body"]))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Generated PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
