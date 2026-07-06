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
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def find_report_dir() -> Path:
    for child in Path.cwd().iterdir():
        if child.is_dir() and child.name == "\u6c47\u62a5":
            return child
    raise FileNotFoundError("report directory not found")


def register_fonts() -> tuple[str, str]:
    normal_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    bold_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    pdfmetrics.registerFont(TTFont("CN", str(normal_path)))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(bold_path)))
    return "CN", "CN-Bold"


def make_styles(font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
            textColor=colors.HexColor("#17324D"),
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=13.5,
            leading=17,
            spaceBefore=4 * mm,
            spaceAfter=1.6 * mm,
            textColor=colors.HexColor("#17324D"),
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName=bold_font,
            fontSize=11.5,
            leading=14.5,
            spaceBefore=2.4 * mm,
            spaceAfter=1 * mm,
            textColor=colors.HexColor("#2F4F68"),
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=font,
            fontSize=10,
            leading=15.1,
            firstLineIndent=0,
            spaceAfter=1.8 * mm,
            alignment=TA_LEFT,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.2,
            leading=13,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=5 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.8,
            leading=14.3,
            leftIndent=6 * mm,
            firstLineIndent=-3.5 * mm,
            spaceAfter=1 * mm,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.2,
            leading=9.2,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=base["BodyText"],
            fontName=bold_font,
            fontSize=7.3,
            leading=9.5,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(text).replace("  ", "&nbsp;&nbsp;"), style)


def parse_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[str]] = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c.replace(":", "").strip()) <= {"-"} for c in cells):
            continue
        rows.append(cells)

    data = []
    for r_idx, row in enumerate(rows):
        style = styles["table_header"] if r_idx == 0 else styles["table"]
        data.append([Paragraph(html.escape(cell), style) for cell in row])

    if rows and len(rows[0]) == 9:
        col_widths = [80, 36, 44, 54, 36, 36, 48, 54, 168]
    else:
        usable = landscape(A4)[0] - 36 * mm
        col_widths = [usable / len(rows[0])] * len(rows[0])

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23445F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D3DA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_story(markdown_text: str, styles: dict[str, ParagraphStyle]):
    story = []
    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue

        if line.startswith("# "):
            story.append(para(line[2:].strip(), styles["title"]))
            i += 1
            continue

        if line.startswith("## "):
            story.append(para(line[3:].strip(), styles["h2"]))
            i += 1
            continue

        if line.startswith("### "):
            story.append(para(line[4:].strip(), styles["h3"]))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(Spacer(1, 1.5 * mm))
            story.append(parse_table(table_lines, styles))
            story.append(Spacer(1, 3 * mm))
            continue

        if line.startswith("- "):
            while i < len(lines) and lines[i].startswith("- "):
                item = lines[i][2:].strip()
                story.append(para(f"- {item}", styles["bullet"]))
                i += 1
            story.append(Spacer(1, 1 * mm))
            continue

        block = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt or nxt.startswith("#") or nxt.startswith("- ") or nxt.startswith("|"):
                break
            block.append(nxt)
            i += 1
        text = " ".join(part.strip() for part in block)
        style = styles["meta"] if text.startswith("日期：") else styles["body"]
        story.append(para(text, style))

    return story


def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawRightString(
        landscape(A4)[0] - 18 * mm,
        9 * mm,
        f"第 {doc.page} 页",
    )
    canvas.restoreState()


def main() -> None:
    report_dir = find_report_dir()
    md_path = next(report_dir.glob("2026-05-25_Task7*.md"))
    pdf_path = md_path.with_suffix(".pdf")
    font, bold_font = register_fonts()
    styles = make_styles(font, bold_font)

    text = md_path.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=14 * mm,
        title="Task7 第三批数据引入后的后续尝试与阶段结论",
        author="thymic gross image analysis team",
    )
    doc.build(build_story(text, styles), onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(pdf_path)


if __name__ == "__main__":
    main()
