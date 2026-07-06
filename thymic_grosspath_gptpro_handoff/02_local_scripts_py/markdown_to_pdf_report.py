from __future__ import annotations

import argparse
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
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


FONT_PATHS = [
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]
FONT_NAME = "TaskMdPdfChinese"


def register_font() -> None:
    for path in FONT_PATHS:
        if path.exists():
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
            return
    raise FileNotFoundError("No Chinese font found in C:\\Windows\\Fonts")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT_NAME,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#12395b"),
            spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_NAME,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=4 * mm,
            spaceAfter=1.5 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_NAME,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=3 * mm,
            spaceAfter=1.2 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName=FONT_NAME,
            fontSize=9.8,
            leading=13,
            textColor=colors.HexColor("#245b3a"),
            spaceBefore=2.2 * mm,
            spaceAfter=0.8 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.6,
            leading=12.4,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=1.0 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.3,
            leading=12.0,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            leftIndent=2 * mm,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName=FONT_NAME,
            fontSize=7.3,
            leading=9.5,
            wordWrap="CJK",
            backColor=colors.HexColor("#f4f6f8"),
            borderColor=colors.HexColor("#d8dee9"),
            borderWidth=0.35,
            borderPadding=4,
            textColor=colors.HexColor("#111827"),
            spaceBefore=1 * mm,
            spaceAfter=1.5 * mm,
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.3,
            leading=12.0,
            wordWrap="CJK",
            leftIndent=3 * mm,
            rightIndent=3 * mm,
            borderColor=colors.HexColor("#d2d6dc"),
            borderWidth=0.35,
            borderPadding=4,
            backColor=colors.HexColor("#f6f8fb"),
            textColor=colors.HexColor("#1f2933"),
            spaceBefore=1.0 * mm,
            spaceAfter=1.0 * mm,
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.5,
            leading=8.2,
            alignment=TA_CENTER,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "td": ParagraphStyle(
            "td",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.35,
            leading=8.2,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
    }


def esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def inline_md(text: str) -> str:
    text = esc(text)
    text = re.sub(r"`([^`]+)`", r'<font face="' + FONT_NAME + r'" color="#1f4e79">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(inline_md(text), style)


def split_table_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def is_table_sep(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells)


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]


def table_flowable(lines: list[str], st: dict[str, ParagraphStyle]) -> Table | None:
    rows = [split_table_row(x) for x in lines if not is_table_sep(x)]
    if not rows:
        return None
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    usable = 185.0
    if max_cols == 1:
        widths = [usable]
    elif max_cols == 2:
        widths = [34.0, usable - 34.0]
    elif max_cols == 3:
        widths = [35.0, 55.0, usable - 90.0]
    elif max_cols == 4:
        widths = [32.0, 50.0, 50.0, usable - 132.0]
    elif max_cols == 5:
        widths = [26.0, 28.0, 50.0, 43.0, usable - 147.0]
    else:
        first = 25.0
        widths = [first] + [(usable - first) / (max_cols - 1)] * (max_cols - 1)

    data = []
    for i, row in enumerate(rows):
        style = st["th"] if i == 0 else st["td"]
        data.append([Paragraph(inline_md(cell), style) for cell in row])
    tbl = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c2cf")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fb")]),
            ]
        )
    )
    return tbl


def flush_paragraph(buffer: list[str], story: list, st: dict[str, ParagraphStyle]) -> None:
    if not buffer:
        return
    text = " ".join(x.strip() for x in buffer if x.strip())
    if text:
        story.append(para(text, st["body"]))
    buffer.clear()


def footer(title: str):
    def _footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(FONT_NAME, 7)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        short = title[:48]
        canvas.drawString(13 * mm, 8 * mm, short)
        canvas.drawRightString(198 * mm, 8 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    return _footer


def build_story(markdown: str, title: str) -> list:
    st = styles()
    story: list = []
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paragraph_buffer: list[str] = []
    bullet_buffer: list[Paragraph] = []
    code_buffer: list[str] = []
    in_code = False
    i = 0
    title_used = False

    def flush_bullets() -> None:
        nonlocal bullet_buffer
        if bullet_buffer:
            story.extend(bullet_buffer)
            story.append(Spacer(1, 0.6 * mm))
            bullet_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(paragraph_buffer, story, st)
            flush_bullets()
            if in_code:
                story.append(Paragraph(esc("\n".join(code_buffer)).replace("\n", "<br/>"), st["code"]))
                code_buffer = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buffer.append(line)
            i += 1
            continue

        if not stripped:
            flush_paragraph(paragraph_buffer, story, st)
            flush_bullets()
            i += 1
            continue

        if re.fullmatch(r"-{3,}|_{3,}|\*{3,}", stripped):
            flush_paragraph(paragraph_buffer, story, st)
            flush_bullets()
            story.append(HRFlowable(width="100%", thickness=0.45, color=colors.HexColor("#d2d6dc"), spaceBefore=1.5 * mm, spaceAfter=1.5 * mm))
            i += 1
            continue

        if is_table_line(line):
            flush_paragraph(paragraph_buffer, story, st)
            flush_bullets()
            table_lines: list[str] = []
            while i < len(lines) and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            tbl = table_flowable(table_lines, st)
            if tbl is not None:
                story.append(tbl)
                story.append(Spacer(1, 1.4 * mm))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph(paragraph_buffer, story, st)
            flush_bullets()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1 and not title_used:
                story.append(para(text, st["title"]))
                title_used = True
            elif level == 1:
                story.append(para(text, st["h1"]))
            elif level == 2:
                story.append(para(text, st["h2"]))
            else:
                story.append(para(text, st["h3"]))
            i += 1
            continue

        bullet_match = re.match(r"^[-*+]\s+(.*)$", stripped)
        if bullet_match:
            flush_paragraph(paragraph_buffer, story, st)
            bullet_buffer.append(para("□ " + bullet_match.group(1), st["bullet"]))
            i += 1
            continue

        ordered_match = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if ordered_match:
            flush_paragraph(paragraph_buffer, story, st)
            bullet_buffer.append(para("□ " + ordered_match.group(1), st["bullet"]))
            i += 1
            continue

        quote_match = re.match(r"^>\s*(.*)$", stripped)
        if quote_match:
            flush_paragraph(paragraph_buffer, story, st)
            flush_bullets()
            story.append(para(quote_match.group(1), st["quote"]))
            i += 1
            continue

        paragraph_buffer.append(line)
        i += 1

    flush_paragraph(paragraph_buffer, story, st)
    flush_bullets()
    if code_buffer:
        story.append(Paragraph(esc("\n".join(code_buffer)).replace("\n", "<br/>"), st["code"]))
    if not title_used:
        story.insert(0, para(title, st["title"]))
    return story


def convert(md_path: Path, out_path: Path | None = None) -> Path:
    register_font()
    out_path = out_path or md_path.with_suffix(".pdf")
    text = md_path.read_text(encoding="utf-8-sig")
    title = md_path.stem
    story = build_story(text, title)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=13 * mm,
    )
    doc.build(story, onFirstPage=footer(title), onLaterPages=footer(title))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UTF-8 Markdown report to a Chinese PDF.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    for input_path in args.inputs:
        out_path = None
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / (input_path.stem + ".pdf")
        written = convert(input_path, out_path)
        print(written)


if __name__ == "__main__":
    main()
