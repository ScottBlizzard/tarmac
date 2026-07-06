from __future__ import annotations

import csv
import html
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "汇报" / "task56_case_review_assets"
CASE_CSV = ASSET_DIR / "case_list.csv"
MANIFEST_CSV = ASSET_DIR / "manifest.csv"
IMAGE_DIR = ASSET_DIR / "images"
PDF_CACHE_DIR = ASSET_DIR / "pdf_cache"
OUTPUT_MD = ROOT / "汇报" / "Task56高置信错例医生回看清单_2026-05-07.md"
OUTPUT_PDF = ROOT / "汇报" / "Task56高置信错例医生回看清单_2026-05-07.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask56CaseReview"


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
        fontSize=14.5,
        leading=19.5,
        textColor=colors.HexColor("#153a5b"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    h2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=12.0,
        leading=16.0,
        textColor=colors.HexColor("#244a73"),
        spaceBefore=2.5 * mm,
        spaceAfter=1.5 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9.9,
        leading=14.5,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#222222"),
        spaceAfter=1.2 * mm,
    )
    note = ParagraphStyle(
        "NoteCN",
        parent=body,
        backColor=colors.HexColor("#f6f8fb"),
        borderColor=colors.HexColor("#d6e0ec"),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=1.5 * mm,
        spaceAfter=2.5 * mm,
    )
    table_header = ParagraphStyle(
        "TableHeaderCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.7,
        leading=10.5,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_body = ParagraphStyle(
        "TableBodyCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=10.3,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    caption = ParagraphStyle(
        "CaptionCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=10.0,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#465a6d"),
        spaceBefore=1 * mm,
        spaceAfter=2 * mm,
    )
    return title, h1, h2, body, note, table_header, table_body, caption


def clean(text: str) -> str:
    return html.escape(str(text)).replace("`", "").replace("**", "")


def build_table(data, widths, header_style, body_style):
    wrapped = []
    for i, row in enumerate(data):
        style = header_style if i == 0 else body_style
        wrapped.append([Paragraph(clean(cell), style) for cell in row])
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table


def image_flowable(path: Path, max_w_mm: float, max_h_mm: float) -> Image:
    cache_path = get_pdf_cache_image(path)
    reader = ImageReader(str(cache_path))
    w, h = reader.getSize()
    max_w = max_w_mm * mm
    max_h = max_h_mm * mm
    scale = min(max_w / w, max_h / h)
    return Image(str(cache_path), width=w * scale, height=h * scale)


def get_pdf_cache_image(path: Path) -> Path:
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PDF_CACHE_DIR / path.name
    if cache_path.exists():
        return cache_path
    with PILImage.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((1800, 1800))
        img.save(cache_path, format="JPEG", quality=82, optimize=True)
    return cache_path


def build_image_table(paths: list[Path], caption_style: ParagraphStyle):
    cells = []
    for idx, path in enumerate(paths, start=1):
        img = image_flowable(path, 78 if len(paths) > 1 else 120, 78)
        cap = Paragraph(clean(f"图 {idx}: {path.name}"), caption_style)
        cells.append([img, cap])
    rows = []
    if len(cells) == 1:
        rows = [[cells[0][0]], [cells[0][1]]]
        return Table(rows, colWidths=[120 * mm], style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    img_row = [cell[0] for cell in cells]
    cap_row = [cell[1] for cell in cells]
    table = Table([img_row, cap_row], colWidths=[78 * mm] * len(cells))
    table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def build_case_note(row: dict[str, str]) -> str:
    task = row["task"]
    true_name = row["true_name"]
    pred_name = row["pred_name"]
    if task == "Task5":
        return f"该病例在三分类任务中真实为 {true_name}，但被高置信度预测为 {pred_name}。这类病例最能反映 TC 与高危胸腺瘤组之间的视觉重叠。"
    if true_name in {"A", "AB"}:
        return f"该病例位于 A/AB 细分边界附近，模型把真实的 {true_name} 预测成 {pred_name}，说明前部低级别类别的形态界限仍然不稳。"
    if true_name in {"B2", "B3", "TC"} or pred_name in {"B2", "B3", "TC"}:
        return f"该病例落在 B2/B3/TC 这条最难边界带上，模型高置信度地把真实的 {true_name} 预测成 {pred_name}，非常值得医生重点回看。"
    return f"该病例是真实类别 {true_name}、预测类别 {pred_name} 的高置信错例，适合用于复核图像表现、标签边界和取材差异。"


def load_rows():
    rows = list(csv.DictReader(CASE_CSV.open(encoding="utf-8-sig")))
    manifest = list(csv.DictReader(MANIFEST_CSV.open(encoding="utf-8-sig")))
    by_key: dict[tuple[str, str], list[str]] = {}
    for item in manifest:
        by_key.setdefault((item["task"], item["case_id"]), []).append(item["export_name"])
    for row in rows:
        row["export_images"] = sorted(by_key.get((row["task"], row["case_id"]), []))
    return rows


def write_markdown(rows: list[dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# Task5 与 Task6 高置信错例医生回看清单")
    lines.append("日期：2026-05-07")
    lines.append("")
    lines.append("## 1. 这份清单是做什么的")
    lines.append("这份清单只收录当前 Task5 和 Task6 最值得医生回看的高置信错例。所谓高置信错例，是指模型不仅判断错了，而且是以很高概率做出了错误判断。这类病例最有信息量，因为它们往往对应真实的边界病例、取材差异、图像质量问题，或者病理标签与大体图视觉特征之间的复杂关系。")
    lines.append("")
    lines.append("## 2. 选例规则")
    lines.append("- Task5：选取真实为 TC、但被模型高置信度错分的前 6 例。")
    lines.append("- Task6：选取最终增强版模型中置信度最高的前 10 例错例。")
    lines.append("- 每例保留当前最主要的原始大体图，若该病例有两张图，则两张都保留。")
    lines.append("")
    lines.append("## 3. 病例清单")
    lines.append("")
    for idx, row in enumerate(rows, start=1):
        lines.append(f"### {idx}. {row['task']} | case_id={row['case_id']}")
        lines.append(f"- 原始病理标签：{row['who_type_raw']}")
        lines.append(f"- 当前任务真值：{row['true_name']}")
        lines.append(f"- 当前任务预测：{row['pred_name']}")
        lines.append(f"- 所在折：{row['fold_id']}")
        lines.append(f"- 预测类别概率：{float(row['pred_prob']):.4f}")
        lines.append(f"- 真实类别概率：{float(row['true_prob']):.4f}")
        lines.append(f"- 置信差值：{float(row['margin']):.4f}")
        lines.append(f"- 回看建议：{build_case_note(row)}")
        for export_name in row["export_images"]:
            rel = f"task56_case_review_assets/images/{export_name}"
            lines.append(f"![{row['case_id']}]({rel})")
        lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_pdf(rows: list[dict[str, str]]) -> None:
    register_font()
    title, h1, h2, body, note, table_header, table_body, caption = build_styles()
    story = []
    story.append(Paragraph("Task5 与 Task6 高置信错例医生回看清单", title))
    story.append(Paragraph("日期：2026-05-07", body))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("1. 这份清单是做什么的", h1))
    story.append(
        Paragraph(
            "这份清单只收录当前 Task5 和 Task6 最值得医生回看的高置信错例。所谓高置信错例，是指模型不仅判断错了，而且是以很高概率做出了错误判断。这类病例最有信息量，因为它们往往对应真实的边界病例、取材差异、图像质量问题，或者病理标签与大体图视觉特征之间的复杂关系。",
            body,
        )
    )
    story.append(Paragraph("2. 选例规则", h1))
    story.append(Paragraph("• Task5：选取真实为 TC、但被模型高置信度错分的前 6 例。", body))
    story.append(Paragraph("• Task6：选取最终增强版模型中置信度最高的前 10 例错例。", body))
    story.append(Paragraph("• 每例保留当前最主要的原始大体图，若该病例有两张图，则两张都保留。", body))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("3. 病例清单", h1))

    for idx, row in enumerate(rows, start=1):
        case_title = f"{idx}. {row['task']} | case_id={row['case_id']}"
        meta_table = build_table(
            [
                ["字段", "内容"],
                ["原始病理标签", row["who_type_raw"]],
                ["当前任务真值", row["true_name"]],
                ["当前任务预测", row["pred_name"]],
                ["所在折", row["fold_id"]],
                ["预测类别概率", f"{float(row['pred_prob']):.4f}"],
                ["真实类别概率", f"{float(row['true_prob']):.4f}"],
                ["置信差值", f"{float(row['margin']):.4f}"],
            ],
            [34 * mm, 128 * mm],
            table_header,
            table_body,
        )
        image_paths = [IMAGE_DIR / name for name in row["export_images"] if (IMAGE_DIR / name).exists()]
        block = [
            Paragraph(case_title, h2),
            meta_table,
            Paragraph(build_case_note(row), note),
        ]
        if image_paths:
            block.append(build_image_table(image_paths, caption))
        block.append(Spacer(1, 4 * mm))
        story.append(KeepTogether(block))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Task5 与 Task6 高置信错例医生回看清单",
        author="OpenAI Codex",
    )
    doc.build(story)


def main() -> None:
    rows = load_rows()
    write_markdown(rows)
    build_pdf(rows)
    print(OUTPUT_MD)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
