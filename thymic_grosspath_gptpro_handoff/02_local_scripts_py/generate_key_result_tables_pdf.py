from __future__ import annotations

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
OUTPUT_MD = ROOT / "汇报" / "重要实验结果数据表_2026-05-08.md"
OUTPUT_PDF = ROOT / "汇报" / "重要实验结果数据表_2026-05-08.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiKeyTables"


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
        spaceAfter=4 * mm,
    )
    subtitle = ParagraphStyle(
        "SubtitleCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9.5,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4b5b6b"),
        spaceAfter=4 * mm,
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
    note = ParagraphStyle(
        "NoteCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=8.8,
        leading=12,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#334455"),
        spaceAfter=2 * mm,
    )
    table_header = ParagraphStyle(
        "TableHeaderCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=8.2,
        leading=10.2,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_body = ParagraphStyle(
        "TableBodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=7.9,
        leading=10.0,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    return title, subtitle, h1, note, table_header, table_body


def fmt(x: float) -> str:
    return f"{x:.4f}"


TASK_DEF = [
    ["任务", "临床问题", "类别定义", "病例数", "主指标", "当前阶段定位"],
    ["Task1", "良性增生 vs 胸腺上皮肿瘤", "benign_hyperplasia vs TET", "160", "AUC / BACC", "基础可分性验证"],
    ["Task2", "胸腺瘤 vs 胸腺癌", "TET vs TC", "140", "AUC / BACC", "恶性程度相关分层"],
    ["Task3", "低危 vs 高危", "A/AB/B1 vs B2/B3", "100", "AUC / BACC", "当前风险分层主线"],
    ["Task4", "WHO 五分类", "A / AB / B1 / B2 / B3", "100", "Macro-F1 / Macro-AUC", "当前细分类主线"],
    ["Task5", "三分类", "A/AB vs B1-3 vs TC", "120", "Macro-F1 / Macro-AUC", "医生提出的新临床友好主线"],
    ["Task6", "六分类", "A / AB / B1 / B2 / B3 / TC", "120", "Macro-F1 / Macro-AUC", "Task5 向细粒度扩展"],
]


CURRENT_BEST = [
    ["任务", "汇报口径", "当前最好配置", "关键结果 1", "关键结果 2", "备注"],
    ["Task1", "AUC-best", "DINOv2 vitl14 + whole", f"AUC = {fmt(0.8707142857)}", f"BACC = {fmt(0.7321428571)}", "排序能力最好"],
    ["Task1", "BACC-best", "DINOv2 vits14 + whole", f"BACC = {fmt(0.7607142857)}", f"AUC = {fmt(0.8160714286)}", "平衡分类更稳"],
    ["Task2", "current-best", "DINOv2 vits14 + whole", f"AUC = {fmt(0.86375)}", f"BACC = {fmt(0.7875)}", "当前双指标都最好"],
    ["Task3", "current-best", "DINOv2 vitl14 + whole + fold-wise threshold", f"AUC = {fmt(0.7620833333)}", f"BACC = {fmt(0.7166666667)}", "当前正式风险分层结果"],
    ["Task4", "F1-best", "DINOv2 vits14 + vitb14 global blend", f"Macro-F1 = {fmt(0.523607)}", f"Macro-AUC = {fmt(0.75325)}", "当前最好 F1"],
    ["Task4", "AUC-best", "DINOv2 vits14 + vitb14 + PLIP blend", f"Macro-AUC = {fmt(0.765625)}", f"Macro-F1 = {fmt(0.522606)}", "当前最好 AUC"],
    ["Task5", "F1-best", "DINOv2 vitl14 + whole", f"Macro-F1 = {fmt(0.6613756614)}", f"Macro-AUC = {fmt(0.7958865741)}", "当前三分类主结果"],
    ["Task5", "AUC-best", "DINOv2 vitl14 + vitb14 feature concat", f"Macro-AUC = {fmt(0.8229467593)}", f"Macro-F1 = {fmt(0.6573105786)}", "AUC 更高"],
    ["Task6", "current-best", "vits14 + vitb14 concat + Task5 vitl14 hierarchical gate", f"Macro-F1 = {fmt(0.5127355498)}", f"Macro-AUC = {fmt(0.8095)}", "当前六分类最好结果"],
]


TASK5_COMPARE = [
    ["方法", "配置", "Macro-F1", "Macro-AUC", "结论"],
    ["单模", "DINOv2 vits14 + whole", fmt(0.5859), fmt(0.7947), "明显弱于 vitl14"],
    ["单模", "DINOv2 vitl14 + whole", fmt(0.6614), fmt(0.7959), "当前最好 F1"],
    ["概率融合", "vitl14 + vits14", fmt(0.5987), fmt(0.7802), "不如 vitl14 单模"],
    ["特征拼接", "vitl14 + vitb14", fmt(0.6573), fmt(0.8229), "当前最好 AUC"],
]


TASK6_COMPARE = [
    ["方法", "配置", "Macro-F1", "Macro-AUC", "结论"],
    ["单模", "DINOv2 vits14 + whole", fmt(0.4847), fmt(0.7647), "中等"],
    ["单模", "DINOv2 vitl14 + whole", fmt(0.4595), fmt(0.7635), "不如 vits14 / vitb14"],
    ["单模", "DINOv2 vitb14 + whole", fmt(0.4864), fmt(0.7993), "当前最好单模"],
    ["概率融合", "vits14 + vitb14", fmt(0.4409), fmt(0.7778), "无效"],
    ["特征拼接", "vits14 + vitb14", fmt(0.5028), fmt(0.8067), "首次把 F1 推过 0.50"],
    ["层级增强", "concat(vits14 + vitb14) + Task5 vitl14 gate", fmt(0.5127), fmt(0.8095), "当前最好结果"],
]


ERROR_SUMMARY = [
    ["任务", "关键错误/薄弱点", "量化结果", "解释"],
    ["Task5", "TC 召回率偏低", "TC recall = 0.5000; TC F1 = 0.5556", "主要难点集中在 TC 与 B1-3 的视觉边界"],
    ["Task5", "主要混淆对", "TC -> B1-3: 7; B1-3 -> TC: 5", "说明三分类里最难的是 TC 与高危 B 组区分"],
    ["Task6", "A 类最难", "A recall = 0.3000; A F1 = 0.3529", "A 容易被拉向 AB / B1 / B2 / B3"],
    ["Task6", "B2 类最难", "B2 recall = 0.3500; B2 F1 = 0.3256", "B2 是最典型的边界聚集类"],
    ["Task6", "最常见混淆", "B3 -> B2: 5; B2 -> B3: 4; B2 -> TC: 4; TC -> B2: 4; A -> AB: 4", "主要卡在 A/AB 和 B2/B3/TC 这两条边界"],
    ["Task6", "增强版是否有效", "总错例 61 -> 58; 跨粗分组大错 39 -> 34", "层级先验主要帮助减少跨组大错"],
]


NEGATIVE_LINES = [
    ["方向", "代表方法", "当前判断"],
    ["旧主线 Fusion", "SuRImage fusion", "在当前数据规模下不稳定，已降级"],
    ["Ordinal loss", "CONDOR", "不如普通 CE，当前不建议继续主推"],
    ["轻量微调", "DINO last-block + MLP", "fold1 即明显负结果，不建议扩成正式 5-fold"],
    ["复杂融合头", "DINO + PLIP stacking / MLP fusion", "没有打过简单全局加权融合"],
    ["BiomedCLIP prompt 线", "CoOp_BiomedCLIP", "已打通，但当前效果不如 DINO 主线"],
]


TABLE_SPECS = [
    ("1. 任务定义与样本规模", TASK_DEF, [18 * mm, 40 * mm, 56 * mm, 16 * mm, 30 * mm, 83 * mm]),
    ("2. 当前最好结果总表", CURRENT_BEST, [18 * mm, 22 * mm, 63 * mm, 35 * mm, 35 * mm, 99 * mm]),
    ("3. Task5 三分类关键方法对比", TASK5_COMPARE, [20 * mm, 68 * mm, 26 * mm, 26 * mm, 110 * mm]),
    ("4. Task6 六分类关键方法对比", TASK6_COMPARE, [20 * mm, 68 * mm, 26 * mm, 26 * mm, 110 * mm]),
    ("5. Task5 / Task6 错误分析摘要", ERROR_SUMMARY, [18 * mm, 44 * mm, 52 * mm, 149 * mm]),
    ("6. 已验证但当前不建议继续主推的方向", NEGATIVE_LINES, [30 * mm, 55 * mm, 175 * mm]),
]


def build_table(data: list[list[str]], col_widths: list[float], header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    wrapped = []
    for row_idx, row in enumerate(data):
        style = header_style if row_idx == 0 else body_style
        wrapped.append([Paragraph(cell.replace("&", "&amp;"), style) for cell in row])
    table = Table(wrapped, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c8d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.8),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return table


def write_markdown() -> None:
    lines = [
        "# 重要实验结果数据表",
        "",
        "日期：2026-05-08",
        "",
        "说明：Task1-Task3 为二分类任务，重点看 AUC 与 BACC；Task4-Task6 为多分类任务，重点看 Macro-F1 与 Macro-AUC。",
        "",
    ]
    for title, data, _ in TABLE_SPECS:
        lines.append(f"## {title}")
        lines.append("")
        header = data[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in data[1:]:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_pdf() -> None:
    register_font()
    title_style, subtitle_style, h1_style, note_style, table_header_style, table_body_style = build_styles()
    story = [
        Paragraph("重要实验结果数据表", title_style),
        Paragraph(
            "日期：2026-05-08　　说明：Task1-Task3 为二分类任务，重点看 AUC 与 BACC；Task4-Task6 为多分类任务，重点看 Macro-F1 与 Macro-AUC。",
            subtitle_style,
        ),
    ]

    for title, data, widths in TABLE_SPECS:
        story.append(Paragraph(title, h1_style))
        story.append(build_table(data, widths, table_header_style, table_body_style))
        story.append(Spacer(1, 3 * mm))

    story.append(
        Paragraph(
            "简要结论：当前最强结果主要集中在 DINOv2 强表征主线。Task3 当前最好结果为 vitl14 whole + fold-wise threshold；Task4 当前最好 F1 为 vits14 + vitb14，全局最好 AUC 为 vits14 + vitb14 + PLIP；Task5 当前已形成较稳的临床友好三分类结果；Task6 当前最好结果来自双 backbone 特征拼接并叠加 Task5 层级先验。",
            note_style,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title="重要实验结果数据表",
        author="OpenAI Codex",
    )
    doc.build(story)


def main() -> None:
    write_markdown()
    build_pdf()
    print(OUTPUT_MD)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
