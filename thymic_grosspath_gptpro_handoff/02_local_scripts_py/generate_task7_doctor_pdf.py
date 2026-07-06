from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "task7_from_task6_struct_aux_manual_364_bax"
SUMMARY_CSV = ROOT / "artifacts" / "image_review" / "thymic_task567_summary.csv"
OUT_DIR = ROOT / "汇报" / "Task7_lowhigh_tc_report_2026-05-13"
OUT_MD = OUT_DIR / "Task7低危高危含TC首轮结果_2026-05-13.md"
OUT_PDF = OUT_DIR / "Task7低危高危含TC首轮结果_2026-05-13.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7Report"


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d"),
        spaceAfter=5 * mm,
    )
    h1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=13.5,
        leading=18,
        textColor=colors.HexColor("#16324f"),
        spaceBefore=3 * mm,
        spaceAfter=1.5 * mm,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.2,
        leading=15,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#222222"),
        spaceAfter=1.2 * mm,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=8 * mm,
        firstLineIndent=-4 * mm,
    )
    table_header = ParagraphStyle(
        "TableHeaderCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.9,
        leading=11,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_body = ParagraphStyle(
        "TableBodyCN",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.8,
        leading=10.8,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    return title, h1, body, bullet, table_header, table_body


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def wrap_table(data: list[list[str]], col_widths: list[float], header_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    rows = []
    for i, row in enumerate(data):
        style = header_style if i == 0 else body_style
        rows.append([Paragraph(cell, style) for cell in row])
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#b9c8d8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
                ("TOPPADDING", (0, 0), (-1, -1), 4.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.0),
            ]
        )
    )
    return table


def load_inputs():
    metrics = pd.read_csv(INPUT_DIR / "oof_metrics_task7.csv").iloc[0].to_dict()
    folds = pd.read_csv(INPUT_DIR / "cv_fold_summary_task7.csv")
    counts = pd.read_csv(INPUT_DIR / "confusion_matrix_counts_task7.csv")
    row_norm = pd.read_csv(INPUT_DIR / "confusion_matrix_row_norm_task7.csv")
    summary = pd.read_csv(SUMMARY_CSV)
    task7_rows = summary[summary["task"] == "task7_lowhigh_tc"].copy()
    return metrics, folds, counts, row_norm, task7_rows


def build_markdown(metrics: dict, folds: pd.DataFrame, counts: pd.DataFrame, row_norm: pd.DataFrame, task7_rows: pd.DataFrame) -> str:
    high_cases = int(task7_rows.loc[task7_rows["label_name"] == "high_risk_group", "num_cases"].iloc[0])
    low_cases = int(task7_rows.loc[task7_rows["label_name"] == "low_risk_group", "num_cases"].iloc[0])
    lines: list[str] = []
    lines.append("# Task7低危高危含TC首轮结果")
    lines.append("")
    lines.append("这份结果单独整理给医生团队，任务定义更贴近临床分层语言。")
    lines.append("")
    lines.append("## 1. 任务定义")
    lines.append("")
    lines.append("- 低危组：A / AB / B1")
    lines.append("- 高危组：B2 / B3 / TC")
    lines.append("")
    lines.append("这条任务线和原来的 Task3 不同。原来的 Task3 只在胸腺瘤内部做低危和高危分层，不包含 TC。Task7 把 TC 并入高危组，更接近临床实际讨论方式。")
    lines.append("")
    lines.append("## 2. 当前纳入病例")
    lines.append("")
    lines.append(f"- 总病例数：{high_cases + low_cases}")
    lines.append(f"- 低危组：{low_cases} 例")
    lines.append(f"- 高危组：{high_cases} 例")
    lines.append("- 每折：低危 12 例，高危 12 例")
    lines.append("")
    lines.append("## 3. 首轮结果")
    lines.append("")
    lines.append("这版结果先没有重跑模型，而是直接从当前保留完整 Task6 OOF 预测的一版结果中折叠得到。来源模型是 Task6 的结构化经验标签辅助版。")
    lines.append("")
    lines.append(f"- AUC = {metrics['auc']:.4f}")
    lines.append(f"- Accuracy = {metrics['accuracy']:.4f}")
    lines.append(f"- Balanced Accuracy = {metrics['balanced_accuracy']:.4f}")
    lines.append(f"- Sensitivity = {metrics['sensitivity']:.4f}")
    lines.append(f"- Specificity = {metrics['specificity']:.4f}")
    lines.append(f"- F1 = {metrics['f1']:.4f}")
    lines.append("")
    lines.append("## 4. 分折结果")
    lines.append("")
    lines.append("| Fold | AUC | Accuracy | BACC | Sensitivity | Specificity | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in folds.iterrows():
        lines.append(
            f"| {int(row['fold_id'])} | {row['auc']:.4f} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} |"
        )
    lines.append("")
    lines.append("## 5. 当前错误分布")
    lines.append("")
    lines.append("按病例计数：")
    lines.append("")
    lines.append("| 真值 \\\\ 预测 | 低危组 | 高危组 |")
    lines.append("| --- | ---: | ---: |")
    for _, row in counts.iterrows():
        lines.append(f"| {row['true_label']} | {int(row['low_risk_group'])} | {int(row['high_risk_group'])} |")
    lines.append("")
    lines.append("按真值行归一化：")
    lines.append("")
    lines.append("| 真值 \\\\ 预测 | 低危组 | 高危组 |")
    lines.append("| --- | ---: | ---: |")
    for _, row in row_norm.iterrows():
        lines.append(f"| {row['true_label']} | {float(row['low_risk_group']):.4f} | {float(row['high_risk_group']):.4f} |")
    lines.append("")
    lines.append("## 6. 当前怎么看这版结果")
    lines.append("")
    lines.append("- 这条结果已经可以先给医生看，不需要先等待完整重跑。")
    lines.append("- 当前低危组保留得更好，特异度较高。")
    lines.append("- 目前的主要提升空间还是高危组召回率，也就是让更多 B2 / B3 / TC 被稳定放进高危。")
    lines.append("")
    lines.append("## 7. 如果要做公平重跑")
    lines.append("")
    lines.append("建议先跑一个紧凑版对照，不要一开始把所有历史分支都拉进来：")
    lines.append("")
    lines.append("- SE-ResNeXt50 + whole")
    lines.append("- DINOv2 vits14 + whole")
    lines.append("- DINOv2 vitl14 + whole")
    lines.append("- DINOv2 vitb14 + whole")
    lines.append("- 如需补一版增强，再加当前经验标签增强链路映射到 Task7")
    lines.append("")
    return "\n".join(lines)


def build_pdf(metrics: dict, folds: pd.DataFrame, counts: pd.DataFrame, row_norm: pd.DataFrame, task7_rows: pd.DataFrame) -> None:
    register_font()
    title_style, h1_style, body_style, bullet_style, table_header_style, table_body_style = build_styles()
    high_cases = int(task7_rows.loc[task7_rows["label_name"] == "high_risk_group", "num_cases"].iloc[0])
    low_cases = int(task7_rows.loc[task7_rows["label_name"] == "low_risk_group", "num_cases"].iloc[0])

    story = []
    story.append(p("Task7低危高危含TC首轮结果", title_style))

    story.append(p("1. 任务定义", h1_style))
    story.append(p("这条任务线采用更贴近临床分层语言的二分类定义。", body_style))
    story.append(p("• 低危组：A / AB / B1", bullet_style))
    story.append(p("• 高危组：B2 / B3 / TC", bullet_style))
    story.append(p("这条任务线和原来的 Task3 不同。原来的 Task3 只在胸腺瘤内部做低危和高危分层，不包含 TC。Task7 把 TC 并入高危组，更接近临床实际讨论方式。", body_style))

    story.append(p("2. 当前纳入病例", h1_style))
    story.append(p(f"总病例数：{high_cases + low_cases}；低危组：{low_cases} 例；高危组：{high_cases} 例；每折：低危 12 例，高危 12 例。", body_style))

    story.append(p("3. 首轮结果", h1_style))
    story.append(p("这版结果先没有重跑模型，而是直接从当前保留完整 Task6 OOF 预测的一版结果中折叠得到。来源模型是 Task6 的结构化经验标签辅助版。", body_style))
    metrics_table = wrap_table(
        [
            ["指标", "结果"],
            ["AUC", f"{metrics['auc']:.4f}"],
            ["Accuracy", f"{metrics['accuracy']:.4f}"],
            ["Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}"],
            ["Sensitivity", f"{metrics['sensitivity']:.4f}"],
            ["Specificity", f"{metrics['specificity']:.4f}"],
            ["F1", f"{metrics['f1']:.4f}"],
        ],
        [48 * mm, 42 * mm],
        table_header_style,
        table_body_style,
    )
    story.append(metrics_table)
    story.append(Spacer(1, 3 * mm))

    story.append(p("4. 分折结果", h1_style))
    fold_data = [["Fold", "AUC", "Accuracy", "BACC", "Sensitivity", "Specificity", "F1"]]
    for _, row in folds.iterrows():
        fold_data.append(
            [
                str(int(row["fold_id"])),
                f"{row['auc']:.4f}",
                f"{row['accuracy']:.4f}",
                f"{row['balanced_accuracy']:.4f}",
                f"{row['sensitivity']:.4f}",
                f"{row['specificity']:.4f}",
                f"{row['f1']:.4f}",
            ]
        )
    story.append(
        wrap_table(
            fold_data,
            [14 * mm, 22 * mm, 24 * mm, 24 * mm, 26 * mm, 26 * mm, 18 * mm],
            table_header_style,
            table_body_style,
        )
    )
    story.append(Spacer(1, 3 * mm))

    story.append(p("5. 当前错误分布", h1_style))
    story.append(p("按病例计数：", body_style))
    count_data = [["真值 \\ 预测", "低危组", "高危组"]]
    for _, row in counts.iterrows():
        count_data.append([str(row["true_label"]), str(int(row["low_risk_group"])), str(int(row["high_risk_group"]))])
    story.append(wrap_table(count_data, [38 * mm, 28 * mm, 28 * mm], table_header_style, table_body_style))
    story.append(Spacer(1, 2 * mm))
    story.append(p("按真值行归一化：", body_style))
    norm_data = [["真值 \\ 预测", "低危组", "高危组"]]
    for _, row in row_norm.iterrows():
        norm_data.append([str(row["true_label"]), f"{float(row['low_risk_group']):.4f}", f"{float(row['high_risk_group']):.4f}"])
    story.append(wrap_table(norm_data, [38 * mm, 28 * mm, 28 * mm], table_header_style, table_body_style))
    story.append(Spacer(1, 3 * mm))

    story.append(p("6. 当前怎么看这版结果", h1_style))
    story.append(p("• 这条结果已经可以先给医生看，不需要先等待完整重跑。", bullet_style))
    story.append(p("• 当前低危组保留得更好，特异度较高。", bullet_style))
    story.append(p("• 目前的主要提升空间还是高危组召回率，也就是让更多 B2 / B3 / TC 被稳定放进高危。", bullet_style))

    story.append(p("7. 如果要做公平重跑", h1_style))
    story.append(p("建议先跑一个紧凑版对照，不要一开始把所有历史分支都拉进来。", body_style))
    story.append(p("• SE-ResNeXt50 + whole", bullet_style))
    story.append(p("• DINOv2 vits14 + whole", bullet_style))
    story.append(p("• DINOv2 vitl14 + whole", bullet_style))
    story.append(p("• DINOv2 vitb14 + whole", bullet_style))
    story.append(p("• 如需补一版增强，再加当前经验标签增强链路映射到 Task7", bullet_style))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Task7低危高危含TC首轮结果",
        author="OpenAI Codex",
    )
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, folds, counts, row_norm, task7_rows = load_inputs()
    OUT_MD.write_text(build_markdown(metrics, folds, counts, row_norm, task7_rows), encoding="utf-8")
    build_pdf(metrics, folds, counts, row_norm, task7_rows)
    print(OUT_MD)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
