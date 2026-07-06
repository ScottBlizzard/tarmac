from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "汇报" / "Task7公平重跑医生版_2026-05-13"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_CSV = ROOT / "reports" / "ThymicGross" / "task7_fair_benchmark_overall_results_2026-05-13.csv"
CONF_CSV = ROOT / "reports" / "ThymicGross" / "task7_fair_benchmark_confusion_results_2026-05-13.csv"
OUT_MD = REPORT_DIR / "Task7公平重跑结果与混淆矩阵_2026-05-13.md"
OUT_PDF = REPORT_DIR / "Task7公平重跑结果与混淆矩阵_2026-05-13.pdf"
HEATMAP_PNG = REPORT_DIR / "task7_confusion_grid.png"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7Fair"


def register_fonts() -> str:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for name in candidates:
        if name in available:
            chosen = name
            break
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.read_csv(OVERALL_CSV)
    conf = pd.read_csv(CONF_CSV)
    return overall, conf


def add_bias_columns(conf: pd.DataFrame) -> pd.DataFrame:
    out = conf.copy()
    out["高危漏判数"] = out["true_high_pred_low"]
    out["低危误判高危数"] = out["true_low_pred_high"]
    out["误差差值"] = out["高危漏判数"] - out["低危误判高危数"]
    tendency = []
    for _, row in out.iterrows():
        diff = int(row["误差差值"])
        if diff >= 3:
            tendency.append("更偏保守，容易把高危放回低危")
        elif diff <= -3:
            tendency.append("更偏激进，容易把低危提到高危")
        else:
            tendency.append("两侧错误相近")
    out["当前偏好"] = tendency
    return out


def draw_heatmap_grid(conf: pd.DataFrame) -> None:
    labels = ["真低危", "真高危"]
    pred_labels = ["判低危", "判高危"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=180)
    axes = axes.flatten()

    for idx, (_, row) in enumerate(conf.iterrows()):
        ax = axes[idx]
        cm = np.array(
            [
                [int(row["true_low_pred_low"]), int(row["true_low_pred_high"])],
                [int(row["true_high_pred_low"]), int(row["true_high_pred_high"])],
            ],
            dtype=float,
        )
        row_sum = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        im = ax.imshow(norm, cmap="Oranges", vmin=0.0, vmax=1.0)
        ax.set_title(str(row["model_name"]), fontsize=10)
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(pred_labels, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(2):
            for j in range(2):
                text = f"{int(cm[i, j])}\n{norm[i, j]:.2f}"
                color = "white" if norm[i, j] > 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    axes[-1].axis("off")
    fig.suptitle("Task7 七个模型的混淆矩阵（数字为计数/比例）", fontsize=15, fontweight="bold", y=0.985)
    cax = fig.add_axes([0.22, 0.875, 0.56, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("按真值行归一化比例", fontsize=10, labelpad=4)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.77, wspace=0.28, hspace=0.36)
    fig.savefig(HEATMAP_PNG, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if col in float_cols:
                vals.append(fmt(float(value)))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown(overall: pd.DataFrame, conf: pd.DataFrame) -> None:
    best_f1 = overall.sort_values(["f1", "auc"], ascending=[False, False]).iloc[0]
    best_auc = overall.sort_values(["auc", "f1"], ascending=[False, False]).iloc[0]
    lines: list[str] = []
    lines.append("# Task7 公平重跑结果与混淆矩阵")
    lines.append("")
    lines.append("Task7 定义：低危组为 A / AB / B1，高危组为 B2 / B3 / TC。")
    lines.append("")
    lines.append("这一版专门回答两个问题：")
    lines.append("")
    lines.append("1. 在同一套 Task7 定义下，不同模型整体结果谁更好。")
    lines.append("2. 各模型更偏向把哪些病例放回低危，或者把哪些病例提前判成高危。")
    lines.append("")
    lines.append("## 1. 七个模型总表")
    lines.append("")
    lines.append(
        markdown_table(
            overall[
                [
                    "model_name",
                    "auc",
                    "accuracy",
                    "balanced_accuracy",
                    "sensitivity",
                    "specificity",
                    "f1",
                ]
            ],
            {"auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"},
        )
    )
    lines.append("")
    lines.append("## 2. 各模型混淆矩阵计数")
    lines.append("")
    lines.append(
        markdown_table(
            conf[
                [
                    "model_name",
                    "true_low_pred_low",
                    "true_low_pred_high",
                    "true_high_pred_low",
                    "true_high_pred_high",
                    "当前偏好",
                ]
            ]
        )
    )
    lines.append("")
    lines.append("## 3. 我们的阶段判断")
    lines.append("")
    lines.append(
        f"- 当前 F1 最好的是 {best_f1['model_name']}，F1={fmt(float(best_f1['f1']))}，AUC={fmt(float(best_f1['auc']))}。"
    )
    lines.append(
        f"- 当前 AUC 最好的是 {best_auc['model_name']}，AUC={fmt(float(best_auc['auc']))}，F1={fmt(float(best_auc['f1']))}。"
    )
    lines.append("- 七个模型都不是主要把低危抬成高危，而是更容易把高危放回低危，说明 Task7 当前主要瓶颈仍然是高危漏判。")
    lines.append("- DINOv2 vitb14 whole 是这轮最稳的单模；experience-aux 版本略改善了高危召回，但还没有超过 vitb14 whole。")
    lines.append("- whole+crop 这条 CNN 双分支在 Task7 上没有带来收益，说明这个任务更适合稳定的 whole-image 主线。")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Task7Title",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#19324f"),
        spaceAfter=4 * mm,
    )
    h1 = ParagraphStyle(
        "Task7H1",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=13.2,
        leading=18,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#16324f"),
        spaceBefore=2.5 * mm,
        spaceAfter=1.5 * mm,
    )
    body = ParagraphStyle(
        "Task7Body",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#222222"),
        wordWrap="CJK",
        spaceAfter=1 * mm,
    )
    table_head = ParagraphStyle(
        "Task7TableHead",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.8,
        leading=10.5,
        textColor=colors.HexColor("#16324f"),
    )
    table_body = ParagraphStyle(
        "Task7TableBody",
        parent=body,
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=10.2,
        textColor=colors.HexColor("#222222"),
    )
    return title, h1, body, table_head, table_body


def wrap_table(data: list[list[str]], widths: list[float], head_style: ParagraphStyle, body_style: ParagraphStyle) -> Table:
    rows = []
    for i, row in enumerate(data):
        style = head_style if i == 0 else body_style
        rows.append([Paragraph(cell, style) for cell in row])
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#b9c8d8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return table


def build_pdf(overall: pd.DataFrame, conf: pd.DataFrame) -> None:
    title_style, h1_style, body_style, table_head, table_body = build_styles()
    best_f1 = overall.sort_values(["f1", "auc"], ascending=[False, False]).iloc[0]
    best_auc = overall.sort_values(["auc", "f1"], ascending=[False, False]).iloc[0]

    story = []
    story.append(p("Task7 公平重跑结果与混淆矩阵", title_style))
    story.append(p("Task7 定义：低危组为 A / AB / B1，高危组为 B2 / B3 / TC。", body_style))
    story.append(
        p(
            "这一版专门回答两个问题：第一，在同一套 Task7 定义下，哪些模型整体结果更好；第二，各模型更偏向把病例往低危还是高危方向放，从而看清它们的错误分布和分类偏好。",
            body_style,
        )
    )
    story.append(Spacer(1, 2 * mm))

    story.append(p("1. 七个模型总表", h1_style))
    overall_table = [["模型", "AUC", "准确率", "平衡准确率", "敏感度", "特异度", "F1"]]
    for _, row in overall.iterrows():
        overall_table.append(
            [
                str(row["model_name"]),
                fmt(float(row["auc"])),
                fmt(float(row["accuracy"])),
                fmt(float(row["balanced_accuracy"])),
                fmt(float(row["sensitivity"])),
                fmt(float(row["specificity"])),
                fmt(float(row["f1"])),
            ]
        )
    story.append(
        wrap_table(
            overall_table,
            [46 * mm, 18 * mm, 20 * mm, 24 * mm, 18 * mm, 18 * mm, 16 * mm],
            table_head,
            table_body,
        )
    )
    story.append(Spacer(1, 2.5 * mm))
    story.append(
        p(
            f"这一轮里，F1 和 AUC 都是 {best_f1['model_name']} 最好，F1={fmt(float(best_f1['f1']))}，AUC={fmt(float(best_f1['auc']))}。从整体上看，DINOv2 主线明显强于传统 CNN；其中 vitb14 whole 是当前最稳的单模。",
            body_style,
        )
    )

    story.append(p("2. 各模型混淆矩阵小图", h1_style))
    story.append(
        p(
            "下面每个小图都按真值行归一化。左上角和右下角越高越好；右上角表示把低危误判成高危；左下角表示把高危漏判回低危。",
            body_style,
        )
    )
    story.append(Image(str(HEATMAP_PNG), width=255 * mm, height=127 * mm))
    story.append(Spacer(1, 2 * mm))

    story.append(p("3. 各模型的错误偏好", h1_style))
    bias_table = [["模型", "真低危判低危", "真低危判高危", "真高危判低危", "真高危判高危", "当前偏好"]]
    for _, row in conf.iterrows():
        bias_table.append(
            [
                str(row["model_name"]),
                str(int(row["true_low_pred_low"])),
                str(int(row["true_low_pred_high"])),
                str(int(row["true_high_pred_low"])),
                str(int(row["true_high_pred_high"])),
                str(row["当前偏好"]),
            ]
        )
    story.append(
        wrap_table(
            bias_table,
            [42 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm, 58 * mm],
            table_head,
            table_body,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        p(
            "从这张表可以直接看出，这七个模型都不是主要把低危提成高危，而是更容易把高危放回低危。也就是说，当前 Task7 最大的问题仍然是高危漏判，而不是过度预警。",
            body_style,
        )
    )
    story.append(
        p(
            "具体来看，SE-ResNeXt50 whole+crop 这版双分支最弱，说明这个任务并不适合继续沿着 dual 方向深挖。vits14+vitb14 concat 能进入第一梯队，但没有超过 vitb14 whole。experience-aux 版本把高危漏判从 20 例压到 19 例，但同时把低危误判高危从 13 例增到 14 例，说明经验标签确实在改变模型偏好，只是当前收益还不够大。",
            body_style,
        )
    )
    story.append(
        p(
            "如果后面要继续提升，我们建议优先看三类病例：第一，肉眼上偏淡白、但病理属于高危的病例；第二，B2 / B3 / TC 边界病例；第三，当前多个模型都稳定错到同一侧的病例。这三类最可能决定 Task7 下一步还能提高多少。",
            body_style,
        )
    )

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title="Task7公平重跑结果与混淆矩阵",
        author="OpenAI Codex",
    )
    doc.build(story)


def main() -> None:
    register_fonts()
    overall, conf_raw = load_inputs()
    conf = add_bias_columns(conf_raw)
    overall = overall.sort_values(["f1", "auc"], ascending=[False, False]).reset_index(drop=True)
    draw_heatmap_grid(conf)
    write_markdown(overall, conf)
    build_pdf(overall, conf)
    print(OUT_MD)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
