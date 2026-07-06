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
OUTPUT_ROOT = ROOT / "outputs" / "task6_fair_benchmark_20260513"
REPORT_ROOT = ROOT / "reports" / "ThymicGross"
DOCTOR_DIR = ROOT / "汇报" / "Task6公平重跑医生版_2026-05-13"
DOCTOR_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_CSV = REPORT_ROOT / "task6_fair_benchmark_overall_results_2026-05-13.csv"
CONF_CSV = REPORT_ROOT / "task6_fair_benchmark_confusion_results_2026-05-13.csv"
STAGE_MD = REPORT_ROOT / "task6_fair_benchmark_stage_results_2026-05-13.md"
OUT_MD = DOCTOR_DIR / "Task6公平重跑结果与混淆矩阵_2026-05-13.md"
OUT_PDF = DOCTOR_DIR / "Task6公平重跑结果与混淆矩阵_2026-05-13.pdf"
HEATMAP_PNG = DOCTOR_DIR / "task6_confusion_grid.png"

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask6Fair"

CLASS_NAMES = ["A", "AB", "B1", "B2", "B3", "TC"]
MODEL_NAME_MAP = {
    "01_srx50_whole": "SE-ResNeXt50 + whole",
    "02_srx50_dual": "SE-ResNeXt50 + whole+crop",
    "03_dino_vits14_whole": "DINOv2 vits14 + whole",
    "04_dino_vitl14_whole": "DINOv2 vitl14 + whole",
    "05_dino_vitb14_whole": "DINOv2 vitb14 + whole",
    "06_dino_vits_vitb_concat": "DINOv2 vits14+vitb14 concat",
    "07_dino_vits_vitb_experience_aux": "DINOv2 vits14+vitb14 + experience-aux",
}


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


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    conf_rows = []

    for model_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        metrics_path = model_dir / "oof_metrics.csv"
        pred_path = model_dir / "oof_case_predictions_mean.csv"
        if not metrics_path.exists() or not pred_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        case_mean = metrics[
            (metrics["split"] == "test_oof")
            & (metrics["level"] == "case")
            & (metrics["aggregation"] == "mean")
        ].iloc[0]

        model_key = model_dir.name
        model_name = MODEL_NAME_MAP.get(model_key, model_key)
        rows.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "accuracy": float(case_mean["accuracy"]),
                "balanced_accuracy": float(case_mean["balanced_accuracy"]),
                "macro_f1": float(case_mean["macro_f1"]),
                "macro_auc": float(case_mean["macro_auc"]),
            }
        )

        pred = pd.read_csv(pred_path)
        cm = np.zeros((6, 6), dtype=int)
        for _, row in pred.iterrows():
            cm[int(row["label_idx"]), int(row["pred_idx"])] += 1

        conf_record = {"model_key": model_key, "model_name": model_name}
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                conf_record[f"true_{true_name}_pred_{pred_name}"] = int(cm[i, j])

        off_diag = []
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                if i == j:
                    continue
                count = int(cm[i, j])
                if count > 0:
                    off_diag.append((count, true_name, pred_name))
        off_diag.sort(reverse=True)
        conf_record["top_error_1"] = (
            f"{off_diag[0][1]}→{off_diag[0][2]} ({off_diag[0][0]})" if len(off_diag) >= 1 else "-"
        )
        conf_record["top_error_2"] = (
            f"{off_diag[1][1]}→{off_diag[1][2]} ({off_diag[1][0]})" if len(off_diag) >= 2 else "-"
        )
        conf_record["top_error_3"] = (
            f"{off_diag[2][1]}→{off_diag[2][2]} ({off_diag[2][0]})" if len(off_diag) >= 3 else "-"
        )
        conf_rows.append(conf_record)

    overall = pd.DataFrame(rows).sort_values(["macro_f1", "macro_auc"], ascending=[False, False]).reset_index(drop=True)
    conf = pd.DataFrame(conf_rows)
    conf["model_name"] = pd.Categorical(conf["model_name"], categories=list(overall["model_name"]), ordered=True)
    conf = conf.sort_values("model_name").reset_index(drop=True)

    OVERALL_CSV.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OVERALL_CSV, index=False, encoding="utf-8-sig")
    conf.to_csv(CONF_CSV, index=False, encoding="utf-8-sig")
    return overall, conf


def confusion_matrix_from_row(row: pd.Series) -> np.ndarray:
    cm = np.zeros((6, 6), dtype=int)
    for i, true_name in enumerate(CLASS_NAMES):
        for j, pred_name in enumerate(CLASS_NAMES):
            cm[i, j] = int(row[f"true_{true_name}_pred_{pred_name}"])
    return cm


def draw_heatmap_grid(conf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 8.4), dpi=180)
    axes = axes.flatten()

    for idx, (_, row) in enumerate(conf.iterrows()):
        ax = axes[idx]
        cm = confusion_matrix_from_row(row).astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        im = ax.imshow(norm, cmap="Oranges", vmin=0.0, vmax=1.0)
        ax.set_title(str(row["model_name"]), fontsize=10)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        for i in range(6):
            for j in range(6):
                text = f"{int(cm[i, j])}\n{norm[i, j]:.2f}"
                color = "white" if norm[i, j] > 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color=color)

    axes[-1].axis("off")
    fig.suptitle("Task6 七个模型的混淆矩阵（数字为计数/比例）", fontsize=15, fontweight="bold", y=0.985)
    cax = fig.add_axes([0.22, 0.875, 0.56, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("按真值行归一化比例", fontsize=10, labelpad=4)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.75, wspace=0.30, hspace=0.38)
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


def write_stage_md(overall: pd.DataFrame, conf: pd.DataFrame) -> None:
    best_f1 = overall.sort_values(["macro_f1", "macro_auc"], ascending=[False, False]).iloc[0]
    best_auc = overall.sort_values(["macro_auc", "macro_f1"], ascending=[False, False]).iloc[0]

    lines = []
    lines.append("# Task6 七模型公平重跑阶段结果")
    lines.append("")
    lines.append("Task6 定义：A / AB / B1 / B2 / B3 / TC 六分类。")
    lines.append("")
    lines.append("## 总表")
    lines.append("")
    lines.append(
        markdown_table(
            overall[["model_name", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]],
            {"accuracy", "balanced_accuracy", "macro_f1", "macro_auc"},
        )
    )
    lines.append("")
    lines.append("## 主要错误方向")
    lines.append("")
    lines.append(markdown_table(conf[["model_name", "top_error_1", "top_error_2", "top_error_3"]]))
    lines.append("")
    lines.append(f"- 当前按 Macro-F1 最好的是 {best_f1['model_name']}，Macro-F1={fmt(best_f1['macro_f1'])}，Macro-AUC={fmt(best_f1['macro_auc'])}。")
    lines.append(f"- 当前按 Macro-AUC 最好的是 {best_auc['model_name']}，Macro-AUC={fmt(best_auc['macro_auc'])}，Macro-F1={fmt(best_auc['macro_f1'])}。")
    lines.append("- 双 DINO 主线明显强于 SE-ResNeXt50。")
    lines.append("- experience-aux 对整体排序能力有帮助，但当前最好 F1 仍来自 concat。")
    STAGE_MD.write_text("\n".join(lines), encoding="utf-8")


def write_doctor_markdown(overall: pd.DataFrame, conf: pd.DataFrame) -> None:
    best_f1 = overall.sort_values(["macro_f1", "macro_auc"], ascending=[False, False]).iloc[0]
    best_auc = overall.sort_values(["macro_auc", "macro_f1"], ascending=[False, False]).iloc[0]
    lines = []
    lines.append("# Task6 公平重跑结果与混淆矩阵")
    lines.append("")
    lines.append("这一页只看同一套 Task6 定义下的七个模型。Task6 定义为 A / AB / B1 / B2 / B3 / TC 六分类。")
    lines.append("")
    lines.append("我们这次主要想看两件事：")
    lines.append("")
    lines.append("1. 在同一套病例和同一套折分下，哪些模型整体结果更好。")
    lines.append("2. 各模型最容易把哪几类混在一起，错误偏向有没有差异。")
    lines.append("")
    lines.append("## 1. 七个模型总表")
    lines.append("")
    lines.append(
        markdown_table(
            overall[["model_name", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]],
            {"accuracy", "balanced_accuracy", "macro_f1", "macro_auc"},
        )
    )
    lines.append("")
    lines.append("## 2. 主要错误方向")
    lines.append("")
    lines.append(markdown_table(conf[["model_name", "top_error_1", "top_error_2", "top_error_3"]]))
    lines.append("")
    lines.append("## 3. 我们目前的判断")
    lines.append("")
    lines.append(f"- 当前按 Macro-F1 最好的是 {best_f1['model_name']}，Macro-F1={fmt(best_f1['macro_f1'])}。")
    lines.append(f"- 当前按 Macro-AUC 最好的是 {best_auc['model_name']}，Macro-AUC={fmt(best_auc['macro_auc'])}。")
    lines.append("- 这轮最主要的混淆，仍然集中在 A/AB、B1/B2、B2/B3、B2/TC 这几条边界上。")
    lines.append("- 双 DINO 的结果明显优于 SE-ResNeXt50，说明强表征在这个任务上更有优势。")
    lines.append("- experience-aux 这条线能提升整体排序能力，但目前最好 F1 仍然来自 concat。")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Task6Title",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#19324f"),
        spaceAfter=4 * mm,
    )
    body = ParagraphStyle(
        "Task6Body",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.2,
        leading=14,
        alignment=TA_LEFT,
    )
    section = ParagraphStyle(
        "Task6Section",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#19324f"),
        spaceBefore=2 * mm,
        spaceAfter=1.5 * mm,
    )
    small = ParagraphStyle(
        "Task6Small",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=8.7,
        leading=11,
        alignment=TA_LEFT,
    )
    return title, body, section, small


def pdf_table_from_df(df: pd.DataFrame, widths: list[float], small: ParagraphStyle, header_bg: str = "#dae8f6") -> Table:
    data = [[p(col, small) for col in df.columns]]
    for _, row in df.iterrows():
        data.append([p(str(row[col]), small) for col in df.columns])
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#19324f")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c7d9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_pdf(overall: pd.DataFrame, conf: pd.DataFrame) -> None:
    title, body, section, small = build_styles()
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=landscape(A4),
        leftMargin=11 * mm,
        rightMargin=11 * mm,
        topMargin=9 * mm,
        bottomMargin=9 * mm,
    )
    story = []
    story.append(p("Task6 公平重跑结果与混淆矩阵", title))
    story.append(
        p(
            "这一页只看同一套 Task6 定义下的七个模型。Task6 定义为 A / AB / B1 / B2 / B3 / TC 六分类。",
            body,
        )
    )
    story.append(
        p(
            "我们这次主要想看两件事：第一，哪些模型整体结果更好；第二，各模型最容易把哪几类混在一起。",
            body,
        )
    )
    story.append(Spacer(1, 2.5 * mm))

    story.append(p("1. 七个模型总表", section))
    overall_view = overall.copy()
    for col in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]:
        overall_view[col] = overall_view[col].map(fmt)
    overall_view = overall_view.rename(
        columns={
            "model_name": "模型",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "macro_f1": "Macro-F1",
            "macro_auc": "Macro-AUC",
        }
    )
    story.append(
        pdf_table_from_df(
            overall_view[["模型", "Accuracy", "Balanced Accuracy", "Macro-F1", "Macro-AUC"]],
            widths=[74 * mm, 26 * mm, 36 * mm, 26 * mm, 26 * mm],
            small=small,
        )
    )
    story.append(Spacer(1, 3 * mm))

    story.append(p("2. 七个模型的混淆矩阵", section))
    story.append(Image(str(HEATMAP_PNG), width=270 * mm, height=155 * mm))
    story.append(Spacer(1, 2 * mm))

    story.append(p("3. 主要错误方向", section))
    conf_view = conf.rename(
        columns={
            "model_name": "模型",
            "top_error_1": "最常见错误1",
            "top_error_2": "最常见错误2",
            "top_error_3": "最常见错误3",
        }
    )
    story.append(
        pdf_table_from_df(
            conf_view[["模型", "最常见错误1", "最常见错误2", "最常见错误3"]],
            widths=[66 * mm, 60 * mm, 60 * mm, 60 * mm],
            small=small,
            header_bg="#e7f1e7",
        )
    )
    story.append(Spacer(1, 2.5 * mm))

    best_f1 = overall.sort_values(["macro_f1", "macro_auc"], ascending=[False, False]).iloc[0]
    best_auc = overall.sort_values(["macro_auc", "macro_f1"], ascending=[False, False]).iloc[0]
    story.append(p("4. 我们目前的判断", section))
    notes = [
        f"当前按 Macro-F1 最好的是 {best_f1['model_name']}，Macro-F1={fmt(best_f1['macro_f1'])}。",
        f"当前按 Macro-AUC 最好的是 {best_auc['model_name']}，Macro-AUC={fmt(best_auc['macro_auc'])}。",
        "这轮最主要的混淆，仍然集中在 A/AB、B1/B2、B2/B3、B2/TC 这几条边界上。",
        "双 DINO 的结果明显优于 SE-ResNeXt50，说明强表征在这个任务上更有优势。",
        "experience-aux 这条线能提升整体排序能力，但目前最好 F1 仍然来自 concat。",
    ]
    for note in notes:
        story.append(p(f"• {note}", body))

    doc.build(story)


def main() -> None:
    register_fonts()
    overall, conf = collect_results()
    draw_heatmap_grid(conf)
    write_stage_md(overall, conf)
    write_doctor_markdown(overall, conf)
    build_pdf(overall, conf)


if __name__ == "__main__":
    main()
