from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "汇报" / "Task56_confusion_report_2026-05-13"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_fonts() -> str:
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
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    except Exception:
        pass
    return chosen


def fmt(v: float) -> str:
    return f"{v:.4f}"


TASK5_LABELS = ["A/AB", "B1-3", "TC"]
TASK6_LABELS = ["A", "AB", "B1", "B2", "B3", "TC"]

TASK5_CM = np.array(
    [
        [30, 9, 1],
        [12, 43, 5],
        [3, 7, 10],
    ],
    dtype=int,
)

TASK6_GATE_CM = np.array(
    [
        [6, 4, 3, 3, 3, 1],
        [2, 15, 2, 1, 0, 0],
        [3, 2, 11, 3, 1, 0],
        [1, 1, 3, 7, 4, 4],
        [1, 1, 2, 5, 10, 1],
        [1, 1, 0, 4, 1, 13],
    ],
    dtype=int,
)


def load_task6_struct_aux() -> tuple[np.ndarray, dict[str, float]]:
    pred_path = ROOT / "outputs" / "task56_struct_aux_manual_task6_full_364_bax" / "oof_case_predictions_mean.csv"
    metrics_path = ROOT / "outputs" / "task56_struct_aux_manual_task6_full_364_bax" / "oof_metrics.csv"
    df = pd.read_csv(pred_path)
    cm = confusion_matrix(df["label_idx"], df["pred_idx"], labels=list(range(6)))
    metrics_df = pd.read_csv(metrics_path)
    mean_row = metrics_df.loc[metrics_df["aggregation"] == "mean"].iloc[0]
    return cm, {
        "accuracy": float(mean_row["accuracy"]),
        "macro_f1": float(mean_row["macro_f1"]),
        "macro_auc": float(mean_row["macro_auc"]),
    }


def top_confusions(cm: np.ndarray, labels: list[str], k: int = 6) -> list[str]:
    pairs: list[tuple[int, str]] = []
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append((count, f"{src} -> {dst}: {count}"))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [x[1] for x in pairs[:k]]


def draw_confusion(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    left = axes[0].imshow(cm, cmap="Blues")
    right = axes[1].imshow(norm, cmap="Oranges", vmin=0.0, vmax=max(0.35, float(norm.max())))

    titles = ["Count matrix", "Row-normalized matrix"]
    arrays = [cm, norm]
    is_norm_flags = [False, True]

    for ax, arr, sub_title, is_norm in zip(axes, arrays, titles, is_norm_flags):
        ax.set_title(sub_title, fontsize=12)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        threshold = 0.55 if is_norm else arr.max() * 0.55
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                text = f"{arr[i, j]:.2f}" if is_norm else f"{int(arr[i, j])}"
                color = "white" if arr[i, j] > threshold else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(left, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(right, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_markdown(struct_meta: dict[str, float], struct_cm: np.ndarray) -> str:
    lines: list[str] = []
    lines.append("# Task5 / Task6 混淆矩阵与错误分布简报")
    lines.append("")
    lines.append("日期：2026-05-13")
    lines.append("")
    lines.append("这份简报主要想让老师和医生直接看到三件事：")
    lines.append("")
    lines.append("1. 哪些类别最容易互相混淆。")
    lines.append("2. 不同阶段的模型更容易往哪一侧偏。")
    lines.append("3. 经验标签加进来以后，错误分布有没有往更合理的方向变化。")
    lines.append("")
    lines.append("## 1. 这次建议重点看的三组结果")
    lines.append("")
    lines.append("| 位置 | 模型 | Macro-F1 | Macro-AUC | 说明 |")
    lines.append("|---|---|---:|---:|---|")
    lines.append("| Task5 主结果 | DINOv2 vitl14 + whole | 0.6614 | 0.7959 | 当前三分类主结果 |")
    lines.append("| Task6 第一版正式最好结果 | vits14 + vitb14 特征拼接 + Task5 层级先验 | 0.5127 | 0.8095 | 我们第一次把六分类稳定推过 0.50 的版本 |")
    lines.append(f"| Task6 后续增强版 | 结构化经验标签辅助版 | {fmt(struct_meta['macro_f1'])} | {fmt(struct_meta['macro_auc'])} | 这版不是最终分数最高，但保留了完整 OOF 预测，能直接复原混淆矩阵，看经验标签把错误推向哪里 |")
    lines.append("")
    lines.append("我们建议这次先不要只给医生看“最后最好”的单一结果，而是按三层来看：")
    lines.append("")
    lines.append("- 先看当前三分类主结果。")
    lines.append("- 再看六分类的第一版正式最好结果。")
    lines.append("- 最后看经验标签增强后，错误分布怎么变化。")
    lines.append("")
    lines.append("如果后面还要补更早的论文复现基线，例如 SuRImage，我们建议单独放在附页。原因是那一阶段的任务定义更接近 Task3 / Task4，不完全等同于现在的 Task5 / Task6，直接混在一张图里不容易解释。")
    lines.append("")
    lines.append("## 2. Task5 主结果")
    lines.append("")
    lines.append("- 模型：DINOv2 vitl14 + whole")
    lines.append("- Macro-F1 = 0.6614")
    lines.append("- Macro-AUC = 0.7959")
    lines.append(f"- 主要错误：{'；'.join(top_confusions(TASK5_CM, TASK5_LABELS, 4))}")
    lines.append("")
    lines.append("Task5 现在已经比较稳了。它不是全局都难，而是问题主要集中在 TC 和 B1-3 的边界。A/AB 这一端相对更清楚。换句话说，这条任务线后面如果继续提高，重点不是重做整体任务，而是想办法把 TC 的召回率再抬一点。")
    lines.append("")
    lines.append("## 3. Task6 第一版正式最好结果")
    lines.append("")
    lines.append("- 模型：vits14 + vitb14 特征拼接 + Task5 层级先验")
    lines.append("- Macro-F1 = 0.5127")
    lines.append("- Macro-AUC = 0.8095")
    lines.append(f"- 主要错误：{'；'.join(top_confusions(TASK6_GATE_CM, TASK6_LABELS, 6))}")
    lines.append("")
    lines.append("这版最清楚地暴露出三条难边界：A 与 AB、B1 与 B2、B2 与 B3/TC。尤其 B2 是最典型的“边界中心类”，会同时向 B1、B3 和 TC 三侧扩散。")
    lines.append("")
    lines.append("## 4. Task6 经验标签增强版")
    lines.append("")
    lines.append("- 模型：结构化经验标签辅助版")
    lines.append(f"- Macro-F1 = {fmt(struct_meta['macro_f1'])}")
    lines.append(f"- Macro-AUC = {fmt(struct_meta['macro_auc'])}")
    lines.append(f"- 主要错误：{'；'.join(top_confusions(struct_cm, TASK6_LABELS, 6))}")
    lines.append("")
    lines.append("这版最重要的价值不是“它是不是分数最高”，而是能看出经验标签会把模型往哪里推。和第一版正式结果相比，我们可以看到一部分 TC / B2 方向的大错被压下来，但同时新的错误更多转向了 A / AB 和 B2 -> AB 这一侧。也就是说，经验标签确实在改变模型的判断偏好。")
    lines.append("")
    lines.append("## 5. 我们现在最想请医生看的边界")
    lines.append("")
    lines.append("1. A 与 AB：哪些图从肉眼上本来就很难分。")
    lines.append("2. B1 与 B2：哪些病例确实缺少清楚的切面或关键形态。")
    lines.append("3. B2 与 B3：这条连续谱本来就是最难边界之一。")
    lines.append("4. B2 / B3 与 TC：尤其是那些没有明显坏死、但病理是真正 TC 的病例。")
    lines.append("")
    lines.append("## 6. 我们目前的判断")
    lines.append("")
    lines.append("- 如果一张图总是被多个模型稳定错到同一侧，通常说明它本身就处在视觉边界上。")
    lines.append("- 如果经验标签加进来以后，错误从一条边界转移到另一条边界，说明模型确实在学经验规则，而不是随机波动。")
    lines.append("- 现阶段最需要的不是再补大量容易病例，而是补更多边界病例，尤其是 B2、TC 和 A/AB 边界样本。")
    lines.append("")
    return "\n".join(lines)


def build_pdf(struct_meta: dict[str, float], struct_cm: np.ndarray, pngs: list[Path], out_pdf: Path) -> None:
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    body_font = "STSong-Light"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNBody", parent=styles["BodyText"], fontName=body_font, fontSize=10.2, leading=14))
    styles.add(ParagraphStyle(name="CNHeading", parent=styles["Heading2"], fontName=body_font, fontSize=13, leading=16, spaceAfter=6))
    styles.add(ParagraphStyle(name="CNTitle", parent=styles["Title"], fontName=body_font, fontSize=18, leading=22, alignment=1))

    story = []
    story.append(Paragraph("Task5 / Task6 混淆矩阵与错误分布简报", styles["CNTitle"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("日期：2026-05-13", styles["CNBody"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "这份简报主要想让老师和医生直接看到三件事：哪些类别最容易互相混淆，不同阶段的模型更容易往哪一侧偏，以及经验标签加进来以后错误分布有没有发生变化。",
            styles["CNBody"],
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    overview = [
        ["位置", "模型", "Macro-F1", "Macro-AUC", "说明"],
        ["Task5 主结果", "DINOv2 vitl14 + whole", "0.6614", "0.7959", "当前三分类主结果"],
        ["Task6 第一版正式最好结果", "vits14 + vitb14 特征拼接 + Task5 层级先验", "0.5127", "0.8095", "第一次把六分类稳定推过 0.50"],
        ["Task6 后续增强版", "结构化经验标签辅助版", fmt(struct_meta["macro_f1"]), fmt(struct_meta["macro_auc"]), "这版能直接复原矩阵，用来看经验标签如何改变错误偏好"],
    ]
    table = Table(overview, colWidths=[2.8 * cm, 6.1 * cm, 2.0 * cm, 2.0 * cm, 4.1 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E8FB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "我们建议这次先不要只给医生看“最后最好”的单一结果，而是按三层来看：先看当前三分类主结果，再看六分类的第一版正式最好结果，最后看经验标签增强后错误分布怎么变化。更早的 SuRImage 复现基线建议单独放在附页，因为它更接近 Task3 / Task4，不完全等同于现在的 Task5 / Task6。",
            styles["CNBody"],
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    sections = [
        (
            "Task5 主结果",
            "这版最重要的信息是：Task5 已经不是全局都难，而是 TC 更容易被拉到 B1-3。A/AB 这一端相对更稳。",
            pngs[0],
        ),
        (
            "Task6 第一版正式最好结果",
            "这版最清楚地暴露出三条难边界：A 与 AB、B1 与 B2、B2 与 B3/TC。尤其 B2 是最典型的边界中心类。",
            pngs[1],
        ),
        (
            "Task6 经验标签增强版",
            "这版最值得看的不是分数，而是错误偏好如何变化：一部分 TC / B2 方向的大错被压下来了，但另一部分错误又转向 A / AB 和 B2 -> AB 这一侧。",
            pngs[2],
        ),
    ]
    for title, text, png in sections:
        story.append(Paragraph(title, styles["CNHeading"]))
        story.append(Paragraph(text, styles["CNBody"]))
        story.append(Spacer(1, 0.12 * cm))
        story.append(Image(str(png), width=17.3 * cm, height=7.2 * cm))
        story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("我们现在最想请医生重点看的边界", styles["CNHeading"]))
    for line in [
        "1. A 与 AB：哪些图从肉眼上本来就很难分。",
        "2. B1 与 B2：哪些病例确实缺少清楚的切面或关键形态。",
        "3. B2 与 B3：这条连续谱本来就是最难边界之一。",
        "4. B2 / B3 与 TC：尤其是那些没有明显坏死、但病理是真正 TC 的病例。",
    ]:
        story.append(Paragraph(line, styles["CNBody"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Paragraph(
            "如果一张图总是被多个模型稳定错到同一侧，我们更倾向于把它理解成真正的视觉边界病例，而不是简单的图像质量问题。",
            styles["CNBody"],
        )
    )
    doc.build(story)


def main() -> None:
    setup_fonts()
    struct_cm, struct_meta = load_task6_struct_aux()

    png1 = OUT_DIR / "task5_confusion.png"
    png2 = OUT_DIR / "task6_gate_confusion.png"
    png3 = OUT_DIR / "task6_struct_aux_confusion.png"

    draw_confusion(TASK5_CM, TASK5_LABELS, "Task5 current main result", png1)
    draw_confusion(TASK6_GATE_CM, TASK6_LABELS, "Task6 first formal best result", png2)
    draw_confusion(struct_cm, TASK6_LABELS, "Task6 structured experience-label assisted result", png3)

    md_path = OUT_DIR / "Task56混淆矩阵与错误分布简报_2026-05-13.md"
    md_path.write_text(build_markdown(struct_meta, struct_cm), encoding="utf-8")

    pdf_path = OUT_DIR / "Task56混淆矩阵与错误分布简报_2026-05-13.pdf"
    build_pdf(struct_meta, struct_cm, [png1, png2, png3], pdf_path)

    print(md_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
