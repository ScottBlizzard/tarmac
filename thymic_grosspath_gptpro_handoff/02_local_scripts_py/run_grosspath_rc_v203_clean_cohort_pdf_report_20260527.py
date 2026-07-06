from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v203_clean_cohort_pdf_report_20260527"
REPORT_DIR = ROOT / "汇报"
PDF_PATH = REPORT_DIR / "2026-05-27_Task7_clean_cohort_exclusion_analysis.pdf"
EXCLUSION_PRESENCE = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_exclusion_presence_in_current_task7.csv"
METRICS = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_clean_vs_full_policy_metrics.csv"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7Clean"


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * float(x):.2f}%"


def esc(x: object) -> str:
    return html.escape("" if pd.isna(x) else str(x))


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(esc(text), style)


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=FONT_NAME,
            fontSize=16,
            leading=21,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#16324f"),
            spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "HeadingCN",
            parent=base["Heading1"],
            fontName=FONT_NAME,
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#12395b"),
            spaceBefore=2 * mm,
            spaceAfter=1 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.7,
            leading=12,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=0.8 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.0,
            leading=8.8,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
        ),
        "th": ParagraphStyle(
            "TableHeaderCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.2,
            alignment=TA_CENTER,
            wordWrap="CJK",
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "TableCellCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.2,
            alignment=TA_CENTER,
            wordWrap="CJK",
            textColor=colors.HexColor("#111827"),
        ),
    }


def make_table(data: list[list[object]], widths: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for i, row in enumerate(data):
        style = styles["th"] if i == 0 else styles["td"]
        rows.append([Paragraph(esc(cell), style) for cell in row])
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
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
    return table


def rows_for_metrics(metrics: pd.DataFrame) -> list[list[object]]:
    keep = metrics.loc[
        metrics["policy"].eq("v195_stable_agreement_release")
        & metrics["cohort"].isin(["full_sensitivity", "clean_primary"])
        & metrics["scope"].isin(["old_data", "third_batch", "strict_external", "all_domains"])
    ].copy()
    order = {"clean_primary": 0, "full_sensitivity": 1}
    scope_order = {"old_data": 0, "third_batch": 1, "strict_external": 2, "all_domains": 3}
    keep["cohort_order"] = keep["cohort"].map(order)
    keep["scope_order"] = keep["scope"].map(scope_order)
    keep = keep.sort_values(["cohort_order", "scope_order"])
    rows = [["队列", "范围", "n", "低危/高危", "Acc", "BAcc", "F1", "FN/FP", "复核/拒识", "自动错误"]]
    cohort_name = {"clean_primary": "clean 主分析", "full_sensitivity": "full 敏感性"}
    scope_name = {
        "old_data": "旧数据",
        "third_batch": "第三批",
        "strict_external": "严格外部",
        "all_domains": "总体",
    }
    for _, r in keep.iterrows():
        rows.append(
            [
                cohort_name[r["cohort"]],
                scope_name[r["scope"]],
                int(r["n"]),
                f"{int(r['low_risk_n'])}/{int(r['high_risk_n'])}",
                pct(r["accuracy"]),
                pct(r["balanced_accuracy"]),
                pct(r["f1"]),
                f"{int(r['fn'])}/{int(r['fp'])}",
                pct(r["review_or_reject_rate"]),
                f"{int(r['auto_error_n'])} ({pct(r['auto_error_rate'])})",
            ]
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    register_font()
    styles = build_styles()
    presence = pd.read_csv(EXCLUSION_PRESENCE)
    metrics = pd.read_csv(METRICS)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=11 * mm,
        bottomMargin=10 * mm,
    )
    story = [
        Paragraph("Task7 Clean Cohort 剔除病例与结果复算说明", styles["title"]),
        Paragraph("2026-05-27；主分析采用 clean cohort，完整队列作为 sensitivity analysis", styles["subtitle"]),
    ]
    story += [
        Paragraph("一、处理原则", styles["h1"]),
        p("我们不物理删除原始数据，而是固定生成 clean primary cohort 与 full sensitivity cohort 两套视图。主文建议报告 clean cohort；补充材料报告 full cohort，以说明结论对特殊组织学/混杂病例剔除不敏感。", styles["body"]),
    ]

    exclusion_rows = [["病例号", "医生备注", "当前主流程状态", "模型状态", "处理"]]
    for _, r in presence.iterrows():
        if bool(r["in_current_task7_main"]):
            model_state = (
                f"{r['domain']}；{r['task_l6_label']}；"
                f"预测={'高危' if int(float(r['final_pred'])) == 1 else '低危'}；"
                f"{'自动放行' if str(r['adaptive_auto_decision']).lower() == 'true' else '复核池'}"
            )
            status = "命中当前699例"
        else:
            model_state = "当前主流程未纳入"
            status = "未命中当前699例"
        exclusion_rows.append([r["original_case_id"], r["doctor_note"], status, model_state, "clean 主分析剔除；full 敏感性保留"])
    story += [
        Paragraph("二、固定剔除清单", styles["h1"]),
        make_table(exclusion_rows, [23 * mm, 28 * mm, 34 * mm, 70 * mm, 56 * mm], styles),
        Spacer(1, 1.5 * mm),
    ]

    story += [
        Paragraph("三、v195 当前主推流程：clean vs full", styles["h1"]),
        p("v195 是当前效率增强主推流程：在 v185 风险控制基础上加入稳定高置信一致放行。下表显示，剔除医生指出的特殊病例后，主结果基本不变。", styles["body"]),
        make_table(rows_for_metrics(metrics), [23 * mm, 23 * mm, 13 * mm, 23 * mm, 18 * mm, 18 * mm, 18 * mm, 16 * mm, 23 * mm, 24 * mm], styles),
    ]

    clean_all = metrics.loc[
        metrics["policy"].eq("v195_stable_agreement_release")
        & metrics["cohort"].eq("clean_primary")
        & metrics["scope"].eq("all_domains")
    ].iloc[0]
    story += [
        Paragraph("四、结论边界", styles["h1"]),
        p(
            f"clean cohort 总体 n={int(clean_all['n'])}，BAcc {pct(clean_all['balanced_accuracy'])}，"
            f"Acc {pct(clean_all['accuracy'])}，FN={int(clean_all['fn'])}，FP={int(clean_all['fp'])}，"
            f"复核/拒识 {pct(clean_all['review_or_reject_rate'])}。当前主流程中实际命中的剔除病例只有 2205101 和 2307206，"
            "且二者原本均被模型判为高危正确，因此 clean cohort 的作用是规范队列定义，不是人为提高分数。",
            styles["body"],
        ),
        p("建议向医生说明：这些病例应从 primary clean cohort 中剔除；如果医生希望强调真实复杂场景，可在补充材料保留 full cohort sensitivity analysis。", styles["body"]),
    ]

    doc.build(story)
    print(f"[v203] wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
