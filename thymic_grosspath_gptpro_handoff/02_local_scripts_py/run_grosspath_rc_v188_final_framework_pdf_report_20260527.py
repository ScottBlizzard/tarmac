from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v188_final_framework_pdf_report_20260527"
REPORT_DIR = ROOT / "汇报"
V186_TABLE = ROOT / "outputs" / "grosspath_rc_v186_final_framework_table_20260527" / "v186_final_framework_operating_table.csv"
V186_CLAIMS = ROOT / "outputs" / "grosspath_rc_v186_final_framework_table_20260527" / "v186_final_framework_claim_table.csv"
V187_CI = ROOT / "outputs" / "grosspath_rc_v187_release_safety_ci_planning_20260527" / "v187_auto_decision_error_ci.csv"
V187_PLAN = ROOT / "outputs" / "grosspath_rc_v187_release_safety_ci_planning_20260527" / "v187_prospective_auto_decision_sample_size.csv"


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):.{digits}f}%"


def register_fonts() -> tuple[str, str]:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No usable Chinese TTF/TTC font found.")
    pdfmetrics.registerFont(TTFont("ReportCN", str(font_path)))
    pdfmetrics.registerFont(TTFont("ReportCN-Bold", str(font_path)))
    return "ReportCN", "ReportCN-Bold"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), style)


def make_table(data: list[list[object]], widths: list[float], font: str, bold: str, left_cols: set[int] | None = None) -> Table:
    left_cols = left_cols or {0}
    head = ParagraphStyle("table_head", fontName=bold, fontSize=7.2, leading=9.2, alignment=TA_CENTER, textColor=colors.white)
    cell = ParagraphStyle("table_cell", fontName=font, fontSize=6.9, leading=8.8, alignment=TA_CENTER)
    left = ParagraphStyle("table_left", fontName=font, fontSize=6.9, leading=8.8, alignment=TA_LEFT)
    rows = []
    for i, row in enumerate(data):
        out = []
        for j, item in enumerate(row):
            out.append(para(item, head if i == 0 else (left if j in left_cols else cell)))
        rows.append(out)
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return t


def label_module(name: str) -> str:
    return {
        "v185 unlabeled shift-adaptive workflow": "v185 无标签偏移自适应流程",
        "v182 fixed stable efficiency workflow": "v182 固定稳定效率版",
        "v118 fixed high-safety fallback": "v118 高安全兜底版",
        "v180 greedy per-fold image-agreement audit": "v180 贪心释放负向审计",
        "30% low-confidence selective review": "30%低置信度复核基线",
        "30% direction-aware image router": "30%方向感知复核",
        "v173 aggressive image-only review corrector": "v173 激进图像复核器",
        "v174 disagreement flip": "v174 不一致翻转",
        "v175 error-enriched flip-risk": "v175 错误富集翻转风险",
    }.get(name, name)


def operating_table_rows(table: pd.DataFrame) -> list[list[object]]:
    rows = [["模块", "定位", "范围", "自动判读", "复核/拒识", "BAcc", "FN", "FP"]]
    keep = [
        "v185 unlabeled shift-adaptive workflow",
        "v182 fixed stable efficiency workflow",
        "v118 fixed high-safety fallback",
        "v180 greedy per-fold image-agreement audit",
        "30% low-confidence selective review",
        "30% direction-aware image router",
        "v173 aggressive image-only review corrector",
        "v174 disagreement flip",
        "v175 error-enriched flip-risk",
    ]
    scope_order = {"all_domains": 0, "strict_external": 1}
    sub = table.loc[table["module"].isin(keep) & table["scope"].isin(scope_order)].copy()
    sub["scope_rank"] = sub["scope"].map(scope_order)
    sub = sub.sort_values(["paper_order", "scope_rank"])
    for _, r in sub.iterrows():
        rows.append(
            [
                label_module(str(r["module"])),
                str(r["tier"]),
                "全域" if r["scope"] == "all_domains" else "严格外部",
                pct(r["auto_decision_rate"], 1),
                pct(r["remaining_review_or_reject_rate"], 1),
                pct(r["balanced_accuracy"], 2),
                int(r["fn"]),
                int(r["fp"]),
            ]
        )
    return rows


def ci_rows(ci: pd.DataFrame) -> list[list[object]]:
    rows = [["策略", "范围", "自动判读例数", "自动错误", "观测错误率", "Wilson95上界"]]
    keep = ["fixed_v118_high_safety", "fixed_v182_stable_release", "v185_unlabeled_shift_adaptive"]
    label = {
        "fixed_v118_high_safety": "v118高安全兜底",
        "fixed_v182_stable_release": "v182效率版",
        "v185_unlabeled_shift_adaptive": "v185自适应流程",
    }
    sub = ci.loc[ci["policy"].isin(keep) & ci["scope"].isin(["all_domains", "strict_external"])].copy()
    sub["policy_rank"] = sub["policy"].map({k: i for i, k in enumerate(keep)})
    sub["scope_rank"] = sub["scope"].map({"all_domains": 0, "strict_external": 1})
    for _, r in sub.sort_values(["policy_rank", "scope_rank"]).iterrows():
        rows.append(
            [
                label[str(r["policy"])],
                "全域" if r["scope"] == "all_domains" else "严格外部",
                int(r["auto_decision_n"]),
                int(r["auto_error_n"]),
                pct(r["auto_error_rate"], 2),
                pct(r["wilson95_high"], 2),
            ]
        )
    return rows


def claim_rows(claims: pd.DataFrame) -> list[list[object]]:
    rows = [["主张", "状态", "证据边界"]]
    for _, r in claims.iterrows():
        rows.append([r["claim"], r["status"], r["evidence"]])
    return rows


def build_pdf(pdf_path: Path) -> None:
    font, bold = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=bold, fontSize=19, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#102A43"))
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName=font, fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#486581"))
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=bold, fontSize=13.5, leading=18, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#243B53"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font, fontSize=9.2, leading=13.8, spaceAfter=5, textColor=colors.HexColor("#1F2933"))
    callout = ParagraphStyle("callout", parent=body, fontName=bold, fontSize=9.6, leading=14.4, leftIndent=8, rightIndent=8, borderPadding=8, backColor=colors.HexColor("#E6F6FF"), textColor=colors.HexColor("#102A43"))

    table = pd.read_csv(V186_TABLE)
    claims = pd.read_csv(V186_CLAIMS)
    ci = pd.read_csv(V187_CI)
    plan = pd.read_csv(V187_PLAN)

    all_rows = table.loc[table["scope"].eq("all_domains")]
    adaptive = all_rows.loc[all_rows["module"].eq("v185 unlabeled shift-adaptive workflow")].iloc[0]
    fixed182 = all_rows.loc[all_rows["module"].eq("v182 fixed stable efficiency workflow")].iloc[0]
    fixed118 = all_rows.loc[all_rows["module"].eq("v118 fixed high-safety fallback")].iloc[0]
    adaptive_ci = ci.loc[ci["policy"].eq("v185_unlabeled_shift_adaptive") & ci["scope"].eq("all_domains")].iloc[0]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    story = [
        para("Task7 风险可控跨域诊断框架阶段汇报", title),
        para("稳定释放、无标签偏移门控与自动判读安全边界 | 2026-05-27", subtitle),
        Spacer(1, 0.28 * cm),
        para(
            f"当前主线已经从“单个分类器”收束为风险可控的选择性诊断框架。推荐部署流程为 v185 无标签偏移自适应流程：全域 BAcc {pct(adaptive['balanced_accuracy'])}，复核/拒识 {pct(adaptive['remaining_review_or_reject_rate'])}，FN={int(adaptive['fn'])}，FP={int(adaptive['fp'])}。"
            f"标准效率版 fixed v182 复核/拒识 {pct(fixed182['remaining_review_or_reject_rate'])}；高安全兜底 fixed v118 复核/拒识 {pct(fixed118['remaining_review_or_reject_rate'])}。",
            callout,
        ),
        para("一、当前 operating points", h1),
        make_table(
            operating_table_rows(table),
            [3.0 * cm, 3.0 * cm, 1.3 * cm, 1.35 * cm, 1.35 * cm, 1.35 * cm, 0.75 * cm, 0.75 * cm],
            font,
            bold,
            {0, 1},
        ),
        para("二、我们现在的框架怎么解释", h1),
        para(
            "第一阶段仍是图像主分类器和多视角概率输出；第二阶段用 two-signal scorecard 和 image-agreement 规则做稳定释放；第三阶段不是强行自动翻转，而是把不稳定病例保留在复核/拒识中；第四阶段用无标签 severe-shift gate 判断批次是否像严格外部偏移，若触发则从效率版回退到高安全版。这样做的重点不是追求最低复核率，而是在跨域风险出现时主动收紧自动判读。",
            body,
        ),
        para(
            "v180 的负向审计很关键：如果每折贪心选择更宽的 release 规则，复核率可以降到 48.21%，但会释放 3 个错误。v182 改成固定稳定规则后，复核率为 56.08%，释放错误为 0。这说明我们的 release 不是任意扩张，而是受稳定性约束控制。",
            body,
        ),
        PageBreak(),
        para("三、自动判读安全边界", h1),
        make_table(
            ci_rows(ci),
            [3.0 * cm, 1.6 * cm, 2.1 * cm, 1.5 * cm, 2.0 * cm, 2.0 * cm],
            font,
            bold,
            {0},
        ),
        Spacer(1, 0.2 * cm),
        para(
            f"这里需要特别区分两个概念：v182/v185 的新增 release 错误为 0，但全部自动判读集合里仍有 1 个残余 FN。v185 全域自动判读 {int(adaptive_ci['auto_decision_n'])} 例，错误 {int(adaptive_ci['auto_error_n'])} 例，Wilson95 上界为 {pct(adaptive_ci['wilson95_high'])}。严格外部自动判读例数少，因此即使 0 错误，置信区间仍然较宽。",
            body,
        ),
        para("四、前瞻验证需要多少自动判读病例", h1),
        make_table(
            [["目标Wilson95上界", "若0错误所需自动判读例数", "若1错误所需自动判读例数"]]
            + [[pct(r["target_wilson95_upper"], 1), int(r["auto_decision_n_needed_if_zero_errors"]), int(r["auto_decision_n_needed_if_one_error"])] for _, r in plan.iterrows()],
            [3.2 * cm, 4.1 * cm, 4.1 * cm],
            font,
            bold,
            {0},
        ),
        para("五、论文主张边界", h1),
        make_table(claim_rows(claims), [4.5 * cm, 2.5 * cm, 8.5 * cm], font, bold, {0, 2}),
        para("六、当前建议写法", h1),
        para(
            "我们建议把论文主线写成：面向跨域大体病理图像的风险可控选择性诊断框架。核心贡献包括多视角风险评分卡、固定稳定释放规则、无标签偏移门控、以及负向审计证明不稳定释放和自动翻转不能直接部署。自动翻转纠偏目前不是成熟模块，应作为后续需要更多真实残余错误样本或医生结构化标签支持的方向。",
            callout,
        ),
    ]
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / "2026-05-27_Task7风险可控跨域诊断框架最终阶段汇报.pdf"
    build_pdf(pdf_path)
    report = {"pdf": str(pdf_path), "source": str(OUT_DIR)}
    (OUT_DIR / "v188_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v188] wrote {pdf_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
