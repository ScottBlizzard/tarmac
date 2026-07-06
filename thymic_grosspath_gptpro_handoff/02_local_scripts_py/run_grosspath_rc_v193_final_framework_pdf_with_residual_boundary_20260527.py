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


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v193_final_framework_pdf_with_residual_boundary_20260527"
REPORT_DIR = ROOT / "汇报"
V192_TABLE = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527" / "v192_final_framework_operating_table.csv"
V192_CLAIMS = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527" / "v192_final_framework_claim_table.csv"
V192_RESIDUAL = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527" / "v192_residual_auto_error_case.csv"
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


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), style)


def make_table(data: list[list[object]], widths: list[float], font: str, bold: str, left_cols: set[int] | None = None) -> Table:
    left_cols = left_cols or {0}
    head = ParagraphStyle("head", fontName=bold, fontSize=7.0, leading=9.0, alignment=TA_CENTER, textColor=colors.white)
    cell = ParagraphStyle("cell", fontName=font, fontSize=6.65, leading=8.4, alignment=TA_CENTER)
    left = ParagraphStyle("left", fontName=font, fontSize=6.65, leading=8.4, alignment=TA_LEFT)
    out = []
    for i, row in enumerate(data):
        converted = []
        for j, item in enumerate(row):
            converted.append(p(item, head if i == 0 else (left if j in left_cols else cell)))
        out.append(converted)
    table = Table(out, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
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
    return table


def module_label(x: str) -> str:
    return {
        "v185 unlabeled shift-adaptive workflow": "v185 无标签偏移自适应流程",
        "v182 fixed stable efficiency workflow": "v182 固定稳定效率版",
        "v118 fixed high-safety fallback": "v118 高安全兜底版",
        "v180 greedy per-fold image-agreement audit": "v180 贪心释放负向审计",
        "v190 fold-wise probability FN sentinel": "v190 概率FN哨兵",
        "v191 learned DINO FN-risk sentinel": "v191 DINO FN风险哨兵",
        "30% low-confidence selective review": "30%低置信度复核基线",
        "30% direction-aware image router": "30%方向感知复核",
        "v173 aggressive image-only review corrector": "v173 激进图像复核器",
        "v174 disagreement flip": "v174 不一致翻转",
        "v175 error-enriched flip-risk": "v175 错误富集翻转风险",
    }.get(x, x)


def tier_label(x: str) -> str:
    if "Recommended deployment" in x:
        return "推荐部署"
    if "Standard efficiency" in x:
        return "标准效率"
    if "High-safety" in x:
        return "高安全兜底"
    if "Negative" in x:
        return "负向审计"
    if "Comparator" in x:
        return "对照"
    if "Candidate" in x:
        return "候选/边界"
    return x


def operating_rows(table: pd.DataFrame) -> list[list[object]]:
    rows = [["模块", "定位", "范围", "复核/拒识", "BAcc", "FN", "FP", "补充"]]
    keep = [
        "v185 unlabeled shift-adaptive workflow",
        "v182 fixed stable efficiency workflow",
        "v118 fixed high-safety fallback",
        "v180 greedy per-fold image-agreement audit",
        "v190 fold-wise probability FN sentinel",
        "v191 learned DINO FN-risk sentinel",
        "30% low-confidence selective review",
        "30% direction-aware image router",
        "v173 aggressive image-only review corrector",
        "v174 disagreement flip",
        "v175 error-enriched flip-risk",
    ]
    scopes = {"all_domains": "全域", "strict_external": "严格外部"}
    sub = table.loc[table["module"].isin(keep) & table["scope"].isin(scopes)].copy()
    sub = sub.sort_values(["paper_order", "scope"])
    for _, r in sub.iterrows():
        extra = ""
        if pd.notna(r.get("sentinel_rescued_fn_n", None)) and str(r.get("sentinel_rescued_fn_n", "")) != "":
            extra = f"救回FN {int(float(r['sentinel_rescued_fn_n']))}"
        elif pd.notna(r.get("released_error_n", None)) and str(r.get("released_error_n", "")) not in ["", "nan"]:
            extra = f"释放错 {int(float(r['released_error_n']))}"
        rows.append(
            [
                module_label(str(r["module"])),
                tier_label(str(r["tier"])),
                scopes[str(r["scope"])],
                pct(r["remaining_review_or_reject_rate"], 1),
                pct(r["balanced_accuracy"], 2),
                int(r["fn"]),
                int(r["fp"]),
                extra,
            ]
        )
    return rows


def ci_rows(ci: pd.DataFrame) -> list[list[object]]:
    rows = [["策略", "范围", "自动判读", "自动错误", "观测错误率", "Wilson95上界"]]
    labels = {
        "fixed_v118_high_safety": "v118高安全",
        "fixed_v182_stable_release": "v182效率",
        "v185_unlabeled_shift_adaptive": "v185自适应",
    }
    sub = ci.loc[ci["policy"].isin(labels) & ci["scope"].isin(["all_domains", "strict_external"])].copy()
    sub["rank"] = sub["policy"].map({k: i for i, k in enumerate(labels)})
    sub["scope_rank"] = sub["scope"].map({"all_domains": 0, "strict_external": 1})
    for _, r in sub.sort_values(["rank", "scope_rank"]).iterrows():
        rows.append(
            [
                labels[str(r["policy"])],
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
    for _, r in claims.tail(6).iterrows():
        rows.append([r["claim"], r["status"], r["evidence"]])
    return rows


def build_pdf(pdf_path: Path) -> None:
    font, bold = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=bold, fontSize=19, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#102A43"))
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName=font, fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#486581"))
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=bold, fontSize=13.2, leading=17, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#243B53"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font, fontSize=9.1, leading=13.6, spaceAfter=5, textColor=colors.HexColor("#1F2933"))
    callout = ParagraphStyle("callout", parent=body, fontName=bold, fontSize=9.5, leading=14.2, leftIndent=8, rightIndent=8, borderPadding=8, backColor=colors.HexColor("#E6F6FF"), textColor=colors.HexColor("#102A43"))

    table = pd.read_csv(V192_TABLE)
    claims = pd.read_csv(V192_CLAIMS)
    residual = pd.read_csv(V192_RESIDUAL, dtype={"original_case_id": str})
    ci = pd.read_csv(V187_CI)
    plan = pd.read_csv(V187_PLAN)

    all_rows = table.loc[table["scope"].eq("all_domains")]
    adaptive = all_rows.loc[all_rows["module"].eq("v185 unlabeled shift-adaptive workflow")].iloc[0]
    v190 = all_rows.loc[all_rows["module"].eq("v190 fold-wise probability FN sentinel")].iloc[0]
    v191 = all_rows.loc[all_rows["module"].eq("v191 learned DINO FN-risk sentinel")].iloc[0]
    res = residual.iloc[0]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    story = [
        p("Task7 风险可控跨域诊断框架最终阶段汇报", title),
        p("加入残余FN边界、稳定释放和无标签偏移门控 | 2026-05-27", subtitle),
        Spacer(1, 0.25 * cm),
        p(
            f"当前推荐部署流程仍为 v185 无标签偏移自适应流程：全域 BAcc {pct(adaptive['balanced_accuracy'])}，复核/拒识 {pct(adaptive['remaining_review_or_reject_rate'])}，FN={int(adaptive['fn'])}，FP={int(adaptive['fp'])}。"
            f"唯一残余自动错误为 {res['original_case_id']}（{res['task_l6_label']}，高危漏判低危）。v190/v191 证明这个病例不能被无泄露的简单哨兵或轻量DINO风险模型稳定救回。",
            callout,
        ),
        p("一、最终 operating points", h1),
        make_table(
            operating_rows(table),
            [2.75 * cm, 1.85 * cm, 1.25 * cm, 1.35 * cm, 1.25 * cm, 0.65 * cm, 0.65 * cm, 1.65 * cm],
            font,
            bold,
            {0, 1, 7},
        ),
        p("二、残余FN边界", h1),
        p(
            f"残余自动错误为 {res['original_case_id']}，标签 {res['task_l6_label']}，模型自动判为低危，prob_mean_core={float(res['prob_mean_core']):.3f}。"
            "我们做了两类补救尝试：v190 用可解释概率规则做 fold-wise 哨兵；v191 用 DINO PCA + 概率特征训练 FN-risk sentinel。"
            f"结果是 v190 额外复核 {int(v190['additional_sentinel_review_n'])} 例、救回FN {int(v190['sentinel_rescued_fn_n'])}；v191 额外复核 {int(v191['additional_sentinel_review_n'])} 例、救回FN {int(v191['sentinel_rescued_fn_n'])}。"
            "full-fit 后验规则能抓住 2516531，但 fold-wise 排除本折后不能稳定复现，因此不能写成正式规则。",
            body,
        ),
        PageBreak(),
        p("三、自动判读安全边界", h1),
        make_table(ci_rows(ci), [2.5 * cm, 1.5 * cm, 1.8 * cm, 1.5 * cm, 1.9 * cm, 1.9 * cm], font, bold, {0}),
        Spacer(1, 0.15 * cm),
        p(
            "这里必须区分“新增释放错误”和“全部自动判读错误”：v182/v185 的新增 release 错误为 0，但全部自动判读集合中仍有 1 个残余FN。"
            "因此论文中应报告 Wilson 置信区间，而不是只写 0 release error。",
            body,
        ),
        p("四、前瞻验证样本量", h1),
        make_table(
            [["目标Wilson95上界", "0错误所需自动判读例数", "1错误所需自动判读例数"]]
            + [[pct(r["target_wilson95_upper"], 1), int(r["auto_decision_n_needed_if_zero_errors"]), int(r["auto_decision_n_needed_if_one_error"])] for _, r in plan.iterrows()],
            [3.0 * cm, 4.0 * cm, 4.0 * cm],
            font,
            bold,
            {0},
        ),
        p("五、主张边界", h1),
        make_table(claim_rows(claims), [4.5 * cm, 2.4 * cm, 8.5 * cm], font, bold, {0, 2}),
        p("六、当前建议写法", h1),
        p(
            "建议把论文主线写成风险可控、偏移自适应的选择性诊断框架。强主张包括：固定稳定释放、无标签偏移门控、严格的负向审计和安全置信边界。"
            "不应主张自动翻转纠偏已经成熟，也不应围绕 2516531 后验制定规则。这个残余病例更适合作为医生回看和前瞻扩展数据中的重点风险类型。",
            callout,
        ),
    ]
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / "2026-05-27_Task7风险可控跨域诊断框架最终阶段汇报_含残余边界.pdf"
    build_pdf(pdf_path)
    report = {"pdf": str(pdf_path), "source": str(OUT_DIR)}
    (OUT_DIR / "v193_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v193] wrote {pdf_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
