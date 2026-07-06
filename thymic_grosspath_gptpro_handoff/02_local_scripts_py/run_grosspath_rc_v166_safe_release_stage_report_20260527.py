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
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v166_safe_release_stage_report_20260527"
REPORT_DIR = ROOT / "汇报"
V165_DIR = ROOT / "outputs" / "grosspath_rc_v165_safe_release_evidence_pack_20260527"
V165_MAIN = V165_DIR / "v165_main_operating_results.csv"
V165_ABLATION = V165_DIR / "v165_safe_release_ablation.csv"
V165_RULES = V165_DIR / "v165_safe_release_rules.csv"
V165_REMAINING = V165_DIR / "v165_remaining_auto_errors_after_safe_release.csv"
V165_CLAIMS = V165_DIR / "v165_claim_evidence_map.csv"
V162_SELECTED = V165_DIR / "v165_frontier_constraint_selection.csv"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def register_fonts() -> tuple[str, str]:
    # ReportLab's TTFont does not support CFF/PostScript OTF outlines.
    regular_candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    regular = next((p for p in regular_candidates if p.exists()), None)
    if regular is None:
        raise FileNotFoundError("No usable Chinese TTF/TTC font found.")
    bold = regular
    pdfmetrics.registerFont(TTFont("ReportCN", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportCN-Bold", str(bold)))
    return "ReportCN", "ReportCN-Bold"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def table(data: list[list[object]], widths: list[float], font: str, bold: str) -> Table:
    converted = []
    body = ParagraphStyle("cell", fontName=font, fontSize=7.2, leading=9.2, alignment=TA_CENTER)
    body_left = ParagraphStyle("cell_left", fontName=font, fontSize=7.2, leading=9.2, alignment=TA_LEFT)
    head = ParagraphStyle("head", fontName=bold, fontSize=7.5, leading=9.5, alignment=TA_CENTER, textColor=colors.white)
    for i, row in enumerate(data):
        out = []
        for j, item in enumerate(row):
            style = head if i == 0 else (body_left if j == 0 else body)
            out.append(p(str(item), style))
        converted.append(out)
    t = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return t


def main_result_table(main: pd.DataFrame) -> list[list[object]]:
    keep = [
        "30% low-confidence selective review baseline",
        "30% image-direction router",
        "v118 high-safety two-signal scorecard",
        "v161 safe-release high-safety scorecard",
    ]
    rows = [["方案", "数据域", "自动放行", "复核/拒识", "BAcc", "FN", "FP", "证据状态"]]
    for _, r in main.loc[main["evidence_line"].isin(keep) & main["eval_domain"].isin(["all_domains", "strict_external"])].iterrows():
        name = {
            "30% low-confidence selective review baseline": "低置信度复核基线",
            "30% image-direction router": "方向感知复核",
            "v118 high-safety two-signal scorecard": "高安全评分卡",
            "v161 safe-release high-safety scorecard": "安全释放评分卡",
        }[r["evidence_line"]]
        domain = "全域" if r["eval_domain"] == "all_domains" else "严格外部"
        rows.append(
            [
                name,
                domain,
                pct(r["auto_pass_rate"]),
                pct(r["review_or_reject_rate"]),
                pct(r["balanced_accuracy"], 2),
                int(r["fn"]),
                int(r["fp"]),
                {
                    "baseline": "基线",
                    "router candidate": "候选",
                    "high-safety baseline": "高安全基线",
                    "auto-correction candidate": "纠偏候选",
                    "current high-safety efficiency candidate": "当前主推候选",
                }.get(str(r["status"]), r["status"]),
            ]
        )
    return rows


def short_constraint(name: str) -> str:
    return {
        "max_30pct_review_best_all_bacc": "≤30%复核",
        "max_40pct_review_best_all_bacc": "≤40%复核",
        "max_60pct_review_best_all_bacc": "≤60%复核",
        "max_60pct_review_best_min_domain_bacc": "≤60%最差域优先",
        "all_bacc_at_least_95_min_review": "全域BAcc≥95%",
        "min_domain_bacc_at_least_95_min_review": "各域BAcc≥95%",
        "near_zero_error_high_safety": "近零错误",
    }.get(name, name)


def short_point(name: str) -> str:
    if name == "v161_safe_release_scorecard":
        return "安全释放评分卡"
    if name.startswith("baseline_low_conf_dev_selected"):
        return "低置信度复核"
    if name.startswith("dev_stable_router_all_domains"):
        return "稳定方向路由"
    if name.startswith("shift_aware_image_directional_candidate"):
        return "图像方向路由"
    return name


def claim_title_cn(claim: str) -> str:
    return {
        "The method should be framed as a risk-controlled workflow rather than a single classifier.": "方法应写成风险可控流程，而不是单一分类器。",
        "Safe-release reduces review burden while keeping the current high-safety error profile.": "安全释放在保持高安全错误结构的同时降低复核负担。",
        "Multi-domain internal constraints are necessary.": "多内部域共同约束是必要条件。",
        "Unlabeled severe-shift gate is not a single fragile statistic.": "无标签 severe-shift 门控不是单一脆弱指标。",
        "Direction-aware routing improves over confidence-only review under strict external shift.": "严格外部偏移下，方向感知路由优于单纯低置信度复核。",
    }.get(claim, claim)


def ablation_table(ablation: pd.DataFrame) -> list[list[object]]:
    rows = [["策略", "释放例数", "释放错误", "复核/拒识", "BAcc", "FN", "FP"]]
    order = [
        "old_only_selected",
        "third_only_selected",
        "leave_domain_intersection",
        "pooled_old_third_zero_error_v161",
        "balanced_zero_error",
    ]
    for _, r in ablation.set_index("strategy").loc[order].reset_index().iterrows():
        label = {
            "old_only_selected": "仅 old 选规则",
            "third_only_selected": "仅 third 选规则",
            "leave_domain_intersection": "留域交集规则",
            "pooled_old_third_zero_error_v161": "old+third 零错误规则",
            "balanced_zero_error": "均衡零错误规则",
        }[r["strategy"]]
        rows.append(
            [
                label,
                int(r["release_n"]),
                int(r["released_error_n"]),
                pct(r["review_rate"]),
                pct(r["balanced_accuracy"], 2),
                int(r["fn"]),
                int(r["fp"]),
            ]
        )
    return rows


def build_pdf(pdf_path: Path) -> None:
    font, bold = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=bold, fontSize=20, leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#102A43"))
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName=font, fontSize=10.5, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#486581"))
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=bold, fontSize=14, leading=18, spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#243B53"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold, fontSize=11, leading=15, spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#334E68"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font, fontSize=9.4, leading=14, spaceAfter=6, textColor=colors.HexColor("#1F2933"))
    callout = ParagraphStyle("callout", parent=body, fontName=bold, fontSize=10, leading=15, leftIndent=8, rightIndent=8, borderPadding=8, backColor=colors.HexColor("#E6F6FF"), textColor=colors.HexColor("#102A43"))

    main = pd.read_csv(V165_MAIN)
    ablation = pd.read_csv(V165_ABLATION)
    rules = pd.read_csv(V165_RULES)
    remaining = pd.read_csv(V165_REMAINING)
    claims = pd.read_csv(V165_CLAIMS)
    frontier = pd.read_csv(V162_SELECTED)

    all_v161 = main.loc[
        main["evidence_line"].eq("v161 safe-release high-safety scorecard")
        & main["eval_domain"].eq("all_domains")
    ].iloc[0]
    strict_v161 = main.loc[
        main["evidence_line"].eq("v161 safe-release high-safety scorecard")
        & main["eval_domain"].eq("strict_external")
    ].iloc[0]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    story = [
        p("Task7 大体病理图像二分类阶段汇报", title),
        p("安全释放评分卡与跨域风险控制框架 | 2026-05-27", subtitle),
        Spacer(1, 0.4 * cm),
        p(
            f"当前主推候选流程为 v161 safe-release high-safety scorecard：全域 BAcc {pct(all_v161['balanced_accuracy'], 2)}，"
            f"复核/拒识 {pct(all_v161['review_or_reject_rate'])}，自动放行 {pct(all_v161['auto_pass_rate'])}，"
            f"FN={int(all_v161['fn'])}、FP={int(all_v161['fp'])}。严格外部集 BAcc {pct(strict_v161['balanced_accuracy'], 2)}，"
            f"FN={int(strict_v161['fn'])}、FP={int(strict_v161['fp'])}。",
            callout,
        ),
        p("一、当前主结果", h1),
        p("下表把普通低置信度复核、方向感知复核、高安全评分卡和安全释放评分卡放在同一口径下比较。核心不是只看准确率，而是同时报告自动放行和复核/拒识比例。", body),
        table(main_result_table(main), [3.6 * cm, 1.5 * cm, 1.7 * cm, 1.8 * cm, 1.5 * cm, 0.8 * cm, 0.8 * cm, 4.1 * cm], font, bold),
        p("二、安全释放规则", h1),
        p("safe-release 的作用是从原本需要复核的病例中，释放一部分两个信号均稳定、且内部多域没有释放错误的病例。", body),
    ]

    rule_rows = [["方向", "释放条件", "内部释放", "外部释放", "释放错误"]]
    for _, r in rules.iterrows():
        direction = "低危释放" if int(r["pred_side"]) == 0 else "高危释放"
        op = "≤" if int(r["pred_side"]) == 0 else "≥"
        cond = f"prob_mean_core {op} {float(r['prob_mean_core_threshold']):.4f}; wholecrop_prob {op} {float(r['wholecrop_prob_threshold']):.4f}; core_agree≥{int(r['core_agree_min'])}"
        rule_rows.append([direction, cond, int(r["internal_release_n"]), int(r["external_release_n"]), int(r["all_released_errors"])])
    story += [
        table(rule_rows, [1.8 * cm, 8.0 * cm, 1.7 * cm, 1.7 * cm, 1.6 * cm], font, bold),
        p("三、消融和边界", h1),
        p("单独使用 old_data 选出的释放规则过于宽松，虽然能大幅降低复核率，但会在 third_batch 释放错误。多内部域共同约束后，释放效率略低，但当前三域释放错误为 0。", body),
        table(ablation_table(ablation), [3.9 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 1.5 * cm, 0.8 * cm, 0.8 * cm], font, bold),
        PageBreak(),
        p("四、安全-效率前沿", h1),
        p("加入 v161 后，≤60% 复核约束下的最佳点由普通 low-confidence 方案变为 safe-release scorecard。说明我们不是单纯提高分类器，而是在风险控制框架里提高高安全流程的可部署性。", body),
    ]
    frontier_rows = [["约束", "选择方案", "复核/拒识", "自动放行", "BAcc", "FN", "FP"]]
    for _, r in frontier.iterrows():
        if r.get("selected", True) in [False, "False"]:
            continue
        frontier_rows.append([short_constraint(str(r["constraint"])), short_point(str(r["operating_point"])), pct(r["review_rate"]), pct(r["auto_pass_rate"]), pct(r["all_bacc"], 2), int(r["all_fn"]), int(r["all_fp"])])
    story += [
        table(frontier_rows, [4.1 * cm, 4.5 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 0.8 * cm, 0.8 * cm], font, bold),
        p("五、剩余错误", h1),
    ]
    if len(remaining) == 0:
        story.append(p("当前 safe-release 后没有自动错误。", body))
    else:
        err_rows = [["数据域", "病例号", "病理", "真值", "预测", "prob_mean_core", "wholecrop"]]
        for _, r in remaining.iterrows():
            err_rows.append([r["domain"], r["original_case_id"], r["task_l6_label"], int(r["label_idx"]), int(r["final_pred"]), f"{float(r['prob_mean_core']):.3f}", f"{float(r['wholecrop_prob']):.3f}"])
        story.append(table(err_rows, [2.0 * cm, 2.0 * cm, 1.5 * cm, 1.1 * cm, 1.1 * cm, 2.2 * cm, 2.0 * cm], font, bold))
        story.append(p("该病例是当前流程保留下来的真实边界：它没有被 safe-release 规则释放错误，但仍属于自动放行中的残余 FN。后续如果继续优化，应优先围绕这类低概率 TC 漏判做机制分析。", body))

    story += [
        p("六、论文写法边界", h1),
    ]
    for _, r in claims.iterrows():
        story += [
            p(claim_title_cn(str(r["claim"])), h2),
            p(f"证据：{r['evidence']} 边界：{r['boundary']}", body),
        ]
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / "2026-05-27_Task7安全释放评分卡与跨域风险控制阶段汇报.pdf"
    build_pdf(pdf_path)

    md_src = V165_DIR / "v165_safe_release_evidence_pack.md"
    md_text = md_src.read_text(encoding="utf-8")
    md_out = OUT_DIR / "v166_safe_release_stage_report_source.md"
    md_out.write_text(md_text, encoding="utf-8")
    report = {
        "pdf": str(pdf_path),
        "source_md": str(md_out),
        "source_pack": str(V165_DIR),
    }
    (OUT_DIR / "v166_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v166] wrote {pdf_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
