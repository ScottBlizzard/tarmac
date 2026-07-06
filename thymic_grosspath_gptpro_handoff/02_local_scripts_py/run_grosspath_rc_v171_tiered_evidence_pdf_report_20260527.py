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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v171_tiered_evidence_pdf_report_20260527"
REPORT_DIR = ROOT / "汇报"
V170_MAIN = ROOT / "outputs" / "grosspath_rc_v170_paper_evidence_tier_pack_20260527" / "v170_main_operating_table_tiered.csv"
V170_MODULE = ROOT / "outputs" / "grosspath_rc_v170_paper_evidence_tier_pack_20260527" / "v170_module_tier_table.csv"
V170_CLAIMS = ROOT / "outputs" / "grosspath_rc_v170_paper_evidence_tier_pack_20260527" / "v170_claim_tier_map.csv"
V170_GUIDE = ROOT / "outputs" / "grosspath_rc_v170_paper_evidence_tier_pack_20260527" / "v170_recommended_writing_guide.md"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def register_fonts() -> tuple[str, str]:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    font = next((p for p in candidates if p.exists()), None)
    if font is None:
        raise FileNotFoundError("No usable Chinese TTF/TTC font found.")
    pdfmetrics.registerFont(TTFont("ReportCN", str(font)))
    pdfmetrics.registerFont(TTFont("ReportCN-Bold", str(font)))
    return "ReportCN", "ReportCN-Bold"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def mk_table(data: list[list[object]], widths: list[float], font: str, bold: str) -> Table:
    head = ParagraphStyle("head", fontName=bold, fontSize=7.2, leading=9, alignment=TA_CENTER, textColor=colors.white)
    cell = ParagraphStyle("cell", fontName=font, fontSize=6.9, leading=8.7, alignment=TA_CENTER)
    left = ParagraphStyle("left", fontName=font, fontSize=6.9, leading=8.7, alignment=TA_LEFT)
    rows = []
    for i, row in enumerate(data):
        rr = []
        for j, item in enumerate(row):
            rr.append(p(str(item), head if i == 0 else (left if j in [0, 1] else cell)))
        rows.append(rr)
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


def short_line(name: str) -> str:
    return {
        "30% low-confidence selective review baseline": "低置信度复核",
        "30% image-direction router": "方向感知复核",
        "v118 high-safety two-signal scorecard": "高安全评分卡",
        "v161 safe-release high-safety scorecard": "安全释放评分卡",
        "severe-shift gated concept auto-correction": "severe-shift 概念纠偏",
    }.get(name, name)


def short_tier(tier: str) -> str:
    return {
        "Comparator": "对照",
        "Tier 1 main/high-safety baseline": "Tier 1 主线",
        "Tier 2 candidate": "Tier 2 候选",
        "Tier 2 efficiency candidate": "Tier 2 效率候选",
        "Tier 2 severe-shift correction candidate": "Tier 2 纠偏候选",
    }.get(tier, tier)


def main_rows(main: pd.DataFrame) -> list[list[object]]:
    rows = [["层级", "方案", "数据域", "自动放行", "复核/拒识", "自动纠偏", "BAcc", "FN", "FP"]]
    order = [
        "30% low-confidence selective review baseline",
        "30% image-direction router",
        "v118 high-safety two-signal scorecard",
        "v161 safe-release high-safety scorecard",
        "severe-shift gated concept auto-correction",
    ]
    for name in order:
        for domain in ["all_domains", "strict_external"]:
            sub = main.loc[main["evidence_line"].eq(name) & main["eval_domain"].eq(domain)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                [
                    short_tier(str(r["tier"])),
                    short_line(name),
                    "全域" if domain == "all_domains" else "严格外部",
                    pct(r["auto_pass_rate"]),
                    pct(r["review_or_reject_rate"]),
                    pct(r["auto_correct_rate"]),
                    pct(r["balanced_accuracy"], 2),
                    int(r["fn"]),
                    int(r["fp"]),
                ]
            )
    return rows


def module_rows(modules: pd.DataFrame) -> list[list[object]]:
    rows = [["层级", "模块", "证据等级", "论文用途"]]
    for _, r in modules.iterrows():
        rows.append([r["tier"], r["module"], r["evidence_grade"], r["recommended_paper_use"]])
    return rows


def claim_rows(claims: pd.DataFrame) -> list[list[object]]:
    rows = [["层级", "主张", "安全写法", "不能这样写"]]
    for _, r in claims.iterrows():
        rows.append([r["claim_tier"], r["claim"], r["safe_wording"], r["do_not_claim"]])
    return rows


def build_pdf(pdf_path: Path) -> None:
    font, bold = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=bold, fontSize=19, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#102A43"))
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName=font, fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#486581"))
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=bold, fontSize=13.2, leading=17, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#243B53"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font, fontSize=9, leading=13.2, spaceAfter=5, textColor=colors.HexColor("#1F2933"))
    callout = ParagraphStyle("callout", parent=body, fontName=bold, fontSize=9.5, leading=14.2, leftIndent=8, rightIndent=8, borderPadding=8, backColor=colors.HexColor("#E6F6FF"), textColor=colors.HexColor("#102A43"))

    main = pd.read_csv(V170_MAIN)
    modules = pd.read_csv(V170_MODULE)
    claims = pd.read_csv(V170_CLAIMS)
    guide = V170_GUIDE.read_text(encoding="utf-8")
    wording = guide.split("## One-paragraph Safe Wording", 1)[-1].strip()

    v118 = main.loc[main["evidence_line"].eq("v118 high-safety two-signal scorecard") & main["eval_domain"].eq("all_domains")].iloc[0]
    v161 = main.loc[main["evidence_line"].eq("v161 safe-release high-safety scorecard") & main["eval_domain"].eq("all_domains")].iloc[0]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    story = [
        p("Task7 论文证据分层与当前主线汇报", title),
        p("risk-controlled cross-domain gross pathology workflow | 2026-05-27", subtitle),
        Spacer(1, 0.3 * cm),
        p(
            f"主线建议：以 v118 高安全评分卡作为最稳主流程（BAcc {pct(v118['balanced_accuracy'], 2)}，复核/拒识 {pct(v118['review_or_reject_rate'])}），"
            f"以 v161 safe-release 作为当前效率候选（BAcc {pct(v161['balanced_accuracy'], 2)}，复核/拒识 {pct(v161['review_or_reject_rate'])}）。"
            "v161 不能写成 nested 锁定阈值，因为 v167/v168 已显示折内重选规则会释放错误。",
            callout,
        ),
        p("一、主 operating points", h1),
        mk_table(main_rows(main), [1.7 * cm, 2.5 * cm, 1.4 * cm, 1.5 * cm, 1.6 * cm, 1.5 * cm, 1.4 * cm, 0.7 * cm, 0.7 * cm], font, bold),
        p("二、模块证据分层", h1),
        mk_table(module_rows(modules), [1.5 * cm, 3.5 * cm, 3.4 * cm, 7.0 * cm], font, bold),
        p("三、主张-证据边界", h1),
        mk_table(claim_rows(claims), [2.1 * cm, 4.5 * cm, 4.6 * cm, 4.2 * cm], font, bold),
        p("四、建议写法", h1),
        p(wording, body),
        p("五、下一步实验与交付物", h1),
        p(
            "下一步不建议继续只追单点最高准确率。更重要的是补两类证据：第一，给 safe-release 做前瞻或更严格的稳定验证，证明降低复核率不会释放高危漏诊；第二，将严格外部集继续保持为验证集，只用旧数据和第三批数据校准门控和候选规则。当前可直接用于汇报的交付物包括 v170 主 operating table、模块 tier 表、claim map，以及 v166/v171 两份 PDF。",
            body,
        ),
    ]
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / "2026-05-27_Task7论文证据分层与当前主线汇报.pdf"
    build_pdf(pdf_path)
    report = {"pdf": str(pdf_path), "source": str(OUT_DIR)}
    (OUT_DIR / "v171_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v171] wrote {pdf_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
