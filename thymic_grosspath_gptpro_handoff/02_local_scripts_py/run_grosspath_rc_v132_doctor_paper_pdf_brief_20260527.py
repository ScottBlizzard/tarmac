from __future__ import annotations

import shutil
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
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "grosspath_rc_v131_final_results_pack_with_fp_leaveout_20260527"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v132_doctor_paper_pdf_brief_20260527"
REPORT_DIR = ROOT / "汇报"
PDF_NAME = "2026-05-27_Task7双向两信号评分卡阶段汇报_医生论文共用版.pdf"


def register_fonts() -> tuple[str, str]:
    simhei = Path("C:/Windows/Fonts/simhei.ttf")
    simsun = Path("C:/Windows/Fonts/simsun.ttc")
    if not simsun.exists():
        simsun = Path("C:/Windows/Fonts/simsunb.ttf")
    pdfmetrics.registerFont(TTFont("SimHei", str(simhei)))
    pdfmetrics.registerFont(TTFont("SimSun", str(simsun)))
    return "SimHei", "SimSun"


def pct_text(value: str) -> str:
    return str(value).replace("nan", "")


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SRC / name, dtype=str).fillna("")


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def make_table(rows: list[list[str]], widths: list[float], font_body: str, font_head: str) -> Table:
    data = [[para(str(cell), ParagraphStyle("th", fontName=font_head, fontSize=8.2, leading=10.5, alignment=TA_CENTER, textColor=colors.white)) for cell in rows[0]]]
    for row in rows[1:]:
        data.append([para(str(cell), ParagraphStyle("td", fontName=font_body, fontSize=8, leading=10.2, alignment=TA_CENTER)) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E5F")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7CC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
            ]
        )
    )
    return table


def add_section(story: list, title: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Spacer(1, 0.15 * cm))
    story.append(para(title, styles["h1"]))
    story.append(Spacer(1, 0.12 * cm))


def image_block(path: Path, caption: str, styles: dict[str, ParagraphStyle], max_w: float = 16.0 * cm, max_h: float = 9.2 * cm) -> list:
    img = Image(str(path))
    ratio = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth = img.imageWidth * ratio
    img.drawHeight = img.imageHeight * ratio
    return [img, Spacer(1, 0.08 * cm), para(caption, styles["caption"])]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("SimSun", 8)
    canvas.setFillColor(colors.HexColor("#6B7C85"))
    canvas.drawString(1.7 * cm, 1.05 * cm, "Task7 大体病理图像风险控制阶段汇报")
    canvas.drawRightString(A4[0] - 1.7 * cm, 1.05 * cm, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    font_head, font_body = register_fonts()

    styles = getSampleStyleSheet()
    style_map = {
        "title": ParagraphStyle(
            "title_cn",
            parent=styles["Title"],
            fontName=font_head,
            fontSize=20,
            leading=25,
            textColor=colors.HexColor("#173843"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle_cn",
            fontName=font_body,
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#4F626A"),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1_cn",
            fontName=font_head,
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#1F4E5F"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body_cn",
            fontName=font_body,
            fontSize=9.4,
            leading=14.2,
            textColor=colors.HexColor("#1E2B32"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet_cn",
            fontName=font_body,
            fontSize=9.2,
            leading=13.4,
            leftIndent=12,
            firstLineIndent=-7,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption_cn",
            fontName=font_body,
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#5D6E76"),
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "callout_cn",
            fontName=font_head,
            fontSize=10.2,
            leading=14,
            textColor=colors.HexColor("#173843"),
            alignment=TA_LEFT,
        ),
    }

    pdf_path = OUT_DIR / PDF_NAME
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.55 * cm,
        title="Task7 双向两信号评分卡阶段汇报",
        author="影响分析项目组",
    )

    story: list = []
    story.append(para("Task7 双向两信号评分卡阶段汇报", style_map["title"]))
    story.append(para("大体病理图像风险可控诊断框架｜2026-05-27", style_map["subtitle"]))

    callout = Table(
        [[para("阶段结论：我们当前最适合主写的高安全候选是 v118/v119 双向两信号评分卡。它在 all-domain 上达到 BAcc 99.81%，控制率 79.97%，FN=1，FP=0；v120 的 100% 结果仅作为后验上限，不作为正式流程。", style_map["callout"])]],
        colWidths=[17.2 * cm],
        hAlign="LEFT",
    )
    callout.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F2F3")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#77A8B3")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(callout)
    story.append(Spacer(1, 0.25 * cm))

    add_section(story, "1. 主结果和证据层级", style_map)
    main = load_csv("v125_main_result_table_formatted.csv")
    main_rows = [["流程", "证据等级", "控制率", "BAcc", "残余错误", "FN", "FP"]]
    for _, r in main.iterrows():
        label = {
            "v91 main locked": "v91 主流程",
            "v118/v119 global bidirectional two-signal scorecard": "v118/v119 双向两信号",
            "v120 post-hoc crop rescue upper bound": "v120 后验上限",
        }.get(r["workflow"], r["workflow"])
        tier = {
            "locked primary": "锁定主流程",
            "deployable candidate after nested support": "高安全候选",
            "post-hoc upper bound only": "后验上限",
        }.get(r["evidence_tier"], r["evidence_tier"])
        main_rows.append([label, tier, r["control_rate"], r["balanced_accuracy"], r["remaining_error_n"], r["fn"], r["fp"]])
    story.append(make_table(main_rows, [3.6 * cm, 3.5 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.4 * cm, 1.4 * cm], font_body, font_head))
    story.append(Spacer(1, 0.18 * cm))
    story.append(para("解释：v91 是证据边界最干净的主流程；v118/v119 是目前最有方法价值的高安全候选；v120 证明 crop 信号存在上限空间，但嵌套验证未通过，因此只放在上限/消融部分。", style_map["body"]))

    add_section(story, "2. 方法创新：双向两信号风险评分卡", style_map)
    story.append(para("我们将错误方向拆成两类：自动低危中的高危漏诊，以及自动高危中的低危误升级。两类风险都用同一组二维信号审查：whole-crop 高危概率和核心模型平均高危概率，只是触发区域相反。这样形成了可解释的双向评分卡，而不是简单堆叠后处理规则。", style_map["body"]))
    fig1 = SRC / "figures" / "v119_bidirectional_two_signal_map.png"
    if fig1.exists():
        story.extend(image_block(fig1, "图 1. 双向两信号风险评分卡：左侧防高危漏诊，右侧防低危误升级。", style_map))

    story.append(PageBreak())
    add_section(story, "3. 消融：two-signal 为什么不是单信号替代", style_map)
    abl = load_csv("v125_two_signal_ablation_table_formatted.csv")
    abl_rows = [["方法", "额外复核", "抓回错误", "干净复核", "控制率", "BAcc", "FN", "FP"]]
    for _, r in abl.iterrows():
        label = {
            "v118_fixed_two_signal": "v118 固定 two-signal",
            "selected_wholecrop_only": "wholecrop-only",
            "selected_core_only": "core-only",
            "selected_two_signal": "扫描最优 two-signal",
        }.get(r["workflow"], r["workflow"])
        abl_rows.append([label, r["extra_review_n"], r["captured_error_n"], r["clean_review_n"], r["control_rate"], r["balanced_accuracy"], r["fn"], r["fp"]])
    story.append(make_table(abl_rows, [4.1 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm, 1.2 * cm, 1.2 * cm], font_body, font_head))
    story.append(Spacer(1, 0.16 * cm))
    story.append(para("同样清零 FP 时，wholecrop-only 需要额外复核 16 例，core-only 需要 17 例，而 two-signal 只需要 8 例。随机复核 8 个自动高危病例，平均只能抓回 0.31 个 FP，达到 3 个 FP 的概率约 0.0009。", style_map["body"]))
    fig2 = SRC / "figures" / "v124_review_efficiency_frontier.png"
    if fig2.exists():
        story.extend(image_block(fig2, "图 2. 复核效率前沿：two-signal 在同等安全目标下约减半复核负担。", style_map, max_h=8.4 * cm))

    add_section(story, "4. 稳定性：避免把错例记住", style_map)
    stab = load_csv("v129_leave_domain_stability_table_formatted.csv")
    stab_rows = [["验证", "控制率", "BAcc", "残余错误", "FN", "FP", "含义"]]
    meaning = {
        "v126_min_leave_domain": "最窄规则会漏 1 个 FP",
        "v127_unbounded_stable_envelope": "FP 清零但复核过高",
        "v128_capped_stable_envelope": "安全平台内 FP 清零",
    }
    for _, r in stab.iterrows():
        label = {
            "v126_min_leave_domain": "留域最窄规则",
            "v127_unbounded_stable_envelope": "无上限包络",
            "v128_capped_stable_envelope": "带上限包络",
        }.get(r["stability_workflow"], r["stability_workflow"])
        stab_rows.append([label, r["control_rate"], r["balanced_accuracy"], r["remaining_error_n"], r["fn"], r["fp"], meaning.get(r["stability_workflow"], "")])
    story.append(make_table(stab_rows, [3.4 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.1 * cm, 1.1 * cm, 4.8 * cm], font_body, font_head))
    story.append(Spacer(1, 0.16 * cm))
    loo = load_csv("v131_leave_one_fp_out_summary_formatted.csv")
    loo_rows = [["选择器", "held-out FP 抓回", "平均控制率", "最差 FP", "最差 FN"]]
    for _, r in loo.iterrows():
        label = "最窄规则" if r["selector"] == "min_review" else "带上限稳定包络"
        loo_rows.append([label, f"{r['heldout_captured_n']}/{r['heldout_cases']}", r["mean_full_control_rate"], r["max_full_fp"], r["max_full_fn"]])
    story.append(make_table(loo_rows, [4.4 * cm, 3.0 * cm, 3.0 * cm, 2.0 * cm, 2.0 * cm], font_body, font_head))
    story.append(Spacer(1, 0.12 * cm))
    story.append(para("留一 FP 错例验证中，带上限稳定包络在不看 held-out FP 的情况下抓回 3/3 个 FP；这说明它不是简单记住全部 FP 错例。但目前 FP 错例只有 3 个，所以这仍是内部稳定性证据，不替代前瞻外部验证。", style_map["body"]))

    story.append(PageBreak())
    add_section(story, "5. 统计边界和写作边界", style_map)
    pair = load_csv("v125_paired_delta_table_formatted.csv")
    pair_rows = [["范围", "v111 BAcc", "v118 BAcc", "ΔBAcc", "CI", "救回", "误伤"]]
    for _, r in pair.iterrows():
        if r["scope"] not in ["all_domains", "third_batch", "strict_external"]:
            continue
        scope = {"all_domains": "全域", "third_batch": "第三批", "strict_external": "严格外部"}.get(r["scope"], r["scope"])
        pair_rows.append([scope, r["base_bacc"], r["new_bacc"], r["delta_bacc"], f"{r['delta_bacc_ci_low']} - {r['delta_bacc_ci_high']}", r["base_wrong_new_correct"], r["base_correct_new_wrong"]])
    story.append(make_table(pair_rows, [2.5 * cm, 2.3 * cm, 2.3 * cm, 2.0 * cm, 3.2 * cm, 1.5 * cm, 1.5 * cm], font_body, font_head))
    story.append(Spacer(1, 0.18 * cm))
    story.append(para("同病例配对显示 v118 相比 v111 救回 3 个错误、无新增错误；但 McNemar p=0.25，不能写成强统计显著。建议表述为同病例错误减少、复核效率提升和风险控制框架更完整。", style_map["body"]))

    add_section(story, "6. 下一步", style_map)
    next_points = [
        "把本结果包扩展为论文 Results 草稿：主流程、方法图、消融、效率前沿、稳定性验证和限制。",
        "若需要继续冲 0 错误，只能寻找能在嵌套或新外部批次中重复出现的 FN 模式；不能围绕 2516531 单例定正式规则。",
        "后续最关键的验证不是继续在当前数据上加规则，而是增加新外部批次，验证 two-signal 安全平台和包络选择是否保持稳定。",
    ]
    for item in next_points:
        story.append(para(f"• {item}", style_map["bullet"]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    report_path = REPORT_DIR / PDF_NAME
    shutil.copy2(pdf_path, report_path)
    return pdf_path


def main() -> None:
    pdf = build_pdf()
    print(pdf)
    print(REPORT_DIR / PDF_NAME)


if __name__ == "__main__":
    main()
