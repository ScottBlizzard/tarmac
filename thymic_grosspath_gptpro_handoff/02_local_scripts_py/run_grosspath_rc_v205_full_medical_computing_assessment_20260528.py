from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
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


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v205_full_medical_computing_assessment_20260528"
REPORT_DIR = ROOT / "汇报"
PDF_PATH = REPORT_DIR / "2026-05-28_Task7医工交叉完整成果与论文完成度详细报告.pdf"
MD_PATH = REPORT_DIR / "2026-05-28_Task7医工交叉完整成果与论文完成度详细报告.md"

V202_METRICS = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_clean_vs_full_policy_metrics.csv"
V202_PRESENCE = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_exclusion_presence_in_current_task7.csv"
V198_CI = ROOT / "outputs" / "grosspath_rc_v198_release_compressed_safety_ci_20260527" / "v198_release_compressed_safety_ci.csv"
V197_TABLE = ROOT / "outputs" / "grosspath_rc_v197_stable_release_evidence_update_20260527" / "v197_stable_release_evidence_table.csv"
V201_SUMMARY = ROOT / "outputs" / "grosspath_rc_v201_stable_supported_domain_flip_20260527" / "v201_stable_supported_domain_flip_summary.csv"
V199_QUALITY = ROOT / "outputs" / "grosspath_rc_v199_directional_error_flip_corrector_20260527" / "v199_directional_error_model_quality.csv"
TASK7_FAIR = ROOT / "reports" / "ThymicGross" / "task7_fair_benchmark_overall_results_2026-05-13.csv"
TASK6_FAIR = ROOT / "reports" / "ThymicGross" / "task6_fair_benchmark_overall_results_2026-05-13.csv"
TASK6_TO_7 = ROOT / "reports" / "ThymicGross" / "task6_folded_to_task7_vs_direct_task7_summary_2026-05-15.csv"
V125_MAIN = ROOT / "outputs" / "grosspath_rc_v125_paper_results_pack_20260527" / "v125_main_result_table.csv"
V86_PRIMARY = ROOT / "outputs" / "grosspath_rc_v86_paper_ready_summary_pack_20260527" / "v86_primary_fixed_workflow_table.csv"

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7Detailed"


def pct(x: object, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def fmt(x: object, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def esc(x: object) -> str:
    return html.escape("" if pd.isna(x) else str(x))


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT_NAME,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#12395b"),
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=5 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_NAME,
            fontSize=13.2,
            leading=18,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_NAME,
            fontSize=10.8,
            leading=15,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=2.4 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.8,
            leading=13.3,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=1.4 * mm,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.2,
            leading=9.6,
            wordWrap="CJK",
            textColor=colors.HexColor("#111827"),
            spaceAfter=0.7 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.4,
            leading=12.5,
            leftIndent=6 * mm,
            firstLineIndent=-4 * mm,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=0.8 * mm,
        ),
        "note": ParagraphStyle(
            "note",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.2,
            leading=12.0,
            borderColor=colors.HexColor("#d2d6dc"),
            borderWidth=0.5,
            borderPadding=5,
            backColor=colors.HexColor("#f5f7fb"),
            textColor=colors.HexColor("#1f2933"),
            wordWrap="CJK",
            spaceAfter=2.0 * mm,
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.8,
            alignment=TA_CENTER,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "td": ParagraphStyle(
            "td",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "td_left": ParagraphStyle(
            "td_left",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.8,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "td_right": ParagraphStyle(
            "td_right",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.8,
            alignment=TA_RIGHT,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(esc(text), styles[style])


def b(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph("• " + esc(text), styles["bullet"])


def tbl(
    rows: list[list[object]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    left_cols: set[int] | None = None,
    right_cols: set[int] | None = None,
) -> Table:
    left_cols = left_cols or set()
    right_cols = right_cols or set()
    out = []
    for i, row in enumerate(rows):
        out_row = []
        for j, cell in enumerate(row):
            style = "th" if i == 0 else ("td_left" if j in left_cols else "td_right" if j in right_cols else "td")
            out_row.append(Paragraph(esc(cell), styles[style]))
        out.append(out_row)
    t = Table(out, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c2cf")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fb")]),
            ]
        )
    )
    return t


def row(df: pd.DataFrame, **kwargs) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for k, v in kwargs.items():
        mask &= df[k].eq(v)
    matched = df.loc[mask]
    if matched.empty:
        raise KeyError(kwargs)
    return matched.iloc[0]


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(197 * mm, 8 * mm, f"第 {doc.page} 页")
    canvas.drawString(14 * mm, 8 * mm, "Task7 胸腺大体病理图像低危/高危风险控制诊断框架")
    canvas.restoreState()


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.read_csv(V202_METRICS),
        "presence": pd.read_csv(V202_PRESENCE),
        "ci": pd.read_csv(V198_CI),
        "release": pd.read_csv(V197_TABLE),
        "v201": pd.read_csv(V201_SUMMARY),
        "quality": pd.read_csv(V199_QUALITY),
        "task7_fair": pd.read_csv(TASK7_FAIR),
        "task6_fair": pd.read_csv(TASK6_FAIR),
        "task6_to_7": pd.read_csv(TASK6_TO_7),
        "v125": pd.read_csv(V125_MAIN),
        "v86": pd.read_csv(V86_PRIMARY),
    }


def build_story() -> tuple[list, dict[str, object], str]:
    register_font()
    styles = make_styles()
    d = load_tables()
    metrics = d["metrics"]
    ci = d["ci"]
    release = d["release"]
    v201 = d["v201"]
    quality = d["quality"]
    presence = d["presence"]
    task7_fair = d["task7_fair"]
    task6_fair = d["task6_fair"]
    v125 = d["v125"]
    v86 = d["v86"]

    clean_all = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="all_domains")
    clean_old = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="old_data")
    clean_third = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="third_batch")
    clean_ext = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="strict_external")
    full_all = row(metrics, policy="v195_stable_agreement_release", cohort="full_sensitivity", scope="all_domains")
    v185_all = row(metrics, policy="v185_unlabeled_shift_adaptive", cohort="full_sensitivity", scope="all_domains")
    v198_v185_all = row(ci, policy="v185_unlabeled_shift_adaptive", scope="all_domains")
    v198_v195_all = row(ci, policy="v195_stable_fixed_agreement_release", scope="all_domains")
    v198_ext = row(ci, policy="v195_stable_fixed_agreement_release", scope="strict_external")
    v201_all = row(v201, scope="all_domains")
    v201_third = row(v201, scope="third_batch")
    fn_quality = quality.loc[
        quality["direction"].eq("fn_high_to_low")
        & quality["scope"].eq("internal_review_candidates_oof")
    ].sort_values("auroc", ascending=False).iloc[0]
    best_task7 = task7_fair.sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_task6 = task6_fair.sort_values("accuracy", ascending=False).iloc[0]

    story: list = []
    story.append(p("Task7 胸腺大体病理图像低危/高危分类", styles, "title"))
    story.append(p("医工交叉完整成果、论文完成度与下一阶段路线详细报告", styles, "title"))
    story.append(p("版本：2026-05-28；用途：供同伴对照柳叶刀/医学期刊、Nature 子刊、CCF/计算机期刊论文要求时使用", styles, "subtitle"))

    story.append(p("报告定位", styles, "h1"))
    story.append(
        p(
            "这份报告不是单次实验记录，而是当前项目的完整阶段总账。它把我们已经完成的医学问题定义、医生沟通、数据治理、模型训练、失败实验、风险控制、外部验证和论文转向统一整理。重点是让同伴能够直接拿去和医工交叉论文进行逐项对照：医学期刊看临床意义、队列、外部验证、安全边界和医生对照；计算机期刊看任务新颖性、方法抽象、消融、跨域泛化、可复现和理论/算法贡献。",
            styles,
        )
    )
    story.append(
        p(
            "最重要的边界：当前最强结果属于 selective diagnosis / risk-controlled workflow，即系统会对一部分病例自动给出低危或高危，对另一部分病例拒识或建议复核。高分数不能解释成“所有病例完全自动诊断已经 99% 以上”；正确解释是“在允许复核/拒识的临床工作流里，自动放行部分错误极少，剩余病例由复核流程承担安全边界”。",
            styles,
            "note",
        )
    )

    story.append(p("一页结论", styles, "h1"))
    for item in [
        f"当前 clean cohort 主分析 n={int(clean_all['n'])}，低危/高危={int(clean_all['low_risk_n'])}/{int(clean_all['high_risk_n'])}；v195 主流程 BAcc {pct(clean_all['balanced_accuracy'])}，Acc {pct(clean_all['accuracy'])}，F1 {pct(clean_all['f1'])}，FN={int(clean_all['fn'])}，FP={int(clean_all['fp'])}。",
        f"v195 自动放行/自动判读覆盖 {pct(clean_all['auto_decision_rate'])}，复核/拒识 {pct(clean_all['review_or_reject_rate'])}；自动错误 {int(clean_all['auto_error_n'])} 例，自动错误 Wilson95 上界 {pct(clean_all['auto_error_wilson95_high'])}。",
        f"严格外部集 n={int(clean_ext['n'])}，低危/高危={int(clean_ext['low_risk_n'])}/{int(clean_ext['high_risk_n'])}；当前 v195 workflow 点估计 BAcc {pct(clean_ext['balanced_accuracy'])}，但复核/拒识 {pct(clean_ext['review_or_reject_rate'])}，说明外部域仍需要严格风险控制。",
        f"v195 相比 v185，把全域自动判读从 {int(v198_v185_all['auto_decision_n'])} 例提高到 {int(v198_v195_all['auto_decision_n'])} 例，复核/拒识从 {pct(v198_v185_all['review_or_reject_rate'])} 降到 {pct(v198_v195_all['review_or_reject_rate'])}，同时自动错误仍为 1 例。",
        f"v201 支持域自动纠偏在第三批中救回 {int(v201_third['rescued_n'])} 例，误伤 {int(v201_third['hurt_n'])} 例；这是正证据，但规模太小，当前只能作为候选增强模块，不能作为主结论。",
        "医学论文路线已经具备 retrospective + external validation + 医生复核材料 + clean/full sensitivity 的雏形；计算机论文路线的关键任务是把当前多版本实验压缩成一个清楚、可复用、可审计的 risk-controlled selective diagnosis 框架。",
    ]:
        story.append(b(item, styles))

    story.append(p("核心指标总表", styles, "h1"))
    main_rows = [["口径", "n", "低危/高危", "Acc", "BAcc", "F1", "敏感性", "特异性", "FN/FP", "自动覆盖", "复核/拒识", "自动错"]]
    for name, r in [
        ("旧数据 clean", clean_old),
        ("第三批", clean_third),
        ("严格外部", clean_ext),
        ("clean 总体", clean_all),
        ("full 敏感性", full_all),
    ]:
        main_rows.append(
            [
                name,
                int(r["n"]),
                f"{int(r['low_risk_n'])}/{int(r['high_risk_n'])}",
                pct(r["accuracy"]),
                pct(r["balanced_accuracy"]),
                pct(r["f1"]),
                pct(r["sensitivity_high"]),
                pct(r["specificity_low"]),
                f"{int(r['fn'])}/{int(r['fp'])}",
                pct(r["auto_decision_rate"]),
                pct(r["review_or_reject_rate"]),
                int(r["auto_error_n"]),
            ]
        )
    story.append(tbl(main_rows, [22, 12, 20, 16, 16, 16, 16, 16, 14, 18, 18, 12], styles))
    story.append(
        p(
            "读表方式：Acc/BAcc/F1 是在“复核/拒识病例不强行自动诊断”的工作流口径下评价最终安全性；自动覆盖和自动错单独说明模型能独立放行多少病例、放行部分出现多少错误。医学报告中建议同时报告这两类指标，避免把选择性诊断误写成全自动诊断。",
            styles,
            "note",
        )
    )

    story.append(PageBreak())
    story.append(p("1. 整体研究方向", styles, "h1"))
    story.append(
        p(
            "建议将论文主线固定为：面向胸腺大体病理图像的风险可控、跨域适配、选择性诊断框架。这个主线比“训练一个胸腺瘤二分类模型”更稳，因为当前项目真正积累的价值不只是单模型准确率，而是围绕医学安全建立了一整套可解释、可拒识、可复核、可外部验证的工作流。",
            styles,
        )
    )
    story.append(
        tbl(
            [
                ["层面", "我们现在做的事", "论文价值", "不能过度宣称的地方"],
                ["医学任务", "把 WHO/病理分型折叠为低危/高危；TC/B2/B3 归高危，A/AB 等归低危", "贴近医生最关心的漏诊高危和误升级低危问题", "不是替代病理诊断，也不是完整 WHO 六分类终局诊断"],
                ["数据治理", "旧数据、第三批、严格外部集分层；医生确认特殊病例进入 exclusion list；clean/full 双口径", "减少 reviewer 对数据后处理和选择性剔除的质疑", "外部集样本量仍小，多中心/前瞻不足"],
                ["模型策略", "基础视觉模型、多图/视图、课程学习、经验标签、风险路由、选择性放行、拒识/复核", "把小样本医学影像从单纯分类提升到安全工作流", "不是所有模块都形成正结果，自动纠偏还不是成熟主模块"],
                ["临床工作流", "自动低危、自动高危、自动纠偏候选、复核/拒识四类输出", "更符合真实临床部署，能解释什么时候不该相信模型", "复核池目前仍偏大，需要后续降低复核负担"],
                ["计算机贡献", "direction-aware risk、stable release、support-domain correction、Wilson safety boundary", "可抽象为小样本跨域医学图像选择性诊断框架", "需要进一步算法化、统一命名、补强 baseline 和消融"],
            ],
            [25, 55, 55, 55],
            styles,
            left_cols={1, 2, 3},
        )
    )

    story.append(p("2. 数据与队列定义", styles, "h1"))
    story.append(
        p(
            "当前所有结果应按三类数据解释。旧数据是主要开发和早期 OOF 验证来源；第三批是医生新增数据，后期作为域扩展/开发验证数据参与框架构建；严格外部集“胸腺瘤+癌”在原则上只作为冻结外部验证，不用于调参。这个边界非常关键：如果拿严格外部集反复调参，就会破坏论文的外部验证可信度。",
            styles,
        )
    )
    story.append(
        tbl(
            [
                ["数据域", "clean n", "低危", "高危", "当前角色", "主要用途", "当前问题"],
                ["旧数据", int(clean_old["n"]), int(clean_old["low_risk_n"]), int(clean_old["high_risk_n"]), "开发/OOF 主体", "基线、课程学习、hard 分层、风险控制规则", "部分特殊组织学病例需 clean/full 双口径"],
                ["第三批", int(clean_third["n"]), int(clean_third["low_risk_n"]), int(clean_third["high_risk_n"]), "域扩展开发验证", "检验同院新批次偏移、提升泛化", "低危比例高，不能单独代表外部泛化"],
                ["严格外部", int(clean_ext["n"]), int(clean_ext["low_risk_n"]), int(clean_ext["high_risk_n"]), "冻结外部测试", "检验跨域、质量差异和拒识能力", "样本量小，复核/拒识比例高，需前瞻扩展"],
                ["clean 总体", int(clean_all["n"]), int(clean_all["low_risk_n"]), int(clean_all["high_risk_n"]), "主分析", "论文主结果", "必须同时给 full 敏感性分析"],
                ["full 总体", int(full_all["n"]), int(full_all["low_risk_n"]), int(full_all["high_risk_n"]), "敏感性分析", "证明剔除特殊病例不改变结论", "不作为主临床口径"],
            ],
            [26, 16, 16, 16, 32, 50, 50],
            styles,
            left_cols={4, 5, 6},
        )
    )

    story.append(p("2.1 医生确认特殊病例处理", styles, "h2"))
    ex_rows = [["病例号", "医生备注", "当前主流程状态", "处理原则"]]
    for _, r in presence.iterrows():
        status = "在当前 Task7 主队列内" if bool(r["in_current_task7_main"]) else "不在当前 Task7 主队列内"
        ex_rows.append(
            [
                r["original_case_id"],
                r["doctor_note"],
                status,
                "clean 主分析剔除；full 敏感性保留；原始数据不删除",
            ]
        )
    story.append(tbl(ex_rows, [24, 38, 45, 75], styles, left_cols={1, 2, 3}))
    story.append(
        p(
            "这一处理方式适合写进论文：不是因为模型错了才临时删除，而是医生事先确认这些病例在组织学/任务定义上更特殊，因此固定为 clean cohort exclusion；同时保留 full cohort 证明主结论稳定。",
            styles,
            "note",
        )
    )

    story.append(PageBreak())
    story.append(p("3. 医学领域已经完成的内容", styles, "h1"))
    for item in [
        "完成了从 Task5/Task6 原始多分类问题到 Task7 低危/高危二分类任务的临床目标重定义。医生明确最重要的问题是二分类风险分层，TC 放入高危，因此 Task7 比单纯六分类更贴近临床安全。",
        "围绕医生关心的问题产出过多轮材料：七模型 Task7/Task6 混淆矩阵、DINOv2 高危漏诊清单、典型边界病例、hard_core 持续错判清单、医生复核清单、质量初筛说明、经验标签说明、第三批和外部集阶段汇报。",
        "对多图病例、切面图/外观图、图片质量、蓝色背景板标准化、尺子/肿瘤大小、注意力区域、发病频次先验、A 型陷阱亚型等医生问题逐一转化为可测试的实验方向或报告说明。",
        "建立了 clean/full 队列机制，避免特殊组织学病例、混合病理或非典型来源病例影响主结论，同时保证 full sensitivity 保留完整透明性。",
        "把“模型不确定”从一句话变成可量化工作流：哪些病例自动放行、哪些病例复核/拒识、自动放行错误率是多少、Wilson 置信上界是多少，均可向医生解释。",
        "形成了可交付给医生的实际材料包：阶段汇报 PDF、hard 复核 PDF、错例清单、病例级 CSV、质量标签与经验标签说明，使项目不是只停留在代码实验。",
    ]:
        story.append(b(item, styles))

    story.append(p("3.1 医生问题与当前对应状态", styles, "h2"))
    story.append(
        tbl(
            [
                ["医生/团队问题", "我们已经做的对应工作", "当前结论", "下一步可补"],
                ["二分类是否比六分类更重要", "建立 Task7：低危 vs 高危；同时比较 Task6 折叠为 Task7 与直接 Task7", "直接二分类更贴近临床目标，六分类可作为辅助而非主线", "补六分类特征是否能提升二分类的正式消融"],
                ["各模型错误分布和偏好", "输出七模型 Task6/Task7 混淆矩阵和 DINO 漏诊清单", "DINOv2 vitb14 whole 是早期最强基线，但仍有高危漏诊", "把最新风险控制 workflow 的错误分布重新整理成论文表"],
                ["双图病例用哪张图", "追踪多图病例使用图 1/图 2、切面图/外观图", "图片选择会影响判断，医生倾向切面图时应固定协议", "制定图像采集 SOP 和多图选择规则"],
                ["背景/色温/曝光是否要统一", "做过质量标签、外部质量门控、无标签域偏移/质量筛查", "单纯质量筛查不能完全解决外部偏移，但能作为拒识依据", "蓝板参考色校正可作为正式预处理消融"],
                ["肿瘤大小和量尺是否有用", "纳入待做建议，尚未形成稳定尺寸模型", "大小可能有临床价值，但也可能成为偏置", "做 ruler/size extraction，按小肿瘤分层分析错误率"],
                ["模型关注区域是否在肿瘤", "讨论 ROI/注意力；未把分割作为主 pipeline", "当前不是先做 GPT 分割再分类；更像图像特征+风险控制", "补 Grad-CAM/注意力或弱 ROI evidence"],
                ["发病频次先验是否要纳入", "讨论 A 型陷阱亚型和真实世界频次", "训练阶段不能简单按临床频次牺牲罕见类安全，但报告要区分 balanced 与 prevalence-weighted", "输出 prevalence-weighted sensitivity analysis"],
                ["特殊病理是否剔除", "固定五个医生确认病例的 clean/full 机制", "当前主队列只命中 2 例；剔除不改变主结论", "医生最终签字确认 exclusion rationale"],
            ],
            [38, 55, 48, 42],
            styles,
            left_cols={0, 1, 2, 3},
        )
    )

    story.append(p("3.2 医学写作的可用主张", styles, "h2"))
    for item in [
        "可写：胸腺大体病理图像中存在可被视觉模型捕捉的低危/高危风险分层信号。",
        "可写：在允许复核/拒识的工作流下，当前系统能显著压低自动放行错误，尤其避免低危误升与高危漏诊的非对称风险。",
        "可写：第三批和严格外部集提示跨批次/跨域图像差异真实存在，因此单一全自动模型不适合直接宣称临床部署。",
        "可写：医生确认的特殊组织学病例应进入 clean/full 敏感性分析，而不是混在主结果里含糊处理。",
        "暂不宜写：系统已经可以完全自动替代医生完成胸腺分型诊断。",
        "暂不宜写：严格外部 100% 点估计足以证明普适泛化；因为 n=108 且复核/拒识比例仍高。",
    ]:
        story.append(b(item, styles))

    story.append(PageBreak())
    story.append(p("4. 计算机领域已经完成的内容", styles, "h1"))
    story.append(
        p(
            "计算机侧从最初的模型试跑，逐步转向风险可控框架。关键变化是：我们不再把所有精力放在单模型提升几个点，而是把任务拆成主分类、错误方向识别、hard 样本识别、稳定放行、外部域检测、自动纠偏候选、拒识/复核安全边界。这个方向比单纯堆模型更适合小样本医学图像，也更容易形成计算机论文的核心方法。",
            styles,
        )
    )
    story.append(
        tbl(
            [
                ["阶段", "代表实验/文件", "做法", "结果/发现", "方法意义"],
                ["公平基线", "2026-05-13 七模型 Task7/Task6", "SE-ResNeXt、DINOv2 vits/b/l、concat、experience aux", f"Task7 早期最佳 {best_task7['model_name']}，BAcc {pct(best_task7['balanced_accuracy'])}；Task6 最佳 Acc {pct(best_task6['accuracy'])}", "确认 DINO 系列明显更适合，Task6 六分类难度更高"],
                ["Task6 折叠 Task7", "task6_folded_to_task7_vs_direct", "六分类预测再合并成二分类 vs 直接二分类", "直接二分类总体更稳，六分类不是自然上界", "医学目标应优先定义为风险分层"],
                ["视图感知/方案B", "cut/outer/mixed specialist", "切面图、外观图、混合图分别建模或路由", "纯 cut specialist 没稳定超过全学；说明外观信息可能提供正则化或数据量收益", "负结果提示不能只凭直觉拆数据"],
                ["课程学习", "easy/medium/hard_core", "先学易例，再分析 hard；难度分层", "easy+medium Acc 0.962；非核心 hard Acc 0.914；hard_core Acc 0.262", "发现瓶颈集中在核心 hard，而不是全体都差"],
                ["经验/反捷径标签", "Task56/Task7 经验标签、anti-shortcut", "标注出血、囊变、边界、坏死、冲突线索等", "大规模弱标签不稳定；少量高可信标签有信号但不够稳定", "经验标签适合作为解释和候选辅助，不宜直接夸大"],
                ["二阶段复核", "v64/v65 旧数据", "模型放行一部分，其余进入复核/纠偏", "旧数据 OOF 曾达到约 0.926；hard_core 65 中覆盖 41，救回 26", "验证“复核池”思路可行，但早期流程仍需防数据泄露和泛化验证"],
                ["跨域风险控制", "v50-v86、v91-v125", "质量/OOD、伪域、conformal、scorecard、utility", "严格外部 v79-strict 曾达到 BAcc 100%，但复核/控制率高", "形成 selective diagnosis 的核心证据链"],
                ["稳定放行", "v182/v195/v196", "跨 fold 稳定高置信一致放行；拒绝 greedy 局部规则", "v195 复核/拒识降至 36.05%，v196 更低但释放 3 个 held-out 错误", "证明固定稳定规则优于贪心调参"],
                ["自动纠偏", "v199-v201", "方向风险模型、支持域翻转", "v201 第三批救回 2 例且 0 误伤；v199/v200 有误伤", "有正证据但仍是候选模块"],
                ["clean cohort", "v202/v203", "医生特殊病例固定剔除清单，clean/full 双口径", "clean n=697，full n=699；主结论不变", "提高医学论文可信度和审稿可解释性"],
            ],
            [22, 30, 40, 46, 44],
            styles,
            left_cols={1, 2, 3, 4},
        )
    )

    story.append(p("4.1 早期七模型基线结果", styles, "h2"))
    fair_rows = [["任务", "最佳模型", "Acc", "BAcc", "AUC/Macro AUC", "F1/Macro F1", "TN/FP/FN/TP 或说明"]]
    fair_rows.append(
        [
            "Task7 二分类",
            best_task7["model_name"],
            pct(best_task7["accuracy"]),
            pct(best_task7["balanced_accuracy"]),
            fmt(best_task7["auc"]),
            pct(best_task7["f1"]),
            f"{int(best_task7['tn'])}/{int(best_task7['fp'])}/{int(best_task7['fn'])}/{int(best_task7['tp'])}",
        ]
    )
    fair_rows.append(
        [
            "Task6 六分类",
            best_task6["model_name"],
            pct(best_task6["accuracy"]),
            pct(best_task6["balanced_accuracy"]),
            fmt(best_task6["macro_auc"]),
            pct(best_task6["macro_f1"]),
            "六分类明显更难，适合作为辅助分析",
        ]
    )
    story.append(tbl(fair_rows, [28, 54, 18, 18, 25, 25, 45], styles, left_cols={1, 6}))
    story.append(
        p(
            "早期基线的作用不是最终分数，而是证明两个事实：第一，DINOv2 vitb14 whole 是早期强基线；第二，Task6 六分类在当前数据量下很难稳定，直接以 Task6 作为主临床任务会拖累论文主线。医生更关心低危/高危，因此 Task7 是更合理的主任务。",
            styles,
            "note",
        )
    )

    story.append(p("4.2 课程学习和 hard_core 发现", styles, "h2"))
    story.append(
        tbl(
            [
                ["分层", "病例数", "正确数", "准确率", "解释"],
                ["easy+medium", 186, 179, "96.2%", "清楚病例已经可以接近临床可用水平"],
                ["可救 hard", 34, 22, "64.7%", "有信号，但需要单独复核/纠偏"],
                ["核心 hard", 65, 17, "26.2%", "多模型容易系统性反向，是旧模型主要瓶颈"],
                ["非核心 hard", 220, 201, "91.4%", "排除核心 hard 后，整体已超过 90%"],
                ["课程学习融合总体", 285, 218, "76.49%", "TN/FP/FN/TP=110/34/33/108，定位瓶颈而非最终部署"],
            ],
            [38, 22, 22, 22, 82],
            styles,
            left_cols={0, 4},
        )
    )
    story.append(
        p(
            "这一阶段的最大价值是问题诊断：如果所有病例混在一起训练，模型会被少量 hard_core 拉偏；如果只看 easy/medium，会误以为任务已经解决。后续 risk-controlled workflow 的逻辑正是从这里来的：先安全放行稳定病例，再把疑难病例隔离到复核/拒识或候选纠偏模块。",
            styles,
            "note",
        )
    )

    story.append(PageBreak())
    story.append(p("5. 当前主流程：v195 风险可控选择性诊断", styles, "h1"))
    story.append(
        p(
            "当前最适合作为主线写入论文的是 v195 stable fixed high-confidence agreement release。它不是简单提高分类器输出，而是在已有风险控制框架上增加稳定高置信一致放行，目标是在不明显增加自动错误的前提下降低复核负担。",
            styles,
        )
    )
    rel_rows = [["流程", "定位", "范围", "BAcc", "复核/拒识", "自动动作", "动作错误", "FN/FP", "结论"]]
    tier_map = {
        "previous recommended framework": "既往推荐",
        "previous efficiency workflow": "既往效率",
        "high-safety fallback": "高安全回退",
        "new candidate efficiency workflow": "新效率候选",
        "negative stability audit": "负向审计",
    }
    scope_map = {"all_domains": "全域", "strict_external": "严格外部"}
    interpretation_map = {
        "v185": "风险控制基础，压缩前基线",
        "v182": "固定稳定放行，v195 前效率流程",
        "v118": "高安全回退，复核负担更高",
        "v195": "复核明显压缩，自动动作 0 错误；不是自动翻转",
        "v196": "复核更低但释放 held-out 错误，否定贪心规则",
    }
    for _, r in release.iterrows():
        label = r["workflow"]
        label = label.replace("v195 stable fixed high-confidence agreement release", "v195 稳定高置信放行")
        label = label.replace("v196 per-fold greedy agreement release audit", "v196 贪心放行审计")
        label = label.replace("v185 unlabeled shift-adaptive workflow", "v185 无标签域偏移自适应")
        label = label.replace("v182 fixed stable efficiency workflow", "v182 固定稳定效率流程")
        label = label.replace("v118 fixed high-safety fallback", "v118 高安全回退")
        interp = "待解释"
        for key, val in interpretation_map.items():
            if key in str(r["workflow"]):
                interp = val
                break
        rel_rows.append(
            [
                label,
                tier_map.get(str(r["tier"]), r["tier"]),
                scope_map.get(str(r["scope"]), r["scope"]),
                pct(r["balanced_accuracy"]),
                pct(r["remaining_review_or_reject_rate"]),
                "" if pd.isna(r["auto_action_n"]) else int(r["auto_action_n"]),
                "" if pd.isna(r["action_error_n"]) else int(r["action_error_n"]),
                f"{int(r['fn'])}/{int(r['fp'])}",
                interp,
            ]
        )
    story.append(tbl(rel_rows, [27, 21, 16, 15, 18, 15, 13, 12, 47], styles, left_cols={0, 1, 8}))
    story.append(
        p(
            "关键解释：v196 看起来复核率更低，但释放了 3 个 held-out 错误，因此应该作为负结果写入，而不是选择更漂亮的复核率。这个负结果很重要，它证明我们没有为了提高覆盖率牺牲安全性。",
            styles,
            "note",
        )
    )

    story.append(p("5.1 自动放行安全边界", styles, "h2"))
    ci_rows = [["流程", "范围", "n", "自动判读", "自动错", "自动错率", "Wilson95 上界", "复核/拒识"]]
    for policy_label, policy in [
        ("v185 原始风险控制", "v185_unlabeled_shift_adaptive"),
        ("v195 稳定放行", "v195_stable_fixed_agreement_release"),
    ]:
        for scope_label, scope in [("旧数据", "old_data"), ("第三批", "third_batch"), ("严格外部", "strict_external"), ("全域", "all_domains")]:
            r = row(ci, policy=policy, scope=scope)
            ci_rows.append(
                [
                    policy_label,
                    scope_label,
                    int(r["n"]),
                    int(r["auto_decision_n"]),
                    int(r["auto_error_n"]),
                    pct(r["auto_error_rate"]),
                    pct(r["wilson95_high"]),
                    pct(r["review_or_reject_rate"]),
                ]
            )
    story.append(tbl(ci_rows, [33, 22, 14, 22, 16, 22, 28, 24], styles))
    story.append(
        p(
            f"v195 全域自动判读 {int(v198_v195_all['auto_decision_n'])}/{int(v198_v195_all['n'])}，自动错 {int(v198_v195_all['auto_error_n'])} 例，Wilson95 上界 {pct(v198_v195_all['wilson95_high'])}。严格外部自动判读 {int(v198_ext['auto_decision_n'])}/{int(v198_ext['n'])}，自动错 0，Wilson95 上界 {pct(v198_ext['wilson95_high'])}。外部样本量仍小，所以下一步需要扩大冻结外部自动判读样本数，才能把上界压低。",
            styles,
            "note",
        )
    )

    story.append(p("5.2 clean/full 敏感性分析", styles, "h2"))
    story.append(
        tbl(
            [
                ["口径", "n", "BAcc", "Acc", "自动覆盖", "复核/拒识", "FN/FP", "解释"],
                ["clean 主分析", int(clean_all["n"]), pct(clean_all["balanced_accuracy"]), pct(clean_all["accuracy"]), pct(clean_all["auto_decision_rate"]), pct(clean_all["review_or_reject_rate"]), f"{int(clean_all['fn'])}/{int(clean_all['fp'])}", "医生确认特殊病例剔除后的主结果"],
                ["full 敏感性", int(full_all["n"]), pct(full_all["balanced_accuracy"]), pct(full_all["accuracy"]), pct(full_all["auto_decision_rate"]), pct(full_all["review_or_reject_rate"]), f"{int(full_all['fn'])}/{int(full_all['fp'])}", "保留全部病例，证明结论不依赖剔除"],
                ["差异", int(full_all["n"] - clean_all["n"]), f"{(float(full_all['balanced_accuracy']) - float(clean_all['balanced_accuracy'])) * 100:+.3f} pct-pt", f"{(float(full_all['accuracy']) - float(clean_all['accuracy'])) * 100:+.3f} pct-pt", "近似不变", "近似不变", "一致", "可作为 reviewer 问题的直接回答"],
            ],
            [25, 13, 17, 17, 18, 18, 15, 59],
            styles,
            left_cols={7},
        )
    )

    story.append(PageBreak())
    story.append(p("6. 自动纠偏：正证据、负证据和边界", styles, "h1"))
    story.append(
        p(
            "自动纠偏是计算机论文最有潜力的方向之一，但目前必须谨慎写。我们已经证明方向风险模型存在信号，也证明某些条件下可以自动救回错误；但直接全域翻转会误伤，因此当前不能把它包装成成熟纠偏器。",
            styles,
        )
    )
    story.append(
        tbl(
            [
                ["实验", "目标", "结果", "当前定位"],
                ["v199 全域方向翻转", "用 FN/FP 方向风险直接改判", "救回 1 例、误伤 1 例；BAcc 降至约 99.70%", "负结果：不能全域自动翻转"],
                ["v200 同域支持约束", "只有训练折同域支持时翻转", "仍出现 1 救回 + 1 误伤", "负结果：阈值和支持还不够稳"],
                ["v201 稳定支持域翻转", "只对第三批稳定 FN 风险规则生效", f"自动救回 {int(v201_third['rescued_n'])} 例，误伤 {int(v201_third['hurt_n'])} 例", "候选正结果：可写为自动纠偏初步证据"],
                ["FN 方向风险模型", "识别高危漏诊方向风险", f"内部复核候选 AUROC {float(fn_quality['auroc']):.3f}，AP {float(fn_quality['average_precision']):.3f}", "有信号，但需转化为稳定决策规则"],
            ],
            [36, 50, 55, 44],
            styles,
            left_cols={1, 2, 3},
        )
    )
    story.append(
        p(
            "建议写法：自动纠偏不是当前主结论，而是 risk-controlled framework 的候选增强模块。可以强调我们已经从“只会拒识”走到“少数条件下能自动救回”，但还没有足够规模证明可广泛替代医生复核。",
            styles,
            "note",
        )
    )

    story.append(p("7. 负结果同样构成论文证据", styles, "h1"))
    story.append(
        tbl(
            [
                ["失败/边界", "实验现象", "我们学到什么", "论文价值"],
                ["只做模型微调", "ConvNeXt/Swin/B3 等没有稳定超过 DINO 主线", "当前瓶颈不是简单换 backbone 即可解决", "支持从单模型转向风险控制框架"],
                ["视图 specialist", "cut/outer/mixed 拆分后没有稳定优于全学", "切面图更统一不等于更好学；外观可能提供正则或样本量收益", "避免 reviewer 质疑为什么不按视图拆分"],
                ["经验标签弱注入", "大规模经验弱标签噪声大，提升不稳定", "医生/模型经验标签更适合解释和高可信小样本辅助", "说明我们没有把未经验证标签强行当真值"],
                ["全域自动纠偏", "v199/v200 误伤原本正确病例", "复核池里既有错例也有正确例，纠偏器必须控制误伤", "展示安全优先的算法选择"],
                ["贪心降低复核率", "v196 复核更低但释放 held-out 错误", "不能为效率牺牲安全", "支持固定稳定规则的必要性"],
                ["严格外部质量门控", "质量差会影响，但只筛质量不能保证高分", "外部泛化不是单纯拍照质量问题", "引出域偏移和拒识机制"],
            ],
            [34, 48, 55, 45],
            styles,
            left_cols={0, 1, 2, 3},
        )
    )

    story.append(PageBreak())
    story.append(p("8. 当前成果与论文路线对照", styles, "h1"))
    story.append(p("8.1 医学期刊路线完成度", styles, "h2"))
    story.append(
        tbl(
            [
                ["维度", "当前完成度", "已有证据", "短板", "建议补强"],
                ["临床问题", "较强", "低危/高危直接对应医生关心的风险分层", "任务定义还需医生最终签字", "请医生确认高危/低危映射和排除标准"],
                ["数据队列", "中等偏强", "旧数据、第三批、严格外部、clean/full", "外部样本量和多中心不足", "扩大冻结外部集，最好前瞻采集"],
                ["模型表现", "强", "clean BAcc 99.81%，严格外部 BAcc 100% 点估计", "selective diagnosis 复核比例较高", "同时报告自动覆盖、自动错和复核率"],
                ["安全性", "较强", "FN/FP、自动错误、Wilson CI、负结果审计", "医生复核流程耗时未量化", "补 reader study 或模拟医生复核节省时间"],
                ["可解释性", "中等", "错例清单、经验标签、质量标签、方向风险", "缺系统注意力/ROI 证据", "补注意力热图、医生盲评一致性"],
                ["临床部署", "不足", "有工作流雏形", "没有前瞻、多中心真实工作流验证", "设计 prospective validation protocol"],
            ],
            [24, 24, 50, 46, 48],
            styles,
            left_cols={0, 2, 3, 4},
        )
    )

    story.append(p("8.2 Nature 子刊 / 医工交叉路线完成度", styles, "h2"))
    story.append(
        tbl(
            [
                ["维度", "当前完成度", "有力部分", "不足", "需要形成的最终形态"],
                ["临床新问题", "较强", "大体病理照片用于胸腺风险分层，公开先例少", "需要系统文献证明空白", "把任务定位成 surgical gross-pathology AI"],
                ["方法创新", "中等", "risk-controlled selective diagnosis、direction-aware routing、stable release", "模块多但还需统一成算法", "一个清楚的框架图 + 算法伪代码 + 固定协议"],
                ["跨域泛化", "中等", "第三批和严格外部均有结果", "外部域少，不足以强 claim", "多外部域或模拟域偏移下的鲁棒性曲线"],
                ["人机协同", "中等", "复核/拒识逻辑清楚", "还没真实 reader study", "医生 alone / AI alone / AI triage / AI+doctor 比较"],
                ["统计严谨", "中等偏强", "Wilson CI、clean/full、负结果审计", "显著性和置信区间需统一", "所有主表附 CI 和 paired test"],
                ["可复现", "中等", "v185-v205 输出完整", "版本号太多，入口混乱", "整理为一个正式 repo、config、frozen seed"],
            ],
            [25, 25, 50, 45, 48],
            styles,
            left_cols={0, 2, 3, 4},
        )
    )

    story.append(p("8.3 CCF / 计算机期刊路线完成度", styles, "h2"))
    story.append(
        tbl(
            [
                ["要求", "当前情况", "风险", "补强方案"],
                ["算法贡献清晰", "已有模块但叙述仍偏实验工程", "容易被认为只是调参和模型集成", "抽象为 Risk-Controlled Selective Diagnosis with Direction-aware Error Routing"],
                ["强 baseline", "已有 DINO/ConvNeXt/Swin/PLIP/BiomedCLIP 等尝试，但需统一表", "模型对照分散", "重跑或整理同一协议下的 backbone baseline"],
                ["消融完整", "v195/v196/v199-v201/v185 有明确消融", "表述还没统一", "主文放三类消融：放行、纠偏、域适配"],
                ["泛化严谨", "有第三批与严格外部", "外部域数少", "增加域偏移模拟、无标签域审计、leave-domain-out"],
                ["理论/统计", "有 Wilson 和 selective risk", "还不够算法论文化", "补 selective risk bound、coverage-risk curve、utility analysis"],
                ["任务可迁移", "框架原则可用于其他小样本医学图像", "目前只在胸腺任务验证", "论文 claim 先收窄，未来扩展其他 gross pathology"],
            ],
            [35, 52, 45, 58],
            styles,
            left_cols={0, 1, 2, 3},
        )
    )

    story.append(PageBreak())
    story.append(p("9. Claim-Evidence Map", styles, "h1"))
    story.append(
        tbl(
            [
                ["可写主张", "证据", "状态", "备注"],
                ["我们建立了 clean cohort 与 full sensitivity 双口径", "v202 clean n=697，full n=699；剔除不改变主结果", "Supported", "医学审稿友好"],
                ["v195 是当前最稳定主流程", f"clean BAcc {pct(clean_all['balanced_accuracy'])}，复核/拒识 {pct(clean_all['review_or_reject_rate'])}，自动错 {int(clean_all['auto_error_n'])}", "Supported", "主结果"],
                ["稳定放行优于贪心放行", "v195 action-error=0；v196 复核更低但释放 3 个 held-out 错误", "Supported", "重要消融"],
                ["严格外部已经完全证明强泛化", "strict external BAcc 100%，但 n=108 且复核/拒识 51.85%", "Partially supported", "必须谨慎"],
                ["自动纠偏已经成熟", "v201 只救回 2 例；v199/v200 有误伤", "Not supported", "作为候选模块"],
                ["方向风险有可用信号", f"FN_high_to_low review candidates AUROC {float(fn_quality['auroc']):.3f}", "Supported as signal", "需稳定规则"],
                ["完全自动临床诊断已经解决", f"clean 仍有 {pct(clean_all['review_or_reject_rate'])} 复核/拒识", "Not supported", "不能这么写"],
                ["任务本身有医学价值", "医生多轮问题集中在二分类、漏诊、复核、图像质量、外部泛化", "Supported", "需医生署名确认"],
            ],
            [54, 78, 28, 34],
            styles,
            left_cols={0, 1, 2, 3},
        )
    )

    story.append(p("10. 已形成的材料资产", styles, "h1"))
    story.append(
        tbl(
            [
                ["材料", "路径/名称", "用途"],
                ["总阶段医学报告", "2026-05-18_Task7课程学习与经验标签阶段汇报_医生版.pdf", "说明课程学习、经验标签、hard 分层"],
                ["hard 医生复核清单", "2026-05-19_Task7核心hard持续错判医生复核清单.pdf", "给医生看持续错判病例和原因"],
                ["第三批引入报告", "2026-05-22_Task7第三批数据引入后适配阶段汇报_医生版.pdf", "说明第三批加入后的泛化和适配"],
                ["外部/后续尝试报告", "2026-05-25_Task7第三批与严格外部集后续尝试阶段汇报_医生版.pdf", "记录第三批、外部、质量门控和泛化问题"],
                ["GPTPro 转向材料", "2026-05-26_胸腺大体图AI项目全景与Nature子刊方向重构_给GPTPro.md；转换.md", "用于论文方向重构和计算机创新思路"],
                ["风险控制阶段报告", "2026-05-27_Task7风险控制工作流阶段汇报_医生版.pdf", "v50-v86 风险控制路线"],
                ["最终框架阶段报告", "2026-05-27_Task7风险可控跨域诊断框架最终阶段汇报_含残余边界.pdf", "v185-v192 最终框架和残余边界"],
                ["clean cohort 报告", "2026-05-27_Task7_clean_cohort_exclusion_analysis.pdf", "医生确认剔除病例和 clean/full 结果"],
                ["本报告", PDF_PATH.name, "完整项目总账，供医学/计算机论文完成度对照"],
            ],
            [36, 88, 62],
            styles,
            left_cols={0, 1, 2},
        )
    )

    story.append(p("11. 下一阶段工作包", styles, "h1"))
    story.append(
        tbl(
            [
                ["优先级", "工作包", "目标", "输出", "成功标准"],
                ["P0", "统一框架命名和固定协议", "把 v185-v205 压缩成一个清楚算法", "流程图、伪代码、固定 config、主表", "同伴看报告能复述 pipeline"],
                ["P0", "主结果表重算和代码冻结", "锁定 clean/full/old/third/external", "main table + supplement CSV", "所有数字可由一条脚本复现"],
                ["P1", "强 baseline 统一整理", "满足计算机审稿对照", "DINOv2/DINOv3/PLIP/BiomedCLIP/ConvNeXt/Swin 表", "同协议、同 split、同指标"],
                ["P1", "医生 reader study 设计", "补医学期刊短板", "医生 alone vs AI vs AI+review", "证明节省复核负担或提高安全"],
                ["P1", "外部前瞻样本量规划", "压低 Wilson 上界", "样本量表和入排标准", "自动判读错误上界可解释"],
                ["P2", "自动纠偏扩大验证", "让 v201 从候选变主模块", "纠偏器消融表", "救回数扩大且误伤仍受控"],
                ["P2", "图像 SOP/质量/ROI", "回应医生拍照和注意力问题", "质量标准、蓝板校正、ROI 可解释图", "能解释拒识原因并指导重拍"],
                ["P2", "临床发病率加权分析", "回应发病频次问题", "balanced vs prevalence-weighted 表", "同时体现公平训练和真实世界风险"],
            ],
            [18, 42, 45, 45, 42],
            styles,
            left_cols={1, 2, 3, 4},
        )
    )

    story.append(p("12. 给同伴检索论文时的对照建议", styles, "h1"))
    for item in [
        "医学期刊检索时，不要只看准确率；重点看样本量、多中心/前瞻、外部验证、reader study、临床终点、安全声明和医生工作流。",
        "Nature Medicine / Nature Biomedical Engineering 类论文重点看：临床需求是否明确、方法是否有一般性、外部泛化是否强、是否有真实工作流验证。",
        "Medical Image Analysis / TMI / CCF 类论文重点看：是否有清楚算法、强 baseline、消融、统计显著性、跨域泛化、可复现代码和理论边界。",
        "我们的优势是任务新、临床沟通扎实、risk-controlled workflow 证据链完整；短板是样本量小、外部域少、方法还需要统一抽象、reader study 和前瞻验证不足。",
        "同伴对照论文时，建议把别人的贡献拆成四列：数据强度、临床验证强度、算法创新强度、泛化/安全证明强度，再和本报告逐项对照。",
    ]:
        story.append(b(item, styles))

    story.append(p("最终判断", styles, "h1"))
    story.append(
        p(
            "最终定位：当前已具备医学 retrospective + external validation 的雏形；若走 Nature 子刊或计算机强刊，必须把多版本实验收敛成统一的 risk-controlled selective diagnosis 框架，并补统一 baseline、消融、外部泛化和 reader study。",
            styles,
            "note",
        )
    )

    meta = {
        "clean_n": int(clean_all["n"]),
        "clean_bacc": float(clean_all["balanced_accuracy"]),
        "clean_review_or_reject_rate": float(clean_all["review_or_reject_rate"]),
        "clean_auto_error_n": int(clean_all["auto_error_n"]),
        "strict_external_n": int(clean_ext["n"]),
        "strict_external_bacc": float(clean_ext["balanced_accuracy"]),
        "strict_external_review_or_reject_rate": float(clean_ext["review_or_reject_rate"]),
        "v195_auto_decision_n_all": int(v198_v195_all["auto_decision_n"]),
        "v195_auto_error_wilson95_high_all": float(v198_v195_all["wilson95_high"]),
        "v201_rescued_n_all": int(v201_all["rescued_n"]),
        "v201_hurt_n_all": int(v201_all["hurt_n"]),
    }
    md = "\n".join(
        [
            "# Task7 医工交叉完整成果与论文完成度详细报告",
            "",
            f"- clean cohort n={meta['clean_n']}，BAcc={pct(meta['clean_bacc'])}，复核/拒识={pct(meta['clean_review_or_reject_rate'])}，自动错={meta['clean_auto_error_n']}。",
            f"- 严格外部 n={meta['strict_external_n']}，BAcc={pct(meta['strict_external_bacc'])}，复核/拒识={pct(meta['strict_external_review_or_reject_rate'])}。",
            f"- v195 全域自动判读 {meta['v195_auto_decision_n_all']} 例，自动错误 Wilson95 上界 {pct(meta['v195_auto_error_wilson95_high_all'])}。",
            f"- v201 自动纠偏救回 {meta['v201_rescued_n_all']} 例，误伤 {meta['v201_hurt_n_all']} 例；仍是候选模块。",
            "",
            "PDF 详版包含医学成果、计算机成果、研究方向、主指标、失败实验、claim-evidence map、期刊路线完成度和下一阶段工作包。",
        ]
    )
    return story, meta, md


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    story, meta, md = build_story()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=13 * mm,
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    MD_PATH.write_text(md, encoding="utf-8")
    (OUT_DIR / "v205_report_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v205] wrote {PDF_PATH}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
