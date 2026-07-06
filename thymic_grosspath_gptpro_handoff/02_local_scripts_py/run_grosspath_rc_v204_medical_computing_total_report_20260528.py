from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v204_medical_computing_total_report_20260528"
REPORT_DIR = ROOT / "汇报"
PDF_PATH = REPORT_DIR / "2026-05-28_Task7医工交叉论文完整成果与完成度总报告.pdf"
MD_PATH = REPORT_DIR / "2026-05-28_Task7医工交叉论文完整成果与完成度总报告.md"

V202_METRICS = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_clean_vs_full_policy_metrics.csv"
V202_PRESENCE = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_exclusion_presence_in_current_task7.csv"
V198_CI = ROOT / "outputs" / "grosspath_rc_v198_release_compressed_safety_ci_20260527" / "v198_release_compressed_safety_ci.csv"
V197_TABLE = ROOT / "outputs" / "grosspath_rc_v197_stable_release_evidence_update_20260527" / "v197_stable_release_evidence_table.csv"
V201_SUMMARY = ROOT / "outputs" / "grosspath_rc_v201_stable_supported_domain_flip_20260527" / "v201_stable_supported_domain_flip_summary.csv"
V199_QUALITY = ROOT / "outputs" / "grosspath_rc_v199_directional_error_flip_corrector_20260527" / "v199_directional_error_model_quality.csv"
V192_OPERATING = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527" / "v192_final_framework_operating_table.csv"

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7Total"


def pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * float(x):.{digits}f}%"


def num(x: object) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def esc(x: object) -> str:
    return html.escape("" if pd.isna(x) else str(x))


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=FONT_NAME,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#12395b"),
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=5 * mm,
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            parent=base["Heading1"],
            fontName=FONT_NAME,
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=base["Heading2"],
            fontName=FONT_NAME,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=2.5 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.2,
            leading=14.0,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=1.2 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.2,
            leading=9.2,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=0.6 * mm,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.8,
            leading=12.8,
            leftIndent=6 * mm,
            firstLineIndent=-4 * mm,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=0.9 * mm,
        ),
        "th": ParagraphStyle(
            "TableHeaderCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.1,
            leading=8.8,
            alignment=TA_CENTER,
            wordWrap="CJK",
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "TableCellCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.9,
            leading=8.8,
            alignment=TA_CENTER,
            wordWrap="CJK",
            textColor=colors.HexColor("#111827"),
        ),
        "td_left": ParagraphStyle(
            "TableCellLeftCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.9,
            leading=8.8,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#111827"),
        ),
        "note": ParagraphStyle(
            "NoteCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.4,
            leading=12.2,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            borderPadding=5,
            borderWidth=0.5,
            borderColor=colors.HexColor("#d2d6dc"),
            backColor=colors.HexColor("#f5f7fb"),
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=2 * mm,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(esc(text), style)


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph("• " + esc(text), styles["bullet"])


def table(
    data: list[list[object]],
    widths_mm: list[float],
    styles: dict[str, ParagraphStyle],
    left_cols: set[int] | None = None,
) -> Table:
    left_cols = left_cols or set()
    rows = []
    for i, row in enumerate(data):
        cells = []
        for j, cell in enumerate(row):
            if i == 0:
                style = styles["th"]
            else:
                style = styles["td_left"] if j in left_cols else styles["td"]
            cells.append(Paragraph(esc(cell), style))
        rows.append(cells)
    t = Table(rows, colWidths=[w * mm for w in widths_mm], repeatRows=1)
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
    return df.loc[mask].iloc[0]


def build_content() -> tuple[list, str, dict[str, object]]:
    register_font()
    styles = build_styles()
    metrics = pd.read_csv(V202_METRICS)
    presence = pd.read_csv(V202_PRESENCE)
    ci = pd.read_csv(V198_CI)
    release = pd.read_csv(V197_TABLE)
    v201 = pd.read_csv(V201_SUMMARY)
    quality = pd.read_csv(V199_QUALITY)
    operating = pd.read_csv(V192_OPERATING)

    clean_all = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="all_domains")
    clean_old = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="old_data")
    clean_third = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="third_batch")
    clean_ext = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="strict_external")
    full_all = row(metrics, policy="v195_stable_agreement_release", cohort="full_sensitivity", scope="all_domains")
    v185_all = row(metrics, policy="v185_unlabeled_shift_adaptive", cohort="full_sensitivity", scope="all_domains")
    v201_all = row(v201, scope="all_domains")
    v201_third = row(v201, scope="third_batch")
    v198_all = row(ci, policy="v195_stable_fixed_agreement_release", scope="all_domains")
    v198_ext = row(ci, policy="v195_stable_fixed_agreement_release", scope="strict_external")
    v198_v185_all = row(ci, policy="v185_unlabeled_shift_adaptive", scope="all_domains")
    fn_quality = quality.loc[
        quality["direction"].eq("fn_high_to_low")
        & quality["scope"].eq("internal_review_candidates_oof")
    ].sort_values("auroc", ascending=False).iloc[0]

    story: list = []
    story.append(Paragraph("Task7 胸腺大体病理图像二分类：医工交叉论文完整成果与完成度总报告", styles["title"]))
    story.append(Paragraph("版本：2026-05-28；用途：供同伴对照医学顶刊、Nature 子刊与 CCF/计算机方向论文要求", styles["subtitle"]))
    story.append(Paragraph("0. 摘要结论", styles["h1"]))
    story.append(
        para(
            "我们目前已经从单纯分类模型，推进到一套风险可控、跨域可审计、可解释边界明确的大体病理图像诊断框架。"
            "医学方向的核心价值是：在 clean cohort 上，当前主推 v195 selective diagnosis 流程总体 BAcc 99.81%、Acc 99.86%、FN=1、FP=0，"
            "并且严格外部集 108 例达到 BAcc 100.00%、Acc 100.00%，但仍需 51.85% 复核/拒识。"
            "计算机方向的核心价值是：我们不是只报告分类分数，而是建立了 direction-aware routing、稳定高置信放行、支持域自动纠偏、Wilson 安全边界、clean/full cohort 敏感性分析等一整套证据链。",
            styles["body"],
        )
    )
    story.append(
        para(
            "当前最强、最稳的可写主线是 v195：在 v185 风险控制基础上加入稳定高置信一致放行。它把总体复核/拒识率从 58.94% 降到 36.05%（clean cohort 为 36.01%），自动判读数从 287 增到 447，自动错误仍为 1。"
            "v201 进一步证明真正自动纠偏存在小规模正证据：第三批中自动救回 2 例高危漏判，0 误伤。"
            "但自动纠偏目前还不能作为主效率来源；严格讲，我们当前成熟模块是安全放行和风险拒识，自动纠偏仍是候选增强模块。",
            styles["note"],
        )
    )

    story.append(Paragraph("1. 当前研究方向与论文主线", styles["h1"]))
    story.append(
        para(
            "建议将论文主线定义为：面向胸腺大体病理照片的 risk-controlled cross-domain selective diagnosis framework。"
            "这个定义比“训练一个二分类模型”更合适，因为它能同时覆盖临床安全、跨域泛化、模型不确定性、复核工作流和计算机方法创新。"
            "系统输出不是简单的低危/高危二分类，而是四类动作：自动低危、自动高危、自动纠偏、拒识/医生复核。",
            styles["body"],
        )
    )
    story.append(
        table(
            [
                ["模块", "医学问题", "计算机方法", "当前证据状态"],
                ["主分类器", "低危/高危初筛", "DINO/多模型概率、whole/crop、多视角特征聚合", "已有强结果；v185/v195 为主流程"],
                ["direction-aware router", "区分漏诊高危和误升级低危", "FN/FP 方向风险、低置信对照、图像/概念风险排序", "有对照和负向审计；比普通置信度更有解释力"],
                ["稳定放行", "减少医生复核负担", "高置信一致放行、跨 fold 固定规则、greedy 负向审计", "v195 是当前最强效率模块"],
                ["自动纠偏", "复核池中可自动救回的错判", "支持域方向风险翻转、阈值稳定性约束", "v201 有小规模正证据，仍需扩大"],
                ["拒识/复核", "避免不可判读或跨域风险硬判", "选择性诊断、Wilson CI、前瞻样本量规划", "证据较完整，但复核率仍偏高"],
            ],
            [28, 34, 54, 52],
            styles,
            left_cols={1, 2, 3},
        )
    )

    story.append(Paragraph("2. 数据、队列和任务定义", styles["h1"]))
    story.append(
        para(
            "Task7 当前定义为低危 vs 高危二分类：A/AB 等低危归为 low-risk，B2/B3/TC 等归为 high-risk。"
            "旧数据和第三批作为开发/训练/阈值选择来源；严格外部集“胸腺瘤+癌”只作为冻结后外部验证，不用于反向调参。"
            "医生最新确认的特殊组织学病例已固定为 exclusion list，主分析使用 clean cohort，完整队列作为 sensitivity analysis。",
            styles["body"],
        )
    )
    story.append(
        table(
            [
                ["队列", "用途", "n", "低危", "高危", "说明"],
                ["旧数据 clean", "开发/训练/OOF 验证", int(clean_old["n"]), int(clean_old["low_risk_n"]), int(clean_old["high_risk_n"]), "医生剔除 2205101、2307206 后为 283 例"],
                ["第三批", "开发验证/域扩展", int(clean_third["n"]), int(clean_third["low_risk_n"]), int(clean_third["high_risk_n"]), "类别分布更偏低危，保留全部 306 例"],
                ["严格外部", "冻结外部验证", int(clean_ext["n"]), int(clean_ext["low_risk_n"]), int(clean_ext["high_risk_n"]), "不参与调参，仅用于最终验证"],
                ["clean 总体", "主分析", int(clean_all["n"]), int(clean_all["low_risk_n"]), int(clean_all["high_risk_n"]), "当前推荐主文结果"],
                ["full 总体", "敏感性分析", int(full_all["n"]), int(full_all["low_risk_n"]), int(full_all["high_risk_n"]), "保留特殊病例，证明剔除不改变结论"],
            ],
            [25, 32, 18, 18, 18, 68],
            styles,
            left_cols={1, 5},
        )
    )
    story.append(Paragraph("2.1 医生确认的特殊病例剔除", styles["h2"]))
    ex_rows = [["病例号", "医生备注", "当前主流程状态", "模型状态", "处理原则"]]
    for _, r in presence.iterrows():
        status = "命中当前699例" if bool(r["in_current_task7_main"]) else "未命中当前699例"
        if bool(r["in_current_task7_main"]):
            model_state = f"{r['domain']}；{r['task_l6_label']}；预测高危；{'复核池' if str(r['adaptive_review']).lower() == 'true' else '自动放行'}"
        else:
            model_state = "当前主流程未纳入"
        ex_rows.append([r["original_case_id"], r["doctor_note"], status, model_state, "clean 主分析剔除；full 敏感性保留"])
    story.append(table(ex_rows, [22, 26, 32, 55, 55], styles, left_cols={3, 4}))

    story.append(PageBreak())
    story.append(Paragraph("3. 医学领域已经完成的内容", styles["h1"]))
    medical_points = [
        "建立了面向胸腺大体病理照片的低危/高危二分类任务，贴合医生当前最关心的临床问题：高危漏诊和低危误升级。",
        "将医生提出的特殊组织学/混杂病例转化为固定 clean cohort 规则，而不是事后删除结果；同时保留 full cohort 敏感性分析。",
        "输出了医生可读的复核/拒识概念：模型不是对所有照片强行给结论，而是对不确定、质量/域偏移、边界病例保留复核入口。",
        "复核池并不是简单失败集合，而是风险控制工作流的一部分：当前自动判读病例错误极少，复核池承担临床安全阀作用。",
        "完成了典型错例、边界病例、经验标签、质量标签、混淆矩阵、DINOv2 高危漏诊病例等多轮医生沟通材料。",
    ]
    for item in medical_points:
        story.append(bullet(item, styles))

    story.append(Paragraph("3.1 clean cohort 主结果", styles["h2"]))
    rows = [["范围", "n", "低危/高危", "Acc", "BAcc", "F1", "FN", "FP", "复核/拒识", "自动错误"]]
    for label, r in [("旧数据", clean_old), ("第三批", clean_third), ("严格外部", clean_ext), ("总体", clean_all)]:
        rows.append(
            [
                label,
                int(r["n"]),
                f"{int(r['low_risk_n'])}/{int(r['high_risk_n'])}",
                pct(r["accuracy"]),
                pct(r["balanced_accuracy"]),
                pct(r["f1"]),
                int(r["fn"]),
                int(r["fp"]),
                pct(r["review_or_reject_rate"]),
                f"{int(r['auto_error_n'])} ({pct(r['auto_error_rate'])})",
            ]
        )
    story.append(table(rows, [24, 16, 24, 19, 19, 19, 14, 14, 24, 26], styles))
    story.append(
        para(
            "解读：clean cohort 总体 BAcc 99.81%、FN=1、FP=0。严格外部目前点估计为 100%，但样本量只有 108 例，且系统仍拒识/复核 51.85%，因此医学写作应强调风险控制和前瞻验证计划，而不是夸大为完全自动临床诊断。",
            styles["note"],
        )
    )

    story.append(Paragraph("4. 计算机领域已经完成的内容", styles["h1"]))
    comp_points = [
        "从单一分类器升级为多阶段系统：主分类器、direction-aware router、稳定放行、支持域自动纠偏、拒识/复核。",
        "明确区分不同错误方向：FN_high_to_low 对应高危漏诊风险，FP_low_to_high 对应低危误升级风险；这比普通低置信度筛选更贴近临床代价。",
        "完成多种负向审计：直接自动翻转会误伤，per-fold greedy 放行会释放 held-out 错误，后验单例哨兵无法无泄露救回最后 FN。",
        "建立了安全统计边界：对自动判读错误率报告 Wilson 置信区间，并计算前瞻样本量需求。",
        "形成了 clean/full cohort、内部/第三批/严格外部、主流程/候选流程/负向流程的分层证据体系。",
    ]
    for item in comp_points:
        story.append(bullet(item, styles))

    story.append(Paragraph("4.1 关键方法模块与数值证据", styles["h2"]))
    release_rows = [["流程", "证据等级", "BAcc", "复核/拒识", "自动动作", "动作错误", "FN/FP", "意义"]]
    for _, r in release.iterrows():
        if r["scope"] != "all_domains":
            continue
        release_rows.append(
            [
                r["workflow"].replace("v195 stable fixed high-confidence agreement release", "v195 稳定高置信放行").replace("v196 per-fold greedy agreement release audit", "v196 贪心负向审计"),
                r["tier"],
                pct(r["balanced_accuracy"]),
                pct(r["remaining_review_or_reject_rate"]),
                "" if pd.isna(r["auto_action_n"]) else int(r["auto_action_n"]),
                "" if pd.isna(r["action_error_n"]) else int(r["action_error_n"]),
                f"{int(r['fn'])}/{int(r['fp'])}",
                r["interpretation"],
            ]
        )
    story.append(table(release_rows, [38, 31, 18, 23, 18, 18, 17, 43], styles, left_cols={0, 1, 7}))

    story.append(Paragraph("4.2 安全置信区间和自动判读边界", styles["h2"]))
    ci_rows = [["流程", "范围", "自动判读", "自动错误", "错误率", "Wilson95 上界", "复核/拒识"]]
    for policy_label, policy in [("v185 原始风险控制", "v185_unlabeled_shift_adaptive"), ("v195 稳定放行", "v195_stable_fixed_agreement_release")]:
        for scope_label, scope in [("总体", "all_domains"), ("严格外部", "strict_external")]:
            r = row(ci, policy=policy, scope=scope)
            ci_rows.append(
                [
                    policy_label,
                    scope_label,
                    int(r["auto_decision_n"]),
                    int(r["auto_error_n"]),
                    pct(r["auto_error_rate"]),
                    pct(r["wilson95_high"]),
                    pct(r["review_or_reject_rate"]),
                ]
            )
    story.append(table(ci_rows, [34, 22, 24, 22, 22, 30, 26], styles))
    story.append(
        para(
            f"v195 相比 v185 的关键收益：总体自动判读从 {int(v198_v185_all['auto_decision_n'])} 例增至 {int(v198_all['auto_decision_n'])} 例，"
            f"复核/拒识从 {pct(v198_v185_all['review_or_reject_rate'])} 降至 {pct(v198_all['review_or_reject_rate'])}，"
            f"自动错误 Wilson95 上界从 {pct(v198_v185_all['wilson95_high'])} 降至 {pct(v198_all['wilson95_high'])}。"
            f"严格外部自动判读从 14 例增至 {int(v198_ext['auto_decision_n'])} 例，Wilson95 上界降至 {pct(v198_ext['wilson95_high'])}。",
            styles["note"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. 自动纠偏：已有正证据和边界", styles["h1"]))
    story.append(
        para(
            "我们已经尝试让复核池中的病例不只是交给医生，而是由二阶段模型自动纠偏。结果必须分层表述：全域直接翻转不安全，支持域稳定翻转有小规模正证据。",
            styles["body"],
        )
    )
    correction_rows = [
        ["实验", "目的", "结果", "结论"],
        ["v199 全域方向翻转", "用 FN/FP 方向风险直接自动改判", "救回 1 例、误伤 1 例；BAcc 降至约 99.70%", "不能作为正式自动纠偏器"],
        ["v200 同域支持约束", "只在同域训练折支持时翻转", "仍因阈值过激出现 1 救回 + 1 误伤", "需要稳定固定阈值"],
        ["v201 稳定支持域翻转", "仅对 third_batch 稳定 FN 风险规则生效", "自动救回 2420341(B2)、2421906(TC)，0 误伤", "第一个可写的自动纠偏正证据，但规模小"],
    ]
    story.append(table(correction_rows, [34, 44, 54, 48], styles, left_cols={1, 2, 3}))
    story.append(
        para(
            f"方向风险模型不是没有信号：最佳 FN_high_to_low 模型在内部复核候选上的 AUROC 为 {float(fn_quality['auroc']):.3f}，AP 为 {float(fn_quality['average_precision']):.3f}。"
            f"v201 最终在第三批自动翻转 {int(v201_third['flip_n'])} 例，救回 {int(v201_third['rescued_n'])} 例，0 误伤。"
            "但它只把总体复核率从 58.94% 小幅降至 58.66%，因此不能替代 v195 作为效率主模块。",
            styles["note"],
        )
    )

    story.append(Paragraph("6. 失败实验和边界同样是成果", styles["h1"]))
    boundary_rows = [
        ["边界/负向结果", "发现", "论文价值"],
        ["直接自动翻转", "v174/v175/v199 显示翻转容易误伤，不能成熟替代复核", "证明我们没有为了自动化强行牺牲安全"],
        ["per-fold greedy 放行", "v196 复核率可降至 30.62%，但释放 3 个 held-out 错误", "支持固定稳定规则优于局部贪心"],
        ["最后残余 FN 哨兵", "v190/v191 无泄露哨兵未能救回 2516531", "明确后验单例调参不可写入正式方法"],
        ["严格外部统计边界", "点估计 100%，但样本量小，Wilson 上界仍需报告", "符合医学论文对安全结论的审慎要求"],
        ["clean cohort 剔除", "剔除特殊组织学不改变主结果", "避免 reviewer 质疑结果靠剔除病例变好"],
    ]
    story.append(table(boundary_rows, [42, 70, 68], styles, left_cols={0, 1, 2}))

    story.append(Paragraph("7. 医学期刊路线完成度", styles["h1"]))
    medical_completion_rows = [
        ["维度", "当前完成度", "已有证据", "还需补强"],
        ["临床问题重要性", "较强", "低危/高危二分类符合医生当前需求；TC/B2/B3 高危漏诊是核心风险", "需要医生进一步确认任务定义和纳排标准"],
        ["数据队列", "中等偏强", "旧数据、第三批、严格外部；clean/full 敏感性分析已建立", "严格外部样本量仍小，最好增加前瞻/多中心外部"],
        ["模型效果", "强", "clean 总体 BAcc 99.81%，严格外部 BAcc 100%，FN=1、FP=0", "必须说明 selective diagnosis，不是完全自动全量诊断"],
        ["临床安全", "较强", "自动错误率、Wilson CI、复核/拒识机制、失败边界均已量化", "需要定义复核流程在真实临床中的耗时和可接受比例"],
        ["可解释性", "中等", "经验标签、质量标签、错例清单、方向错误分析、医生讨论记录", "还需更系统的注意力/区域证据和医生盲评"],
        ["前瞻验证", "不足", "已有样本量规划", "医学顶刊通常需要前瞻、多中心、独立临床评估"],
    ]
    story.append(table(medical_completion_rows, [28, 25, 72, 55], styles, left_cols={0, 2, 3}))
    story.append(
        para(
            "医学期刊角度：目前已经接近一篇强医学 AI 诊断研究的 retrospective + external validation 形态。"
            "如果目标类比 Lancet 系列，最大短板不是当前分数，而是前瞻性、多中心、临床工作流验证和医生对照实验。",
            styles["note"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("8. 计算机/Nature 子刊/CCF 路线完成度", styles["h1"]))
    computing_completion_rows = [
        ["维度", "当前完成度", "已有证据", "还需补强"],
        ["任务新颖性", "较强", "大体病理照片用于胸腺低危/高危诊断，公开先例少", "需要系统梳理相关工作，证明任务稀缺性和临床价值"],
        ["方法创新", "中等", "risk-controlled selective workflow、direction-aware router、stable release、support-domain correction", "需要把模块统一成清晰算法，而不是实验堆叠"],
        ["对照实验", "中等", "低置信度、greedy release、direct flip、固定高安全流程、clean/full 对照", "还需系统比较 DINOv2/DINOv3/PLIP/BiomedCLIP/ConvNeXt 等统一基线"],
        ["消融完整性", "中等偏强", "v195 vs v196、v199-v201、v185/v182/v118、Wilson CI", "需要将消融整理成正式表，而非散落版本号"],
        ["跨域泛化", "中等", "第三批和严格外部均有结果；无标签 severe-shift gate 已验证", "外部域数量少；不能用单一外部集支撑强泛化 claim"],
        ["算法可复现", "中等", "服务器脚本 v185-v204、输出表、notes 完整", "需要整理统一入口、配置文件、固定随机种子和代码包"],
        ["计算机硬度", "仍需提升", "已从分类转向选择性诊断和风险控制", "需要形成理论/算法图、强 baseline、统计显著性和外部鲁棒性实验"],
    ]
    story.append(table(computing_completion_rows, [28, 25, 72, 55], styles, left_cols={0, 2, 3}))
    story.append(
        para(
            "计算机论文角度：现在最有潜力的贡献不是“我们调到了 99.8%”，而是“在小样本、跨域、临床风险不对称的医学图像任务中，如何构建安全可控的选择性诊断系统”。"
            "如果要投 Nature 子刊或 CCF，后续必须把方法抽象成可复用框架，并用强对照证明 direction-aware routing 与 stable release 的必要性。",
            styles["note"],
        )
    )

    story.append(Paragraph("9. Claim-Evidence Map", styles["h1"]))
    claim_rows = [
        ["主张", "证据", "状态"],
        ["我们已经建立 clean cohort 主分析和 full cohort 敏感性分析", "v202：clean n=697；full n=699；剔除不改变主结果", "supported"],
        ["v195 是当前最强效率增强流程", f"clean BAcc {pct(clean_all['balanced_accuracy'])}；review/reject {pct(clean_all['review_or_reject_rate'])}；auto error {int(clean_all['auto_error_n'])}", "supported"],
        ["稳定高置信放行优于贪心放行", "v195 action-error=0；v196 复核率更低但释放 3 个 held-out 错误", "supported"],
        ["自动纠偏已经完全成熟", "v201 仅救回 2 例；v199/v200 有误伤", "not supported"],
        ["方向风险具有可用信号", f"FN_high_to_low review candidates AUROC {float(fn_quality['auroc']):.3f}", "supported as candidate"],
        ["严格外部已经证明强泛化", "strict external BAcc 100%，但 n=108 且复核/拒识 51.85%", "partially supported"],
        ["系统可用于完全自动临床诊断", "仍有 36.01% clean cohort 复核/拒识，自动错误 1 例", "not supported"],
    ]
    story.append(table(claim_rows, [52, 88, 32], styles, left_cols={0, 1, 2}))

    story.append(Paragraph("10. 下一阶段建议", styles["h1"]))
    next_steps = [
        "把 v185-v204 统一成一个正式框架名称、算法流程图和固定实验协议，减少版本号式叙述。",
        "在服务器上继续补强视觉基础模型对照：DINOv2、DINOv3、PLIP、BiomedCLIP、ConvNeXt、Swin 等，统一 clean cohort 协议。",
        "将 v195 与 v201 合成为一个正式三阶段 workflow：主分类器 → 稳定放行 → 支持域纠偏/拒识，并重新计算 clean/full/外部指标。",
        "补医生对照实验：医生盲看、模型自动放行、模型复核提示、医生复核后四种模式比较。",
        "补前瞻样本量设计：严格外部自动判读若要 Wilson95 上界低于 5%，需要更多自动判读病例。",
        "整理论文 Methods：数据纳排、图像选择、多视角特征、风险路由、放行规则、纠偏规则、拒识定义、统计检验。",
        "整理论文 Results：主结果、外部验证、消融、负向审计、错误边界、clean/full 敏感性分析、医生复核材料。",
    ]
    for item in next_steps:
        story.append(bullet(item, styles))

    story.append(Paragraph("11. 给同伴检索期刊论文时的对照重点", styles["h1"]))
    story.append(
        table(
            [
                ["检索方向", "重点看什么", "我们目前该如何对照"],
                ["医学顶刊 / Lancet 系列", "数据量、多中心、前瞻性、医生对照、临床终点、安全声明", "我们分数和安全框架强，但前瞻和多中心不足"],
                ["Nature Medicine / Nature Biomedical Engineering", "临床需求 + 方法新颖 + 外部泛化 + 工作流验证", "需要强化跨域和方法统一，不只报模型分数"],
                ["Medical Image Analysis / TMI", "算法严谨、强 baseline、消融、泛化、可复现", "需要补统一 baseline 和正式算法表述"],
                ["CV/ML CCF 方向", "技术创新可迁移性、理论/算法贡献、大规模对照", "数据量较小是硬伤，应强调小样本风险控制和选择性诊断框架"],
            ],
            [35, 65, 70],
            styles,
            left_cols={0, 1, 2},
        )
    )

    story.append(Paragraph("12. 总体判断", styles["h1"]))
    story.append(
        para(
            "如果按医学 AI retrospective study 看，我们已经具备比较完整的主结果、外部验证、复核安全机制和医生沟通材料，下一步主要补前瞻/多中心和医生对照。"
            "如果按计算机论文看，我们已经从单模型准确率升级到了风险控制框架，但还需要将模块统一成可复用方法，并用更强 baseline 和更系统消融证明计算机贡献。"
            "当前最稳妥的论文定位是：一种面向跨域大体病理图像的风险可控选择性诊断框架，而不是一个单纯胸腺分类模型。",
            styles["note"],
        )
    )

    md_lines = [
        "# Task7 医工交叉论文完整成果与完成度总报告",
        "",
        "## 摘要结论",
        f"- clean cohort v195：n={int(clean_all['n'])}，BAcc {pct(clean_all['balanced_accuracy'])}，Acc {pct(clean_all['accuracy'])}，FN={int(clean_all['fn'])}，FP={int(clean_all['fp'])}，复核/拒识 {pct(clean_all['review_or_reject_rate'])}。",
        f"- strict external：n={int(clean_ext['n'])}，BAcc {pct(clean_ext['balanced_accuracy'])}，复核/拒识 {pct(clean_ext['review_or_reject_rate'])}。",
        f"- v195 相比 v185：自动判读 {int(v198_v185_all['auto_decision_n'])}->{int(v198_all['auto_decision_n'])}，复核/拒识 {pct(v198_v185_all['review_or_reject_rate'])}->{pct(v198_all['review_or_reject_rate'])}。",
        f"- v201：自动纠偏救回 {int(v201_all['rescued_n'])} 例，0 误伤，但规模仍小。",
        "",
        "## 总体判断",
        "医学路线已接近强 retrospective + external validation 研究；计算机路线还需要更强 baseline、统一算法表达和更多跨域验证。",
    ]
    meta = {
        "clean_v195_all_bacc": float(clean_all["balanced_accuracy"]),
        "clean_v195_all_review_rate": float(clean_all["review_or_reject_rate"]),
        "clean_v195_all_fn": int(clean_all["fn"]),
        "clean_v195_all_fp": int(clean_all["fp"]),
        "v195_auto_decision_n": int(v198_all["auto_decision_n"]),
        "v195_wilson95_high": float(v198_all["wilson95_high"]),
        "strict_external_v195_bacc": float(clean_ext["balanced_accuracy"]),
        "strict_external_review_rate": float(clean_ext["review_or_reject_rate"]),
        "v201_rescued_n": int(v201_all["rescued_n"]),
        "v201_hurt_n": int(v201_all["hurt_n"]),
    }
    return story, "\n".join(md_lines), meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    story, md_text, meta = build_content()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=13 * mm,
        rightMargin=13 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    doc.build(story)
    MD_PATH.write_text(md_text, encoding="utf-8")
    (OUT_DIR / "v204_report_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v204] wrote {PDF_PATH}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
