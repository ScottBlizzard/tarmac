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
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v206_joint_team_report_20260531"
REPORT_DIR = ROOT / "汇报"
PDF_PATH = REPORT_DIR / "2026-05-31_Task7项目进展与论文完成度综合报告_医生导师共读版.pdf"
MD_PATH = REPORT_DIR / "2026-05-31_Task7项目进展与论文完成度综合报告_医生导师共读版.md"

V202_METRICS = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_clean_vs_full_policy_metrics.csv"
V198_CI = ROOT / "outputs" / "grosspath_rc_v198_release_compressed_safety_ci_20260527" / "v198_release_compressed_safety_ci.csv"
V197_TABLE = ROOT / "outputs" / "grosspath_rc_v197_stable_release_evidence_update_20260527" / "v197_stable_release_evidence_table.csv"
V201_SUMMARY = ROOT / "outputs" / "grosspath_rc_v201_stable_supported_domain_flip_20260527" / "v201_stable_supported_domain_flip_summary.csv"
TASK7_FAIR = ROOT / "reports" / "ThymicGross" / "task7_fair_benchmark_overall_results_2026-05-13.csv"
TASK6_FAIR = ROOT / "reports" / "ThymicGross" / "task6_fair_benchmark_overall_results_2026-05-13.csv"

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7Joint"


def pct(x: object, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def esc(x: object) -> str:
    return html.escape("" if pd.isna(x) else str(x))


def row(df: pd.DataFrame, **kwargs) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for k, v in kwargs.items():
        mask &= df[k].eq(v)
    matched = df.loc[mask]
    if matched.empty:
        raise KeyError(kwargs)
    return matched.iloc[0]


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def styles() -> dict[str, ParagraphStyle]:
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
            spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_NAME,
            fontSize=13.2,
            leading=18,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=3.5 * mm,
            spaceAfter=1.8 * mm,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_NAME,
            fontSize=10.8,
            leading=15,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=2.3 * mm,
            spaceAfter=1 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.8,
            leading=13.2,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=1.1 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.4,
            leading=12.3,
            leftIndent=6 * mm,
            firstLineIndent=-4 * mm,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=0.7 * mm,
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
            spaceAfter=1.8 * mm,
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.7,
            alignment=TA_CENTER,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "td": ParagraphStyle(
            "td",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.7,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "td_left": ParagraphStyle(
            "td_left",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.7,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "td_right": ParagraphStyle(
            "td_right",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.8,
            leading=8.7,
            alignment=TA_RIGHT,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
    }


def para(text: str, st: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(esc(text), st[name])


def bullet(text: str, st: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph("• " + esc(text), st["bullet"])


def table(
    rows: list[list[object]],
    widths: list[float],
    st: dict[str, ParagraphStyle],
    left_cols: set[int] | None = None,
    right_cols: set[int] | None = None,
) -> Table:
    left_cols = left_cols or set()
    right_cols = right_cols or set()
    data = []
    for i, r in enumerate(rows):
        row_cells = []
        for j, c in enumerate(r):
            style = "th" if i == 0 else ("td_left" if j in left_cols else "td_right" if j in right_cols else "td")
            row_cells.append(Paragraph(esc(c), st[style]))
        data.append(row_cells)
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
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


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(13 * mm, 8 * mm, "Task7 胸腺大体病理图像项目进展与论文完成度综合报告")
    canvas.drawRightString(198 * mm, 8 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_story() -> tuple[list, dict[str, object], str]:
    register_font()
    st = styles()
    metrics = pd.read_csv(V202_METRICS)
    ci = pd.read_csv(V198_CI)
    release = pd.read_csv(V197_TABLE)
    v201 = pd.read_csv(V201_SUMMARY)
    task7_fair = pd.read_csv(TASK7_FAIR)
    task6_fair = pd.read_csv(TASK6_FAIR)

    clean_all = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="all_domains")
    clean_old = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="old_data")
    clean_third = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="third_batch")
    clean_ext = row(metrics, policy="v195_stable_agreement_release", cohort="clean_primary", scope="strict_external")
    full_all = row(metrics, policy="v195_stable_agreement_release", cohort="full_sensitivity", scope="all_domains")
    v185_ci = row(ci, policy="v185_unlabeled_shift_adaptive", scope="all_domains")
    v195_ci = row(ci, policy="v195_stable_fixed_agreement_release", scope="all_domains")
    v195_ext_ci = row(ci, policy="v195_stable_fixed_agreement_release", scope="strict_external")
    v201_all = row(v201, scope="all_domains")
    v201_third = row(v201, scope="third_batch")
    best_task7 = task7_fair.sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_task6 = task6_fair.sort_values("accuracy", ascending=False).iloc[0]

    story: list = []
    story.append(para("Task7 胸腺大体病理图像低危/高危风险分层", st, "title"))
    story.append(para("项目进展与论文完成度综合报告", st, "title"))
    story.append(para("医生团队与导师团队共读版｜2026-05-31", st, "subtitle"))

    story.append(para("报告目的", st, "h1"))
    story.append(
        para(
            "这份报告把项目从早期复现到当前风险控制工作流的进展合并成一个共读版本。医生团队可以重点看临床任务、队列边界、当前结果和需要医学确认的问题；导师团队可以重点看计算机贡献、论文完成度、证据短板和下一步可形成论文创新的位置。",
            st,
        )
    )
    story.append(
        para(
            "当前最重要的表述边界是：我们现在的高分结果属于 risk-controlled selective diagnosis，即模型对稳定病例自动放行，对不稳定病例复核或拒识；不能写成全量病例已经完全自动诊断 99% 以上。这个边界是医学安全和论文可信度的核心。",
            st,
            "note",
        )
    )

    story.append(para("一页结论", st, "h1"))
    for x in [
        f"项目进度已经非常完整：从 SuRImage 复现、Task1-6、Task7、课程学习、经验标签、第三批适配、严格外部压力测试，到 GrossPath-RC 风险控制工作流和 clean/full cohort，已经形成连续证据链。",
        f"当前主流程 v195 在 clean cohort 上 n={int(clean_all['n'])}，Acc {pct(clean_all['accuracy'])}，BAcc {pct(clean_all['balanced_accuracy'])}，F1 {pct(clean_all['f1'])}，FN={int(clean_all['fn'])}，FP={int(clean_all['fp'])}。",
        f"v195 clean cohort 自动覆盖 {pct(clean_all['auto_decision_rate'])}，复核/拒识 {pct(clean_all['review_or_reject_rate'])}，自动错误 {int(clean_all['auto_error_n'])} 例；全域自动错误 Wilson95 上界 {pct(v195_ci['wilson95_high'])}。",
        f"严格外部集 n={int(clean_ext['n'])}，当前 workflow 点估计 BAcc {pct(clean_ext['balanced_accuracy'])}，但复核/拒识 {pct(clean_ext['review_or_reject_rate'])}，说明外部泛化结果必须按选择性诊断解释。",
        f"医学期刊方向：已经具备较完整的回顾性 AI 研究雏形，短板是前瞻、多中心、医生 reader study 和正式临床工作流评价。",
        f"计算机/Nature 子刊方向：潜力不在单纯 DINO 分类，而在小样本、跨域、风险可控、可拒识的大体病理图像诊断框架；短板是算法表达、统一 baseline、消融和可复现代码还要继续压实。",
    ]:
        story.append(bullet(x, st))

    story.append(para("当前推荐论文主线", st, "h1"))
    story.append(
        table(
            [
                ["对象", "建议主线", "为什么这样写"],
                ["医生团队", "胸腺大体病理照片能够为低危/高危风险分层提供辅助信息；模型对稳定病例可自动提示，对不确定病例建议复核。", "符合临床安全逻辑，不把模型包装成替代病理医生。"],
                ["医学期刊", "回顾性队列 + 扩展开发队列 + 严格外部压力测试 + 风险可控选择性诊断工作流。", "医学期刊更看重临床问题、队列、外部验证、安全和医生可接受性。"],
                ["导师/计算机方向", "GrossPath-RC：面向小样本跨域大体病理图像的风险可控选择性诊断框架。", "比单纯调模型更有方法贡献，能连接 selective prediction、domain shift、uncertainty 和 clinical workflow。"],
                ["Nature 子刊/医工交叉", "把临床新任务、大体图像数据、跨域失败、风险控制工作流和医生复核闭环组织成一个完整系统。", "必须体现医学问题和计算机方法都成立，不能只靠一个高分表。"],
            ],
            [30, 87, 68],
            st,
            left_cols={1, 2},
        )
    )

    story.append(PageBreak())
    story.append(para("1. 项目进度总览", st, "h1"))
    story.append(
        para(
            "项目不是从一个模型结果直接跳到 v195，而是经历了九个阶段。这个过程本身很重要，因为它说明我们已经系统排查了模型、数据、任务定义、外部泛化、医生复核和安全边界。",
            st,
        )
    )
    story.append(
        table(
            [
                ["阶段", "完成内容", "阶段结论", "当前用途"],
                ["1 基础复现", "SuRImage、SE-ResNeXt、whole/crop、5-fold 病例级评估", "大体图有粗分层信号，但传统 CNN/Fusion 不稳定", "早期可行性和复现对照"],
                ["2 Task1-6", "建立多任务体系，验证 DINOv2、PLIP、Task5/Task6", "DINOv2 是稳定强表征，Task6 六分类受样本量限制", "说明任务选择和强基线"],
                ["3 Task7 冲刺", "低危/高危二分类、多图、视图、cut/outer/mixed、backbone 替换", "直接二分类更贴近医生需求，但全量强制分类仍不够", "确定 Task7 主任务"],
                ["4 课程学习", "easy/medium/hard_core 分层、经验标签、hard 复核清单", "瓶颈集中在核心 hard，不是所有病例都难", "解释错误来源和复核必要性"],
                ["5 二阶段复核", "一次通过、复核池、候选纠偏器、多模型分歧", "复核思路有效，但早期纠偏器未成熟", "风险控制工作流前身"],
                ["6 第三批与外部", "第三批 306 例适配，严格外部 108 例压力测试", "强制分类跨域下降，外部必须谨慎解释", "推动转向 risk-controlled workflow"],
                ["7 框架重构", "明确 GrossPath-RC、数据边界、可写/不可写 claim", "论文主线从单模型变成风险可控框架", "计算机贡献雏形"],
                ["8 v0-v2 工作流", "选择性诊断、质量门控、概念、stacking、guard", "多模型一致性和质量门控有用，stacking/guard 不是主线", "形成候选模块和负结果证据"],
                ["9 v195/clean cohort", "稳定高置信放行、Wilson CI、clean/full、v201 候选纠偏", "v195 是当前最稳主流程，v201 仅作候选增强", "当前主结果和论文核心"],
            ],
            [22, 58, 58, 47],
            st,
            left_cols={1, 2, 3},
        )
    )

    story.append(para("2. 数据与队列边界", st, "h1"))
    story.append(
        table(
            [
                ["队列", "规模", "角色", "能怎么写", "不能怎么写"],
                ["早期 Task1-6", "160 例/172 图", "早期任务体系和强表征验证", "早期开发队列", "不能和后期 Task7 直接横比"],
                ["Task7 旧数据", "285 例", "内部开发、OOF、课程学习、hard 分层", "内部开发/OOF 队列", "不能称外部验证"],
                ["第三批", "306 例", "扩展开发/适配队列", "时间外或扩展开发队列", "不能再称 strict external"],
                ["严格外部", "108 例", "冻结外部压力测试", "严格外部/外部压力测试", "不能用调参后最高分冒充干净验证"],
                ["clean cohort", "697 例", "医生确认特殊病例后的主分析", "主分析队列", "不能写成删错例提分"],
                ["full cohort", "699 例", "完整敏感性分析", "full sensitivity", "不能替代 clean 主口径"],
            ],
            [30, 23, 48, 42, 42],
            st,
            left_cols={2, 3, 4},
        )
    )
    story.append(
        para(
            "医生已提出 5 个特殊组织学或混杂病例用于 clean/full 处理，其中当前 Task7 主队列实际命中 2205101 和 2307206 两例。clean 和 full 结果几乎一致，说明剔除规则没有人为改变主结论。",
            st,
            "note",
        )
    )

    story.append(PageBreak())
    story.append(para("3. 医学团队关心的问题与当前答案", st, "h1"))
    story.append(
        table(
            [
                ["医生关心的问题", "我们已经做了什么", "目前答案", "仍需医生支持"],
                ["二分类是否最重要", "将 Task2/3 思路合并为 Task7 低危/高危；TC/B2/B3 归高危", "Task7 是最适合临床汇报的主任务", "确认最终低危/高危映射和边界亚型"],
                ["模型错在哪里", "混淆矩阵、DINO 漏诊清单、hard_core 清单、典型错例 PDF", "错误集中在边界、线索冲突和核心 hard", "对 hard_core 是否可由大体图判断做医生复核"],
                ["多图/切面图怎么处理", "追踪图1/图2、切面/外观/混合视图，尝试视图 specialist", "统一选图 SOP 比盲目多图混合更重要", "最终确认多图病例优先使用哪类图"],
                ["照片质量和背景是否影响", "质量标签、严格外部质量门控、蓝板/曝光讨论", "质量会影响，但不是外部下降的唯一原因", "建立重拍标准和可判读标准"],
                ["模型是否知道肿瘤区域/大小", "讨论 ROI、尺子、大小变量，尚未形成成熟模块", "可能有价值，但需避免引入偏置", "提供尺寸或 ROI 标注可作为后续增强"],
                ["结果能否临床使用", "建立自动放行 + 复核/拒识工作流", "可作为辅助判读框架，不是替代病理诊断", "设计医生 reader study 和真实流程评价"],
            ],
            [37, 55, 45, 45],
            st,
            left_cols={0, 1, 2, 3},
        )
    )

    story.append(para("4. 当前主结果", st, "h1"))
    story.append(
        table(
            [
                ["口径", "n", "低危/高危", "Acc", "BAcc", "F1", "FN/FP", "自动覆盖", "复核/拒识", "自动错"],
                ["旧数据 clean", int(clean_old["n"]), f"{int(clean_old['low_risk_n'])}/{int(clean_old['high_risk_n'])}", pct(clean_old["accuracy"]), pct(clean_old["balanced_accuracy"]), pct(clean_old["f1"]), f"{int(clean_old['fn'])}/{int(clean_old['fp'])}", pct(clean_old["auto_decision_rate"]), pct(clean_old["review_or_reject_rate"]), int(clean_old["auto_error_n"])],
                ["第三批", int(clean_third["n"]), f"{int(clean_third['low_risk_n'])}/{int(clean_third['high_risk_n'])}", pct(clean_third["accuracy"]), pct(clean_third["balanced_accuracy"]), pct(clean_third["f1"]), f"{int(clean_third['fn'])}/{int(clean_third['fp'])}", pct(clean_third["auto_decision_rate"]), pct(clean_third["review_or_reject_rate"]), int(clean_third["auto_error_n"])],
                ["严格外部", int(clean_ext["n"]), f"{int(clean_ext['low_risk_n'])}/{int(clean_ext['high_risk_n'])}", pct(clean_ext["accuracy"]), pct(clean_ext["balanced_accuracy"]), pct(clean_ext["f1"]), f"{int(clean_ext['fn'])}/{int(clean_ext['fp'])}", pct(clean_ext["auto_decision_rate"]), pct(clean_ext["review_or_reject_rate"]), int(clean_ext["auto_error_n"])],
                ["clean 总体", int(clean_all["n"]), f"{int(clean_all['low_risk_n'])}/{int(clean_all['high_risk_n'])}", pct(clean_all["accuracy"]), pct(clean_all["balanced_accuracy"]), pct(clean_all["f1"]), f"{int(clean_all['fn'])}/{int(clean_all['fp'])}", pct(clean_all["auto_decision_rate"]), pct(clean_all["review_or_reject_rate"]), int(clean_all["auto_error_n"])],
                ["full 总体", int(full_all["n"]), f"{int(full_all['low_risk_n'])}/{int(full_all['high_risk_n'])}", pct(full_all["accuracy"]), pct(full_all["balanced_accuracy"]), pct(full_all["f1"]), f"{int(full_all['fn'])}/{int(full_all['fp'])}", pct(full_all["auto_decision_rate"]), pct(full_all["review_or_reject_rate"]), int(full_all["auto_error_n"])],
            ],
            [23, 13, 20, 16, 16, 16, 15, 18, 19, 14],
            st,
        )
    )
    story.append(
        para(
            f"v195 相比 v185 的主要进步不是让分数从 99.81% 再提高，而是在安全基本不变的前提下，把全域自动判读从 {int(v185_ci['auto_decision_n'])} 例提高到 {int(v195_ci['auto_decision_n'])} 例，复核/拒识从 {pct(v185_ci['review_or_reject_rate'])} 降到 {pct(v195_ci['review_or_reject_rate'])}。这对临床工作流更有意义。",
            st,
            "note",
        )
    )

    story.append(PageBreak())
    story.append(para("5. 计算机方向：我们真正做出的框架", st, "h1"))
    story.append(
        para(
            "导师团队需要看到的是：我们不是只把 DINO 特征拿来做二分类，而是在小样本、跨域、外部图像质量不稳定、临床风险不对称的约束下，逐步形成了一个可审计的风险控制框架。",
            st,
        )
    )
    story.append(
        table(
            [
                ["模块", "解决的问题", "当前证据", "论文价值"],
                ["强视觉表征", "大体图是否有可学习信号", f"早期 Task7 七模型中最佳 DINOv2 vitb14 whole BAcc {pct(best_task7['balanced_accuracy'])}；Task6 最佳 Acc {pct(best_task6['accuracy'])}", "证明任务基础和强 baseline"],
                ["难度分层", "为什么全量分类上不去", "easy+medium 约 96.2%，非核心 hard 约 91.4%，核心 hard 约 26.2%", "说明瓶颈集中在核心 hard"],
                ["经验/概念标签", "如何解释干扰线索和边界病例", "弱标签直接训练不稳定，高可信标签有信号但不够稳定", "适合解释和医生沟通，不宜夸大为主模型"],
                ["选择性诊断", "哪些病例可以自动处理，哪些应复核", "v195 clean 自动覆盖 63.99%，复核/拒识 36.01%，自动错 1", "连接 selective prediction 和临床复核"],
                ["稳定放行", "如何避免贪心调参造成假提升", "v196 复核更低但释放 3 个 held-out 错误，因此否定", "体现安全优先和负结果审计"],
                ["域偏移/外部压力", "为什么外部强制分类下降", "严格外部强制分类曾明显低于域内，后期 workflow 通过拒识控制风险", "支撑 domain shift 和 OOD 风险控制"],
                ["自动纠偏候选", "复核池中能否由模型救回一部分", f"v201 第三批救回 {int(v201_third['rescued_n'])} 例，误伤 {int(v201_third['hurt_n'])} 例", "有潜力，但当前只能写候选增强"],
            ],
            [30, 42, 60, 50],
            st,
            left_cols={1, 2, 3},
        )
    )

    story.append(para("6. 负结果为什么重要", st, "h1"))
    for x in [
        "cut/outer/mixed 分开训练没有稳定超过全量学习，说明单凭“视图更统一”不能保证性能提升。",
        "ConvNeXt、Swin、B3 等 backbone 替换没有稳定超过 DINO 主线，说明当前瓶颈不是简单换模型。",
        "大规模经验弱标签噪声较大，直接加入训练不稳定；少量高可信标签更适合做解释和辅助。",
        "贪心降低复核率会释放 held-out 错误，说明论文不能只选漂亮数字，必须把安全边界写清楚。",
        "严格外部强制分类下降说明跨域泛化是真问题，也正是 risk-controlled workflow 的必要性来源。",
    ]:
        story.append(bullet(x, st))

    story.append(PageBreak())
    story.append(para("7. 论文完成度判断", st, "h1"))
    story.append(
        table(
            [
                ["方向", "当前完成度", "已经具备", "主要短板", "最适合的下一步"],
                ["普通医学 AI 回顾性研究", "较高", "明确临床任务、队列分层、clean/full、外部压力测试、医生沟通材料、主结果清楚", "前瞻、多中心、医生对照不足", "补 reader study 和医生确认材料"],
                ["柳叶刀/顶级医学期刊路线", "中等", "临床问题有意义，安全工作流清楚", "样本量、多中心、前瞻、真实临床终点不足", "设计前瞻验证和多中心扩展"],
                ["医工交叉/Nature 子刊路线", "中等偏上雏形", "临床新任务 + 风险控制框架 + 外部压力测试 + 负结果审计", "方法还需统一抽象，外部域少，缺 reader study", "把 GrossPath-RC 写成系统框架，补消融和外部验证"],
                ["TMI/MIA/CCF 计算机路线", "中等", "有 selective diagnosis、domain shift、stable release、Wilson CI", "强 baseline、算法伪代码、统计显著性、可复现代码需要补", "统一协议重跑/整理 baseline 和消融"],
            ],
            [38, 25, 55, 48, 42],
            st,
            left_cols={0, 2, 3, 4},
        )
    )
    story.append(
        para(
            "综合判断：医学方向已经接近一篇较完整的回顾性医学 AI 工作；如果要冲更高层级期刊，最缺的是前瞻、多中心和医生 reader study。计算机方向已有可发展的框架，但还不能只靠现有版本号堆结果，必须统一成一个可复现算法并补强对照。",
            st,
            "note",
        )
    )

    story.append(para("8. 现在可以对外说什么，不能说什么", st, "h1"))
    story.append(
        table(
            [
                ["可说", "不能说"],
                ["我们已经完成从单模型分类到风险可控选择性诊断工作流的转变。", "不能说模型已对所有病例全自动诊断达到 99% 或 100%。"],
                ["大体图像中存在低危/高危风险分层信号，稳定病例可由模型自动辅助判断。", "不能说大体图像已经可以替代病理诊断或 WHO 完整分型。"],
                ["第三批是扩展开发/适配队列，说明同院新批次也存在分布变化。", "不能把第三批称作严格外部验证。"],
                ["严格外部集是压力测试，workflow 点估计好，但复核/拒识比例高。", "不能单独报严格外部 100% 点估计而不报复核比例。"],
                ["clean cohort 是医生确认任务边界后的主分析，full cohort 用于敏感性分析。", "不能写成删除模型错例后提分。"],
                ["v201 自动纠偏有初步正证据。", "不能写成自动纠偏模块已经成熟。"],
            ],
            [92, 92],
            st,
            left_cols={0, 1},
        )
    )

    story.append(PageBreak())
    story.append(para("9. 需要医生团队确认的问题", st, "h1"))
    for x in [
        "确认 Task7 低危/高危最终映射：哪些病理亚型必须归高危，哪些边界亚型需要单列说明。",
        "确认 clean cohort exclusion list 的医学理由，最好形成正式会议记录或医生签字版说明。",
        "对 hard_core 持续错判病例做小规模复核：判断哪些是图像质量/选图问题，哪些是大体图本身不可稳定判别。",
        "共同制定图像采集 SOP：切面图优先级、多图选择、背景板、光照、主体占比、是否需要重拍。",
        "设计 reader study：医生 alone、AI alone、AI 自动放行 + 医生复核、AI 提示后医生复核几种模式对照。",
    ]:
        story.append(bullet(x, st))

    story.append(para("10. 需要导师团队决定的问题", st, "h1"))
    for x in [
        "论文主投方向：优先做医学 AI 回顾性研究，还是继续补计算机方法后冲医工交叉/Nature 子刊。",
        "计算机贡献是否固定为 GrossPath-RC：风险可控选择性诊断、方向风险、稳定放行、域偏移门控、Wilson 安全边界。",
        "是否集中资源补统一 baseline 和消融：DINOv2/DINOv3/PLIP/BiomedCLIP/ConvNeXt/Swin 在同一协议下整理。",
        "是否开源或半开源可复现实验脚本：至少固定 split、特征、阈值选择、Wilson CI 和主结果表。",
        "是否把 strict external 继续扩大为真正多中心外部验证，还是先以外部压力测试和工作流安全为主线。",
    ]:
        story.append(bullet(x, st))

    story.append(para("11. 建议的下一阶段工作包", st, "h1"))
    story.append(
        table(
            [
                ["优先级", "工作", "目标产出", "完成标准"],
                ["P0", "固定最终数据字典和队列流程图", "old/third/strict external/clean/full 的病例流图", "医生和导师对数据角色没有歧义"],
                ["P0", "统一 v195 主结果表", "主文表 + supplement 表 + Wilson CI", "所有数字可一键复现"],
                ["P1", "整理 GrossPath-RC 算法表达", "框架图、伪代码、模块消融", "从版本号叙述变成方法叙述"],
                ["P1", "补 reader study 方案", "医生对照实验设计和样本清单", "能回答模型是否提升医生效率/安全"],
                ["P1", "补强 baseline 和负结果表", "统一协议对照表", "计算机审稿人能看到公平比较"],
                ["P2", "扩大外部验证或前瞻样本", "新冻结外部集或前瞻方案", "降低严格外部置信区间不确定性"],
                ["P2", "图像 SOP 与质量控制", "拍照标准和重拍建议", "医生团队可执行"],
            ],
            [18, 52, 60, 55],
            st,
            left_cols={1, 2, 3},
        )
    )

    story.append(para("最终判断", st, "h1"))
    story.append(
        para(
            "我们现在已经不是一个只有零散实验结果的项目，而是一个有完整阶段记录、明确医学问题、清楚数据边界、外部压力测试、负结果审计和风险控制主流程的医工交叉项目。最稳妥的共同表述是：我们已经建立了面向胸腺大体病理图像的低危/高危风险可控辅助判读框架；它在稳定病例上表现很好，但仍需要医生复核机制、更多外部验证和 reader study 才能走向高水平论文和临床部署。",
            st,
            "note",
        )
    )

    meta = {
        "clean_n": int(clean_all["n"]),
        "clean_bacc": float(clean_all["balanced_accuracy"]),
        "clean_review_rate": float(clean_all["review_or_reject_rate"]),
        "strict_external_n": int(clean_ext["n"]),
        "strict_external_bacc": float(clean_ext["balanced_accuracy"]),
        "strict_external_review_rate": float(clean_ext["review_or_reject_rate"]),
        "v195_auto_decision_all": int(v195_ci["auto_decision_n"]),
        "v195_wilson95_high_all": float(v195_ci["wilson95_high"]),
        "v201_rescued_all": int(v201_all["rescued_n"]),
        "v201_hurt_all": int(v201_all["hurt_n"]),
    }
    md = "\n".join(
        [
            "# Task7 项目进展与论文完成度综合报告（医生导师共读版）",
            "",
            f"- clean cohort: n={meta['clean_n']}，BAcc={pct(meta['clean_bacc'])}，复核/拒识={pct(meta['clean_review_rate'])}。",
            f"- strict external: n={meta['strict_external_n']}，BAcc={pct(meta['strict_external_bacc'])}，复核/拒识={pct(meta['strict_external_review_rate'])}。",
            f"- v195 全域自动判读 {meta['v195_auto_decision_all']} 例，自动错误 Wilson95 上界 {pct(meta['v195_wilson95_high_all'])}。",
            "- 报告定位：医生团队看临床价值和复核需求，导师团队看论文完成度和计算机贡献边界。",
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
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    MD_PATH.write_text(md, encoding="utf-8")
    (OUT_DIR / "v206_joint_team_report_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v206] wrote {PDF_PATH}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
