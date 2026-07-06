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
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v207_joint_team_report_data_enriched_20260531"
REPORT_DIR = ROOT / "汇报"
PDF_PATH = REPORT_DIR / "2026-05-31_Task7项目进展与论文完成度综合报告_正式汇报版.pdf"
MD_PATH = REPORT_DIR / "2026-05-31_Task7项目进展与论文完成度综合报告_正式汇报版.md"

V202_METRICS = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527" / "v202_clean_vs_full_policy_metrics.csv"
V198_CI = ROOT / "outputs" / "grosspath_rc_v198_release_compressed_safety_ci_20260527" / "v198_release_compressed_safety_ci.csv"
V197_TABLE = ROOT / "outputs" / "grosspath_rc_v197_stable_release_evidence_update_20260527" / "v197_stable_release_evidence_table.csv"
V201_SUMMARY = ROOT / "outputs" / "grosspath_rc_v201_stable_supported_domain_flip_20260527" / "v201_stable_supported_domain_flip_summary.csv"
TASK7_FAIR = ROOT / "reports" / "ThymicGross" / "task7_fair_benchmark_overall_results_2026-05-13.csv"
TASK6_FAIR = ROOT / "reports" / "ThymicGross" / "task6_fair_benchmark_overall_results_2026-05-13.csv"

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7JointV207"


def pct(x: object, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def num(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


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
            fontSize=17.2,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#12395b"),
            spaceAfter=3.2 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.8,
            leading=12.2,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_NAME,
            fontSize=12.4,
            leading=16.5,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=3.0 * mm,
            spaceAfter=1.4 * mm,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_NAME,
            fontSize=10.2,
            leading=14,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=2.0 * mm,
            spaceAfter=0.8 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.4,
            leading=12.4,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=0.95 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.1,
            leading=11.7,
            leftIndent=6 * mm,
            firstLineIndent=-4 * mm,
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=0.55 * mm,
        ),
        "note": ParagraphStyle(
            "note",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.0,
            leading=11.8,
            borderColor=colors.HexColor("#d2d6dc"),
            borderWidth=0.45,
            borderPadding=4.5,
            backColor=colors.HexColor("#f5f7fb"),
            textColor=colors.HexColor("#1f2933"),
            wordWrap="CJK",
            spaceAfter=1.4 * mm,
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.4,
            leading=8.1,
            alignment=TA_CENTER,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "td": ParagraphStyle(
            "td",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.35,
            leading=8.0,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "td_left": ParagraphStyle(
            "td_left",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.35,
            leading=8.0,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "td_right": ParagraphStyle(
            "td_right",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=6.35,
            leading=8.0,
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
    header_color: str = "#1f4e79",
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
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c2cf")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.4),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
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


OLD_STAGE_ROWS = [
    {
        "stage": "Task7 七模型公平基线",
        "n": 120,
        "acc": 0.7667,
        "bacc": 0.7667,
        "f1": 0.7500,
        "auc": 0.8172,
        "cm": "TN50/FP10/FN18/TP42",
        "note": "早期七模型中最佳 DINOv2 vitb14 whole；证明任务可学，但不足以支撑临床。",
    },
    {
        "stage": "课程学习融合",
        "n": 285,
        "acc": 0.7649,
        "bacc": 0.7649,
        "f1": 0.7633,
        "auc": None,
        "cm": "TN110/FP34/FN33/TP108",
        "note": "旧数据 5-fold OOF；性能未大幅提升，但把 easy/medium/hard_core 分层问题暴露出来。",
    },
    {
        "stage": "31例经验标签候选",
        "n": 285,
        "acc": 0.7684,
        "bacc": 0.7685,
        "f1": 0.7676,
        "auc": None,
        "cm": "TN110/FP34/FN32/TP109",
        "note": "经验标签有微弱正信号，但噪声和泛化不足，不能作为主模型。",
    },
    {
        "stage": "早期二阶段纠偏",
        "n": 285,
        "acc": 0.8070,
        "bacc": 0.8065,
        "f1": 0.7955,
        "auc": None,
        "cm": "TN123/FP21/FN34/TP107",
        "note": "第一次证明先判简单病例、再处理困难病例的方向有效。",
    },
    {
        "stage": "41号候选融合基础模型",
        "n": 285,
        "acc": 0.8351,
        "bacc": 0.8348,
        "f1": 0.8291,
        "auc": None,
        "cm": "TN124/FP20/FN27/TP114",
        "note": "多候选融合后形成更强基础模型，为 64 号二阶段提供基础。",
    },
    {
        "stage": "64号二阶段自动复核流程",
        "n": 285,
        "acc": 0.9263,
        "bacc": 0.9263,
        "f1": 0.9253,
        "auc": 0.9456,
        "cm": "TN134/FP10/FN11/TP130",
        "note": "旧数据 OOF 中最有说服力的自动二阶段结果；直接放行 78.6%，复核池也能保持 90% 左右。",
    },
]

HARD_ROWS = [
    ["easy", 151, 148, "98.0%", "模型稳定，适合自动放行的主体病例。"],
    ["medium", 35, 31, "88.6%", "存在少量边界或干扰线索，但仍有明显可学信号。"],
    ["easy+medium", 186, 179, "96.2%", "说明非核心困难病例不是主瓶颈。"],
    ["salvage hard", 34, 22, "64.7%", "部分 hard 可通过二阶段或复核器救回。"],
    ["hard_core", 65, 17, "26.2%", "真正瓶颈，常出现线索冲突或大体图不可稳定判别。"],
    ["non-core hard", 220, 201, "91.4%", "如果避开核心 hard，旧数据可接近临床可接受水平。"],
]

THIRD_EXTERNAL_FORCED_ROWS = [
    ["旧数据适配后 OOF", 285, "Acc 92.28%", "BAcc 92.27%", "F1 92.14%", "AUC 94.86%", "TN134/FP10/FN12/TP129", "旧数据保持高水平。"],
    ["第三批全量", 306, "Acc 83.01%", "BAcc 76.80%", "F1 66.67%", "AUC 81.82%", "TN202/FP22/FN30/TP52", "同院新批次仍有明显域偏移。"],
    ["第三批 holdout", 234, "Acc 87.18%", "BAcc 73.25%", "F1 57.14%", "AUC 73.94%", "TN184/FP12/FN18/TP20", "低危多、高危少；高危召回不足。"],
    ["严格外部 all", 108, "Acc 64.81%", "BAcc 62.75%", "F1 53.66%", "AUC 60.83%", "TN48/FP13/FN25/TP22", "强制分类明显下降，是转向风险控制的关键证据。"],
    ["严格外部可判读", 105, "Acc 63.81%", "BAcc 62.20%", "F1 53.66%", "AUC 60.38%", "TN45/FP13/FN25/TP22", "单纯剔除差图不能解决全部外部问题。"],
]


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
    story.append(para("项目进展、核心数据链与论文完成度综合报告", st, "title"))
    story.append(para("医生团队与导师团队共读版｜2026-05-31", st, "subtitle"))

    story.append(para("汇报范围", st, "h1"))
    story.append(
        para(
            "本阶段汇报围绕项目从早期单模型分类、旧数据二阶段复核、第三批适配、严格外部压力测试，到当前风险控制选择性诊断框架的完整证据链展开。早期单模型约 0.76，二阶段逐步推到 0.80、0.835，64 号流程在旧数据 OOF 达到 0.926；随后第三批和严格外部强制分类暴露跨域问题，推动当前方法转向 risk-controlled selective diagnosis。",
            st,
            "note",
        )
    )

    story.append(para("一页结论", st, "h1"))
    for x in [
        "项目最强证据不是某一个孤立高分，而是一条连续演进链：强表征基线 → 课程学习分层 → 经验标签/困难病例分析 → 二阶段复核 → 第三批/外部压力测试 → 风险控制选择性诊断。",
        "旧数据阶段最有利结果是 64 号二阶段自动复核流程：n=285，5-fold OOF，Acc 92.63%，BAcc 92.63%，F1 92.53%，AUC 94.56%，直接放行 78.6%，直接放行准确率 93.30%，复核池准确率 90.16%。",
        f"当前主流程 v195 在 clean cohort 上 n={int(clean_all['n'])}，Acc {pct(clean_all['accuracy'])}，BAcc {pct(clean_all['balanced_accuracy'])}，F1 {pct(clean_all['f1'])}，自动覆盖 {pct(clean_all['auto_decision_rate'])}，复核/拒识 {pct(clean_all['review_or_reject_rate'])}。",
        f"严格外部集当前 workflow 点估计 BAcc {pct(clean_ext['balanced_accuracy'])}，但复核/拒识 {pct(clean_ext['review_or_reject_rate'])}；因此外部结果必须按选择性诊断解释，不能写成全量自动分类 100%。",
        "医学方向已经具备较完整回顾性 AI 研究雏形；计算机方向的核心创新应聚焦小样本、跨域、可拒识、风险可控的大体病理图像诊断框架，而不是单纯换 backbone。",
    ]:
        story.append(bullet(x, st))

    story.append(para("数据链总览", st, "h1"))
    story.append(
        table(
            [
                ["阶段", "数据/验证", "核心数字", "说明"],
                ["早期公平基线", "Task7 七模型，病例级评估", "最佳 DINOv2 vitb14 whole：BAcc 76.67%，AUC 81.72%", "证明大体图有信号，但直接分类不够。"],
                ["旧数据课程学习", "285 例，5-fold OOF", "Acc 76.49%，BAcc 76.49%", "分数不惊艳，但发现 easy/medium/hard_core。"],
                ["旧数据二阶段", "285 例，5-fold OOF", "64 号流程 Acc/BAcc 92.63%，AUC 94.56%", "这是转向计算机框架前最有利的实证结果。"],
                ["第三批适配", "306 例扩展开发", "强制分类第三批 full Acc 83.01%，holdout Acc 87.18%", "说明同院新批次也有偏移。"],
                ["严格外部压力", "108 例冻结外部", "强制分类 Acc 64.81%，BAcc 62.75%", "说明必须加入质量/域偏移/拒识机制。"],
                ["风险控制工作流", "clean 697 例", f"BAcc {pct(clean_all['balanced_accuracy'])}，复核/拒识 {pct(clean_all['review_or_reject_rate'])}", "当前最适合论文主线的安全口径。"],
            ],
            [34, 42, 58, 52],
            st,
            left_cols={1, 2, 3},
        )
    )

    story.append(para("1. 旧数据阶段：从 0.76 到 0.926 的关键证据链", st, "h1"))
    story.append(
        para(
            "这一页是整份报告最关键的补充。旧数据阶段虽然还没有完全转成后来的 GrossPath-RC 框架，但已经把“先识别稳定病例，再处理困难病例”的方向跑通了。后续所有风险控制、复核池、拒识、hard_core 分析，本质上都是从这条二阶段证据链发展出来的。",
            st,
        )
    )
    old_rows = [["阶段", "n", "Acc", "BAcc", "F1", "AUC", "混淆矩阵", "结论"]]
    for r in OLD_STAGE_ROWS:
        old_rows.append(
            [
                r["stage"],
                r["n"],
                pct(r["acc"]),
                pct(r["bacc"]),
                pct(r["f1"]),
                "NA" if r["auc"] is None else pct(r["auc"]),
                r["cm"],
                r["note"],
            ]
        )
    story.append(table(old_rows, [29, 12, 15, 15, 15, 15, 29, 56], st, left_cols={0, 6, 7}))
    story.append(
        para(
            "解释：64 号不是外部集结果，而是旧数据病例级 OOF 二阶段结果；它的价值在于证明二阶段机制本身成立。后面第三批和外部测试说明这个机制要想用于真实部署，必须再升级成跨域风险控制框架。",
            st,
            "note",
        )
    )

    story.append(
        KeepTogether(
            [
                para("64号二阶段流程细节", st, "h1"),
                table(
                    [
                        ["指标", "数值", "解释"],
                        ["总样本", "285 例", "旧数据 Task7，低危/高危病例级 5-fold OOF。"],
                        ["总体性能", "Acc 92.63%，BAcc 92.63%，F1 92.53%，AUC 94.56%", "二阶段后的最终自动输出表现。"],
                        ["混淆矩阵", "TN134 / FP10 / FN11 / TP130", "低危误升级 10 例，高危误降级 11 例。"],
                        ["直接放行", "224/285 = 78.6%", "模型认为稳定、可直接输出的病例比例。"],
                        ["直接放行准确率", "93.30%", "放行部分不是简单挑容易样本，仍覆盖近八成病例。"],
                        ["复核池", "61/285 = 21.4%", "模型认为需要二阶段复核或纠偏的病例。"],
                        ["复核池准确率", "90.16%", "早期复核器已经能在困难病例中保持较高正确率。"],
                        ["hard_core 覆盖", "41/65 = 63.08%", "能识别出相当部分核心困难病例进入复核。"],
                        ["救回/误伤", "rescued 26，hurt 0", "说明纠偏方向有效，但还需要跨域验证。"],
                    ],
                    [38, 55, 92],
                    st,
                    left_cols={0, 1, 2},
                    header_color="#245b3a",
                ),
            ]
        )
    )

    story.append(para("2. 难度分层：为什么二阶段能成立", st, "h1"))
    story.append(
        para(
            "课程学习阶段的核心发现不是单模型分数，而是病例难度结构。模型并不是所有病例都判断不好，而是在一小部分核心 hard 上反复出错；非核心病例已经接近较高水平。这正是后续“自动放行 + 复核/拒识”的医学和计算机依据。",
            st,
        )
    )
    story.append(
        table(
            [["分层", "n", "正确", "准确率", "含义"]] + HARD_ROWS,
            [35, 16, 16, 20, 98],
            st,
            left_cols={0, 4},
        )
    )
    story.append(para("旧数据二阶段给出的临床解释", st, "h2"))
    for x in [
        "easy/medium 说明大体图中确实有稳定可学习信息，不是纯随机或纯病理不可见。",
        "hard_core 说明一部分病例存在大体图线索冲突、取材/切面/外观不一致、或本身需要显微病理才能稳定判断。",
        "因此，合理系统不应该强迫模型对所有病例给出同等置信度的结论，而应把稳定病例自动放行，把核心困难病例提示复核。",
        "这条逻辑比单纯追求全量 accuracy 更符合医生使用场景，也更容易写成高质量医工交叉论文。",
    ]:
        story.append(bullet(x, st))

    story.append(para("经验标签在这里的真实作用", st, "h2"))
    story.append(
        table(
            [
                ["经验标签用途", "结论", "论文中应如何表述"],
                ["解释模型为什么犹豫", "有价值", "可作为错例分析、医生沟通和概念整理材料。"],
                ["直接大规模弱监督训练", "不稳定", "不能说经验标签已经稳定提升主模型。"],
                ["少量高可信标签辅助", "有候选价值", "可写成探索性实验或补充材料。"],
                ["hard_core 识别", "有启发", "可以帮助定义复核池，但还要模型化和外部验证。"],
            ],
            [43, 37, 105],
            st,
            left_cols={0, 1, 2},
        )
    )

    story.append(para("3. 第三批与严格外部：为什么必须转成风险控制框架", st, "h1"))
    story.append(
        para(
            "如果只看旧数据 64 号流程，项目已经很漂亮；但第三批和严格外部说明，同院新批次和外院/异质图像都会带来明显分布变化。这个结果不是失败，而是把论文主线从“一个分类器”推向“跨域风险可控系统”的关键证据。",
            st,
        )
    )
    story.append(
        table(
            [["队列", "n", "Acc", "BAcc", "F1", "AUC", "混淆矩阵", "结论"]] + THIRD_EXTERNAL_FORCED_ROWS,
            [28, 13, 18, 18, 18, 18, 32, 41],
            st,
            left_cols={0, 6, 7},
        )
    )
    story.append(
        para(
            "解释：第三批不是严格外部验证，更适合称为扩展开发/适配队列；严格外部集应保持冻结，只能用于最终压力测试。我们可以根据旧数据 + 第三批改进泛化能力，但不能根据严格外部分数反向调参。",
            st,
            "note",
        )
    )

    story.append(para("外部下降带来的三条结论", st, "h2"))
    for x in [
        "单纯强制二分类在跨域图像上不安全，尤其是高危召回会下降。",
        "质量筛查能发现坏图，但不能完全解决外部域偏移；图像风格、取材、切面、主体比例、拍摄流程都可能改变模型表征。",
        "真正可写的方向不是回避外部下降，而是把它变成 risk-controlled workflow：能判断的自动放行，不能判断的拒识/复核。",
    ]:
        story.append(bullet(x, st))

    story.append(
        KeepTogether(
            [
                para("4. 当前主流程 v195：风险控制选择性诊断结果", st, "h1"),
                table(
                    [
                        ["口径", "n", "低危/高危", "Acc", "BAcc", "F1", "FN/FP", "自动覆盖", "复核/拒识", "自动错"],
                        ["旧数据 clean", int(clean_old["n"]), f"{int(clean_old['low_risk_n'])}/{int(clean_old['high_risk_n'])}", pct(clean_old["accuracy"]), pct(clean_old["balanced_accuracy"]), pct(clean_old["f1"]), f"{int(clean_old['fn'])}/{int(clean_old['fp'])}", pct(clean_old["auto_decision_rate"]), pct(clean_old["review_or_reject_rate"]), int(clean_old["auto_error_n"])],
                        ["第三批", int(clean_third["n"]), f"{int(clean_third['low_risk_n'])}/{int(clean_third['high_risk_n'])}", pct(clean_third["accuracy"]), pct(clean_third["balanced_accuracy"]), pct(clean_third["f1"]), f"{int(clean_third['fn'])}/{int(clean_third['fp'])}", pct(clean_third["auto_decision_rate"]), pct(clean_third["review_or_reject_rate"]), int(clean_third["auto_error_n"])],
                        ["严格外部", int(clean_ext["n"]), f"{int(clean_ext['low_risk_n'])}/{int(clean_ext['high_risk_n'])}", pct(clean_ext["accuracy"]), pct(clean_ext["balanced_accuracy"]), pct(clean_ext["f1"]), f"{int(clean_ext['fn'])}/{int(clean_ext['fp'])}", pct(clean_ext["auto_decision_rate"]), pct(clean_ext["review_or_reject_rate"]), int(clean_ext["auto_error_n"])],
                        ["clean 总体", int(clean_all["n"]), f"{int(clean_all['low_risk_n'])}/{int(clean_all['high_risk_n'])}", pct(clean_all["accuracy"]), pct(clean_all["balanced_accuracy"]), pct(clean_all["f1"]), f"{int(clean_all['fn'])}/{int(clean_all['fp'])}", pct(clean_all["auto_decision_rate"]), pct(clean_all["review_or_reject_rate"]), int(clean_all["auto_error_n"])],
                        ["full 总体", int(full_all["n"]), f"{int(full_all['low_risk_n'])}/{int(full_all['high_risk_n'])}", pct(full_all["accuracy"]), pct(full_all["balanced_accuracy"]), pct(full_all["f1"]), f"{int(full_all['fn'])}/{int(full_all['fp'])}", pct(full_all["auto_decision_rate"]), pct(full_all["review_or_reject_rate"]), int(full_all["auto_error_n"])],
                    ],
                    [22, 12, 20, 16, 16, 16, 15, 18, 19, 14],
                    st,
                ),
            ]
        )
    )
    story.append(
        para(
            f"v195 的关键贡献是安全释放更多病例：相比 v185，全域自动判读从 {int(v185_ci['auto_decision_n'])} 例提高到 {int(v195_ci['auto_decision_n'])} 例，复核/拒识从 {pct(v185_ci['review_or_reject_rate'])} 降到 {pct(v195_ci['review_or_reject_rate'])}，自动错误 Wilson95 上界为 {pct(v195_ci['wilson95_high'])}。严格外部集自动错误 Wilson95 上界为 {pct(v195_ext_ci['wilson95_high'])}，但因为样本较少且复核比例高，应谨慎表述。",
            st,
            "note",
        )
    )

    story.append(para("从 64 号到 v195：两者不是同一个口径", st, "h2"))
    story.append(
        table(
            [
                ["项目", "64号二阶段", "v195 风险控制"],
                ["验证对象", "旧数据 285 例 5-fold OOF", "clean/full + 第三批 + 严格外部"],
                ["主要目的", "证明二阶段自动复核机制可以显著提高旧数据表现", "在跨域条件下控制自动输出风险"],
                ["输出方式", "放行 + 复核池纠偏后给最终类别", "稳定病例自动输出，不稳定病例复核/拒识"],
                ["最有利数字", "Acc/BAcc 92.63%，放行 78.6%", f"clean BAcc {pct(clean_all['balanced_accuracy'])}，自动覆盖 {pct(clean_all['auto_decision_rate'])}"],
                ["论文定位", "机制发现和早期强证据", "当前主流程和安全口径"],
            ],
            [38, 73, 73],
            st,
            left_cols={0, 1, 2},
        )
    )

    story.append(para("5. 医学团队关心的问题与当前答案", st, "h1"))
    story.append(
        table(
            [
                ["医生关心的问题", "我们已经做了什么", "目前答案", "仍需医生支持"],
                ["二分类是否最重要", "将 Task2/3 思路合并为 Task7 低危/高危；TC/B2/B3 归高危", "Task7 是最适合临床汇报的主任务", "确认最终低危/高危映射和边界亚型"],
                ["六分类是否能提高二分类", "比较 Task6、Task7、六分类合并二分类输出", "六分类可提供错误喜好和亚型信息，但当前样本量下不如直接 Task7 稳", "继续补 B2/B3/稀有亚型病例"],
                ["模型错在哪里", "混淆矩阵、DINO 漏诊清单、hard_core 清单、典型错例 PDF", "错误集中在边界、线索冲突和核心 hard", "判断哪些大体图本身不可判别"],
                ["多图/切面图怎么处理", "追踪图1/图2、切面/外观/混合视图，尝试 view specialist", "统一选图 SOP 比盲目多图混合更重要", "确认多图病例优先切面图还是代表性图"],
                ["照片质量和背景是否影响", "质量标签、外部质量门控、蓝板/曝光讨论", "质量会影响，但不是外部下降唯一原因", "制定重拍标准和可判读标准"],
                ["模型是否知道肿瘤区域/大小", "讨论 ROI、尺子、大小变量，尚未成熟", "可能有价值，但需避免偏置", "如能提供尺寸/ROI，可作为后续增强"],
                ["结果能否临床使用", "建立自动放行 + 复核/拒识工作流", "可作为辅助判读框架，不替代病理诊断", "设计 reader study 和真实流程评价"],
            ],
            [35, 53, 44, 51],
            st,
            left_cols={0, 1, 2, 3},
        )
    )

    story.append(para("数据与队列边界", st, "h1"))
    story.append(
        table(
            [
                ["队列", "规模", "角色", "能怎么写", "不能怎么写"],
                ["早期 Task1-6", "160例/172图", "早期任务体系和强表征验证", "早期开发队列", "不能和后期 Task7 直接横比"],
                ["Task7 旧数据", "285例", "内部开发、OOF、课程学习、hard 分层", "内部开发/OOF 队列", "不能称外部验证"],
                ["第三批", "306例", "扩展开发/适配队列", "时间外或扩展开发队列", "不能称 strict external"],
                ["严格外部", "108例", "冻结外部压力测试", "严格外部/外部压力测试", "不能用调参后最高分冒充干净验证"],
                ["clean cohort", "697例", "医生确认特殊病例后的主分析", "主分析队列", "不能写成删错例提分"],
                ["full cohort", "699例", "完整敏感性分析", "full sensitivity", "不能替代 clean 主口径"],
            ],
            [28, 23, 48, 42, 44],
            st,
            left_cols={2, 3, 4},
        )
    )

    story.append(para("6. 计算机方向：目前真正能成立的创新点", st, "h1"))
    story.append(
        para(
            "计算机贡献不能只写“我们换了很多模型”。真正有价值的是把小样本医学大体图像中的不确定性、跨域偏移、质量差异和临床风险统一到一个可审计流程里。64 号二阶段是方法雏形，v195 是安全化后的主流程。",
            st,
        )
    )
    story.append(
        table(
            [
                ["创新点", "解决的问题", "对应证据", "还要补什么"],
                ["难度感知学习", "不是所有病例同等难，hard_core 是核心瓶颈", "easy+medium 96.2%，hard_core 26.2%", "更正式的难度定义和统计验证"],
                ["二阶段复核机制", "稳定病例和困难病例应走不同路径", "64号 Acc/BAcc 92.63%，复核池 21.4%", "外部数据上的自动纠偏能力"],
                ["风险可控选择性诊断", "跨域条件下强制输出不安全", f"v195 clean 自动覆盖 {pct(clean_all['auto_decision_rate'])}，自动错 1", "更低复核率和 reader study"],
                ["稳定释放准则", "防止为了覆盖率牺牲安全", "v196 因释放 held-out 错误被否定", "写成明确算法规则"],
                ["跨域压力测试", "外部图像质量和风格变化带来风险", "严格外部强制分类 BAcc 62.75%", "更多中心或更规范外部集"],
                ["概念/经验标签桥接", "医生能理解模型为什么犹豫", "经验标签有解释价值，训练增益不稳定", "少量高可信概念标签消融"],
                ["统计安全边界", "小样本高分需要置信区间", f"v195 全域自动错误 Wilson95 上界 {pct(v195_ci['wilson95_high'])}", "正式报告 CI 和敏感性分析"],
            ],
            [35, 43, 55, 52],
            st,
            left_cols={0, 1, 2, 3},
        )
    )

    story.append(
        KeepTogether(
            [
                para("已尝试但不宜作为主结论的方向", st, "h1"),
                *[
                    bullet(x, st)
                    for x in [
                        "cut/outer/mixed specialist 没有稳定超过全量学习，说明单凭视图拆分不能保证性能提升。",
                        "ConvNeXt、Swin、EfficientNet-B3 等替换没有稳定超过 DINO 主线，说明瓶颈不是简单换 backbone。",
                        "大规模经验弱标签直接训练不稳定；少量高可信经验标签适合解释和候选辅助。",
                        "贪心降低复核率会释放 held-out 错误，因此主流程必须优先选择稳定版本，而不是最高覆盖版本。",
                        "PLIP、BiomedCLIP、DINOv3 等结果可作为表征探索，但严格外部集不能被用作反向调参依据。",
                    ]
                ],
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                para("7. 当前论文完成度", st, "h1"),
                table(
                    [
                        ["方向", "当前完成度", "已经具备", "主要短板", "下一步"],
                        ["普通医学 AI 回顾性研究", "较高", "明确临床任务、队列分层、clean/full、外部压力测试、医生材料", "前瞻、多中心、reader study 不足", "补医生 reader study 和病例流图"],
                        ["柳叶刀/顶级医学期刊路线", "中等", "临床问题有意义，安全工作流清楚", "样本量、多中心、真实临床终点不足", "优先扩外部和前瞻设计"],
                        ["医工交叉/Nature 子刊路线", "中等偏上雏形", "新任务 + 风险控制框架 + 外部压力 + 负结果审计", "方法抽象、外部域数量、算法消融不足", "把 GrossPath-RC 写成系统框架"],
                        ["TMI/MIA/CCF 计算机路线", "中等", "selective diagnosis、domain shift、stable release、Wilson CI", "统一 baseline、伪代码、统计显著性、开源协议不足", "补统一协议和消融"],
                    ],
                    [35, 25, 55, 48, 42],
                    st,
                    left_cols={0, 2, 3, 4},
                ),
            ]
        )
    )

    story.append(para("我们现在可以对外说什么", st, "h1"))
    story.append(
        table(
            [
                ["可说", "不能说"],
                ["我们已经完成从单模型分类到风险可控选择性诊断工作流的转变。", "不能说模型已对所有病例全自动诊断达到 99% 或 100%。"],
                ["旧数据 64 号二阶段证明二阶段机制在内部 OOF 上成立。", "不能把 64 号写成外部验证结果。"],
                ["大体图像中存在低危/高危分层信号，稳定病例可自动辅助判断。", "不能说大体图像已经可以替代病理诊断或 WHO 完整分型。"],
                ["第三批是扩展开发/适配队列，说明同院新批次也存在分布变化。", "不能把第三批称作严格外部验证。"],
                ["严格外部集是压力测试，workflow 点估计好，但复核/拒识比例高。", "不能单独报严格外部 100% 点估计而不报复核比例。"],
                ["v201 自动纠偏有初步正证据。", "不能写成自动纠偏模块已经成熟。"],
            ],
            [92, 92],
            st,
            left_cols={0, 1},
        )
    )

    story.append(para("8. 后续工作包", st, "h1"))
    story.append(
        table(
            [
                ["优先级", "工作", "目标产出", "完成标准"],
                ["P0", "固定最终数据字典和病例流图", "old/third/strict external/clean/full 的正式流图", "医生和导师对数据角色无歧义"],
                ["P0", "固化 64号二阶段与 v195 主结果", "旧数据机制表 + 当前 workflow 主表", "两条结果链能一键复现"],
                ["P0", "明确外部集冻结规则", "严格外部不能调参的书面规则", "论文中可经得住数据泄露质疑"],
                ["P1", "整理 GrossPath-RC 算法表达", "框架图、伪代码、模块消融", "从版本号叙述变成方法叙述"],
                ["P1", "补医生 reader study", "医生 alone、AI alone、AI+复核流程对照", "回答是否提升医生效率和安全"],
                ["P1", "统一 baseline 和负结果表", "DINOv2/v3/PLIP/BiomedCLIP/CNN 等公平表", "计算机审稿人能看到公平比较"],
                ["P2", "扩大外部验证或前瞻样本", "新冻结外部集或前瞻方案", "降低外部结果置信区间不确定性"],
                ["P2", "图像 SOP 与质量控制", "拍照标准、重拍标准、可判读标准", "医生团队可执行"],
            ],
            [18, 52, 60, 55],
            st,
            left_cols={1, 2, 3},
        )
    )

    story.append(para("建议给医生和导师的共同表述", st, "h1"))
    story.append(
        para(
            "我们目前已经建立了一条比较完整的证据链：早期单模型证明大体图像存在可学习信号；课程学习和经验回看发现核心困难病例；旧数据 64 号二阶段流程把内部 OOF 提升到 92.63%，证明先放行稳定病例、再处理困难病例的机制有效；第三批和严格外部暴露跨域强制分类风险；因此当前主线升级为风险可控选择性诊断框架，在 clean cohort 上保持很高安全性，同时对不稳定病例给出复核/拒识。后续论文的关键不是再单纯追一个全量 accuracy，而是把这套机制用病例流图、统一消融、外部冻结验证和医生 reader study 做完整。",
            st,
            "note",
        )
    )

    meta = {
        "old_stage64_acc": 0.9263,
        "old_stage64_bacc": 0.9263,
        "old_stage64_auto_release_rate": 0.786,
        "old_stage64_review_rate": 0.214,
        "old_stage64_review_acc": 0.9016,
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
        "best_task7_bacc": float(best_task7["balanced_accuracy"]),
        "best_task6_acc": float(best_task6["accuracy"]),
    }
    md = "\n".join(
        [
            "# Task7 项目进展与论文完成度综合报告",
            "",
            "## 核心数据链",
            "- Task7 七模型公平基线最佳 DINOv2 vitb14 whole：BAcc 76.67%，AUC 81.72%。",
            "- 旧数据课程学习：Acc/BAcc 76.49%。",
            "- 旧数据早期二阶段：Acc 80.70%。",
            "- 旧数据 41号候选融合：Acc 83.51%。",
            "- 旧数据 64号二阶段自动复核：Acc/BAcc 92.63%，AUC 94.56%，直接放行 78.6%，复核池 21.4%。",
            f"- 当前 v195 clean cohort：n={meta['clean_n']}，BAcc={pct(meta['clean_bacc'])}，复核/拒识={pct(meta['clean_review_rate'])}。",
            f"- 严格外部 workflow：n={meta['strict_external_n']}，BAcc={pct(meta['strict_external_bacc'])}，复核/拒识={pct(meta['strict_external_review_rate'])}。",
            "",
            "## 表述边界",
            "- 64号是旧数据 OOF 二阶段机制证据，不是外部验证。",
            "- v195 是风险控制选择性诊断，不是全量自动诊断。",
            "- 第三批是扩展开发/适配队列，严格外部集保持冻结压力测试口径。",
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
    (OUT_DIR / "v207_joint_team_report_data_enriched_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[v207] wrote {PDF_PATH}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
