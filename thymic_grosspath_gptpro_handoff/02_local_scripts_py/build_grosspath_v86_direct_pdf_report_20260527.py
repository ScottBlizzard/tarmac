from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(r"D:\影响分析")
V86_DIR = ROOT / "outputs" / "grosspath_rc_v86_paper_ready_summary_pack_20260527"
OUT_DIR = ROOT / "汇报"
OUT_PDF = OUT_DIR / "2026-05-27_Task7风险可控跨域诊断框架阶段汇报_v50-v86.pdf"

BLUE = colors.HexColor("#1F4D78")
LIGHT_BLUE = colors.HexColor("#E8EEF5")
LIGHT_GRAY = colors.HexColor("#F2F4F7")
DARK = colors.HexColor("#222222")
MUTED = colors.HexColor("#666666")
GRID = colors.HexColor("#D6DCE5")

WORKFLOW_ZH = {
    "Standard risk-control workflow (v50)": "v50 标准风险控制",
    "High-risk miss protection (v75)": "v75 高危漏诊保护",
    "Bidirectional light guard (v79-light)": "v79-light 双向轻量保护",
    "Bidirectional strict guard (v79-strict)": "v79-strict 双向严格保护",
}

DOMAIN_ZH = {
    "old_data": "旧数据",
    "third_batch": "第三批",
    "strict_external": "严格外部集",
}

CLAIM_ZH = {
    "main": "主结果",
    "candidate": "候选",
}


def register_fonts() -> None:
    # Built-in CID font avoids depending on local Office/Windows font registration.
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="STSong-Light",
            fontSize=20,
            leading=25,
            textColor=BLUE,
            alignment=TA_LEFT,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="STSong-Light",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="STSong-Light",
            fontSize=13,
            leading=17,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=9,
            leading=13,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=7.4,
            leading=9.2,
            textColor=DARK,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "small_left": ParagraphStyle(
            "small_left",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=7.4,
            leading=9.2,
            textColor=DARK,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "note": ParagraphStyle(
            "note",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=9,
            leading=13,
            textColor=DARK,
            alignment=TA_LEFT,
            leftIndent=6,
            rightIndent=6,
            spaceAfter=5,
            wordWrap="CJK",
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), style)


def table(data: list[list[str]], col_widths: list[float], style_map: dict[str, ParagraphStyle], left_cols: set[int] | None = None) -> Table:
    left_cols = left_cols or set()
    rendered = []
    for r_idx, row in enumerate(data):
        out_row = []
        for c_idx, value in enumerate(row):
            style = style_map["small_left"] if c_idx in left_cols and r_idx != 0 else style_map["small"]
            out_row.append(p(value, style))
        rendered.append(out_row)
    t = Table(rendered, colWidths=[w * inch for w in col_widths], repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GRAY),
                ("TEXTCOLOR", (0, 0), (-1, 0), DARK),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def add_note_box(story, text: str, s: dict[str, ParagraphStyle]) -> None:
    t = Table([[p(text, s["note"])]], colWidths=[9.65 * inch], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B8C7D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 8))


def bullets(story, items: list[str], s: dict[str, ParagraphStyle]) -> None:
    for item in items:
        story.append(p(f"• {item}", s["body"]))


def three_domain_rows(primary: pd.DataFrame) -> list[list[str]]:
    rows = [["数据域", "流程", "控制率", "BAcc", "Sens", "Spec", "FN", "FP"]]
    for _, r in primary.iterrows():
        rows.append(
            [
                DOMAIN_ZH.get(r["domain"], r["domain"]),
                WORKFLOW_ZH.get(r["workflow"], r["workflow"]),
                r["control_rate"],
                r["balanced_accuracy"],
                r["sensitivity"],
                r["specificity"],
                str(int(r["fn"])),
                str(int(r["fp"])),
            ]
        )
    return rows


def strict_rows(strict: pd.DataFrame) -> list[list[str]]:
    rows = [["流程", "控制率", "BAcc / 95%CI", "Sens / 95%CI", "Spec / 95%CI", "FN", "FP", "定位"]]
    for _, r in strict.iterrows():
        rows.append(
            [
                WORKFLOW_ZH.get(r["workflow"], r["workflow"]),
                r["control_rate"],
                f"{r['balanced_accuracy']}\n{r['balanced_accuracy_wilson_95ci']}",
                f"{r['sensitivity']}\n{r['sensitivity_wilson_95ci']}",
                f"{r['specificity']}\n{r['specificity_wilson_95ci']}",
                str(int(r["fn"])),
                str(int(r["fp"])),
                CLAIM_ZH.get(r["claim_tier"], r["claim_tier"]),
            ]
        )
    return rows


def ablation_rows(ablation: pd.DataFrame) -> list[list[str]]:
    rows = [["数据域", "模块", "控制率变化", "BAcc变化", "FN变化", "FP变化"]]
    keep = ablation[ablation["domain"].isin(["third_batch", "strict_external"])]
    for _, r in keep.iterrows():
        rows.append(
            [
                DOMAIN_ZH.get(r["domain"], r["domain"]),
                r["module_label"],
                r["delta_control_rate"],
                r["delta_bacc"],
                str(int(r["delta_fn"])),
                str(int(r["delta_fp"])),
            ]
        )
    return rows


def paired_rows(paired: pd.DataFrame) -> list[list[str]]:
    rows = [["数据域", "比较", "BAcc变化", "bootstrap 95%CI", "FN变化", "FP变化", "McNemar p"]]
    keep = paired[
        paired["comparison"].isin(
            [
                "Light full workflow vs v50",
                "Strict full workflow vs v50",
                "Light low-risk guard vs v75",
            ]
        )
    ]
    for _, r in keep.iterrows():
        rows.append(
            [
                DOMAIN_ZH.get(r["domain"], r["domain"]),
                r["comparison"],
                r["delta_bacc"],
                r["delta_bacc_bootstrap_95ci"],
                str(int(r["delta_fn"])),
                str(int(r["delta_fp"])),
                str(r["mcnemar_p"]),
            ]
        )
    return rows


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("STSong-Light", 7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(5.5 * inch, 0.28 * inch, f"Task7 风险可控跨域诊断框架阶段汇报 | v50-v86 | 第 {doc.page} 页")
    canvas.restoreState()


def build_pdf() -> Path:
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = styles()

    primary = pd.read_csv(V86_DIR / "v86_primary_fixed_workflow_table.csv")
    strict = pd.read_csv(V86_DIR / "v86_strict_external_focus_table.csv")
    adaptive = pd.read_csv(V86_DIR / "v86_adaptive_workflow_table.csv")
    ablation = pd.read_csv(V86_DIR / "v86_module_ablation_table.csv")
    paired = pd.read_csv(V86_DIR / "v86_paired_delta_stats_table.csv")

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=landscape(letter),
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.5 * inch,
        title="Task7 风险可控跨域诊断框架阶段汇报",
    )

    story = []
    story.append(p("Task7 风险可控跨域诊断框架阶段汇报", s["title"]))
    story.append(p("v50-v86 结果汇总 | 2026-05-27 | 胸腺大体病理图像二分类", s["subtitle"]))
    add_note_box(
        story,
        "本阶段我们的重点已经从单纯追求一个分类器分数，转为构建“自动诊断 + 风险门控 + 复核触发”的可部署流程。"
        "目前最稳的主推版本是 v79-light：严格外部集 BAcc 99.18%，高危漏诊 0 例，低危误升级 1 例，控制率 82.41%。"
        "v79-strict 在该外部集上达到 0 错例，但仍按候选高安全版本处理，后续需要更多外部批次验证。",
        s,
    )

    story.append(p("一、目前最能对外讲清楚的结论", s["h1"]))
    bullets(
        story,
        [
            "v50 是稳定基础流程，在旧数据、第三批、严格外部集上分别达到 99.65%、95.29%、97.30% BAcc，说明基础二阶段框架已经能跨批次工作。",
            "v75 主要减少高危漏诊；v79 主要减少低危误升级。两个模块解决的是不同错误类型，不是简单叠模型。",
            "v79-light 是当前主推高安全版本；v79-strict 结果更好，但因为严格外部集样本量有限，现阶段只作为高安全候选版本。",
        ],
        s,
    )

    story.append(p("二、三域固定工作流结果", s["h1"]))
    story.append(p("我们保留旧数据、第三批和严格外部集三组结果，更看重跨域一致性，而不是单一数据集的最高点估计。", s["body"]))
    story.append(table(three_domain_rows(primary), [0.85, 2.05, 0.8, 0.75, 0.75, 0.75, 0.4, 0.4], s, left_cols={1}))

    story.append(PageBreak())
    story.append(p("三、严格外部集焦点结果", s["h1"]))
    story.append(p("严格外部集是最关键的泛化检验。这里同时报告点估计和 Wilson 95% 置信区间，避免把小样本 100% 误读成真实世界确定性。", s["body"]))
    story.append(table(strict_rows(strict), [2.2, 0.75, 1.35, 1.35, 1.35, 0.35, 0.35, 0.6], s, left_cols={0}))

    story.append(p("四、模块贡献拆解", s["h1"]))
    story.append(p("我们把高危漏诊保护和低危误升级保护拆开评估，目的是真正说明每个模块承担什么临床风险。", s["body"]))
    story.append(table(ablation_rows(ablation), [1.0, 2.65, 1.0, 1.0, 0.7, 0.7], s, left_cols={1}))

    story.append(PageBreak())
    story.append(p("五、配对统计和表述边界", s["h1"]))
    story.append(
        p(
            "同病例配对比较更接近真实增益判断。第三批上 v79-light 相比 v50 的 BAcc 提升有正向区间支持；严格外部集错误数明显减少，但受样本量限制，统计上应写成趋势和错误减少，而不是强显著。",
            s["body"],
        )
    )
    story.append(table(paired_rows(paired), [1.0, 2.25, 0.9, 1.45, 0.65, 0.65, 0.8], s, left_cols={1}))

    story.append(p("六、部署型流程和论文贡献", s["h1"]))
    adaptive_light = adaptive[adaptive["policy_id"] == "adaptive_light_on_shift"].iloc[0]
    fixed_light = adaptive[adaptive["policy_id"] == "v79_light_lowrisk_guard"].iloc[0]
    story.append(
        p(
            f"如果只追求已有三域的最高安全性，固定 v79-light/strict 更强；如果考虑真实部署，v82 的无标签批次审计更有方法学价值。"
            f"例如 adaptive v50->v79-light 的全域控制率为 {adaptive_light['overall_control_rate']}，低于固定 v79-light 的 {fixed_light['overall_control_rate']}，"
            f"但能在识别到严重批次偏移时自动切换安全流程。",
            s["body"],
        )
    )
    bullets(
        story,
        [
            "计算机侧主线：无标签批次审计识别采集域偏移，再按风险等级选择不同复核强度。",
            "医学侧主线：高危漏诊和低危误升级分开控制，结果可以对应临床安全诉求。",
            "论文写法：v50 是稳定基线，v79-light 是主推高安全流程，v79-strict 是候选上限，v82 是部署式自适应框架。",
        ],
        s,
    )

    story.append(p("七、不能夸大的部分", s["h1"]))
    bullets(
        story,
        [
            "严格外部集已经被我们用于大量分析，后续论文中必须区分正式盲法验证结果和暴露后的探索性发现。",
            "v79-strict 的 0 错例是很有价值的候选结果，但 Wilson 区间下界仍约 93%，不能直接写成真实世界一定 100%。",
            "质量门控和外部域偏移解释是合理方向，但还需要更多外部医院批次来验证阈值稳定性。",
        ],
        s,
    )

    story.append(p("八、下一步", s["h1"]))
    bullets(
        story,
        [
            "继续把质量、视图、批次偏移做成开发集可标定的模块，减少依赖外部集分数反向调参。",
            "补充更多外部批次或前瞻性样本，验证 v79-light/v79-strict 的安全性是否稳定。",
            "把 v82-v85 的统计、消融和决策效用图表整理成论文 Results 的固定主表和补充材料。",
        ],
        s,
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT_PDF


if __name__ == "__main__":
    print(build_pdf())
