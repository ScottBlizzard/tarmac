from __future__ import annotations

import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "汇报"
SOURCE_MD = REPORT_DIR / "2026-05-21_Task7旧数据二阶段模型阶段精华_医生版.md"
OUTPUT_PDF = REPORT_DIR / "2026-05-21_Task7旧数据二阶段模型阶段精华_医生版.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7PreThird"


REPORT_TEXT = """# Task7二分类旧数据阶段模型进展汇报

日期：2026-05-21

本报告整理的是我们在引入第三批数据之前，基于 batch1 与 batch2 共 285 例旧数据完成的一段核心工作。这里的 Task7 指低危与高危二分类：低危包括 A、AB、B1，高危包括 B2、B3、胸腺癌。所有结果均按 5-fold OOF 口径汇总，即每一例的预测都来自没有见过该例的折外模型。

## 1. 阶段结论

这一阶段最重要的变化，不是简单把单个模型从 0.76 慢慢调到更高，而是我们把问题拆成了“哪些病例可以稳定判读、哪些病例需要复核、复核后能否自动纠偏”。在课程学习之后，单一主模型整体准确率约 0.765；进一步做候选模型融合后，基础模型达到 0.835；最后在不使用医生肉眼所见文字、不按病例号查表的前提下，二阶段自动复核流程在旧数据 OOF 上达到 0.926。

这个结果说明，旧数据中确实存在一批模型可以直接稳定处理的病例，也存在一批需要二阶段复核的疑难病例。复核器利用的是不同图像模型之间的预测分歧、置信度和错误模式，而不是人工复核结果。换句话说，我们目前已经形成了一条较完整的自动流程：主模型先给出初判，模型自己判断是否需要复核，疑难病例再交给专门复核器纠偏。

需要强调的是，这仍然是旧数据内部交叉验证结果，不等同于外部泛化结果。它的意义在于证明路线可行，并为后续新增数据验证和医生复核清单提供明确方向。

## 2. 数据和评估口径

旧数据共 285 例，低危 144 例，高危 141 例，二分类分布基本均衡。我们使用 5-fold 交叉验证评估，每一折都只用训练折训练模型，在折外样本上输出预测。后续融合、路由和复核也按折内训练、折外评估的原则处理，避免把测试折结果反向用于调参。

混淆矩阵中，TN 表示低危判低危，FP 表示低危误判高危，FN 表示高危误判低危，TP 表示高危判高危。医生最关心的漏判高危对应 FN。

## 3. 课程学习之后我们看清楚的问题

课程学习阶段把病例按模型难度分成 easy、medium、可救 hard 和核心 hard。这个分层不是临床诊断分层，而是当前图像模型体系下的判读难度分层。它帮助我们确认了一个关键事实：模型不是在所有病例上平均地差，真正拉低总体准确率的是核心 hard。

| 分组 | 例数 | 正确数 | Accuracy | 我们的理解 |
| --- | ---: | ---: | ---: | --- |
| easy | 151 | 148 | 0.980 | 基本稳定，可作为可靠学习基础 |
| medium | 35 | 31 | 0.886 | 仍有边界错误，但整体可控 |
| easy+medium | 186 | 179 | 0.962 | 说明清楚病例已经可以达到较高水平 |
| 可救 hard | 34 | 22 | 0.647 | 有可救信号，适合单独处理 |
| 核心 hard | 65 | 17 | 0.262 | 主要瓶颈，多模型容易系统性反向 |
| 非核心 hard | 220 | 201 | 0.914 | 不含核心 hard 时已超过 90% |

课程学习融合后的总体结果为 Accuracy 0.7649，Balanced Accuracy 0.7649，F1 0.7633，混淆矩阵 TN/FP/FN/TP 为 110/34/33/108。相比此前最好严格融合结果，错误数减少 8 例，其中低危误升高危减少 6 例，高危漏判低危减少 2 例。

这一步的价值在于把模型瓶颈定位清楚。后续如果继续把所有 hard 直接塞回训练，容易破坏 easy 和 medium 已经学稳的边界；更合理的方向是把疑难病例单独识别出来，再做复核纠偏。

## 4. 经验标签和医生肉眼所见带来的启发

课程学习之后，我们尝试过多种经验标签方案。大规模弱标签覆盖更广，但噪声也更大，直接加入训练后并没有稳定提升；少量高可信核心标签更有信号，但提升幅度仍不稳定。31 例核心经验标签在一个候选融合口径下把准确率推到 0.7684，但细网格复查后没有形成稳定突破。

这给我们的判断是：经验标签方向有价值，但不能简单追求数量。真正有用的标签应该少而精，最好由医生确认，且要有低危和高危的正反对照。出血、坏死、囊性变、边界、切面均一、主体信息不足等线索都可能影响模型，但它们不能直接等同于某一类诊断。

医生提供的肉眼所见表格也帮助我们确认了另一件事：文字描述里确实包含和疑难病例相关的线索。但从自动模型角度看，我们不能依赖病例号去查医生描述，也不能把已知病理相关文字作为新病例推理输入。因此，我们把医生文字主要用于理解错误和设计复核思路，而不是作为最终自动流程的核心输入。

## 5. 从单模型到候选模型融合

在二阶段复核之前，我们先把多个候选图像模型的输出做了融合。这个阶段只使用图像模型输出行为，包括不同模型的预测概率、类别判断、置信度和模型间分歧。融合后的 41 号基础模型达到 Accuracy 0.8351，Balanced Accuracy 0.8348，F1 0.8291，混淆矩阵为 124/20/27/114。

这一步比课程学习主结果明显提高，说明多个图像模型之间并不是完全重复的。不同模型在疑难病例上会暴露出不同方向的偏差和不确定性，这些信息可以被后续复核器利用。

| 阶段 | Accuracy | Balanced Accuracy | F1 | TN/FP/FN/TP | 说明 |
| --- | ---: | ---: | ---: | --- | --- |
| 课程学习融合 | 0.7649 | 0.7649 | 0.7633 | 110/34/33/108 | 定位 hard_core 瓶颈 |
| 31例核心经验标签候选结果 | 0.7684 | 0.7685 | 0.7676 | 110/34/32/109 | 有信号但稳定性不足 |
| 早期二阶段纠偏 | 0.8070 | 0.8065 | 0.7955 | 123/21/34/107 | 证明复核方向有效 |
| 41号候选模型融合基础模型 | 0.8351 | 0.8348 | 0.8291 | 124/20/27/114 | 作为后续主模型基础 |
| 64号二阶段自动复核流程 | 0.9263 | 0.9263 | 0.9253 | 134/10/11/130 | 当前旧数据阶段最好结果 |

## 6. 二阶段自动复核流程

64 号流程是这一阶段最重要的结果。它不使用医生肉眼所见文字，不使用病理文本，也不按病例号查表。它的输入只来自推理时可获得的信息：主模型和候选图像模型的预测概率、类别输出、模型间分歧、置信度、不确定性，以及图像数量和选图规则等信息。

流程分两步。第一步，主模型先给所有病例做初判，同时路由器判断哪些病例可以直接通过，哪些病例需要进入复核池。第二步，复核器只处理进入复核池的病例，重新判断是否需要纠偏；没有进入复核池的病例保留主模型结果。

64 号最佳结果如下：

| 指标 | 数值 |
| --- | ---: |
| 全量 Accuracy | 0.9263 |
| Balanced Accuracy | 0.9263 |
| F1 | 0.9253 |
| AUC | 0.9456 |
| 混淆矩阵 TN/FP/FN/TP | 134/10/11/130 |
| 直接放行病例 | 224/285，78.6% |
| 直接放行准确率 | 93.30% |
| 进入复核池病例 | 61/285，21.4% |
| 复核池准确率 | 90.16% |
| hard_core 覆盖 | 41/65，63.08% |
| 救回/误伤 | 救回 26 例，误伤 0 例 |

从临床风险角度看，FN 从 27 例降到 11 例，FP 从 20 例降到 10 例。也就是说，二阶段流程不是单纯把高危判得更多，而是在高危漏判和低危误升两个方向上都同步改善。

## 7. 为什么复核器能起作用

复核器并不是重新从零看一张图，而是看一组图像模型对同一病例的反应模式。某些核心 hard 病例会让主模型高置信度走向错误方向，但不同候选模型之间的概率分歧、类别变化和置信度组合会暴露出“这个病例不适合直接相信主模型”的信号。

可以把这个流程理解为模型内部的二次判读：第一遍给出初判，第二遍专门检查初判是否落在已知易错模式中。如果落入易错模式，就交给复核器用另一套判别边界处理。这个机制解释了为什么单个模型只有 0.835，但二阶段组合可以进一步提高。

我们后续也尝试把 64 号复核器的能力蒸馏到单独 DINO 图像特征模型里，但目前没有成功。DINO student 单独最好只有 Accuracy 0.7123；加入融合后可以达到 0.9263，但没有超过 64 号。这说明复核能力主要来自模型群体行为和错误模式，而不是当前冻结 DINO embedding 单独可以直接读出的视觉空间。

## 8. 我们排除过的方向

这一阶段也有一些负结果值得保留。经验标签不能越多越好，弱标签过多会干扰已经稳定的边界；医生肉眼所见文字直接建模有启发性，但不适合作为新病例自动流程的依赖；单独用 DINO 冻结特征去学习 64 号复核器，目前承接不了纠偏能力；联合调整复核池阈值和复核器阈值的结果为 Accuracy 0.9193，低于 64 号固定策略。

这些负结果让我们把路线收窄到一个更稳的方向：保留 64 号二阶段自动复核流程作为旧数据阶段主结果，同时继续研究如何让复核器更好地泛化到新数据。

## 9. 目前我们希望医生团队关注的问题

第一，旧数据中剩余错误已经明显减少，但仍有 21 例最终错判。我们后续可以把这些病例单独列出，尤其关注仍被判低危的高危病例，判断它们从大体图像上是否本身缺少高危提示，还是模型没有看到关键区域。

第二，hard_core 并不是简单“不可判别”。64 号流程已经能覆盖 41/65 个核心 hard，并且救回 26 例，说明其中相当一部分有可利用信息。我们希望医生后续帮助区分：哪些 hard_core 是图像选择或拍照角度造成的，哪些是大体形态本身确实高度重叠。

第三，后续新增数据仍然非常关键。旧数据 OOF 结果已经证明二阶段路线有潜力，但最终能否作为可靠模型，需要用新增病例验证。新增数据最好尽量保留标准切面图、必要时保留外观图，并记录医生认为最适合判读的一张主图。

## 10. 下一步安排

我们会把 64 号流程作为旧数据阶段的主线结果继续维护，同时准备两项工作。第一项是整理旧数据剩余错判病例的医生复核清单，把模型初判、复核器判断、错误方向和疑似干扰线索放在一起。第二项是用新增数据做外部验证，重点看二阶段流程是否还能稳定识别疑难病例、是否仍能降低高危漏判。

目前我们的阶段判断是：单纯追求一个更强单模型已经不是最有效路线。更合理的系统应该是“稳定病例直接通过，疑难病例自动复核，仍不稳定的病例再交给医生重点确认”。这一套流程在旧数据内部已经达到 90% 以上，下一步的关键是验证它在新增数据上的稳定性。
"""


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def clean_inline(text: str) -> str:
    text = text.replace("`", "").replace("**", "")
    return html.escape(text)


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName=FONT_NAME,
            fontSize=17,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#14324d"),
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "HeadingCN",
            parent=styles["Heading1"],
            fontName=FONT_NAME,
            fontSize=12.0,
            leading=16,
            textColor=colors.HexColor("#14324d"),
            spaceBefore=2.4 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.0,
            leading=13.2,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            spaceAfter=1.3 * mm,
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.1,
            leading=8.8,
            wordWrap="CJK",
            textColor=colors.HexColor("#14324d"),
        ),
        "table_body": ParagraphStyle(
            "TableBodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.0,
            leading=8.7,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
        ),
    }


def parse_markdown(text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    para: list[str] = []
    idx = 0

    def flush() -> None:
        nonlocal para
        if para:
            blocks.append(("p", " ".join(para).strip()))
            para = []

    while idx < len(lines):
        raw = lines[idx].rstrip()
        line = raw.strip()
        if not line:
            flush()
            idx += 1
            continue
        if line.startswith("# "):
            flush()
            blocks.append(("title", line[2:].strip()))
            idx += 1
            continue
        if line.startswith("## "):
            flush()
            blocks.append(("h1", line[3:].strip()))
            idx += 1
            continue
        if line.startswith("|"):
            flush()
            rows: list[list[str]] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_line = lines[idx].strip()
                marker = table_line.replace("|", "").replace("-", "").replace(":", "").strip()
                if marker:
                    rows.append([cell.strip() for cell in table_line.strip("|").split("|")])
                idx += 1
            blocks.append(("table", rows))
            continue
        para.append(line)
        idx += 1
    flush()
    return blocks


def col_widths(ncols: int, width: float) -> list[float]:
    usable = width - 3 * mm
    presets = {
        2: [0.48, 0.52],
        4: [0.28, 0.20, 0.20, 0.32],
        5: [0.24, 0.15, 0.15, 0.18, 0.28],
        6: [0.21, 0.12, 0.14, 0.13, 0.18, 0.22],
    }
    ratios = presets.get(ncols, [1 / ncols] * ncols)
    return [usable * r for r in ratios]


def make_table(rows: list[list[str]], width: float, styles: dict[str, ParagraphStyle]) -> Table:
    ncols = max(len(r) for r in rows)
    normalized = [r + [""] * (ncols - len(r)) for r in rows]
    wrapped = []
    for ridx, row in enumerate(normalized):
        style = styles["table_head"] if ridx == 0 else styles["table_body"]
        wrapped.append([Paragraph(clean_inline(cell), style) for cell in row])
    table = Table(wrapped, colWidths=col_widths(ncols, width), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf2")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c9d7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
            ]
        )
    )
    return table


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_MD.write_text(REPORT_TEXT, encoding="utf-8")
    register_font()
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    story = []
    for kind, payload in parse_markdown(REPORT_TEXT):
        if kind == "title":
            story.append(Paragraph(clean_inline(str(payload)), styles["title"]))
            story.append(Spacer(1, 1.5 * mm))
        elif kind == "h1":
            story.append(Paragraph(clean_inline(str(payload)), styles["h1"]))
        elif kind == "p":
            story.append(Paragraph(clean_inline(str(payload)), styles["body"]))
        elif kind == "table":
            story.append(make_table(payload, doc.width, styles))
            story.append(Spacer(1, 2.0 * mm))
    doc.build(story)
    print(f"Generated markdown: {SOURCE_MD}")
    print(f"Generated PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
