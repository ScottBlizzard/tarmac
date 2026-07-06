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
SOURCE_MD = REPORT_DIR / "2026-05-22_Task7第三批数据引入后适配阶段汇报_医生版.md"
OUTPUT_PDF = REPORT_DIR / "2026-05-22_Task7第三批数据引入后适配阶段汇报_医生版.pdf"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]
FONT_NAME = "Task7ThirdBatchCN"


REPORT_TEXT = """# Task7第三批数据引入后模型适配阶段汇报

日期：2026-05-22

这份汇报整理的是第三批 306 例数据引入后，我们围绕 Task7 低危/高危二分类做的阶段性工作。这里的低危包括 A、AB、B1，高危包括 B2、B3、胸腺癌。第三批数据先按外部测试集处理，不直接全部并入训练；后续为了判断模型能否适应新数据，我们再从第三批里划出一小部分做适配，其余仍作为 holdout 评估。

## 1. 第三批数据带来的变化

第三批数据和旧数据来自同一医院体系，但从模型角度看并不是完全同分布。旧数据 285 例在 Task7 上基本均衡，低危 144 例、高危 141 例；第三批 306 例低危明显更多，低危 224 例、高危 82 例。六分类构成上，第三批主要是 AB、B2、TC 和少量 B1，没有 A 和 B3。

| 数据集 | 例数 | 六分类构成 | Task7构成 | 我们的判断 |
| --- | ---: | --- | --- | --- |
| 旧数据 | 285 | A 44、AB 50、B1 50、B2 60、B3 24、TC 57 | 低危144、高危141 | 二分类基本均衡，适合作为主训练与OOF评估基础 |
| 第三批 | 306 | AB 212、B1 12、B2 29、TC 53 | 低危224、高危82 | 低危占多数，且低危几乎集中在AB；高危中B2/TC为主 |
| 第三批适配划分 | 72 | 从第三批中抽取 | 低危28、高危44 | 偏向高危和错误高发样本，用于适配与策略选择 |
| 第三批holdout | 234 | 第三批剩余病例 | 低危196、高危38 | 保留为评估集，不参与策略选择 |

我们还做了图像层面的快速核查。第三批可用图像分辨率中位数约 24MP，高于旧数据本地可回看样本约 15.9MP；但第三批整体亮度略低、主体相对更小、蓝板背景占比更大，边缘阴影和拍摄构图差异也更明显。DINO 特征空间中，旧数据和第三批可以被区分，域分类 AUC 约 0.739，第三批被判为“第三批域”的中位概率约 0.926。也就是说，虽然医院来源一致，但模型看到的图像分布已经发生了可测量偏移。

## 2. 直接外部测试结果

我们先把旧数据阶段形成的二阶段思路直接放到第三批 306 例上测试。由于最早 64 号流程的精确 checkpoint 当时没有完整保留，这一版是按 64 号思路在旧数据上重建的 frozen 64-style 流程，第三批只用于一次外部评分，不参与训练和调参。

| 测试口径 | Accuracy | Balanced Accuracy | F1 | TN/FP/FN/TP | 说明 |
| --- | ---: | ---: | ---: | --- | --- |
| 旧数据64号参考结果 | 0.9263 | 0.9263 | 0.9253 | 134/10/11/130 | 旧数据OOF阶段的内部参考上限 |
| 第三批306例直接外测 | 0.7124 | 0.6760 | 0.5269 | 169/55/33/49 | 直接迁移后明显下降，说明需要适配或重新校准 |

这个结果说明，旧数据上的高准确率不能直接等同于第三批上的泛化能力。第三批主要有两类错误：一类是 AB/B1 因结节、出血、碎片化、红色组织等表现被推成高危；另一类是 B2/TC 因主体较小、切面较淡、坏死出血不突出或形态较规整，被压成低危。

## 3. 适配策略和约束

我们没有把第三批全部混入训练，而是采用“旧数据保护 + 小样本适配 + holdout 验证”的方式。适配策略只能使用旧数据 OOF 和第三批 72 例适配集进行选择；第三批 holdout 的 234 例只做最后评估，不参与策略选择。

我们给适配策略加了一个硬约束：旧数据准确率和均衡准确率需要保持在约 0.92 以上。原因是旧数据阶段已经证明二阶段流程可以在内部达到 0.926 左右，如果为了第三批把旧数据性能明显打掉，就不是一个可接受的通用模型，而更像是为某一批数据单独调出来的模型。

## 4. 目前最重要的结果对照

| 版本 | 选择口径 | 旧数据表现 | 第三批holdout表现 | holdout TN/FP/FN/TP | 阶段判断 |
| --- | --- | --- | --- | --- | --- |
| holdout基础线 | 旧数据模型直接迁移 | 旧基线0.9263 | Acc 0.7436，BAcc 0.7090，F1 0.4545 | 149/47/13/25 | 第三批上有明显低危误升和高危漏判 |
| 当前可部署适配版 | 只用旧数据+adapt72选择 | Acc 0.9228，BAcc 0.9227 | Acc 0.8034，BAcc 0.7129，F1 0.4889 | 166/30/16/22 | 总体准确率提升约6个百分点，低危误升明显减少，但高危真阳性下降 |
| 0.87参考上限版 | 使用holdout标签作参考筛选 | Acc 0.9228，BAcc 0.9227 | Acc 0.8761，BAcc 0.7139，F1 0.5538 | 187/9/20/18 | 不是可部署结果；主要靠减少低危误升提高总体准确率，高危漏判反而增加 |
| 保高危召回参考版 | 使用holdout标签作参考筛选 | Acc 0.9228，BAcc 0.9227 | Acc 0.7521，BAcc 0.7142，F1 0.4630 | 151/45/13/25 | 保住高危TP后，总体准确率提升很小 |
| 二层轻量融合 | 只用旧数据+adapt72选择 | Acc 0.9228 | Acc 0.7094，BAcc 0.6992，F1约0.50 | 140/56/12/26 | 过度偏向高危，低危误升过多 |
| 冻结crop补救器 | 只用旧数据+adapt72选择 | Acc 0.9263，BAcc 0.9263 | Acc 0.8034，BAcc 0.7129，F1 0.4889 | 166/30/16/22 | 只救回旧数据1例，对第三批holdout没有新增收益 |

这张表里最需要区分的是“当前可部署适配版”和“0.87参考上限版”。可部署适配版的策略选择没有看 holdout 标签，因此更接近真实使用流程；0.87 那一版是为了判断第三批上理论上还有没有可挖空间，使用 holdout 标签做了参考筛选，只能作为上限观察，不能作为最终模型结果汇报。

## 5. 对0.87结果的解释

0.8761 这个准确率看起来接近 0.90，但它不是医学上最理想的提升方式。第三批 holdout 里低危 196 例、高危 38 例，低危占比高；如果模型大量减少低危误升高危，整体 Accuracy 会明显提高。0.87 参考版正是这种情况：低危误升从 47 例降到 9 例，但高危真阳性从 25 例降到 18 例，高危漏判从 13 例升到 20 例。

所以我们对 0.87 的定位是：它说明第三批上低危误升可以被明显压下去，模型还有适配空间；但它也提示，如果只盯总体 Accuracy，容易牺牲高危召回。后续策略不能只追求 0.87 或 0.90 的总体数字，还要同时看 FN，也就是高危打成低危的病例数。

## 6. 我们尝试后得到的判断

第一，第三批适配最容易改善的是低危误升高危。当前可部署适配版已经把 holdout 的 FP 从 47 降到 30；0.87 参考版甚至能降到 9。这说明 AB/B1 被误升的问题有较强的可校准空间。

第二，高危漏判更难改善。当前可部署适配版虽然总体 Accuracy 提高，但 TP 从 25 降到 22；保高危召回的参考版可以维持 TP=25，但 Accuracy 只能从 0.7436 小幅升到 0.7521。这说明第三批里被压成低危的 B2/TC，单靠当前冻结特征和轻量校准还不够。

第三，第三批 72 例适配集本身偏高危、偏难例。如果策略在 adapt72 上过度追求高危召回，放到 holdout 上容易造成低危大量误升。因此后续适配不能只看 adapt72 的总体分数，还需要把低危误升和高危漏判分开约束。

第四，冻结 crop 特征没有解决问题。我们补做了只针对低判病例的 crop rescue corrector，它能在旧数据上救回 1 例，使旧数据回到 0.9263，但在第三批 holdout 上没有新增收益。这提示单纯换成裁剪图冻结特征还不够，下一步需要真正微调 crop 视角下的 DINO 特征。

## 7. 正在继续跑的工作

服务器上已经继续挂起了 crop fine-tune 的 OOF 补跑。这个实验不是简单用冻结 crop 特征做分类，而是在适配数据上微调 DINO 的后两层和分类头，再把 fold 2 到 fold 5 补齐，和已有 fold 1 合并成完整 OOF。它的目的很明确：看裁剪后的肿瘤主体信息在真正微调后，能不能补上第三批高危漏判这一块。

这部分目前还在运行中，因此没有写进最终结果表作为已完成结论。等 fold 2-5 完成后，我们会重点看三个指标：旧数据是否仍能守住 0.92 左右，第三批 holdout Accuracy 是否继续提高，以及高危 FN 是否真正下降。

## 8. 下一步我们建议的工作顺序

第一步，保留当前可部署适配版作为阶段性主结果。它在不看 holdout 标签的前提下，把第三批 holdout 从 0.7436 提到 0.8034，同时旧数据仍保持在 0.9228。

第二步，继续追 crop fine-tune 的完整 OOF。如果微调 crop 能把高危 FN 拉下来，同时不显著增加低危 FP，就有可能把第三批 holdout 从 0.80 往上推，并且比 0.87 参考版更安全。

第三步，把第三批的错误拆成两张清单：低危误升高危和高危漏判低危。低危误升主要对应“模型过度警惕”，高危漏判对应“医学风险更高”。后续医生复核和模型改进应该优先看高危漏判清单。

第四步，如果后续要把第三批并入训练，我们建议只把一部分作为适配训练数据，另一部分继续保留为外部验证。直接全量混入训练虽然可能提高当前批次分数，但会失去判断泛化能力的依据。

## 9. 目前给医生团队的阶段性结论

引入第三批后，我们确认旧数据阶段的 0.926 内部结果不能直接外推到新批次；第三批存在类别构成、图像构图和模型特征空间上的可见偏移。经过小样本适配后，在不使用 holdout 标签调参的前提下，第三批 holdout 准确率从 0.7436 提升到 0.8034，旧数据性能仍保持在 0.9228。参考分析中存在 0.8761 的总体准确率上限，但这个结果主要来自低危误升的大幅减少，同时高危漏判增加，因此不能作为最终可部署方案。下一步重点不是单纯追总体准确率，而是继续提升高危召回，尤其是第三批 B2/TC 被压成低危的病例。
"""


def register_font() -> None:
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))
            return
    raise FileNotFoundError("No Chinese font found in Windows fonts directory.")


def clean(text: str) -> str:
    return html.escape(text.replace("`", ""))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=FONT_NAME,
            fontSize=17,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#16324F"),
            spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "HeadingCN",
            parent=base["Heading1"],
            fontName=FONT_NAME,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#16324F"),
            spaceBefore=2.5 * mm,
            spaceAfter=1.5 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.0,
            leading=13.3,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
            spaceAfter=1.5 * mm,
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.2,
            leading=9.0,
            wordWrap="CJK",
            textColor=colors.HexColor("#16324F"),
        ),
        "table_body": ParagraphStyle(
            "TableBodyCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.0,
            leading=8.8,
            wordWrap="CJK",
            textColor=colors.HexColor("#222222"),
        ),
    }


def parse_markdown(text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    para: list[str] = []
    idx = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            blocks.append(("p", " ".join(para).strip()))
            para = []

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_para()
            idx += 1
            continue
        if stripped.startswith("# "):
            flush_para()
            blocks.append(("title", stripped[2:].strip()))
            idx += 1
            continue
        if stripped.startswith("## "):
            flush_para()
            blocks.append(("h1", stripped[3:].strip()))
            idx += 1
            continue
        if stripped.startswith("|"):
            flush_para()
            rows: list[list[str]] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_line = lines[idx].strip()
                marker = table_line.replace("|", "").replace("-", "").replace(":", "").strip()
                if marker:
                    rows.append([cell.strip() for cell in table_line.strip("|").split("|")])
                idx += 1
            blocks.append(("table", rows))
            continue
        para.append(stripped)
        idx += 1
    flush_para()
    return blocks


def widths_for(ncols: int) -> list[float] | None:
    presets_mm = {
        2: [78, 100],
        4: [36, 18, 70, 54],
        5: [36, 25, 27, 32, 58],
        6: [28, 31, 30, 35, 26, 28],
    }
    if ncols not in presets_mm:
        return None
    return [value * mm for value in presets_mm[ncols]]


def build_pdf() -> None:
    register_font()
    styles = build_styles()
    story = []

    for block_type, payload in parse_markdown(REPORT_TEXT):
        if block_type == "title":
            story.append(Paragraph(clean(str(payload)), styles["title"]))
            continue
        if block_type == "h1":
            story.append(Paragraph(clean(str(payload)), styles["h1"]))
            continue
        if block_type == "p":
            story.append(Paragraph(clean(str(payload)), styles["body"]))
            continue
        if block_type == "table":
            rows = payload
            if not rows:
                continue
            rendered = []
            for row_idx, row in enumerate(rows):
                style = styles["table_head"] if row_idx == 0 else styles["table_body"]
                rendered.append([Paragraph(clean(cell), style) for cell in row])
            table = Table(rendered, colWidths=widths_for(len(rows[0])), repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324F")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C7D6")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 2.5 * mm))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Task7第三批数据引入后模型适配阶段汇报",
        author="胸腺影像分析项目组",
    )
    doc.build(story)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_MD.write_text(REPORT_TEXT, encoding="utf-8")
    build_pdf()
    print(SOURCE_MD)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
