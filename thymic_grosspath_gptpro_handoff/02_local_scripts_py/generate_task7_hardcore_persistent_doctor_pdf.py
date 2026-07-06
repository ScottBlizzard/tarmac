from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "outputs" / "task7_curriculum_runs" / "49_hardcore_persistent_review"
INPUT_CSV = WORK_DIR / "hardcore_persistent_wrong_43_review_table.csv"
OUTPUT_CSV = WORK_DIR / "hardcore_persistent_wrong_43_doctor_review_table.csv"
OUTPUT_MD = ROOT / "汇报" / "2026-05-19_Task7核心hard持续错判医生复核清单.md"
OUTPUT_PDF = ROOT / "汇报" / "2026-05-19_Task7核心hard持续错判医生复核清单.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHeiTask7HardcoreReview"


CASE_REVIEWS = {
    "2524091": "标本较小，形态和边缘不规整，局部淡白小结节伴暗红边缘；模型可能把小标本的不规则和颜色混杂当成高危。",
    "2421303": "多条小碎片样切面，主体信息少，边缘暗红，低危 A 型没有呈现典型完整、规整切面；模型可能被碎片化和不规则带偏。",
    "2416158": "切面较大，灰白实性区合并暗红/坏死样边缘，视野内复杂度高；低危 B1 被模型当成高危样复杂肿瘤。",
    "2320997": "切面体积大，多叶灰白结节伴暗红边缘，整体不像典型单一低危外观；模型很可能被分叶、多结节和红色组织误导。",
    "2304401": "更接近外观图，表面红黄混杂、凸凹不平，缺少清楚切面信息；模型可能把外观复杂度当成高危证据。",
    "2304324": "多块碎片样组织，红黄混杂，条索样切面不完整；A 型低危被碎片化和不规则外观误升。",
    "2419725": "主体较小，暗红坏死样和边缘不清楚较明显；模型可能在信息不足时更依赖暗红/坏死样线索。",
    "2417646": "偏外观图，标本暗红、表面粗糙，切面信息不足；低危 A 型被模型按高危样外观处理。",
    "2514780": "暗红坏死样外观明显，主体呈团块状且表面不规则；B1 低危容易被这种复杂外观误升。",
    "2113064": "两个灰白切面较实性，局部多结节感，低危 AB 但不像非常典型的规整低危切面。",
    "2114234": "切面较大，灰白实性区旁伴暗红组织和裂隙，视觉上接近复杂或出血样病灶。",
    "2206051": "红黄混杂、脂肪样组织和切面混在一起，主体边界不够清楚；模型可能被取材混杂误导。",
    "2208429": "长条状红黄组织，切面不完整，局部暗红；低危 AB 被模型误认为高危样不规则组织。",
    "2301043": "标本小，淡白切面边缘带红色，信息量有限；模型可能把小标本边缘红色和切面不典型当成高危。",
    "2300708": "小块红黄混杂组织，脂肪样成分明显，主体不够完整；更像取材/背景干扰导致误升。",
    "2414793": "圆形切面伴黑褐坏死样区域，虽然真值为 B1，但坏死样颜色很容易把模型推向高危。",
    "2302818": "两个小条状切面，形态不规则且信息量少；模型可能把非完整切面当成高危样线索。",
    "2506040": "小块暗红组织，伴坏死样/出血样颜色，B1 低危被红黑复杂外观误升。",
    "2414792": "多枚小条状碎片，主体信息少，边缘不规整；模型在信息不足时倾向按高危处理。",
    "2209199": "较大切面内有腔隙、暗红和黄白混杂，视觉复杂度高；B1 低危被模型当成高危样病灶。",
    "2207648": "红黄混杂，切面不完整，脂肪样组织和肿瘤组织边界不清；低危 AB 被复杂取材带偏。",
    "2213450": "大范围红黄混杂和碎片化，局部出血样明显；B1 低危被模型误升高危的风险较高。",
    "2521394": "小块暗红组织，主体小且缺少典型低危切面；模型可能主要依据暗红和形态不规整判断。",
    "2115307": "两个圆钝淡黄结节，切面较均一，缺少明显高危样复杂结构；B2 高危被模型当成低危样结节。",
    "2419639": "主体较小且远景，圆钝小结节为主，图像信息有限；B2 高危被低危样小结节外观掩盖。",
    "2422397": "三枚小条状/小结节样组织，主体信息少且缺少明显坏死或侵袭样表现；B3 被模型低估。",
    "2412958": "切面有灰白圆钝区，虽然周围有暗红组织，但模型可能更依赖圆钝、相对清楚的切面。",
    "2215086": "小圆形粉白结节旁有红色组织，整体像包裹较好的小结节；B2 高危被低危样结节外观带偏。",
    "2410565": "胸腺癌但图像中红黄脂肪样和圆钝组织混杂，缺少模型稳定识别的高危主体区域。",
    "2414021": "标本极小，仅见细条状组织，主体信息严重不足；B2 高危几乎没有足够图像线索。",
    "2307709": "胸腺癌切面呈两个较圆的灰白结节，边缘相对清楚，模型容易把它理解为低危/边界样结节。",
    "2208027": "红黄混杂并带脂肪样成分，局部切面较圆钝，模型可能被非肿瘤组织和低危样区域稀释。",
    "2215167": "大块淡白均一切面，分叶但整体较规整，缺少明显坏死出血；B2 高危表现得很像低危切面。",
    "2205439": "小块灰白结节，形态圆钝、主体小，缺少高危视觉线索；B2 被模型低估。",
    "2418322": "标本很小，远景下主体信息少，整体更像小结节；B2 高危被低危样小体积外观带偏。",
    "2409869": "条状黄白切面伴少量暗红，整体没有形成强高危视觉模式；模型可能按较规整切面处理。",
    "2510266": "外观暗红但切面呈两个灰白结节，模型可能被结节边界和局部均一性吸引，未抓住高危特征。",
    "2500696": "小块暗红标本，主体占比和切面信息有限；B3 高危在图像上缺少稳定可识别区域。",
    "2215103": "大块淡白/灰白均一切面，整体圆钝规整；B2 高危被模型明显当成低危样切面。",
    "2200732": "胸腺癌但图像主体较小，多结节灰黄，视野和切面信息有限；外观更接近 A/AB 样低危模式。",
    "2120121": "B3 但红黄组织、腔隙和脂肪样成分混杂，模型可能被低危样/脂肪样区域冲淡高危判断。",
    "2214223": "B3 图像更偏外观和混合组织，红黄复杂但高危主体不集中；模型可能没有抓到有效切面证据。",
    "2207409": "B2 切面灰白并伴脂肪和暗红组织，区域混杂，模型可能被相对均一的灰白切面或低危样区域带偏。",
}


def doctor_question(direction: str) -> str:
    if direction == "低危误升高危":
        return "请医生判断这些出血、坏死样、碎片化或外观复杂线索是否只是非特异性干扰，低危病例中是否应单独标注为取材/图像因素。"
    return "请医生判断该图是否确实缺少高危大体提示，是否有应优先观察的局部区域，或该病例仅凭这张大体图本身就难以区分。"


def clean(text: str) -> str:
    return html.escape(str(text).replace("`", ""))


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCN", parent=base["Title"], fontName=FONT_NAME, fontSize=17, leading=23, alignment=TA_CENTER, textColor=colors.HexColor("#17324d")),
        "h1": ParagraphStyle("H1CN", parent=base["Heading1"], fontName=FONT_NAME, fontSize=12.2, leading=16, textColor=colors.HexColor("#17324d"), spaceBefore=3 * mm, spaceAfter=1.2 * mm),
        "body": ParagraphStyle("BodyCN", parent=base["BodyText"], fontName=FONT_NAME, fontSize=9.0, leading=13.0, alignment=TA_LEFT, wordWrap="CJK", spaceAfter=1.2 * mm),
        "caption": ParagraphStyle("CaptionCN", parent=base["BodyText"], fontName=FONT_NAME, fontSize=8.0, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#4a5568")),
        "th": ParagraphStyle("THCN", parent=base["BodyText"], fontName=FONT_NAME, fontSize=6.4, leading=7.7, wordWrap="CJK", textColor=colors.HexColor("#17324d")),
        "td": ParagraphStyle("TDCN", parent=base["BodyText"], fontName=FONT_NAME, fontSize=6.15, leading=7.5, wordWrap="CJK"),
    }


def build_table(rows: list[list[str]], doc_width: float, st: dict[str, ParagraphStyle]) -> Table:
    col_widths = [19 * mm, 17 * mm, 25 * mm, 75 * mm, doc_width - 136 * mm]
    wrapped = []
    for ridx, row in enumerate(rows):
        style = st["th"] if ridx == 0 else st["td"]
        wrapped.append([Paragraph(clean(cell), style) for cell in row])
    table = Table(wrapped, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
                ("GRID", (0, 0), (-1, -1), 0.28, colors.HexColor("#b6c7d6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.0),
                ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
            ]
        )
    )
    return table


def fit_image(path: Path, doc_width: float) -> Image:
    img = Image(str(path))
    max_width = doc_width
    max_height = 210 * mm
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawCentredString(A4[0] / 2, 7 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = [["病理号", "WHO/真值", "模型判断", "我们回看判断", "请医生确认"]]
    for row in df.itertuples(index=False):
        model_text = (
            f"{row.model_pred}；p高={row.prob_high_risk_group:.3f}\n"
            f"Stage2={row.prob_stage2_easy_medium:.3f}，Stage3={row.prob_stage3_salvage_hard:.3f}"
        )
        rows.append(
            [
                str(row.original_case_id),
                f"{row.task_l6_label} / {row.true_risk}",
                model_text,
                str(row.our_review_detailed),
                str(row.doctor_question),
            ]
        )
    return rows


def write_markdown(df: pd.DataFrame) -> None:
    lines = [
        "# Task7 核心 hard 持续错判医生复核清单",
        "",
        "日期：2026-05-19",
        "",
        "这份清单只整理核心 hard 中最顽固的一批病例：Stage2、Stage3 和最终融合三路都没有做对。我们的目的不是让医生从零开始看错例，而是先把模型判断、模型置信度和我们能看到的图像干扰线索整理出来，方便医生重点复核。",
        "",
        f"本次共 {len(df)} 例，其中低危误升高危 {(df['error_direction'] == '低危误升高危').sum()} 例，高危漏判低危 {(df['error_direction'] == '高危漏判低危').sum()} 例。",
        "",
        "## 1. 总体观察",
        "",
        "低危误升高危主要集中在出血/坏死样暗红区域、红黄组织混杂、碎片化、不规则、外观图或主体信息不足。模型容易把这些非特异性复杂表现当成高危证据。",
        "",
        "高危漏判低危主要集中在小标本、淡白或灰白均一切面、圆钝结节、边界相对清楚、主体信息不足。模型容易把这些低危样外观当成低危证据，即使真值是 B2、B3 或胸腺癌。",
        "",
    ]
    for direction, sub in df.groupby("error_direction", sort=False):
        lines.extend([f"## {direction}", ""])
        lines.append("| 病理号 | WHO/真值 | 模型判断 | 我们回看判断 | 请医生确认 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in sub.itertuples(index=False):
            model = f"{row.model_pred}；p高={row.prob_high_risk_group:.3f}"
            lines.append(f"| {row.original_case_id} | {row.task_l6_label}/{row.true_risk} | {model} | {row.our_review_detailed} | {row.doctor_question} |")
        lines.append("")
    lines.extend(
        [
            "## 联系图",
            "",
            "![低危误升高危联系图](D:/影响分析/outputs/task7_curriculum_runs/49_hardcore_persistent_review/hardcore_persistent_low_to_high_contact_sheet.jpg)",
            "",
            "![高危漏判低危联系图](D:/影响分析/outputs/task7_curriculum_runs/49_hardcore_persistent_review/hardcore_persistent_high_to_low_contact_sheet.jpg)",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["original_case_id"] = df["original_case_id"].astype(str)
    df["our_review_detailed"] = df["original_case_id"].map(CASE_REVIEWS).fillna(df["our_review_initial"])
    df["doctor_question"] = df["error_direction"].map(doctor_question)
    df = df.sort_values(["error_direction", "wrong_confidence"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    write_markdown(df)

    register_font()
    st = styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=11 * mm,
        rightMargin=11 * mm,
        topMargin=11 * mm,
        bottomMargin=14 * mm,
    )
    story = [
        Paragraph("Task7 核心 hard 持续错判医生复核清单", st["title"]),
        Spacer(1, 2 * mm),
        Paragraph("这份清单整理的是核心 hard 中最顽固的一批病例：Stage2、Stage3 和最终融合三路都没有做对。我们先把模型判断、置信度、图像干扰线索和需要医生确认的问题放在一起，方便集中复核。", st["body"]),
        Paragraph(f"本次共 {len(df)} 例：低危误升高危 {(df['error_direction'] == '低危误升高危').sum()} 例，高危漏判低危 {(df['error_direction'] == '高危漏判低危').sum()} 例。", st["body"]),
        Paragraph("总体观察", st["h1"]),
        Paragraph("低危误升高危多与出血/坏死样暗红区域、红黄混杂、碎片化、不规则、外观图或主体信息不足有关；高危漏判低危多与小标本、淡白或灰白均一切面、圆钝结节、边界相对清楚、主体信息不足有关。", st["body"]),
    ]
    for direction, sub in df.groupby("error_direction", sort=False):
        story.append(Paragraph(direction, st["h1"]))
        story.append(build_table(build_rows(sub), doc.width, st))
        story.append(Spacer(1, 2 * mm))

    story.append(PageBreak())
    story.append(Paragraph("低危误升高危联系图", st["h1"]))
    story.append(fit_image(WORK_DIR / "hardcore_persistent_low_to_high_contact_sheet.jpg", doc.width))
    story.append(Paragraph("每张图下方标注病理号、WHO 类型、模型错误方向和主模型高危概率。", st["caption"]))
    story.append(PageBreak())
    story.append(Paragraph("高危漏判低危联系图", st["h1"]))
    story.append(fit_image(WORK_DIR / "hardcore_persistent_high_to_low_contact_sheet.jpg", doc.width))
    story.append(Paragraph("这组病例主要用于请医生判断：大体图是否本身缺少高危提示，或是否存在模型没有关注到的关键区域。", st["caption"]))
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_MD}")
    print(f"wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
