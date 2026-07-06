from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"

CORE_IN = EXP_DIR / "task56_experience_label_core_round2_manualmerge.csv"
SOFT_IN = EXP_DIR / "task56_experience_label_train_candidates_soft_manualmerge.csv"
STRICT_IN = EXP_DIR / "task56_experience_label_train_candidates_strict_manualmerge.csv"

CORE_OUT = EXP_DIR / "task56_experience_label_core_round2_manualstruct.csv"
SOFT_OUT = EXP_DIR / "task56_experience_label_train_candidates_soft_manualstruct.csv"
STRICT_OUT = EXP_DIR / "task56_experience_label_train_candidates_strict_manualstruct.csv"
SUMMARY_OUT = EXP_DIR / "task56_experience_label_manualstruct_summary.md"


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def derive_pale_uniform(text: str) -> str:
    return "yes" if contains_any(text, ["淡白", "淡粉", "均匀", "均一", "灰白", "粉白"]) else "no"


def derive_round_smooth(text: str) -> str:
    kws = ["圆钝", "圆整", "圆形", "球形", "光滑", "边界清楚", "边界较清楚", "包膜完整", "规整", "成团规整", "表面完整", "圆滑"]
    return "yes" if contains_any(text, kws) else "no"


def derive_microcystic(text: str) -> str:
    kws = ["微囊", "囊隙", "囊样", "裂隙", "小腔", "筛孔", "空腔"]
    return "yes" if contains_any(text, kws) else "no"


def derive_multinodular(text: str) -> str:
    kws = ["多发", "卫星", "结节", "双叶", "分叶", "分区", "多结节", "双结节", "双枚", "两枚"]
    return "yes" if contains_any(text, kws) else "no"


def derive_hemonec(text: str) -> str:
    marked = ["坏死", "黑红", "血腔", "空洞", "坏死底", "大片出血", "囊性出血", "大片", "乳头样突起", "厚壁出血性囊腔"]
    mild = ["出血", "暗红", "红染", "针尖样出血", "斑驳", "深色", "血性"]
    if contains_any(text, marked):
        return "marked"
    if contains_any(text, mild):
        return "mild"
    return "none"


def derive_irregularity(text: str) -> str:
    kws = ["不规则", "粗糙", "碎裂", "外生", "乳头", "壁结节", "坏死底", "附蒂"]
    return "high" if contains_any(text, kws) else "low"


def derive_confound_target(text: str) -> str:
    if contains_any(text, ["胸腺癌", "TC"]):
        return "TC"
    if contains_any(text, ["AB型", "A型", "A/AB", "AB"]):
        return "A_AB"
    if "B1" in text:
        return "B1"
    if "B2" in text:
        return "B2"
    if "B3" in text:
        return "B3"
    if contains_any(text, ["B组", "B型"]):
        return "B_group"
    return "none"


def derive_view_limit(text: str) -> str:
    kws = ["小标本", "视野小", "无切面", "仅外观", "标本较小", "仅表面", "只见外观", "体积小"]
    return "yes" if contains_any(text, kws) else "no"


def add_manual_struct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    support = out["exp_round2_key_discriminative_clues"].fillna("").astype(str).str.strip()
    confound = out["exp_round2_confounding_clues"].fillna("").astype(str).str.strip()
    both = (support + "；" + confound).astype(str)

    out["exp_manual_pale_uniform"] = support.map(derive_pale_uniform)
    out["exp_manual_round_smooth"] = support.map(derive_round_smooth)
    out["exp_manual_microcystic"] = support.map(derive_microcystic)
    out["exp_manual_multinodular"] = support.map(derive_multinodular)
    out["exp_manual_hemonec"] = support.map(derive_hemonec)
    out["exp_manual_irregularity"] = support.map(derive_irregularity)
    out["exp_manual_confound_target"] = confound.map(derive_confound_target)
    out["exp_manual_view_limit"] = both.map(derive_view_limit)
    return out


def summarize(df: pd.DataFrame) -> list[str]:
    cols = [
        "exp_manual_pale_uniform",
        "exp_manual_round_smooth",
        "exp_manual_microcystic",
        "exp_manual_multinodular",
        "exp_manual_hemonec",
        "exp_manual_irregularity",
        "exp_manual_confound_target",
        "exp_manual_view_limit",
    ]
    lines = ["# 人工修订经验标签结构化摘要", ""]
    lines.append(f"- 核心集行数：`{len(df)}`")
    lines.append("")
    for col in cols:
        lines.append(f"## {col}")
        vc = df[col].value_counts(dropna=False)
        for key, val in vc.items():
            lines.append(f"- `{key}`: `{val}`")
        lines.append("")
    return lines


def main() -> None:
    core = pd.read_csv(CORE_IN, dtype={"case_id": str})
    soft = pd.read_csv(SOFT_IN, dtype={"case_id": str})
    strict = pd.read_csv(STRICT_IN, dtype={"case_id": str})

    core_struct = add_manual_struct(core)
    soft_struct = add_manual_struct(soft)
    strict_struct = add_manual_struct(strict)

    core_struct.to_csv(CORE_OUT, index=False, encoding="utf-8-sig")
    soft_struct.to_csv(SOFT_OUT, index=False, encoding="utf-8-sig")
    strict_struct.to_csv(STRICT_OUT, index=False, encoding="utf-8-sig")
    SUMMARY_OUT.write_text("\n".join(summarize(core_struct)), encoding="utf-8")

    print(CORE_OUT.resolve())
    print(SOFT_OUT.resolve())
    print(STRICT_OUT.resolve())


if __name__ == "__main__":
    main()
