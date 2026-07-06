from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "汇报"
EXP_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"

DOCTOR_ORIG = REPORT_DIR / "Task56经验标签核心集_医生版_2026-05-10.csv"
DOCTOR_MANUAL = REPORT_DIR / "Task56经验标签核心集_医生版_人工修订2026-05-10.csv"
DOCTOR_MANUAL_UTF8 = EXP_DIR / "task56_experience_label_doctor_manual_utf8.csv"

CORE_ROUND2 = EXP_DIR / "task56_experience_label_core_round2_merged.csv"
CORE_ROUND2_MANUAL = EXP_DIR / "task56_experience_label_core_round2_manualmerge.csv"
SOFT_OUT = EXP_DIR / "task56_experience_label_train_candidates_soft_manualmerge.csv"
STRICT_OUT = EXP_DIR / "task56_experience_label_train_candidates_strict_manualmerge.csv"
SUMMARY_OUT = EXP_DIR / "task56_experience_label_manualmerge_summary.md"


STABILITY_STRICT = {"stable", "稳定"}
STABILITY_SOFT = {"stable", "uncertain", "稳定", "犹豫"}


def main() -> None:
    doctor_manual = pd.read_csv(DOCTOR_MANUAL, encoding="gbk", dtype={"病例ID": str})
    doctor_orig = pd.read_csv(DOCTOR_ORIG, encoding="utf-8-sig", dtype={"病例ID": str})

    doctor_manual.to_csv(DOCTOR_MANUAL_UTF8, index=False, encoding="utf-8-sig")

    changed_keys = []
    merged = pd.read_csv(CORE_ROUND2, dtype={"case_id": str})

    manual_subset = doctor_manual.rename(
        columns={
            "病例ID": "case_id",
            "图像文件": "image_name",
            "主要支持线索": "exp_round2_key_discriminative_clues",
            "主要干扰线索": "exp_round2_confounding_clues",
            "主要边界轴": "exp_round1_boundary_axis",
            "回看后Task5判断": "exp_round2_revised_task5_guess",
            "回看后Task6判断": "exp_round2_revised_task6_guess",
            "稳定性": "exp_round2_label_stability",
        }
    )[
        [
            "case_id",
            "image_name",
            "exp_round2_key_discriminative_clues",
            "exp_round2_confounding_clues",
            "exp_round1_boundary_axis",
            "exp_round2_revised_task5_guess",
            "exp_round2_revised_task6_guess",
            "exp_round2_label_stability",
        ]
    ].copy()

    old_subset = doctor_orig.rename(columns={"病例ID": "case_id", "图像文件": "image_name"})
    old_subset = old_subset[
        [
            "case_id",
            "image_name",
            "主要支持线索",
            "主要干扰线索",
            "主要边界轴",
            "回看后Task5判断",
            "回看后Task6判断",
            "稳定性",
        ]
    ].rename(
        columns={
            "主要支持线索": "old_support",
            "主要干扰线索": "old_confound",
            "主要边界轴": "old_boundary",
            "回看后Task5判断": "old_task5",
            "回看后Task6判断": "old_task6",
            "稳定性": "old_stability",
        }
    )

    compare = manual_subset.merge(old_subset, on=["case_id", "image_name"], how="left")
    for _, row in compare.iterrows():
        diffs = []
        if str(row["exp_round2_key_discriminative_clues"]) != str(row["old_support"]):
            diffs.append("support")
        if str(row["exp_round2_confounding_clues"]) != str(row["old_confound"]):
            diffs.append("confound")
        if str(row["exp_round1_boundary_axis"]) != str(row["old_boundary"]):
            diffs.append("boundary")
        if str(row["exp_round2_revised_task5_guess"]) != str(row["old_task5"]):
            diffs.append("task5_guess")
        if str(row["exp_round2_revised_task6_guess"]) != str(row["old_task6"]):
            diffs.append("task6_guess")
        if str(row["exp_round2_label_stability"]) != str(row["old_stability"]):
            diffs.append("stability")
        if diffs:
            changed_keys.append((row["case_id"], row["image_name"], ",".join(diffs)))

    merged = merged.merge(
        manual_subset,
        on=["case_id", "image_name"],
        how="left",
        suffixes=("", "_manual"),
    )
    for col in [
        "exp_round2_key_discriminative_clues",
        "exp_round2_confounding_clues",
        "exp_round1_boundary_axis",
        "exp_round2_revised_task5_guess",
        "exp_round2_revised_task6_guess",
        "exp_round2_label_stability",
    ]:
        manual_col = f"{col}_manual"
        if manual_col in merged.columns:
            merged[col] = merged[manual_col].combine_first(merged[col])
            merged = merged.drop(columns=[manual_col])

    merged.to_csv(CORE_ROUND2_MANUAL, index=False, encoding="utf-8-sig")

    task5_match = merged["exp_round2_revised_task5_guess"] == merged["task5_label"]
    task6_match = merged["exp_round2_revised_task6_guess"] == merged["task6_label"]
    both_match = task5_match & task6_match
    stability = merged["exp_round2_label_stability"].fillna("").astype(str).str.strip()
    strict = merged[stability.isin(STABILITY_STRICT) & both_match].copy()
    soft = merged[stability.isin(STABILITY_SOFT) & both_match].copy()
    strict.to_csv(STRICT_OUT, index=False, encoding="utf-8-sig")
    soft.to_csv(SOFT_OUT, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 人工修订经验标签合并摘要")
    lines.append("")
    lines.append(f"- 医生版原始行数：`{len(doctor_orig)}`")
    lines.append(f"- 人工修订版行数：`{len(doctor_manual)}`")
    lines.append(f"- 核心集行数：`{len(merged)}`")
    lines.append(f"- 检测到有字段变化的图像：`{len(changed_keys)}`")
    lines.append(f"- strict 候选：`{len(strict)}`")
    lines.append(f"- soft 候选：`{len(soft)}`")
    lines.append("")
    lines.append("## 变化字段")
    lines.append("")
    field_counts: dict[str, int] = {}
    for _, _, diff_str in changed_keys:
        for item in diff_str.split(","):
            field_counts[item] = field_counts.get(item, 0) + 1
    for k, v in sorted(field_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{k}`: `{v}`")
    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(DOCTOR_MANUAL_UTF8.resolve())
    print(CORE_ROUND2_MANUAL.resolve())
    print(SOFT_OUT.resolve())
    print(STRICT_OUT.resolve())


if __name__ == "__main__":
    main()
