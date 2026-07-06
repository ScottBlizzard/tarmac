from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"
IN_FILE = EXP_DIR / "task56_experience_label_core_round2_merged.csv"


def main() -> None:
    df = pd.read_csv(IN_FILE)
    task5_match = df["exp_round2_revised_task5_guess"] == df["task5_label"]
    task6_match = df["exp_round2_revised_task6_guess"] == df["task6_label"]
    both_match = task5_match & task6_match

    strict = df[(df["exp_round2_label_stability"] == "stable") & both_match].copy()
    soft = df[df["exp_round2_label_stability"].isin(["stable", "uncertain"]) & both_match].copy()

    strict_file = EXP_DIR / "task56_experience_label_train_candidates_strict.csv"
    soft_file = EXP_DIR / "task56_experience_label_train_candidates_soft.csv"
    strict.to_csv(strict_file, index=False, encoding="utf-8-sig")
    soft.to_csv(soft_file, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 经验标签训练候选层摘要")
    lines.append("")
    lines.append(f"- Core round2 source rows: `{len(df)}`")
    lines.append(f"- Strict candidates (`stable` + truth matched): `{len(strict)}`")
    lines.append(f"- Soft candidates (`stable/uncertain` + truth matched): `{len(soft)}`")
    lines.append("")
    lines.append("## Strict 按 Task6 类别")
    lines.append("")
    for cls, count in strict["task6_label"].value_counts().sort_index().items():
        lines.append(f"- `{cls}`: `{int(count)}`")
    lines.append("")
    lines.append("## Soft 按 Task6 类别")
    lines.append("")
    for cls, count in soft["task6_label"].value_counts().sort_index().items():
        lines.append(f"- `{cls}`: `{int(count)}`")
    lines.append("")
    lines.append("## Soft 按 selection tier")
    lines.append("")
    for tier, count in soft["selection_tier"].value_counts().sort_index().items():
        lines.append(f"- `{tier}`: `{int(count)}`")

    (EXP_DIR / "task56_experience_label_train_candidates_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(strict_file.resolve())
    print(soft_file.resolve())


if __name__ == "__main__":
    main()
