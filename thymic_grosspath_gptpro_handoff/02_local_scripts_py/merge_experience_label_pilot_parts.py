from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PART_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling" / "pilot_parts"
OUT_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"


ROUND1_COLS = [
    "exp_round1_overall_pattern",
    "exp_round1_color_pattern",
    "exp_round1_surface_pattern",
    "exp_round1_structure_pattern",
    "exp_round1_hemorrhage_necrosis",
    "exp_round1_lowrisk_impression",
    "exp_round1_bgroup_impression",
    "exp_round1_tc_impression",
    "exp_round1_human_difficulty",
    "exp_round1_boundary_axis",
]


def frame_to_pipe_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(["index"] + cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for idx, row in df.iterrows():
        vals = [str(idx)] + [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    files = sorted(PART_DIR.glob("task56_experience_label_pilot_part*.csv"))
    if not files:
        raise FileNotFoundError(f"No part CSVs found in {PART_DIR}")

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(["task6_label", "case_id", "image_name"]).reset_index(drop=True)

    out_csv = OUT_DIR / "task56_experience_label_pilot_24_round1_merged.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append("# Task56 经验标签试标 Round1 合并摘要")
    summary_lines.append("")
    summary_lines.append(f"- Rows: `{len(merged)}`")
    summary_lines.append(f"- Filled round1 rows: `{int(merged[ROUND1_COLS].notna().all(axis=1).sum())}`")
    summary_lines.append("")
    summary_lines.append("## 分布概览")
    summary_lines.append("")
    for col in ROUND1_COLS:
        summary_lines.append(f"### {col}")
        vc = merged[col].fillna("").replace("", "<blank>").value_counts()
        for key, val in vc.items():
            summary_lines.append(f"- `{key}`: `{val}`")
        summary_lines.append("")

    summary_lines.append("## Task6 类别 x 边界轴")
    summary_lines.append("")
    ctab = pd.crosstab(merged["task6_label"], merged["exp_round1_boundary_axis"])
    summary_lines.append(frame_to_pipe_table(ctab))
    summary_lines.append("")

    summary_lines.append("## Task6 类别 x 人眼难度")
    summary_lines.append("")
    ctab2 = pd.crosstab(merged["task6_label"], merged["exp_round1_human_difficulty"])
    summary_lines.append(frame_to_pipe_table(ctab2))
    summary_lines.append("")

    out_md = OUT_DIR / "task56_experience_label_pilot_24_round1_summary.md"
    out_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print(out_csv.resolve())
    print(out_md.resolve())


if __name__ == "__main__":
    main()
