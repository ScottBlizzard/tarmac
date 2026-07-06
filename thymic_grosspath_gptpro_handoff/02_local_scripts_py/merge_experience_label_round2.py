from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PART_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling" / "pilot_parts"
OUT_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"
BASE_FILE = OUT_DIR / "task56_experience_label_pilot_24_round1_merged.csv"


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

ROUND2_COLS = [
    "exp_round2_revised_task5_guess",
    "exp_round2_revised_task6_guess",
    "exp_round2_key_discriminative_clues",
    "exp_round2_confounding_clues",
    "exp_round2_label_stability",
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

    base = pd.read_csv(BASE_FILE)
    merged = base.copy()

    for f in files:
        part = pd.read_csv(f)
        for _, row in part.iterrows():
            image_name_field = str(row.get("image_name", ""))
            looks_like_filename = image_name_field.endswith(".JPG") and ("/" not in image_name_field) and ("\\" not in image_name_field)
            corrupted_part3 = (not looks_like_filename) and str(row.get("who_type_raw", "")).endswith(".JPG")
            if looks_like_filename:
                key_image = row["image_name"]
            elif corrupted_part3:
                key_image = row["who_type_raw"]
            else:
                continue
            mask = merged["image_name"] == key_image
            if mask.sum() != 1:
                continue
            if corrupted_part3:
                remap = {
                    "exp_round2_revised_task5_guess": row.get("exp_round1_boundary_axis", ""),
                    "exp_round2_revised_task6_guess": row.get("exp_round2_revised_task5_guess", ""),
                    "exp_round2_key_discriminative_clues": row.get("exp_round2_revised_task6_guess", ""),
                    "exp_round2_confounding_clues": row.get("exp_round2_key_discriminative_clues", ""),
                    "exp_round2_label_stability": row.get("exp_round2_confounding_clues", ""),
                }
                for col, val in remap.items():
                    merged.loc[mask, col] = val
            else:
                for col in ROUND2_COLS:
                    merged.loc[mask, col] = row.get(col, "")

    merged = merged.sort_values(["task6_label", "case_id", "image_name"]).reset_index(drop=True)

    out_csv = OUT_DIR / "task56_experience_label_pilot_24_round2_merged.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append("# Task56 经验标签试标 Round2 合并摘要")
    summary_lines.append("")
    summary_lines.append(f"- Rows: `{len(merged)}`")
    summary_lines.append(f"- Complete round1 rows: `{int(merged[ROUND1_COLS].replace('', pd.NA).notna().all(axis=1).sum())}`")
    summary_lines.append(f"- Complete round2 rows: `{int(merged[ROUND2_COLS].replace('', pd.NA).notna().all(axis=1).sum())}`")
    summary_lines.append("")

    summary_lines.append("## Round2 稳定性分布")
    summary_lines.append("")
    vc = merged["exp_round2_label_stability"].fillna("").replace("", "<blank>").value_counts()
    for key, val in vc.items():
        summary_lines.append(f"- `{key}`: `{val}`")
    summary_lines.append("")

    summary_lines.append("## Task6 类别 x Round2 稳定性")
    summary_lines.append("")
    ctab = pd.crosstab(merged["task6_label"], merged["exp_round2_label_stability"])
    summary_lines.append(frame_to_pipe_table(ctab))
    summary_lines.append("")

    summary_lines.append("## Task6 真值 x Round2 修正猜测")
    summary_lines.append("")
    ctab2 = pd.crosstab(merged["task6_label"], merged["exp_round2_revised_task6_guess"])
    summary_lines.append(frame_to_pipe_table(ctab2))
    summary_lines.append("")

    summary_lines.append("## 高风险摇摆图")
    summary_lines.append("")
    unstable = merged[merged["exp_round2_label_stability"].isin(["uncertain", "conflicting_clues"])].copy()
    if len(unstable) == 0:
        summary_lines.append("- None")
    else:
        cols = [
            "case_id",
            "image_name",
            "task6_label",
            "exp_round1_boundary_axis",
            "exp_round2_revised_task6_guess",
            "exp_round2_label_stability",
            "exp_round2_key_discriminative_clues",
            "exp_round2_confounding_clues",
        ]
        summary_lines.append(frame_to_pipe_table(unstable[cols].set_index("case_id")))
    summary_lines.append("")

    out_md = OUT_DIR / "task56_experience_label_pilot_24_round2_summary.md"
    out_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print(out_csv.resolve())
    print(out_md.resolve())


if __name__ == "__main__":
    main()
