from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"
BASE_FILE = EXP_DIR / "task56_experience_label_core_workset.csv"
BATCH_DIR = EXP_DIR / "core_batches"

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
    base = pd.read_csv(BASE_FILE)
    for col in ROUND1_COLS:
        base[col] = base[col].astype("object")

    files = sorted(BATCH_DIR.glob("task56_core_round1_batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"No round1 core batch CSVs found in {BATCH_DIR}")

    for f in files:
        part = pd.read_csv(f)
        for _, row in part.iterrows():
            key_case = row["case_id"]
            key_image = row["image_name"]
            mask = (base["case_id"] == key_case) & (base["image_name"] == key_image)
            if mask.sum() != 1:
                continue
            for col in ROUND1_COLS:
                val = row.get(col, "")
                if pd.isna(val):
                    continue
                if str(val) == "":
                    continue
                base.loc[mask, col] = val

    out_csv = EXP_DIR / "task56_experience_label_core_round1_merged.csv"
    base.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append("# Task56 经验标签核心工作集 Round1 合并摘要")
    summary_lines.append("")
    filled = base[ROUND1_COLS].replace("", pd.NA).notna().all(axis=1).sum()
    summary_lines.append(f"- Total workset images: `{len(base)}`")
    summary_lines.append(f"- Round1 complete images: `{int(filled)}`")
    summary_lines.append(f"- Remaining incomplete images: `{int(len(base) - filled)}`")
    summary_lines.append("")
    summary_lines.append("## Round1 完整率按 tier")
    summary_lines.append("")
    tier_done = (
        base.assign(round1_done=base[ROUND1_COLS].replace("", pd.NA).notna().all(axis=1))
        .groupby("selection_tier")["round1_done"]
        .agg(["sum", "count"])
    )
    tier_done["ratio"] = (tier_done["sum"] / tier_done["count"]).round(3)
    summary_lines.append(frame_to_pipe_table(tier_done))
    summary_lines.append("")
    summary_lines.append("## Round1 完整率按 Task6 类别")
    summary_lines.append("")
    cls_done = (
        base.assign(round1_done=base[ROUND1_COLS].replace("", pd.NA).notna().all(axis=1))
        .groupby("task6_label")["round1_done"]
        .agg(["sum", "count"])
    )
    cls_done["ratio"] = (cls_done["sum"] / cls_done["count"]).round(3)
    summary_lines.append(frame_to_pipe_table(cls_done))
    summary_lines.append("")
    summary_lines.append("## 边界轴分布（已完成 Round1 的图）")
    summary_lines.append("")
    done_df = base[base[ROUND1_COLS].replace("", pd.NA).notna().all(axis=1)].copy()
    if len(done_df) > 0:
        vc = done_df["exp_round1_boundary_axis"].value_counts()
        for k, v in vc.items():
            summary_lines.append(f"- `{k}`: `{v}`")
    else:
        summary_lines.append("- None")

    out_md = EXP_DIR / "task56_experience_label_core_round1_summary.md"
    out_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print(out_csv.resolve())
    print(out_md.resolve())


if __name__ == "__main__":
    main()
