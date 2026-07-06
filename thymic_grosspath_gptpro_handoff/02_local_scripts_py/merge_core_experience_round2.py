from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"
BASE_FILE = EXP_DIR / "task56_experience_label_core_round1_merged.csv"
BATCH_DIR = EXP_DIR / "core_round2_batches"

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
    base = pd.read_csv(BASE_FILE)
    for col in ROUND2_COLS:
        if col not in base.columns:
            base[col] = ""
        base[col] = base[col].astype("object")

    files = sorted(BATCH_DIR.glob("task56_core_round2_batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"No round2 core batch CSVs found in {BATCH_DIR}")

    for f in files:
        part = pd.read_csv(f)
        for _, row in part.iterrows():
            mask = (base["case_id"] == row["case_id"]) & (base["image_name"] == row["image_name"])
            if mask.sum() != 1:
                continue
            for col in ROUND2_COLS:
                val = row.get(col, "")
                if pd.isna(val) or str(val) == "":
                    continue
                base.loc[mask, col] = val

    out_csv = EXP_DIR / "task56_experience_label_core_round2_merged.csv"
    base.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append("# Task56 经验标签核心集 Round2 合并摘要")
    summary_lines.append("")
    summary_lines.append(f"- Total images: `{len(base)}`")
    complete = int(base[ROUND2_COLS].replace("", pd.NA).notna().all(axis=1).sum())
    summary_lines.append(f"- Round2 complete images: `{complete}`")
    summary_lines.append(f"- Remaining incomplete images: `{int(len(base) - complete)}`")
    summary_lines.append("")

    summary_lines.append("## Round2 稳定性分布")
    summary_lines.append("")
    vc = base["exp_round2_label_stability"].fillna("").replace("", "<blank>").value_counts()
    for key, val in vc.items():
        summary_lines.append(f"- `{key}`: `{int(val)}`")
    summary_lines.append("")

    summary_lines.append("## Task6 类别 x Round2 稳定性")
    summary_lines.append("")
    ctab = pd.crosstab(base["task6_label"], base["exp_round2_label_stability"])
    summary_lines.append(frame_to_pipe_table(ctab))
    summary_lines.append("")

    summary_lines.append("## Task6 真值 x Round2 修正猜测")
    summary_lines.append("")
    ctab2 = pd.crosstab(base["task6_label"], base["exp_round2_revised_task6_guess"])
    summary_lines.append(frame_to_pipe_table(ctab2))
    summary_lines.append("")

    unstable = base[base["exp_round2_label_stability"].isin(["uncertain", "conflicting_clues"])].copy()
    summary_lines.append("## 不稳定图像")
    summary_lines.append("")
    if len(unstable) == 0:
        summary_lines.append("- None")
    else:
        cols = [
            "case_id",
            "image_name",
            "selection_tier",
            "task6_label",
            "exp_round1_boundary_axis",
            "exp_round2_revised_task6_guess",
            "exp_round2_label_stability",
            "exp_round2_key_discriminative_clues",
            "exp_round2_confounding_clues",
        ]
        summary_lines.append(frame_to_pipe_table(unstable[cols].set_index("case_id")))

    out_md = EXP_DIR / "task56_experience_label_core_round2_summary.md"
    out_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print(out_csv.resolve())
    print(out_md.resolve())


if __name__ == "__main__":
    main()
