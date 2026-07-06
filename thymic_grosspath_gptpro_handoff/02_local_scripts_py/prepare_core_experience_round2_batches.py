from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"
IN_FILE = EXP_DIR / "task56_experience_label_core_round1_merged.csv"
OUT_DIR = EXP_DIR / "core_round2_batches"

BATCH_SIZE = 24

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


def main() -> None:
    df = pd.read_csv(IN_FILE)
    missing = [c for c in ROUND1_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing round1 columns: {missing}")

    for col in ROUND2_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("task56_core_round2_batch_*.csv"):
        old.unlink()

    sort_df = df.copy()
    sort_df["selection_tier"] = pd.Categorical(
        sort_df["selection_tier"],
        categories=["A_high_conf_wrong", "B_doctor_review", "C_boundary", "D_prototype"],
        ordered=True,
    )
    sort_df["exp_round1_human_difficulty"] = pd.Categorical(
        sort_df["exp_round1_human_difficulty"],
        categories=["high", "medium", "low"],
        ordered=True,
    )
    sort_df = sort_df.sort_values(
        [
            "selection_tier",
            "exp_round1_human_difficulty",
            "task6_label",
            "case_id",
            "image_name",
        ]
    ).reset_index(drop=True)

    total = len(sort_df)
    batch_paths: list[Path] = []
    for start in range(0, total, BATCH_SIZE):
        batch = sort_df.iloc[start : start + BATCH_SIZE].copy()
        batch_path = OUT_DIR / f"task56_core_round2_batch_{len(batch_paths) + 1}.csv"
        batch.to_csv(batch_path, index=False, encoding="utf-8-sig")
        batch_paths.append(batch_path)

    summary_lines = []
    summary_lines.append("# Task56 经验标签核心集 Round2 工作集摘要")
    summary_lines.append("")
    summary_lines.append(f"- Source file: `{IN_FILE.name}`")
    summary_lines.append(f"- Total images for round2: `{total}`")
    summary_lines.append(f"- Batch size: `{BATCH_SIZE}`")
    summary_lines.append(f"- Batch count: `{len(batch_paths)}`")
    summary_lines.append("")
    summary_lines.append("## 按 tier 分布")
    summary_lines.append("")
    tier_counts = sort_df["selection_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        summary_lines.append(f"- `{tier}`: `{int(count)}`")
    summary_lines.append("")
    summary_lines.append("## 按 Task6 类别分布")
    summary_lines.append("")
    cls_counts = sort_df["task6_label"].value_counts().sort_index()
    for cls, count in cls_counts.items():
        summary_lines.append(f"- `{cls}`: `{int(count)}`")
    summary_lines.append("")
    summary_lines.append("## 轮次文件")
    summary_lines.append("")
    for p in batch_paths:
        summary_lines.append(f"- `{p.name}`")

    (EXP_DIR / "task56_experience_label_core_round2_workset_summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    for p in batch_paths:
        print(p.resolve())


if __name__ == "__main__":
    main()
