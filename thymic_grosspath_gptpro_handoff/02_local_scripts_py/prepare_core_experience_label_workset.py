from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"
CORE_IMAGES = EXP_DIR / "task56_experience_label_core_images_all.csv"
PILOT_ROUND2 = EXP_DIR / "task56_experience_label_pilot_24_round2_merged.csv"


EXP_COLS = [
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
    "exp_round2_revised_task5_guess",
    "exp_round2_revised_task6_guess",
    "exp_round2_key_discriminative_clues",
    "exp_round2_confounding_clues",
    "exp_round2_label_stability",
    "exp_round3_doctor_question",
    "exp_round3_should_use_for_training",
    "exp_round3_notes",
]


def main() -> None:
    core = pd.read_csv(CORE_IMAGES)
    for col in EXP_COLS:
        if col not in core.columns:
            core[col] = ""

    pilot = pd.read_csv(PILOT_ROUND2)
    pilot_key_cols = ["case_id", "image_name"] + EXP_COLS
    pilot = pilot[pilot_key_cols].copy()
    pilot = pilot.drop_duplicates(subset=["case_id", "image_name"])

    merged = core.merge(
        pilot,
        on=["case_id", "image_name"],
        how="left",
        suffixes=("", "_pilot"),
    )

    for col in EXP_COLS:
        pcol = f"{col}_pilot"
        if pcol in merged.columns:
            merged[col] = merged[pcol].where(merged[pcol].notna(), merged[col])
            merged = merged.drop(columns=[pcol])

    out_csv = EXP_DIR / "task56_experience_label_core_workset.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    unlabeled = merged[merged["exp_round1_overall_pattern"].isna() | (merged["exp_round1_overall_pattern"] == "")]
    unlabeled = unlabeled.copy()

    # Balance by tier and class in round-robin order.
    sort_order_tier = {
        "A_high_conf_wrong": 0,
        "B_doctor_review": 1,
        "C_boundary": 2,
        "D_prototype": 3,
    }
    unlabeled["tier_rank"] = unlabeled["selection_tier"].map(sort_order_tier).fillna(9)
    unlabeled = unlabeled.sort_values(["tier_rank", "task6_label", "case_id", "image_name"]).reset_index(drop=True)
    unlabeled = unlabeled.drop(columns=["tier_rank"])

    batch_dir = EXP_DIR / "core_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    parts = [unlabeled.iloc[i::3].copy().reset_index(drop=True) for i in range(3)]
    for idx, part in enumerate(parts, 1):
        part.to_csv(batch_dir / f"task56_core_round1_batch_{idx}.csv", index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append("# Task56 经验标签核心工作集准备摘要")
    summary_lines.append("")
    summary_lines.append(f"- Core workset images: `{len(merged)}`")
    summary_lines.append(f"- Prefilled from pilot_24: `{len(merged) - len(unlabeled)}`")
    summary_lines.append(f"- Remaining unlabeled images: `{len(unlabeled)}`")
    summary_lines.append("")
    summary_lines.append("## Unlabeled by tier")
    summary_lines.append("")
    for tier, cnt in unlabeled["selection_tier"].value_counts().items():
        summary_lines.append(f"- `{tier}`: `{cnt}`")
    summary_lines.append("")
    summary_lines.append("## Unlabeled by Task6 class")
    summary_lines.append("")
    for label, cnt in unlabeled["task6_label"].value_counts().sort_index().items():
        summary_lines.append(f"- `{label}`: `{cnt}`")
    summary_lines.append("")
    summary_lines.append("## Batch sizes")
    summary_lines.append("")
    for idx, part in enumerate(parts, 1):
        summary_lines.append(f"- `batch_{idx}`: `{len(part)}` images")

    (EXP_DIR / "task56_experience_label_core_workset_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(out_csv.resolve())
    for idx in range(1, 4):
        print((batch_dir / f"task56_core_round1_batch_{idx}.csv").resolve())


if __name__ == "__main__":
    main()
