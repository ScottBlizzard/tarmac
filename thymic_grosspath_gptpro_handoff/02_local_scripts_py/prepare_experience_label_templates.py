from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "ThymicGross" / "image_review_batches" / "full_visual_review_merged.csv"
OUT_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"


BASE_COLS = [
    "review_index",
    "batch_id",
    "case_id",
    "who_type_raw",
    "image_name",
    "image_relpath",
    "local_image_path",
    "task5_label",
    "task6_label",
    "overall_quality",
    "image_clarity",
    "exposure",
    "background_clutter",
    "specimen_ratio",
    "ai_visual_summary",
    "ai_color_tone",
    "ai_shape_margin",
    "ai_texture_pattern",
    "ai_salient_clues",
    "ai_background_note",
    "ai_blind_guess_task5",
    "ai_blind_guess_task6",
    "ai_guess_confidence",
    "ai_priority_note",
]


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


def add_exp_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EXP_COLS:
        out[col] = ""
    return out


def choose_one_image_per_case(df: pd.DataFrame) -> pd.DataFrame:
    quality_order = {"good": 0, "fair": 1, "poor": 2}
    priority_order = {
        "worth doctor review": 0,
        "likely boundary case": 1,
        "likely quality/view issue": 2,
        "visually straightforward": 3,
    }
    temp = df.copy()
    temp["quality_rank"] = temp["overall_quality"].map(quality_order).fillna(9)
    temp["priority_rank"] = temp["ai_priority_note"].map(priority_order).fillna(9)
    temp = temp.sort_values(
        ["case_id", "quality_rank", "priority_rank", "review_index"],
        ascending=[True, True, True, True],
    )
    temp = temp.groupby("case_id", as_index=False).first()
    return temp.drop(columns=["quality_rank", "priority_rank"])


def balanced_pilot(df: pd.DataFrame, per_class: int = 4) -> pd.DataFrame:
    one_per_case = choose_one_image_per_case(df)
    picks = []
    for label, group in one_per_case.groupby("task6_label", sort=False):
        boundary = group[group["ai_priority_note"].isin(["likely boundary case", "worth doctor review"])]
        straight = group[group["ai_priority_note"] == "visually straightforward"]
        mixed = group[~group.index.isin(boundary.index.union(straight.index))]

        selected_parts = [
            boundary.head(2),
            straight.head(1),
        ]
        selected = pd.concat(selected_parts).drop_duplicates(subset=["case_id"])
        needed = per_class - len(selected)
        if needed > 0:
            remainder = pd.concat([mixed, straight.iloc[1:], boundary.iloc[2:]]).drop_duplicates(subset=["case_id"])
            remainder = remainder[~remainder["case_id"].isin(selected["case_id"])]
            selected = pd.concat([selected, remainder.head(needed)])
        selected = selected.head(per_class)
        picks.append(selected)

    pilot = pd.concat(picks, ignore_index=True)
    return pilot.sort_values(["task6_label", "case_id", "image_name"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE)

    task56 = df[df["task6_label"].notna()].copy()
    task56 = task56[BASE_COLS]
    task56_full = add_exp_cols(task56)
    task56_full.to_csv(OUT_DIR / "task56_experience_label_full_template.csv", index=False, encoding="utf-8-sig")

    pilot = balanced_pilot(task56, per_class=4)
    pilot = add_exp_cols(pilot)
    pilot.to_csv(OUT_DIR / "task56_experience_label_pilot_24.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        {
            "file": [
                "task56_experience_label_full_template.csv",
                "task56_experience_label_pilot_24.csv",
            ],
            "rows": [len(task56_full), len(pilot)],
            "notes": [
                "All Task5/Task6 core images with empty experience-label columns.",
                "Balanced pilot subset: 4 images per Task6 class, with boundary and straightforward cases mixed.",
            ],
        }
    )
    summary.to_csv(OUT_DIR / "task56_experience_label_template_summary.csv", index=False, encoding="utf-8-sig")
    print((OUT_DIR / "task56_experience_label_full_template.csv").resolve())
    print((OUT_DIR / "task56_experience_label_pilot_24.csv").resolve())


if __name__ == "__main__":
    main()
