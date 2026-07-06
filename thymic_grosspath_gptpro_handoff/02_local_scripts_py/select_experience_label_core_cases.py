from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VISUAL_REVIEW = ROOT / "reports" / "ThymicGross" / "image_review_batches" / "full_visual_review_merged.csv"
HIGH_CONF = ROOT / "汇报" / "task56_case_review_assets" / "case_list.csv"
OUT_DIR = ROOT / "reports" / "ThymicGross" / "experience_labeling"


def rank_prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    qrank = {"good": 0, "fair": 1, "poor": 2}
    prank = {
        "worth doctor review": 0,
        "likely boundary case": 1,
        "visually straightforward": 2,
        "likely quality/view issue": 3,
    }
    out["quality_rank"] = out["overall_quality"].map(qrank).fillna(9)
    out["priority_rank"] = out["ai_priority_note"].map(prank).fillna(9)
    return out


def choose_case_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = rank_prepare(df)
    df = df.sort_values(
        ["case_id", "priority_rank", "quality_rank", "review_index"],
        ascending=[True, True, True, True],
    )
    return df.groupby("case_id", as_index=False).first()


def choose_specific_case_rows(df: pd.DataFrame, priority: str) -> pd.DataFrame:
    sub = df[df["ai_priority_note"] == priority].copy()
    return choose_case_rows(sub)


def pick_per_class(df: pd.DataFrame, selected: set[int], n_per_class: int) -> pd.DataFrame:
    rows = []
    for label in ["A", "AB", "B1", "B2", "B3", "TC"]:
        group = df[(df["task6_label"] == label) & (~df["case_id"].isin(selected))].copy()
        rows.append(group.head(n_per_class))
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=df.columns)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(VISUAL_REVIEW)
    core = df[df["task6_label"].notna()].copy()
    core["case_id"] = core["case_id"].astype(int)

    wrong = pd.read_csv(HIGH_CONF)
    wrong["case_id"] = wrong["case_id"].astype(int)
    wrong_case_ids = set(wrong["case_id"].tolist())
    wrong_case_task = wrong.groupby("case_id")["task"].apply(lambda s: "|".join(sorted(set(s)))).to_dict()

    core_case = choose_case_rows(core)
    worth_rows = choose_specific_case_rows(core, "worth doctor review")
    boundary_rows = choose_specific_case_rows(core, "likely boundary case")
    proto_rows = choose_case_rows(core[(core["ai_priority_note"] == "visually straightforward") & (core["overall_quality"] == "good")].copy())

    selected_frames = []
    selected_ids: set[int] = set()

    tier_a = core_case[core_case["case_id"].isin(wrong_case_ids)].copy()
    tier_a["selection_tier"] = "A_high_conf_wrong"
    tier_a["selection_reason"] = tier_a["case_id"].map(lambda x: f"high_conf_wrong_{wrong_case_task.get(x,'Task56')}")
    selected_frames.append(tier_a)
    selected_ids.update(tier_a["case_id"].tolist())

    tier_b = worth_rows[~worth_rows["case_id"].isin(selected_ids)].copy()
    tier_b["selection_tier"] = "B_doctor_review"
    tier_b["selection_reason"] = "worth_doctor_review"
    selected_frames.append(tier_b)
    selected_ids.update(tier_b["case_id"].tolist())

    tier_c = pick_per_class(boundary_rows, selected_ids, n_per_class=4).copy()
    tier_c["selection_tier"] = "C_boundary"
    tier_c["selection_reason"] = "likely_boundary_case"
    selected_frames.append(tier_c)
    selected_ids.update(tier_c["case_id"].tolist())

    tier_d = pick_per_class(proto_rows, selected_ids, n_per_class=2).copy()
    tier_d["selection_tier"] = "D_prototype"
    tier_d["selection_reason"] = "stable_visual_prototype"
    selected_frames.append(tier_d)
    selected_ids.update(tier_d["case_id"].tolist())

    selected_cases = pd.concat(selected_frames, ignore_index=True)
    selected_cases = selected_cases.sort_values(["selection_tier", "task6_label", "case_id"]).reset_index(drop=True)

    keep_cols = [
        "selection_tier",
        "selection_reason",
        "case_id",
        "task5_label",
        "task6_label",
        "who_type_raw",
        "ai_priority_note",
        "overall_quality",
        "image_name",
        "image_relpath",
        "local_image_path",
        "ai_visual_summary",
        "ai_blind_guess_task5",
        "ai_blind_guess_task6",
        "ai_guess_confidence",
    ]
    selected_cases_out = selected_cases[keep_cols].copy()
    selected_cases_out.to_csv(OUT_DIR / "task56_experience_label_core_cases.csv", index=False, encoding="utf-8-sig")

    selected_case_ids = set(selected_cases["case_id"].tolist())
    selected_images = core[core["case_id"].isin(selected_case_ids)].copy()
    meta = selected_cases[["case_id", "selection_tier", "selection_reason"]].drop_duplicates()
    selected_images = selected_images.merge(meta, on="case_id", how="left")
    image_keep_cols = [
        "selection_tier",
        "selection_reason",
        "case_id",
        "task5_label",
        "task6_label",
        "who_type_raw",
        "image_name",
        "image_relpath",
        "local_image_path",
        "overall_quality",
        "ai_priority_note",
        "ai_visual_summary",
        "ai_blind_guess_task5",
        "ai_blind_guess_task6",
        "ai_guess_confidence",
    ]
    selected_images = selected_images[image_keep_cols].sort_values(["selection_tier", "task6_label", "case_id", "image_name"])
    selected_images.to_csv(OUT_DIR / "task56_experience_label_core_images_all.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# Task56 经验标签高价值核心层筛选说明")
    lines.append("")
    lines.append(f"- Core Task56 images available: `{len(core)}`")
    lines.append(f"- Core Task56 cases available: `{core['case_id'].nunique()}`")
    lines.append(f"- Selected core cases: `{selected_cases['case_id'].nunique()}`")
    lines.append(f"- Selected core images (all images of selected cases): `{len(selected_images)}`")
    lines.append("")
    lines.append("## 分层规则")
    lines.append("")
    lines.append("- `A_high_conf_wrong`: 已知高置信错例病例，优先级最高")
    lines.append("- `B_doctor_review`: 当前被标为 `worth doctor review` 的病例")
    lines.append("- `C_boundary`: 每个 Task6 类额外补 `4` 个边界病例")
    lines.append("- `D_prototype`: 每个 Task6 类额外补 `2` 个清晰、直观的稳定原型")
    lines.append("")
    lines.append("## 结果统计")
    lines.append("")
    lines.append("### 按 tier")
    lines.append("")
    for tier, sub in selected_cases.groupby("selection_tier"):
        lines.append(f"- `{tier}`: `{sub['case_id'].nunique()}` cases")
    lines.append("")
    lines.append("### 按 Task6 类别")
    lines.append("")
    vc = selected_cases["task6_label"].value_counts().sort_index()
    for label, count in vc.items():
        lines.append(f"- `{label}`: `{count}` cases")
    lines.append("")
    lines.append("## 使用建议")
    lines.append("")
    lines.append("- 先优先给 `task56_experience_label_core_cases.csv` 中这些病例补经验标签。")
    lines.append("- 真正接训练时，用 `task56_experience_label_core_images_all.csv`，这样多图病例不会被截断。")
    lines.append("- 如果这批经验标签训练有效，再逐步扩展到剩余核心图。")

    (OUT_DIR / "task56_experience_label_core_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print((OUT_DIR / "task56_experience_label_core_cases.csv").resolve())
    print((OUT_DIR / "task56_experience_label_core_images_all.csv").resolve())
    print((OUT_DIR / "task56_experience_label_core_selection_summary.md").resolve())


if __name__ == "__main__":
    main()
