from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a small balanced high-confidence anti-shortcut label core.")
    parser.add_argument("--weakplus-csv", required=True)
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--curriculum-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--per-bucket", type=int, default=12)
    return parser.parse_args()


def source_score(source: str) -> int:
    score = 0
    if "salvage_review" in source:
        score += 5
    if "manualstruct" in source:
        score += 4
    if "weak_round" in source:
        score += 1
    return score


def difficulty_score(value: str) -> int:
    return {"hard_salvage_teacher": 3, "medium": 2, "easy": 1}.get(value, 0)


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.weakplus_csv, dtype=str).fillna("")
    registry = pd.read_csv(args.registry_csv, dtype=str).fillna("")
    curriculum = pd.read_csv(args.curriculum_csv, dtype=str).fillna("")
    df = (
        labels.merge(
            registry[["case_id", "task_l6_label", "task_l7_label", "who_type_raw"]],
            on="case_id",
            how="left",
        )
        .merge(curriculum[["case_id", "difficulty", "difficulty_fine"]], on="case_id", how="left")
        .copy()
    )
    df["risk"] = df["task_l7_label"].map({"low_risk_group": "low", "high_risk_group": "high"})
    df = df[df["difficulty_fine"].isin(["easy", "medium", "hard_salvage_teacher"])].copy()
    df["high_risk_like"] = (
        (df["exp_manual_hemonec"] != "none")
        | (df["exp_manual_irregularity"] == "high")
        | df["exp_manual_confound_target"].isin(["B2", "B3", "TC"])
    )
    df["low_risk_like"] = (
        (df["exp_manual_pale_uniform"] == "yes")
        | (df["exp_manual_round_smooth"] == "yes")
        | (df["exp_manual_hemonec"] == "none")
        | (df["exp_manual_irregularity"] == "low")
        | df["exp_manual_confound_target"].isin(["A_AB", "B1"])
    )
    df["source_score"] = df["antishortcut_source"].map(source_score)
    df["difficulty_score"] = df["difficulty_fine"].map(difficulty_score)
    df["rarity_score"] = df["task_l6_label"].map({"A": 1, "AB": 1, "B1": 1, "B2": 2, "B3": 3, "TC": 2}).fillna(0)

    buckets = {
        "low_with_highrisk_like": (df["risk"] == "low") & df["high_risk_like"],
        "high_with_highrisk_like": (df["risk"] == "high") & df["high_risk_like"],
        "high_with_lowrisk_like": (df["risk"] == "high") & df["low_risk_like"],
        "low_with_lowrisk_like": (df["risk"] == "low") & df["low_risk_like"],
    }

    selected: dict[tuple[str, str], dict[str, object]] = {}
    bucket_rows = []
    for bucket, mask in buckets.items():
        sub = df.loc[mask].copy()
        sub = sub.sort_values(
            ["difficulty_score", "source_score", "rarity_score", "case_id"],
            ascending=[False, False, False, True],
        ).head(args.per_bucket)
        bucket_rows.append({"bucket": bucket, "n": len(sub), "low": int((sub["risk"] == "low").sum()), "high": int((sub["risk"] == "high").sum())})
        for _, row in sub.iterrows():
            key = (row["case_id"], row["image_name"])
            if key not in selected:
                payload = row.to_dict()
                payload["antishortcut_bucket"] = bucket
                selected[key] = payload
            else:
                selected[key]["antishortcut_bucket"] = str(selected[key]["antishortcut_bucket"]) + "|" + bucket

    out = pd.DataFrame(selected.values())
    out["main_sample_weight"] = out["antishortcut_bucket"].map(
        lambda text: 1.45 if ("low_with_highrisk_like" in text or "high_with_lowrisk_like" in text) else 1.20
    )
    keep_cols = [
        "case_id",
        "image_name",
        "antishortcut_source",
        "antishortcut_bucket",
        "main_sample_weight",
        "task_l6_label",
        "task_l7_label",
        "difficulty",
        "difficulty_fine",
        "exp_manual_pale_uniform",
        "exp_manual_round_smooth",
        "exp_manual_microcystic",
        "exp_manual_multinodular",
        "exp_manual_hemonec",
        "exp_manual_irregularity",
        "exp_manual_confound_target",
        "exp_manual_view_limit",
    ]
    out = out[keep_cols].sort_values(["antishortcut_bucket", "case_id"]).reset_index(drop=True)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"wrote {out_path} rows={len(out)}")
    print(pd.DataFrame(bucket_rows).to_string(index=False))
    print("\nselected risk")
    print(out["task_l7_label"].value_counts().to_string())
    print("\nselected difficulty")
    print(out["difficulty_fine"].value_counts().to_string())
    print("\nselected source")
    print(out["antishortcut_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
