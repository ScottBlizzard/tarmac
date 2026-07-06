from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cut-oriented Task7 registry from merged registry and view predictions.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--view-pred-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--mixed-margin", type=float, default=0.08)
    return parser.parse_args()


def infer_view_mode(df: pd.DataFrame, mixed_margin: float) -> pd.Series:
    cut_prob = df["view_prob_cut_surface"].astype(float)
    outer_prob = df["view_prob_outer_surface"].astype(float)
    pred_view_type = df["pred_view_type"].astype(str)
    diff = cut_prob - outer_prob

    mode = pred_view_type.copy()
    mixed_mask = pred_view_type.eq("mixed")
    mode.loc[mixed_mask & (diff >= mixed_margin)] = "cut_heavy"
    mode.loc[mixed_mask & (diff <= -mixed_margin)] = "outer_heavy"
    mode.loc[mixed_mask & ~(diff >= mixed_margin) & ~(diff <= -mixed_margin)] = "balanced_mixed"
    return mode


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry_csv)
    view_df = pd.read_csv(args.view_pred_csv, encoding="utf-8-sig")

    registry["case_id"] = registry["case_id"].astype(str)
    view_df["case_id"] = view_df["case_id"].astype(str)

    if "pred_view_mode" not in view_df.columns:
        view_df["pred_view_mode"] = infer_view_mode(view_df, mixed_margin=args.mixed_margin)

    keep_cols = [
        "case_id",
        "pred_view_type",
        "pred_view_mode",
        "view_prob_cut_surface",
        "view_prob_outer_surface",
        "view_prob_mixed",
        "view_prob_unclear",
    ]
    view_small = view_df[keep_cols].drop_duplicates(subset=["case_id"]).copy()
    merged = registry.merge(view_small, on="case_id", how="left", validate="one_to_one")

    cut_oriented = merged[merged["pred_view_mode"].isin(["cut_surface", "cut_heavy"])].copy()
    cut_oriented.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "total_cases": int(len(merged)),
        "cut_oriented_cases": int(len(cut_oriented)),
        "cut_oriented_view_mode_counts": cut_oriented["pred_view_mode"].value_counts().to_dict(),
        "cut_oriented_task7_counts": cut_oriented["task_l7_label"].value_counts().to_dict() if "task_l7_label" in cut_oriented.columns else {},
        "cut_oriented_who_counts": cut_oriented["who_type_raw"].value_counts().to_dict() if "who_type_raw" in cut_oriented.columns else {},
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
