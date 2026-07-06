from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v49_v37_v48_case_transitions_20260527"


POLICIES = {
    "p2_baseline": {"kind": "none", "budget": 0.0},
    "v37_any_hard_dev97": {"kind": "score", "score": "any", "budget": 0.600},
    "v48_direction_raw_dev97": {"kind": "score", "score": "direction_raw", "budget": 0.675},
    "v48_fn_ranker_dev97": {"kind": "score", "score": "fn", "budget": 0.725},
}


def review_by_score(score: np.ndarray, budget: float) -> np.ndarray:
    return v30.top_budget(score, float(budget))


def apply_review(df: pd.DataFrame, review: np.ndarray) -> np.ndarray:
    y = df["label_idx"].to_numpy(dtype=int)
    pred = df["p2_pred"].to_numpy(dtype=int).copy()
    pred[review] = y[review]
    return pred


def direction_label(df: pd.DataFrame) -> np.ndarray:
    y = df["label_idx"].to_numpy(dtype=int)
    p = df["p2_pred"].to_numpy(dtype=int)
    wrong = y != p
    return np.select(
        [wrong & (y == 1) & (p == 0), wrong & (y == 0) & (p == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )


def policy_metrics(df: pd.DataFrame, name: str, review: np.ndarray) -> dict[str, float | int | str]:
    y = df["label_idx"].to_numpy(dtype=int)
    final = apply_review(df, review)
    m = v30.metrics_binary(y, final)
    p2_wrong = df["p2_pred"].to_numpy(dtype=int) != y
    err_dir = direction_label(df)
    m.update(
        {
            "policy": name,
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_p2_wrong_n": int((review & p2_wrong).sum()),
            "captured_p2_wrong_rate": float((review & p2_wrong).sum() / p2_wrong.sum()) if p2_wrong.sum() else 0.0,
            "captured_fn_n": int((review & (err_dir == "FN_high_to_low")).sum()),
            "captured_fp_n": int((review & (err_dir == "FP_low_to_high")).sum()),
            "remaining_error_n": int((final != y).sum()),
        }
    )
    return m


def summarize_remaining(cases: pd.DataFrame, policy: str) -> pd.DataFrame:
    sub = cases.loc[cases[f"{policy}_final_correct"].eq(0)].copy()
    rows = []
    for group_col in ["task_l6_label", "quality_status", "quality_score", "p2_error_direction"]:
        if group_col not in sub.columns:
            continue
        grouped = sub.groupby(group_col, dropna=False).size().reset_index(name="remaining_error_n")
        grouped.insert(0, "group_by", group_col)
        grouped = grouped.rename(columns={group_col: "group_value"})
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical

    masks = v48.error_masks(dev)
    any_dev, any_ext, *_ = v48.generic_oof_external_scores(dev, ext, features, masks["any_wrong"])
    fn_dev, fn_ext, *_ = v48.generic_oof_external_scores(dev, ext, features, masks["fn_high_to_low"])
    fp_dev, fp_ext, *_ = v48.generic_oof_external_scores(dev, ext, features, masks["fp_low_to_high"])
    direction_ext = np.where(ext["p2_pred"].to_numpy(dtype=int) == 0, fn_ext, fp_ext)
    scores = {"any": any_ext, "fn": fn_ext, "fp": fp_ext, "direction_raw": direction_ext}

    base_cols = [
        "case_id",
        "original_case_id",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "image_name",
        "quality_status",
        "quality_score",
        "p2_pred",
        "main_prob",
        "robust_prob",
        "prob_mean_core",
    ]
    cases = ext[[c for c in base_cols if c in ext.columns]].copy()
    y = ext["label_idx"].to_numpy(dtype=int)
    p2 = ext["p2_pred"].to_numpy(dtype=int)
    cases["p2_error_direction"] = direction_label(ext)
    cases["p2_correct"] = (p2 == y).astype(int)
    cases["risk_any"] = any_ext
    cases["risk_fn"] = fn_ext
    cases["risk_fp"] = fp_ext
    cases["risk_direction_raw"] = direction_ext

    metrics_rows = []
    for policy_name, cfg in POLICIES.items():
        if cfg["kind"] == "none":
            review = np.zeros(len(ext), dtype=bool)
        else:
            review = review_by_score(scores[str(cfg["score"])], float(cfg["budget"]))
        final = apply_review(ext, review)
        cases[f"{policy_name}_review"] = review.astype(int)
        cases[f"{policy_name}_final_pred"] = final
        cases[f"{policy_name}_final_correct"] = (final == y).astype(int)
        metrics_rows.append(policy_metrics(ext, policy_name, review))

    metrics = pd.DataFrame(metrics_rows)
    cases["v48_raw_incremental_review_vs_v37"] = (
        cases["v48_direction_raw_dev97_review"].eq(1) & cases["v37_any_hard_dev97_review"].eq(0)
    ).astype(int)
    cases["v48_raw_incremental_rescue_vs_v37"] = (
        cases["v48_raw_incremental_review_vs_v37"].eq(1) & cases["p2_correct"].eq(0)
    ).astype(int)
    cases["v48_fn_incremental_review_vs_v48_raw"] = (
        cases["v48_fn_ranker_dev97_review"].eq(1) & cases["v48_direction_raw_dev97_review"].eq(0)
    ).astype(int)
    cases["v48_fn_incremental_rescue_vs_v48_raw"] = (
        cases["v48_fn_incremental_review_vs_v48_raw"].eq(1) & cases["p2_correct"].eq(0)
    ).astype(int)

    remaining_frames = []
    for policy_name in POLICIES:
        rem = summarize_remaining(cases, policy_name)
        if not rem.empty:
            rem.insert(0, "policy", policy_name)
            remaining_frames.append(rem)
    remaining = pd.concat(remaining_frames, ignore_index=True) if remaining_frames else pd.DataFrame()

    incremental = pd.DataFrame(
        [
            {
                "comparison": "v48_direction_raw_vs_v37",
                "extra_review_n": int(cases["v48_raw_incremental_review_vs_v37"].sum()),
                "extra_rescued_p2_error_n": int(cases["v48_raw_incremental_rescue_vs_v37"].sum()),
                "extra_review_precision": float(cases["v48_raw_incremental_rescue_vs_v37"].sum() / cases["v48_raw_incremental_review_vs_v37"].sum()),
                "extra_rescued_fn_n": int(((cases["v48_raw_incremental_rescue_vs_v37"].eq(1)) & cases["p2_error_direction"].eq("FN_high_to_low")).sum()),
                "extra_rescued_fp_n": int(((cases["v48_raw_incremental_rescue_vs_v37"].eq(1)) & cases["p2_error_direction"].eq("FP_low_to_high")).sum()),
            },
            {
                "comparison": "v48_fn_ranker_vs_v48_direction_raw",
                "extra_review_n": int(cases["v48_fn_incremental_review_vs_v48_raw"].sum()),
                "extra_rescued_p2_error_n": int(cases["v48_fn_incremental_rescue_vs_v48_raw"].sum()),
                "extra_review_precision": float(cases["v48_fn_incremental_rescue_vs_v48_raw"].sum() / cases["v48_fn_incremental_review_vs_v48_raw"].sum()),
                "extra_rescued_fn_n": int(((cases["v48_fn_incremental_rescue_vs_v48_raw"].eq(1)) & cases["p2_error_direction"].eq("FN_high_to_low")).sum()),
                "extra_rescued_fp_n": int(((cases["v48_fn_incremental_rescue_vs_v48_raw"].eq(1)) & cases["p2_error_direction"].eq("FP_low_to_high")).sum()),
            },
        ]
    )

    metrics.to_csv(OUT_DIR / "v49_policy_metrics.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v49_external_case_transitions.csv", index=False, encoding="utf-8-sig")
    remaining.to_csv(OUT_DIR / "v49_remaining_error_breakdown.csv", index=False, encoding="utf-8-sig")
    incremental.to_csv(OUT_DIR / "v49_incremental_rescue_summary.csv", index=False, encoding="utf-8-sig")

    print("Policy metrics:")
    print(metrics[["policy", "review_rate", "balanced_accuracy", "accuracy", "fn", "fp", "captured_fn_n", "captured_fp_n", "remaining_error_n"]].to_string(index=False))
    print("\nIncremental rescue:")
    print(incremental.to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
