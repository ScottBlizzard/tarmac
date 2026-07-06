from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v51_workflow_validation_20260527 as v51  # noqa: E402
import run_grosspath_rc_v54_constrained_policy_search_20260527 as v54  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v58_fp_buffer_after_v54_20260527"
BASE_POLICY = "fusion_rank::dir_plus_pred_low_fn::budget=0.650"
ADDON_RATES = np.round(np.arange(0.0, 0.151, 0.01), 3)

SCENARIOS = [
    {"scenario": "sens985_spec97", "sens_min": 0.985, "spec_min": 0.970},
    {"scenario": "sens985_spec98", "sens_min": 0.985, "spec_min": 0.980},
    {"scenario": "sens99_spec97", "sens_min": 0.990, "spec_min": 0.970},
    {"scenario": "sens99_spec98", "sens_min": 0.990, "spec_min": 0.980},
]


def add_pred_high_buffer(df: pd.DataFrame, base_review: np.ndarray, score: np.ndarray, addon_rate: float) -> np.ndarray:
    review = base_review.copy()
    n_add = int(round(len(df) * addon_rate))
    if n_add <= 0:
        return review
    p2 = df["p2_pred"].to_numpy(dtype=int)
    available = (~review) & (p2 == 1)
    idx = np.flatnonzero(available)
    if len(idx) == 0:
        return review
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    review[order[: min(n_add, len(order))]] = True
    return review


def rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="mergesort")
    out = np.zeros(len(score), dtype=float)
    if len(score) > 1:
        out[order] = np.linspace(0.0, 1.0, len(score))
    return out


def addon_scores(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Higher score means more likely to be a low-risk case incorrectly upgraded to high-risk.
    main_margin = -df["main_margin_abs"].to_numpy(dtype=float) if "main_margin_abs" in df.columns else np.zeros(len(df))
    robust_margin = -df["robust_margin_abs"].to_numpy(dtype=float) if "robust_margin_abs" in df.columns else np.zeros(len(df))
    prob_std = df["core_prob_std"].to_numpy(dtype=float) if "core_prob_std" in df.columns else np.zeros(len(df))
    return {
        "fp_risk": scores["fp"],
        "any_risk": scores["any"],
        "direction_risk": scores["direction"],
        "uncertain_main": main_margin,
        "uncertain_robust": robust_margin,
        "model_std": prob_std,
        "fp_plus_uncertain": 0.7 * rank01(scores["fp"]) + 0.3 * rank01(main_margin),
        "fp_plus_std": 0.7 * rank01(scores["fp"]) + 0.3 * rank01(prob_std),
    }


def metric(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    final = v51.final_prediction(df, review)
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(df)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "remaining_error_n": int((final != y).sum()),
        }
    )
    return m


def build_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    _grid, ext_reviews = v54.build_policy_grid(dev, ext, dev_scores, ext_scores)
    # Rebuild dev base review from candidate score.
    base_dev = None
    for cand in v54.build_score_candidates(dev, ext, dev_scores, ext_scores):
        if cand["family"] == "fusion_rank" and cand["name"] == "dir_plus_pred_low_fn":
            base_dev = v54.top_by_score(cand["dev_score"], 0.650)
            break
    if base_dev is None:
        raise RuntimeError("Base policy not found")
    base_ext = ext_reviews[BASE_POLICY]

    dev_addons = addon_scores(dev, dev_scores)
    ext_addons = addon_scores(ext, ext_scores)
    rows = []
    for addon_name in dev_addons:
        for addon_rate in ADDON_RATES:
            dev_review = add_pred_high_buffer(dev, base_dev, dev_addons[addon_name], float(addon_rate))
            ext_review = add_pred_high_buffer(ext, base_ext, ext_addons[addon_name], float(addon_rate))
            rows.append(
                {
                    "addon_name": addon_name,
                    "addon_rate": float(addon_rate),
                    **{f"dev_{k}": v for k, v in metric(dev, dev_review).items()},
                    **{f"external_{k}": v for k, v in metric(ext, ext_review).items()},
                }
            )
    return pd.DataFrame(rows), ext


def select_by_dev(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        ok = grid.loc[(grid["dev_sensitivity"] >= sc["sens_min"]) & (grid["dev_specificity"] >= sc["spec_min"])].copy()
        if ok.empty:
            chosen = grid.sort_values(["dev_sensitivity", "dev_specificity", "dev_balanced_accuracy"], ascending=[False, False, False]).iloc[0]
            met = 0
        else:
            chosen = ok.sort_values(
                ["dev_review_rate", "dev_specificity", "dev_balanced_accuracy"],
                ascending=[True, False, False],
            ).iloc[0]
            met = 1
        row = chosen.to_dict()
        row.update({"scenario": sc["scenario"], "sens_min": sc["sens_min"], "spec_min": sc["spec_min"], "constraints_met": met})
        rows.append(row)
    return pd.DataFrame(rows)


def external_oracle(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        ok = grid.loc[(grid["external_sensitivity"] >= sc["sens_min"]) & (grid["external_specificity"] >= sc["spec_min"])].copy()
        if ok.empty:
            continue
        chosen = ok.sort_values(["external_review_rate", "external_balanced_accuracy"], ascending=[True, False]).iloc[0]
        row = chosen.to_dict()
        row.update({"scenario": sc["scenario"], "sens_min": sc["sens_min"], "spec_min": sc["spec_min"]})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, _ext = build_grid()
    selected = select_by_dev(grid)
    oracle = external_oracle(grid)
    grid.to_csv(OUT_DIR / "v58_fp_buffer_grid.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v58_dev_selected_fp_buffer.csv", index=False, encoding="utf-8-sig")
    oracle.to_csv(OUT_DIR / "v58_external_oracle_fp_buffer.csv", index=False, encoding="utf-8-sig")

    show_cols = [
        "scenario",
        "addon_name",
        "addon_rate",
        "dev_review_rate",
        "dev_sensitivity",
        "dev_specificity",
        "external_review_rate",
        "external_balanced_accuracy",
        "external_sensitivity",
        "external_specificity",
        "external_fn",
        "external_fp",
    ]
    print("Dev-selected FP buffer:")
    print(selected[show_cols].to_string(index=False))
    if not oracle.empty:
        print("\nExternal oracle upper bound:")
        print(oracle[show_cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
