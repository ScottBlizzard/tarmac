from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v150_severe_shift_hybrid_correct_review_20260527"
V147_CASES = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_cases.csv"

INTERNAL_POLICY = "dev_stable_router_all_domains"
SEVERE_POLICY = "shift_aware_concept_directional_candidate"
INTERNAL_REVIEW_BUDGET = 0.30
AUTO_BUDGETS = [0.0, 0.1, 0.2, 0.3, 0.4]
REVIEW_EXTRA_BUDGETS = [0.0, 0.1, 0.2, 0.3, 0.4]


def apply_internal_review(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.loc[
        df["policy"].eq(INTERNAL_POLICY)
        & np.isclose(df["review_budget"].astype(float), INTERNAL_REVIEW_BUDGET)
        & df["eval_domain"].isin(["old_data", "third_batch"])
    ].copy()
    sub["auto_flip"] = False
    sub["doctor_review"] = sub["review"].astype(bool)
    sub["final_pred"] = sub["base_pred"].astype(int)
    sub.loc[sub["doctor_review"], "final_pred"] = sub.loc[sub["doctor_review"], "label_idx"].astype(int)
    return sub


def apply_severe_hybrid(df: pd.DataFrame, auto_budget: float, review_extra_budget: float) -> pd.DataFrame:
    sub = df.loc[
        df["policy"].eq(SEVERE_POLICY)
        & np.isclose(df["review_budget"].astype(float), INTERNAL_REVIEW_BUDGET)
        & df["eval_domain"].eq("strict_external")
    ].copy()
    sub = sub.sort_values("risk_score", ascending=False).reset_index(drop=True)
    n = len(sub)
    n_auto = int(round(auto_budget * n))
    n_review = int(round(review_extra_budget * n))
    sub["auto_flip"] = False
    sub["doctor_review"] = False
    if n_auto > 0:
        sub.loc[: n_auto - 1, "auto_flip"] = True
    remaining = sub.index[~sub["auto_flip"].astype(bool)].to_numpy()
    if n_review > 0 and len(remaining) > 0:
        sub.loc[remaining[: min(n_review, len(remaining))], "doctor_review"] = True
    sub["final_pred"] = sub["base_pred"].astype(int)
    sub.loc[sub["auto_flip"], "final_pred"] = 1 - sub.loc[sub["auto_flip"], "base_pred"].astype(int)
    sub.loc[sub["doctor_review"], "final_pred"] = sub.loc[sub["doctor_review"], "label_idx"].astype(int)
    return sub


def summarize(scope: str, cases: pd.DataFrame, auto_budget: float, review_extra_budget: float) -> dict[str, object]:
    y = cases["label_idx"].astype(int).to_numpy()
    base_pred = cases["base_pred"].astype(int).to_numpy()
    final_pred = cases["final_pred"].astype(int).to_numpy()
    prob = cases["base_prob"].to_numpy(float)
    auto_flip = cases["auto_flip"].astype(bool).to_numpy()
    review = cases["doctor_review"].astype(bool).to_numpy()
    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = base_wrong & ~final_wrong
    hurt = ~base_wrong & final_wrong
    row = {
        "scope": scope,
        "severe_auto_flip_budget": float(auto_budget),
        "severe_extra_review_budget": float(review_extra_budget),
        "n": int(len(cases)),
        "auto_flip_n": int(auto_flip.sum()),
        "auto_flip_rate": float(auto_flip.mean()) if len(cases) else np.nan,
        "doctor_review_n": int(review.sum()),
        "doctor_review_rate": float(review.mean()) if len(cases) else np.nan,
        "non_review_auto_n": int((~review).sum()),
        "base_errors": int(base_wrong.sum()),
        "final_errors": int(final_wrong.sum()),
        "net_errors_reduced": int(base_wrong.sum() - final_wrong.sum()),
        "rescued_n": int(rescued.sum()),
        "hurt_n": int(hurt.sum()),
        "auto_flip_rescued_n": int((auto_flip & base_wrong & ~final_wrong).sum()),
        "auto_flip_hurt_n": int((auto_flip & ~base_wrong & final_wrong).sum()),
    }
    row.update({f"base_{k}": v for k, v in metrics(y, base_pred, prob).items()})
    row.update({f"final_{k}": v for k, v in metrics(y, final_pred, prob).items()})
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V147_CASES, dtype={"case_id": str})
    internal = apply_internal_review(df)
    rows = []
    case_frames = []
    for auto_budget in AUTO_BUDGETS:
        for review_budget in REVIEW_EXTRA_BUDGETS:
            severe = apply_severe_hybrid(df, auto_budget, review_budget)
            all_cases = pd.concat([internal, severe], ignore_index=True)
            rows.append(summarize("old_third_internal_review", internal, auto_budget, review_budget))
            rows.append(summarize("strict_external_hybrid", severe, auto_budget, review_budget))
            rows.append(summarize("all_three_domains_hybrid", all_cases, auto_budget, review_budget))
            tmp = all_cases.copy()
            tmp["severe_auto_flip_budget"] = auto_budget
            tmp["severe_extra_review_budget"] = review_budget
            case_frames.append(tmp)
    summary = pd.DataFrame(rows).sort_values(
        ["scope", "severe_auto_flip_budget", "severe_extra_review_budget", "final_balanced_accuracy"],
        ascending=[True, True, True, False],
    )
    cases = pd.concat(case_frames, ignore_index=True)
    summary.to_csv(OUT_DIR / "v150_hybrid_correct_review_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v150_hybrid_correct_review_cases.csv", index=False, encoding="utf-8-sig")
    report = {
        "internal_policy": INTERNAL_POLICY,
        "internal_review_budget": INTERNAL_REVIEW_BUDGET,
        "severe_policy": SEVERE_POLICY,
        "boundary": "This is an exploratory severe-shift hybrid curve. Auto-correct/review budgets are scanned on the labeled strict external set and must not be treated as a locked no-leak policy.",
    }
    (OUT_DIR / "v150_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v150] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
