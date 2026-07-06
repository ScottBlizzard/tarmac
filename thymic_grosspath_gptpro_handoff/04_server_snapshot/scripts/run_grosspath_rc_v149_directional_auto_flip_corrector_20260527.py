from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v149_directional_auto_flip_corrector_20260527"
V147_CASES = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_cases.csv"

FOCUS_POLICIES = [
    "baseline_low_conf_dev_selected",
    "dev_stable_router_all_domains",
    "shift_aware_image_directional_candidate",
    "shift_aware_concept_directional_candidate",
]
DOMAINS = ["old_data", "third_batch", "strict_external", "all_three_domains"]


def add_auto_flip(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["auto_flip"] = out["review"].astype(bool)
    out["flip_pred"] = out["base_pred"].astype(int)
    out.loc[out["auto_flip"], "flip_pred"] = 1 - out.loc[out["auto_flip"], "base_pred"].astype(int)
    y = out["label_idx"].astype(int)
    base_pred = out["base_pred"].astype(int)
    flip_pred = out["flip_pred"].astype(int)
    out["base_wrong"] = base_pred.ne(y)
    out["flip_wrong"] = flip_pred.ne(y)
    out["rescued_by_flip"] = out["base_wrong"] & ~out["flip_wrong"]
    out["hurt_by_flip"] = ~out["base_wrong"] & out["flip_wrong"]
    out["rescued_fn_by_flip"] = out["rescued_by_flip"] & y.eq(1) & base_pred.eq(0)
    out["rescued_fp_by_flip"] = out["rescued_by_flip"] & y.eq(0) & base_pred.eq(1)
    out["hurt_to_fn_by_flip"] = out["hurt_by_flip"] & y.eq(1) & flip_pred.eq(0)
    out["hurt_to_fp_by_flip"] = out["hurt_by_flip"] & y.eq(0) & flip_pred.eq(1)
    return out


def summarize_one(policy: str, budget: float, domain: str, df: pd.DataFrame) -> dict[str, object]:
    sub = df.loc[df["policy"].eq(policy) & np.isclose(df["review_budget"].astype(float), budget)].copy()
    if domain != "all_three_domains":
        sub = sub.loc[sub["eval_domain"].eq(domain)].copy()
    sub = add_auto_flip(sub)
    y = sub["label_idx"].astype(int).to_numpy()
    base_pred = sub["base_pred"].astype(int).to_numpy()
    flip_pred = sub["flip_pred"].astype(int).to_numpy()
    prob = sub["base_prob"].to_numpy(float)
    review = sub["review"].astype(bool).to_numpy()
    row = {
        "policy": policy,
        "eval_domain": domain,
        "flip_budget": float(budget),
        "n": int(len(sub)),
        "flip_n": int(review.sum()),
        "flip_rate": float(review.mean()) if len(sub) else np.nan,
        "base_errors": int((base_pred != y).sum()),
        "flip_errors": int((flip_pred != y).sum()),
        "net_errors_reduced": int((base_pred != y).sum() - (flip_pred != y).sum()),
        "rescued_by_flip": int(sub["rescued_by_flip"].sum()),
        "hurt_by_flip": int(sub["hurt_by_flip"].sum()),
        "rescued_fn_by_flip": int(sub["rescued_fn_by_flip"].sum()),
        "rescued_fp_by_flip": int(sub["rescued_fp_by_flip"].sum()),
        "hurt_to_fn_by_flip": int(sub["hurt_to_fn_by_flip"].sum()),
        "hurt_to_fp_by_flip": int(sub["hurt_to_fp_by_flip"].sum()),
    }
    row.update({f"base_{k}": v for k, v in metrics(y, base_pred, prob).items()})
    row.update({f"auto_flip_{k}": v for k, v in metrics(y, flip_pred, prob).items()})
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V147_CASES, dtype={"case_id": str})
    df = df.loc[df["policy"].isin(FOCUS_POLICIES)].copy()
    rows = []
    for policy in FOCUS_POLICIES:
        for budget in sorted(df["review_budget"].astype(float).unique()):
            for domain in DOMAINS:
                rows.append(summarize_one(policy, float(budget), domain, df))
    summary = pd.DataFrame(rows).sort_values(
        ["eval_domain", "flip_budget", "auto_flip_balanced_accuracy"],
        ascending=[True, True, False],
    )
    case_out = add_auto_flip(df)
    summary.to_csv(OUT_DIR / "v149_auto_flip_corrector_summary.csv", index=False, encoding="utf-8-sig")
    case_out.to_csv(OUT_DIR / "v149_auto_flip_corrector_cases.csv", index=False, encoding="utf-8-sig")
    report = {
        "interpretation": "The high-risk routed subset is automatically flipped instead of reviewed. This tests whether the router has enough precision to serve as an automatic corrector.",
        "focus_policies": FOCUS_POLICIES,
    }
    (OUT_DIR / "v149_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v149] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
