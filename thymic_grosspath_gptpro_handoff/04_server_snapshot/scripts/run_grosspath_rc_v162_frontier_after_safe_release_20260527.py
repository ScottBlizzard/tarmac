from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v160_safety_efficiency_operating_frontier_20260527 import (
    constraint_table,
    frontier,
    pct,
)


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v162_frontier_after_safe_release_20260527"
V160_POOL = ROOT / "outputs" / "grosspath_rc_v160_safety_efficiency_operating_frontier_20260527" / "v160_operating_point_pool.csv"
V161_SUMMARY = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_summary.csv"


def v161_pool_row() -> dict[str, object]:
    s = pd.read_csv(V161_SUMMARY)
    sub = s.loc[s["workflow"].eq("v161_safe_release_scorecard")].copy()
    all_row = sub.loc[sub["scope"].eq("all_domains")].iloc[0]
    domains = sub.loc[sub["scope"].isin(["old_data", "third_batch", "strict_external"])].copy()
    strict = domains.loc[domains["scope"].eq("strict_external")].iloc[0]
    return {
        "operating_point": "v161_safe_release_scorecard",
        "source": "v161_internal_zero_error_safe_release",
        "policy": "v161_safe_release_scorecard",
        "review_rate": float(all_row["review_rate"]),
        "auto_pass_rate": float(all_row["auto_pass_rate"]),
        "auto_correct_rate": 0.0,
        "all_bacc": float(all_row["balanced_accuracy"]),
        "all_acc": float(all_row["accuracy"]),
        "all_f1": float(all_row["f1"]),
        "all_fn": int(all_row["fn"]),
        "all_fp": int(all_row["fp"]),
        "strict_bacc": float(strict["balanced_accuracy"]),
        "strict_fn": int(strict["fn"]),
        "strict_fp": int(strict["fp"]),
        "min_domain_bacc": float(domains["balanced_accuracy"].min()),
        "max_domain_fn": int(domains["fn"].max()),
        "max_domain_fp": int(domains["fp"].max()),
        "status": "efficiency-improved high-safety candidate",
        "review_rate_round": round(float(all_row["review_rate"]), 3),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(V160_POOL)
    pool = pd.concat([pool, pd.DataFrame([v161_pool_row()])], ignore_index=True)
    pool = pool.drop_duplicates("operating_point", keep="last")
    selected = constraint_table(pool)
    front = frontier(pool)

    pool.to_csv(OUT_DIR / "v162_operating_point_pool_with_safe_release.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v162_constraint_selected_points.csv", index=False, encoding="utf-8-sig")
    front.to_csv(OUT_DIR / "v162_all_bacc_review_frontier.csv", index=False, encoding="utf-8-sig")

    min95 = selected.loc[selected["constraint"].eq("min_domain_bacc_at_least_95_min_review")].iloc[0]
    zero = selected.loc[selected["constraint"].eq("near_zero_error_high_safety")].iloc[0]
    max60 = selected.loc[selected["constraint"].eq("max_60pct_review_best_all_bacc")].iloc[0]

    md = [
        "# v162 Frontier After Safe-release",
        "",
        "## Main Findings",
        "",
        (
            f"- After adding v161, the minimum-review point satisfying min-domain BAcc >=95% is "
            f"`{min95['operating_point']}` with review/reject {pct(min95['review_rate'])}, "
            f"auto-pass {pct(min95['auto_pass_rate'])}, all-domain BAcc {pct(min95['all_bacc'])}."
        ),
        (
            f"- The near-zero-error point is also `{zero['operating_point']}` with FN={int(zero['all_fn'])}, "
            f"FP={int(zero['all_fp'])}, review/reject {pct(zero['review_rate'])}."
        ),
        (
            f"- Under a <=60% review constraint, the best all-domain BAcc point becomes `{max60['operating_point']}` "
            f"with BAcc {pct(max60['all_bacc'])}."
        ),
        "",
        "## Interpretation",
        "",
        "v161 changes the deployment story: high safety no longer necessarily means about 80% review. With internally selected safe-release rules, the current candidate keeps the same near-zero-error profile while moving the review burden into the high-50% range. The remaining open problem is prospective validation of these safe-release thresholds and further reduction of review burden.",
    ]
    (OUT_DIR / "v162_frontier_after_safe_release.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "pool_rows": int(len(pool)),
        "frontier_rows": int(len(front)),
        "min_domain_ge95_point": str(min95["operating_point"]),
        "min_domain_ge95_review_rate": float(min95["review_rate"]),
        "near_zero_error_point": str(zero["operating_point"]),
        "near_zero_error_review_rate": float(zero["review_rate"]),
        "max60_best_point": str(max60["operating_point"]),
        "max60_best_all_bacc": float(max60["all_bacc"]),
    }
    (OUT_DIR / "v162_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v162] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
