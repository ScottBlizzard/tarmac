from __future__ import annotations

import json
import math

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v198_release_compressed_safety_ci_20260527"
V187_CI = ROOT / "outputs" / "grosspath_rc_v187_release_safety_ci_planning_20260527" / "v187_auto_decision_error_ci.csv"
V195_SUMMARY = ROOT / "outputs" / "grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527" / "v195_selected_candidate_summary.csv"


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = pd.read_csv(V187_CI)
    selected = pd.read_csv(V195_SUMMARY)

    rows = []
    for _, r in selected.iterrows():
        scope = str(r["scope"])
        n = int(r["n"])
        review_n = int(r["remaining_review_n"])
        auto_n = n - review_n
        auto_error_n = int(r["system_errors_after_review"])
        low, high = wilson_interval(auto_error_n, auto_n)
        rows.append(
            {
                "policy": "v195_stable_fixed_agreement_release",
                "scope": scope,
                "n": n,
                "auto_decision_n": auto_n,
                "auto_error_n": auto_error_n,
                "auto_error_rate": auto_error_n / auto_n if auto_n else math.nan,
                "wilson95_low": low,
                "wilson95_high": high,
                "review_or_reject_n": review_n,
                "review_or_reject_rate": float(r["remaining_review_rate"]),
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
            }
        )
    v195_ci = pd.DataFrame(rows)

    # Keep the previous v185 rows beside v195 for direct comparison.
    prior_keep = prior.loc[prior["policy"].eq("v185_unlabeled_shift_adaptive")].copy()
    prior_keep["n"] = prior_keep["auto_decision_n"].astype(int) + prior_keep["review_or_reject_n"].astype(int)
    prior_keep["balanced_accuracy"] = ""
    prior_keep["fn"] = ""
    prior_keep["fp"] = ""
    prior_keep = prior_keep[
        [
            "policy",
            "scope",
            "n",
            "auto_decision_n",
            "auto_error_n",
            "auto_error_rate",
            "wilson95_low",
            "wilson95_high",
            "review_or_reject_n",
            "review_or_reject_rate",
            "balanced_accuracy",
            "fn",
            "fp",
        ]
    ]
    table = pd.concat([prior_keep, v195_ci], ignore_index=True)
    table.to_csv(OUT_DIR / "v198_release_compressed_safety_ci.csv", index=False, encoding="utf-8-sig")

    all_v185 = table.loc[table["policy"].eq("v185_unlabeled_shift_adaptive") & table["scope"].eq("all_domains")].iloc[0]
    all_v195 = table.loc[table["policy"].eq("v195_stable_fixed_agreement_release") & table["scope"].eq("all_domains")].iloc[0]
    strict_v185 = table.loc[table["policy"].eq("v185_unlabeled_shift_adaptive") & table["scope"].eq("strict_external")].iloc[0]
    strict_v195 = table.loc[table["policy"].eq("v195_stable_fixed_agreement_release") & table["scope"].eq("strict_external")].iloc[0]

    md = [
        "# v198 Release-compressed Safety CI",
        "",
        "## Main Result",
        "",
        (
            f"v195 increases all-domain automatic decisions from {int(all_v185['auto_decision_n'])} to "
            f"{int(all_v195['auto_decision_n'])} while keeping the observed automatic error count at "
            f"{int(all_v195['auto_error_n'])}. The all-domain review/reject rate drops from "
            f"{pct(all_v185['review_or_reject_rate'])} to {pct(all_v195['review_or_reject_rate'])}."
        ),
        "",
        "## Confidence-bound Safety",
        "",
        (
            f"- All-domain: v185 observed auto-error {pct(all_v185['auto_error_rate'])}, Wilson95 upper "
            f"{pct(all_v185['wilson95_high'])}; v195 observed auto-error {pct(all_v195['auto_error_rate'])}, "
            f"Wilson95 upper {pct(all_v195['wilson95_high'])}."
        ),
        (
            f"- Strict external: v185 auto decisions {int(strict_v185['auto_decision_n'])}, Wilson95 upper "
            f"{pct(strict_v185['wilson95_high'])}; v195 auto decisions {int(strict_v195['auto_decision_n'])}, "
            f"Wilson95 upper {pct(strict_v195['wilson95_high'])}."
        ),
        "",
        "## Interpretation",
        "",
        "v195 should be treated as the current efficiency-upgraded safety workflow: it compresses the review/reject pool and tightens the all-domain automatic-decision confidence bound. The strict external confidence interval remains limited by sample size, even though the point estimate has zero automatic errors.",
    ]
    (OUT_DIR / "v198_release_compressed_safety_ci.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "all_domain_v185_auto_n": int(all_v185["auto_decision_n"]),
        "all_domain_v195_auto_n": int(all_v195["auto_decision_n"]),
        "all_domain_v185_review_rate": float(all_v185["review_or_reject_rate"]),
        "all_domain_v195_review_rate": float(all_v195["review_or_reject_rate"]),
        "all_domain_v185_wilson95_high": float(all_v185["wilson95_high"]),
        "all_domain_v195_wilson95_high": float(all_v195["wilson95_high"]),
        "strict_external_v185_auto_n": int(strict_v185["auto_decision_n"]),
        "strict_external_v195_auto_n": int(strict_v195["auto_decision_n"]),
        "strict_external_v185_wilson95_high": float(strict_v185["wilson95_high"]),
        "strict_external_v195_wilson95_high": float(strict_v195["wilson95_high"]),
    }
    (OUT_DIR / "v198_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v198] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
