from __future__ import annotations

import json
from math import ceil, sqrt

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v187_release_safety_ci_planning_20260527"
V185_CASES = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_cases.csv"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def n_for_wilson_upper_zero(target_upper: float, z: float = 1.959963984540054) -> int:
    n = 1
    while True:
        _, upper = wilson(0, n, z)
        if upper <= target_upper:
            return n
        n += 1


def n_for_wilson_upper_one(target_upper: float, z: float = 1.959963984540054) -> int:
    n = 2
    while True:
        _, upper = wilson(1, n, z)
        if upper <= target_upper:
            return n
        n += 1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["fixed_v118_review", "fixed_v182_review", "adaptive_review"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])

    policies = {
        "fixed_v118_high_safety": "fixed_v118_review",
        "fixed_v182_stable_release": "fixed_v182_review",
        "v185_unlabeled_shift_adaptive": "adaptive_review",
    }
    rows = []
    for policy, review_col in policies.items():
        auto = ~df[review_col]
        auto_error = auto & df["base_wrong"]
        for scope, mask in [
            ("old_data", df["domain"].eq("old_data")),
            ("third_batch", df["domain"].eq("third_batch")),
            ("strict_external", df["domain"].eq("strict_external")),
            ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])),
        ]:
            n = int((auto & mask).sum())
            k = int((auto_error & mask).sum())
            lo, hi = wilson(k, n)
            rows.append(
                {
                    "policy": policy,
                    "scope": scope,
                    "auto_decision_n": n,
                    "auto_error_n": k,
                    "auto_error_rate": float(k / n) if n else float("nan"),
                    "wilson95_low": lo,
                    "wilson95_high": hi,
                    "review_or_reject_n": int((df[review_col] & mask).sum()),
                    "review_or_reject_rate": float((df[review_col] & mask).mean()),
                }
            )
    ci = pd.DataFrame(rows)
    ci.to_csv(OUT_DIR / "v187_auto_decision_error_ci.csv", index=False, encoding="utf-8-sig")

    planning = []
    for target in [0.10, 0.075, 0.05, 0.025, 0.01]:
        planning.append(
            {
                "target_wilson95_upper": target,
                "auto_decision_n_needed_if_zero_errors": n_for_wilson_upper_zero(target),
                "auto_decision_n_needed_if_one_error": n_for_wilson_upper_one(target),
            }
        )
    plan = pd.DataFrame(planning)
    plan.to_csv(OUT_DIR / "v187_prospective_auto_decision_sample_size.csv", index=False, encoding="utf-8-sig")

    all_adaptive = ci.loc[ci["policy"].eq("v185_unlabeled_shift_adaptive") & ci["scope"].eq("all_domains")].iloc[0]
    strict_adaptive = ci.loc[
        ci["policy"].eq("v185_unlabeled_shift_adaptive") & ci["scope"].eq("strict_external")
    ].iloc[0]
    all_v182 = ci.loc[ci["policy"].eq("fixed_v182_stable_release") & ci["scope"].eq("all_domains")].iloc[0]

    md = [
        "# v187 Release Safety CI and Prospective Planning",
        "",
        "## Current Wilson Boundary",
        "",
        (
            f"- v185 adaptive all-domain auto decisions: n={int(all_adaptive['auto_decision_n'])}, "
            f"errors={int(all_adaptive['auto_error_n'])}, observed error rate "
            f"{100 * float(all_adaptive['auto_error_rate']):.2f}%, Wilson95 upper "
            f"{100 * float(all_adaptive['wilson95_high']):.2f}%."
        ),
        (
            f"- v185 adaptive strict external auto decisions: n={int(strict_adaptive['auto_decision_n'])}, "
            f"errors={int(strict_adaptive['auto_error_n'])}, Wilson95 upper "
            f"{100 * float(strict_adaptive['wilson95_high']):.2f}%."
        ),
        (
            f"- fixed v182 all-domain auto decisions: n={int(all_v182['auto_decision_n'])}, "
            f"errors={int(all_v182['auto_error_n'])}, Wilson95 upper "
            f"{100 * float(all_v182['wilson95_high']):.2f}%."
        ),
        "",
        "## Prospective Planning",
        "",
        "- If future prospective auto decisions have zero errors, 35 auto-decided cases are needed for a Wilson95 upper bound <=10%, and 73 are needed for <=5%.",
        "- If one error is allowed, 53 auto-decided cases are needed for <=10%, and 110 are needed for <=5%.",
        "",
        "## Writing Boundary",
        "",
        "Observed zero-error release should be reported with Wilson intervals and framed as a current-split safety estimate, not a guaranteed clinical error rate.",
    ]
    (OUT_DIR / "v187_release_safety_ci_planning.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "adaptive_all_auto_n": int(all_adaptive["auto_decision_n"]),
        "adaptive_all_auto_errors": int(all_adaptive["auto_error_n"]),
        "adaptive_all_wilson95_upper": float(all_adaptive["wilson95_high"]),
        "adaptive_strict_external_auto_n": int(strict_adaptive["auto_decision_n"]),
        "adaptive_strict_external_wilson95_upper": float(strict_adaptive["wilson95_high"]),
        "fixed_v182_all_auto_n": int(all_v182["auto_decision_n"]),
        "fixed_v182_all_wilson95_upper": float(all_v182["wilson95_high"]),
        "n_zero_errors_for_upper_05": int(plan.loc[plan["target_wilson95_upper"].eq(0.05), "auto_decision_n_needed_if_zero_errors"].iloc[0]),
    }
    (OUT_DIR / "v187_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v187] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
