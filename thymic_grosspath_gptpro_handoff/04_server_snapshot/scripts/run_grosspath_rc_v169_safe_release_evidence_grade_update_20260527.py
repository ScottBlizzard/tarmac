from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool, pct


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v169_safe_release_evidence_grade_update_20260527"
V161_SUMMARY = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_summary.csv"
V161_CASES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_cases.csv"
V164_FOCUS = ROOT / "outputs" / "grosspath_rc_v164_stable_safe_release_strategy_compare_20260527" / "v164_all_domain_strategy_focus.csv"
V167_SUMMARY = ROOT / "outputs" / "grosspath_rc_v167_nested_safe_release_validation_20260527" / "v167_nested_safe_release_summary.csv"
V168_SUMMARY = ROOT / "outputs" / "grosspath_rc_v168_conservative_nested_safe_release_20260527" / "v168_conservative_nested_summary.csv"


def v161_fold_audit() -> pd.DataFrame:
    df = pd.read_csv(V161_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v161_safe_release_from_review", "v161_final_review_or_reject", "base_wrong"]:
        df[col] = as_bool(df[col])
    rows = []
    internal = df.loc[df["domain"].isin(["old_data", "third_batch"])].copy()
    for (domain, fold), sub in internal.groupby(["domain", "fold_id"]):
        rel = sub["v161_safe_release_from_review"]
        rows.append(
            {
                "domain": domain,
                "fold_id": int(fold),
                "n": int(len(sub)),
                "released_n": int(rel.sum()),
                "released_error_n": int((rel & sub["base_wrong"]).sum()),
                "final_review_rate": float(sub["v161_final_review_or_reject"].mean()),
                "auto_pass_rate": float((~sub["v161_final_review_or_reject"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def evidence_grade() -> pd.DataFrame:
    v161 = pd.read_csv(V161_SUMMARY)
    v164 = pd.read_csv(V164_FOCUS)
    v167 = pd.read_csv(V167_SUMMARY)
    v168 = pd.read_csv(V168_SUMMARY)
    v161_all = v161.loc[v161["workflow"].eq("v161_safe_release_scorecard") & v161["scope"].eq("all_domains")].iloc[0]
    v161_ext = v161.loc[v161["workflow"].eq("v161_safe_release_scorecard") & v161["scope"].eq("strict_external")].iloc[0]
    old_only = v164.loc[v164["strategy"].eq("old_only_selected")].iloc[0]
    nested = v167.loc[v167["workflow"].eq("nested_internal_safe_release") & v167["scope"].eq("internal_all")].iloc[0]
    inner = v168.loc[v168["workflow"].eq("outer_inner_fold_intersection") & v168["scope"].eq("internal_all")].iloc[0]
    return pd.DataFrame(
        [
            {
                "evidence_item": "Fixed old+third safe-release candidate",
                "source": "v161/v162/v164",
                "review_rate": float(v161_all["review_rate"]),
                "auto_pass_rate": float(v161_all["auto_pass_rate"]),
                "balanced_accuracy": float(v161_all["balanced_accuracy"]),
                "fn": int(v161_all["fn"]),
                "fp": int(v161_all["fp"]),
                "released_error_n": int(v161_all["released_error_n"]),
                "evidence_grade": "current-data high-safety candidate",
                "paper_use": "Main efficiency candidate, with explicit prospective-validation caveat.",
            },
            {
                "evidence_item": "Strict external observation under fixed candidate",
                "source": "v161",
                "review_rate": float(v161_ext["review_rate"]),
                "auto_pass_rate": float(v161_ext["auto_pass_rate"]),
                "balanced_accuracy": float(v161_ext["balanced_accuracy"]),
                "fn": int(v161_ext["fn"]),
                "fp": int(v161_ext["fp"]),
                "released_error_n": int(v161_ext["released_error_n"]),
                "evidence_grade": "external observation, not selection",
                "paper_use": "External validation row; do not claim threshold was tuned on external.",
            },
            {
                "evidence_item": "Nested fold-wise max-release selection",
                "source": "v167",
                "review_rate": float(nested["review_rate"]),
                "auto_pass_rate": float(nested["auto_pass_rate"]),
                "balanced_accuracy": float(nested["balanced_accuracy"]),
                "fn": int(nested["fn"]),
                "fp": int(nested["fp"]),
                "released_error_n": int(nested["released_error_n"]),
                "evidence_grade": "does not pass strict nested safety",
                "paper_use": "Boundary/limitation: maximizing release inside folds is unstable.",
            },
            {
                "evidence_item": "Conservative inner-fold intersection",
                "source": "v168",
                "review_rate": float(inner["review_rate"]),
                "auto_pass_rate": float(inner["auto_pass_rate"]),
                "balanced_accuracy": float(inner["balanced_accuracy"]),
                "fn": int(inner["fn"]),
                "fp": int(inner["fp"]),
                "released_error_n": int(inner["released_error_n"]),
                "evidence_grade": "does not pass strict nested safety",
                "paper_use": "Negative result: simple conservative intersection is insufficient.",
            },
            {
                "evidence_item": "Old-only aggressive release",
                "source": "v164",
                "review_rate": float(old_only["review_rate"]),
                "auto_pass_rate": float(old_only["auto_pass_rate"]),
                "balanced_accuracy": float(old_only["balanced_accuracy"]),
                "fn": int(old_only["fn"]),
                "fp": int(old_only["fp"]),
                "released_error_n": int(old_only["released_error_n"]),
                "evidence_grade": "negative ablation",
                "paper_use": "Shows why multi-domain constraints are required.",
            },
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold = v161_fold_audit()
    grade = evidence_grade()
    fold.to_csv(OUT_DIR / "v169_v161_fixed_candidate_fold_audit.csv", index=False, encoding="utf-8-sig")
    grade.to_csv(OUT_DIR / "v169_safe_release_evidence_grade.csv", index=False, encoding="utf-8-sig")

    main = grade.loc[grade["evidence_item"].eq("Fixed old+third safe-release candidate")].iloc[0]
    nested = grade.loc[grade["evidence_item"].eq("Nested fold-wise max-release selection")].iloc[0]
    old_only = grade.loc[grade["evidence_item"].eq("Old-only aggressive release")].iloc[0]
    md = [
        "# v169 Safe-release Evidence Grade Update",
        "",
        "## Revised Claim",
        "",
        (
            f"The safe-release module should be written as a current-data high-safety efficiency candidate: "
            f"BAcc {pct(main['balanced_accuracy'])}, review/reject {pct(main['review_rate'])}, "
            f"auto-pass {pct(main['auto_pass_rate'])}, FN={int(main['fn'])}, FP={int(main['fp'])}."
        ),
        "",
        "## What Cannot Be Claimed",
        "",
        (
            f"It should not be called a nested-validated locked threshold. v167 nested max-release leaves "
            f"{int(nested['released_error_n'])} released errors, and v168 conservative intersection still leaves released errors."
        ),
        "",
        "## Why The Module Is Still Useful",
        "",
        (
            f"The old-only ablation releases too aggressively and leaves {int(old_only['released_error_n'])} released errors. "
            "This supports the method design: safe-release must be constrained by multiple internal domains and reported with a safety boundary."
        ),
        "",
        "## Recommended Wording",
        "",
        "We developed a multi-domain constrained safe-release module to reduce the review burden of the high-safety scorecard. On the current old+third development set and strict external observation, the fixed old+third rule reduces review from about 80% to 57.5% while preserving the current near-zero-error profile. However, fold-wise nested experiments show that reselecting release thresholds inside smaller folds can release errors; therefore, the module is reported as a strong efficiency candidate requiring prospective validation, not as a locked deployment threshold.",
    ]
    (OUT_DIR / "v169_safe_release_evidence_grade_update.md").write_text("\n".join(md), encoding="utf-8")
    report = {
        "main_candidate_review_rate": float(main["review_rate"]),
        "main_candidate_bacc": float(main["balanced_accuracy"]),
        "main_candidate_fn": int(main["fn"]),
        "main_candidate_fp": int(main["fp"]),
        "nested_released_error_n": int(nested["released_error_n"]),
        "old_only_released_error_n": int(old_only["released_error_n"]),
        "evidence_grade": str(main["evidence_grade"]),
    }
    (OUT_DIR / "v169_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v169] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
