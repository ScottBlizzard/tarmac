from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v181_current_system_after_nested_audit_20260527"
V179 = ROOT / "outputs" / "grosspath_rc_v179_current_system_evidence_table_20260527" / "v179_current_system_operating_table.csv"
V180 = ROOT / "outputs" / "grosspath_rc_v180_nested_v178_image_agreement_release_20260527" / "v180_nested_workflow_summary.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v179 = pd.read_csv(V179)
    v180 = pd.read_csv(V180)

    rows = []
    keep_modules = [
        "v118 locked high-safety two-signal scorecard",
        "v161 safe-release scorecard",
        "v178 safe-release + image agreement",
        "30% low-confidence selective review",
        "30% direction-aware image router",
        "v173 aggressive image-only review corrector",
        "v174 disagreement flip",
        "v175 error-enriched flip-risk",
    ]
    table = v179.loc[v179["module"].isin(keep_modules)].copy()
    table["revised_status_after_v180"] = table["tier"]
    table.loc[
        table["module"].eq("v178 safe-release + image agreement"),
        "revised_status_after_v180",
    ] = "Current-data candidate; downgraded by v180 nested audit"
    table.loc[
        table["module"].eq("v161 safe-release scorecard"),
        "revised_status_after_v180",
    ] = "Recommended efficiency candidate"
    table.loc[
        table["module"].eq("v118 locked high-safety two-signal scorecard"),
        "revised_status_after_v180",
    ] = "Recommended high-safety baseline"

    nested = v180.loc[v180["workflow"].eq("v180_nested_image_agreement_union")].copy()
    for _, r in nested.iterrows():
        rows.append(
            {
                "module": "v180 nested image-agreement audit",
                "tier": "Stability audit / caution",
                "role": "nested validation of v178 incremental image release",
                "scope": r["scope"],
                "auto_decision_rate": 1.0 - float(r["final_review_rate"]),
                "remaining_review_or_reject_rate": float(r["final_review_rate"]),
                "auto_correct_or_release_rate": "",
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "accuracy": float(r["accuracy"]),
                "f1": float(r["f1"]),
                "auc": float(r["auc"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "auto_action_error_n": int(r["released_error_n"]),
                "source": "v180",
                "writing_status": "Stability audit; do not promote v178",
                "revised_status_after_v180": "Negative stability evidence for v178",
            }
        )
    revised = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    revised.to_csv(OUT_DIR / "v181_revised_current_system_operating_table.csv", index=False, encoding="utf-8-sig")

    all_rows = revised.loc[revised["scope"].eq("all_domains")].copy()
    v118 = all_rows.loc[all_rows["module"].eq("v118 locked high-safety two-signal scorecard")].iloc[0]
    v161 = all_rows.loc[all_rows["module"].eq("v161 safe-release scorecard")].iloc[0]
    v178 = all_rows.loc[all_rows["module"].eq("v178 safe-release + image agreement")].iloc[0]
    v180r = all_rows.loc[all_rows["module"].eq("v180 nested image-agreement audit")].iloc[0]

    claims = pd.DataFrame(
        [
            {
                "claim": "Primary high-safety workflow remains v118 two-signal scorecard.",
                "status_after_v180": "supported",
                "evidence": f"BAcc {pct(v118['balanced_accuracy'])}, review {pct(v118['remaining_review_or_reject_rate'])}, FN={int(v118['fn'])}, FP={int(v118['fp'])}.",
                "writing_boundary": "Stable main safety baseline.",
            },
            {
                "claim": "Safe-release can reduce review burden with current zero released errors.",
                "status_after_v180": "supported as efficiency candidate",
                "evidence": f"v161 BAcc {pct(v161['balanced_accuracy'])}, review {pct(v161['remaining_review_or_reject_rate'])}, released errors {v161['auto_action_error_n']}.",
                "writing_boundary": "Still needs prospective validation, but stronger than v178 incremental rule.",
            },
            {
                "claim": "Image-agreement can safely improve v161 review burden.",
                "status_after_v180": "downgraded",
                "evidence": f"Full-fit v178 review {pct(v178['remaining_review_or_reject_rate'])}, but nested v180 released errors {int(v180r['auto_action_error_n'])} and BAcc {pct(v180r['balanced_accuracy'])}.",
                "writing_boundary": "Keep as current-data ablation, not as recommended workflow.",
            },
            {
                "claim": "Automatic flipping is a mature correction module.",
                "status_after_v180": "not supported",
                "evidence": "v174/v175 flip-risk produced many hurts; v180 further shows release expansion can destabilize.",
                "writing_boundary": "Future work only.",
            },
        ]
    )
    claims.to_csv(OUT_DIR / "v181_revised_claim_boundary_table.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v181 Current System After Nested Audit",
        "",
        "## Revised Recommendation",
        "",
        (
            f"- Main safety baseline: v118, BAcc {pct(v118['balanced_accuracy'])}, review "
            f"{pct(v118['remaining_review_or_reject_rate'])}, FN={int(v118['fn'])}, FP={int(v118['fp'])}."
        ),
        (
            f"- Recommended efficiency candidate: v161, BAcc {pct(v161['balanced_accuracy'])}, review "
            f"{pct(v161['remaining_review_or_reject_rate'])}, released errors {v161['auto_action_error_n']}."
        ),
        (
            f"- Downgraded candidate: v178 full-fit looks good, but v180 nested audit gives released errors "
            f"{int(v180r['auto_action_error_n'])}, BAcc {pct(v180r['balanced_accuracy'])}, review "
            f"{pct(v180r['remaining_review_or_reject_rate'])}."
        ),
        "",
        "## Paper Boundary",
        "",
        "Do not promote v178 as the recommended operating point. Use it to show that we actively tested image-agreement release and rejected it after nested audit. This strengthens the rigor of the final framework.",
    ]
    (OUT_DIR / "v181_current_system_after_nested_audit.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "v118_bacc": float(v118["balanced_accuracy"]),
        "v118_review_rate": float(v118["remaining_review_or_reject_rate"]),
        "v161_bacc": float(v161["balanced_accuracy"]),
        "v161_review_rate": float(v161["remaining_review_or_reject_rate"]),
        "v178_full_fit_review_rate": float(v178["remaining_review_or_reject_rate"]),
        "v180_nested_bacc": float(v180r["balanced_accuracy"]),
        "v180_nested_review_rate": float(v180r["remaining_review_or_reject_rate"]),
        "v180_released_errors": int(v180r["auto_action_error_n"]),
    }
    (OUT_DIR / "v181_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v181] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
