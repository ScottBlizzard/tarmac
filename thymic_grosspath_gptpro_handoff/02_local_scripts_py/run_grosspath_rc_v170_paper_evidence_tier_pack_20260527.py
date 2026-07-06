from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v170_paper_evidence_tier_pack_20260527"
V165_MAIN = ROOT / "outputs" / "grosspath_rc_v165_safe_release_evidence_pack_20260527" / "v165_main_operating_results.csv"
V165_CLAIMS = ROOT / "outputs" / "grosspath_rc_v165_safe_release_evidence_pack_20260527" / "v165_claim_evidence_map.csv"
V169_GRADE = ROOT / "outputs" / "grosspath_rc_v169_safe_release_evidence_grade_update_20260527" / "v169_safe_release_evidence_grade.csv"
V157_MODULE = ROOT / "outputs" / "grosspath_rc_v157_framework_claim_status_after_v156_20260527" / "v157_framework_module_status.csv"
V156_SUMMARY = ROOT / "outputs" / "grosspath_rc_v156_severe_shift_gate_safety_boundary_20260527" / "v156_focus_safety_table.csv"
V158_GATE = ROOT / "outputs" / "grosspath_rc_v158_unlabeled_gate_threshold_stability_20260527" / "v158_gate_component_stability.csv"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):+.{digits}f} pp"


def main_operating_table() -> pd.DataFrame:
    main = pd.read_csv(V165_MAIN)
    keep = [
        "30% low-confidence selective review baseline",
        "30% image-direction router",
        "v118 high-safety two-signal scorecard",
        "v161 safe-release high-safety scorecard",
        "severe-shift gated concept auto-correction",
    ]
    out = main.loc[
        main["evidence_line"].isin(keep)
        & main["eval_domain"].isin(["all_domains", "strict_external"])
    ].copy()
    out["tier"] = out["evidence_line"].map(
        {
            "30% low-confidence selective review baseline": "Comparator",
            "30% image-direction router": "Tier 2 candidate",
            "v118 high-safety two-signal scorecard": "Tier 1 main/high-safety baseline",
            "v161 safe-release high-safety scorecard": "Tier 2 efficiency candidate",
            "severe-shift gated concept auto-correction": "Tier 2 severe-shift correction candidate",
        }
    )
    out["paper_position"] = out["evidence_line"].map(
        {
            "30% low-confidence selective review baseline": "confidence-only selective review comparator",
            "30% image-direction router": "direction-aware router candidate; trend-level external gain",
            "v118 high-safety two-signal scorecard": "most defensible high-safety operating point",
            "v161 safe-release high-safety scorecard": "current best efficiency candidate, not nested-locked",
            "severe-shift gated concept auto-correction": "mechanistic correction candidate under severe shift",
        }
    )
    cols = [
        "tier",
        "evidence_line",
        "paper_position",
        "eval_domain",
        "n",
        "auto_pass_rate",
        "review_or_reject_rate",
        "auto_correct_rate",
        "balanced_accuracy",
        "accuracy",
        "f1",
        "fn",
        "fp",
        "status",
        "source",
    ]
    return out[cols].sort_values(["tier", "evidence_line", "eval_domain"])


def module_tier_table() -> pd.DataFrame:
    modules = pd.read_csv(V157_MODULE)
    rows = []
    for _, r in modules.iterrows():
        module = str(r["module"])
        if module.startswith("Primary"):
            tier = "Tier 1"
            grade = "main framework backbone"
            recommended = "Use as the base diagnostic engine and report safety/efficiency operating points."
        elif module == "Reject / clinical review fallback":
            tier = "Tier 1"
            grade = "required safety layer"
            recommended = "Report review/reject burden together with accuracy."
        elif module == "Direction-aware router":
            tier = "Tier 2"
            grade = "candidate; point-estimate gain"
            recommended = "Report as an improvement direction over confidence-only review, with CI caveat."
        elif module.startswith("Image-distilled"):
            tier = "Tier 2"
            grade = "severe-shift candidate"
            recommended = "Use as mechanistic evidence for concept-guided correction, not global auto-flip."
        elif module.startswith("Unlabeled shift"):
            tier = "Tier 2"
            grade = "safety gate candidate"
            recommended = "Write as necessary condition for auto-correction; needs prospective threshold calibration."
        else:
            tier = "Tier 3"
            grade = str(r["current_status"])
            recommended = str(r["next_action"])
        rows.append(
            {
                "tier": tier,
                "module": module,
                "evidence": r["evidence"],
                "evidence_grade": grade,
                "recommended_paper_use": recommended,
                "limitation": r["main_limitation"],
            }
        )
    rows.append(
        {
            "tier": "Tier 2",
            "module": "Multi-domain constrained safe-release",
            "evidence": "v161/v162/v164: review/reject 57.51%, BAcc 99.81%, FN=1, FP=0; v167/v168 nested re-selection releases errors.",
            "evidence_grade": "current-data efficiency candidate; not nested-locked",
            "recommended_paper_use": "Use as the main efficiency candidate with explicit prospective-validation caveat.",
            "limitation": "Fold-wise threshold re-selection is unstable; the fixed old+third rule needs prospective validation.",
        }
    )
    return pd.DataFrame(rows)


def claim_tier_map() -> pd.DataFrame:
    grade = pd.read_csv(V169_GRADE)
    gate = pd.read_csv(V158_GATE)
    clean_gate = int(gate["current_safety_status"].eq("clean_current_split").sum())
    gate_n = int(len(gate))
    gated = pd.read_csv(V156_SUMMARY)
    severe = gated.loc[
        gated["scenario"].eq("current_unlabeled_gate_only_severe_shift")
        & gated["eval_domain"].eq("strict_external")
    ].iloc[0]
    no_gate = gated.loc[
        gated["scenario"].eq("no_gate_apply_all_domains")
        & gated["eval_domain"].eq("all_three_domains")
    ].iloc[0]
    fixed = grade.loc[grade["evidence_item"].eq("Fixed old+third safe-release candidate")].iloc[0]
    nested = grade.loc[grade["evidence_item"].eq("Nested fold-wise max-release selection")].iloc[0]
    rows = [
        {
            "claim_tier": "Tier 1 supported",
            "claim": "The project is a risk-controlled cross-domain workflow rather than a single classifier.",
            "evidence": "v159/v165 separate classifier, router, correction, safe-release, and review/reject operating points.",
            "safe_wording": "We propose a risk-controlled workflow for gross pathology image diagnosis.",
            "do_not_claim": "Do not imply every module is fully locked or prospectively validated.",
        },
        {
            "claim_tier": "Tier 1 supported",
            "claim": "The two-signal scorecard provides the most defensible high-safety baseline.",
            "evidence": "v118/v159/v165: all-domain BAcc 99.81%, FN=1, FP=0; strict external BAcc 100%.",
            "safe_wording": "The high-safety scorecard is the most conservative operating point.",
            "do_not_claim": "Do not hide that review/reject burden is high before safe-release.",
        },
        {
            "claim_tier": "Tier 2 candidate",
            "claim": "Safe-release substantially improves efficiency while preserving current-data safety.",
            "evidence": f"v161/v169: review/reject {pct(fixed['review_rate'])}, BAcc {pct(fixed['balanced_accuracy'])}, FN={int(fixed['fn'])}, FP={int(fixed['fp'])}; nested re-selection leaves {int(nested['released_error_n'])} released errors.",
            "safe_wording": "Safe-release is a strong current-data efficiency candidate requiring prospective validation.",
            "do_not_claim": "Do not call it nested-validated or deployment-locked.",
        },
        {
            "claim_tier": "Tier 2 candidate",
            "claim": "Direction-aware routing can improve review selection under strict external shift.",
            "evidence": "v148/v159: strict external image-direction router BAcc 80.59% vs low-confidence 78.10% at ~30% review.",
            "safe_wording": "Direction-aware routing improves the point estimate over confidence-only review.",
            "do_not_claim": "Do not claim statistically proven superiority; prior CI crossed zero.",
        },
        {
            "claim_tier": "Tier 2 candidate",
            "claim": "Concept-guided auto-correction is useful only under severe-shift gating.",
            "evidence": f"v156: gated strict external delta {pp(severe['delta_bacc'])}; no-gate all-domain delta {pp(no_gate['delta_bacc'])}.",
            "safe_wording": "Concept-direction correction is a severe-shift candidate that requires a reliable gate.",
            "do_not_claim": "Do not use it as a global auto-flip rule.",
        },
        {
            "claim_tier": "Tier 2 candidate",
            "claim": "Unlabeled shift gate is a necessary safety precondition.",
            "evidence": f"v158: {clean_gate}/{gate_n} no-label gate variants cleanly trigger strict external in current data.",
            "safe_wording": "Unlabeled batch shift audit can support conditional workflow switching.",
            "do_not_claim": "Do not claim universal threshold stability without additional external batches.",
        },
    ]
    return pd.DataFrame(rows)


def writing_guide() -> str:
    return "\n".join(
        [
            "# v170 Paper Evidence Tier Pack",
            "",
            "## Recommended Results Story",
            "",
            "1. Start with the problem: ordinary confidence-based selective prediction is not enough under cross-domain acquisition shift.",
            "2. Present the framework: primary classifier, direction-aware router, severe-shift gate, concept-direction correction, safe-release, and review/reject fallback.",
            "3. Use v118 as the conservative high-safety baseline, because it has the cleanest safety posture.",
            "4. Present v161 safe-release as the current efficiency candidate, explicitly bounded by v167/v168 nested results.",
            "5. Present direction-aware routing and concept-guided auto-correction as mechanistic candidate modules, not final deployment rules.",
            "6. Close with residual boundary: the remaining automatic error and the need for prospective validation.",
            "",
            "## One-paragraph Safe Wording",
            "",
            "We built a risk-controlled diagnostic workflow for Task7 binary gross pathology classification. The workflow separates automatic classification, direction-aware risk routing, severe-shift detection, candidate automatic correction, safe-release, and review/reject fallback. The conservative two-signal scorecard provides the most defensible high-safety baseline, while the multi-domain constrained safe-release module substantially reduces review burden on the current data. Because fold-wise re-selection of release thresholds can still release errors, safe-release is reported as a strong efficiency candidate requiring prospective validation rather than a locked deployment threshold.",
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_table = main_operating_table()
    modules = module_tier_table()
    claims = claim_tier_map()
    carryover = pd.read_csv(V165_CLAIMS)

    main_table.to_csv(OUT_DIR / "v170_main_operating_table_tiered.csv", index=False, encoding="utf-8-sig")
    modules.to_csv(OUT_DIR / "v170_module_tier_table.csv", index=False, encoding="utf-8-sig")
    claims.to_csv(OUT_DIR / "v170_claim_tier_map.csv", index=False, encoding="utf-8-sig")
    carryover.to_csv(OUT_DIR / "v170_v165_claim_carryover.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "v170_recommended_writing_guide.md").write_text(writing_guide(), encoding="utf-8")

    v118 = main_table.loc[
        main_table["evidence_line"].eq("v118 high-safety two-signal scorecard")
        & main_table["eval_domain"].eq("all_domains")
    ].iloc[0]
    v161 = main_table.loc[
        main_table["evidence_line"].eq("v161 safe-release high-safety scorecard")
        & main_table["eval_domain"].eq("all_domains")
    ].iloc[0]
    report = {
        "main_tier_rows": int(len(main_table)),
        "module_rows": int(len(modules)),
        "claim_rows": int(len(claims)),
        "tier1_high_safety_bacc": float(v118["balanced_accuracy"]),
        "tier1_high_safety_review_rate": float(v118["review_or_reject_rate"]),
        "safe_release_candidate_bacc": float(v161["balanced_accuracy"]),
        "safe_release_candidate_review_rate": float(v161["review_or_reject_rate"]),
        "safe_release_candidate_status": str(v161["tier"]),
    }
    (OUT_DIR / "v170_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v170] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
