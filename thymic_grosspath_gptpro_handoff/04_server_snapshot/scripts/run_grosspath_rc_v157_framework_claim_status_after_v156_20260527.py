from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v157_framework_claim_status_after_v156_20260527"
V148 = ROOT / "outputs" / "grosspath_rc_v148_router_card_comparative_evidence_20260527" / "v148_policy_vs_low_conf_paired_comparison.csv"
V152_MODULE = ROOT / "outputs" / "grosspath_rc_v152_framework_evidence_pack_20260527" / "v152_module_evidence_table.csv"
V155 = ROOT / "outputs" / "grosspath_rc_v155_autocorrect_evidence_update_20260527" / "v155_autocorrect_evidence_update.csv"
V156_REPORT = ROOT / "outputs" / "grosspath_rc_v156_severe_shift_gate_safety_boundary_20260527" / "v156_run_report.json"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):+.{digits}f} pp"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v148 = pd.read_csv(V148)
    v152 = pd.read_csv(V152_MODULE)
    v155 = pd.read_csv(V155)
    v156 = json.loads(V156_REPORT.read_text(encoding="utf-8"))

    strict_image = v148.loc[
        v148["domain"].eq("strict_external") & v148["policy"].eq("shift_aware_image_directional_candidate")
    ].iloc[0]
    strict_concept_review = v148.loc[
        v148["domain"].eq("strict_external") & v148["policy"].eq("shift_aware_concept_directional_candidate")
    ].iloc[0]
    concept_autocorrect = v155.loc[v155["line"].eq("pseudo-severe concept-direction auto-corrector")].iloc[0]

    module_rows = [
        {
            "stage": "Stage 1",
            "module": "Primary image classifier / selective auto-pass backbone",
            "evidence": "Previous v50/v79/v118-v119 family establishes the high-safety workflow and two-signal risk card on old+third+strict external.",
            "current_status": "main workflow backbone",
            "paper_claim": "Use as the base diagnostic engine; do not present later auto-correction experiments as replacing the primary classifier.",
            "main_limitation": "The remaining innovation must come from risk control and cross-domain handling, not from claiming a single stronger classifier.",
            "next_action": "Keep the backbone fixed when testing safety modules, unless a new model is validated by the same leave-domain and external protocol.",
        },
        {
            "stage": "Stage 2",
            "module": "Direction-aware router",
            "evidence": (
                f"v148 strict external image-direction router BAcc {pct(strict_image['policy_system_bacc'])} "
                f"vs low-conf {pct(strict_image['baseline_system_bacc'])}, Delta {pp(strict_image['delta_bacc'])}; "
                f"concept-direction review Delta {pp(strict_concept_review['delta_bacc'])}."
            ),
            "current_status": "candidate supported by point-estimate gain",
            "paper_claim": "Direction-aware routing is more informative than ordinary confidence for severe-shift triage, but the matched CI still crosses zero.",
            "main_limitation": "External sample size is small; significance is trend-level.",
            "next_action": "Use as a framework component and evaluate prospective external batches; avoid claiming statistically proven superiority yet.",
        },
        {
            "stage": "Stage 3",
            "module": "Image-distilled concept / pseudo-severe automatic correction",
            "evidence": (
                f"v155 concept-direction rule: {concept_autocorrect['strict_external_behavior']}; "
                f"ordinary-domain boundary: {concept_autocorrect['ordinary_domain_boundary']}"
            ),
            "current_status": "severe-shift-only candidate",
            "paper_claim": "Doctor-derived concepts are not useful as naive direct inputs, but they can supervise a direction-aware correction signal under severe shift.",
            "main_limitation": "Unsafe as a global auto-flip rule; needs a reliable severe-shift gate.",
            "next_action": "Develop a no-label severe-shift gate and a second severe-shift validation batch before locking automatic correction.",
        },
        {
            "stage": "Stage 4",
            "module": "Unlabeled shift gate for safe correction",
            "evidence": (
                f"v156 internal max shift index {v156['internal_max_shift_index']:.3f}, strict external "
                f"{v156['strict_external_shift_index']:.3f}; gated strict-external Delta BAcc "
                f"{pp(v156['strict_external_delta_bacc_when_gated'])}; false old/third gate "
                f"{pp(v156['old_false_gate_delta_bacc'])}/{pp(v156['third_false_gate_delta_bacc'])}."
            ),
            "current_status": "core safety boundary evidence",
            "paper_claim": "Automatic correction must be conditional on batch-level shift detection; otherwise it can harm ordinary domains.",
            "main_limitation": "Current separation is shown on existing batches; more external batches are required to calibrate a deployable threshold.",
            "next_action": "Treat uncertain gate cases as review/rejection, not auto-correction.",
        },
        {
            "stage": "Stage 5",
            "module": "Reject / clinical review fallback",
            "evidence": "v151/v153 show safe all-domain auto-flip collapses to zero or near-zero coverage; v156 shows false severe-shift triggers are harmful.",
            "current_status": "required safety layer",
            "paper_claim": "Rejection is not a failure mode; it is the safety mechanism that prevents unsafe correction when the model cannot verify the domain state.",
            "main_limitation": "Review burden must be reported with accuracy; high accuracy without review-rate accounting is incomplete.",
            "next_action": "Report auto-pass, auto-correct, and reject/review proportions together for every dataset.",
        },
    ]
    module_status = pd.DataFrame(module_rows)
    module_status.to_csv(OUT_DIR / "v157_framework_module_status.csv", index=False, encoding="utf-8-sig")

    claim_rows = [
        {
            "claim": "The framework is more than a single classifier.",
            "evidence": "It contains primary classifier, direction-aware router, concept-distilled correction, unlabeled shift gate, and rejection layer.",
            "status": "supported as framework design",
            "writing_boundary": "Must still quantify each module separately and avoid implying every module is fully locked.",
        },
        {
            "claim": "Concept information is useful.",
            "evidence": "v141 direct fusion is nearly negative, v142/v144/v154-v156 show concept value appears in routing/correction rather than direct classification.",
            "status": "supported with nuance",
            "writing_boundary": "Write as structured intermediate supervision / risk signal, not as simple multimodal fusion.",
        },
        {
            "claim": "Automatic correction can improve severe external shift.",
            "evidence": f"v156 gated severe-shift Delta BAcc {pp(v156['strict_external_delta_bacc_when_gated'])}.",
            "status": "candidate",
            "writing_boundary": "Cannot be claimed as deployable without severe-shift gate validation and prospective external validation.",
        },
        {
            "claim": "Unconditional auto-correction is unsafe.",
            "evidence": f"v156 no-gate all-domain Delta BAcc {pp(v156['all_domain_delta_bacc_without_gate'])}.",
            "status": "supported",
            "writing_boundary": "This is a strength of the risk-control framing: it explains why naive post-processing is rejected.",
        },
    ]
    claim_status = pd.DataFrame(claim_rows)
    claim_status.to_csv(OUT_DIR / "v157_claim_status.csv", index=False, encoding="utf-8-sig")

    carryover = v152[["module", "main_result", "status", "interpretation"]].copy()
    carryover.to_csv(OUT_DIR / "v157_previous_module_carryover.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v157 Framework Claim Status After v156",
        "",
        "## One-sentence Story",
        "",
        "The current project should be written as a risk-controlled, cross-domain gross pathology diagnosis framework, not as a single-model accuracy race.",
        "",
        "## Module Status",
        "",
        "| Stage | Module | Current status | Evidence | Writing boundary |",
        "|---|---|---|---|---|",
    ]
    for r in module_status.itertuples(index=False):
        md.append(f"| {r.stage} | {r.module} | {r.current_status} | {r.evidence} | {r.main_limitation} |")
    md += [
        "",
        "## Claim Boundaries",
        "",
        "| Claim | Status | Evidence | Boundary |",
        "|---|---|---|---|",
    ]
    for r in claim_status.itertuples(index=False):
        md.append(f"| {r.claim} | {r.status} | {r.evidence} | {r.writing_boundary} |")
    md += [
        "",
        "## Next Experimental Priority",
        "",
        "1. Calibrate the no-label severe-shift gate on old+third and any future non-test external-like data.",
        "2. Keep the strict external set as validation only; do not select thresholds using its labels.",
        "3. For every future result, report auto-pass rate, auto-correct rate, review/reject rate, Acc, BAcc, F1, FN, FP, rescued cases, and hurt cases.",
        "4. If another severe-shift development batch becomes available, validate whether the v154/v156 concept-direction correction remains beneficial before moving it from candidate to locked module.",
    ]
    (OUT_DIR / "v157_framework_claim_status_after_v156.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "stage_count": int(len(module_status)),
        "claim_count": int(len(claim_status)),
        "main_message": "risk-controlled cross-domain framework; auto-correction is severe-shift gated candidate, not global post-processing",
    }
    (OUT_DIR / "v157_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v157] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
