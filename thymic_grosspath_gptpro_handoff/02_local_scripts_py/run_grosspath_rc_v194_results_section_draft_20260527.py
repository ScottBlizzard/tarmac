from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v194_results_section_draft_20260527"
V192_TABLE = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527" / "v192_final_framework_operating_table.csv"
V192_CLAIMS = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527" / "v192_final_framework_claim_table.csv"
V187_CI = ROOT / "outputs" / "grosspath_rc_v187_release_safety_ci_planning_20260527" / "v187_auto_decision_error_ci.csv"
V187_PLAN = ROOT / "outputs" / "grosspath_rc_v187_release_safety_ci_planning_20260527" / "v187_prospective_auto_decision_sample_size.csv"
V189_ERRORS = ROOT / "outputs" / "grosspath_rc_v189_adaptive_error_review_anatomy_20260527" / "v189_adaptive_auto_residual_errors.csv"
V191_QUALITY = ROOT / "outputs" / "grosspath_rc_v191_dino_fn_risk_sentinel_20260527" / "v191_fn_risk_model_quality.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def get_row(table: pd.DataFrame, module: str, scope: str = "all_domains") -> pd.Series:
    return table.loc[table["module"].eq(module) & table["scope"].eq(scope)].iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(V192_TABLE)
    claims = pd.read_csv(V192_CLAIMS)
    ci = pd.read_csv(V187_CI)
    plan = pd.read_csv(V187_PLAN)
    residual = pd.read_csv(V189_ERRORS, dtype={"original_case_id": str})
    quality = pd.read_csv(V191_QUALITY)

    adaptive = get_row(table, "v185 unlabeled shift-adaptive workflow")
    fixed182 = get_row(table, "v182 fixed stable efficiency workflow")
    fixed118 = get_row(table, "v118 fixed high-safety fallback")
    greedy = get_row(table, "v180 greedy per-fold image-agreement audit")
    lowconf_ext = get_row(table, "30% low-confidence selective review", "strict_external")
    router_ext = get_row(table, "30% direction-aware image router", "strict_external")
    v190 = get_row(table, "v190 fold-wise probability FN sentinel")
    v191 = get_row(table, "v191 learned DINO FN-risk sentinel")
    v173 = get_row(table, "v173 aggressive image-only review corrector")
    v174 = get_row(table, "v174 disagreement flip")
    v175 = get_row(table, "v175 error-enriched flip-risk")
    adaptive_ci = ci.loc[ci["policy"].eq("v185_unlabeled_shift_adaptive") & ci["scope"].eq("all_domains")].iloc[0]
    strict_ci = ci.loc[ci["policy"].eq("v185_unlabeled_shift_adaptive") & ci["scope"].eq("strict_external")].iloc[0]
    case = residual.iloc[0]
    best_fn_model = quality.sort_values("internal_lowrisk_auc", ascending=False).iloc[0]
    n_zero_05 = int(plan.loc[plan["target_wilson95_upper"].eq(0.05), "auto_decision_n_needed_if_zero_errors"].iloc[0])
    n_one_05 = int(plan.loc[plan["target_wilson95_upper"].eq(0.05), "auto_decision_n_needed_if_one_error"].iloc[0])

    md = f"""# v194 Results Section Draft

## Mini-outline

- Main result: a risk-controlled, shift-adaptive selective diagnosis framework reaches high balanced accuracy while explicitly controlling automatic decisions.
- Deployment policy: unlabeled severe-shift auditing switches the system from the standard efficiency workflow to a high-safety fallback.
- Stable release: fixed image-agreement release improves review efficiency only when constrained by fold-wise stability; greedy release is rejected by audit.
- Negative evidence: automatic flipping and residual FN sentinels are not reliable enough to replace review under the current sample size.
- Safety boundary: automatic decision error rates require Wilson confidence intervals and prospective sample-size planning.

## Results Draft

**Paragraph role: opening / main result.**  
The final Task7 system should be interpreted as a risk-controlled selective diagnosis framework rather than a standalone binary classifier. Under the recommended v185 unlabeled shift-adaptive workflow, the system achieved an all-domain balanced accuracy of {pct(adaptive['balanced_accuracy'])} with a review/reject rate of {pct(adaptive['remaining_review_or_reject_rate'])}, leaving FN={int(adaptive['fn'])} and FP={int(adaptive['fp'])}. This operating point preserved the same all-domain balanced accuracy as both the fixed v182 efficiency workflow and the fixed v118 high-safety fallback, while avoiding the high review burden of the fixed high-safety mode ({pct(fixed118['remaining_review_or_reject_rate'])}).

**Paragraph role: deployment mechanism.**  
The unlabeled shift-adaptive policy provides the deployment-time risk-control mechanism. For within-internal-shift batches, the system uses the fixed v182 stable release workflow, which keeps all-domain BAcc at {pct(fixed182['balanced_accuracy'])} and reduces review/reject to {pct(fixed182['remaining_review_or_reject_rate'])}. When the severe-shift gate flags a batch as strict-external-like, the system falls back to v118 high-safety review; on the current strict external split this raises the review/reject rate from {pct(get_row(table, 'v182 fixed stable efficiency workflow', 'strict_external')['remaining_review_or_reject_rate'])} to {pct(get_row(table, 'v185 unlabeled shift-adaptive workflow', 'strict_external')['remaining_review_or_reject_rate'])}. This result supports the system-level claim that risk control is conditional on the unlabeled batch state, not only on per-case confidence.

**Paragraph role: ablation / stable release.**  
The safe-release module was only retained after stability auditing. The fixed v182 stable release rule kept BAcc at {pct(fixed182['balanced_accuracy'])} with zero released errors and a review/reject rate of {pct(fixed182['remaining_review_or_reject_rate'])}. In contrast, the greedy per-fold image-agreement audit lowered the review/reject rate to {pct(greedy['remaining_review_or_reject_rate'])}, but released 3 errors and reduced BAcc to {pct(greedy['balanced_accuracy'])}. This contrast is important because it shows that lower review burden alone is not accepted as improvement; release rules must satisfy stability constraints.

**Paragraph role: comparator evidence.**  
Direction-aware routing provided mechanism-level evidence beyond confidence-only selective prediction. At a comparable 30% review budget on the strict external split, the image-direction router reached BAcc {pct(router_ext['balanced_accuracy'])}, compared with {pct(lowconf_ext['balanced_accuracy'])} for the low-confidence selective-review baseline. The absolute performance of these 30% review policies was below the final safety-oriented workflow, but the comparison supports the claim that error direction and shift state contain useful information not captured by confidence alone.

**Paragraph role: negative correction evidence.**  
Automatic correction by direct flipping was not supported. The aggressive image-only corrector reduced review/reject to {pct(v173['remaining_review_or_reject_rate'])}, but retained automatic errors and was therefore treated as a safety-boundary candidate rather than a recommended workflow. The disagreement-flip policy produced severe harm (BAcc {pct(v174['balanced_accuracy'])}, FN={int(v174['fn'])}, FP={int(v174['fp'])}), and the error-enriched flip-risk strategy also failed to transfer to the current residual-error distribution (BAcc {pct(v175['balanced_accuracy'])}). These negative results justify keeping uncertain cases in review/rejection rather than forcing automatic correction.

**Paragraph role: residual error boundary.**  
The final residual automatic error was case {case['original_case_id']} ({case['task_l6_label']}), a high-risk-to-low-risk false negative. A simple fold-wise probability sentinel added {int(v190['additional_sentinel_review_n'])} reviews but rescued {int(v190['sentinel_rescued_fn_n'])} FN cases, and a learned DINO/probability FN-risk sentinel added {int(v191['additional_sentinel_review_n'])} reviews but also rescued {int(v191['sentinel_rescued_fn_n'])} FN cases. Although the best learned FN-risk model reached internal low-risk AUROC {float(best_fn_model['internal_lowrisk_auc']):.3f}, it did not stably trigger review for the residual FN under fold-wise evaluation. This supports treating the remaining FN as a residual risk boundary rather than post-hoc tuning a case-specific rule.

**Paragraph role: statistical safety boundary.**  
Automatic decision safety was reported with confidence intervals rather than only with point estimates. The v185 adaptive workflow made {int(adaptive_ci['auto_decision_n'])} all-domain automatic decisions with {int(adaptive_ci['auto_error_n'])} error, corresponding to an observed automatic error rate of {pct(adaptive_ci['auto_error_rate'])} and a Wilson95 upper bound of {pct(adaptive_ci['wilson95_high'])}. On strict external cases, the adaptive workflow made only {int(strict_ci['auto_decision_n'])} automatic decisions; even with zero observed error, the Wilson95 upper bound remained {pct(strict_ci['wilson95_high'])}. A prospective validation set would need {n_zero_05} zero-error automatic decisions to bound the Wilson95 upper limit below 5%, or {n_one_05} automatic decisions if one error is allowed.

## Self-review Checklist

- Clarity: The section states that the contribution is a risk-controlled workflow, not a raw classifier.
- Flow: Results move from main operating point to mechanism, ablation, comparator, negative evidence, residual boundary, and statistical boundary.
- Terminology: The terms v185 adaptive workflow, fixed v182 efficiency workflow, and fixed v118 fallback are used consistently.
- Unsupported claims: The draft does not claim mature automatic flipping or complete clinical safety.
- Missing evidence: Larger prospective external validation remains required, especially for strict external automatic-decision confidence intervals.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| The main contribution is a risk-controlled selective diagnosis framework. | v185 BAcc {pct(adaptive['balanced_accuracy'])}, review/reject {pct(adaptive['remaining_review_or_reject_rate'])}, FN={int(adaptive['fn'])}, FP={int(adaptive['fp'])}. | supported |
| Stable release improves efficiency without increasing current released errors. | fixed v182 review/reject {pct(fixed182['remaining_review_or_reject_rate'])}, released errors 0; greedy v180 releases 3 errors. | supported as current-split/stability-audited candidate |
| Direction-aware routing improves over confidence-only routing under strict external shift. | strict external BAcc {pct(router_ext['balanced_accuracy'])} vs {pct(lowconf_ext['balanced_accuracy'])} at comparable 30% review. | trend-level support |
| Automatic flipping is ready to replace review. | v174 and v175 both fail with substantial harm or poor transfer. | not supported |
| The final FN can be fixed by a simple non-leaky sentinel. | v190/v191 rescue 0 FN under nested evaluation. | not supported |
| Current automatic-decision safety needs confidence intervals. | v185 all-domain Wilson95 upper {pct(adaptive_ci['wilson95_high'])}; strict external upper {pct(strict_ci['wilson95_high'])}. | supported |
"""

    out = OUT_DIR / "v194_results_section_draft.md"
    out.write_text(md, encoding="utf-8")
    report = {
        "draft": str(out),
        "adaptive_bacc": float(adaptive["balanced_accuracy"]),
        "adaptive_review_rate": float(adaptive["remaining_review_or_reject_rate"]),
        "residual_case_id": str(case["original_case_id"]),
        "wilson95_upper_all": float(adaptive_ci["wilson95_high"]),
        "wilson95_upper_strict_external": float(strict_ci["wilson95_high"]),
    }
    (OUT_DIR / "v194_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v194] wrote {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
