# v195+ Pre-integration Gap Audit

This audit is read-only. It checks what would be needed before optionally connecting the v195+ sidecar to the original project.

## Result

- passed: `True`
- recommended_strategy: `keep_sidecar_only_until_optional_hooks_are_explicitly_approved`
- future_main_project_touch_n: `2`
- missing_optional_hook_n: `2`
- blocked_gap_n: `0`
- original_project_code_modified: `False`

## Gap Table

| Gap | Status | Candidate Paths | Next Action |
|---|---|---|---|
| `v185_base_output_available` | `ready` | scripts/run_grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527.py;scripts/run_grosspath_rc_v187_release_safety_ci_planning_20260527.py;scripts/run_grosspath_rc_v189_adaptive_error_review_anatomy_20260527.py;scripts/run_grosspath_rc_v190_fn_sentinel_nested_scan_20260527.py;scripts/run_grosspath_rc_v191_dino_fn_risk_sentinel_20260527.py;scripts/run_grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527.py;scripts/run_grosspath_rc_v199_directional_error_flip_corrector_20260527.py;scripts/run_grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527.py | reuse existing v185 output as sidecar base input |
| `v173_corrector_output_available` | `ready` | scripts/run_grosspath_rc_v173_image_only_review_corrector_20260527.py;scripts/run_grosspath_rc_v174_disagree_flip_release_policy_20260527.py;scripts/run_grosspath_rc_v178_safe_release_union_with_image_agreement_20260527.py;scripts/run_grosspath_rc_v180_nested_v178_image_agreement_release_20260527.py;scripts/run_grosspath_rc_v182_stable_fixed_image_agreement_release_20260527.py;scripts/run_grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527.py;scripts/run_grosspath_rc_v196_nested_adaptive_release_validation_20260527.py;scripts/run_grosspath_rc_v199_directional_error_flip_corrector_20260527.py | reuse existing v173 corrector output as sidecar corrector input |
| `v195_current_flow_identified` | `ready` | scripts/run_grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527.py;scripts/run_grosspath_rc_v196_nested_adaptive_release_validation_20260527.py | keep v195 as comparator and do not rewrite it for sidecar use |
| `v195_plus_manifest_frozen` | `ready` | experiments/risk_control_rejection_20260621/configs/v195_plus_candidate_manifest.json | treat manifest as frozen policy definition for sidecar |
| `sidecar_package_ready` | `ready` | experiments/risk_control_rejection_20260621/staging/v195_plus_sidecar_package | review staged sidecar package before any integration decision |
| `sidecar_invocation_hook` | `missing_optional_hook` | - | if approved later, add one optional invocation after v185/v173 outputs are materialized |
| `runtime_decision_consumer_hook` | `missing_optional_hook` | - | if approved later, define a downstream consumer for label-free runtime decisions |
| `audit_files_blocked_from_runtime` | `ready` | - | keep audit decisions and label/error fields out of any runtime path |

## Minimal Future Touches

If integration is explicitly approved later, the smallest main-project change should be:

1. Add an optional sidecar invocation after v185 base cases and v173 corrector outputs exist.
2. Consume only `v195_plus_runtime_decisions.csv`-style label-free columns.
3. Keep `v195_plus_audit_decisions.csv`, labels, and error fields out of runtime code.
4. Keep v195 unchanged as the comparator/fallback.

## Interpretation

The current recommendation remains sidecar-only. The required upstream outputs already exist,
but the original project has no dedicated v195+ invocation or downstream consumer hook. That is
acceptable because integration has not been approved.