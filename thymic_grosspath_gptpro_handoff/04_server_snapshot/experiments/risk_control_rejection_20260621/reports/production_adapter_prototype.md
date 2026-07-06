# Production Adapter Prototype

## Purpose

This adapter maps current project outputs into the stable `risk_control_v1` policy feature interface.
It is isolated under the experiment directory and does not modify production code.

## Domain Counts

- old_data: n=285, missing dino_fn_risk=142
- strict_external: n=108, missing dino_fn_risk=47
- third_batch: n=306, missing dino_fn_risk=77

## Review Pool

- v195 review/reject cases available for release rule: 252

## Sources

- `outputs/grosspath_rc_v201_stable_supported_domain_flip_20260527/v201_stable_supported_domain_flip_cases.csv`
- `outputs/grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527/v195_selected_candidate_action_cases.csv`
- `outputs/grosspath_rc_v77_batch_shift_audit_policy_switch_20260527/v77_unlabeled_batch_shift_audit.csv`
- `outputs/grosspath_rc_v191_dino_fn_risk_sentinel_20260527/v191_fn_risk_case_outputs.csv`