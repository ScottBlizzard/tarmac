# Main Project Integration Dry Run

## Result

- passed: True
- recommended_strategy: `sidecar_stable_export_only`
- integration_allowed_file_n: 8
- read_only_file_n: 18
- forbidden_candidate_n: 0
- main_project_write_required: False
- strict_external_relaxation_allowed: False
- original_project_code_modified: False

## Dry-run Steps

| Step | Target | Action | Required | Uses Strict External Labels |
| --- | --- | --- | --- | --- |
| `add_stable_runtime_feature_export` | original_project_optional_export_boundary | export risk_control_v1 runtime columns only | True | False |
| `add_preflight_contract_validation` | sidecar_or_optional_main_project_boundary | validate exported features against final_policy_contract.json | True | False |
| `add_sidecar_invocation_or_wrapper_adapter` | sidecar boundary preferred | call deployable wrapper without importing frozen what-if artifacts | True | False |
| `keep_review_evidence_read_only` | reports and outputs | retain reports/result JSON as audit evidence only | True | False |

## Blocked Actions

- `do_not_integrate_relaxed_or_what_if_artifacts`
- `do_not_use_strict_external_labels_for_threshold_selection`
- `do_not_move_review_evidence_into_runtime_logic`
- `do_not_relax_strict_external_severe_shift_fallback_without_new_independent_validation`

## Interpretation

This is a dry-run audit only. It recommends preserving the sidecar/stable-export
boundary and does not authorize in-place original-project code changes.