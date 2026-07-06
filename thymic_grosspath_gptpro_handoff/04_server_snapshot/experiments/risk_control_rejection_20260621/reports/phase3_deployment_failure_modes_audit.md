# Phase 3 Deployment Failure Modes Audit

## Result

- passed: True
- scenario_n: 6
- blocked_scenario_n: 6
- failed_scenario_n: 0
- unsafe_auto_allowed: False
- main_project_write_required: False
- strict_external_relaxation_allowed: False
- strict_external_labels_used_for_selection: False
- original_project_code_modified: False

## Failure Cases

| Scenario | Expected Blocked | Blocked | Passed | Reason |
| --- | --- | --- | --- | --- |
| missing_runtime_required_column | True | True | True | missing_required|interface_missing_runtime_required |
| bad_policy_feature_version | True | True | True | unexpected_feature_version |
| forbidden_runtime_selector_column | True | True | True | forbidden_runtime_selector|interface_forbidden_runtime_selector |
| decision_output_order_drift | True | True | True | decision_output_order_drift |
| runtime_decision_row_count_mismatch | True | True | True | runtime_decision_row_count_mismatch |
| relaxed_artifact_in_package_manifest | True | True | True | relaxed_or_what_if_artifact_in_package |

## Interpretation

This audit intentionally corrupts the sidecar runtime interface, decision interface, and staging
manifest. Passing means these corruptions are detected as blocked conditions. It does not
authorize original-project writes, strict_external threshold relaxation, or label-dependent
runtime behavior.