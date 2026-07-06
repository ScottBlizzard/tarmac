# v195+ Sidecar Stability and Interface Audit

This audit checks that the isolated v195+ sidecar has a stable, label-free runtime interface and reproduces the frozen Phase2 metrics.

## Result

- Passed: `True`.
- Checks: `28`.
- Failed: `0`.

## Failed Checks

_empty_

## File Inventory

| path | exists | size |
| --- | --- | --- |
| experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_runtime_decisions.csv | True | 154126 |
| experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_audit_decisions.csv | True | 182979 |
| experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_summary.csv | True | 380 |
| experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_sidecar_report.json | True | 3760 |

## All Checks

| check_id | passed | detail |
| --- | --- | --- |
| file_exists::v195_plus_runtime_decisions.csv | True | {'path': 'experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_runtime_decisions.csv', 'exists': True, 'size': 154126} |
| file_exists::v195_plus_audit_decisions.csv | True | {'path': 'experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_audit_decisions.csv', 'exists': True, 'size': 182979} |
| file_exists::v195_plus_summary.csv | True | {'path': 'experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_summary.csv', 'exists': True, 'size': 380} |
| file_exists::v195_plus_sidecar_report.json | True | {'path': 'experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_sidecar_report.json', 'exists': True, 'size': 3760} |
| manifest_exists | True | experiments/risk_control_rejection_20260621/configs/v195_plus_candidate_manifest.json |
| manifest_phase2_candidate_n_7 | True | candidate_n=7 |
| manifest_strict_external_not_selection | True | False |
| runtime_exact_column_contract | True | ['domain', 'case_id', 'original_case_id', 'final_pred', 'prob_mean_core', 'v185_auto_decision', 'v185_review_or_reject', 'v195_release_from_review', 'v195_auto_decision', 'v195_review_or_reject', 'v195_plus_release_from_review', 'v195_plus_new_release_vs_v195', 'v195_plus_auto_decision', 'v195_plus_review_or_reject', 'v195_plus_release_reason', 'v195_plus_releasing_candidate_n', 'v195_plus_releasing_candidate_ids', 'v195_plus_policy_id'] |
| runtime_no_audit_or_label_columns | True | [] |
| runtime_row_count_699 | True | runtime_rows=699 |
| audit_row_count_699 | True | audit_rows=699 |
| audit_contains_audit_columns | True | ['task_l6_label', 'label_idx', 'v195_auto_error', 'v195_auto_high_risk_fn', 'v195_plus_auto_error', 'v195_plus_auto_high_risk_fn', 'v195_plus_new_release_error', 'v195_plus_new_release_high_risk_fn'] |
| runtime_case_id_unique | True | case_id duplicates absent |
| v195_strict_auto_n | True | observed=52, expected=52 |
| v195_strict_review_n | True | observed=56, expected=56 |
| v195_strict_auto_error_n | True | observed=0, expected=0 |
| v195_strict_auto_high_risk_fn_n | True | observed=0, expected=0 |
| v195_plus_strict_auto_n | True | observed=57, expected=57 |
| v195_plus_strict_review_n | True | observed=51, expected=51 |
| v195_plus_strict_auto_error_n | True | observed=0, expected=0 |
| v195_plus_strict_auto_high_risk_fn_n | True | observed=0, expected=0 |
| builder_reports_pass | True | a=True, b=True |
| builder_runtime_idempotent | True | two in-memory builds match |
| builder_audit_idempotent | True | two in-memory builds match |
| runtime_disk_matches_builder | True | disk output matches builder |
| audit_disk_matches_builder | True | disk output matches builder |
| summary_disk_matches_builder | True | summary output matches builder |
| sidecar_boundary_flags | True | sidecar_only=True, original_project_code_modified=False |