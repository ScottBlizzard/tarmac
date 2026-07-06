# Phase 3 Staging Package Full Rehearsal

## Result

- passed: True
- package_local_wrapper_loaded: True
- contract_passed: True
- feature_rows: 699
- decision_rows: 699
- released_n: 29
- strict_external_released_n: 0
- forbidden_columns_present: -
- reference_summary_match: True
- main_project_write_required: False
- original_project_code_modified: False

## Interpretation

This rehearsal runs the staged package-local deployable wrapper on the full stable feature
table. It checks that the staged package reproduces the sidecar summary while preserving
the strict_external hard fallback and the no-main-project-write boundary.