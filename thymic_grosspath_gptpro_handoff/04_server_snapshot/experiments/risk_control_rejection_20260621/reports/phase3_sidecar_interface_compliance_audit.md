# Phase 3 Sidecar Interface Compliance Audit

## Result

- passed: True
- interface_version: `sidecar_external_interface_v1`
- runtime rows: 699
- decision rows: 699
- row_count_matches: True
- runtime_required_column_n: 11
- decision_output_column_n: 11
- forbidden_selector_column_n: 8
- audit_only_column_n: 3
- decision_order_matches: True
- main_project_write_required: False
- strict_external_relaxation_allowed: False
- original_project_code_modified: False

## Drift Checks

- runtime_missing_required_columns: []
- runtime_forbidden_columns_present: []
- runtime_audit_only_columns_present: []
- decision_missing_output_columns: []
- decision_extra_output_columns: []
- decision_forbidden_columns_present: []

## Interpretation

This audit checks that the current sidecar runtime feature export and decision output still
conform to the frozen external interface. It is a drift guard only; it does not authorize
original-project writes or any strict_external relaxation.