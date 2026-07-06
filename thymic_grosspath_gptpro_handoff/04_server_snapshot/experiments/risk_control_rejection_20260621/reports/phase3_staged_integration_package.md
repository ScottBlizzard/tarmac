# Phase 3 Staged Integration Package

## Result

- passed: True
- package_dir: `/workspace/thymic_project/experiments/risk_control_rejection_20260621/staging/hard_gate_integration_package`
- files_copied: 26
- integration_allowed_file_n: 8
- stale_files_removed: 34

## Manifest

| Source | Class | Integration Allowed | Required | Copied | Forbidden | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| `integration/risk_control_gate_entrypoint.py` | deployable_code | True | True | True | False | copied |
| `integration/risk_control_deployable_wrapper.py` | deployable_code | True | True | True | False | copied |
| `configs/final_policy_contract.json` | runtime_contract | True | True | True | False | copied |
| `templates/risk_control_v1_runtime_feature_template.csv` | runtime_contract | True | True | True | False | copied |
| `reports/phase3_integration_readiness_audit.md` | review_evidence | False | True | True | False | copied |
| `outputs/phase3_allowed_integration_items.csv` | review_evidence | False | True | True | False | copied |
| `sidecar_original_project_adapter/minimal_export_contract.md` | export_contract | False | True | True | False | copied |
| `sidecar_original_project_adapter/minimal_export_contract_audit.py` | export_contract | False | True | True | False | copied |
| `sidecar_original_project_adapter/bridge_outputs/minimal_export_contract_audit.json` | export_contract | False | True | True | False | copied |
| `scripts/final_policy_runner.py` | package_support_code | True | True | True | False | copied |
| `scripts/policy_contract_validator.py` | package_support_code | True | True | True | False | copied |
| `scripts/audit_rejection_layer.py` | package_support_code | True | True | True | False | copied |
| `scripts/release_rejected_audit.py` | package_support_code | True | True | True | False | copied |
| `reports/phase2_final_experiment_report.md` | review_evidence | False | False | True | False | copied |
| `reports/phase2_hard_gate_deployable_frontier.md` | review_evidence | False | False | True | False | copied |
| `reports/phase3_sidecar_external_interface_freeze_audit.md` | review_evidence | False | False | True | False | copied |
| `reports/phase3_sidecar_interface_compliance_audit.md` | review_evidence | False | False | True | False | copied |
| `reports/phase3_deployment_failure_modes_audit.md` | review_evidence | False | False | True | False | copied |
| `outputs/phase3_integration_readiness_report.json` | review_evidence | False | False | True | False | copied |
| `outputs/phase2_hard_gate_deployable_frontier_report.json` | review_evidence | False | False | True | False | copied |
| `outputs/phase3_sidecar_external_interface_freeze_report.json` | review_evidence | False | False | True | False | copied |
| `outputs/phase3_sidecar_interface_compliance_report.json` | review_evidence | False | False | True | False | copied |
| `outputs/phase3_deployment_failure_modes_report.json` | review_evidence | False | False | True | False | copied |
| `outputs/phase3_deployment_failure_modes_cases.csv` | review_evidence | False | False | True | False | copied |
| `outputs/wrapper_deployable_report.json` | review_evidence | False | False | True | False | copied |
| `README.md` | package_documentation | False | True | True | False | generated |

## Interpretation

This package is a staging artifact only. It is intended to make a future original-project
integration review deterministic while preserving the current no-modification constraint.