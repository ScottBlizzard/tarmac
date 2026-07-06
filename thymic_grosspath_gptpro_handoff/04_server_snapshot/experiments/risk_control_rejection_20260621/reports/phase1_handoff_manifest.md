# Phase 1 Handoff Manifest

## Result

- all checks passed: True
- checks: 82

## Required Artifacts And Status Checks

| Type | Path | Passed | Detail |
| --- | --- | --- | --- |
| artifact_exists | `plans/phase1_implementation_plan.md` | True | present |
| artifact_exists | `configs/final_policy_contract.json` | True | present |
| artifact_exists | `reports/phase1_candidate_validation_report.md` | True | present |
| artifact_exists | `reports/phase1_engineering_interface_report.md` | True | present |
| artifact_exists | `reports/final_safety_claims_audit.md` | True | present |
| artifact_exists | `reports/deployment_readiness_matrix.md` | True | present |
| artifact_exists | `reports/deployment_interface_guard.md` | True | present |
| artifact_exists | `reports/runtime_feature_template_audit.md` | True | present |
| artifact_exists | `reports/runtime_template_wrapper_dry_run.md` | True | present |
| artifact_exists | `reports/synthetic_deployment_scenario_audit.md` | True | present |
| artifact_exists | `reports/integration_smoke_audit.md` | True | present |
| artifact_exists | `reports/wrapper_decision_schema_audit.md` | True | present |
| artifact_exists | `reports/new_data_preflight_runner.md` | True | present |
| artifact_exists | `reports/new_data_preflight_failure_audit.md` | True | present |
| artifact_exists | `reports/existing_external_generalization_preflight_audit.md` | True | present |
| artifact_exists | `reports/existing_external_casebook_audit.md` | True | present |
| artifact_exists | `reports/strict_external_override_candidate_audit.md` | True | present |
| artifact_exists | `reports/known_domain_intersection_override_audit.md` | True | present |
| artifact_exists | `reports/integration_handoff_guide.md` | True | present |
| artifact_exists | `reports/phase1_reproducibility_runner.md` | True | present |
| artifact_exists | `integration/risk_control_gate_entrypoint.py` | True | present |
| artifact_exists | `integration/risk_control_deployable_wrapper.py` | True | present |
| artifact_exists | `templates/risk_control_v1_runtime_feature_template.csv` | True | present |
| artifact_exists | `scripts/runtime_feature_template_audit.py` | True | present |
| artifact_exists | `scripts/runtime_template_wrapper_dry_run.py` | True | present |
| artifact_exists | `scripts/synthetic_deployment_scenario_audit.py` | True | present |
| artifact_exists | `scripts/wrapper_decision_schema_audit.py` | True | present |
| artifact_exists | `scripts/new_data_preflight_runner.py` | True | present |
| artifact_exists | `scripts/new_data_preflight_failure_audit.py` | True | present |
| artifact_exists | `scripts/existing_external_generalization_preflight_audit.py` | True | present |
| artifact_exists | `scripts/existing_external_casebook_audit.py` | True | present |
| artifact_exists | `scripts/strict_external_override_candidate_audit.py` | True | present |
| artifact_exists | `scripts/known_domain_intersection_override_audit.py` | True | present |
| artifact_exists | `outputs/final_policy_runner_summary.csv` | True | present |
| artifact_exists | `outputs/runtime_decision_export.csv` | True | present |
| artifact_exists | `outputs/adapter_runtime_decision_export.csv` | True | present |
| artifact_exists | `outputs/integration_deployable_decisions.csv` | True | present |
| artifact_exists | `outputs/wrapper_deployable_decisions.csv` | True | present |
| artifact_exists | `outputs/wrapper_deployable_report.json` | True | present |
| artifact_exists | `outputs/deployment_readiness_matrix.csv` | True | present |
| artifact_exists | `outputs/final_safety_claims.csv` | True | present |
| artifact_exists | `outputs/deployment_interface_guard.csv` | True | present |
| artifact_exists | `outputs/runtime_feature_template_audit_report.json` | True | present |
| artifact_exists | `outputs/runtime_template_wrapper_dry_run_decisions.csv` | True | present |
| artifact_exists | `outputs/runtime_template_wrapper_dry_run_report.json` | True | present |
| artifact_exists | `outputs/synthetic_deployment_scenario_decisions.csv` | True | present |
| artifact_exists | `outputs/synthetic_deployment_scenario_report.json` | True | present |
| artifact_exists | `outputs/integration_smoke_audit_report.json` | True | present |
| artifact_exists | `outputs/wrapper_decision_schema_audit.csv` | True | present |
| artifact_exists | `outputs/wrapper_decision_schema_audit_report.json` | True | present |
| artifact_exists | `outputs/new_data_preflight_decisions.csv` | True | present |
| artifact_exists | `outputs/new_data_preflight_report.json` | True | present |
| artifact_exists | `outputs/new_data_preflight_failure_audit_report.json` | True | present |
| artifact_exists | `outputs/existing_external_preflight_decisions.csv` | True | present |
| artifact_exists | `outputs/existing_external_preflight_summary.csv` | True | present |
| artifact_exists | `outputs/existing_external_generalization_preflight_report.json` | True | present |
| artifact_exists | `outputs/existing_external_casebook.csv` | True | present |
| artifact_exists | `outputs/existing_external_casebook_summary.csv` | True | present |
| artifact_exists | `outputs/existing_external_casebook_report.json` | True | present |
| artifact_exists | `outputs/strict_external_override_candidate_cases.csv` | True | present |
| artifact_exists | `outputs/strict_external_override_candidate_summary.csv` | True | present |
| artifact_exists | `outputs/strict_external_override_candidate_report.json` | True | present |
| artifact_exists | `outputs/known_domain_intersection_override_cases.csv` | True | present |
| artifact_exists | `outputs/known_domain_intersection_override_summary.csv` | True | present |
| artifact_exists | `outputs/known_domain_intersection_override_report.json` | True | present |
| artifact_exists | `outputs/phase1_reproducibility_runner_report.json` | True | present |
| status_value | `outputs/phase1_reproducibility_runner_report.json` | True | all_passed=True, expected=True |
| status_value | `outputs/final_safety_claims_report.json` | True | all_claims_passed=True, expected=True |
| status_value | `outputs/deployment_interface_guard_report.json` | True | all_passed=True, expected=True |
| status_value | `outputs/runtime_feature_template_audit_report.json` | True | passed=True, expected=True |
| status_value | `outputs/runtime_template_wrapper_dry_run_report.json` | True | passed=True, expected=True |
| status_value | `outputs/synthetic_deployment_scenario_report.json` | True | passed=True, expected=True |
| status_value | `outputs/integration_smoke_audit_report.json` | True | passed=True, expected=True |
| status_value | `outputs/wrapper_decision_schema_audit_report.json` | True | all_passed=True, expected=True |
| status_value | `outputs/new_data_preflight_report.json` | True | passed=True, expected=True |
| status_value | `outputs/new_data_preflight_failure_audit_report.json` | True | passed=True, expected=True |
| status_value | `outputs/existing_external_generalization_preflight_report.json` | True | passed=True, expected=True |
| status_value | `outputs/existing_external_casebook_report.json` | True | passed=True, expected=True |
| status_value | `outputs/strict_external_override_candidate_report.json` | True | passed=True, expected=True |
| status_value | `outputs/known_domain_intersection_override_report.json` | True | passed=True, expected=True |
| status_value | `outputs/wrapper_deployable_report.json` | True | contract_passed=True, expected=True |
| status_value | `outputs/candidate_lock_audit_report.json` | True | passed=True, expected=True |