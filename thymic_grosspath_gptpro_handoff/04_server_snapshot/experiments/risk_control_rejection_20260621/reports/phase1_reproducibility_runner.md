# Phase 1 Reproducibility Runner

## Result

| Step | Return Code | Seconds | Command |
| ---: | ---: | ---: | --- |
| 1 | 0 | 2.67 | `/root/miniconda3/envs/thymic_baseline/bin/python -m unittest /workspace/thymic_project/experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py -v` |
| 2 | 0 | 1.36 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/stable_policy_feature_interface.py` |
| 3 | 0 | 1.37 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/production_adapter_prototype.py` |
| 4 | 0 | 1.43 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/final_policy_runner.py` |
| 5 | 0 | 1.35 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/final_policy_runner.py --features /workspace/thymic_project/experiments/risk_control_rejection_20260621/outputs/adapter_policy_features.csv --output-prefix adapter_final_policy_runner` |
| 6 | 0 | 0.57 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/final_policy_consistency_audit.py` |
| 7 | 0 | 0.53 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/policy_contract_validator.py` |
| 8 | 0 | 1.30 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/runtime_decision_export.py` |
| 9 | 0 | 1.33 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/runtime_decision_export.py --features /workspace/thymic_project/experiments/risk_control_rejection_20260621/outputs/adapter_policy_features.csv --output-prefix adapter_runtime_decision_export` |
| 10 | 0 | 1.46 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/input_order_invariance_audit.py` |
| 11 | 0 | 1.52 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/input_order_invariance_audit.py --features /workspace/thymic_project/experiments/risk_control_rejection_20260621/outputs/adapter_policy_features.csv --output-prefix adapter_input_order_invariance` |
| 12 | 0 | 1.40 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/dino_missing_policy_sensitivity.py` |
| 13 | 0 | 2.30 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/fraction_granularity_stress_audit.py` |
| 14 | 0 | 2.96 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/fold_fraction_granularity_stress_audit.py` |
| 15 | 0 | 3.02 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/fold_boundary_failure_casebook.py` |
| 16 | 0 | 0.58 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/fold_safe_frontier.py` |
| 17 | 0 | 0.45 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/candidate_lock_audit.py` |
| 18 | 0 | 1.32 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/locked_candidate_subgroup_stress_audit.py` |
| 19 | 0 | 1.25 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/locked_candidate_boundary_margin_audit.py` |
| 20 | 0 | 1.26 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/strict_external_mode_diff_audit.py` |
| 21 | 0 | 0.61 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/shift_gate_evidence_audit.py` |
| 22 | 0 | 0.49 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/deployment_readiness_matrix.py` |
| 23 | 0 | 0.52 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/final_safety_claims_audit.py` |
| 24 | 0 | 0.50 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/deployment_interface_guard.py` |
| 25 | 0 | 0.44 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/runtime_feature_template_audit.py` |
| 26 | 0 | 1.27 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/runtime_template_wrapper_dry_run.py` |
| 27 | 0 | 1.44 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/synthetic_deployment_scenario_audit.py` |
| 28 | 0 | 1.35 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/integration_smoke_audit.py` |
| 29 | 0 | 1.25 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/integration/risk_control_deployable_wrapper.py` |
| 30 | 0 | 0.50 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/wrapper_decision_schema_audit.py` |
| 31 | 0 | 1.26 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/new_data_preflight_runner.py` |
| 32 | 0 | 1.29 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/new_data_preflight_failure_audit.py` |
| 33 | 0 | 1.29 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/existing_external_generalization_preflight_audit.py` |
| 34 | 0 | 1.30 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/existing_external_casebook_audit.py` |
| 35 | 0 | 2.80 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/strict_external_override_candidate_audit.py` |
| 36 | 0 | 2.00 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/known_domain_intersection_override_audit.py` |
| 37 | 0 | 0.80 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase1_handoff_manifest.py` |