# Phase 2 Reproducibility Runner

## Result

| Step | Return Code | Seconds | Command |
| ---: | ---: | ---: | --- |
| 1 | 0 | 1.88 | `/root/miniconda3/envs/thymic_baseline/bin/python -m unittest /workspace/thymic_project/experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py -v` |
| 2 | 0 | 0.43 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_shift_gate_calibration_audit.py` |
| 3 | 0 | 0.49 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_shift_gate_sensitivity_grid.py` |
| 4 | 0 | 1.12 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_relaxed_gate_policy_audit.py` |
| 5 | 0 | 1.62 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_relaxed_intersection_policy_audit.py` |
| 6 | 0 | 0.92 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_leave_one_known_gate_audit.py` |
| 7 | 0 | 1.04 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_leave_one_gate_sensitivity_grid.py` |
| 8 | 0 | 1.63 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_case_level_robust_core_grid.py` |
| 9 | 0 | 0.45 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_shift_metric_ablation_audit.py` |
| 10 | 0 | 3.22 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase2_hard_gate_deployable_frontier.py` |
| 11 | 0 | 0.48 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_integration_readiness_audit.py` |
| 12 | 0 | 1.25 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/sidecar_original_project_adapter/original_output_stable_feature_bridge.py` |
| 13 | 0 | 1.08 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/sidecar_original_project_adapter/minimal_export_contract_audit.py` |
| 14 | 0 | 4.83 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_build_staged_integration_package.py` |
| 15 | 0 | 1.98 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_staging_package_smoke_audit.py` |
| 16 | 0 | 0.36 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_main_project_integration_dry_run.py` |
| 17 | 0 | 1.24 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_staging_package_full_rehearsal.py` |
| 18 | 0 | 0.71 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_staging_package_artifact_integrity_audit.py` |
| 19 | 0 | 0.55 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_release_handoff_bundle_audit.py` |
| 20 | 0 | 1.14 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/sidecar_original_project_adapter/original_project_sidecar_adapter.py` |
| 21 | 0 | 0.44 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_sidecar_external_interface_freeze_audit.py` |
| 22 | 0 | 0.38 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_sidecar_interface_compliance_audit.py` |
| 23 | 0 | 0.40 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_deployment_failure_modes_audit.py` |
| 24 | 0 | 0.40 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_main_project_sidecar_patch_plan.py` |
| 25 | 0 | 5.14 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_main_project_gap_audit.py` |
| 26 | 0 | 0.40 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_export_hook_candidate_prioritization.py` |
| 27 | 0 | 0.41 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_export_hook_patch_blueprint.py` |
| 28 | 0 | 0.40 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_missing_runtime_hooks_blueprint.py` |
| 29 | 0 | 0.42 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_main_project_patch_rehearsal_package.py` |
| 30 | 0 | 0.39 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_patch_rehearsal_safety_audit.py` |
| 31 | 0 | 0.39 | `/root/miniconda3/envs/thymic_baseline/bin/python /workspace/thymic_project/experiments/risk_control_rejection_20260621/scripts/phase3_main_project_integration_decision_gate.py` |