# Phase 2 Final Experiment Report

## Scope

All Phase 2 work was kept under `experiments/risk_control_rejection_20260621/`. No original
project code was modified.

Phase 2 tested whether the strict_external severe-shift fallback could be safely relaxed using the
project's existing external-generalization data. strict_external labels were not used for threshold
or gate selection. They were used only after frozen decisions/candidates were produced for audit
error counts.

## Deployable Decision

Do not relax the strict_external batch gate.

The deployable policy should remain:

```text
strict_external hard_shift_gate = severe_unknown_shift
strict_external release_from_v195_review = 0
strict_external fallback = v195
```

The strongest deployable hard-gate frontier remains:

```text
granularity = domain
release_fraction = 0.15
old_data + third_batch released_n = 29
third_batch released_n = 18
strict_external released_n = 0
new released errors = 0
known-domain fold-safe = true
```

## Main Results

### Strict_external shift evidence

The default label-free severe gate remains supported:

```text
severe_multiplier = 1.5
min_exceeded_metrics = 3
strict_exceeded_metric_n = 4 / 4
recommendation = keep_severe_unknown_shift_fallback
```

Metric ablation showed the hard gate is not caused by one fragile metric:

```text
default_leave_one_out_all_severe = true
default_single_metric_severe_n = 4 / 4
batch_shift_index strict/known_max = 2.40x
domain_auc_cv strict/known_max = 1.87x
mean_outside_ref_05_95_rate strict/known_max = 4.45x
quality_proxy_mean strict/known_max = 13.95x
```

### Relaxed gate what-if

A 2.5x relaxed gate can produce attractive frozen strict_external audit numbers, but it is not
stable enough to deploy:

```text
relaxed gate = 2.5x, min_exceeded_metrics = 3
quota-15 strict_external released_n = 8
quota-15 strict_external released errors = 0
intersection strict_external candidate_n = 21
intersection strict_external candidate errors = 0
promote_to_deployable = false
```

Leave-one-known calibration rejected the 2.5x relaxation:

```text
heldout_known_stable_non_severe = true
strict_gate_stable_across_known_domains = false
strict_severe_train_domain_n = 1 / 2
```

Under the default min-exceeded-metrics setting, strict_external only becomes stably non-severe at a
5.0x multiplier, which is too loose to justify as a conservative medical deployment gate.

### Case-level tightening

Known-domain intersection thresholds produced a frozen strict_external candidate set:

```text
model_disagreement_rate <= 0.5
dino_fn_risk_max <= 0.3696929763812454
known candidates = 80, errors = 0
strict_external frozen candidates = 21, errors = 0
```

Shrinking those thresholds trades coverage for conservatism:

```text
shrink 1.00 -> strict candidates 21, errors 0
shrink 0.90 -> strict candidates 19, errors 0
shrink 0.75 -> strict candidates 12, errors 0
shrink 0.50 -> strict candidates 1, errors 0
shrink 0.25 -> strict candidates 0, errors 0
```

This does not solve the deployment problem because any strict_external release still requires
overriding the severe batch gate.

### Hard-gate deployable frontier

When the hard gate is kept unchanged and candidate selection requires fold-safe old_data +
third_batch evidence, no coverage improvement beyond the current deployable policy was found:

```text
selected granularity = domain
selected release_fraction = 0.15
third_batch released_n = 18
internal_known released_n = 29
strict_external released_n = 0
all-domain released errors = 0
known fold-safe = true
```

An aggregate-only scan briefly favored `domain / 0.20`, but that was rejected because fold-level
stress evidence made the higher fraction too optimistic.

## Verification

Fresh verification after staging package smoke audit completion:

```text
unit tests = 114 passed
phase2_reproducibility_runner steps = 31 / 31 passed
```

Primary reproducibility entry point:

```bash
python experiments/risk_control_rejection_20260621/scripts/phase2_reproducibility_runner.py
```

## Files Added In Phase 2

Scripts:

```text
scripts/phase2_shift_gate_calibration_audit.py
scripts/phase2_shift_gate_sensitivity_grid.py
scripts/phase2_relaxed_gate_policy_audit.py
scripts/phase2_relaxed_intersection_policy_audit.py
scripts/phase2_leave_one_known_gate_audit.py
scripts/phase2_leave_one_gate_sensitivity_grid.py
scripts/phase2_case_level_robust_core_grid.py
scripts/phase2_shift_metric_ablation_audit.py
scripts/phase2_hard_gate_deployable_frontier.py
scripts/phase3_integration_readiness_audit.py
scripts/phase3_build_staged_integration_package.py
scripts/phase3_staging_package_smoke_audit.py
scripts/phase3_main_project_integration_dry_run.py
scripts/phase3_staging_package_full_rehearsal.py
scripts/phase3_staging_package_artifact_integrity_audit.py
scripts/phase3_release_handoff_bundle_audit.py
scripts/phase3_sidecar_external_interface_freeze_audit.py
scripts/phase3_sidecar_interface_compliance_audit.py
scripts/phase3_deployment_failure_modes_audit.py
scripts/phase3_main_project_sidecar_patch_plan.py
scripts/phase3_main_project_gap_audit.py
scripts/phase3_export_hook_candidate_prioritization.py
scripts/phase3_export_hook_patch_blueprint.py
scripts/phase3_missing_runtime_hooks_blueprint.py
scripts/phase3_main_project_patch_rehearsal_package.py
scripts/phase3_patch_rehearsal_safety_audit.py
scripts/phase3_main_project_integration_decision_gate.py
scripts/phase2_reproducibility_runner.py
sidecar_original_project_adapter/original_project_sidecar_adapter.py
sidecar_original_project_adapter/original_output_stable_feature_bridge.py
sidecar_original_project_adapter/minimal_export_contract_audit.py
```

Key reports:

```text
reports/phase2_shift_gate_progress_report.md
reports/phase2_shift_gate_calibration_audit.md
reports/phase2_leave_one_gate_sensitivity_grid.md
reports/phase2_case_level_robust_core_grid.md
reports/phase2_shift_metric_ablation_audit.md
reports/phase2_hard_gate_deployable_frontier.md
reports/phase2_final_experiment_report.md
reports/phase3_integration_readiness_audit.md
reports/phase3_staged_integration_package.md
reports/phase3_staging_package_smoke_audit.md
reports/phase3_main_project_integration_dry_run.md
reports/phase3_staging_package_full_rehearsal.md
reports/phase3_staging_package_artifact_integrity_audit.md
reports/phase3_release_handoff_bundle_audit.md
reports/phase3_sidecar_external_interface_freeze_audit.md
reports/phase3_sidecar_interface_compliance_audit.md
reports/phase3_deployment_failure_modes_audit.md
reports/phase3_main_project_sidecar_patch_plan.md
reports/phase3_main_project_gap_audit.md
reports/phase3_export_hook_candidate_prioritization.md
reports/phase3_export_hook_patch_blueprint.md
reports/phase3_missing_runtime_hooks_blueprint.md
reports/phase3_main_project_patch_rehearsal_package.md
reports/phase3_patch_rehearsal_safety_audit.md
reports/phase3_main_project_integration_decision_gate.md
sidecar_original_project_adapter/README.md
sidecar_original_project_adapter/stable_feature_bridge_report.md
sidecar_original_project_adapter/minimal_export_contract.md
sidecar_original_project_adapter/original_output_compatibility_report.md
```

Phase 3 staging outputs:

```text
outputs/phase3_integration_readiness_manifest.csv
outputs/phase3_allowed_integration_items.csv
outputs/phase3_integration_readiness_report.json
outputs/phase3_staged_integration_package_manifest.csv
outputs/phase3_staged_integration_package_report.json
outputs/phase3_staging_package_smoke_report.json
outputs/phase3_main_project_integration_dry_run_steps.csv
outputs/phase3_main_project_integration_dry_run_report.json
outputs/phase3_staging_package_full_rehearsal_decisions.csv
outputs/phase3_staging_package_full_rehearsal_report.json
outputs/phase3_staging_package_artifact_integrity_inventory.csv
outputs/phase3_staging_package_artifact_integrity_report.json
outputs/phase3_release_handoff_bundle_index.csv
outputs/phase3_release_handoff_bundle_report.json
outputs/phase3_sidecar_external_interface_freeze_schema.csv
outputs/phase3_sidecar_external_interface_freeze_report.json
outputs/phase3_sidecar_interface_compliance_report.json
outputs/phase3_deployment_failure_modes_cases.csv
outputs/phase3_deployment_failure_modes_report.json
outputs/phase3_main_project_sidecar_patch_plan.csv
outputs/phase3_main_project_sidecar_patch_plan_report.json
outputs/phase3_main_project_gap_audit.csv
outputs/phase3_main_project_gap_audit_report.json
outputs/phase3_export_hook_candidate_prioritization.csv
outputs/phase3_export_hook_candidate_prioritization_report.json
outputs/phase3_export_hook_patch_blueprint.csv
outputs/phase3_export_hook_patch_blueprint_report.json
outputs/phase3_missing_runtime_hooks_blueprint.csv
outputs/phase3_missing_runtime_hooks_blueprint_report.json
outputs/phase3_main_project_patch_rehearsal_package.csv
outputs/phase3_main_project_patch_rehearsal_package_report.json
outputs/phase3_patch_rehearsal_safety_audit_issues.csv
outputs/phase3_patch_rehearsal_safety_audit_report.json
outputs/phase3_main_project_integration_decision_gate_issues.csv
outputs/phase3_main_project_integration_decision_gate_report.json
outputs/handoff_file_classification_manifest.csv
outputs/handoff_file_classification_manifest.json
staging/hard_gate_integration_package/
patch_rehearsal/
sidecar_original_project_adapter/outputs/
sidecar_original_project_adapter/outputs/sidecar_runtime_features.csv
sidecar_original_project_adapter/outputs/sidecar_input_contract_audit.json
sidecar_original_project_adapter/outputs/sidecar_risk_control_decisions.csv
sidecar_original_project_adapter/outputs/sidecar_risk_control_report.json
sidecar_original_project_adapter/bridge_outputs/sidecar_stable_features_from_original_outputs.csv
sidecar_original_project_adapter/bridge_outputs/sidecar_bridge_risk_control_decisions.csv
sidecar_original_project_adapter/bridge_outputs/sidecar_bridge_report.json
sidecar_original_project_adapter/bridge_outputs/sidecar_bridge_reference_parity.json
sidecar_original_project_adapter/bridge_outputs/minimal_export_contract_audit.json
staging/hard_gate_integration_package/sidecar_original_project_adapter/minimal_export_contract.md
staging/hard_gate_integration_package/sidecar_original_project_adapter/minimal_export_contract_audit.py
staging/hard_gate_integration_package/sidecar_original_project_adapter/bridge_outputs/minimal_export_contract_audit.json
staging/hard_gate_integration_package/scripts/final_policy_runner.py
staging/hard_gate_integration_package/scripts/policy_contract_validator.py
staging/hard_gate_integration_package/scripts/audit_rejection_layer.py
staging/hard_gate_integration_package/scripts/release_rejected_audit.py
staging/hard_gate_integration_package/outputs/handoff_file_classification_manifest.csv
staging/hard_gate_integration_package/outputs/handoff_file_classification_manifest.json
```

## Sidecar Adapter Result

The recommended original-project connection is now a sidecar, not an in-place code change.
It reads the stable runtime feature table, validates the contract, blocks relaxed/error columns,
and then calls the existing hard-gate deployable wrapper.

Current default sidecar run:

```text
input = outputs/adapter_policy_features.csv
decision_rows = 699
released_n = 29
strict_external_released_n = 0
review_or_reject_n = 223
contract_passed = true
forbidden_columns_present = []
original_project_code_modified = false
sidecar_only = true
```

This gives a practical integration path while preserving the scientific boundary: no original
project code is touched, and no relaxed strict_external what-if logic is promoted.

Historical raw-output compatibility was also checked:

```text
v195_selected_candidate_action_cases.csv:
  passed = false
  blocked = auto_correct_action, action_error, rescued_by_action, hurt_by_action

v185_unlabeled_shift_adaptive_cases.csv:
  passed = false
  mapped base aliases = true
  missing = model_disagreement_rate, dino_fn_risk_max, dino_fn_risk_missing
```

This means the sidecar can run on the stable feature export, but raw historical v185/v195 output
tables need a stable-feature export step before deployment use.

## Stable Feature Bridge Result

The stable-feature bridge now performs that export step inside the experiment sidecar directory. It
rebuilds deployable features from existing project artifacts:

```text
v201_stable_supported_domain_flip_cases.csv
v195_selected_candidate_action_cases.csv
v77_unlabeled_batch_shift_audit.csv
v191_fn_risk_case_outputs.csv
```

Bridge result:

```text
feature_rows = 699
decision_rows = 699
released_n = 29
strict_external_released_n = 0
contract_passed = true
reference_parity_passed = true
reference_row_count_delta = 0
reference_mismatched_columns = []
original_project_code_modified = false
sidecar_only = true
```

This closes the no-modification integration loop: the project can now rebuild the stable feature
table from existing outputs, validate it, and run the hard-gate sidecar without changing original
project code.

## Minimal Export Contract Result

The minimal export contract audit defines the smallest optional source surface needed if the
original project later chooses to feed the sidecar directly. It audits source schemas and the
materialized runtime feature table, but it does not modify the original project.

Audited source artifacts:

```text
v201_stable_supported_domain_flip_cases.csv
v195_selected_candidate_action_cases.csv
v77_unlabeled_batch_shift_audit.csv
v191_fn_risk_case_outputs.csv
```

Audit result:

```text
passed = true
runtime_contract_passed = true
runtime_rows = 699
runtime_columns = 29
runtime_forbidden_columns_present = []
v201_cases pred_disagree_v185 columns = 12
original_project_code_modified = false
sidecar_only = true
```

Interpretation: if formal main-project integration is ever approved, the lowest-risk change is to
add an optional stable export that provides these fields, then keep the deployable risk-control
logic in the sidecar/wrapper boundary. The audit also confirms that relaxed what-if and post-hoc
error columns remain blocked from the runtime path.

## Staging Package Smoke Audit Result

The staging package now includes package-local helper scripts needed by the deployable wrapper:

```text
scripts/final_policy_runner.py
scripts/policy_contract_validator.py
scripts/audit_rejection_layer.py
scripts/release_rejected_audit.py
```

Smoke audit result:

```text
passed = true
package_local_wrapper_loaded = true
classification_manifest_present = true
contract_passed = true
decision_rows = 1
missing_required_files = []
forbidden_columns_present = []
files_copied = 26
integration_allowed_file_n = 8
stale_files_removed = 34
```

Interpretation: the staged handoff package can load the deployable wrapper from its own package
boundary and validate the runtime template without relying on relaxed/what-if artifacts. This is
still a staging artifact inside the experiment directory, not an in-place original-project change.
The builder now clears the previous package directory before copying, so stale forbidden files
cannot persist from an earlier staging attempt.

## Handoff Classification Manifest

The staging builder now emits a machine-readable file classification manifest in both locations:

```text
outputs/handoff_file_classification_manifest.csv
outputs/handoff_file_classification_manifest.json
staging/hard_gate_integration_package/outputs/handoff_file_classification_manifest.csv
staging/hard_gate_integration_package/outputs/handoff_file_classification_manifest.json
```

The manifest separates deployable/runtime files from read-only evidence:

```text
integration_allowed_file_n = 8
deployable_code = integration entrypoint + wrapper
runtime_contract = final policy contract + runtime feature template
package_support_code = package-local helper scripts
export_contract = sidecar minimal export contract files
review_evidence = reports and result JSON/CSV files
```

Only `deployable_code`, `runtime_contract`, and package-local support scripts are marked
`integration_allowed = true`. Reports, frozen audit outputs, and export-contract evidence remain
read-only review materials.
The sidecar interface freeze/compliance reports and JSON outputs are included in this read-only
evidence class, so they improve reviewability without increasing the runtime integration surface.

## Main Project Integration Dry Run Result

The main-project integration dry run translates the staging manifest into a concrete integration
boundary. It does not modify original project code and does not authorize relaxed strict_external
logic.

Dry-run result:

```text
passed = true
recommended_strategy = sidecar_stable_export_only
integration_allowed_file_n = 8
read_only_file_n = 18
forbidden_candidate_n = 0
main_project_write_required = false
strict_external_relaxation_allowed = false
original_project_code_modified = false
```

Required integration steps, if the project later approves a production connection:

```text
add_stable_runtime_feature_export
add_preflight_contract_validation
add_sidecar_invocation_or_wrapper_adapter
keep_review_evidence_read_only
```

Blocked actions:

```text
do_not_integrate_relaxed_or_what_if_artifacts
do_not_use_strict_external_labels_for_threshold_selection
do_not_move_review_evidence_into_runtime_logic
do_not_relax_strict_external_severe_shift_fallback_without_new_independent_validation
```

Interpretation: the safest main-project path remains a stable export plus sidecar/wrapper
invocation. This preserves the no-modification boundary for the current experiment and keeps
review evidence separate from runtime logic.

## Staging Package Full Rehearsal Result

The full rehearsal runs the staged package-local deployable wrapper on the full stable feature
table instead of the one-row template used by the smoke audit.

Full rehearsal result:

```text
passed = true
package_local_wrapper_loaded = true
contract_passed = true
feature_rows = 699
decision_rows = 699
released_n = 29
review_or_reject_n = 223
strict_external_released_n = 0
forbidden_columns_present = []
reference_summary_match = true
main_project_write_required = false
original_project_code_modified = false
```

Interpretation: the staged package can reproduce the sidecar summary on the full stable feature
table while preserving the hard strict_external fallback. This is stronger than the smoke audit
because it tests the actual 699-row runtime surface through the staged package boundary.

## Staging Package Artifact Integrity Result

The artifact integrity audit hashes every file in the staged package and separates manifest-listed
files from generated handoff manifests and Python cache files.

Integrity result:

```text
passed = true
file_n = 34
integration_allowed_file_n = 8
read_only_file_n = 26
generated_manifest_file_n = 2
generated_cache_file_n = 6
unexpected_unmanifested_file_n = 0
missing_manifest_file_n = 0
forbidden_path_present_n = 0
relaxed_artifacts_present = false
main_project_write_required = false
original_project_code_modified = false
```

Interpretation: the staged package now has a content-addressed inventory. No relaxed/what-if
artifact is present, and the only unmanifested files are generated package manifests or Python cache
files created while running the staged package locally.

## Release Handoff Bundle Result

The release handoff bundle audit creates a single review index from the artifact integrity
inventory. It separates files into integration candidates, read-only evidence, generated support,
and blocked artifacts.

Bundle result:

```text
passed = true
recommended_handoff = review_staging_package_without_main_project_write
bundle_file_n = 34
integration_candidate_file_n = 8
read_only_evidence_file_n = 18
generated_support_file_n = 8
blocked_artifact_file_n = 0
runner_all_passed = true
runner_steps = 31 / 31
forbidden_path_present_n = 0
relaxed_artifacts_present = false
main_project_write_required = false
strict_external_relaxation_allowed = false
original_project_code_modified = false
```

Reproducibility command:

```bash
python experiments/risk_control_rejection_20260621/scripts/phase2_reproducibility_runner.py
```

Interpretation: this is now the safest single entry point for human review or later production
handoff. It does not authorize original-project writes, and it keeps strict_external relaxation
blocked unless new independent validation is added later.

## Sidecar External Interface Freeze Result

The sidecar external interface freeze audit locks the stable runtime input schema and deployable
decision output schema as a versioned interface.

Interface freeze result:

```text
passed = true
interface_version = sidecar_external_interface_v1
contract_id = risk_control_rejection_v1_dino_tiebreak
feature_version = risk_control_v1
runtime_required_column_n = 11
audit_only_column_n = 3
forbidden_selector_column_n = 8
decision_output_column_n = 11
missing_runtime_required_columns = []
forbidden_present_in_runtime_required = []
audit_only_present_in_runtime_required = []
forbidden_present_in_decision_output = []
interface_signature_sha256 = c667d639d89a8fc8c759e21318fec70523c90291d01e8b85b6c8ae342669434b
main_project_write_required = false
strict_external_relaxation_allowed = false
original_project_code_modified = false
```

Interpretation: future integration should treat `sidecar_external_interface_v1` as the stable
boundary. The original project should only export the frozen runtime-required columns and consume
the frozen decision-output columns; audit-only, error, and relaxed selector fields remain outside
the runtime interface.

## Sidecar Interface Compliance Result

The sidecar interface compliance audit checks the current sidecar runtime feature export and
decision output against the frozen interface.

Compliance result:

```text
passed = true
interface_version = sidecar_external_interface_v1
runtime_row_count = 699
decision_row_count = 699
row_count_matches = true
runtime_required_column_n = 11
decision_output_column_n = 11
forbidden_selector_column_n = 8
audit_only_column_n = 3
runtime_missing_required_columns = []
runtime_forbidden_columns_present = []
runtime_audit_only_columns_present = []
decision_missing_output_columns = []
decision_extra_output_columns = []
decision_forbidden_columns_present = []
decision_order_matches = true
main_project_write_required = false
strict_external_relaxation_allowed = false
original_project_code_modified = false
```

Interpretation: the latest sidecar outputs conform to the frozen interface. This closes the
stale-output risk in the reproducibility chain by requiring the sidecar adapter to run before
interface freeze and compliance checks.

## Deployment Failure Modes Result

The deployment failure modes audit intentionally corrupts the sidecar runtime interface, decision
interface, and package manifest to ensure the candidate deployment path fails closed.

Failure-mode result:

```text
passed = true
scenario_n = 6
blocked_scenario_n = 6
failed_scenario_n = 0
unsafe_auto_allowed = false
main_project_write_required = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
```

Blocked scenarios:

```text
missing_runtime_required_column
bad_policy_feature_version
forbidden_runtime_selector_column
decision_output_order_drift
runtime_decision_row_count_mismatch
relaxed_artifact_in_package_manifest
```

Interpretation: the sidecar path now has a concrete fail-closed audit for common integration
mistakes. Bad feature versions, forbidden selector columns, decision schema drift, row-count drift,
and accidental relaxed/what-if package artifacts are blocked rather than allowed to create new
automatic decisions.

## Main-project Sidecar Patch Plan Result

The sidecar patch plan audit converts the earlier dry-run recommendation into a concrete future
patch checklist. It does not modify the original project.

Patch-plan result:

```text
passed = true
recommended_strategy = prepare_sidecar_patch_only_after_review
future_main_project_touch_count = 4
main_project_write_performed = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
required_evidence_missing = []
```

Future main-project touch points, if explicitly approved later:

```text
stable_runtime_feature_export_hook
preflight_contract_validation_hook
sidecar_invocation_hook
decision_output_consumer_hook
```

Non-main-project gate:

```text
ci_reproducibility_gate
```

Interpretation: the safest future integration remains a minimal sidecar patch, not broad in-place
rewiring. Review evidence stays outside runtime logic, strict_external fallback remains unchanged,
and relaxed/what-if artifacts remain blocked.

## Main-project Gap Audit Result

The gap audit scans the original project read-only and maps the patch-plan touch points to current
candidate hooks or explicit gaps. It writes only experiment-side audit artifacts.

Gap-audit result:

```text
passed = true
recommended_strategy = keep_sidecar_until_minimal_hooks_are_reviewed
future_touchpoint_n = 4
candidate_hook_found_n = 1
missing_minimal_hook_n = 3
main_project_write_performed = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
```

Found candidate:

```text
stable_runtime_feature_export_hook
```

Still missing minimal hooks:

```text
preflight_contract_validation_hook
sidecar_invocation_hook
decision_output_consumer_hook
```

Interpretation: original prediction export sites can support a future stable runtime-feature
export, but the project does not yet have the contract preflight, sidecar invocation, or decision
consumer hooks. The current recommendation therefore remains sidecar-first until a minimal patch is
explicitly reviewed.

## Export-hook Candidate Prioritization Result

The export-hook prioritization audit narrows the broad stable-feature export candidate list from the
gap audit. It prefers package-level maintained training/export modules over one-off historical
scripts.

Prioritization result:

```text
passed = true
recommended_strategy = review_core_package_export_candidates_before_any_main_project_patch
candidate_n = 70
shortlist_n = 6
top_candidate_path = thymic_surimage/train.py
main_project_write_performed = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
```

Preferred candidates:

```text
thymic_surimage/train.py
thymic_surimage/strict_train.py
thymic_baseline/aggregate_cv_run.py
thymic_baseline/train.py
thymic_surimage/ablation_train.py
thymic_surimage/evaluate_strict_pipeline.py
```

Interpretation: if a future minimal main-project patch is approved, the first review target should
be the core prediction-export boundary around `thymic_surimage/train.py`, with `strict_train.py` and
baseline aggregation as adjacent candidates. Historical dated scripts should remain low-priority
unless they are proven to be the active production entry point.

## Export-hook Patch Blueprint Result

The patch blueprint audit reads the prioritized source files and identifies concrete candidate
functions and line numbers for a future stable runtime-feature export hook. It still does not modify
the original project.

Blueprint result:

```text
passed = true
recommended_strategy = review_blueprint_before_any_main_project_write
candidate_reviewed_n = 6
blueprint_ready_n = 4
manual_review_required_n = 2
top_ready_path = thymic_surimage/train.py
top_ready_function = save_standard_outputs
main_project_write_performed = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
```

Top concrete insertion point:

```text
path = thymic_surimage/train.py
function = save_standard_outputs
suggested_insert_after_line = 395
hook_kind = post_case_prediction_export
```

Interpretation: if a minimal patch is explicitly approved later, the safest first draft is a
sidecar-safe exporter call immediately after `case_predictions_mean` is written in
`save_standard_outputs`. This remains a blueprint only; it does not authorize writing into
`thymic_surimage/train.py`.

## Missing Runtime Hooks Blueprint Result

The missing runtime hooks blueprint sequences the remaining production-facing hooks that would be
needed if a future main-project patch is explicitly approved. It still performs no main-project
write.

Blueprint result:

```text
passed = true
recommended_strategy = review_ordered_runtime_hooks_before_any_main_project_write
hook_n = 4
ready_hook_n = 4
blocked_hook_n = 0
top_export_path = thymic_surimage/train.py
top_export_function = save_standard_outputs
interface_version = sidecar_external_interface_v1
decision_order_matches = true
row_count_matches = true
main_project_write_performed = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
```

Ordered hook blueprint:

```text
1. stable_runtime_feature_export_hook -> thymic_surimage/train.py::save_standard_outputs
2. preflight_contract_validation_hook -> scripts/policy_contract_validator.py
3. sidecar_invocation_hook -> integration/risk_control_deployable_wrapper.py
4. decision_output_consumer_hook -> sidecar_external_interface_v1 decision columns
```

Interpretation: the future reviewed patch should be a small ordered chain, not a broad rewrite.
The runtime path remains locked to the hard-gate deployable policy and the frozen sidecar interface.

## Main-project Patch Rehearsal Package Result

The patch rehearsal package converts the ordered hook blueprint into review-only patch artifacts.
It does not apply a patch and does not modify the original project.

Rehearsal result:

```text
passed = true
recommended_strategy = human_review_non_applying_patch_rehearsal_before_any_main_project_write
draft_artifact_n = 4
ready_artifact_n = 4
blocked_artifact_n = 0
top_export_path = thymic_surimage/train.py
top_export_function = save_standard_outputs
patch_draft_path = patch_rehearsal/main_project_sidecar_hook_draft.patch.txt
patch_applied = false
main_project_write_performed = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
forbidden_terms_present = []
```

Interpretation: the project now has a concrete, non-applying review artifact for the smallest
future main-project hook chain. This remains outside the original project and should not be applied
without explicit approval and review.

## Patch Rehearsal Safety Audit Result

The patch rehearsal safety audit checks that the draft remains a review-only artifact and cannot be
mistaken for an applied production patch.

Safety audit result:

```text
passed = true
issue_n = 0
draft_artifact_n = 4
do_not_apply_marker_present = true
review_only_marker_present = true
deployable_mode_marker_present = true
patch_applied = false
main_project_write_performed = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
forbidden_terms_present = []
```

Interpretation: the rehearsal package is suitable for human review as a non-applying artifact. It
still does not authorize original-project edits.

## Main-project Integration Decision Gate Result

The decision gate combines the runner, handoff, runtime-hook blueprint, patch rehearsal package,
and safety audit into one pre-integration decision. Passing this gate means the evidence is ready
for human review, not that original-project writes are allowed.

Decision gate result:

```text
passed = true
decision = review_only_wait_for_explicit_main_project_write_approval
ready_for_human_review = true
auto_apply_allowed = false
blocking_issue_n = 0
runner_steps = 31 / 31
runtime_hooks_ready_n = 4
patch_rehearsal_issue_n = 0
main_project_write_performed = false
main_project_write_required = false
strict_external_relaxation_allowed = false
strict_external_labels_used_for_selection = false
original_project_code_modified = false
```

Interpretation: this is the current stopping rule before any original-project code change. The
next transition, if approved later, should start from human review of the non-applying patch
rehearsal package, not from automatic application.

## Next Stage

The next stage should not keep trying to loosen strict_external on the same 108 cases. The
scientifically defensible path is:

1. Preserve the deployable hard gate and current wrapper behavior.
2. Treat the 8-case and 21-case strict_external releases as frozen what-if evidence only.
3. If project code is modified later, integrate only the stable interface, reproducibility checks,
   and hard-gate deployable policy, not the relaxed strict_external candidate releases.
4. Any future strict_external relaxation requires additional independent external validation or a
   formally predeclared prospective audit.

Phase 3 has prepared a reviewable hard-gate staging package, a runnable sidecar adapter, and a
stable-feature bridge without modifying original project code. Original-project in-place integration
remains a separate decision and is not required for continued validation.
