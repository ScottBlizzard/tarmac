# Phase 2 Shift Gate Progress Report

## Scope

This stage continues under `experiments/risk_control_rejection_20260621/` only. No original project
code was modified.

The goal was to test whether the current strict_external hard fallback is clearly over-conservative,
using the project's existing external-generalization data and label-free shift metrics.

Strict_external labels were not used to choose gate thresholds. They were used only after frozen
decisions/candidates were produced for audit error counts.

## Experiments Completed

### 1. Label-free metric-space shift calibration

Script:

```text
scripts/phase2_shift_gate_calibration_audit.py
```

Default calibration:

```text
known domains = old_data, third_batch
audit domain = strict_external
severe multiplier = 1.5
minimum exceeded metrics = 3
```

Result:

```text
strict_exceeded_metric_n = 4 / 4
strict_gate_supported = true
recommendation = keep_severe_unknown_shift_fallback
```

Interpretation: under the default label-free calibration, strict_external still has enough
unsupervised shift evidence to keep the severe-shift fallback.

### 2. Severe gate sensitivity grid

Script:

```text
scripts/phase2_shift_gate_sensitivity_grid.py
```

First multiplier where strict_external is no longer severe:

| Minimum Exceeded Metrics | First Not-severe Multiplier |
| ---: | ---: |
| 1 | 15.0 |
| 2 | 5.0 |
| 3 | 2.5 |
| 4 | 2.0 |

Interpretation: with the default requirement of at least 3 exceeded metrics, the threshold must be
loosened from 1.5x to 2.5x before strict_external stops being severe. This suggests the current
hard gate is not a near-boundary artifact.

### 3. Relaxed gate quota-15 frozen what-if

Script:

```text
scripts/phase2_relaxed_gate_policy_audit.py
```

What-if gate:

```text
severe multiplier = 2.5
minimum exceeded metrics = 3
strict_external gate = normal_or_known_shift
```

Result:

```text
current strict_external released_n = 0
relaxed strict_external released_n = 8
relaxed strict_external released_error_n = 0
relaxed strict_external released_high_risk_fn_n = 0
promote_to_deployable = false
```

Interpretation: if the gate were relaxed enough to treat strict_external as normal/known shift, the
existing quota-15 rule would release 8 strict_external cases with no observed released error in this
frozen audit. This remains a what-if, not a deployable recommendation.

### 4. Relaxed gate + known-domain intersection frozen what-if

Script:

```text
scripts/phase2_relaxed_intersection_policy_audit.py
```

What-if gate and thresholds:

```text
severe multiplier = 2.5
minimum exceeded metrics = 3
model_disagreement_rate <= 0.5
dino_fn_risk_max <= 0.3696929763812454
```

Result:

```text
known_allowed_candidate_n = 80
known_allowed_error_n = 0
known_allowed_high_risk_fn_n = 0
strict_external_allowed_candidate_n = 21
strict_external_allowed_error_n = 0
strict_external_allowed_high_risk_fn_n = 0
promote_to_deployable = false
```

Interpretation: this is the strongest Phase 2 frozen what-if so far. It combines a relaxed
label-free gate with the Phase 1 known-domain intersection thresholds. It gives 21 strict_external
candidate releases with 0 observed errors, but it still cannot be promoted because the relaxed gate
itself has not been independently validated.

### 5. Leave-one-known relaxed gate stress

Script:

```text
scripts/phase2_leave_one_known_gate_audit.py
```

Stress setting:

```text
severe multiplier = 2.5
minimum exceeded metrics = 3
calibration = one known domain at a time
```

Result:

```text
heldout_known_stable_non_severe = true
strict_gate_stable_across_known_domains = false
strict_severe_train_domain_n = 1 / 2
promote_relaxed_gate = false
```

Interpretation: the 2.5x relaxed gate does not misclassify the held-out known domain as severe, but
strict_external changes classification depending on whether old_data or third_batch is used as the
single calibration source. This means the 2.5x relaxation is too sensitive to promote.

### 6. Leave-one-known sensitivity grid

Script:

```text
scripts/phase2_leave_one_gate_sensitivity_grid.py
```

First stable strict_external classifications:

| Minimum Exceeded Metrics | First Stable Strict Non-severe | First Stable Strict Severe |
| ---: | ---: | ---: |
| 1 | - | 2.0 |
| 2 | 10.0 | 1.5 |
| 3 | 5.0 | 1.0 |
| 4 | 2.0 | 1.0 |

Interpretation: under the default requirement of at least 3 exceeded metrics, strict_external only
becomes stably non-severe across leave-one-known calibration at a 5.0x multiplier. That is much
looser than the 2.5x threshold that produced the attractive 21-case frozen what-if, so the relaxed
gate remains insufficiently justified for deployment.

### 7. Case-level robust core grid

Script:

```text
scripts/phase2_case_level_robust_core_grid.py
```

This experiment kept strict_external labels out of selection and shrank the known-domain
intersection thresholds after they were derived from old_data + third_batch:

| Shrink Factor | Disagreement Threshold | DINO FN-risk Threshold | Known Candidates | Known Errors | Strict Candidates | Strict Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.500 | 0.3696929763812454 | 80 | 0 | 21 | 0 |
| 0.90 | 0.450 | 0.33272367874312087 | 47 | 0 | 19 | 0 |
| 0.75 | 0.375 | 0.27726973228593405 | 34 | 0 | 12 | 0 |
| 0.50 | 0.250 | 0.1848464881906227 | 18 | 0 | 1 | 0 |
| 0.25 | 0.125 | 0.09242324409531134 | 0 | 0 | 0 | 0 |

Result:

```text
selected_safe_shrink_factor = 1.0
selected_strict_external_candidate_n = 21
selected_strict_external_error_n = 0
promote_to_deployable = false
```

Interpretation: case-level threshold tightening gives a controllable coverage/safety tradeoff and
all tested shrink factors have 0 observed released errors in this frozen audit. However, the best
frozen coverage remains the original intersection rule at 21 strict_external cases. This does not
solve the batch-gate problem, because those 21 releases still require treating strict_external as
non-severe under a relaxation that failed leave-one-known stability.

### 8. Shift metric ablation audit

Script:

```text
scripts/phase2_shift_metric_ablation_audit.py
```

This experiment checked whether the default severe-shift classification was driven by one
unsupervised metric or by multiple independent-looking signals. It used only batch-shift metrics,
not strict_external labels.

All-metric result by multiplier:

| Multiplier | Exceeded Metrics | Severe Under 75% Rule | Exceeded Metric Names |
| ---: | ---: | --- | --- |
| 1.0 | 4 / 4 | true | batch_shift_index, domain_auc_cv, mean_outside_ref_05_95_rate, quality_proxy_mean |
| 1.25 | 4 / 4 | true | batch_shift_index, domain_auc_cv, mean_outside_ref_05_95_rate, quality_proxy_mean |
| 1.5 | 4 / 4 | true | batch_shift_index, domain_auc_cv, mean_outside_ref_05_95_rate, quality_proxy_mean |
| 2.0 | 3 / 4 | true | batch_shift_index, mean_outside_ref_05_95_rate, quality_proxy_mean |
| 2.5 | 2 / 4 | false | mean_outside_ref_05_95_rate, quality_proxy_mean |
| 3.0 | 2 / 4 | false | mean_outside_ref_05_95_rate, quality_proxy_mean |
| 4.0 | 2 / 4 | false | mean_outside_ref_05_95_rate, quality_proxy_mean |
| 5.0 | 1 / 4 | false | quality_proxy_mean |
| 10.0 | 1 / 4 | false | quality_proxy_mean |
| 15.0 | 0 / 4 | false | - |

Default 1.5x result:

```text
default_exceeded_metric_n = 4 / 4
default_leave_one_out_all_severe = true
default_single_metric_severe_n = 4 / 4
audit_to_known_max_ratio_by_metric:
  batch_shift_index = 2.40x
  domain_auc_cv = 1.87x
  mean_outside_ref_05_95_rate = 4.45x
  quality_proxy_mean = 13.95x
promote_relaxed_gate = false
```

Interpretation: the default hard gate is not explained by one fragile metric. At 1.5x, every
metric exceeds its known-domain maximum by enough to trigger severe evidence, and every
leave-one-out subset still classifies strict_external as severe. The 2.5x relaxation turns off the
all-metric severe decision only because batch_shift_index and domain_auc_cv no longer exceed their
loosened thresholds; quality_proxy_mean and outside-reference rate remain strongly shifted.

### 9. Hard-gate deployable frontier

Script:

```text
scripts/phase2_hard_gate_deployable_frontier.py
```

This experiment scanned release fraction and quota granularity while keeping the deployable
severe-shift gate unchanged. Any `severe_unknown_shift` domain was forced back to v195. Candidate
selection also required old_data + third_batch fold-safe evidence, preventing a purely aggregate
zero-error result from being selected if it was unstable across folds.

Selected candidate:

```text
granularity = domain
release_fraction = 0.15
third_batch_released_n = 18
third_batch_released_error_n = 0
third_batch_released_high_risk_fn_n = 0
internal_known_released_n = 29
all_domains_released_n = 29
all_domains_released_error_n = 0
all_domains_released_high_risk_fn_n = 0
strict_external_released_n = 0
known_fold_safe_zero_new_release_error = true
promote_gate_relaxation = false
```

Important correction during this audit: the aggregate-only scan initially favored
`domain / 0.20` with 39 released cases and 0 aggregate released errors, but that candidate was too
optimistic because fold-level stress evidence had already shown higher fractions can be unstable.
After adding the fold-safe requirement, the selected frontier returned to `domain / 0.15`, matching
the existing deployable behavior.

Interpretation: under the current hard batch gate, there is no defensible Phase 2 coverage
improvement beyond the existing deployable policy. strict_external remains 0-release by design, and
third_batch stays at 18 released cases with 0 newly released errors.

### 10. Phase 3 integration readiness and staging

Scripts:

```text
scripts/phase3_integration_readiness_audit.py
scripts/phase3_build_staged_integration_package.py
```

The integration readiness audit converted Phase 2 results into an explicit allowed/blocked
boundary:

```text
recommendation = integrate_hard_gate_only
allowed_items = hard_gate_deployable_policy
blocked_items =
  relaxed_strict_external_quota15
  relaxed_strict_external_intersection
  case_level_robust_core_strict_candidates
blocked_as_expected = true
guards_ok = true
original_project_code_modified = false
```

The staged package then copied only hard-gate deployable integration artifacts into:

```text
staging/hard_gate_integration_package
```

Staging result:

```text
files_copied = 26
integration_allowed_file_n = 8
stale_files_removed = 34
missing_required = []
forbidden_copied = []
original_project_code_modified = false
```

Interpretation: Phase 3 did not change original project code. It created a reviewable staging
package for future integration and explicitly excluded relaxed/what-if strict_external artifacts.
The staging builder now clears the previous package directory before copying, so stale forbidden
files cannot persist from an earlier staging attempt.
The staging package now also includes the minimal export contract document, its audit script, and
the generated audit JSON so a future main-project export review can start from the stable sidecar
boundary rather than from relaxed exploratory artifacts. It also includes package-local helper
scripts required by the deployable wrapper. A machine-readable handoff classification manifest is
written to both the experiment outputs and the staging package:
The sidecar interface freeze/compliance reports and JSON outputs are now copied as read-only review
evidence, not as runtime integration candidates.

```text
outputs/handoff_file_classification_manifest.csv
outputs/handoff_file_classification_manifest.json
staging/hard_gate_integration_package/outputs/handoff_file_classification_manifest.csv
staging/hard_gate_integration_package/outputs/handoff_file_classification_manifest.json
```

### 11. Sidecar original-project adapter

Script:

```text
sidecar_original_project_adapter/original_project_sidecar_adapter.py
```

The sidecar is the recommended connection pattern for the original project. It is a runnable
adapter under the experiment directory rather than an in-place change to original project code. It
reads a project output table, maps limited stable aliases, validates the runtime feature contract,
blocks relaxed/error selector columns, and calls the hard-gate deployable wrapper.

Default run:

```text
input_rows = 699
runtime_rows = 699
decision_rows = 699
released_n = 29
review_or_reject_n = 223
strict_external_released_n = 0
contract_passed = true
forbidden_columns_present = []
original_project_code_modified = false
sidecar_only = true
```

Interpretation: the project now has a practical no-modification route for running the deployable
hard-gate policy on existing project outputs. It still preserves the core Phase 2 decision:
strict_external is not released by the deployable path.

### 12. Stable feature bridge from original outputs

Script:

```text
sidecar_original_project_adapter/original_output_stable_feature_bridge.py
```

The bridge reconstructs the stable `risk_control_v1` feature table from existing project artifacts:

```text
v201_stable_supported_domain_flip_cases.csv
v195_selected_candidate_action_cases.csv
v77_unlabeled_batch_shift_audit.csv
v191_fn_risk_case_outputs.csv
```

Result:

```text
feature_rows = 699
runtime_rows = 699
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

Interpretation: raw v185/v195 output tables are not directly deployable, but existing project
artifacts contain enough information to rebuild the stable feature table outside the original code
path. The bridge reproduces the current adapter feature table and preserves strict_external hard
fallback.

### 13. Minimal export contract audit

Script:

```text
sidecar_original_project_adapter/minimal_export_contract_audit.py
```

This audit defines the smallest optional export surface needed if the original project later feeds
the risk-control sidecar directly. It is deliberately a dry-run contract audit, not a main-project
code change.

Audited sources:

```text
v201_stable_supported_domain_flip_cases.csv
v195_selected_candidate_action_cases.csv
v77_unlabeled_batch_shift_audit.csv
v191_fn_risk_case_outputs.csv
```

Result:

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

Interpretation: the existing project artifacts already contain the information needed to materialize
the stable sidecar interface. If main-project integration is approved later, the safest approach is
an optional stable export boundary, not moving relaxed what-if logic into the production code path.

### 14. Staging package smoke audit

Script:

```text
scripts/phase3_staging_package_smoke_audit.py
```

This audit loads the deployable wrapper from the staged handoff package itself, validates the
package-local runtime feature template against the package-local contract, and checks that no
forbidden runtime columns are emitted.

Result:

```text
passed = true
package_local_wrapper_loaded = true
classification_manifest_present = true
integration_allowed_file_n = 8
contract_passed = true
decision_rows = 1
missing_required_files = []
forbidden_columns_present = []
strict_external_released_n = 0
```

Interpretation: the handoff package is now reviewable as a package boundary. It still does not
modify original project code, but it contains the helper scripts needed to load the deployable
wrapper locally instead of relying on hidden imports from the experiment tree. The classification
manifest marks deployable code, runtime contracts, and package support scripts as integration
allowed, while reports, frozen result files, and export-contract evidence remain read-only review
materials.

### 15. Main-project integration dry run

Script:

```text
scripts/phase3_main_project_integration_dry_run.py
```

This dry run converts the staging classification manifest into an explicit main-project connection
plan. It is a no-write audit: no original project code is changed, strict_external labels are not
used for selection, and relaxed/what-if artifacts remain blocked.

Result:

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

Required steps if integration is approved later:

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

Interpretation: the dry run supports the same recommendation as the sidecar work. The best current
path is a stable feature export and sidecar/wrapper invocation, not direct promotion of exploratory
strict_external relaxation into the original project.

### 16. Staging package full rehearsal

Script:

```text
scripts/phase3_staging_package_full_rehearsal.py
```

This rehearsal runs the staged package-local deployable wrapper on the full stable feature table
instead of the one-row smoke template. It checks that the staged package reproduces the sidecar
summary without writing to original project code.

Result:

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

Interpretation: the staged handoff package is no longer only structurally smoke-tested. It can run
the full 699-row stable feature table through its own package-local wrapper and reproduce the
sidecar summary while preserving the hard strict_external fallback.

### 17. Staging package artifact integrity audit

Script:

```text
scripts/phase3_staging_package_artifact_integrity_audit.py
```

This audit hashes every staged package file and separates manifest-listed files from generated
handoff manifests and Python cache files. It is designed to prevent relaxed/what-if artifacts from
being accidentally treated as integration candidates.

Result:

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

Interpretation: the staging package is now content-addressed for handoff review. No forbidden
relaxed/what-if path is present, and no unexpected unmanifested file is present.

### 18. Release handoff bundle audit

Script:

```text
scripts/phase3_release_handoff_bundle_audit.py
```

This audit turns the artifact integrity inventory into a single handoff index. It separates
integration candidates, read-only evidence, generated support files, and blocked artifacts.

Result:

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

Interpretation: the handoff now has one review entry point. It points back to the full
reproducibility runner and keeps review evidence, generated files, and runtime candidates separated.

### 19. Sidecar external interface freeze audit

Script:

```text
scripts/phase3_sidecar_external_interface_freeze_audit.py
```

This audit freezes the sidecar external interface as a versioned schema. It records runtime
required columns, audit-only columns, forbidden selector columns, and deployable decision output
columns with a deterministic signature.

Result:

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

Interpretation: if integration is approved later, the original project should connect only through
`sidecar_external_interface_v1`; experimental audit/error/relaxed fields remain outside runtime.

### 20. Sidecar interface compliance audit

Script:

```text
scripts/phase3_sidecar_interface_compliance_audit.py
```

This audit checks the current sidecar runtime feature export and decision output against the
frozen interface schema.

Result:

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

Interpretation: the sidecar adapter now runs before interface freeze and compliance checks in the
reproducibility chain, so the frozen interface is checked against current sidecar outputs instead
of stale files.

### 21. Deployment failure modes audit

Script:

```text
scripts/phase3_deployment_failure_modes_audit.py
```

This audit intentionally corrupts the sidecar runtime interface, decision output interface, and
staging package manifest. The expected behavior is fail-closed blocking, not automatic decision
release.

Result:

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

Interpretation: the candidate sidecar path now has explicit evidence that common integration
mistakes are blocked. This hardens external generalization by preventing unsafe automatic decisions
from malformed runtime features, schema drift, stale/relaxed artifacts, or row-count mismatch.

### 22. Main-project sidecar patch plan

Script:

```text
scripts/phase3_main_project_sidecar_patch_plan.py
```

This audit creates a future minimal patch checklist without touching the original project.

Result:

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

Future touch points:

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

Interpretation: if original-project integration is explicitly approved later, the patch should be
limited to a stable feature export, contract validation, sidecar invocation, and decision-output
consumption. This still blocks relaxed/what-if artifacts and strict_external label use.

### 23. Main-project gap audit

Script:

```text
scripts/phase3_main_project_gap_audit.py
```

This audit scans the original project read-only and maps the future patch touch points to candidate
hook locations or remaining gaps.

Result:

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

Still missing:

```text
preflight_contract_validation_hook
sidecar_invocation_hook
decision_output_consumer_hook
```

Interpretation: the original project has candidate prediction-export boundaries, but it still lacks
the minimum runtime safety hooks needed for direct integration. Continue using the sidecar package
unless a reviewed minimal patch is explicitly approved.

### 24. Export-hook candidate prioritization

Script:

```text
scripts/phase3_export_hook_candidate_prioritization.py
```

This audit ranks the broad stable export-hook candidate set from the gap audit and narrows it to
package-level candidates for later human review.

Result:

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

Interpretation: the next integration review, if approved later, should begin at the maintained
package export path rather than dated one-off scripts. This remains read-only planning evidence.

### 25. Export-hook patch blueprint

Script:

```text
scripts/phase3_export_hook_patch_blueprint.py
```

This audit reads the prioritized source files and identifies concrete future insertion points for a
stable runtime-feature export hook. It does not edit those files.

Result:

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

Top insertion point:

```text
thymic_surimage/train.py:395
function = save_standard_outputs
hook_kind = post_case_prediction_export
```

Interpretation: a future reviewed patch can be kept very narrow: add an approved sidecar-safe
exporter call after case-level predictions are written. No main-project write has been performed.

### 26. Missing runtime hooks blueprint

Script:

```text
scripts/phase3_missing_runtime_hooks_blueprint.py
```

This blueprint sequences the remaining runtime hooks that would be needed for a future approved
main-project patch. It performs no main-project write.

Result:

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

Interpretation: future integration should be reviewed as an ordered four-hook chain. The deployable
decision remains the conservative hard-gate policy; relaxed strict_external what-if evidence is not
part of this runtime path.

### 27. Main-project patch rehearsal package

Script:

```text
scripts/phase3_main_project_patch_rehearsal_package.py
```

This package converts the ordered runtime-hook blueprint into review-only patch artifacts. It does
not apply a patch and does not modify the original project.

Result:

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

Interpretation: there is now a concrete non-applying review artifact for the future minimal
main-project hook chain. It is evidence for review, not an authorization to edit the original
project.

### 28. Patch rehearsal safety audit

Script:

```text
scripts/phase3_patch_rehearsal_safety_audit.py
```

This audit verifies that the patch rehearsal package remains review-only and cannot be confused
with an applied production patch.

Result:

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

Interpretation: the non-applying patch rehearsal package is now guarded by a dedicated safety
audit. It remains review evidence only.

### 29. Main-project integration decision gate

Script:

```text
scripts/phase3_main_project_integration_decision_gate.py
```

This gate combines the runner, handoff, runtime-hook blueprint, patch rehearsal package, and safety
audit into one pre-integration decision.

Result:

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

Interpretation: evidence can be reviewed, but automatic original-project modification remains
blocked. Any main-project write still requires explicit approval.

## Verification

Current verification:

```text
unit tests = 114 passed
phase2_reproducibility_runner steps = 31 / 31 passed
```

## Current Decision

Do not modify original project code yet.

The current deployable wrapper should remain unchanged:

```text
strict_external released_n = 0
strict_external fallback = severe_unknown_shift_fallback_to_v195
```

The 2.5x relaxed gate failed leave-one-known stability for strict_external. The 21-case relaxed
intersection result should therefore remain frozen evidence only. Case-level tightening can reduce
the candidate set, but it does not provide enough independent evidence to override the severe batch
gate. Metric ablation supports the current hard fallback because the default strict_external severe
classification is not dependent on a single shift metric. The hard-gate deployable frontier did not
find a fold-safe coverage improvement beyond the existing `domain / 0.15` deployable policy.

Phase 2 should close with the current deployable decision and a clear frozen what-if section rather
than forcing gate relaxation. Phase 3 can proceed only as hard-gate staging/review work unless the
user explicitly authorizes original-project code changes.
