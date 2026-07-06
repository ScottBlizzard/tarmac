# Phase 3 Main-project Sidecar Patch Plan

## Result

- passed: True
- recommended_strategy: `prepare_sidecar_patch_only_after_review`
- future_main_project_touch_count: 4
- main_project_write_performed: False
- strict_external_relaxation_allowed: False
- strict_external_labels_used_for_selection: False
- original_project_code_modified: False
- required_evidence_missing: []

## Minimal Future Patch Touches

| Change | Target Area | Future Main-project Touch | Minimal Action |
| --- | --- | --- | --- |
| `stable_runtime_feature_export_hook` | original_project_output_boundary | True | export exactly the frozen risk_control_v1 runtime-required columns |
| `preflight_contract_validation_hook` | original_project_or_sidecar_boundary | True | validate exported rows against the frozen contract before any decision call |
| `sidecar_invocation_hook` | original_project_inference_boundary | True | call the sidecar deployable wrapper and keep strict_external severe fallback unchanged |
| `decision_output_consumer_hook` | original_project_decision_boundary | True | consume only the frozen decision-output columns from sidecar_external_interface_v1 |
| `ci_reproducibility_gate` | experiment_review_or_ci | False | run the full experiment reproducibility runner before any patch review |

## Blocked Actions

- `do_not_integrate_relaxed_or_what_if_artifacts`
- `do_not_copy_review_evidence_into_runtime_logic`
- `do_not_use_strict_external_labels_for_threshold_selection`
- `do_not_relax_strict_external_fallback_without_new_independent_validation`
- `do_not_modify_original_project_until_patch_review_is_explicitly_approved`

## Interpretation

This is a patch plan only. It lists the smallest future touch points if explicit
main-project integration is approved later. The current experiment still performs no
original-project writes and does not relax strict_external fallback behavior.