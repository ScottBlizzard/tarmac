# Phase 3 Missing Runtime Hooks Blueprint

## Result

- passed: True
- recommended_strategy: `review_ordered_runtime_hooks_before_any_main_project_write`
- hook_n: 4
- ready_hook_n: 4
- blocked_hook_n: 0
- top_export_path: `thymic_surimage/train.py`
- top_export_function: `save_standard_outputs`
- interface_version: `sidecar_external_interface_v1`
- decision_order_matches: True
- row_count_matches: True
- main_project_write_performed: False
- strict_external_relaxation_allowed: False
- strict_external_labels_used_for_selection: False
- original_project_code_modified: False

## Ordered Hook Blueprint

| Seq | Hook | Status | Target | Approved Source | Minimal Action |
| ---: | --- | --- | --- | --- | --- |
| 1 | `stable_runtime_feature_export_hook` | `blueprint_ready` | `thymic_surimage/train.py` | `outputs/phase3_export_hook_patch_blueprint.csv` | export frozen risk_control_v1 runtime-required columns from the case-prediction output boundary |
| 2 | `preflight_contract_validation_hook` | `blueprint_ready` | `runtime_feature_export_boundary` | `scripts/policy_contract_validator.py` | validate feature version, required columns, and forbidden selector columns before any decision call |
| 3 | `sidecar_invocation_hook` | `blueprint_ready` | `post_preflight_runtime_features` | `integration/risk_control_deployable_wrapper.py` | invoke the deployable sidecar wrapper in shift_aware_deployable mode only |
| 4 | `decision_output_consumer_hook` | `blueprint_ready` | `downstream_decision_consumer` | `outputs/phase3_sidecar_external_interface_freeze_schema.csv` | consume only frozen deployable decision columns and preserve row-count/order checks |

## Interpretation

This is an ordered blueprint for the runtime hooks still missing from the original
project. It records the approved experiment-side source for each hook and keeps
strict_external labels, relaxed what-if policies, and audit-only fields out of
the deployable path. It performs no main-project writes.