# Phase 3 Sidecar External Interface Freeze Audit

## Result

- passed: True
- interface_version: `sidecar_external_interface_v1`
- contract_id: `risk_control_rejection_v1_dino_tiebreak`
- feature_version: `risk_control_v1`
- runtime_required_column_n: 11
- audit_only_column_n: 3
- forbidden_selector_column_n: 8
- decision_output_column_n: 11
- interface_signature_sha256: `c667d639d89a8fc8c759e21318fec70523c90291d01e8b85b6c8ae342669434b`
- main_project_write_required: False
- strict_external_relaxation_allowed: False
- original_project_code_modified: False

## Frozen Columns

| Role | Ordinal | Column |
| --- | ---: | --- |
| audit_only_input | 0 | `label_idx` |
| audit_only_input | 1 | `task_l6_label` |
| audit_only_input | 2 | `fold_id` |
| decision_output | 0 | `policy_mode` |
| decision_output | 1 | `domain` |
| decision_output | 2 | `case_id` |
| decision_output | 3 | `v195_auto_decision` |
| decision_output | 4 | `v195_review_or_reject` |
| decision_output | 5 | `hard_shift_gate` |
| decision_output | 6 | `release_from_v195_review` |
| decision_output | 7 | `auto_decision` |
| decision_output | 8 | `review_or_reject` |
| decision_output | 9 | `final_pred` |
| decision_output | 10 | `release_fraction` |
| forbidden_selector | 0 | `action_error` |
| forbidden_selector | 1 | `auto_correct_action` |
| forbidden_selector | 2 | `rescued_by_action` |
| forbidden_selector | 3 | `hurt_by_action` |
| forbidden_selector | 4 | `auto_error` |
| forbidden_selector | 5 | `auto_high_risk_fn` |
| forbidden_selector | 6 | `released_error` |
| forbidden_selector | 7 | `released_high_risk_fn` |
| runtime_required | 0 | `policy_feature_version` |
| runtime_required | 1 | `domain` |
| runtime_required | 2 | `case_id` |
| runtime_required | 3 | `v195_review_or_reject` |
| runtime_required | 4 | `v195_auto_decision` |
| runtime_required | 5 | `hard_shift_gate` |
| runtime_required | 6 | `model_disagreement_rate` |
| runtime_required | 7 | `dino_fn_risk_max` |
| runtime_required | 8 | `dino_fn_risk_missing` |
| runtime_required | 9 | `final_pred` |
| runtime_required | 10 | `prob_mean_core` |

## Interpretation

This audit freezes the sidecar external interface as a versioned schema. Runtime required
columns, audit-only inputs, forbidden selector columns, and decision outputs are recorded
with a deterministic signature. It does not authorize original-project writes or
strict_external relaxation.