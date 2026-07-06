# v195+ Minimal Hook Blueprint

This is a read-only integration blueprint. It does not modify the original project.

## Result

- passed: `True`
- recommended_strategy: `add_new_versioned_flow_script_only_after_explicit_integration_approval`
- future_new_flow: `scripts/run_grosspath_rc_v203_v195_plus_sidecar_policy_20260623.py`
- future_output_dir: `outputs/grosspath_rc_v203_v195_plus_sidecar_policy_20260623`
- original_project_code_modified: `False`
- existing_v195_script_should_be_modified: `False`

## Blueprint

| Seq | Hook | Future Target | Change Type | Minimal Action |
|---:|---|---|---|---|
| 1 | `add_new_v195_plus_flow_script` | `scripts/run_grosspath_rc_v203_v195_plus_sidecar_policy_20260623.py` | `add_new_file_only` | create a new versioned flow script that invokes the sidecar runner after v185/v173 outputs exist |
| 2 | `optional_runtime_decision_consumer` | `downstream consumer not yet identified` | `manual_review_required` | only after approval, consume label-free v195_plus runtime decision columns instead of audit decisions |
| 3 | `pre_merge_reproducibility_gate` | `experiment-side command` | `review_gate` | run sidecar audit, package smoke audit, and preintegration gap audit before any future patch is applied |

## Patch Draft

- `experiments/risk_control_rejection_20260621/patch_rehearsal/v195_plus/v195_plus_new_flow_patch_draft.patch.txt`

## Interpretation

The safest future integration is a new versioned flow script, not editing v195.
The downstream consumer remains undefined and should be reviewed separately.