# Stable Policy Feature Interface

Feature version: `risk_control_v1`.

## Required Runtime Columns

- `policy_feature_version`
- `domain`
- `case_id`
- `original_case_id`
- `task_l6_label`
- `label_idx`
- `final_pred`
- `prob_mean_core`
- `fold_id`
- `v195_review_or_reject`
- `v195_auto_decision`
- `hard_shift_gate`
- `model_disagreement_rate`
- `dino_fn_risk_max`
- `dino_fn_risk_missing`

## Domain Counts

- old_data: n=285
- strict_external: n=108
- third_batch: n=306

## Notes

`dino_fn_risk_max` is materialized here so downstream policy runners no longer need to read
the historical v191 raw artifact directly.