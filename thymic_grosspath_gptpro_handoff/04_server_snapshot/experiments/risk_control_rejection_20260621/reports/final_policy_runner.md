# Final Policy Runner

## Rule

- Primary release score: `model_disagreement_rate` ascending.
- Secondary release score: `dino_fn_risk_max` ascending.
- Residual deterministic tie-break: `case_id` ascending.
- `case_only_exploratory` ignores severe shift for frozen display.
- `shift_aware_deployable` falls back to v195 for `severe_unknown_shift` batches.

## Summary

| Mode | Scope | Auto | Review | Released | Released Errors | Auto Errors | High-risk FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| case_only_exploratory | old_data | 221 | 64 | 11 | 0 | 0 | 0 |
| case_only_exploratory | third_batch | 203 | 103 | 18 | 0 | 1 | 1 |
| case_only_exploratory | strict_external | 60 | 48 | 8 | 0 | 0 | 0 |
| case_only_exploratory | all_domains | 484 | 215 | 37 | 0 | 1 | 1 |
| shift_aware_deployable | old_data | 221 | 64 | 11 | 0 | 0 | 0 |
| shift_aware_deployable | third_batch | 203 | 103 | 18 | 0 | 1 | 1 |
| shift_aware_deployable | strict_external | 52 | 56 | 0 | 0 | 0 | 0 |
| shift_aware_deployable | all_domains | 476 | 223 | 29 | 0 | 1 | 1 |