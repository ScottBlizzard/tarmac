# v135 Strict External Phase-1 Selective Rejection Audit

This audit re-runs phase-1 selective rejection on the correct v135 strict-external baseline
`main_prob`, `balanced_accuracy`, threshold `0.595`. It does not use v185/v201 post-processed
`final_pred` as the model prediction.

## Strict External Summary

| policy | margin_threshold | auto_decision_n | review_or_reject_n | auto_error_n | auto_high_risk_fn_n | auto_error_wilson95_high | system_if_review_corrected_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_all_auto_v135_main_prob | 0 | 108 | 0 | 38 | 25 | 0.445591 | 0.648148 |
| internal_selected_zero_auto_error | 0.404595 | 4 | 104 | 2 | 2 | 0.849961 | 0.981481 |
| internal_selected_zero_high_risk_fn | 0.404595 | 4 | 104 | 2 | 2 | 0.849961 | 0.981481 |
| strict_oracle_zero_auto_error | 0.481069 | 1 | 107 | 0 | 0 | 0.793451 | 1 |
| strict_oracle_zero_high_risk_fn | 0.481069 | 1 | 107 | 0 | 0 | 0.793451 | 1 |

## Selected On Internal Only

| constraint | selected_on | margin_threshold | internal_auto_decision_n | internal_auto_decision_rate | internal_auto_error_n | internal_auto_high_risk_fn_n | internal_auto_error_wilson95_high | selection_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero_auto_error | internal_oof_old_third | 0.404595 | 45 | 0.0761421 | 0 | 0 | 0.0786516 | selected |
| zero_high_risk_fn | internal_oof_old_third | 0.404595 | 45 | 0.0761421 | 0 | 0 | 0.0786516 | selected |
| wilson3_no_highrisk_fn | internal_oof_old_third |  | 0 | 0 | 0 | 0 |  | no_internal_candidate |

## Interpretation

- `internal_selected_*` rows are frozen strict-external evaluations after selecting the margin threshold on internal old+third only.
- `strict_oracle_*` rows are post-hoc strict-external display rows and must not be treated as deployable evidence.