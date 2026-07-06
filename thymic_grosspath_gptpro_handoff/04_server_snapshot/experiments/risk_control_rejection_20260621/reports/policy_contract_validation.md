# Policy Contract Validation

## Summary

| Input | Rows | Passed | Missing Required | Unexpected Versions | Forbidden Columns | Audit-only Present |
| --- | ---: | --- | --- | --- | --- | --- |
| experiments/risk_control_rejection_20260621/outputs/stable_policy_features.csv | 699 | True | - | - | - | label_idx, task_l6_label, fold_id |
| experiments/risk_control_rejection_20260621/outputs/adapter_policy_features.csv | 699 | True | - | - | - | label_idx, task_l6_label, fold_id |

Audit-only columns are allowed in experiment evaluation tables, but they are not part of
the release selector. Forbidden selector columns must not appear in the stable runtime
feature interface.