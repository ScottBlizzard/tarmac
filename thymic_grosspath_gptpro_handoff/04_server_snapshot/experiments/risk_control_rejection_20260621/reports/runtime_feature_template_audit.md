# Runtime Feature Template Audit

## Purpose

This audit checks that the deployment feature template exactly matches the policy
contract runtime columns and does not include labels, fold metadata, or selector error
columns.

## Result

- passed: True
- template rows: 1
- template columns: 11
- order matches contract: True
- missing required columns: -
- extra columns: -
- audit-only columns present: -
- forbidden selector columns present: -
- unexpected feature versions: -

## Runtime Columns

| Position | Column |
| ---: | --- |
| 1 | `policy_feature_version` |
| 2 | `domain` |
| 3 | `case_id` |
| 4 | `v195_review_or_reject` |
| 5 | `v195_auto_decision` |
| 6 | `hard_shift_gate` |
| 7 | `model_disagreement_rate` |
| 8 | `dino_fn_risk_max` |
| 9 | `dino_fn_risk_missing` |
| 10 | `final_pred` |
| 11 | `prob_mean_core` |