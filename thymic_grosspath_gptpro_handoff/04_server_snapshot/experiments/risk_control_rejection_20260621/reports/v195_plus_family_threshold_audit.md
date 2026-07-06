# v195+ Family and Threshold Audit

This audit tests whether the conservative Phase2 candidate depends on one fragile model family or one narrow threshold neighborhood.
Strict external is reported as a frozen audit only.

## Decision Gate

- v195 strict auto: `52`.
- Phase2 full strict auto: `57`.
- Without `tabular_hgb_l2` strict auto: `52`.
- Without DINO feature-set rules strict auto: `57`.
- Decision: PASS_WITH_CAUTION

## Family Ablation

| policy | candidate_n | release_case_n_all_domains | strict_auto_n | strict_review_n | strict_auto_error_n | strict_high_risk_fn_n | strict_auto_gain_vs_v195 | old_data_auto_error_n | third_batch_auto_error_n | known_domain_auto_error_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_full | 7 | 238 | 57 | 51 | 0 | 0 | 5 | 0 | 1 | 1 |
| phase2_without_tabular_hgb_l2 | 6 | 172 | 52 | 56 | 0 | 0 | 0 | 0 | 1 | 1 |
| phase2_without_dino_feature_set | 5 | 233 | 57 | 51 | 0 | 0 | 5 | 0 | 1 | 1 |
| phase2_only_v161_review_policy | 6 | 237 | 57 | 51 | 0 | 0 | 5 | 0 | 1 | 1 |
| phase2_only_tabular_feature_set | 5 | 233 | 57 | 51 | 0 | 0 | 5 | 0 | 1 | 1 |
| phase2_only_dino_feature_set | 2 | 12 | 14 | 94 | 0 | 0 | -38 | 0 | 1 | 1 |

## Threshold Sensitivity

| policy | candidate_n | release_case_n_all_domains | strict_auto_n | strict_review_n | strict_auto_error_n | strict_high_risk_fn_n | strict_auto_gain_vs_v195 | old_data_auto_error_n | third_batch_auto_error_n | known_domain_auto_error_n | threshold_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_exact_threshold_0.985 | 2 | 171 | 52 | 56 | 0 | 0 | 0 | 0 | 1 | 1 | threshold == 0.985 |
| phase2_threshold_ge_0.985 | 7 | 238 | 57 | 51 | 0 | 0 | 5 | 0 | 1 | 1 | threshold >= 0.985 |
| phase2_exact_threshold_0.990 | 3 | 221 | 56 | 52 | 0 | 0 | 4 | 0 | 1 | 1 | threshold == 0.990 |
| phase2_threshold_ge_0.990 | 5 | 222 | 56 | 52 | 0 | 0 | 4 | 0 | 1 | 1 | threshold >= 0.990 |
| phase2_exact_threshold_0.995 | 2 | 74 | 29 | 79 | 0 | 0 | -23 | 0 | 1 | 1 | threshold == 0.995 |
| phase2_threshold_ge_0.995 | 2 | 74 | 29 | 79 | 0 | 0 | -23 | 0 | 1 | 1 | threshold >= 0.995 |

## Interpretation

- If removing one family collapses the strict_external gain below v195, Phase2 should be treated as fragile.
- Here, removing `tabular_hgb_l2` removes the strict_external gain but does not make the policy worse than v195; this is a concentration warning, not a hard failure.
- Threshold rows are sensitivity evidence, not a new selection step.