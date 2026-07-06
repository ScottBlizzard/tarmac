# v195+ Error Attribution

This audit separates automatic errors inherited from the v185/v195 baseline from errors newly introduced by v195+ release policies.

## Decision Gate

- Phase2 newly introduced known-domain auto errors versus v195: `0`.
- Decision: PASS

## Policy-Domain Error Summary

| policy | domain | auto_error_n | auto_high_risk_fn_n | release_from_review_error_n | new_error_vs_v195_n |
| --- | --- | --- | --- | --- | --- |
| v185 | old_data | 0 | 0 | 0 | 0 |
| v185 | strict_external | 0 | 0 | 0 | 0 |
| v185 | third_batch | 1 | 1 | 0 | 0 |
| v195 | old_data | 0 | 0 | 0 | 0 |
| v195 | strict_external | 0 | 0 | 0 | 0 |
| v195 | third_batch | 1 | 1 | 0 | 0 |
| phase1 | old_data | 0 | 0 | 0 | 0 |
| phase1 | strict_external | 0 | 0 | 0 | 0 |
| phase1 | third_batch | 1 | 1 | 0 | 0 |
| phase2 | old_data | 0 | 0 | 0 | 0 |
| phase2 | strict_external | 0 | 0 | 0 | 0 |
| phase2 | third_batch | 1 | 1 | 0 | 0 |

## Full Policy Evaluation

| policy | domain | n | auto_n | auto_rate | review_n | review_rate | release_from_review_n | auto_error_n | auto_high_risk_fn_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v185_baseline_no_v195_release | old_data | 285 | 131 | 0.459649 | 154 | 0.540351 | 0 | 0 | 0 |
| v185_baseline_no_v195_release | third_batch | 306 | 142 | 0.464052 | 164 | 0.535948 | 0 | 1 | 1 |
| v185_baseline_no_v195_release | strict_external | 108 | 14 | 0.12963 | 94 | 0.87037 | 0 | 0 | 0 |
| v185_baseline_no_v195_release | all | 699 | 287 | 0.410587 | 412 | 0.589413 | 0 | 1 | 1 |
| v195_selected_single_rule | old_data | 285 | 210 | 0.736842 | 75 | 0.263158 | 79 | 0 | 0 |
| v195_selected_single_rule | third_batch | 306 | 185 | 0.604575 | 121 | 0.395425 | 43 | 1 | 1 |
| v195_selected_single_rule | strict_external | 108 | 52 | 0.481481 | 56 | 0.518519 | 38 | 0 | 0 |
| v195_selected_single_rule | all | 699 | 447 | 0.639485 | 252 | 0.360515 | 160 | 1 | 1 |
| phase1_all_safe_union | old_data | 285 | 235 | 0.824561 | 50 | 0.175439 | 104 | 0 | 0 |
| phase1_all_safe_union | third_batch | 306 | 234 | 0.764706 | 72 | 0.235294 | 92 | 1 | 1 |
| phase1_all_safe_union | strict_external | 108 | 63 | 0.583333 | 45 | 0.416667 | 49 | 0 | 0 |
| phase1_all_safe_union | all | 699 | 532 | 0.761087 | 167 | 0.238913 | 245 | 1 | 1 |
| phase2_min10_both_domain_union | old_data | 285 | 235 | 0.824561 | 50 | 0.175439 | 104 | 0 | 0 |
| phase2_min10_both_domain_union | third_batch | 306 | 233 | 0.761438 | 73 | 0.238562 | 91 | 1 | 1 |
| phase2_min10_both_domain_union | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |
| phase2_min10_both_domain_union | all | 699 | 525 | 0.751073 | 174 | 0.248927 | 238 | 1 | 1 |

## Auto-Error Casebook

| domain | case_id | original_case_id | task_l6_label | label_idx | final_pred | prob_mean_core | adaptive_review | v185_auto | v185_error | v185_high_risk_fn | v185_release_from_review | v195_auto | v195_error | v195_high_risk_fn | v195_release_from_review | phase1_auto | phase1_error | phase1_high_risk_fn | phase1_release_from_review | phase2_auto | phase2_error | phase2_high_risk_fn | phase2_release_from_review | raw_prediction_error | raw_high_risk_fn | error_origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| third_batch | third_TC_2516531 | 2516531 | TC | 1 | 0 | 0.427882 | False | True | True | True | False | True | True | True | False | True | True | True | False | True | True | True | False | True | True | inherited_from_v185_base_auto |

## Interpretation

- If an error appears under `v185`, it is inherited from the base automatic decision and not introduced by v195+ release.
- If an error first appears under `phase2`, Phase2 should not be promoted without further pruning.
- Strict external remains a frozen audit set, not a selection source.