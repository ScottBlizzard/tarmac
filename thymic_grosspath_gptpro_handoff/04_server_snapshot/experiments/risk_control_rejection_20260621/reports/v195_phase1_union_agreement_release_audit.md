# v195 Phase-1 Union Agreement Release Audit

This audit keeps v195 as the main workflow. It reconstructs agreement-release candidates from
v185 base decisions and v173 correctors, selects candidates using old+third only, then reports
strict_external as a frozen audit.

## Evaluation

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
| v195_union_internal_zero_supported_agreement_rules | old_data | 285 | 235 | 0.824561 | 50 | 0.175439 | 104 | 0 | 0 |
| v195_union_internal_zero_supported_agreement_rules | third_batch | 306 | 234 | 0.764706 | 72 | 0.235294 | 92 | 1 | 1 |
| v195_union_internal_zero_supported_agreement_rules | strict_external | 108 | 63 | 0.583333 | 45 | 0.416667 | 49 | 0 | 0 |
| v195_union_internal_zero_supported_agreement_rules | all | 699 | 532 | 0.761087 | 167 | 0.238913 | 245 | 1 | 1 |

## Top Internal-Zero Candidates

| candidate_id | old_data_action_n | third_batch_action_n | strict_external_action_n | old_data_action_error_n | third_batch_action_error_n | strict_external_action_error_n |
| --- | --- | --- | --- | --- | --- | --- |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.985 | 79 | 43 | 38 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.990 | 67 | 23 | 31 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.995 | 44 | 13 | 15 | 0 | 0 | 0 |
| v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.995 | 3 | 0 | 6 | 0 | 0 | 0 |
| v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 | 0 | 2 | 5 | 0 | 0 | 0 |
| v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.975 | 3 | 6 | 3 | 0 | 0 | 0 |
| v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.985 | 2 | 5 | 2 | 0 | 0 | 0 |
| v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.990 | 1 | 3 | 2 | 0 | 0 | 0 |
| v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.995 | 0 | 2 | 2 | 0 | 0 | 0 |
| v118_review_or_control||tabular_logreg_c1||agreement_release_only||0.995 | 10 | 3 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca128_logreg_c01||agreement_release_only||0.985 | 4 | 8 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca128_logreg_c01||agreement_release_only||0.990 | 3 | 7 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca128_logreg_c01||agreement_release_only||0.995 | 3 | 5 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca64_logreg_c03||agreement_release_only||0.900 | 1 | 2 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 | 1 | 2 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca64_logreg_c03||agreement_release_only||0.925 | 0 | 2 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_dino_pca64_logreg_c03||agreement_release_only||0.950 | 0 | 2 | 0 | 0 | 0 | 0 |
| v118_review_or_control||tabular_dino_extra_trees_d4||agreement_release_only||0.700 | 0 | 1 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca64_logreg_c03||agreement_release_only||0.950 | 0 | 1 | 0 | 0 | 0 | 0 |

## Interpretation

- The union policy is selected without strict_external labels: candidates must have zero action errors on old_data and third_batch and nonzero internal support.
- On strict_external, this increases auto decisions from 52 to 63 and reduces review/reject from 56 to 45, with zero observed auto errors.
- This remains an experimental candidate and should be stress-audited before replacing the single-rule v195 deployment policy.