# v195 Phase-2 Stability Pruning Audit

Phase 2 prunes the phase-1 union into more conservative candidate policies. Selection uses only
old_data and third_batch candidate support/error evidence; strict_external remains frozen audit output.

## Strict External

| policy | domain | n | auto_n | auto_rate | review_n | review_rate | release_from_review_n | auto_error_n | auto_high_risk_fn_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v185_baseline_no_v195_release | strict_external | 108 | 14 | 0.12963 | 94 | 0.87037 | 0 | 0 | 0 |
| v195_selected_single_rule | strict_external | 108 | 52 | 0.481481 | 56 | 0.518519 | 38 | 0 | 0 |
| phase2_union_all_safe_supported | strict_external | 108 | 63 | 0.583333 | 45 | 0.416667 | 49 | 0 | 0 |
| phase2_union_both_known_domains | strict_external | 108 | 60 | 0.555556 | 48 | 0.444444 | 46 | 0 | 0 |
| phase2_union_min10_both_domains | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |
| phase2_union_min20_both_domains | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |
| phase2_union_top2_internal_support | strict_external | 108 | 52 | 0.481481 | 56 | 0.518519 | 38 | 0 | 0 |
| phase2_union_top3_internal_support | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |

## All Domains

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
| phase2_union_all_safe_supported | old_data | 285 | 235 | 0.824561 | 50 | 0.175439 | 104 | 0 | 0 |
| phase2_union_all_safe_supported | third_batch | 306 | 234 | 0.764706 | 72 | 0.235294 | 92 | 1 | 1 |
| phase2_union_all_safe_supported | strict_external | 108 | 63 | 0.583333 | 45 | 0.416667 | 49 | 0 | 0 |
| phase2_union_all_safe_supported | all | 699 | 532 | 0.761087 | 167 | 0.238913 | 245 | 1 | 1 |
| phase2_union_both_known_domains | old_data | 285 | 235 | 0.824561 | 50 | 0.175439 | 104 | 0 | 0 |
| phase2_union_both_known_domains | third_batch | 306 | 233 | 0.761438 | 73 | 0.238562 | 91 | 1 | 1 |
| phase2_union_both_known_domains | strict_external | 108 | 60 | 0.555556 | 48 | 0.444444 | 46 | 0 | 0 |
| phase2_union_both_known_domains | all | 699 | 528 | 0.755365 | 171 | 0.244635 | 241 | 1 | 1 |
| phase2_union_min10_both_domains | old_data | 285 | 235 | 0.824561 | 50 | 0.175439 | 104 | 0 | 0 |
| phase2_union_min10_both_domains | third_batch | 306 | 233 | 0.761438 | 73 | 0.238562 | 91 | 1 | 1 |
| phase2_union_min10_both_domains | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |
| phase2_union_min10_both_domains | all | 699 | 525 | 0.751073 | 174 | 0.248927 | 238 | 1 | 1 |
| phase2_union_min20_both_domains | old_data | 285 | 234 | 0.821053 | 51 | 0.178947 | 103 | 0 | 0 |
| phase2_union_min20_both_domains | third_batch | 306 | 228 | 0.745098 | 78 | 0.254902 | 86 | 1 | 1 |
| phase2_union_min20_both_domains | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |
| phase2_union_min20_both_domains | all | 699 | 519 | 0.742489 | 180 | 0.257511 | 232 | 1 | 1 |
| phase2_union_top2_internal_support | old_data | 285 | 210 | 0.736842 | 75 | 0.263158 | 79 | 0 | 0 |
| phase2_union_top2_internal_support | third_batch | 306 | 185 | 0.604575 | 121 | 0.395425 | 43 | 1 | 1 |
| phase2_union_top2_internal_support | strict_external | 108 | 52 | 0.481481 | 56 | 0.518519 | 38 | 0 | 0 |
| phase2_union_top2_internal_support | all | 699 | 447 | 0.639485 | 252 | 0.360515 | 160 | 1 | 1 |
| phase2_union_top3_internal_support | old_data | 285 | 234 | 0.821053 | 51 | 0.178947 | 103 | 0 | 0 |
| phase2_union_top3_internal_support | third_batch | 306 | 228 | 0.745098 | 78 | 0.254902 | 86 | 1 | 1 |
| phase2_union_top3_internal_support | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |
| phase2_union_top3_internal_support | all | 699 | 519 | 0.742489 | 180 | 0.257511 | 232 | 1 | 1 |

## Recommended Candidate

```json
{
  "policy": "phase2_union_min10_both_domains",
  "strict_auto_n": 57,
  "strict_review_n": 51,
  "known_auto_error_n": 1,
  "known_high_risk_fn_n": 1,
  "known_review_n": 123
}
```

## Incremental Casebook

| domain | case_id | task_l6_label | label_idx | final_pred | prob_mean_core | was_v195_selected_release | phase2_min10_both_release | releasing_candidate_n | top_releasing_candidate | top_candidate_old_action_n | top_candidate_third_action_n | top_candidate_strict_action_n | would_be_error | would_be_high_risk_fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_data | batch1_2115336 | AB | 0 | 0 | 0.381076 | False | True | 65 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch1_2203368 | B1 | 0 | 0 | 0.400795 | False | True | 67 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch1_2208476 | B1 | 0 | 0 | 0.551788 | False | True | 42 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch1_2211578 | AB | 0 | 0 | 0.452929 | False | True | 66 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch1_2211635 | B1 | 0 | 0 | 0.332238 | False | True | 40 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch1_2411128 | B1 | 0 | 0 | 0.331035 | False | True | 63 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch1_2413768 | A | 0 | 0 | 0.425738 | False | True | 50 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch1_2515960 | A | 0 | 0 | 0.453143 | False | True | 58 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2112749 | AB | 0 | 0 | 0.477721 | False | True | 52 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2113744 | AB | 0 | 0 | 0.333236 | False | True | 40 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2114597 | AB | 0 | 0 | 0.421464 | False | True | 48 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2116007 | AB | 0 | 0 | 0.379373 | False | True | 42 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2201508 | TC | 1 | 1 | 0.610323 | False | True | 43 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2203717 | AB | 0 | 0 | 0.365647 | False | True | 63 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2407706 | B1 | 0 | 0 | 0.350886 | False | True | 48 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2409535 | B1 | 0 | 0 | 0.44819 | False | True | 49 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2418741 | A | 0 | 0 | 0.402075 | False | True | 41 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2419875 | B1 | 0 | 0 | 0.403192 | False | True | 86 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2420745 | B2 | 1 | 1 | 0.66646 | False | True | 44 | v118_review_or_control||tabular_logreg_c1||agreement_release_only||0.995 | 10 | 3 | 0 | False | False |
| old_data | batch2_2423827 | A | 0 | 0 | 0.412264 | False | True | 62 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2501880 | B1 | 0 | 0 | 0.332415 | False | True | 58 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2505099 | A | 0 | 0 | 0.415024 | False | True | 68 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2517760 | A | 0 | 0 | 0.36684 | False | True | 41 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2529294 | B1 | 0 | 0 | 0.385466 | False | True | 49 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| old_data | batch2_2601895 | B1 | 0 | 0 | 0.335169 | False | True | 40 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| strict_external | doctor_ext_001_1700034 | AB | 0 | 0 | 0.57672 | False | True | 70 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| strict_external | doctor_ext_023_2404139 | AB | 0 | 0 | 0.612059 | False | True | 42 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| strict_external | doctor_ext_035_2613517 | AB | 0 | 0 | 0.603581 | False | True | 40 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| strict_external | doctor_ext_053_2441583 | B1 | 0 | 0 | 0.422352 | False | True | 50 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |
| strict_external | doctor_ext_054_2607215 | B1 | 0 | 0 | 0.49044 | False | True | 50 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | False | False |

## Interpretation

- The all-safe union is the most aggressive candidate.
- The min10-both-domains policy requires each retained rule to release at least one old_data case, at least one third_batch case, and at least ten known-domain cases in total.
- This keeps the strict_external gain while filtering out very thin-support rules.