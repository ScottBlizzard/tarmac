# v195 Phase-2 Full Stability Audit

This completes the planned Phase 2 audit for the recommended v195+ candidate.

## Recommended Evaluation

| policy | domain | n | auto_n | auto_rate | review_n | review_rate | release_from_review_n | auto_error_n | auto_high_risk_fn_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_union_min10_both_domains | old_data | 285 | 235 | 0.824561 | 50 | 0.175439 | 104 | 0 | 0 |
| phase2_union_min10_both_domains | third_batch | 306 | 233 | 0.761438 | 73 | 0.238562 | 91 | 1 | 1 |
| phase2_union_min10_both_domains | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |
| phase2_union_min10_both_domains | all | 699 | 525 | 0.751073 | 174 | 0.248927 | 238 | 1 | 1 |

## Candidate Ablation

| candidate_id | old_data_action_n | third_batch_action_n | strict_external_action_n | strict_unique_auto_loss_if_removed | strict_only_auto_n | strict_only_auto_error_n | strict_only_high_risk_fn_n | without_strict_auto_n | without_strict_auto_error_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | 5 | 47 | 0 | 0 | 52 | 0 |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.985 | 79 | 43 | 38 | 1 | 52 | 0 | 0 | 56 | 0 |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.990 | 67 | 23 | 31 | 0 | 45 | 0 | 0 | 57 | 0 |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.995 | 44 | 13 | 15 | 0 | 29 | 0 | 0 | 57 | 0 |
| v118_review_or_control||tabular_logreg_c1||agreement_release_only||0.995 | 10 | 3 | 0 | 0 | 14 | 0 | 0 | 57 | 0 |
| v161_final_review_or_reject||dino_pca128_logreg_c01||agreement_release_only||0.985 | 4 | 8 | 0 | 0 | 14 | 0 | 0 | 57 | 0 |
| v161_final_review_or_reject||dino_pca128_logreg_c01||agreement_release_only||0.990 | 3 | 7 | 0 | 0 | 14 | 0 | 0 | 57 | 0 |

## Leave-Domain Stress

| policy | domain | n | auto_n | auto_rate | review_n | review_rate | release_from_review_n | auto_error_n | auto_high_risk_fn_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| select_old_data_min10_test_third_batch | third_batch | 306 | 305 | 0.996732 | 1 | 0.00326797 | 163 | 9 | 7 |
| select_third_batch_min10_test_old_data | old_data | 285 | 234 | 0.821053 | 51 | 0.178947 | 103 | 0 | 0 |

## Strict Incremental Casebook

| domain | case_id | task_l6_label | label_idx | final_pred | prob_mean_core | would_be_error | would_be_high_risk_fn | candidate_n | candidate_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_external | doctor_ext_001_1700034 | AB | 0 | 0 | 0.57672 | False | False | 70 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.500;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.600;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.700;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.800;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.850;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.900;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.925;v118_review_or_control||dino_pca64_logreg_c03||agreement_release_only||0.500 |
| strict_external | doctor_ext_023_2404139 | AB | 0 | 0 | 0.612059 | False | False | 42 | v118_review_or_control||tabular_dino_extra_trees_d4||agreement_release_only||0.500;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.500;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.600;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.700;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.800;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.850;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.900;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.925 |
| strict_external | doctor_ext_035_2613517 | AB | 0 | 0 | 0.603581 | False | False | 40 | v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.500;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.600;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.700;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.800;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.850;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.900;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.925;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.950 |
| strict_external | doctor_ext_053_2441583 | B1 | 0 | 0 | 0.422352 | False | False | 50 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.500;v118_review_or_control||dino_pca64_logreg_c03||agreement_release_only||0.500;v118_review_or_control||dino_pca64_logreg_c03||agreement_release_only||0.600;v118_review_or_control||tabular_dino_extra_trees_d4||agreement_release_only||0.500;v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.500;v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.600;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.500;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.600 |
| strict_external | doctor_ext_054_2607215 | B1 | 0 | 0 | 0.49044 | False | False | 50 | v118_review_or_control||tabular_dino_extra_trees_d4||agreement_release_only||0.500;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.500;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.600;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.700;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.800;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.850;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.900;v118_review_or_control||tabular_hgb_l2||agreement_release_only||0.925 |

## High-Level Subgroup Issues

| domain | grouping | n | auto_n | review_n | auto_error_n | auto_high_risk_fn_n | task_l6_label | label_idx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| third_batch | domain | 306 | 233 | 73 | 1 | 1 |  |  |
| third_batch | domain+task_l6_label | 53 | 41 | 12 | 1 | 1 | TC |  |
| third_batch | domain+label_idx | 82 | 57 | 25 | 1 | 1 |  | 1 |