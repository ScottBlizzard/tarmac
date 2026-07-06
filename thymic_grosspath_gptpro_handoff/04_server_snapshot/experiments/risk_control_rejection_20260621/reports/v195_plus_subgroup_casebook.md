# v195+ Subgroup and Casebook Audit

This audit summarizes how Phase1 and Phase2 change automatic interpretation by domain and label, and lists strict_external cases newly released beyond v195.

## Decision Gate

- Phase2 strict incremental errors: `0`.
- Phase2 strict incremental high-risk FN: `0`.
- Decision: PASS

## Strict Incremental Label Summary

| policy | task_l6_label | incremental_n | error_n | high_risk_fn_n | min_known_support_n | max_known_support_n |
| --- | --- | --- | --- | --- | --- | --- |
| phase1_all_safe_union | AB | 5 | 0 | 0 | 2 | 117 |
| phase1_all_safe_union | B1 | 2 | 0 | 0 | 117 | 117 |
| phase1_all_safe_union | B2 | 1 | 0 | 0 | 2 | 2 |
| phase1_all_safe_union | B2_B3_mixed | 1 | 0 | 0 | 9 | 9 |
| phase1_all_safe_union | B3 | 1 | 0 | 0 | 2 | 9 |
| phase1_all_safe_union | TC | 1 | 0 | 0 | 2 | 9 |
| phase2_min10_both_domain_union | AB | 3 | 0 | 0 | 117 | 117 |
| phase2_min10_both_domain_union | B1 | 2 | 0 | 0 | 117 | 117 |

## High-Risk Groups With Review Decrease

| policy | grouping | group_key | n | auto_n | review_n | release_from_review_n | auto_error_n | auto_high_risk_fn_n | domain | task_l6_label | label_idx | v195_auto_n | v195_review_n | v195_auto_error_n | v195_auto_high_risk_fn_n | auto_delta_vs_v195 | review_delta_vs_v195 | auto_error_delta_vs_v195 | auto_high_risk_fn_delta_vs_v195 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1_all_safe_union | domain_label_idx | old_data|1 | 141 | 115 | 26 | 36 | 0 | 0 | old_data |  | 1 | 113 | 28 | 0 | 0 | 2 | -2 | 0 | 0 |
| phase1_all_safe_union | domain_label_idx | strict_external|1 | 47 | 17 | 30 | 13 | 0 | 0 | strict_external |  | 1 | 13 | 34 | 0 | 0 | 4 | -4 | 0 | 0 |
| phase2_min10_both_domain_union | domain_label_idx | old_data|1 | 141 | 115 | 26 | 36 | 0 | 0 | old_data |  | 1 | 113 | 28 | 0 | 0 | 2 | -2 | 0 | 0 |

## Strict Incremental Casebook

| policy | domain | case_id | original_case_id | task_l6_label | label_idx | final_pred | prob_mean_core | would_be_error | would_be_high_risk_fn | releasing_candidate_n | max_known_support_n | min_known_support_n | top_releasing_candidate | candidate_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1_all_safe_union | strict_external | doctor_ext_001_1700034 | 1700034 | AB | 0 | 0 | 0.57672 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase1_all_safe_union | strict_external | doctor_ext_022_2403653 | 2403653 | AB | 0 | 0 | 0.248177 | False | False | 1 | 2 | 2 | v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 | v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 |
| phase1_all_safe_union | strict_external | doctor_ext_023_2404139 | 2404139 | AB | 0 | 0 | 0.612059 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase1_all_safe_union | strict_external | doctor_ext_026_2444000 | 2444000 | AB | 0 | 0 | 0.076981 | False | False | 1 | 2 | 2 | v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 | v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 |
| phase1_all_safe_union | strict_external | doctor_ext_035_2613517 | 2613517 | AB | 0 | 0 | 0.603581 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase1_all_safe_union | strict_external | doctor_ext_053_2441583 | 2441583 | B1 | 0 | 0 | 0.422352 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase1_all_safe_union | strict_external | doctor_ext_054_2607215 | 2607215 | B1 | 0 | 0 | 0.49044 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase1_all_safe_union | strict_external | doctor_ext_089_2316953 | 2316953 | B2_B3_mixed | 1 | 1 | 0.715148 | False | False | 1 | 9 | 9 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.975 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.975 |
| phase1_all_safe_union | strict_external | doctor_ext_098_2025136 | 2025136 | B2 | 1 | 1 | 0.924162 | False | False | 1 | 2 | 2 | v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 | v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 |
| phase1_all_safe_union | strict_external | doctor_ext_100_1913309 | 1913309 | B3 | 1 | 1 | 0.871249 | False | False | 5 | 9 | 2 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.975 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.975;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.985;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.990;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.995;v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 |
| phase1_all_safe_union | strict_external | doctor_ext_106_2023767 | 2023767 | TC | 1 | 1 | 0.916585 | False | False | 5 | 9 | 2 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.975 | v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.975;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.985;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.990;v118_review_or_control||dino_pca128_logreg_c01||agreement_release_only||0.995;v118_review_or_control||tabular_dino_pca64_logreg_c03||agreement_release_only||0.925 |
| phase2_min10_both_domain_union | strict_external | doctor_ext_001_1700034 | 1700034 | AB | 0 | 0 | 0.57672 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase2_min10_both_domain_union | strict_external | doctor_ext_023_2404139 | 2404139 | AB | 0 | 0 | 0.612059 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase2_min10_both_domain_union | strict_external | doctor_ext_035_2613517 | 2613517 | AB | 0 | 0 | 0.603581 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase2_min10_both_domain_union | strict_external | doctor_ext_053_2441583 | 2441583 | B1 | 0 | 0 | 0.422352 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |
| phase2_min10_both_domain_union | strict_external | doctor_ext_054_2607215 | 2607215 | B1 | 0 | 0 | 0.49044 | False | False | 1 | 117 | 117 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 |

## Interpretation

- Phase2 newly released strict_external cases are compared against v195, not against v185.
- A high-risk review decrease is not automatically an error; it flags groups where the final report must be explicit about safety evidence.
- If Phase2 had newly released a high-risk FN with weak internal support, it would fail this gate.