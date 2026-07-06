# v195+ Candidate Freeze Protocol

This document freezes the candidate definitions for the v195+ rejection/release experiment.
It is generated from the reconstruction script and should be treated as the candidate contract for the next audits.

## Scope

- Main workflow: v195 adaptive autocorrect / agreement-release review-compression flow.
- Source base cases: `outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv`.
- Source corrector outputs: `outputs/grosspath_rc_v173_image_only_review_corrector_20260527/v173_corrector_case_outputs.csv`.
- Selection domains: `old_data`, `third_batch`.
- Frozen audit domain: `strict_external`.
- Strict external labels are not used for candidate selection.

## Frozen Candidates

- `phase1_all_safe_union`: exploratory upper-bound candidate. It keeps all agreement-release rules with zero action errors on both known domains and nonzero known-domain support.
- `phase2_min10_both_domain_union`: conservative main candidate. It further requires support in both known domains and at least 10 known-domain releases.

## Strict External Frozen Audit

| policy | domain | n | auto_n | auto_rate | review_n | review_rate | release_from_review_n | auto_error_n | auto_high_risk_fn_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v185_baseline_no_v195_release | strict_external | 108 | 14 | 0.12963 | 94 | 0.87037 | 0 | 0 | 0 |
| v195_selected_single_rule | strict_external | 108 | 52 | 0.481481 | 56 | 0.518519 | 38 | 0 | 0 |
| phase1_all_safe_union | strict_external | 108 | 63 | 0.583333 | 45 | 0.416667 | 49 | 0 | 0 |
| phase2_min10_both_domain_union | strict_external | 108 | 57 | 0.527778 | 51 | 0.472222 | 43 | 0 | 0 |

## Phase 2 Candidate Records

| candidate_id | old_data_action_n | third_batch_action_n | strict_external_action_n | old_data_action_error_n | third_batch_action_error_n | strict_external_action_error_n |
| --- | --- | --- | --- | --- | --- | --- |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.985 | 79 | 43 | 38 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.990 | 67 | 23 | 31 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_hgb_l2||agreement_release_only||0.990 | 61 | 56 | 33 | 0 | 0 | 0 |
| v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.995 | 44 | 13 | 15 | 0 | 0 | 0 |
| v118_review_or_control||tabular_logreg_c1||agreement_release_only||0.995 | 10 | 3 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca128_logreg_c01||agreement_release_only||0.985 | 4 | 8 | 0 | 0 | 0 | 0 |
| v161_final_review_or_reject||dino_pca128_logreg_c01||agreement_release_only||0.990 | 3 | 7 | 0 | 0 | 0 | 0 |

## Reproduction Checks

```json
[
  {
    "name": "phase1_all_safe_union",
    "checked": true,
    "passed": true,
    "current_n": 24,
    "existing_n": 24,
    "missing_from_current": [],
    "extra_in_current": []
  },
  {
    "name": "phase2_min10_both_domain_union",
    "checked": true,
    "passed": true,
    "current_n": 7,
    "existing_n": 7,
    "missing_from_current": [],
    "extra_in_current": []
  }
]
```

## Decision

PASS: manifest candidate ids reproduce existing Phase 1/2 outputs.