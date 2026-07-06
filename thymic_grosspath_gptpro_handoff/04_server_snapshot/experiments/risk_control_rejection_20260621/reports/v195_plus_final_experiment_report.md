# v195+ External-Generalization Rejection Experiment Report

## Executive Summary

This experiment evaluates a v195+ rejection/release enhancement for the confirmed main workflow:

```text
v195 adaptive autocorrect / agreement-release review-compression flow
```

The experiment does not retrain the classifier. It only changes which v185/v195 review-or-reject cases may be safely released to automatic interpretation using agreement-release evidence from existing corrector outputs.

All candidate selection uses `old_data` and `third_batch` only. `strict_external` is used only as a frozen descriptive audit.

Main conclusion:

- The current v195 baseline on `strict_external` automatically interprets 52/108 cases, rejects/reviews 56/108, and has 0 automatic errors.
- The conservative v195+ Phase2 candidate automatically interprets 57/108 cases, rejects/reviews 51/108, and has 0 automatic errors.
- This is a modest strict_external improvement: 5 additional automatic cases and a rejection/review reduction from 51.85% to 47.22%.
- The gain is real under the frozen audit, but should not be called prospective external validation because `strict_external` has only 108 cases.

## Data Scope

| Domain | N |
|---|---:|
| old_data | 285 |
| third_batch | 306 |
| strict_external | 108 |
| total | 699 |

Main source files:

```text
outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv
outputs/grosspath_rc_v173_image_only_review_corrector_20260527/v173_corrector_case_outputs.csv
```

## Candidate Definitions

### v195 Baseline

Selected rule:

```text
v161_final_review_or_reject||tabular_logreg_c1||agreement_release_only||0.985
```

Release condition:

```text
adaptive_review == True
corrector_pred == final_pred
corrector_confidence >= 0.985
```

### Phase1 Exploratory Candidate

Definition:

```text
Union of all agreement-release candidates with:
- old_data action_error_n == 0
- third_batch action_error_n == 0
- old_data action_n + third_batch action_n > 0
```

Candidate count: 24.

Interpretation: exploratory upper bound. It gives the highest strict_external coverage, but includes thin-support rules.

### Phase2 Conservative Candidate

Definition:

```text
Union of agreement-release candidates with:
- old_data action_error_n == 0
- third_batch action_error_n == 0
- old_data action_n > 0
- third_batch action_n > 0
- old_data action_n + third_batch action_n >= 10
```

Candidate count: 7.

Interpretation: current conservative v195+ candidate. It removes thin-support rules and requires support in both known domains.

## Main Results

### Strict External

| Policy | Auto | Review/Reject | Auto Error | High-Risk FN |
|---|---:|---:|---:|---:|
| v185 baseline, no v195 release | 14/108 | 94/108 | 0 | 0 |
| v195 selected single rule | 52/108 | 56/108 | 0 | 0 |
| Phase1 exploratory union | 63/108 | 45/108 | 0 | 0 |
| Phase2 conservative union | 57/108 | 51/108 | 0 | 0 |

Compared with v195, Phase2 adds 5 automatic strict_external decisions and reduces review/reject by 4.63 percentage points.

### Known Domains

| Policy | old_data Auto/Error | third_batch Auto/Error |
|---|---:|---:|
| v185 baseline | 131/285, error 0 | 142/306, error 1 |
| v195 selected single rule | 210/285, error 0 | 185/306, error 1 |
| Phase1 exploratory union | 235/285, error 0 | 234/306, error 1 |
| Phase2 conservative union | 235/285, error 0 | 233/306, error 1 |

The one third_batch automatic error is inherited from the v185 baseline automatic decision. Phase1 and Phase2 introduce 0 new known-domain automatic errors relative to v195.

## Error Attribution

Phase2 decision gate passed:

```text
phase2_new_known_domain_auto_error_n = 0
```

Interpretation:

- The observed third_batch automatic error is not caused by v195+ releasing a reviewed case.
- It is already present when no v195 release is applied.
- Phase2 does not add a new automatic error on old_data or third_batch.

## Family and Threshold Sensitivity

Phase2 passed with caution.

| Variant | strict_external Auto | Auto Error | Gain vs v195 |
|---|---:|---:|---:|
| Phase2 full | 57/108 | 0 | +5 |
| Phase2 without tabular_hgb_l2 | 52/108 | 0 | 0 |
| Phase2 without DINO feature-set rules | 57/108 | 0 | +5 |
| Phase2 only v161 review policy | 57/108 | 0 | +5 |

Important caution:

The strict_external gain disappears when `tabular_hgb_l2` is removed. This does not make Phase2 unsafe, because that rule has strong old+third support, but it means the external gain is concentrated rather than broadly distributed across model families.

Threshold sensitivity:

| Variant | strict_external Auto | Auto Error |
|---|---:|---:|
| threshold == 0.985 | 52/108 | 0 |
| threshold >= 0.985 | 57/108 | 0 |
| threshold == 0.990 | 56/108 | 0 |
| threshold >= 0.990 | 56/108 | 0 |
| threshold >= 0.995 | 29/108 | 0 |

These rows are sensitivity evidence only. They are not used to reselect a new strict_external-tuned threshold.

## Subgroup and Casebook Findings

Phase2 newly releases 5 strict_external cases beyond v195:

```text
phase2_strict_incremental_n = 5
phase2_strict_incremental_error_n = 0
phase2_strict_incremental_high_risk_fn_n = 0
```

High-risk review-decrease warning:

- Phase2 reduces review in the old_data `label_idx=1` group by 2 cases compared with v195.
- No new automatic error or high-risk FN appears in that group.
- This should be disclosed as a subgroup safety check, not hidden.

## Reproducibility

Primary runner:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/scripts/v195_plus_reproducibility_runner.py
```

Runner result:

```text
passed = true
```

Validated checks:

- v195 strict_external auto 52/108, auto error 0.
- Phase1 strict_external auto 63/108, auto error 0.
- Phase2 strict_external auto 57/108, auto error 0.
- Candidate freeze passed.
- Error attribution passed.
- Family/threshold audit passed with caution.
- Subgroup/casebook audit passed.

## Generated Evidence Files

Candidate freeze:

```text
configs/v195_plus_candidate_manifest.json
reports/v195_plus_candidate_protocol.md
outputs/v195_plus_candidate_freeze_audit_report.json
```

Error attribution:

```text
outputs/v195_plus_error_attribution.csv
outputs/v195_plus_error_attribution_summary.csv
reports/v195_plus_error_attribution.md
```

Family/threshold audit:

```text
outputs/v195_plus_family_ablation.csv
outputs/v195_plus_threshold_sensitivity.csv
reports/v195_plus_family_threshold_audit.md
```

Subgroup/casebook audit:

```text
outputs/v195_plus_subgroup_audit.csv
outputs/v195_plus_strict_incremental_casebook.csv
reports/v195_plus_subgroup_casebook.md
```

Reproducibility:

```text
scripts/v195_plus_reproducibility_runner.py
outputs/v195_plus_reproducibility_report.json
reports/v195_plus_reproducibility_runner.md
```

## Limitations

1. `strict_external` has only 108 cases. Zero observed automatic errors does not prove the true error rate is zero.
2. `strict_external` is a frozen descriptive audit, not a prospective validation set.
3. No new model was trained. This work improves selective release/rejection logic only.
4. The strict_external gain is modest: Phase2 adds 5 automatic decisions compared with v195.
5. The Phase2 gain is concentrated in the `tabular_hgb_l2` rule family.
6. The original project code has not been integrated or modified.

## Recommendation

Do not replace the original v195 main workflow yet.

Recommended next state:

```text
Keep Phase2 as the conservative experimental v195+ candidate.
Package it as a sidecar flow if integration is requested.
Do not directly rewrite the original project pipeline until the user explicitly approves integration.
```

Phase1 should remain exploratory evidence only. Phase2 is the candidate worth carrying forward because it gives a small strict_external coverage gain without introducing observed automatic errors in the audited data, while maintaining clearer internal support requirements.
