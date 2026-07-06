# v195+ External-Generalization Rejection Completion Plan

## Purpose

This plan continues the current external-generalization work on the confirmed main workflow:

```text
main workflow = v195 adaptive autocorrect / agreement-release review-compression flow
```

The goal is not to retrain the classifier and not to prove raw model accuracy improved. The goal is to strengthen the v195 rejection/review policy so that, on strict external data, more cases can be automatically interpreted while keeping observed automatic error risk no worse than v195.

All work must remain isolated under:

```text
experiments/risk_control_rejection_20260621/
```

Do not modify original project scripts or outputs unless the user explicitly authorizes integration.

## Current Baseline and Candidate State

### Confirmed v195 Baseline

Source workflow:

```text
outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv
outputs/grosspath_rc_v173_image_only_review_corrector_20260527/v173_corrector_case_outputs.csv
outputs/grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527/
```

Strict external baseline:

| Policy | Auto | Review/Reject | Auto Error | High-Risk FN |
|---|---:|---:|---:|---:|
| v195 selected single rule | 52/108 | 56/108 | 0 | 0 |

The v195 selected rule is:

```text
v161_final_review_or_reject
tabular_logreg_c1
agreement_release_only
corrector_confidence >= 0.985
corrector_pred == final_pred
```

### Phase 1 Exploratory Candidate

Definition:

```text
Union of all agreement-release candidates with:
- old_data action_error_n = 0
- third_batch action_error_n = 0
- old_data + third_batch internal support > 0
```

Strict external result:

| Policy | Auto | Review/Reject | Auto Error | High-Risk FN |
|---|---:|---:|---:|---:|
| Phase 1 all-safe union | 63/108 | 45/108 | 0 | 0 |

Interpretation:

This is the exploratory upper-bound candidate. It improves coverage most, but includes thin-support rules. It should not be treated as the main candidate without stronger stability evidence.

### Phase 2 Conservative Candidate

Definition:

```text
Union of agreement-release candidates with:
- old_data action_error_n = 0
- third_batch action_error_n = 0
- old_data action_n > 0
- third_batch action_n > 0
- old_data action_n + third_batch action_n >= 10
```

Strict external result:

| Policy | Auto | Review/Reject | Auto Error | High-Risk FN |
|---|---:|---:|---:|---:|
| Phase 2 min10 both-domain union | 57/108 | 51/108 | 0 | 0 |

Interpretation:

This is the current conservative v195+ candidate. It gives a smaller improvement than Phase 1, but it filters out thin-support rules and is better suited as the main candidate if the remaining audits pass.

## Key Risk Found So Far

Single-domain rule selection is unsafe.

Observed leave-domain stress:

| Selection | Test Domain | Auto | Review/Reject | Auto Error | High-Risk FN |
|---|---|---:|---:|---:|---:|
| select on old_data only | third_batch | 305/306 | 1/306 | 9 | 7 |
| select on third_batch only | old_data | 234/285 | 51/285 | 0 | 0 |

Implication:

Do not relax candidate selection to single-domain evidence. Any deployable-style v195+ candidate must require support from both known domains.

## Non-Negotiable Selection Discipline

1. `strict_external` labels must not be used to select rules, thresholds, model families, or tie-breakers.
2. `strict_external` may only be used for frozen audit reporting after a rule is selected from `old_data + third_batch`.
3. Any result on `strict_external` is descriptive evidence, not prospective validation.
4. v135 raw 64.81% results are not the main workflow and must not be mixed into v195+ headline comparisons.
5. Do not claim full external generalization success from 108 strict external cases.
6. Do not modify original project code while this remains an experiment.

## Remaining Work

### Phase A: Candidate Definition Freeze

Status: completed

Goal:

Freeze the exact candidate definitions before any more metric comparisons.

Tasks:

- [x] Write a candidate manifest for `phase1_all_safe_union`.
- [x] Write a candidate manifest for `phase2_min10_both_domain_union`.
- [x] Record source files, rule filters, thresholds, and included candidate ids.
- [x] State which candidate is exploratory and which candidate is the conservative main candidate.
- [x] Add a guard note that strict external was not used for selection.

Outputs:

```text
experiments/risk_control_rejection_20260621/configs/v195_plus_candidate_manifest.json
experiments/risk_control_rejection_20260621/reports/v195_plus_candidate_protocol.md
```

Decision gate:

Passed. The generated manifest reproduces the exact candidate ids used in existing Phase 1/2 outputs:

- `phase1_all_safe_union`: 24/24 candidate ids matched.
- `phase2_min10_both_domain_union`: 7/7 candidate ids matched.

Generated files:

```text
experiments/risk_control_rejection_20260621/scripts/v195_plus_candidate_freeze_protocol.py
experiments/risk_control_rejection_20260621/configs/v195_plus_candidate_manifest.json
experiments/risk_control_rejection_20260621/outputs/v195_plus_candidate_freeze_audit_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_candidate_protocol.md
```

### Phase B: Residual Error Attribution

Status: completed

Goal:

Clarify whether the known third_batch auto error is inherited from v185/v195 or introduced by v195+ release.

Tasks:

- [x] Identify the third_batch auto-error case under v195 baseline.
- [x] Identify whether Phase 1 or Phase 2 newly releases any additional error cases.
- [x] Separate errors into:
  - baseline automatic error,
  - v195 release action error,
  - v195+ newly introduced release error.
- [x] Produce a short casebook for all auto-error cases across old_data, third_batch, strict_external.

Outputs:

```text
outputs/v195_plus_error_attribution.csv
reports/v195_plus_error_attribution.md
```

Decision gate:

Passed. Phase 2 introduces `0` new auto errors on old_data + third_batch relative to v195.

Generated files:

```text
experiments/risk_control_rejection_20260621/scripts/v195_plus_error_attribution.py
experiments/risk_control_rejection_20260621/outputs/v195_plus_error_attribution.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_error_attribution_summary.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_error_attribution_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_error_attribution.md
```

### Phase C: Rule-Family and Threshold Sensitivity

Status: completed

Goal:

Check whether the candidate depends too heavily on one model family or one fragile threshold.

Tasks:

- [x] Evaluate Phase 2 without `tabular_hgb_l2`.
- [x] Evaluate Phase 2 without DINO-based correctors.
- [x] Evaluate Phase 2 using only `v161_final_review_or_reject` candidates.
- [x] Evaluate threshold neighborhoods around 0.985, 0.990, 0.995.
- [x] Report strict external as frozen audit only.

Outputs:

```text
outputs/v195_plus_family_ablation.csv
outputs/v195_plus_threshold_sensitivity.csv
reports/v195_plus_family_threshold_audit.md
```

Decision gate:

Passed with caution. Phase 2 does not become worse than v195 after family removal, but the strict_external gain disappears when `tabular_hgb_l2` is removed:

- v195 strict auto: 52/108.
- Phase 2 full strict auto: 57/108.
- Phase 2 without `tabular_hgb_l2`: 52/108.
- Phase 2 without DINO feature-set rules: 57/108.

This means the strict_external gain is concentrated in a high-support `tabular_hgb_l2` rule, not in a thin-support DINO rule. Keep this as a caution in the final report.

Generated files:

```text
experiments/risk_control_rejection_20260621/scripts/v195_plus_family_threshold_audit.py
experiments/risk_control_rejection_20260621/outputs/v195_plus_family_ablation.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_threshold_sensitivity.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_family_threshold_audit_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_family_threshold_audit.md
```

### Phase D: Subgroup and Casebook Finalization

Status: completed

Goal:

Make the safety profile interpretable by label group and released-case type.

Tasks:

- [x] Produce subgroup rows by domain, label, and domain-label.
- [x] Produce strict external incremental casebook for Phase 1 and Phase 2 separately.
- [x] Report which labels are newly released by Phase 2 compared with v195.
- [x] Highlight any high-risk groups where review decreases.

Outputs:

```text
outputs/v195_plus_subgroup_audit.csv
outputs/v195_plus_strict_incremental_casebook.csv
reports/v195_plus_subgroup_casebook.md
```

Decision gate:

Passed. Phase 2 newly releases 5 strict_external cases beyond v195, with 0 errors and 0 high-risk FN. No weak-support high-risk incremental case was found.

Generated files:

```text
experiments/risk_control_rejection_20260621/scripts/v195_plus_subgroup_casebook.py
experiments/risk_control_rejection_20260621/outputs/v195_plus_subgroup_audit.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_strict_incremental_casebook.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_subgroup_casebook_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_subgroup_casebook.md
```

### Phase E: Reproducibility Runner

Status: completed

Goal:

Create one command that regenerates the v195+ evidence without relying on conversation context.

Tasks:

- [x] Build a runner that executes:
  - v195 Phase 1 union audit,
  - v195 Phase 2 stability pruning audit,
  - full stability audit,
  - remaining Phase B-D audits.
- [x] Validate expected output files exist.
- [x] Validate headline metrics match expected values.
- [x] Avoid writing outside the experiment directory.

Outputs:

```text
scripts/v195_plus_reproducibility_runner.py
outputs/v195_plus_reproducibility_report.json
reports/v195_plus_reproducibility_runner.md
```

Decision gate:

Passed. The runner validates the core v195+ evidence chain:

- v195 strict_external auto 52/108, auto error 0.
- Phase 1 strict_external auto 63/108, auto error 0.
- Phase 2 strict_external auto 57/108, auto error 0.
- Candidate freeze, error attribution, family/threshold, and subgroup/casebook gates passed.

Generated files:

```text
experiments/risk_control_rejection_20260621/scripts/v195_plus_reproducibility_runner.py
experiments/risk_control_rejection_20260621/outputs/v195_plus_reproducibility_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_reproducibility_runner.md
```

### Phase F: Final Stage Report

Status: completed

Goal:

Write the final experiment-stage report, not a deployment claim.

Required content:

- [x] v195 baseline results.
- [x] Phase 1 exploratory upper-bound results.
- [x] Phase 2 conservative candidate results.
- [x] Error attribution.
- [x] Rule-family and threshold sensitivity.
- [x] Subgroup/casebook analysis.
- [x] Reproducibility command.
- [x] Limitations:
  - strict external n=108,
  - frozen descriptive audit only,
  - no prospective external validation,
  - no model retraining,
  - no original project integration.

Outputs:

```text
reports/v195_plus_final_experiment_report.md
```

Decision gate:

Completed. The final experiment-stage report has been written:

```text
experiments/risk_control_rejection_20260621/reports/v195_plus_final_experiment_report.md
```

The user can now decide whether to:

1. keep as experimental evidence only,
2. package as a sidecar v195+ flow,
3. request integration into original project code.

## Current Recommendation

Do not integrate yet.

Continue with Phase A, then Phase B-D, then runner, then final report.

The current conservative candidate is:

```text
phase2_union_min10_both_domains
```

The current exploratory upper-bound candidate is:

```text
phase1_all_safe_union
```
