# Phase 1 Release Experiment Report

## Scope

This stage kept all work isolated under `experiments/risk_control_rejection_20260621/`.
Existing project code and previous project outputs were used only as read-only inputs.

Strict external remained a frozen audit set. It was not used to choose features, thresholds,
release fractions, or candidate policies.

## What Was Added

1. Added v195 risk labels to `case_risk_features.csv`:
   - `confirmed_safe`: v195 automatic decision and base prediction correct.
   - `confirmed_error`: v195 automatic decision and base prediction wrong.
   - `unlabeled_or_uncertain`: v195 reviewed/rejected cases.

2. Added old+third fold stability audits for simple risk rankings.

3. Added `safe_distribution_risk.py`:
   - Fits a robust safe-profile only on old+third `confirmed_safe` v195 automatic cases.
   - Scores all cases after fitting.
   - Produces fold-out internal audits and frozen strict-external displays.

4. Added `release_rejected_audit.py`:
   - Keeps all v195 automatic decisions unchanged.
   - Only considers v195 reviewed/rejected cases for possible release.
   - Releases the lowest-risk reviewed/rejected cases by candidate score.
   - Separates total automatic errors from newly introduced released-case errors.

## Verification

The current helper test suite passes:

```text
python -m unittest experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py -v
Ran 10 tests
OK
```

## v195 Baseline

Reconstructed v195 behavior:

| Scope | n | Auto | Review/Reject | Auto Errors | High-risk FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| old_data | 285 | 210 | 75 | 0 | 0 |
| third_batch | 306 | 185 | 121 | 1 | 1 |
| strict_external | 108 | 52 | 56 | 0 | 0 |
| all_domains | 699 | 447 | 252 | 1 | 1 |

Risk label counts:

| Label | Count |
| --- | ---: |
| confirmed_safe | 446 |
| unlabeled_or_uncertain | 252 |
| confirmed_error | 1 |

## Safe-Distribution Score Result

The semi-supervised safe-distribution score is useful as an audit signal, but it should not
replace v195 rejection logic.

Reason: when used as a standalone ranking gate, third_batch still leaves multiple automatic
high-risk false negatives even at high review rates. This confirms that the safer direction is
not “replace v195 with a new risk ranker”, but “start from v195 and only release a controlled
low-risk subset of v195 rejected cases.”

## Releasing v195 Rejected Cases

Internal old+third aggregate audit found that `model_disagreement_rate` was the strongest simple
release score.

Using whole-domain internal ranking:

| Score | Release Fraction of v195 Review | Released | Remaining Review | New Released Errors | Total Auto Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| model_disagreement_rate | 0.25 | 48 | 148 | 0 | 1 |
| model_disagreement_rate | 0.20 | 39 | 157 | 0 | 1 |
| model_disagreement_rate | 0.15 | 29 | 167 | 0 | 1 |
| model_disagreement_rate | 0.10 | 19 | 177 | 0 | 1 |
| model_disagreement_rate | 0.05 | 9 | 187 | 0 | 1 |

However, fold-level audit is stricter. In third_batch, `model_disagreement_rate` remains
new-error-free only through release fraction `0.15`. At `0.20` and above, fold-level release
introduces additional errors in third_batch.

Therefore the conservative Phase 1 candidate is:

```text
Base policy: v195
Release candidates: only v195 reviewed/rejected cases
Release score: model_disagreement_rate
Release fraction: 0.15 of v195 reviewed/rejected cases
Selection basis: old+third internal and fold stability only
```

## Frozen Strict External Display

For the conservative candidate above, strict external frozen display is:

| Score | Release Fraction | Released | Remaining Review | New Released Errors | Total Auto Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| model_disagreement_rate | 0.15 | 8 | 48 | 0 | 0 |

This lowers strict external review/reject count from 56 to 48, corresponding to a review/reject
rate of 44.4%. This is encouraging, but it must remain an audit observation, not a tuning basis.

## Current Interpretation

The strongest result so far is not a new standalone rejection model. It is a conservative
extension of v195:

- Keep v195 as the safety baseline.
- Do not alter existing v195 automatic decisions.
- Release only a small, internally justified low-disagreement subset of v195 rejected cases.
- Treat strict external improvement as frozen confirmation only.

The 25% internal aggregate release looks attractive, but fold-level third_batch auditing makes it
too aggressive for a Phase 1 safety candidate. The 15% release setting is the current defensible
candidate because it is selected from old+third stability constraints and still gives a useful
strict-external frozen reduction.

## Generated Outputs

- `outputs/case_risk_features.csv`
- `outputs/old_third_fold_coverage_risk_curves.csv`
- `outputs/old_third_fold_stability_summary.csv`
- `outputs/safe_distribution_scores.csv`
- `outputs/safe_distribution_frozen_coverage_risk_curves.csv`
- `outputs/safe_distribution_fold_coverage_risk_curves.csv`
- `outputs/safe_distribution_fold_stability_summary.csv`
- `outputs/v195_rejected_release_curves.csv`
- `outputs/v195_rejected_release_internal_summary.csv`
- `outputs/v195_rejected_release_fold_curves.csv`
- `outputs/v195_rejected_release_fold_stability_summary.csv`
