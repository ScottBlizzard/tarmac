# Phase 1 Risk-Controlled Rejection Layer Summary

## Boundary

- This experiment is isolated under `experiments/risk_control_rejection_20260621/`.
- Existing project scripts and outputs were treated as read-only inputs.
- Strict external is a frozen audit set; no threshold or feature choice is selected from strict external labels.
- Coverage improvements on strict external are exploratory only.

## v195 Reconstruction

- All domains: n=699, auto=447, review/reject=252, auto errors=1, BAcc=99.81%.
- Strict external: n=108, auto=52, review/reject=56, auto errors=0, Wilson95 upper=6.88%.

## First Risk-Ranking Audit

The current ranking outputs are baselines for auditing, not deployable thresholds. They are intended to expose coverage-risk behavior before any model training.

Top low-error internal settings by old+third aggregate auto errors:

- hybrid_score at review=0.80: old+third auto errors=1, mean review=80.0%.
- router_score at review=0.80: old+third auto errors=1, mean review=80.0%.
- disagreement_score at review=0.70: old+third auto errors=2, mean review=70.1%.
- disagreement_score at review=0.80: old+third auto errors=2, mean review=80.0%.
- disagreement_score at review=0.60: old+third auto errors=3, mean review=60.1%.
- hybrid_score at review=0.70: old+third auto errors=3, mean review=70.1%.
- router_score at review=0.70: old+third auto errors=3, mean review=70.1%.
- uncertainty_score at review=0.70: old+third auto errors=3, mean review=70.1%.
- uncertainty_score at review=0.80: old+third auto errors=3, mean review=80.0%.
- hybrid_score at review=0.60: old+third auto errors=4, mean review=60.1%.

## Old+Third Fold Stability Audit

Strict external is excluded from this stability audit. Each fold is ranked and evaluated within its own old/third subset.

- old_data | router_score at review=0.40: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=40.4%.
- old_data | disagreement_score at review=0.50: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=49.1%.
- old_data | router_score at review=0.50: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=49.1%.
- old_data | disagreement_score at review=0.60: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=59.6%.
- old_data | hybrid_score at review=0.60: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=59.6%.
- old_data | router_score at review=0.60: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=59.6%.
- old_data | disagreement_score at review=0.70: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=70.2%.
- old_data | hybrid_score at review=0.70: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=70.2%.
- old_data | router_score at review=0.70: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=70.2%.
- old_data | uncertainty_score at review=0.70: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=70.2%.
- old_data | disagreement_score at review=0.80: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=80.7%.
- old_data | hybrid_score at review=0.80: fold auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=80.7%.

## Generated Files

- `outputs/case_risk_features.csv`
- `outputs/batch_shift_features.csv`
- `outputs/v195_baseline_reconstruction.csv`
- `outputs/coverage_risk_curves.csv`
- `outputs/old_third_fold_coverage_risk_curves.csv`
- `outputs/old_third_fold_stability_summary.csv`
- `outputs/phase1_audit_summary.csv`