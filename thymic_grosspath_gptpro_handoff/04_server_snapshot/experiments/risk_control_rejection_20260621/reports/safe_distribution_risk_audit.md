# Safe Distribution Risk Audit

## Boundary

- The robust safety profile is fitted only on old+third `confirmed_safe` v195 automatic cases.
- Strict external is scored after fitting and remains a frozen audit display.
- This is a semi-supervised ranking baseline, not a deployable threshold.

## Training Set

- Confirmed-safe internal training cases: 394.
- Case-level features: base_uncertainty, base_entropy, model_disagreement_rate, corrector_confidence_mean, corrector_prob_high_mean, router_risk_mean, router_risk_max.

## Fold-Out Internal Stability

- old_data review=0.40: auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=40.4%.
- old_data review=0.50: auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=49.1%.
- old_data review=0.60: auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=59.6%.
- old_data review=0.70: auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=70.2%.
- old_data review=0.80: auto errors sum/max=0/0, high-risk FN sum/max=0/0, mean review=80.7%.
- old_data review=0.20: auto errors sum/max=1/1, high-risk FN sum/max=0/0, mean review=19.3%.
- old_data review=0.30: auto errors sum/max=1/1, high-risk FN sum/max=0/0, mean review=29.8%.

## Frozen Strict External Display

- safe_distance_case_score review=0.20: auto=86, review=22, auto errors=0, high-risk FN=0.
- safe_distance_case_score review=0.30: auto=76, review=32, auto errors=0, high-risk FN=0.
- safe_distance_case_score review=0.40: auto=65, review=43, auto errors=0, high-risk FN=0.
- safe_distance_case_score review=0.50: auto=54, review=54, auto errors=0, high-risk FN=0.
- safe_distance_case_score review=0.60: auto=43, review=65, auto errors=0, high-risk FN=0.
- safe_distance_case_score review=0.70: auto=32, review=76, auto errors=0, high-risk FN=0.
- safe_distance_case_score review=0.80: auto=22, review=86, auto errors=0, high-risk FN=0.
- safe_distance_full_score review=0.20: auto=86, review=22, auto errors=0, high-risk FN=0.
- safe_distance_full_score review=0.30: auto=76, review=32, auto errors=0, high-risk FN=0.
- safe_distance_full_score review=0.40: auto=65, review=43, auto errors=0, high-risk FN=0.
- safe_distance_full_score review=0.50: auto=54, review=54, auto errors=0, high-risk FN=0.
- safe_distance_full_score review=0.60: auto=43, review=65, auto errors=0, high-risk FN=0.
- safe_distance_full_score review=0.70: auto=32, review=76, auto errors=0, high-risk FN=0.
- safe_distance_full_score review=0.80: auto=22, review=86, auto errors=0, high-risk FN=0.

## Generated Files

- `outputs/safe_distribution_scores.csv`
- `outputs/safe_distribution_frozen_coverage_risk_curves.csv`
- `outputs/safe_distribution_fold_coverage_risk_curves.csv`
- `outputs/safe_distribution_fold_stability_summary.csv`