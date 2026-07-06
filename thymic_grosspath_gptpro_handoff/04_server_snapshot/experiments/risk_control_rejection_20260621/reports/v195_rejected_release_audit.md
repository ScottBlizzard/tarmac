# v195 Rejected-Case Release Audit

## Boundary

- v195 automatic decisions are kept unchanged.
- Only cases already reviewed/rejected by v195 are candidates for release.
- Internal summaries use old+third only; strict external rows are frozen display and must not be used for choosing release settings.

## Internal Candidate Settings

- model_disagreement_rate release=0.25: released=48, remaining review=148, released errors=0, released high-risk FN=0, total auto errors=1, total high-risk FN=1, mean review=24.9%.
- model_disagreement_rate release=0.20: released=39, remaining review=157, released errors=0, released high-risk FN=0, total auto errors=1, total high-risk FN=1, mean review=26.4%.
- model_disagreement_rate release=0.15: released=29, remaining review=167, released errors=0, released high-risk FN=0, total auto errors=1, total high-risk FN=1, mean review=28.1%.
- model_disagreement_rate release=0.10: released=19, remaining review=177, released errors=0, released high-risk FN=0, total auto errors=1, total high-risk FN=1, mean review=29.7%.
- model_disagreement_rate release=0.05: released=9, remaining review=187, released errors=0, released high-risk FN=0, total auto errors=1, total high-risk FN=1, mean review=31.4%.
- safe_distance_case_score release=0.05: released=9, remaining review=187, released errors=0, released high-risk FN=0, total auto errors=1, total high-risk FN=1, mean review=31.4%.
- safe_distance_full_score release=0.05: released=9, remaining review=187, released errors=0, released high-risk FN=0, total auto errors=1, total high-risk FN=1, mean review=31.4%.

## Fold Stability Check

- old_data | model_disagreement_rate release=0.50: released=36, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=13.7%.
- old_data | safe_distance_case_score release=0.50: released=36, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=13.7%.
- old_data | safe_distance_full_score release=0.50: released=36, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=13.7%.
- old_data | hybrid_score release=0.40: released=29, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=16.1%.
- old_data | model_disagreement_rate release=0.40: released=29, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=16.1%.
- old_data | router_risk_mean release=0.40: released=29, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=16.1%.
- old_data | safe_distance_case_score release=0.40: released=29, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=16.1%.
- old_data | safe_distance_full_score release=0.40: released=29, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=16.1%.
- old_data | hybrid_score release=0.30: released=20, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=19.3%.
- old_data | model_disagreement_rate release=0.30: released=20, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=19.3%.
- old_data | router_risk_mean release=0.30: released=20, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=19.3%.
- old_data | safe_distance_case_score release=0.30: released=20, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=19.3%.
- old_data | safe_distance_full_score release=0.30: released=20, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=19.3%.
- old_data | hybrid_score release=0.25: released=16, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=20.7%.
- old_data | model_disagreement_rate release=0.25: released=16, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=20.7%.
- old_data | router_risk_mean release=0.25: released=16, released errors sum/max=0/0, released high-risk FN sum/max=0/0, total auto errors sum/max=0/0, mean review=20.7%.

## Frozen Strict External Display

- base_uncertainty release=0.05: released=2, remaining review=54, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- base_uncertainty release=0.10: released=5, remaining review=51, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- base_uncertainty release=0.15: released=8, remaining review=48, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- base_uncertainty release=0.20: released=11, remaining review=45, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- base_uncertainty release=0.25: released=14, remaining review=42, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- base_uncertainty release=0.30: released=16, remaining review=40, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- base_uncertainty release=0.40: released=22, remaining review=34, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- base_uncertainty release=0.50: released=28, remaining review=28, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.05: released=2, remaining review=54, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.10: released=5, remaining review=51, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.15: released=8, remaining review=48, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.20: released=11, remaining review=45, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.25: released=14, remaining review=42, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.30: released=16, remaining review=40, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.40: released=22, remaining review=34, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- hybrid_score release=0.50: released=28, remaining review=28, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.05: released=2, remaining review=54, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.10: released=5, remaining review=51, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.15: released=8, remaining review=48, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.20: released=11, remaining review=45, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.25: released=14, remaining review=42, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.30: released=16, remaining review=40, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.40: released=22, remaining review=34, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- model_disagreement_rate release=0.50: released=28, remaining review=28, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.05: released=2, remaining review=54, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.10: released=5, remaining review=51, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.15: released=8, remaining review=48, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.20: released=11, remaining review=45, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.25: released=14, remaining review=42, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.30: released=16, remaining review=40, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.40: released=22, remaining review=34, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- router_risk_mean release=0.50: released=28, remaining review=28, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.05: released=2, remaining review=54, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.10: released=5, remaining review=51, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.15: released=8, remaining review=48, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.20: released=11, remaining review=45, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.25: released=14, remaining review=42, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.30: released=16, remaining review=40, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.40: released=22, remaining review=34, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_case_score release=0.50: released=28, remaining review=28, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.05: released=2, remaining review=54, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.10: released=5, remaining review=51, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.15: released=8, remaining review=48, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.20: released=11, remaining review=45, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.25: released=14, remaining review=42, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.30: released=16, remaining review=40, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.40: released=22, remaining review=34, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.
- safe_distance_full_score release=0.50: released=28, remaining review=28, released errors=0, released high-risk FN=0, total auto errors=0, total high-risk FN=0.

## Generated Files

- `outputs/v195_rejected_release_curves.csv`
- `outputs/v195_rejected_release_internal_summary.csv`
- `outputs/v195_rejected_release_fold_curves.csv`
- `outputs/v195_rejected_release_fold_stability_summary.csv`