# v195+ Reproducibility Runner

This runner regenerates the v195+ candidate freeze, error attribution, family/threshold audit, and subgroup/casebook audit from scripts under the experiment directory.

## Result

- Passed: `True`.

## Step Results

| script | returncode | passed |
| --- | --- | --- |
| experiments/risk_control_rejection_20260621/scripts/v195_phase1_union_agreement_release_audit.py | 0 | True |
| experiments/risk_control_rejection_20260621/scripts/v195_phase2_stability_pruning_audit.py | 0 | True |
| experiments/risk_control_rejection_20260621/scripts/v195_phase2_full_stability_audit.py | 0 | True |
| experiments/risk_control_rejection_20260621/scripts/v195_plus_candidate_freeze_protocol.py | 0 | True |
| experiments/risk_control_rejection_20260621/scripts/v195_plus_error_attribution.py | 0 | True |
| experiments/risk_control_rejection_20260621/scripts/v195_plus_family_threshold_audit.py | 0 | True |
| experiments/risk_control_rejection_20260621/scripts/v195_plus_subgroup_casebook.py | 0 | True |

## Metric Checks

| name | passed |
| --- | --- |
| v195_strict_auto_52_error_0 | True |
| phase1_strict_auto_63_error_0 | True |
| phase2_strict_auto_57_error_0 | True |
| candidate_freeze_passed | True |
| phase2_no_new_known_error | True |
| family_threshold_passed_with_caution | True |
| subgroup_casebook_passed | True |

## Expected Files

| path | exists | size |
| --- | --- | --- |
| experiments/risk_control_rejection_20260621/configs/v195_plus_candidate_manifest.json | True | 26482 |
| experiments/risk_control_rejection_20260621/reports/v195_plus_candidate_protocol.md | True | 3245 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_candidate_freeze_audit_report.json | True | 2119 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_error_attribution.csv | True | 617 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_error_attribution_report.json | True | 7879 |
| experiments/risk_control_rejection_20260621/reports/v195_plus_error_attribution.md | True | 3894 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_family_ablation.csv | True | 2967 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_threshold_sensitivity.csv | True | 2304 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_family_threshold_audit_report.json | True | 9972 |
| experiments/risk_control_rejection_20260621/reports/v195_plus_family_threshold_audit.md | True | 2513 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_subgroup_audit.csv | True | 11466 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_strict_incremental_casebook.csv | True | 5247 |
| experiments/risk_control_rejection_20260621/outputs/v195_plus_subgroup_casebook_report.json | True | 2171 |
| experiments/risk_control_rejection_20260621/reports/v195_plus_subgroup_casebook.md | True | 8010 |