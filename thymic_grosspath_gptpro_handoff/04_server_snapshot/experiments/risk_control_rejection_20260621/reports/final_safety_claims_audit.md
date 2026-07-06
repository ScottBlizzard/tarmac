# Final Safety Claims Audit

## Result

- all claims passed: True
- claim rows: 4

## Claims

| Claim | Passed | Evidence | Detail |
| --- | --- | --- | --- |
| candidate_contract_locked | True | outputs/candidate_lock_audit_report.json | Executable contract is locked to domain@15%. |
| fold_selected_zero_released_error | True | outputs/fold_safe_frontier_report.json | Selected fold frontier has zero released errors and zero released high-risk FN. |
| deployable_zero_new_release_error | True | outputs/final_policy_runner_summary.csv | shift_aware_deployable released_error_n=0, released_high_risk_fn_n=0. |
| strict_external_no_deployable_release | True | outputs/deployment_readiness_matrix.csv | strict_external shift-aware released_n=0; case-only strict_external remains frozen_display_only. |