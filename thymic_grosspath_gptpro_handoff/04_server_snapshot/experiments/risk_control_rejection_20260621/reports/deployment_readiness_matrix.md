# Deployment Readiness Matrix

## Purpose

This matrix prevents frozen strict_external display results from being read as deployable
coverage claims. The deployable policy is `shift_aware_deployable`; case-only strict_external
rows are audit display only.

## Matrix

| Scope | Mode | Status | Deployable Claim | Released | Released Errors | High-risk FN | Evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| old_data | case_only_exploratory | exploratory_internal_reference | False | 11 | 0 | 0 | case_only_mode_is_not_the_deployable_policy |
| third_batch | case_only_exploratory | exploratory_internal_reference | False | 18 | 0 | 0 | case_only_mode_is_not_the_deployable_policy |
| strict_external | case_only_exploratory | frozen_display_only | False | 8 | 0 | 0 | strict_external_case_only_result_is_not_deployable |
| all_domains | case_only_exploratory | exploratory_internal_reference | False | 37 | 0 | 0 | case_only_mode_is_not_the_deployable_policy |
| old_data | shift_aware_deployable | deployable_release | True | 11 | 0 | 0 | shift_aware_policy_with_zero_released_errors |
| third_batch | shift_aware_deployable | deployable_release | True | 18 | 0 | 0 | shift_aware_policy_with_zero_released_errors |
| strict_external | shift_aware_deployable | deployable_fallback_to_v195 | True | 0 | 0 | 0 | severe_unknown_shift_hard_gate |
| all_domains | shift_aware_deployable | deployable_release | True | 29 | 0 | 0 | shift_aware_policy_with_zero_released_errors |