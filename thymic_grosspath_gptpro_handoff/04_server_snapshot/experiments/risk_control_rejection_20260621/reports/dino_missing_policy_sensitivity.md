# DINO Missing Policy Sensitivity

## Purpose

This frozen audit compares how missing `dino_fn_risk_max` handling affects the same
15% case-level release rule. It is not used to tune strict_external; strict_external remains
a frozen display.

## Summary

| Missing Policy | Scope | Released | Released Missing DINO | Released Errors | Auto Errors | High-risk FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sort_last | old_data | 11 | 0 | 0 | 0 | 0 |
| sort_last | third_batch | 18 | 0 | 0 | 1 | 1 |
| sort_last | strict_external | 8 | 0 | 0 | 0 | 0 |
| sort_last | all_domains | 37 | 0 | 0 | 1 | 1 |
| sort_first | old_data | 11 | 3 | 0 | 0 | 0 |
| sort_first | third_batch | 18 | 3 | 0 | 1 | 1 |
| sort_first | strict_external | 8 | 0 | 0 | 0 | 0 |
| sort_first | all_domains | 37 | 6 | 0 | 1 | 1 |
| median | old_data | 11 | 3 | 0 | 0 | 0 |
| median | third_batch | 18 | 3 | 0 | 1 | 1 |
| median | strict_external | 8 | 0 | 0 | 0 | 0 |
| median | all_domains | 37 | 6 | 0 | 1 | 1 |