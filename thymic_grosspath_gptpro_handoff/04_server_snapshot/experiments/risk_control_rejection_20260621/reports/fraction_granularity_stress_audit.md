# Fraction Granularity Stress Audit

## Purpose

This audit scans release fraction and quota granularity using old_data + third_batch as
the selection evidence. strict_external is reported only as a frozen display.

## Selected Internal-safe Candidate

- granularity: `domain`
- release_fraction: `0.2`
- internal released_n: `39`
- internal released_error_n: `0`
- internal released_high_risk_fn_n: `0`

## Summary

| Granularity | Fraction | Scope | Released | Released Errors | Released High-risk FN | Auto Errors | Auto High-risk FN |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| domain | 0.050 | internal_known | 9 | 0 | 0 | 1 | 1 |
| domain | 0.050 | strict_external | 2 | 0 | 0 | 0 | 0 |
| domain | 0.100 | internal_known | 19 | 0 | 0 | 1 | 1 |
| domain | 0.100 | strict_external | 5 | 0 | 0 | 0 | 0 |
| domain | 0.125 | internal_known | 24 | 0 | 0 | 1 | 1 |
| domain | 0.125 | strict_external | 7 | 0 | 0 | 0 | 0 |
| domain | 0.150 | internal_known | 29 | 0 | 0 | 1 | 1 |
| domain | 0.150 | strict_external | 8 | 0 | 0 | 0 | 0 |
| domain | 0.175 | internal_known | 34 | 0 | 0 | 1 | 1 |
| domain | 0.175 | strict_external | 9 | 0 | 0 | 0 | 0 |
| domain | 0.200 | internal_known | 39 | 0 | 0 | 1 | 1 |
| domain | 0.200 | strict_external | 11 | 0 | 0 | 0 | 0 |
| domain_label | 0.050 | internal_known | 4 | 0 | 0 | 1 | 1 |
| domain_label | 0.050 | strict_external | 1 | 0 | 0 | 0 | 0 |
| domain_label | 0.100 | internal_known | 16 | 1 | 1 | 2 | 2 |
| domain_label | 0.100 | strict_external | 3 | 0 | 0 | 0 | 0 |
| domain_label | 0.125 | internal_known | 20 | 1 | 1 | 2 | 2 |
| domain_label | 0.125 | strict_external | 5 | 0 | 0 | 0 | 0 |
| domain_label | 0.150 | internal_known | 24 | 1 | 1 | 2 | 2 |
| domain_label | 0.150 | strict_external | 6 | 0 | 0 | 0 | 0 |
| domain_label | 0.175 | internal_known | 28 | 3 | 3 | 4 | 4 |
| domain_label | 0.175 | strict_external | 6 | 0 | 0 | 0 | 0 |
| domain_label | 0.200 | internal_known | 36 | 3 | 3 | 4 | 4 |
| domain_label | 0.200 | strict_external | 8 | 0 | 0 | 0 | 0 |
| domain_shift | 0.050 | internal_known | 9 | 0 | 0 | 1 | 1 |
| domain_shift | 0.050 | strict_external | 2 | 0 | 0 | 0 | 0 |
| domain_shift | 0.100 | internal_known | 19 | 0 | 0 | 1 | 1 |
| domain_shift | 0.100 | strict_external | 5 | 0 | 0 | 0 | 0 |
| domain_shift | 0.125 | internal_known | 24 | 0 | 0 | 1 | 1 |
| domain_shift | 0.125 | strict_external | 7 | 0 | 0 | 0 | 0 |
| domain_shift | 0.150 | internal_known | 29 | 0 | 0 | 1 | 1 |
| domain_shift | 0.150 | strict_external | 8 | 0 | 0 | 0 | 0 |
| domain_shift | 0.175 | internal_known | 34 | 0 | 0 | 1 | 1 |
| domain_shift | 0.175 | strict_external | 9 | 0 | 0 | 0 | 0 |
| domain_shift | 0.200 | internal_known | 39 | 0 | 0 | 1 | 1 |
| domain_shift | 0.200 | strict_external | 11 | 0 | 0 | 0 | 0 |