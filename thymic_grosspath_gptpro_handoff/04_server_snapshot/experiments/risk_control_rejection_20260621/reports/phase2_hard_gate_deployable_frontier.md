# Phase 2 Hard-gate Deployable Frontier

## Scope

This audit scans case-level release fraction and quota granularity while keeping the
deployable severe-shift hard gate unchanged. Any severe_unknown_shift domain falls back
to v195, so strict_external cannot be released by this frontier.

## Selected Candidate

- granularity: `domain`
- release_fraction: `0.15`
- third_batch released: `18`
- internal_known released: `29`
- strict_external released: `0`
- all-domain released errors: `0`
- all-domain released high-risk FN: `0`
- known-domain fold-safe: `True`

## Frontier Summary

| Granularity | Fraction | Scope | Released | Released Errors | Released High-risk FN | Auto Errors | Auto High-risk FN |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| domain | 0.050 | internal_known | 9 | 0 | 0 | 1 | 1 |
| domain | 0.050 | old_data | 3 | 0 | 0 | 0 | 0 |
| domain | 0.050 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain | 0.050 | third_batch | 6 | 0 | 0 | 1 | 1 |
| domain | 0.100 | internal_known | 19 | 0 | 0 | 1 | 1 |
| domain | 0.100 | old_data | 7 | 0 | 0 | 0 | 0 |
| domain | 0.100 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain | 0.100 | third_batch | 12 | 0 | 0 | 1 | 1 |
| domain | 0.125 | internal_known | 24 | 0 | 0 | 1 | 1 |
| domain | 0.125 | old_data | 9 | 0 | 0 | 0 | 0 |
| domain | 0.125 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain | 0.125 | third_batch | 15 | 0 | 0 | 1 | 1 |
| domain | 0.150 | internal_known | 29 | 0 | 0 | 1 | 1 |
| domain | 0.150 | old_data | 11 | 0 | 0 | 0 | 0 |
| domain | 0.150 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain | 0.150 | third_batch | 18 | 0 | 0 | 1 | 1 |
| domain | 0.175 | internal_known | 34 | 0 | 0 | 1 | 1 |
| domain | 0.175 | old_data | 13 | 0 | 0 | 0 | 0 |
| domain | 0.175 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain | 0.175 | third_batch | 21 | 0 | 0 | 1 | 1 |
| domain | 0.200 | internal_known | 39 | 0 | 0 | 1 | 1 |
| domain | 0.200 | old_data | 15 | 0 | 0 | 0 | 0 |
| domain | 0.200 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain | 0.200 | third_batch | 24 | 0 | 0 | 1 | 1 |
| domain_label | 0.050 | internal_known | 4 | 0 | 0 | 1 | 1 |
| domain_label | 0.050 | old_data | 0 | 0 | 0 | 0 | 0 |
| domain_label | 0.050 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_label | 0.050 | third_batch | 4 | 0 | 0 | 1 | 1 |
| domain_label | 0.100 | internal_known | 16 | 1 | 1 | 2 | 2 |
| domain_label | 0.100 | old_data | 5 | 0 | 0 | 0 | 0 |
| domain_label | 0.100 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_label | 0.100 | third_batch | 11 | 1 | 1 | 2 | 2 |
| domain_label | 0.125 | internal_known | 20 | 1 | 1 | 2 | 2 |
| domain_label | 0.125 | old_data | 7 | 0 | 0 | 0 | 0 |
| domain_label | 0.125 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_label | 0.125 | third_batch | 13 | 1 | 1 | 2 | 2 |
| domain_label | 0.150 | internal_known | 24 | 1 | 1 | 2 | 2 |
| domain_label | 0.150 | old_data | 9 | 0 | 0 | 0 | 0 |
| domain_label | 0.150 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_label | 0.150 | third_batch | 15 | 1 | 1 | 2 | 2 |
| domain_label | 0.175 | internal_known | 28 | 3 | 3 | 4 | 4 |
| domain_label | 0.175 | old_data | 9 | 0 | 0 | 0 | 0 |
| domain_label | 0.175 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_label | 0.175 | third_batch | 19 | 3 | 3 | 4 | 4 |
| domain_label | 0.200 | internal_known | 36 | 3 | 3 | 4 | 4 |
| domain_label | 0.200 | old_data | 13 | 0 | 0 | 0 | 0 |
| domain_label | 0.200 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_label | 0.200 | third_batch | 23 | 3 | 3 | 4 | 4 |
| domain_shift | 0.050 | internal_known | 9 | 0 | 0 | 1 | 1 |
| domain_shift | 0.050 | old_data | 3 | 0 | 0 | 0 | 0 |
| domain_shift | 0.050 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_shift | 0.050 | third_batch | 6 | 0 | 0 | 1 | 1 |
| domain_shift | 0.100 | internal_known | 19 | 0 | 0 | 1 | 1 |
| domain_shift | 0.100 | old_data | 7 | 0 | 0 | 0 | 0 |
| domain_shift | 0.100 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_shift | 0.100 | third_batch | 12 | 0 | 0 | 1 | 1 |
| domain_shift | 0.125 | internal_known | 24 | 0 | 0 | 1 | 1 |
| domain_shift | 0.125 | old_data | 9 | 0 | 0 | 0 | 0 |
| domain_shift | 0.125 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_shift | 0.125 | third_batch | 15 | 0 | 0 | 1 | 1 |
| domain_shift | 0.150 | internal_known | 29 | 0 | 0 | 1 | 1 |
| domain_shift | 0.150 | old_data | 11 | 0 | 0 | 0 | 0 |
| domain_shift | 0.150 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_shift | 0.150 | third_batch | 18 | 0 | 0 | 1 | 1 |
| domain_shift | 0.175 | internal_known | 34 | 0 | 0 | 1 | 1 |
| domain_shift | 0.175 | old_data | 13 | 0 | 0 | 0 | 0 |
| domain_shift | 0.175 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_shift | 0.175 | third_batch | 21 | 0 | 0 | 1 | 1 |
| domain_shift | 0.200 | internal_known | 39 | 0 | 0 | 1 | 1 |
| domain_shift | 0.200 | old_data | 15 | 0 | 0 | 0 | 0 |
| domain_shift | 0.200 | strict_external | 0 | 0 | 0 | 0 | 0 |
| domain_shift | 0.200 | third_batch | 24 | 0 | 0 | 1 | 1 |

## Interpretation

This frontier can improve or validate known/normal-shift deployment behavior, but it cannot
increase strict_external coverage while the hard gate remains active. Candidate selection
requires both aggregate zero new release errors and old_data/third_batch fold-safe evidence.