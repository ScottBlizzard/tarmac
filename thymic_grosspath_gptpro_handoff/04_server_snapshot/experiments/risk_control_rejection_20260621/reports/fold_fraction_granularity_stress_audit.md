# Fold Fraction Granularity Stress Audit

## Purpose

This audit checks release fraction and quota granularity inside old_data and third_batch
folds. It is the safety gate for the aggregate fraction/granularity scan.

## Selected Fold-safe Candidate

- granularity: `domain`
- release_fraction: `0.15`
- fold released_n_sum: `24`
- released_error_n_sum: `0`
- released_high_risk_fn_n_sum: `0`

## Fold Summary

| Granularity | Fraction | Domain | Released Sum | Released Error Sum | Released Error Max | Released High-risk FN Sum | Fold-safe |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| domain | 0.050 | old_data | 0 | 0 | 0 | 0 | True |
| domain | 0.050 | third_batch | 4 | 0 | 0 | 0 | True |
| domain | 0.100 | old_data | 5 | 0 | 0 | 0 | True |
| domain | 0.100 | third_batch | 9 | 0 | 0 | 0 | True |
| domain | 0.125 | old_data | 6 | 0 | 0 | 0 | True |
| domain | 0.125 | third_batch | 14 | 0 | 0 | 0 | True |
| domain | 0.150 | old_data | 9 | 0 | 0 | 0 | True |
| domain | 0.150 | third_batch | 15 | 0 | 0 | 0 | True |
| domain | 0.175 | old_data | 11 | 0 | 0 | 0 | True |
| domain | 0.175 | third_batch | 20 | 1 | 1 | 1 | False |
| domain | 0.200 | old_data | 13 | 0 | 0 | 0 | True |
| domain | 0.200 | third_batch | 22 | 1 | 1 | 1 | False |
| domain_label | 0.050 | old_data | 0 | 0 | 0 | 0 | True |
| domain_label | 0.050 | third_batch | 2 | 0 | 0 | 0 | True |
| domain_label | 0.100 | old_data | 0 | 0 | 0 | 0 | True |
| domain_label | 0.100 | third_batch | 7 | 0 | 0 | 0 | True |
| domain_label | 0.125 | old_data | 0 | 0 | 0 | 0 | True |
| domain_label | 0.125 | third_batch | 9 | 0 | 0 | 0 | True |
| domain_label | 0.150 | old_data | 0 | 0 | 0 | 0 | True |
| domain_label | 0.150 | third_batch | 11 | 0 | 0 | 0 | True |
| domain_label | 0.175 | old_data | 1 | 0 | 0 | 0 | True |
| domain_label | 0.175 | third_batch | 13 | 0 | 0 | 0 | True |
| domain_label | 0.200 | old_data | 2 | 0 | 0 | 0 | True |
| domain_label | 0.200 | third_batch | 16 | 0 | 0 | 0 | True |
| domain_shift | 0.050 | old_data | 0 | 0 | 0 | 0 | True |
| domain_shift | 0.050 | third_batch | 4 | 0 | 0 | 0 | True |
| domain_shift | 0.100 | old_data | 5 | 0 | 0 | 0 | True |
| domain_shift | 0.100 | third_batch | 9 | 0 | 0 | 0 | True |
| domain_shift | 0.125 | old_data | 6 | 0 | 0 | 0 | True |
| domain_shift | 0.125 | third_batch | 14 | 0 | 0 | 0 | True |
| domain_shift | 0.150 | old_data | 9 | 0 | 0 | 0 | True |
| domain_shift | 0.150 | third_batch | 15 | 0 | 0 | 0 | True |
| domain_shift | 0.175 | old_data | 11 | 0 | 0 | 0 | True |
| domain_shift | 0.175 | third_batch | 20 | 1 | 1 | 1 | False |
| domain_shift | 0.200 | old_data | 13 | 0 | 0 | 0 | True |
| domain_shift | 0.200 | third_batch | 22 | 1 | 1 | 1 | False |