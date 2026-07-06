# Fold Boundary Failure Casebook

## Purpose

This casebook lists cases that become newly released errors in fold-level fraction/granularity
stress audits. It explains why looser aggregate policies are rejected.

## Unsafe Policy Rows

| Granularity | Fraction | Domain | Released Error Sum | Released High-risk FN Sum |
| --- | ---: | --- | ---: | ---: |
| domain | 0.175 | third_batch | 1 | 1 |
| domain | 0.200 | third_batch | 1 | 1 |
| domain_shift | 0.175 | third_batch | 1 | 1 |
| domain_shift | 0.200 | third_batch | 1 | 1 |

## Released Error Cases

| Fraction | Granularity | Domain | Fold | Case ID | Label | y | pred | Disagreement | DINO FN Risk | High-risk FN |
| ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.175 | domain | third_batch | 1 | third_B2_2517222 | B2 | 1 | 0 | 0.000000 | 0.679124 | True |
| 0.175 | domain_shift | third_batch | 1 | third_B2_2517222 | B2 | 1 | 0 | 0.000000 | 0.679124 | True |
| 0.200 | domain | third_batch | 1 | third_B2_2517222 | B2 | 1 | 0 | 0.000000 | 0.679124 | True |
| 0.200 | domain_shift | third_batch | 1 | third_B2_2517222 | B2 | 1 | 0 | 0.000000 | 0.679124 | True |