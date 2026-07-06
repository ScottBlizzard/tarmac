# Phase 2 Leave-one Gate Sensitivity Grid

## Scope

This audit scans relaxed severe-shift multipliers under leave-one-known-domain calibration.
It uses only label-free shift metrics.

## First Stable Strict-external Classification

| Minimum Exceeded Metrics | First Stable Strict Non-severe | First Stable Strict Severe |
| ---: | ---: | ---: |
| 1 | - | 2.00 |
| 2 | 10.00 | 1.50 |
| 3 | 5.00 | 1.00 |
| 4 | 2.00 | 1.00 |

## Grid

| Multiplier | Min Metrics | Heldout Known Stable | Strict Severe Train Domains | All Strict Non-severe | All Strict Severe |
| ---: | ---: | --- | ---: | --- | --- |
| 1.00 | 1 | False | 2/2 | False | True |
| 1.25 | 1 | False | 2/2 | False | True |
| 1.50 | 1 | False | 2/2 | False | True |
| 2.00 | 1 | True | 2/2 | False | True |
| 2.50 | 1 | True | 2/2 | False | True |
| 3.00 | 1 | True | 2/2 | False | True |
| 4.00 | 1 | True | 2/2 | False | True |
| 5.00 | 1 | True | 2/2 | False | True |
| 10.00 | 1 | True | 2/2 | False | True |
| 15.00 | 1 | True | 1/2 | False | False |
| 1.00 | 2 | False | 2/2 | False | True |
| 1.25 | 2 | False | 2/2 | False | True |
| 1.50 | 2 | True | 2/2 | False | True |
| 2.00 | 2 | True | 2/2 | False | True |
| 2.50 | 2 | True | 2/2 | False | True |
| 3.00 | 2 | True | 2/2 | False | True |
| 4.00 | 2 | True | 2/2 | False | True |
| 5.00 | 2 | True | 1/2 | False | False |
| 10.00 | 2 | True | 0/2 | True | False |
| 15.00 | 2 | True | 0/2 | True | False |
| 1.00 | 3 | True | 2/2 | False | True |
| 1.25 | 3 | True | 2/2 | False | True |
| 1.50 | 3 | True | 2/2 | False | True |
| 2.00 | 3 | True | 2/2 | False | True |
| 2.50 | 3 | True | 1/2 | False | False |
| 3.00 | 3 | True | 1/2 | False | False |
| 4.00 | 3 | True | 1/2 | False | False |
| 5.00 | 3 | True | 0/2 | True | False |
| 10.00 | 3 | True | 0/2 | True | False |
| 15.00 | 3 | True | 0/2 | True | False |
| 1.00 | 4 | True | 2/2 | False | True |
| 1.25 | 4 | True | 2/2 | False | True |
| 1.50 | 4 | True | 2/2 | False | True |
| 2.00 | 4 | True | 0/2 | True | False |
| 2.50 | 4 | True | 0/2 | True | False |
| 3.00 | 4 | True | 0/2 | True | False |
| 4.00 | 4 | True | 0/2 | True | False |
| 5.00 | 4 | True | 0/2 | True | False |
| 10.00 | 4 | True | 0/2 | True | False |
| 15.00 | 4 | True | 0/2 | True | False |

## Interpretation

A deployable relaxation would need the strict_external classification to be stable across
known-domain calibration choices. If stable non-severe behavior requires a very loose
multiplier, the relaxed gate should remain frozen what-if evidence only.