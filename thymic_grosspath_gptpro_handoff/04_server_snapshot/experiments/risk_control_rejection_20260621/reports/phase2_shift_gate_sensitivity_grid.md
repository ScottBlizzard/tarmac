# Phase 2 Shift Gate Sensitivity Grid

## Scope

This audit scans label-free severe-shift multipliers and the number of metrics required to
support a severe gate. It uses only batch-shift metrics and does not use strict_external
labels.

## First Multiplier Where Strict External Is No Longer Severe

| Minimum Exceeded Metrics | First Not-severe Multiplier |
| ---: | ---: |
| 1 | 15.00 |
| 2 | 5.00 |
| 3 | 2.50 |
| 4 | 2.00 |

## Grid

| Severe Multiplier | Minimum Exceeded Metrics | Strict Exceeded Metrics | Severe Supported |
| ---: | ---: | ---: | --- |
| 1.00 | 1 | 4 | True |
| 1.25 | 1 | 4 | True |
| 1.50 | 1 | 4 | True |
| 2.00 | 1 | 3 | True |
| 2.50 | 1 | 2 | True |
| 3.00 | 1 | 2 | True |
| 4.00 | 1 | 2 | True |
| 5.00 | 1 | 1 | True |
| 10.00 | 1 | 1 | True |
| 15.00 | 1 | 0 | False |
| 1.00 | 2 | 4 | True |
| 1.25 | 2 | 4 | True |
| 1.50 | 2 | 4 | True |
| 2.00 | 2 | 3 | True |
| 2.50 | 2 | 2 | True |
| 3.00 | 2 | 2 | True |
| 4.00 | 2 | 2 | True |
| 5.00 | 2 | 1 | False |
| 10.00 | 2 | 1 | False |
| 15.00 | 2 | 0 | False |
| 1.00 | 3 | 4 | True |
| 1.25 | 3 | 4 | True |
| 1.50 | 3 | 4 | True |
| 2.00 | 3 | 3 | True |
| 2.50 | 3 | 2 | False |
| 3.00 | 3 | 2 | False |
| 4.00 | 3 | 2 | False |
| 5.00 | 3 | 1 | False |
| 10.00 | 3 | 1 | False |
| 15.00 | 3 | 0 | False |
| 1.00 | 4 | 4 | True |
| 1.25 | 4 | 4 | True |
| 1.50 | 4 | 4 | True |
| 2.00 | 4 | 3 | False |
| 2.50 | 4 | 2 | False |
| 3.00 | 4 | 2 | False |
| 4.00 | 4 | 2 | False |
| 5.00 | 4 | 1 | False |
| 10.00 | 4 | 1 | False |
| 15.00 | 4 | 0 | False |

## Interpretation

The default Phase 2 calibration uses multiplier 1.5 and requires 3 exceeded metrics. If strict
external remains severe across nearby looser settings, this supports keeping the hard fallback.
If it loses severe status only at very loose multipliers, that is evidence against relaxing the
gate without stronger independent data.