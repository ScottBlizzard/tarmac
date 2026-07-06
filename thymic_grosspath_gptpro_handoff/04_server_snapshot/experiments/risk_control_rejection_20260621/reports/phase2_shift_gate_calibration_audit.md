# Phase 2 Shift Gate Calibration Audit

## Scope

This is a label-free metric-space calibration audit. It uses only old_data and third_batch
to set conservative severe-shift thresholds, then evaluates strict_external as a frozen
audit domain. It does not use strict_external labels or release errors to tune the gate.

## Calibration Rule

- known domains: old_data, third_batch
- audit domain: strict_external
- severe multiplier: 1.5
- minimum exceeded metrics for severe support: 3

## Thresholds

| Metric | Known Max | Severe Threshold | Strict Value | Strict/Known Max | Exceeds |
| --- | ---: | ---: | ---: | ---: | --- |
| batch_shift_index | 0.969486 | 1.454229 | 2.324184 | 2.40x | True |
| domain_auc_cv | 0.531717 | 0.797575 | 0.994501 | 1.87x | True |
| mean_outside_ref_05_95_rate | 0.098856 | 0.148284 | 0.439815 | 4.45x | True |
| quality_proxy_mean | 0.021950 | 0.032925 | 0.306096 | 13.95x | True |

## Metric-space Stress

| Source Domain | Stress Multiplier | Exceeded Metrics |
| --- | ---: | ---: |
| third_batch | 1.00 | 0 |
| third_batch | 1.25 | 0 |
| third_batch | 1.50 | 0 |
| third_batch | 2.00 | 3 |
| old_data | 1.00 | 0 |
| old_data | 1.25 | 0 |
| old_data | 1.50 | 0 |
| old_data | 2.00 | 3 |

## Result

- strict exceeded metrics: 4 / 4
- strict gate supported: True
- recommendation: `keep_severe_unknown_shift_fallback`

## Interpretation

Under this conservative label-free calibration, strict_external still has enough unsupervised
shift evidence to keep the deployable severe-shift fallback. This does not rule out future
relaxation, but it means Phase 2 has not yet produced evidence to override the hard gate.