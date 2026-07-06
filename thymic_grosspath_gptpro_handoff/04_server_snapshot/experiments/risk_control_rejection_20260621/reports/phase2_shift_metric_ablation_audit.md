# Phase 2 Shift Metric Ablation Audit

## Scope

This audit checks whether strict_external severe-shift classification is driven by one
unsupervised metric or by multiple metrics. It uses only label-free batch-shift values.

## All-metric Gate Across Multipliers

| Multiplier | Exceeded Metrics | Severe | Exceeded Metric Names |
| ---: | ---: | --- | --- |
| 1.00 | 4/4 | True | batch_shift_index,domain_auc_cv,mean_outside_ref_05_95_rate,quality_proxy_mean |
| 1.25 | 4/4 | True | batch_shift_index,domain_auc_cv,mean_outside_ref_05_95_rate,quality_proxy_mean |
| 1.50 | 4/4 | True | batch_shift_index,domain_auc_cv,mean_outside_ref_05_95_rate,quality_proxy_mean |
| 2.00 | 3/4 | True | batch_shift_index,mean_outside_ref_05_95_rate,quality_proxy_mean |
| 2.50 | 2/4 | False | mean_outside_ref_05_95_rate,quality_proxy_mean |
| 3.00 | 2/4 | False | mean_outside_ref_05_95_rate,quality_proxy_mean |
| 4.00 | 2/4 | False | mean_outside_ref_05_95_rate,quality_proxy_mean |
| 5.00 | 1/4 | False | quality_proxy_mean |
| 10.00 | 1/4 | False | quality_proxy_mean |
| 15.00 | 0/4 | False |  |

## Default 1.5x Leave-one-out

| Held-out Metric | Remaining Metrics | Exceeded Metrics | Severe |
| --- | --- | ---: | --- |
| batch_shift_index | domain_auc_cv,mean_outside_ref_05_95_rate,quality_proxy_mean | 3/3 | True |
| domain_auc_cv | batch_shift_index,mean_outside_ref_05_95_rate,quality_proxy_mean | 3/3 | True |
| mean_outside_ref_05_95_rate | batch_shift_index,domain_auc_cv,quality_proxy_mean | 3/3 | True |
| quality_proxy_mean | batch_shift_index,domain_auc_cv,mean_outside_ref_05_95_rate | 3/3 | True |

## Default 1.5x Single Metrics

| Metric | Strict/Known Max Ratio | Severe Alone |
| --- | ---: | --- |
| batch_shift_index | 2.40x | True |
| domain_auc_cv | 1.87x | True |
| mean_outside_ref_05_95_rate | 4.45x | True |
| quality_proxy_mean | 13.95x | True |

## Result

- default exceeded metrics: 4 / 4
- default leave-one-out all severe: True
- default single-metric severe count: 4 / 4
- promote relaxed gate: False

## Interpretation

The default severe classification is not dependent on one metric if every leave-one-out subset
still classifies strict_external as severe. This supports keeping the hard fallback unless
a separate, stable relaxation rule is justified.