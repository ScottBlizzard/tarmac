# Shift Gate Margin Audit

## Scope

This audit uses unsupervised batch-shift metrics only. It does not use strict_external labels
to tune a threshold.

## Metric Margins

| Metric | Internal Max | Strict External | Gap | Ratio |
| --- | ---: | ---: | ---: | ---: |
| batch_shift_index | 0.969486 | 2.324184 | 1.354698 | 2.40x |
| domain_auc_cv | 0.531717 | 0.994501 | 0.462784 | 1.87x |
| mean_outside_ref_05_95_rate | 0.098856 | 0.439815 | 0.340959 | 4.45x |
| quality_proxy_mean | 0.021950 | 0.306096 | 0.284146 | 13.95x |

## Strict External Policy Consequence

| Mode | Auto | Review | Delta Review | Released | Released Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 52 | 56 | 0 | 0 | 0 |
| case_only_exploratory | 60 | 48 | -8 | 8 | 0 |
| shift_aware_deployable | 52 | 56 | 0 | 0 | 0 |

## Interpretation

The strict_external batch is far outside the old/third internal envelope on every available
unsupervised shift metric. This supports treating the hard-gated deployable policy separately
from the case-only exploratory frozen display.