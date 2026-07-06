# Shift Gate Evidence Audit

## Purpose

This audit documents why strict_external deployable mode falls back to v195: strict_external
is the only domain marked `severe_unknown_shift`, and its unsupervised shift metrics are
outside the internal envelope.

## Gate Summary

- severe domains: strict_external
- normal domains: old_data, third_batch
- max strict/internal ratio: 13.945

## Shift Metrics

| Metric | Internal Max | Strict External | Ratio |
| --- | ---: | ---: | ---: |
| batch_shift_index | 0.969486 | 2.324184 | 2.40x |
| domain_auc_cv | 0.531717 | 0.994501 | 1.87x |
| mean_outside_ref_05_95_rate | 0.098856 | 0.439815 | 4.45x |
| quality_proxy_mean | 0.021950 | 0.306096 | 13.95x |