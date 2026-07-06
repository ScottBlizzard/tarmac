# Phase 2 Relaxed Intersection Policy Audit

## Scope

This frozen what-if combines the label-free relaxed shift gate with the Phase 1 known-domain
intersection thresholds. Strict_external labels are used only after candidates are selected
for error counting.

## Gate And Thresholds

- severe multiplier: 2.5
- minimum exceeded metrics: 3
- gate by domain: {'third_batch': 'normal_or_known_shift', 'old_data': 'normal_or_known_shift', 'strict_external': 'normal_or_known_shift'}
- intersection thresholds: {'model_disagreement_rate': 0.5, 'dino_fn_risk_max': 0.3696929763812454}

## Summary

| Domain | Candidates | Gate-allowed Candidates | Errors | High-risk FN |
| --- | ---: | ---: | ---: | ---: |
| old_data | 43 | 43 | 0 | 0 |
| third_batch | 37 | 37 | 0 | 0 |
| strict_external | 21 | 21 | 0 | 0 |
| all_domains | 101 | 101 | 0 | 0 |

## Result

- known allowed candidates: 80
- strict_external allowed candidates: 21
- strict_external allowed errors: 0
- promote to deployable: False

## Interpretation

This is the most permissive Phase 2 frozen what-if so far, because it combines a relaxed
label-free gate with known-domain zero-error intersection thresholds. It is still not promoted
to deployable because the relaxed gate setting itself has not been validated independently.