# Policy Mode Comparison

## Modes

- `baseline`: reconstructed v195.
- `case_only_exploratory`: v195 plus 15% low-disagreement release with `dino_fn_risk_max` and `case_id` tie-break.
- `shift_aware_deployable`: same DINO tie-break release rule, but severe unknown shift batches fall back to v195.

## Results

| Scope | Mode | Auto | Review | Delta Review | Released | Released Errors | Auto Errors | High-risk FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_domains | baseline | 447 | 252 | 0 | 0 | 0 | 1 | 1 |
| all_domains | case_only_exploratory | 484 | 215 | -37 | 37 | 0 | 1 | 1 |
| all_domains | shift_aware_deployable | 476 | 223 | -29 | 29 | 0 | 1 | 1 |
| old_data | baseline | 210 | 75 | 0 | 0 | 0 | 0 | 0 |
| old_data | case_only_exploratory | 221 | 64 | -11 | 11 | 0 | 0 | 0 |
| old_data | shift_aware_deployable | 221 | 64 | -11 | 11 | 0 | 0 | 0 |
| strict_external | baseline | 52 | 56 | 0 | 0 | 0 | 0 | 0 |
| strict_external | case_only_exploratory | 60 | 48 | -8 | 8 | 0 | 0 | 0 |
| strict_external | shift_aware_deployable | 52 | 56 | 0 | 0 | 0 | 0 | 0 |
| third_batch | baseline | 185 | 121 | 0 | 0 | 0 | 1 | 1 |
| third_batch | case_only_exploratory | 203 | 103 | -18 | 18 | 0 | 1 | 1 |
| third_batch | shift_aware_deployable | 203 | 103 | -18 | 18 | 0 | 1 | 1 |

## Interpretation

The DINO case-only candidate shows the maximum frozen strict-external benefit observed so far,
but it does not enforce the severe-shift hard fallback. The shift-aware deployable mode is
the conservative Phase 1 deployment interpretation: old_data and third_batch receive the
audited 15% release, while strict_external remains at the v195 review count because its
unsupervised batch gate is `severe_unknown_shift`.