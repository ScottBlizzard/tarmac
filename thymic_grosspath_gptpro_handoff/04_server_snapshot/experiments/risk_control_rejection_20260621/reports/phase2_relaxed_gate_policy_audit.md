# Phase 2 Relaxed Gate Policy Audit

## Scope

This is a frozen what-if audit. The relaxed gate is selected from label-free shift metrics,
and strict_external labels are used only after decisions for error counting.

## Gate Setting

- severe multiplier: 2.5
- minimum exceeded metrics: 3
- gate by domain: {'third_batch': 'normal_or_known_shift', 'old_data': 'normal_or_known_shift', 'strict_external': 'normal_or_known_shift'}

## Summary

| Variant | Scope | Released | Released Errors | High-risk FN | Auto | Review |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current_hard_gate | old_data | 11 | 0 | 0 | 221 | 64 |
| current_hard_gate | third_batch | 18 | 0 | 0 | 203 | 103 |
| current_hard_gate | strict_external | 0 | 0 | 0 | 52 | 56 |
| current_hard_gate | all_domains | 29 | 0 | 0 | 476 | 223 |
| relaxed_gate_what_if | old_data | 11 | 0 | 0 | 221 | 64 |
| relaxed_gate_what_if | third_batch | 18 | 0 | 0 | 203 | 103 |
| relaxed_gate_what_if | strict_external | 8 | 0 | 0 | 60 | 48 |
| relaxed_gate_what_if | all_domains | 37 | 0 | 0 | 484 | 215 |

## Result

- current strict_external releases: 0
- relaxed strict_external releases: 8
- relaxed strict_external released errors: 0
- promote to deployable: False

## Interpretation

This audit quantifies the potential coverage gain if the severe gate were relaxed enough to
treat strict_external as normal/known shift. It is not a deployment recommendation; Phase 2
still needs stronger evidence before changing the hard gate.