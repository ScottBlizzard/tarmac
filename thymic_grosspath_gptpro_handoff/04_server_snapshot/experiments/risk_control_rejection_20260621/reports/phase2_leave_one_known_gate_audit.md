# Phase 2 Leave-one-known Gate Audit

## Scope

This audit calibrates the relaxed severe-shift gate on one known domain at a time, then
checks the held-out known domain and strict_external. It uses label-free shift metrics only.

## Result

- severe multiplier: 2.5
- minimum exceeded metrics: 3
- heldout known stable non-severe: True
- strict gate stable across known domains: False
- promote relaxed gate: False

## Rows

| Train Domain | Heldout Known | Heldout Exceeded | Heldout Severe | Strict Exceeded | Strict Severe |
| --- | --- | ---: | --- | ---: | --- |
| old_data | third_batch | 0 | False | 2 | False |
| third_batch | old_data | 0 | False | 3 | True |

## Interpretation

The relaxed gate should not be promoted unless it is stable across known-domain calibration
choices. If strict_external flips between severe and non-severe depending on which known
domain is used for calibration, the relaxed gate is too sensitive for deployment.