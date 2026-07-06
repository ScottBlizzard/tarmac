# Phase 2 Case-level Robust Core Grid

## Scope

This audit shrinks the Phase 1 known-domain intersection thresholds to find lower-risk
case-level cores. Strict_external labels are used only after candidate selection for frozen
error counting.

## Base Thresholds

- model_disagreement_rate: 0.500000
- dino_fn_risk_max: 0.369693

## Grid

| Shrink | Disagreement Thr | DINO Thr | Known Candidates | Known Errors | Strict Candidates | Strict Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.500000 | 0.369693 | 80 | 0 | 21 | 0 |
| 0.90 | 0.450000 | 0.332724 | 47 | 0 | 19 | 0 |
| 0.75 | 0.375000 | 0.277270 | 34 | 0 | 12 | 0 |
| 0.50 | 0.250000 | 0.184846 | 18 | 0 | 1 | 0 |
| 0.25 | 0.125000 | 0.092423 | 0 | 0 | 0 | 0 |

## Result

- selected safe shrink factor: 1.0
- selected strict_external candidates: 21
- selected strict_external errors: 0
- promote to deployable: False

## Interpretation

This audit can identify a stricter case-level core, but it does not solve the batch-gate
problem. Any strict_external candidate count remains frozen what-if evidence unless the
batch gate itself is justified independently.