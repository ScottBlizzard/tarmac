# Quota Veto No-Backfill Release Search

## Boundary

- Rank score is `model_disagreement_rate`.
- Veto score is `safe_distance_case_score`.
- Veto removes candidates after quota selection and does not backfill.
- Search and selection use old+third only; strict external is frozen display.

## Selected Policy

- Release fraction: `0.15`.
- Veto threshold: `0.7302192052776949`.
- Internal released: 14; internal new errors: 0.
- Internal remaining review/reject: 182.
- Frozen strict external released: 6; remaining review/reject: 50.
- Frozen strict external new errors: 0; total auto errors: 0.