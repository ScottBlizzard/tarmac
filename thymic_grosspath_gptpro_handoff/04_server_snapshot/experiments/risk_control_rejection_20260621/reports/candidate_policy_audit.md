# Candidate Policy Audit

## Policy

- Base policy: `v195`.
- Release score: `model_disagreement_rate`.
- Release fraction: `0.15` of v195 reviewed/rejected cases per evaluated batch.
- Existing v195 automatic decisions are unchanged.
- Strict external remains frozen audit only.

## Domain Results

- old_data: released=11, review/reject=64, released errors=0, auto errors=0, high-risk FN=0, delta review=-11.
- third_batch: released=18, review/reject=103, released errors=0, auto errors=1, high-risk FN=1, delta review=-18.
- strict_external: released=8, review/reject=48, released errors=0, auto errors=0, high-risk FN=0, delta review=-8.
- all_domains: released=37, review/reject=215, released errors=0, auto errors=1, high-risk FN=1, delta review=-37.

## Fold Summary

- old_data: released=9, released errors sum/max=0/0, released high-risk FN sum/max=0/0, mean review=23.2%.
- third_batch: released=15, released errors sum/max=0/0, released high-risk FN sum/max=0/0, mean review=34.6%.