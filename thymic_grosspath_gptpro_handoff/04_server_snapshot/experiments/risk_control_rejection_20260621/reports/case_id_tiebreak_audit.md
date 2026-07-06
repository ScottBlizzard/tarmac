# Case ID Tie-Break Audit

## Rule

- Primary score: `model_disagreement_rate` ascending.
- Deterministic tie-break: `case_id` ascending.

## Domain Results

- old_data: released=11, released errors=0, review=64, auto errors=0.
- third_batch: released=18, released errors=0, review=103, auto errors=1.
- strict_external: released=8, released errors=0, review=48, auto errors=0.
- all_domains: released=37, released errors=0, review=215, auto errors=1.

## Fold Results

- old_data: released=9, released errors sum/max=0/0, review=66.
- third_batch: released=15, released errors sum/max=0/0, review=106.