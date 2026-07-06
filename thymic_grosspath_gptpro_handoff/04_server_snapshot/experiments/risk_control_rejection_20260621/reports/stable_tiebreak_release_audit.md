# Stable Tie-Break Release Audit

## Rule

- Primary score: `model_disagreement_rate` ascending.
- Secondary score: `safe_distance_case_score` ascending.
- Final deterministic tie-break: `case_id` ascending.

- old_data: released=11, released errors=0, review=64, auto errors=0, high-risk FN=0.
- third_batch: released=18, released errors=2, review=103, auto errors=3, high-risk FN=3.
- strict_external: released=8, released errors=0, review=48, auto errors=0, high-risk FN=0.
- all_domains: released=37, released errors=2, review=215, auto errors=3, high-risk FN=3.