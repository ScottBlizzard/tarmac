# DINO Shift-Aware Candidate Audit

## Rule

- If a batch has `hard_shift_gate = severe_unknown_shift`, use v195 only.
- Otherwise, release the lowest 15% v195 reviewed cases by `model_disagreement_rate`, then `dino_fn_risk_max`, then `case_id`.
- Strict external remains a frozen audit display.

## Domain Results

- old_data: gate=normal_or_known_shift, released=11, released errors=0, review=64, auto errors=0.
- third_batch: gate=normal_or_known_shift, released=18, released errors=0, review=103, auto errors=1.
- strict_external: gate=severe_unknown_shift, released=0, released errors=0, review=56, auto errors=0.
- all_domains: gate=domain_applied, released=29, released errors=0, review=223, auto errors=1.

## Fold Results

- old_data: released=9, released errors sum/max=0/0, review=66.
- third_batch: released=15, released errors sum/max=0/0, review=106.