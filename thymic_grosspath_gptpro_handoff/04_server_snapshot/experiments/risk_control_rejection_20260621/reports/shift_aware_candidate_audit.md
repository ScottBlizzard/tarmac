# Shift-Aware Candidate Audit

## Rule

- If a batch has `hard_shift_gate = severe_unknown_shift`, use v195 only and release no reviewed cases.
- Otherwise, release the lowest 15% v195 reviewed cases by `model_disagreement_rate`, with `case_id` tie-break.
- Strict external remains a frozen audit display.

## Domain Results

- old_data: gate=normal_or_known_shift, released=11, released errors=0, review=64, auto errors=0.
- third_batch: gate=normal_or_known_shift, released=18, released errors=0, review=103, auto errors=1.
- strict_external: gate=severe_unknown_shift, released=0, released errors=0, review=56, auto errors=0.
- all_domains: gate=domain_applied, released=29, released errors=0, review=223, auto errors=1.

## Fold Results

- old_data: released=9, released errors sum/max=0/0, review=66.
- third_batch: released=15, released errors sum/max=0/0, review=106.

## Interpretation

This is the conservative deployment interpretation of the Phase 1 plan. Under the existing
unsupervised hard gate, strict external is classified as severe unknown shift, so the release
layer does not lower its review count. The 48-review strict-external result from the case-only
candidate should therefore be treated as an exploratory frozen display, not the hard-gated
deployment result.