# Synthetic Deployment Scenario Audit

## Purpose

This audit exercises the deployable wrapper with a small label-free scenario containing
normal-shift reviewed cases, severe-shift reviewed cases, and a severe-shift v195-auto
case. It verifies deployment semantics without using audit labels.

## Result

- passed: True
- contract passed: True
- release fraction: 0.5
- feature rows: 4
- decision rows: 4
- released rows: 1
- review/reject rows: 2
- strict external released rows: 0
- audit-only columns present: -
- forbidden selector columns present: -

## Checks

| Check | Passed |
| --- | --- |
| `normal_low_released` | True |
| `normal_high_not_released` | True |
| `severe_review_not_released` | True |
| `severe_review_stays_review` | True |
| `severe_v195_auto_stays_auto` | True |