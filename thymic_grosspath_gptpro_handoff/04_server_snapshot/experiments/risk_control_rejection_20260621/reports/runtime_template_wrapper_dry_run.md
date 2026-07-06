# Runtime Template Wrapper Dry Run

## Purpose

This dry run sends the label-free runtime feature template through the deployable wrapper
to confirm that a new-data-shaped input can produce deployment decisions without labels,
fold metadata, or error-derived selector columns.

## Result

- passed: True
- contract passed: True
- template rows: 1
- decision rows: 1
- released rows: 0
- review/reject rows: 0
- strict external released rows: 0
- audit-only columns present: -
- forbidden selector columns present: -

## Decision Columns

| Position | Column |
| ---: | --- |
| 1 | `policy_mode` |
| 2 | `domain` |
| 3 | `case_id` |
| 4 | `v195_auto_decision` |
| 5 | `v195_review_or_reject` |
| 6 | `hard_shift_gate` |
| 7 | `release_from_v195_review` |
| 8 | `auto_decision` |
| 9 | `review_or_reject` |
| 10 | `final_pred` |
| 11 | `release_fraction` |