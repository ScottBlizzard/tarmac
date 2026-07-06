# Runtime Decision Export

## Purpose

This export removes audit-only columns before applying the final policy, then checks that
the decisions match the audit-table path. It demonstrates that release/review decisions do
not depend on labels or fold metadata.

## Result

- decision rows: 1398
- parity rows compared: 1398
- mismatch rows: 0
- audit columns removed: label_idx, task_l6_label, fold_id