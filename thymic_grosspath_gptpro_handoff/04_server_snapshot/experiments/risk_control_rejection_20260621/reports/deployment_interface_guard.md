# Deployment Interface Guard

## Purpose

This guard checks that runtime decision exports do not carry audit-only labels or fold
metadata, and that their label-free parity reports have zero mismatches.

## Results

| Export | Passed | Rows | Forbidden Columns | Missing Required Columns | Parity Mismatches |
| --- | --- | ---: | --- | --- | ---: |
| runtime_decision_export | True | 1398 | - | - | 0 |
| adapter_runtime_decision_export | True | 1398 | - | - | 0 |