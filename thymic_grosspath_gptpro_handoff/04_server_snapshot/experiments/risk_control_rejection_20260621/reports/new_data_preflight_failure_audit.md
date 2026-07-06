# New Data Preflight Failure Audit

## Purpose

This negative audit confirms that the new-data preflight blocks a label-free runtime
feature CSV if it includes an error-derived selector column.

## Result

- passed: True
- blocked feature rows: 1
- blocked decision rows: 0
- blocked forbidden columns: released_error
- blocked contract passed: False
- blocked decision schema passed: False