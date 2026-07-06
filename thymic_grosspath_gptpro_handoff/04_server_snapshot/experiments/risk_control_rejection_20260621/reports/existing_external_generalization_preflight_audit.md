# Existing External Generalization Preflight Audit

## Purpose

This audit uses the project's existing external/generalization domains as deployment
simulation inputs. The runtime frame removes audit-only columns before invoking the
deployable wrapper; labels are used only after decisions are produced to summarize
released errors and high-risk false negatives.

## Result

- passed: True
- external domains: third_batch, strict_external
- decision rows: 414
- released rows: 18
- audit columns removed: label_idx, task_l6_label, fold_id
- forbidden columns present in decisions: -
- strict external release ok: True

## Domain Summary

| Domain | Rows | Decisions | Released | Review/Reject | Strict Released | Released Errors | High-risk FN | Contract | Schema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| third_batch | 306 | 306 | 18 | 103 | 0 | 0 | 0 | True | True |
| strict_external | 108 | 108 | 0 | 56 | 0 | 0 | 0 | True | True |