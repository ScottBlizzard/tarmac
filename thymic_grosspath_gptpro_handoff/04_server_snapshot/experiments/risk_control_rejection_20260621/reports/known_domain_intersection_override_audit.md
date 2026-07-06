# Known-domain Intersection Override Audit

## Purpose

This audit takes the componentwise conservative intersection of the zero-error thresholds
selected separately on each known domain. It then freezes that intersection and audits
old_data, third_batch, and strict_external. This remains an exploratory frozen audit and
does not modify deployable behavior.

## Result

- passed: True
- interpretation: `conservative_intersection_frozen_audit_only_not_deployable`
- promote to deployable: False
- intersection model disagreement threshold: 0.500000
- intersection DINO FN risk threshold: 0.369693
- known-domain candidates: 80
- known-domain candidate errors: 0
- strict external frozen candidates: 21
- strict external frozen candidate errors: 0

## Single-domain Thresholds

| Domain | Disagreement Threshold | DINO FN Risk Threshold |
| --- | ---: | ---: |
| old_data | 0.500000 | 0.877680 |
| third_batch | 0.583333 | 0.369693 |

## Summary

| Type | Trained On | Audited Domain | Candidates | Errors | High-risk FN | Safe |
| --- | --- | --- | ---: | ---: | ---: | --- |
| single_known_domain_threshold | old_data | old_data | 46 | 0 | 0 | True |
| single_known_domain_threshold | third_batch | third_batch | 43 | 0 | 0 | True |
| intersection_known_domains | old_data,third_batch | old_data | 43 | 0 | 0 | True |
| intersection_known_domains | old_data,third_batch | third_batch | 37 | 0 | 0 | True |
| intersection_frozen_audit | old_data,third_batch | strict_external | 21 | 0 | 0 | True |

## Interpretation

This intersection removes the known leave-one-domain unsafe expansion while preserving
a nonzero strict_external frozen candidate set. It is still not promoted because strict_external
is severe-shift gated in deployable mode and the known-domain evidence remains limited.