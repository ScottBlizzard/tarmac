# Strict External Override Candidate Audit

## Purpose

This exploratory audit asks whether a stricter case-level override could be selected using
only old_data and third_batch, then frozen before inspecting strict_external. It does not
change the deployable policy; strict_external remains severe-shift gated.

## Result

- passed: True
- interpretation: `exploratory_frozen_audit_only_not_deployable`
- promote to deployable: False
- selection domains: old_data, third_batch
- frozen audit domains: strict_external
- model disagreement threshold: 0.583333
- DINO FN risk threshold: 0.369693
- selection candidates: 86
- selection candidate errors: 0
- strict external frozen candidates: 22
- strict external frozen candidate errors: 0
- leave-one-known-domain all safe: False

## Summary

| Type | Trained On | Audited Domain | Candidates | Errors | High-risk FN | Safe |
| --- | --- | --- | ---: | ---: | ---: | --- |
| selection_domains | old_data,third_batch | old_data | 43 | 0 | 0 | True |
| selection_domains | old_data,third_batch | third_batch | 43 | 0 | 0 | True |
| frozen_audit | old_data,third_batch | strict_external | 22 | 0 | 0 | True |
| leave_one_known_domain | third_batch | old_data | 43 | 0 | 0 | True |
| leave_one_known_domain | old_data | third_batch | 92 | 6 | 6 | False |

## Interpretation

The selected threshold is useful as a frozen audit probe. It is not promoted because the
leave-one-known-domain audit can be unsafe when the threshold is selected from a single
known domain, demonstrating limited cross-domain stability.