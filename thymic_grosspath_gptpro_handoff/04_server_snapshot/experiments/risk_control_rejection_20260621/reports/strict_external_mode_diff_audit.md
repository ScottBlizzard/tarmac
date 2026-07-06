# Strict External Mode Difference Audit

## Purpose

This audit separates strict_external frozen-display releases from deployable behavior.
`case_only_exploratory` may show releases for scientific audit, but `shift_aware_deployable`
falls back to v195 because strict_external is marked `severe_unknown_shift`.

## Summary

- case-only strict_external releases: 8
- deployable strict_external releases: 0
- frozen-only releases: 8
- frozen-only released errors: 0
- frozen-only released high-risk FN: 0
- deployable reason: `severe_unknown_shift_fallback_to_v195`

## Frozen-only Cases

| Case ID | Label | y | pred | Disagreement | DINO FN Risk | Released Error | Status | Deployable Action |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| doctor_ext_001_1700034 | AB | 0 | 0 | 0.000000 | 0.175911 | False | frozen_display_only | v195_fallback_review |
| doctor_ext_003_1805322 | AB | 0 | 0 | 0.000000 | 0.266641 | False | frozen_display_only | v195_fallback_review |
| doctor_ext_015_2308319 | AB | 0 | 0 | 0.000000 | 0.252699 | False | frozen_display_only | v195_fallback_review |
| doctor_ext_017_2321137 | AB | 0 | 0 | 0.000000 | 0.247655 | False | frozen_display_only | v195_fallback_review |
| doctor_ext_026_2444000 | AB | 0 | 0 | 0.000000 | 0.250778 | False | frozen_display_only | v195_fallback_review |
| doctor_ext_033_2600883 | AB | 0 | 0 | 0.000000 | 0.226565 | False | frozen_display_only | v195_fallback_review |
| doctor_ext_039_2440727 | A | 0 | 0 | 0.000000 | 0.232843 | False | frozen_display_only | v195_fallback_review |
| doctor_ext_053_2441583 | B1 | 0 | 0 | 0.000000 | 0.249807 | False | frozen_display_only | v195_fallback_review |