# Existing External Casebook Audit

## Purpose

This audit enriches label-free deployable decisions for the project's existing external
domains with post-decision labels and risk features. It inspects which reviewed cases were
released, which remained in review, and which strict_external cases would have been
case-only release candidates but were blocked by the severe-shift deployable fallback.

## Result

- passed: True
- casebook rows: 414
- released rows: 18
- released errors: 0
- released high-risk FN: 0
- strict external released rows: 0
- strict external shift-blocked release candidates: 8

## Status Summary

| Domain | Status | n | Released | Would-be Errors | Would-be High-risk FN | Released Errors | Released High-risk FN | DINO Missing | Rank Range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict_external | reviewed_not_released | 48 | 0 | 0 | 0 | 0 | 0 | 34 | 9-56 |
| strict_external | shift_blocked_release_candidate | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 1-8 |
| strict_external | v195_auto_retained | 52 | 0 | 0 | 0 | 0 | 0 | 13 | - |
| third_batch | released_from_review | 18 | 18 | 0 | 0 | 0 | 0 | 0 | 1-18 |
| third_batch | reviewed_not_released | 103 | 0 | 8 | 6 | 0 | 0 | 21 | 19-121 |
| third_batch | v195_auto_retained | 185 | 0 | 1 | 1 | 0 | 0 | 56 | - |

## Released Or Shift-blocked Cases

| Domain | Status | Case ID | Label | y | pred | Rank | Disagreement | DINO FN Risk | Error | High-risk FN |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| strict_external | shift_blocked_release_candidate | doctor_ext_001_1700034 | AB | 0 | 0 | 1 | 0.000000 | 0.175911 | False | False |
| strict_external | shift_blocked_release_candidate | doctor_ext_033_2600883 | AB | 0 | 0 | 2 | 0.000000 | 0.226565 | False | False |
| strict_external | shift_blocked_release_candidate | doctor_ext_039_2440727 | A | 0 | 0 | 3 | 0.000000 | 0.232843 | False | False |
| strict_external | shift_blocked_release_candidate | doctor_ext_017_2321137 | AB | 0 | 0 | 4 | 0.000000 | 0.247655 | False | False |
| strict_external | shift_blocked_release_candidate | doctor_ext_053_2441583 | B1 | 0 | 0 | 5 | 0.000000 | 0.249807 | False | False |
| strict_external | shift_blocked_release_candidate | doctor_ext_026_2444000 | AB | 0 | 0 | 6 | 0.000000 | 0.250778 | False | False |
| strict_external | shift_blocked_release_candidate | doctor_ext_015_2308319 | AB | 0 | 0 | 7 | 0.000000 | 0.252699 | False | False |
| strict_external | shift_blocked_release_candidate | doctor_ext_003_1805322 | AB | 0 | 0 | 8 | 0.000000 | 0.266641 | False | False |
| third_batch | released_from_review | third_AB_2515958 | AB | 0 | 0 | 1 | 0.000000 | 0.259258 | False | False |
| third_batch | released_from_review | third_AB_2406696 | AB | 0 | 0 | 2 | 0.000000 | 0.264317 | False | False |
| third_batch | released_from_review | third_AB_2313987 | AB | 0 | 0 | 3 | 0.000000 | 0.294915 | False | False |
| third_batch | released_from_review | third_AB_2510224 | AB | 0 | 0 | 4 | 0.000000 | 0.320591 | False | False |
| third_batch | released_from_review | third_AB_2500842 | AB | 0 | 0 | 5 | 0.000000 | 0.328258 | False | False |
| third_batch | released_from_review | third_AB_2512465 | AB | 0 | 0 | 6 | 0.000000 | 0.337418 | False | False |
| third_batch | released_from_review | third_AB_2521040 | AB | 0 | 0 | 7 | 0.000000 | 0.345039 | False | False |
| third_batch | released_from_review | third_AB_2500987 | AB | 0 | 0 | 8 | 0.000000 | 0.348514 | False | False |
| third_batch | released_from_review | third_AB_2513157 | AB | 0 | 0 | 9 | 0.000000 | 0.353859 | False | False |
| third_batch | released_from_review | third_AB_2510659 | AB | 0 | 0 | 10 | 0.000000 | 0.358084 | False | False |
| third_batch | released_from_review | third_AB_2421733 | AB | 0 | 0 | 11 | 0.000000 | 0.365535 | False | False |
| third_batch | released_from_review | third_AB_2412683 | AB | 0 | 0 | 12 | 0.000000 | 0.379128 | False | False |
| third_batch | released_from_review | third_AB_2509110 | AB | 0 | 0 | 13 | 0.000000 | 0.380023 | False | False |
| third_batch | released_from_review | third_AB_2412764 | AB | 0 | 0 | 14 | 0.000000 | 0.383177 | False | False |
| third_batch | released_from_review | third_AB_2506622 | AB | 0 | 0 | 15 | 0.000000 | 0.385157 | False | False |
| third_batch | released_from_review | third_AB_2213730 | AB | 0 | 0 | 16 | 0.000000 | 0.393466 | False | False |
| third_batch | released_from_review | third_AB_2517828 | AB | 0 | 0 | 17 | 0.000000 | 0.398902 | False | False |
| third_batch | released_from_review | third_AB_2200377 | AB | 0 | 0 | 18 | 0.000000 | 0.399299 | False | False |