# Locked Candidate Boundary Margin Audit

## Purpose

This audit inspects the next reviewed cases immediately after the locked `domain@15%`
release cutoff. It checks whether a small relaxation would expose errors.

## Domain Summary

| Domain | Released | Next n | Next Errors | Next High-risk FN | Cutoff Disagreement | Cutoff DINO | Next Case IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old_data | 11 | 5 | 0 | 0 | 0.000000 | 0.194529 | batch2_2120339;batch1_2404744;batch2_2118344;batch1_2213215;batch2_2111334 |
| third_batch | 18 | 5 | 0 | 0 | 0.000000 | 0.399299 | third_AB_2409353;third_AB_2606116;third_AB_2421456;third_B1_2501379;third_AB_2308722 |
| strict_external | 8 | 5 | 0 | 0 | 0.000000 | 0.266641 | doctor_ext_022_2403653;doctor_ext_041_2518212;doctor_ext_011_2244122;doctor_ext_036_2612426;doctor_ext_027_2447503 |

## Next-case Errors

No next reviewed case in the lookahead band is an error.