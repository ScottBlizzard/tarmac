# Candidate Subgroup Audit

## Purpose

This audit checks whether the current candidate policy concentrates releases or errors in specific labels or domains.

## Label Summary

- AB: n=301, released=30, released errors=0, auto errors=0.
- B1: n=73, released=5, released errors=0, auto errors=0.
- B2: n=117, released=1, released errors=0, auto errors=0.
- TC: n=115, released=1, released errors=0, auto errors=1.
- A: n=52, released=0, released errors=0, auto errors=0.
- B3: n=34, released=0, released errors=0, auto errors=0.
- B2_B3_mixed: n=3, released=0, released errors=0, auto errors=0.
- MNT_assumed_low: n=3, released=0, released errors=0, auto errors=0.
- B1_B2_mixed: n=1, released=0, released errors=0, auto errors=0.

## Domain + Label Released Groups

- third_batch / AB: n=212, released=18, released errors=0, high-risk FN=0, release rate=8.5%.
- strict_external / AB: n=39, released=8, released errors=0, high-risk FN=0, release rate=20.5%.
- old_data / B1: n=50, released=5, released errors=0, high-risk FN=0, release rate=10.0%.
- old_data / AB: n=50, released=4, released errors=0, high-risk FN=0, release rate=8.0%.
- old_data / B2: n=60, released=1, released errors=0, high-risk FN=0, release rate=1.7%.
- old_data / TC: n=57, released=1, released errors=0, high-risk FN=0, release rate=1.8%.