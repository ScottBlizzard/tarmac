# Candidate Lock Audit

## Purpose

This audit prevents the aggregate 20% stress result from accidentally replacing the
fold-safe selected candidate. The locked candidate is the simple `domain@15%` policy.

## Result

- passed: True
- selected granularity: `domain`
- selected release fraction: `0.15`
- contract granularity: `domain`
- contract release fraction: `0.15`
- frontier max released_n_sum: `24`
- selected frontier released_n_sum: `24`