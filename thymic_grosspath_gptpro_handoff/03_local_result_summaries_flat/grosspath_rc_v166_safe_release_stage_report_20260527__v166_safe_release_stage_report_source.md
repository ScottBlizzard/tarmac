# v165 Safe-release Evidence Pack

## Current Main Result

The current high-safety efficiency candidate is v161 safe-release: all-domain BAcc 99.8%, review/reject 57.5%, auto-pass 42.5%, FN=1, FP=0.
On strict external, the same workflow gives BAcc 100.0%, review/reject 68.5%, FN=0, FP=0.

## Key Ablation

Old-only safe-release is the negative control: it lowers all-domain review to 38.9%, but releases 5 errors and leaves FN=5, FP=1. This supports the multi-domain constraint design.

## Remaining Error

After v161, the remaining automatic error table contains 1 case(s). This should be kept in the Results boundary rather than hidden.

## Files

- v165_main_operating_results.csv
- v165_safe_release_rules.csv
- v165_safe_release_ablation.csv
- v165_frontier_constraint_selection.csv
- v165_remaining_auto_errors_after_safe_release.csv
- v165_claim_evidence_map.csv