# Phase 3 Integration Readiness Audit

## Purpose

This audit converts Phase 2 results into a strict integration boundary. It does not modify
the original project. It only states which experiment outputs may be used if original project
integration is requested later.

## Recommendation

`integrate_hard_gate_only`

## Allowed Items

- `hard_gate_deployable_policy`: copy_or_adapt_only_this_deployable_policy_if_original_project_is_modified

## Manifest

| Item | Status | Passed | Evidence | Action |
| --- | --- | --- | --- | --- |
| `phase2_reproducibility` | guard_passed | True | steps=31/31, all_passed=True | required_before_any_integration |
| `hard_gate_deployable_policy` | allowed | True | granularity=domain, fraction=0.15, third_released=18, strict_released=0, fold_safe=True | copy_or_adapt_only_this_deployable_policy_if_original_project_is_modified |
| `relaxed_strict_external_quota15` | blocked | False | released=8, errors=0, promote=False | keep_as_frozen_what_if_only |
| `relaxed_strict_external_intersection` | blocked | False | candidates=21, errors=0, promote=False | keep_as_frozen_what_if_only |
| `case_level_robust_core_strict_candidates` | blocked | False | candidates=21, errors=0, promote=False | keep_as_frozen_what_if_only |
| `shift_metric_hard_gate_evidence` | guard_passed | True | default_leave_one_out_all_severe=True, promote_relaxed_gate=False | retain_hard_gate_evidence_in_methods_and_audit_docs |
| `leave_one_known_relaxation_guard` | guard_passed | True | strict_gate_stable=False, promote_relaxed_gate=False | block_gate_relaxation_until_independent_validation |

## Interpretation

The only integration-ready policy is the hard-gate deployable policy. The relaxed
strict_external quota and intersection results must remain frozen what-if evidence.