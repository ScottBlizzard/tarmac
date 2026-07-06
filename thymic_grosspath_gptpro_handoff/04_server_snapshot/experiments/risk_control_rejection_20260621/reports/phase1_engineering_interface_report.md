# Phase 1 Engineering Interface Report

## Scope

This stage did not modify existing project code or historical outputs. All new work stayed under:

```text
experiments/risk_control_rejection_20260621/
```

The purpose was to move the selected DINO tie-break candidate from an audit-only script toward a
stable, reproducible interface that future code changes can follow.

## What Was Added

### Stable Feature Interface

New script:

```text
scripts/stable_policy_feature_interface.py
```

Outputs:

```text
outputs/stable_policy_features.csv
outputs/stable_policy_feature_manifest.json
reports/stable_policy_feature_interface.md
```

The stable table materializes the selected runtime features:

```text
policy_feature_version = risk_control_v1
model_disagreement_rate
dino_fn_risk_max
dino_fn_risk_missing
hard_shift_gate
v195_review_or_reject
v195_auto_decision
final_pred
```

`dino_fn_risk_max` still comes upstream from the historical v191 DINO/FN sentinel artifact, but the
policy runner no longer reads v191 directly. This reduces the maintenance risk when the policy is
implemented later.

Important finding:

```text
dino_fn_risk_missing = 266 / 699 cases
released cases with missing dino_fn_risk_max = 0
```

Missing DINO risk values are sorted as low priority for release.

### Production Adapter Prototype

New script:

```text
scripts/production_adapter_prototype.py
```

Outputs:

```text
outputs/adapter_policy_features.csv
outputs/production_adapter_report.json
reports/production_adapter_prototype.md
```

The adapter maps existing project outputs into the same `risk_control_v1` feature interface used by
the isolated runner. It is still isolated in the experiment directory and does not change production
code.

Current adapter result:

```text
rows = 699
v195_review_or_reject_n = 252
dino_fn_risk_missing = 266
old_data = 285
third_batch = 306
strict_external = 108
```

The adapter was also run through the final policy runner:

```text
outputs/adapter_final_policy_runner_cases.csv
outputs/adapter_final_policy_runner_summary.csv
outputs/adapter_final_policy_runner_report.json
```

The adapter runner summary matched the stable-feature runner summary exactly on all compared key
counts:

```text
matched summary rows = 8
mismatched counts = 0
```

### Executable Policy Contract Validation

New script:

```text
scripts/policy_contract_validator.py
```

Outputs:

```text
outputs/policy_contract_validation_report.json
reports/policy_contract_validation.md
```

This validates both the stable feature table and adapter feature table against:

```text
configs/final_policy_contract.json
```

Current result:

```text
stable_policy_features.csv: passed=true, rows=699, forbidden_columns_present=0
adapter_policy_features.csv: passed=true, rows=699, forbidden_columns_present=0
dino_fn_risk_max null count: 266 in both tables
```

The null DINO count is accepted because `dino_fn_risk_missing` is a required explicit companion
flag and missing DINO values are sorted last by the release rule.

### Label-free Runtime Decision Export

New script:

```text
scripts/runtime_decision_export.py
```

Outputs:

```text
outputs/runtime_decision_export.csv
outputs/runtime_decision_export_parity_diff.csv
outputs/runtime_decision_export_report.json
outputs/adapter_runtime_decision_export.csv
outputs/adapter_runtime_decision_export_parity_diff.csv
outputs/adapter_runtime_decision_export_report.json
reports/runtime_decision_export.md
```

This export drops audit-only columns before applying the final policy:

```text
label_idx
task_l6_label
fold_id
```

Current result:

```text
stable feature path decision rows = 1398
adapter feature path decision rows = 1398
audit-column parity mismatch rows = 0
stable-vs-adapter runtime decision mismatch rows = 0
```

This confirms that release/review decisions do not depend on labels or fold metadata. Labels are
only needed by audit summaries.

### Isolated Final Policy Runner

New script:

```text
scripts/final_policy_runner.py
```

Outputs:

```text
outputs/final_policy_runner_cases.csv
outputs/final_policy_runner_summary.csv
outputs/final_policy_runner_report.json
reports/final_policy_runner.md
```

The runner supports two modes:

```text
case_only_exploratory
shift_aware_deployable
```

Rules:

```text
Release only v195 reviewed/rejected cases.
Release fraction = 15% within each domain/batch.
Primary sort = model_disagreement_rate ascending.
Secondary sort = dino_fn_risk_max ascending.
Residual tie-break = case_id ascending.
Deployable mode falls back to v195 for severe_unknown_shift.
```

## Runner Results

| Mode | Scope | Auto | Review | Released | Released Errors | Auto Errors | High-risk FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| case_only_exploratory | old_data | 221 | 64 | 11 | 0 | 0 | 0 |
| case_only_exploratory | third_batch | 203 | 103 | 18 | 0 | 1 | 1 |
| case_only_exploratory | strict_external | 60 | 48 | 8 | 0 | 0 | 0 |
| case_only_exploratory | all_domains | 484 | 215 | 37 | 0 | 1 | 1 |
| shift_aware_deployable | old_data | 221 | 64 | 11 | 0 | 0 | 0 |
| shift_aware_deployable | third_batch | 203 | 103 | 18 | 0 | 1 | 1 |
| shift_aware_deployable | strict_external | 52 | 56 | 0 | 0 | 0 | 0 |
| shift_aware_deployable | all_domains | 476 | 223 | 29 | 0 | 1 | 1 |

## Consistency Audit

New script:

```text
scripts/final_policy_consistency_audit.py
```

Outputs:

```text
outputs/final_policy_consistency_diff.csv
outputs/final_policy_consistency_report.json
reports/final_policy_consistency_audit.md
```

Result:

```text
mismatch_rows = 0
```

This means the final runner reproduces the selected DINO case-only and DINO shift-aware audit
summaries on all compared key counts.

## Verification

Fresh command run:

```bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python -m unittest experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py -v
python experiments/risk_control_rejection_20260621/scripts/stable_policy_feature_interface.py
python experiments/risk_control_rejection_20260621/scripts/production_adapter_prototype.py
python experiments/risk_control_rejection_20260621/scripts/final_policy_runner.py
python experiments/risk_control_rejection_20260621/scripts/final_policy_runner.py --features experiments/risk_control_rejection_20260621/outputs/adapter_policy_features.csv --output-prefix adapter_final_policy_runner
python experiments/risk_control_rejection_20260621/scripts/final_policy_consistency_audit.py
python experiments/risk_control_rejection_20260621/scripts/policy_contract_validator.py
python experiments/risk_control_rejection_20260621/scripts/runtime_decision_export.py
python experiments/risk_control_rejection_20260621/scripts/runtime_decision_export.py --features experiments/risk_control_rejection_20260621/outputs/adapter_policy_features.csv --output-prefix adapter_runtime_decision_export
python experiments/risk_control_rejection_20260621/scripts/known_domain_intersection_override_audit.py
python experiments/risk_control_rejection_20260621/scripts/phase1_reproducibility_runner.py
```

Observed:

```text
Ran 72 tests
OK
final_policy_consistency mismatch_rows = 0
adapter runner summary mismatched counts = 0
policy contract validation passed for stable and adapter feature tables
runtime decision audit-column parity mismatch rows = 0
runtime decision stable-vs-adapter mismatch rows = 0
phase1 reproducibility runner all_passed = true
```

### Input Order Invariance Audit

New script:

```text
scripts/input_order_invariance_audit.py
```

Outputs:

```text
outputs/input_order_invariance_diff.csv
outputs/input_order_invariance_report.json
outputs/adapter_input_order_invariance_diff.csv
outputs/adapter_input_order_invariance_report.json
reports/input_order_invariance_audit.md
```

Result:

```text
shuffle seeds = 0, 1, 2, 7, 13, 29, 101
stable feature path mismatch rows = 0 / 1398
adapter feature path mismatch rows = 0 / 1398
```

This verifies that the final decision export is not sensitive to input CSV row order.

### One-command Reproducibility Runner

New script:

```text
scripts/phase1_reproducibility_runner.py
```

Outputs:

```text
outputs/phase1_reproducibility_runner_report.json
reports/phase1_reproducibility_runner.md
```

Current result:

```text
steps_planned = 37
steps_run = 37
all_passed = true
unit tests = 72 passed
```

The runner rebuilds stable features, adapter features, final summaries, contract validation,
runtime decision exports, input-order invariance checks, DINO missing-value sensitivity outputs,
and fraction/granularity pressure audits.

The new pressure-audit steps are:

```text
scripts/fraction_granularity_stress_audit.py
scripts/fold_fraction_granularity_stress_audit.py
scripts/fold_boundary_failure_casebook.py
scripts/fold_safe_frontier.py
scripts/candidate_lock_audit.py
scripts/locked_candidate_subgroup_stress_audit.py
scripts/locked_candidate_boundary_margin_audit.py
scripts/strict_external_mode_diff_audit.py
scripts/shift_gate_evidence_audit.py
scripts/deployment_readiness_matrix.py
scripts/final_safety_claims_audit.py
scripts/deployment_interface_guard.py
scripts/runtime_feature_template_audit.py
scripts/runtime_template_wrapper_dry_run.py
scripts/synthetic_deployment_scenario_audit.py
scripts/integration_smoke_audit.py
integration/risk_control_deployable_wrapper.py
scripts/wrapper_decision_schema_audit.py
scripts/new_data_preflight_runner.py
scripts/new_data_preflight_failure_audit.py
scripts/existing_external_generalization_preflight_audit.py
scripts/existing_external_casebook_audit.py
scripts/strict_external_override_candidate_audit.py
scripts/known_domain_intersection_override_audit.py
scripts/phase1_handoff_manifest.py
```

The fold audit is the safety gate. Aggregate internal-known metrics can make 20% look acceptable,
but fold-level third_batch auditing rejects 17.5% and 20% because they release one high-risk FN.
The boundary casebook identifies that failure as `third_B2_2517222`.
The fold-safe frontier keeps `domain@15%` and `domain_shift@15%` as the maximum-release safe
policies; the simpler `domain@15%` remains the selected rule.
The candidate lock audit verifies that the executable contract also specifies `domain@15%`.
The locked candidate subgroup stress audit reports 0 flagged subgroup rows across label, fold,
shift-gate, and DINO-missing splits.
The boundary margin audit reports 0 errors in the first 5 next-reviewed cases after the selected
cutoff for old_data, third_batch, and strict_external.
The strict external mode-difference audit separates 8 frozen-display releases from deployable
behavior, where strict_external has 0 deployable releases due to severe-shift fallback.
The shift-gate evidence audit confirms strict_external is the only severe-shift domain
(`max_strict_to_internal_ratio=13.945213`), which documents the deployable fallback without using
strict_external labels for tuning.
The deployment readiness matrix adds the handoff guardrail: strict_external case-only results are
`frozen_display_only`, and strict_external deployable behavior is `deployable_fallback_to_v195`.
The final safety claims audit makes those handoff statements executable; all 4 current claims pass.
The deployment interface guard verifies that stable and adapter runtime decision exports contain no
audit-only labels/fold metadata and have zero label-free parity mismatches.
The runtime feature template audit verifies that the later deployment CSV template exactly matches
the 11 runtime-required contract columns and excludes audit/error selector fields.
The runtime template wrapper dry run sends that label-free template through the deployable wrapper
and confirms it emits a deployment decision with no audit-only or error-derived selector columns.
The synthetic deployment scenario audit verifies the deployable semantics on label-free rows:
normal-shift reviewed cases release by risk order, severe-shift reviewed cases stay in review, and
severe-shift v195-auto cases stay automatic.
The integration smoke audit exercises `integration/risk_control_gate_entrypoint.py` on adapter
features, producing 699 deployment decisions with 29 deployable releases and 0 strict_external
releases.
The deployable wrapper validates adapter features against `final_policy_contract.json` before
emitting deployment decisions; the current wrapper run has `contract_passed=true`,
`released_n=29`, and `strict_external_released_n=0`.
The wrapper decision schema audit verifies 4 deployment-facing decision CSVs share the required
decision schema and carry no audit-only or error-derived selector columns.
The new-data preflight runner combines contract validation, deployable wrapper execution, and
decision-schema inspection into one command for later label-free runtime feature CSVs.
The new-data preflight failure audit confirms that an input carrying `released_error` is rejected
at contract validation and emits 0 decision rows.
The existing external generalization preflight audit applies the same label-free deployable wrapper
path to the project's existing external/generalization domains, `third_batch` and `strict_external`.
It removes `label_idx`, `task_l6_label`, and `fold_id` before runtime decisions, then uses labels
only for post-decision audit. Current results are: `third_batch` 306 rows, 18 released, 0 released
errors, 0 released high-risk FN; `strict_external` 108 rows, 0 released, 0 released errors.
The existing external casebook audit then expands those decisions into 414 case-level rows. It
reports 18 third_batch released cases with 0 released errors and 0 released high-risk FN, plus 8
strict_external case-only release candidates that are blocked by the severe-shift deployable
fallback.
The strict external override candidate audit selects a threshold only from old_data and third_batch,
then freezes it before evaluating strict_external. It finds 86 old+third candidates with 0 errors
and 22 strict_external frozen candidates with 0 errors, but the leave-one-known-domain audit is not
stable (`old_data -> third_batch` gives 92 candidates with 6 errors), so the candidate is explicitly
marked `exploratory_frozen_audit_only_not_deployable`.
The known-domain intersection override audit then takes the componentwise conservative intersection
of the old_data-only and third_batch-only zero-error thresholds. It finds 80 known-domain candidates
with 0 errors and 21 strict_external frozen candidates with 0 errors, but remains
`conservative_intersection_frozen_audit_only_not_deployable` because strict_external is still gated
as severe unknown shift.
The integration handoff guide documents why the wrapper remains the preferred interface before
in-place modification of historical scripts.
The handoff manifest checks 82 core artifacts/status values and currently reports no failed checks.

## Additional Robustness Audit

The DINO missing-value sensitivity audit was added:

```text
scripts/dino_missing_policy_sensitivity.py
outputs/dino_missing_policy_sensitivity_summary.csv
reports/dino_missing_policy_sensitivity.md
```

Result:

```text
sort_last released missing DINO cases = 0 / 37 released
sort_first released missing DINO cases = 6 / 37 released
median fill released missing DINO cases = 6 / 37 released
released_error_n = 0 for all missing-value policies in the current audit
```

This supports keeping `dino_fn_risk_max` missing values sorted last as the more conservative
engineering default.

## Current Interpretation

This stage makes the candidate implementable without relying on hidden row order or direct reads
from the v191 historical artifact at policy time. It does not change the scientific conclusion:

```text
case_only_exploratory strict_external: 48 review, 0 released errors, frozen display only
shift_aware_deployable strict_external: 56 review, 0 released errors, conservative deployment mode
```
