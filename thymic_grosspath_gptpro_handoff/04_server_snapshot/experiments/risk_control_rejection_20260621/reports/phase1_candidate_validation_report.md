# Phase 1 Candidate Validation Report

## Scope

All work in this stage remained under `experiments/risk_control_rejection_20260621/`.
Existing project files and historical outputs were treated as read-only inputs.

Strict external was used only as a frozen audit display. It was not used for selecting the
candidate policy, release fraction, labels, thresholds, or veto rules.

## Current Candidate

```text
Base policy: v195
Release target: only v195 reviewed/rejected cases
Release score: model_disagreement_rate
Release rule: release the lowest-risk 15% within each evaluated batch/domain
Tie handling: model_disagreement_rate ascending, then dino_fn_risk_max ascending, then case_id ascending
```

Existing v195 automatic decisions are unchanged.
`case_id` is used only as a final residual tie-break; it is not treated as a risk signal.
`dino_fn_risk_max` currently comes from the v191 DINO/FN sentinel artifact and must be promoted
to a stable feature interface before production implementation.

This promotion has now been prototyped in the isolated experiment:

```text
stable feature table: outputs/stable_policy_features.csv
feature version: risk_control_v1
final runner outputs: outputs/final_policy_runner_cases.csv / outputs/final_policy_runner_summary.csv
production adapter feature table: outputs/adapter_policy_features.csv
adapter runner outputs: outputs/adapter_final_policy_runner_cases.csv / outputs/adapter_final_policy_runner_summary.csv
```

The stable table contains `dino_fn_risk_missing` because 266/699 cases have no non-null DINO/FN
risk value in the historical v191 artifact. The current DINO release rule does not release any
case with missing `dino_fn_risk_max`; missing values are sorted as lowest priority for release.

The production adapter prototype reconstructs the same stable feature interface from current
project outputs. It produced 699 rows, 252 v195 review/reject cases, and 266 missing DINO/FN risk
values. Its final-runner summary matched the stable-feature final-runner summary exactly on all
8 mode/scope rows and all compared count columns.

The executable contract validator now checks both feature tables against
`configs/final_policy_contract.json`. Both current inputs pass. The validator also confirms that no
forbidden selector columns such as action/error flags are present in either stable runtime feature
table.

A label-free runtime decision export was also added. It drops `label_idx`, `task_l6_label`, and
`fold_id` before applying the final policy. The exported decisions match the audit-table decisions
with 0 mismatches, and the stable-feature and adapter-feature decision exports also match with
0 mismatches across 1398 mode/case rows.

An input-order invariance audit shuffled both stable and adapter feature tables with seeds
0, 1, 2, 7, 13, 29, and 101. Runtime decisions stayed identical in all runs
(`mismatch_rows=0`), confirming that the large tie groups are resolved deterministically.

A one-command reproducibility runner was added at
`scripts/phase1_reproducibility_runner.py`. It currently runs 37 checks end to end and passes all
of them, including 72 unit tests.

The isolated final runner was checked against the selected DINO audit outputs:

```text
final_policy_consistency mismatch_rows = 0
compared modes = case_only_exploratory, shift_aware_deployable
```

## Candidate Performance

| Scope | v195 Auto | Candidate Auto | Review Change | Released | Released Errors | Auto Errors | High-risk FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old_data | 210 | 221 | -11 | 11 | 0 | 0 | 0 |
| third_batch | 185 | 203 | -18 | 18 | 0 | 1 | 1 |
| strict_external | 52 | 60 | -8 | 8 | 0 | 0 | 0 |
| all_domains | 447 | 484 | -37 | 37 | 0 | 1 | 1 |

The one all-domain automatic error is the original v195 third_batch TC error, not a newly
released case.

## Negative Experiments

Several alternatives were tested and rejected:

- Fixed `model_disagreement_rate` threshold: unsafe on third_batch.
- Conjunctive two-feature rules: internally safe, but released only 2 strict external cases.
- Quota + safe-distance veto with backfill: selected an overly conservative veto; released only 2 strict external cases.
- Quota + safe-distance veto without backfill: safe, but released fewer cases than the current 15% quota.
- AB-only release: safe, but loses 7 safe old_data non-AB releases without improving strict external.

Current interpretation: the 15% per-batch quota is the best Phase 1 candidate among tested rules.

## Deployment Mode Split

There are now two explicitly separated interpretations:

| Scope | Mode | Auto | Review | Review Change | Released | Released Errors | Auto Errors | High-risk FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old_data | case_only_exploratory | 221 | 64 | -11 | 11 | 0 | 0 | 0 |
| old_data | shift_aware_deployable | 221 | 64 | -11 | 11 | 0 | 0 | 0 |
| third_batch | case_only_exploratory | 203 | 103 | -18 | 18 | 0 | 1 | 1 |
| third_batch | shift_aware_deployable | 203 | 103 | -18 | 18 | 0 | 1 | 1 |
| strict_external | case_only_exploratory | 60 | 48 | -8 | 8 | 0 | 0 | 0 |
| strict_external | shift_aware_deployable | 52 | 56 | 0 | 0 | 0 | 0 | 0 |

`case_only_exploratory` applies the case-level 15% release rule to strict external as a frozen
display. `shift_aware_deployable` enforces the severe-shift hard fallback. Because existing
unsupervised shift audit marks strict external as `severe_unknown_shift`, the deployable
hard-gated interpretation does not lower strict external review count.

This is an important boundary: the current evidence supports safe internal release and an
encouraging frozen strict-external display, but not a deployable claim that strict external
review can be reduced under the conservative hard-gate policy.

The hard-gate split is not a near-threshold artifact in the current unsupervised metrics:

| Metric | Internal Max | Strict External | Ratio |
| --- | ---: | ---: | ---: |
| batch_shift_index | 0.969486 | 2.324184 | 2.40x |
| domain_auc_cv | 0.531717 | 0.994501 | 1.87x |
| mean_outside_ref_05_95_rate | 0.098856 | 0.439815 | 4.45x |
| quality_proxy_mean | 0.021950 | 0.306096 | 13.95x |

These are unsupervised shift metrics only; no strict_external labels were used to set a new
threshold here.

The dedicated shift-gate evidence audit records:

```text
severe_domains = strict_external
normal_domains = old_data, third_batch
max_strict_to_internal_ratio = 13.945213
```

## Subgroup Findings

Released cases by label under the selected DINO tie-break policy:

| Label | n | Released | Released Errors | Auto Errors |
| --- | ---: | ---: | ---: | ---: |
| AB | 301 | 28 | 0 | 0 |
| B1 | 73 | 5 | 0 | 0 |
| A | 52 | 4 | 0 | 0 |
| B2 | 117 | 0 | 0 | 0 |
| TC | 115 | 0 | 0 | 1 |

Strict external exploratory release is:

```text
strict_external / AB: released=6, released errors=0
strict_external / B1: released=1, released errors=0
strict_external / A: released=1, released errors=0
```

The selected DINO rule releases no B2 or TC cases. This is a useful safety property because the
known boundary failures were high-risk B2 false negatives, and the original v195 all-domain
automatic error remains a TC case that was already automatic under v195.

## DINO Missing-Value Sensitivity

Because 266/699 cases have missing `dino_fn_risk_max`, the missing-value policy was audited
separately under the same 15% case-level release rule.

| Missing Policy | Scope | Released | Released Missing DINO | Released Errors | Auto Errors | High-risk FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sort_last | old_data | 11 | 0 | 0 | 0 | 0 |
| sort_last | third_batch | 18 | 0 | 0 | 1 | 1 |
| sort_last | strict_external | 8 | 0 | 0 | 0 | 0 |
| sort_first | old_data | 11 | 3 | 0 | 0 | 0 |
| sort_first | third_batch | 18 | 3 | 0 | 1 | 1 |
| sort_first | strict_external | 8 | 0 | 0 | 0 | 0 |
| median | old_data | 11 | 3 | 0 | 0 | 0 |
| median | third_batch | 18 | 3 | 0 | 1 | 1 |
| median | strict_external | 8 | 0 | 0 | 0 | 0 |

This does not change the selected rule. `sort_last` remains preferable because it avoids releasing
cases with unknown DINO/FN risk while preserving the same aggregate safety profile in the current
audit. Strict external remains a frozen display only.

## Fraction and Granularity Stress Audit

The release fraction was scanned at 5%, 10%, 12.5%, 15%, 17.5%, and 20% across three quota
granularities: `domain`, `domain + task_l6_label`, and `domain + hard_shift_gate`.

The aggregate internal-known scan alone would select `domain` at 20%:

```text
internal_known released_n = 39
released_error_n = 0
released_high_risk_fn_n = 0
```

That aggregate result is not accepted as the final rule because fold-level auditing rejects the
looser boundary. In third_batch folds, `domain` and `domain_shift` become unsafe at 17.5% and 20%:

| Granularity | Fraction | Domain | Released Sum | Released Error Sum | Released Error Max | Released High-risk FN Sum | Fold-safe |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| domain | 0.150 | old_data | 9 | 0 | 0 | 0 | true |
| domain | 0.150 | third_batch | 15 | 0 | 0 | 0 | true |
| domain | 0.175 | third_batch | 20 | 1 | 1 | 1 | false |
| domain | 0.200 | third_batch | 22 | 1 | 1 | 1 | false |
| domain_shift | 0.175 | third_batch | 20 | 1 | 1 | 1 | false |
| domain_shift | 0.200 | third_batch | 22 | 1 | 1 | 1 | false |

The fold-safe selector therefore keeps the original conservative candidate:

```text
granularity = domain
release_fraction = 0.15
fold released_n_sum = 24
released_error_n_sum = 0
released_high_risk_fn_n_sum = 0
```

This confirms that the 15% rule is the highest-release scanned candidate that remains safe across
old_data and third_batch folds. Strict external was not used for this selection.

A fold boundary failure casebook was generated for the unsafe rows. All unsafe rows point to the
same newly released high-risk false negative:

```text
case_id = third_B2_2517222
domain = third_batch
fold_id = 1
task_l6_label = B2
label_idx = 1
final_pred = 0
model_disagreement_rate = 0.0
dino_fn_risk_max = 0.679124
unsafe policies = domain/domain_shift at 17.5% and 20%
```

The fold-safe frontier confirms that the maximum-release safe policies are:

| Granularity | Fraction | Fold Released Sum | Released Errors | Released High-risk FN |
| --- | ---: | ---: | ---: | ---: |
| domain | 0.150 | 24 | 0 | 0 |
| domain_shift | 0.150 | 24 | 0 | 0 |

`domain_shift` is equivalent here because old_data and third_batch are not severe-shift batches
under the current hard gate, so `domain` at 15% remains the simpler selected rule.

The candidate lock audit now explicitly checks the contract and frontier:

```text
selected_granularity = domain
selected_release_fraction = 0.15
contract_granularity = domain
contract_release_fraction = 0.15
frontier_max_released_n_sum = 24
selected_frontier_released_n_sum = 24
passed = true
```

`configs/final_policy_contract.json` now explicitly includes `quota_granularity = domain`.

## Locked Candidate Subgroup Stress

The locked `domain@15%` candidate was re-audited by:

- label
- domain + label
- domain + fold
- domain + hard_shift_gate
- domain + DINO missingness

Result:

```text
summary_rows = 48
flagged_subgroup_rows = 0
released_error_n = 0 in all subgroup families
released_high_risk_fn_n = 0 in all subgroup families
```

This means the current selected release set does not create a hidden newly released error in any
audited label, fold, shift-gate, or DINO-missing subgroup. Strict external remains a frozen display.

## Locked Candidate Boundary Margin

The next 5 reviewed cases immediately after the `domain@15%` cutoff were audited per domain:

| Domain | Released | Next Reviewed | Next Errors | Next High-risk FN |
| --- | ---: | ---: | ---: | ---: |
| old_data | 11 | 5 | 0 | 0 |
| third_batch | 18 | 5 | 0 | 0 |
| strict_external | 8 | 5 | 0 | 0 |

This indicates that the selected cutoff is not immediately adjacent to an observed error in the
first 5 next-reviewed cases. It does not justify loosening the rule, because the fold stress audit
already shows that 17.5% and 20% release `third_B2_2517222` as a high-risk FN.

## Strict External Mode Difference

The strict external case-only and deployable interpretations were compared directly:

```text
case_only_released_n = 8
deployable_released_n = 0
frozen_only_released_n = 8
frozen_only_released_error_n = 0
frozen_only_released_high_risk_fn_n = 0
deployable_reason = severe_unknown_shift_fallback_to_v195
```

The 8 strict_external releases are therefore retained only as frozen-display evidence. They are not
a deployable coverage claim under the conservative hard-shift gate.

## Deployment Readiness Matrix

The deployment readiness matrix converts the mode split into explicit status rows:

```text
rows = 8
deployable_claim_rows = 4
frozen_display_rows = 1
deployable_fallback_rows = 1
statuses = deployable_fallback_to_v195, deployable_release, exploratory_internal_reference, frozen_display_only
```

This matrix is the handoff guardrail: strict_external `case_only_exploratory` remains
`frozen_display_only`, while strict_external `shift_aware_deployable` is
`deployable_fallback_to_v195` with 0 released cases.

## Final Safety Claims

The final safety claims audit consolidates the executable evidence:

```text
all_claims_passed = true
claim_rows = 4
failed_claims = []
```

The four checked claims are: executable contract locked to `domain@15%`, selected fold frontier has
zero released errors and zero released high-risk FN, deployable mode has zero newly released errors,
and strict_external has no deployable release under the severe-shift fallback.

## Deployment Interface Guard

The deployment interface guard checks the stable and adapter runtime decision exports:

```text
all_passed = true
rows = 2
failed_exports = []
```

It verifies that runtime exports contain no `label_idx`, `task_l6_label`, or `fold_id` columns, and
that both label-free parity reports have 0 mismatches.

## Handoff Manifest

The Phase 1 handoff manifest checks the core files needed to resume or modify the project safely:

```text
all_passed = true
checks = 82
failed_checks = []
```

This is the current handoff boundary for future code changes: use the final policy contract,
candidate validation report, engineering interface report, runtime exports, safety claims, interface
guard, and reproducibility runner together rather than relying on memory of this session.

## Integration Smoke

An isolated integration entrypoint was added under `integration/` without modifying project code.
It applies the deployable risk-control gate to adapter policy features and emits deployment
decisions only:

```text
decision_rows = 699
released_n = 29
review_or_reject_n = 223
strict_external_released_n = 0
forbidden_columns_present = []
passed = true
```

This is the current code-level handoff artifact for later production integration.

## Deployable Wrapper

The external wrapper validates the feature table against `final_policy_contract.json` before
emitting deployment decisions:

```text
contract_passed = true
decision_rows = 699
released_n = 29
review_or_reject_n = 223
strict_external_released_n = 0
forbidden_columns_present = []
```

This wrapper is the preferred next-step interface for new data runs. It keeps the original project
code untouched while enforcing the stable contract and deployable severe-shift fallback.
The integration handoff guide records why the wrapper remains the preferred interface before any
in-place modification of historical project scripts.

## Runtime Feature Template Audit

The runtime feature template is now executable-checked against `final_policy_contract.json`:

```text
template rows = 1
template columns = 11
passed = true
order_matches_contract = true
missing_required_columns = []
extra_columns = []
audit_only_columns_present = []
forbidden_columns_present = []
```

This gives later deployment runs a concrete label-free input shape. The template contains only the
11 runtime columns required by the contract and excludes `label_idx`, `task_l6_label`, `fold_id`,
and all error-derived selector columns.

The same template was also sent through the deployable wrapper as a dry run:

```text
passed = true
contract_passed = true
template_rows = 1
decision_rows = 1
audit_only_columns_present = []
forbidden_columns_present = []
```

This verifies that a new-data-shaped, label-free input can produce deployment decisions through the
same wrapper path used for adapter features.

A synthetic label-free deployment scenario was also added to verify the wrapper semantics directly:

```text
passed = true
feature_rows = 4
decision_rows = 4
released_n = 1
strict_external_released_n = 0
normal_low_released = true
normal_high_not_released = true
severe_review_not_released = true
severe_review_stays_review = true
severe_v195_auto_stays_auto = true
```

This confirms that normal-shift reviewed cases can be released by risk order, while severe-shift
reviewed cases fall back to v195 review and severe-shift v195-auto cases remain automatic.

Finally, a wrapper decision schema audit checks all deployment-facing decision CSVs:

```text
all_passed = true
rows = 4
failed_exports = []
```

The checked exports are `integration_deployable_decisions`, `wrapper_deployable_decisions`,
`runtime_template_wrapper_dry_run`, and `synthetic_deployment_scenario`. Each must contain the
same deployment decision columns and must not include audit labels, fold metadata, or error-derived
selector fields.

The new-data preflight runner is now the single deployment-adjacent command for later label-free
runtime feature CSVs:

```text
passed = true
contract_passed = true
decision_schema_passed = true
feature_rows = 1
decision_rows = 1
schema_failed_exports = []
```

It performs contract validation, deployable wrapper execution, and output schema inspection in one
step. The default run uses the runtime feature template as a minimal label-free input; later
deployment feature CSVs should pass through the same command.

A negative new-data preflight audit verifies that forbidden error-derived selector columns are
blocked before any decision rows are emitted:

```text
passed = true
blocked_feature_rows = 1
blocked_decision_rows = 0
blocked_forbidden_columns = released_error
blocked contract passed = false
blocked decision schema passed = false
```

## Existing External Generalization Preflight

To avoid confusing the deployment-template preflight with the actual external-generalization audit,
a dedicated audit now runs the deployable wrapper on the project's existing external/generalization
domains from `adapter_policy_features.csv`:

```text
external domains = third_batch, strict_external
runtime audit columns removed = label_idx, task_l6_label, fold_id
decision rows = 414
released_n = 18
forbidden_columns_present = []
passed = true
```

Domain-level results:

| Domain | Rows | Released | Review/Reject | Strict Released | Released Errors | High-risk FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| third_batch | 306 | 18 | 103 | 0 | 0 | 0 |
| strict_external | 108 | 0 | 56 | 0 | 0 | 0 |

The runtime input is label-free; labels are used only after decisions are emitted to audit released
errors and released high-risk false negatives. This is the current external-generalization evidence
path for the existing project datasets, not a placeholder dataset.

## Existing External Casebook Audit

A casebook audit now expands the existing external preflight decisions into all 414 external-domain
cases. It classifies each case as retained v195 auto, released from v195 review, reviewed but not
released, or strict_external shift-blocked release candidate.

Current casebook result:

```text
casebook_rows = 414
released_n = 18
released_error_n = 0
released_high_risk_fn_n = 0
strict_external_released_n = 0
strict_external_shift_blocked_candidate_n = 8
passed = true
```

Status-level summary:

| Domain | Status | n | Released | Would-be Errors | Would-be High-risk FN | Released Errors | DINO Missing | Rank Range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| third_batch | released_from_review | 18 | 18 | 0 | 0 | 0 | 0 | 1-18 |
| third_batch | reviewed_not_released | 103 | 0 | 8 | 6 | 0 | 21 | 19-121 |
| third_batch | v195_auto_retained | 185 | 0 | 1 | 1 | 0 | 56 | - |
| strict_external | shift_blocked_release_candidate | 8 | 0 | 0 | 0 | 0 | 0 | 1-8 |
| strict_external | reviewed_not_released | 48 | 0 | 0 | 0 | 0 | 34 | 9-56 |
| strict_external | v195_auto_retained | 52 | 0 | 0 | 0 | 0 | 13 | - |

Interpretation: the deployable wrapper safely releases 18 `third_batch` reviewed cases with zero
released errors. In strict_external, the top 8 case-only release candidates are visible for frozen
audit and have zero observed errors, but deployable mode still blocks them because strict_external is
the severe-shift domain. This preserves the original rule: strict_external labels are descriptive
audit evidence, not a tuning signal for relaxing the hard gate.

## Strict External Override Candidate Audit

An exploratory override audit was added to test whether a more permissive strict_external release
rule can be designed without tuning on strict_external labels. The rule is selected only on
`old_data + third_batch` reviewed cases using runtime-safe thresholds:

```text
model_disagreement_rate <= 0.583333
dino_fn_risk_max <= 0.369693
dino_fn_risk_missing = false
```

Selection-domain result:

```text
old_data candidates = 43, errors = 0, high-risk FN = 0
third_batch candidates = 43, errors = 0, high-risk FN = 0
combined selection candidates = 86, errors = 0, high-risk FN = 0
```

Frozen strict_external audit after threshold selection:

```text
strict_external candidates = 22
strict_external candidate errors = 0
strict_external candidate high-risk FN = 0
```

However, the leave-one-known-domain check blocks promotion:

| Trained On | Audited Domain | Candidates | Errors | High-risk FN | Safe |
| --- | --- | ---: | ---: | ---: | --- |
| third_batch | old_data | 43 | 0 | 0 | true |
| old_data | third_batch | 92 | 6 | 6 | false |

Interpretation: the 22 strict_external candidates are useful frozen evidence, but the candidate is
not stable enough to become deployable. It remains `exploratory_frozen_audit_only_not_deployable`
and `promote_to_deployable = false`.

## Known-domain Intersection Override Audit

A stricter follow-up override audit was added to remove the unsafe leave-one-known-domain expansion.
Instead of selecting one pooled threshold from old_data + third_batch, it first selects a zero-error
threshold in each known domain, then takes the componentwise conservative intersection.

Single-domain thresholds:

| Selection Domain | model_disagreement_rate | dino_fn_risk_max |
| --- | ---: | ---: |
| old_data | 0.500000 | 0.877680 |
| third_batch | 0.583333 | 0.369693 |

Intersection threshold:

```text
model_disagreement_rate <= 0.500000
dino_fn_risk_max <= 0.369693
dino_fn_risk_missing = false
```

Known-domain result:

```text
old_data candidates = 43, errors = 0, high-risk FN = 0
third_batch candidates = 37, errors = 0, high-risk FN = 0
combined known candidates = 80, errors = 0, high-risk FN = 0
```

Frozen strict_external audit after threshold intersection:

```text
strict_external candidates = 21
strict_external candidate errors = 0
strict_external candidate high-risk FN = 0
```

Interpretation: this is the safest strict_external release hypothesis found so far because it is
stable across both known external-generalization sources used for threshold selection. It still does
not become deployable in Phase 1 because strict_external remains a severe unknown shift under the
hard gate, and the audit is too small to justify overriding that gate. It is recorded as
`conservative_intersection_frozen_audit_only_not_deployable` with
`promote_to_deployable = false`.

## Boundary Findings

Increasing from 15% to 20% is unsafe in fold-level third_batch auditing:

```text
third_batch fold 1, B2, case third_B2_2517222:
  label_idx=1, final_pred=0, model_disagreement_rate=0.0
```

This explains why `model_disagreement_rate` alone cannot safely loosen beyond 15% under current
evidence.

An additional tie audit found that the 15% boundary often falls inside a large
`model_disagreement_rate = 0.0` tie group. A naive implementation that relies on input row order
is therefore not deployable. A deterministic `case_id` tie-break reproduced the numeric candidate,
but it is arbitrary. A follow-up DINO tie-break search found a non-arbitrary secondary signal:
`dino_fn_risk_max` ascending.

| Scope | Released | Released Errors | Review | Auto Errors |
| --- | ---: | ---: | ---: | ---: |
| old_data | 11 | 0 | 64 | 0 |
| third_batch | 18 | 0 | 103 | 1 |
| strict_external | 8 | 0 | 48 | 0 |
| all_domains | 37 | 0 | 215 | 1 |

Fold-level re-audit under the DINO tie rule released 9 old_data and 15 third_batch cases with
0 newly released errors in both domains. The DINO rule and the earlier `case_id` rule do not select
exactly the same cases, but both avoid the known third_batch B2 boundary-error cases and produce the
same aggregate/fold safety profile.

## Statistical Audit

Noninferiority flag used here:

```text
no extra high-risk FN
and candidate auto-error Wilson95 upper <= 3%
```

| Scope | Candidate Auto n | Auto Errors | Wilson95 Upper | Flag |
| --- | ---: | ---: | ---: | --- |
| all_domains | 484 | 1 | 1.16% | pass |
| old_data | 221 | 0 | 1.71% | pass |
| third_batch | 203 | 1 | 2.74% | pass |
| strict_external | 60 | 0 | 6.02% | frozen display only |

Strict external does not pass the 3% statistical flag because 60 automatic cases are too few,
not because an error occurred.

## Future Validation Size

For zero automatic errors:

| Wilson95 Upper Target | Required Auto n | Required Total n at 50% Auto Rate |
| --- | ---: | ---: |
| 3% | 125 | 250 |
| 2% | 189 | 378 |
| 1% | 381 | 762 |

## Generated Artifacts

- `outputs/candidate_policy_summary.csv`
- `outputs/candidate_subgroup_by_label.csv`
- `outputs/candidate_subgroup_by_domain_label.csv`
- `outputs/dino_candidate_subgroup_by_label.csv`
- `outputs/dino_candidate_subgroup_by_domain_label.csv`
- `outputs/dino_candidate_casebook_cases.csv`
- `outputs/dino_candidate_casebook_summary.csv`
- `outputs/stable_policy_features.csv`
- `outputs/stable_policy_feature_manifest.json`
- `outputs/adapter_policy_features.csv`
- `outputs/production_adapter_report.json`
- `outputs/final_policy_runner_cases.csv`
- `outputs/final_policy_runner_summary.csv`
- `outputs/adapter_final_policy_runner_cases.csv`
- `outputs/adapter_final_policy_runner_summary.csv`
- `outputs/adapter_final_policy_runner_report.json`
- `outputs/final_policy_consistency_diff.csv`
- `outputs/policy_contract_validation_report.json`
- `outputs/runtime_decision_export.csv`
- `outputs/runtime_decision_export_parity_diff.csv`
- `outputs/runtime_decision_export_report.json`
- `outputs/adapter_runtime_decision_export.csv`
- `outputs/adapter_runtime_decision_export_parity_diff.csv`
- `outputs/adapter_runtime_decision_export_report.json`
- `outputs/input_order_invariance_diff.csv`
- `outputs/input_order_invariance_report.json`
- `outputs/adapter_input_order_invariance_diff.csv`
- `outputs/adapter_input_order_invariance_report.json`
- `outputs/dino_missing_policy_sensitivity_cases.csv`
- `outputs/dino_missing_policy_sensitivity_summary.csv`
- `outputs/dino_missing_policy_sensitivity_report.json`
- `outputs/fraction_granularity_stress_cases.csv`
- `outputs/fraction_granularity_stress_summary.csv`
- `outputs/fraction_granularity_stress_selected.csv`
- `outputs/fraction_granularity_stress_report.json`
- `outputs/fold_fraction_granularity_stress_summary.csv`
- `outputs/fold_fraction_granularity_stress_selected.csv`
- `outputs/fold_fraction_granularity_stress_report.json`
- `outputs/fold_boundary_failure_cases.csv`
- `outputs/fold_boundary_failure_unsafe_policies.csv`
- `outputs/fold_boundary_failure_casebook_report.json`
- `outputs/fold_safe_policy_summary.csv`
- `outputs/fold_safe_frontier.csv`
- `outputs/fold_safe_frontier_report.json`
- `outputs/candidate_lock_audit_report.json`
- `outputs/locked_candidate_subgroup_cases.csv`
- `outputs/locked_candidate_subgroup_summary.csv`
- `outputs/locked_candidate_subgroup_report.json`
- `outputs/locked_candidate_boundary_margin_summary.csv`
- `outputs/locked_candidate_boundary_margin_casebook.csv`
- `outputs/locked_candidate_boundary_margin_report.json`
- `outputs/strict_external_mode_diff_cases.csv`
- `outputs/strict_external_mode_diff_summary.csv`
- `outputs/strict_external_mode_diff_report.json`
- `outputs/shift_gate_evidence_summary.csv`
- `outputs/shift_gate_evidence_report.json`
- `outputs/deployment_readiness_matrix.csv`
- `outputs/deployment_readiness_matrix_report.json`
- `outputs/final_safety_claims.csv`
- `outputs/final_safety_claims_report.json`
- `outputs/deployment_interface_guard.csv`
- `outputs/deployment_interface_guard_report.json`
- `outputs/runtime_feature_template_audit_report.json`
- `outputs/runtime_template_wrapper_dry_run_decisions.csv`
- `outputs/runtime_template_wrapper_dry_run_report.json`
- `outputs/synthetic_deployment_scenario_decisions.csv`
- `outputs/synthetic_deployment_scenario_report.json`
- `outputs/wrapper_decision_schema_audit.csv`
- `outputs/wrapper_decision_schema_audit_report.json`
- `outputs/new_data_preflight_decisions.csv`
- `outputs/new_data_preflight_report.json`
- `outputs/new_data_preflight_failure_audit_report.json`
- `outputs/existing_external_preflight_decisions.csv`
- `outputs/existing_external_preflight_summary.csv`
- `outputs/existing_external_generalization_preflight_report.json`
- `outputs/existing_external_casebook.csv`
- `outputs/existing_external_casebook_summary.csv`
- `outputs/existing_external_casebook_report.json`
- `outputs/strict_external_override_candidate_cases.csv`
- `outputs/strict_external_override_candidate_summary.csv`
- `outputs/strict_external_override_candidate_report.json`
- `outputs/known_domain_intersection_override_cases.csv`
- `outputs/known_domain_intersection_override_summary.csv`
- `outputs/known_domain_intersection_override_report.json`
- `outputs/integration_deployable_decisions.csv`
- `outputs/integration_smoke_audit_report.json`
- `outputs/wrapper_deployable_decisions.csv`
- `outputs/wrapper_deployable_report.json`
- `outputs/phase1_handoff_manifest.csv`
- `outputs/phase1_handoff_manifest_report.json`
- `outputs/phase1_reproducibility_runner_report.json`
- `outputs/candidate_casebook_cases.csv`
- `outputs/dino_candidate_statistical_audit.csv`
- `outputs/case_id_tiebreak_summary.csv`
- `outputs/case_id_tiebreak_fold_summary.csv`
- `outputs/dino_tiebreak_internal_summary.csv`
- `outputs/dino_tiebreak_fold_summary.csv`
- `outputs/dino_shift_aware_candidate_summary.csv`
- `outputs/shift_aware_candidate_summary.csv`
- `outputs/policy_mode_comparison.csv`
- `outputs/shift_gate_metric_margins.csv`
- `outputs/validation_sample_size.csv`
- `reports/candidate_policy_audit.md`
- `reports/candidate_subgroup_audit.md`
- `reports/dino_candidate_subgroup_audit.md`
- `reports/dino_candidate_casebook.md`
- `reports/stable_policy_feature_interface.md`
- `reports/production_adapter_prototype.md`
- `reports/final_policy_runner.md`
- `reports/final_policy_consistency_audit.md`
- `reports/policy_contract_validation.md`
- `reports/runtime_decision_export.md`
- `reports/input_order_invariance_audit.md`
- `reports/dino_missing_policy_sensitivity.md`
- `reports/fraction_granularity_stress_audit.md`
- `reports/fold_fraction_granularity_stress_audit.md`
- `reports/fold_boundary_failure_casebook.md`
- `reports/fold_safe_frontier.md`
- `reports/candidate_lock_audit.md`
- `reports/locked_candidate_subgroup_stress_audit.md`
- `reports/locked_candidate_boundary_margin_audit.md`
- `reports/strict_external_mode_diff_audit.md`
- `reports/shift_gate_evidence_audit.md`
- `reports/deployment_readiness_matrix.md`
- `reports/final_safety_claims_audit.md`
- `reports/deployment_interface_guard.md`
- `reports/runtime_feature_template_audit.md`
- `reports/runtime_template_wrapper_dry_run.md`
- `reports/synthetic_deployment_scenario_audit.md`
- `reports/wrapper_decision_schema_audit.md`
- `reports/new_data_preflight_runner.md`
- `reports/new_data_preflight_failure_audit.md`
- `reports/existing_external_generalization_preflight_audit.md`
- `reports/existing_external_casebook_audit.md`
- `reports/strict_external_override_candidate_audit.md`
- `reports/known_domain_intersection_override_audit.md`
- `reports/integration_smoke_audit.md`
- `reports/integration_handoff_guide.md`
- `reports/phase1_handoff_manifest.md`
- `reports/phase1_reproducibility_runner.md`
- `reports/phase1_engineering_interface_report.md`
- `reports/candidate_casebook.md`
- `reports/dino_candidate_statistical_audit.md`
- `reports/case_id_tiebreak_audit.md`
- `reports/dino_tiebreak_audit.md`
- `reports/dino_shift_aware_candidate_audit.md`
- `reports/shift_aware_candidate_audit.md`
- `reports/policy_mode_comparison.md`
- `reports/shift_gate_margin_audit.md`
- `reports/validation_sample_size.md`
