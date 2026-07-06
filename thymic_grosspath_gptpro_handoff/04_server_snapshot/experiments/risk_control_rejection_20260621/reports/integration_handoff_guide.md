# Integration Handoff Guide

## Current Position

The risk-control rejection module is ready as an isolated external wrapper, not as an in-place
modification of the historical project scripts.

The current integration boundary is:

```text
stable/project outputs
  -> production_adapter_prototype.py
  -> adapter_policy_features.csv
  -> integration/risk_control_deployable_wrapper.py
  -> wrapper_deployable_decisions.csv
```

## Why Not Modify Original Project Code Yet

The original project contains many historical experiment scripts around v185, v195, v201, and v202.
Those scripts are useful sources, but they are not a clean production entrypoint. Directly inserting
the gate into one historical script would create maintenance ambiguity: future runs could bypass the
gate, use a different version lineage, or accidentally mix audit labels into runtime behavior.

The wrapper approach keeps the deployed behavior explicit:

- validate `risk_control_v1` feature contract first;
- use only runtime-safe selector features;
- apply only `shift_aware_deployable`;
- force severe unknown shift batches back to v195 behavior;
- emit deployment decisions without `label_idx`, `task_l6_label`, or `fold_id`.

## Current Command

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/integration/risk_control_deployable_wrapper.py \
  --features experiments/risk_control_rejection_20260621/outputs/adapter_policy_features.csv \
  --out experiments/risk_control_rejection_20260621/outputs/wrapper_deployable_decisions.csv \
  --report experiments/risk_control_rejection_20260621/outputs/wrapper_deployable_report.json
```

## Runtime Feature Template

For later deployment runs, the runtime feature table should match:

```text
templates/risk_control_v1_runtime_feature_template.csv
```

This template intentionally excludes `label_idx`, `task_l6_label`, and `fold_id`. Those fields may
exist in audit tables, but they are not required by the deployable wrapper.

The template is checked by:

```bash
python experiments/risk_control_rejection_20260621/scripts/runtime_feature_template_audit.py
```

Current audit result:

```text
passed = true
column_count = 11
order_matches_contract = true
extra_columns = []
audit_only_columns_present = []
forbidden_columns_present = []
```

The template has also been dry-run through the deployable wrapper:

```bash
python experiments/risk_control_rejection_20260621/scripts/runtime_template_wrapper_dry_run.py
```

Current dry-run result:

```text
passed = true
contract_passed = true
template_rows = 1
decision_rows = 1
audit_only_columns_present = []
forbidden_columns_present = []
```

The wrapper semantics are also checked on a synthetic label-free deployment scenario:

```bash
python experiments/risk_control_rejection_20260621/scripts/synthetic_deployment_scenario_audit.py
```

Current scenario result:

```text
passed = true
released_n = 1
strict_external_released_n = 0
normal_low_released = true
severe_review_not_released = true
severe_v195_auto_stays_auto = true
```

Deployment-facing decision CSVs are schema-checked by:

```bash
python experiments/risk_control_rejection_20260621/scripts/wrapper_decision_schema_audit.py
```

Current schema audit result:

```text
all_passed = true
rows = 4
failed_exports = []
```

For later deployment runs, use the preflight runner as the single command after materializing a
label-free runtime feature CSV:

```bash
python experiments/risk_control_rejection_20260621/scripts/new_data_preflight_runner.py \
  --features path/to/new_runtime_features.csv \
  --out path/to/new_deployable_decisions.csv \
  --report path/to/new_preflight_report.json
```

Current template-based preflight result:

```text
passed = true
contract_passed = true
decision_schema_passed = true
feature_rows = 1
decision_rows = 1
schema_failed_exports = []
```

The negative preflight audit confirms that error-derived selector columns are blocked:

```bash
python experiments/risk_control_rejection_20260621/scripts/new_data_preflight_failure_audit.py
```

Current negative audit result:

```text
passed = true
blocked_decision_rows = 0
blocked_forbidden_columns = released_error
blocked contract passed = false
```

## Existing External Generalization Audit

The current external-generalization experiment should use the project's existing external domains,
not an imagined future collection. The dedicated audit command is:

```bash
python experiments/risk_control_rejection_20260621/scripts/existing_external_generalization_preflight_audit.py
```

It reads `outputs/adapter_policy_features.csv`, filters the existing external/generalization
domains, removes audit-only columns before runtime decisions, and then uses labels only after
decisions are emitted for audit summaries.

Current result:

```text
passed = true
external_domains = third_batch, strict_external
decision_rows = 414
released_n = 18
forbidden_columns_present = []
third_batch: rows=306, released=18, released_error_n=0, released_high_risk_fn_n=0
strict_external: rows=108, released=0, strict_external_released_n=0
```

This is the deployment-wrapper check for the already available external-generalization datasets.
The generic `new_data_preflight_runner.py` remains useful as an interface check for any later
label-free runtime feature CSV, but it is not the evidence source for the current external
generalization result.

For case-level inspection of those same existing external domains, run:

```bash
python experiments/risk_control_rejection_20260621/scripts/existing_external_casebook_audit.py
```

Current casebook result:

```text
passed = true
casebook_rows = 414
released_n = 18
released_error_n = 0
released_high_risk_fn_n = 0
strict_external_released_n = 0
strict_external_shift_blocked_candidate_n = 8
```

This casebook is the preferred artifact for inspecting which external cases were released or
blocked. The strict_external shift-blocked candidates are frozen audit evidence only; they should
not be used to tune strict_external release behavior.

An exploratory override candidate can also be inspected:

```bash
python experiments/risk_control_rejection_20260621/scripts/strict_external_override_candidate_audit.py
```

Current result:

```text
passed = true
interpretation = exploratory_frozen_audit_only_not_deployable
selection_domains = old_data, third_batch
model_disagreement_rate_threshold = 0.583333
dino_fn_risk_max_threshold = 0.369693
selection_candidate_n = 86
selection_released_error_n = 0
strict_external_candidate_n = 22
strict_external_candidate_error_n = 0
leave_one_known_domain_all_safe = false
promote_to_deployable = false
```

This is not an integration candidate. Its purpose is to document that a looser frozen strict_external
display is possible, while also documenting the cross-domain stability failure that blocks deployment.

A stricter conservative override hypothesis can also be inspected:

```bash
python experiments/risk_control_rejection_20260621/scripts/known_domain_intersection_override_audit.py
```

Current result:

```text
passed = true
interpretation = conservative_intersection_frozen_audit_only_not_deployable
selection_domains = old_data, third_batch
intersection_model_disagreement_rate_threshold = 0.500000
intersection_dino_fn_risk_max_threshold = 0.369693
known_candidate_n = 80
known_candidate_error_n = 0
known_candidate_high_risk_fn_n = 0
strict_external_candidate_n = 21
strict_external_candidate_error_n = 0
strict_external_candidate_high_risk_fn_n = 0
promote_to_deployable = false
```

This is the strongest Phase 1 frozen strict_external hypothesis because it is stable separately on
old_data and third_batch before strict_external is viewed. It still is not an integration candidate:
the deployable wrapper must continue to apply the severe-shift fallback and release 0 strict_external
cases.

## Current Wrapper Result

```text
contract_passed = true
decision_rows = 699
released_n = 29
review_or_reject_n = 223
strict_external_released_n = 0
forbidden_columns_present = []
```

## Conditions For Later In-place Integration

Only consider modifying original project code after all of the following are true:

- there is a single agreed inference entrypoint for new data;
- that entrypoint can materialize `risk_control_v1` features without labels;
- the wrapper output is accepted as the expected deployment decision format;
- strict_external and any later unknown external batches remain shift-gated without label tuning;
- the one-command runner remains green after any copied integration path is added.

Until then, the wrapper is the safer production-adjacent interface.
