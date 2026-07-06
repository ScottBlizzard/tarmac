# v195+ Sidecar Handoff

## Final Index

The single entry point for the current v195+ work is:

```text
experiments/risk_control_rejection_20260621/reports/v195_plus_final_handoff_index.md
```

Machine-readable index:

```text
experiments/risk_control_rejection_20260621/outputs/v195_plus_final_handoff_index.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_final_handoff_index.json
experiments/risk_control_rejection_20260621/outputs/v195_plus_final_handoff_report.json
```

## What Was Added

An isolated v195+ Phase2 sidecar flow was added under:

```text
experiments/risk_control_rejection_20260621/v195_plus_sidecar/
```

It does not modify original project code.

## Command

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/v195_plus_sidecar/v195_plus_sidecar_runner.py
```

## Outputs

```text
v195_plus_sidecar/outputs/v195_plus_runtime_decisions.csv
v195_plus_sidecar/outputs/v195_plus_audit_decisions.csv
v195_plus_sidecar/outputs/v195_plus_summary.csv
v195_plus_sidecar/outputs/v195_plus_sidecar_report.json
```

`v195_plus_runtime_decisions.csv` is label-free and is the table to use for future runtime-style
handoff.

`v195_plus_audit_decisions.csv` contains labels and error columns for experiment verification only.

## Verification Result

The sidecar run passed.

| Policy | Domain | Auto | Review | Auto Error | High-Risk FN |
|---|---|---:|---:|---:|---:|
| v195 | strict_external | 52/108 | 56/108 | 0 | 0 |
| v195_plus | strict_external | 57/108 | 51/108 | 0 | 0 |

Runtime output audit:

```text
runtime_rows = 699
runtime_forbidden_present = []
audit_rows = 699
audit_has_labels = true
```

## Stability and Interface Audit

Command:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/v195_plus_sidecar/v195_plus_sidecar_audit.py
```

Result:

```text
passed = true
failed_n = 0
```

Audit report:

```text
experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_sidecar_audit_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_sidecar_audit.md
```

## Pre-integration Gap Audit

Command:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/scripts/v195_plus_preintegration_gap_audit.py
```

Result:

```text
passed = true
blocked_gap_n = 0
future_main_project_touch_n = 2
missing_optional_hook_n = 2
```

The two missing optional hooks are expected because original-project integration has not been
approved:

```text
sidecar_invocation_hook
runtime_decision_consumer_hook
```

Audit report:

```text
experiments/risk_control_rejection_20260621/outputs/v195_plus_preintegration_gap_audit.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_preintegration_gap_audit_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_preintegration_gap_audit.md
```

## Minimal Hook Blueprint

Command:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/scripts/v195_plus_minimal_hook_blueprint.py
python experiments/risk_control_rejection_20260621/scripts/v195_plus_minimal_hook_blueprint_safety_audit.py
```

Result:

```text
blueprint passed = true
safety audit passed = true
future_new_flow = scripts/run_grosspath_rc_v203_v195_plus_sidecar_policy_20260623.py
existing_v195_script_should_be_modified = false
```

This is a non-applied draft. It recommends adding a new versioned v203-style flow only if explicit
integration approval is given later. It does not modify v195.

Artifacts:

```text
experiments/risk_control_rejection_20260621/outputs/v195_plus_minimal_hook_blueprint.csv
experiments/risk_control_rejection_20260621/outputs/v195_plus_minimal_hook_blueprint_report.json
experiments/risk_control_rejection_20260621/outputs/v195_plus_minimal_hook_blueprint_safety_report.json
experiments/risk_control_rejection_20260621/reports/v195_plus_minimal_hook_blueprint.md
experiments/risk_control_rejection_20260621/reports/v195_plus_minimal_hook_blueprint_safety_audit.md
experiments/risk_control_rejection_20260621/patch_rehearsal/v195_plus/v195_plus_new_flow_patch_draft.patch.txt
```

## Boundary

This is still an experimental sidecar package. It should not be copied into the original project
pipeline unless integration is explicitly approved.
