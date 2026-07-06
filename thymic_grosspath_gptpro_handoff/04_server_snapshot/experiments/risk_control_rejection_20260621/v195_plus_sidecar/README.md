# v195+ Sidecar Flow

## Purpose

This directory contains an isolated v195+ sidecar flow for the Phase2 conservative candidate.

It does not modify original project code. It reads existing v185/v173 outputs plus the frozen v195+
candidate manifest, then writes new decision files under this sidecar directory.

## Policy

Candidate:

```text
phase2_min10_both_domain_union
```

Definition:

```text
Union of agreement-release candidates with:
- old_data action_error_n == 0
- third_batch action_error_n == 0
- old_data action_n > 0
- third_batch action_n > 0
- old_data action_n + third_batch action_n >= 10
```

This is not model retraining. It is a rejection/release policy applied on top of the confirmed v195
main workflow.

## Inputs

Default source files:

```text
outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv
outputs/grosspath_rc_v173_image_only_review_corrector_20260527/v173_corrector_case_outputs.csv
experiments/risk_control_rejection_20260621/configs/v195_plus_candidate_manifest.json
```

## Command

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/v195_plus_sidecar/v195_plus_sidecar_runner.py
```

Optional arguments:

```text
--manifest PATH
--candidate-key phase2_min10_both_domain_union
--out-dir PATH
```

## Outputs

Default outputs:

```text
experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_runtime_decisions.csv
experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_audit_decisions.csv
experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_summary.csv
experiments/risk_control_rejection_20260621/v195_plus_sidecar/outputs/v195_plus_sidecar_report.json
```

`v195_plus_runtime_decisions.csv` is the label-free runtime-style output.

`v195_plus_audit_decisions.csv` includes labels and error fields for experiment verification only.

## Expected Check

The default run should reproduce the Phase2 strict_external result:

```text
v195 strict_external auto = 52/108
v195+ strict_external auto = 57/108
v195+ strict_external auto_error = 0
```

The sidecar report fails if this expected check is not met.

## Stability and Interface Audit

After generating outputs, run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/v195_plus_sidecar/v195_plus_sidecar_audit.py
```

Current verified audit:

```text
passed = true
failed_n = 0
```

The audit checks:

- expected sidecar output files exist,
- runtime output has the exact label-free column contract,
- runtime output contains no audit/label/error columns,
- strict_external metrics reproduce v195 and v195+ Phase2 expectations,
- repeated in-memory builds are idempotent,
- disk outputs match the sidecar builder,
- sidecar boundary flags remain `sidecar_only = true` and `original_project_code_modified = false`.

## Boundary

This flow is intentionally sidecar-only:

```text
original_project_code_modified = false
sidecar_only = true
```

Do not copy this into the original project pipeline unless integration is explicitly approved.
