# v195+ Sidecar Package Smoke Audit

This audit runs the package-local sidecar runner from the staged handoff package and checks that it emits label-free runtime output.

## Result

- Passed: `True`.
- Package: `experiments/risk_control_rejection_20260621/staging/v195_plus_sidecar_package`.
- Smoke outputs: `experiments/risk_control_rejection_20260621/outputs/v195_plus_sidecar_package_smoke_outputs`.

## Checks

| Check | Passed | Detail |
|---|---:|---|
| package_report_passed | True | experiments/risk_control_rejection_20260621/staging/v195_plus_sidecar_package/outputs/v195_plus_sidecar_package_report.json |
| package_local_runner_passed | True | returncode=0 |
| smoke_runtime_rows_699 | True | runtime_rows=699 |
| smoke_runtime_label_free | True | [] |
| smoke_sidecar_report_passed | True | passed=True |
| package_excludes_audit_decisions | True | v195_plus_audit_decisions.csv absent from package |