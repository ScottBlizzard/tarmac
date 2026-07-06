# v195+ Final Handoff Index

This is the single entry point for the v195+ external-generalization rejection work.
It links the plan, experiment reports, sidecar, staged package, pre-integration audit, and non-applied hook blueprint.

## Status

- passed: `True`
- artifact_n: `19`
- original_project_code_modified: `False`
- recommended_current_state: `keep_v195_plus_as_sidecar_package; do_not_modify_original_project_without_explicit_approval`

## Headline Result

| Policy | strict_external Auto | strict_external Review | Auto Error |
|---|---:|---:|---:|
| v195 | 52/108 | 56/108 | 0 |
| v195+ Phase2 sidecar | 57/108 | 51/108 | 0 |

## Pass Flags

| Check | Passed |
|---|---:|
| `reproducibility_passed` | True |
| `sidecar_passed` | True |
| `sidecar_audit_passed` | True |
| `package_passed` | True |
| `package_smoke_passed` | True |
| `preintegration_gap_passed` | True |
| `minimal_hook_blueprint_passed` | True |
| `minimal_hook_blueprint_safety_passed` | True |

## Key Artifacts

| stage | artifact_id | path | exists | runtime_candidate | forbidden_runtime | role |
| --- | --- | --- | --- | --- | --- | --- |
| plan | completion_plan | plans/v195_plus_completion_plan.md | True | False | False | full phase checklist and status |
| candidate_freeze | candidate_manifest | configs/v195_plus_candidate_manifest.json | True | True | False | frozen Phase1/Phase2 candidate definitions |
| candidate_freeze | candidate_protocol | reports/v195_plus_candidate_protocol.md | True | False | False | human-readable candidate protocol |
| experiment | final_experiment_report | reports/v195_plus_final_experiment_report.md | True | False | False | main scientific result and limitations |
| experiment | reproducibility_runner | scripts/v195_plus_reproducibility_runner.py | True | False | False | regenerate core v195+ evidence chain |
| experiment | reproducibility_report | outputs/v195_plus_reproducibility_report.json | True | False | False | runner pass/fail and headline metrics |
| sidecar | sidecar_runner | v195_plus_sidecar/v195_plus_sidecar_runner.py | True | True | False | isolated v195+ sidecar decision builder |
| sidecar | sidecar_audit | v195_plus_sidecar/v195_plus_sidecar_audit.py | True | True | False | sidecar interface and stability audit |
| sidecar | runtime_decisions | v195_plus_sidecar/outputs/v195_plus_runtime_decisions.csv | True | False | False | label-free reference decision output |
| sidecar | audit_decisions | v195_plus_sidecar/outputs/v195_plus_audit_decisions.csv | True | False | True | audit-only labeled decisions; forbidden from runtime |
| sidecar | sidecar_handoff | reports/v195_plus_sidecar_handoff.md | True | False | False | rolling handoff notes for sidecar/package/gap/blueprint |
| package | staged_sidecar_package | staging/v195_plus_sidecar_package | True | False | False | minimal staged handoff package |
| package | package_report | outputs/v195_plus_sidecar_package_report.json | True | False | False | package build pass/fail |
| package | package_smoke_report | outputs/v195_plus_sidecar_package_smoke_report.json | True | False | False | package-local runner smoke test |
| preintegration | gap_audit | reports/v195_plus_preintegration_gap_audit.md | True | False | False | read-only original-project integration gap audit |
| preintegration | gap_audit_report | outputs/v195_plus_preintegration_gap_audit_report.json | True | False | False | gap audit pass/fail |
| hook_blueprint | minimal_hook_blueprint | reports/v195_plus_minimal_hook_blueprint.md | True | False | False | future non-applied hook plan |
| hook_blueprint | patch_draft | patch_rehearsal/v195_plus/v195_plus_new_flow_patch_draft.patch.txt | True | False | False | non-applied v203 new-flow patch sketch |
| hook_blueprint | blueprint_safety_report | outputs/v195_plus_minimal_hook_blueprint_safety_report.json | True | False | False | safety check that draft was not applied and v195 is untouched |

## Runtime Boundary

- Runtime candidates are limited to the sidecar runner/audit and frozen manifest.
- `v195_plus_audit_decisions.csv` is audit-only and forbidden from runtime.
- The original v195 script should not be modified for this sidecar path.
- Future integration, if approved, should add a new v203-style flow script.

## Decision Options

- `stop_here_and_archive_sidecar_evidence`
- `request_explicit_approval_to_create_new_v203_flow_script`
- `collect_additional_external_validation_before_any_integration`