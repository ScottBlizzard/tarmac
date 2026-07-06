# Phase 3 Release Handoff Bundle Audit

## Result

- passed: True
- recommended_handoff: `review_staging_package_without_main_project_write`
- bundle_file_n: 34
- integration_candidate_file_n: 8
- read_only_evidence_file_n: 18
- generated_support_file_n: 8
- blocked_artifact_file_n: 0
- reproducibility_command: `python experiments/risk_control_rejection_20260621/scripts/phase2_reproducibility_runner.py`
- main_project_write_required: False
- strict_external_relaxation_allowed: False
- original_project_code_modified: False

## Bundle Index

| Path | Role | Class | Origin | SHA256 | Instruction |
| --- | --- | --- | --- | --- | --- |
| `integration/__pycache__/risk_control_deployable_wrapper.cpython-311.pyc` | generated_support | unmanifested | generated_cache | `76f7d90b42c2fc2e19ddbc7e4ae8f5cfae480f3ce592ce3352a6b8cd7b5b88e7` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `integration/__pycache__/risk_control_gate_entrypoint.cpython-311.pyc` | generated_support | unmanifested | generated_cache | `1b5ad3159893f3f46e7a26edc07456ffea9a0824a2b6b4467601fa2570e1112f` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `outputs/handoff_file_classification_manifest.csv` | generated_support | unmanifested | generated_manifest | `cd3ab4741eb0876b68e7b32001d79f1ce5231c154ed002e60c68d25c2a819b80` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `outputs/handoff_file_classification_manifest.json` | generated_support | unmanifested | generated_manifest | `fc425343a9485c0cb4467516cc3f547cbd371fb35ee05a5f16b2e226d2eee54c` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `scripts/__pycache__/audit_rejection_layer.cpython-311.pyc` | generated_support | unmanifested | generated_cache | `fa940f0b0a2f38ee4710be9d75765a1f1c1ad4bab6335e642cb232800ec71050` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `scripts/__pycache__/final_policy_runner.cpython-311.pyc` | generated_support | unmanifested | generated_cache | `3f12b31f334a8d0a8771ac4c3526b091c883ced94b483fb4278ecc1214fbdffc` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `scripts/__pycache__/policy_contract_validator.cpython-311.pyc` | generated_support | unmanifested | generated_cache | `2e51f1f21c373a5d05cf228407789c91a8ffa57acc5854d921fa900967f50e0f` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `scripts/__pycache__/release_rejected_audit.cpython-311.pyc` | generated_support | unmanifested | generated_cache | `d77088b8c29bc40145b884f0c02cd7c5a2ced3312f052db46bf050ad206d154c` | generated_by_staging_or_python_runtime_keep_for_audit_context |
| `configs/final_policy_contract.json` | integration_candidate | runtime_contract | manifested | `37f4f2d2449145a87006c5252fd91964cacfee596157259f3df88659ff2146ac` | review_before_optional_sidecar_or_main_project_connection |
| `integration/risk_control_deployable_wrapper.py` | integration_candidate | deployable_code | manifested | `3e9ff8a93d1a3125e6ec87a93786b92a5470faa61429f8e0aa6bdef5b2d6e676` | review_before_optional_sidecar_or_main_project_connection |
| `integration/risk_control_gate_entrypoint.py` | integration_candidate | deployable_code | manifested | `a0ac7b5693ad9076a102d3e735d6cb3b17fae584aa8aa9785827777ae52f2789` | review_before_optional_sidecar_or_main_project_connection |
| `scripts/audit_rejection_layer.py` | integration_candidate | package_support_code | manifested | `af7a9a2741f0d7a9c3780b81d6dcc1ad244f6a98c16a7c21bc356506ff8aac7a` | review_before_optional_sidecar_or_main_project_connection |
| `scripts/final_policy_runner.py` | integration_candidate | package_support_code | manifested | `2cdb5314d751aa350061d898f9a8acceeb7c33366ddfc321b7417ff5b6d51c89` | review_before_optional_sidecar_or_main_project_connection |
| `scripts/policy_contract_validator.py` | integration_candidate | package_support_code | manifested | `a9590abce521e29c54a0b9ff5d8f87420f7bf8a69f6c4b5cbfac991414d93b6b` | review_before_optional_sidecar_or_main_project_connection |
| `scripts/release_rejected_audit.py` | integration_candidate | package_support_code | manifested | `9a80ab362bb1160fb65c5c17d84288e6c8804775299ea08440db4158b228b770` | review_before_optional_sidecar_or_main_project_connection |
| `templates/risk_control_v1_runtime_feature_template.csv` | integration_candidate | runtime_contract | manifested | `3e52d6983a61fdedc79bf1015ccb5f2cf08a101c802ff2add336454d29e462c2` | review_before_optional_sidecar_or_main_project_connection |
| `README.md` | read_only_evidence | package_documentation | manifested | `ece3b41dbb2b04249484009d6fdb0aac6c28e134bf7a7f4f9db0bde6328564b0` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/phase2_hard_gate_deployable_frontier_report.json` | read_only_evidence | review_evidence | manifested | `6af182da17497513f5d0d6d1bc55e4cb59990d079cc9b8b11f56235f5144e33e` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/phase3_allowed_integration_items.csv` | read_only_evidence | review_evidence | manifested | `057ec2dc36301aba29697f0fd6b47a4929e3bb7cb3929f5d4eb4338c5a2b5caa` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/phase3_deployment_failure_modes_cases.csv` | read_only_evidence | review_evidence | manifested | `034b00b862c2409038e04077b2e82814ec7681af7d4c2fbf96c6d4262f692307` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/phase3_deployment_failure_modes_report.json` | read_only_evidence | review_evidence | manifested | `7d92898234fe6bfd4341f8aa83659db02c132406478b85a170acfff711a3ecd5` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/phase3_integration_readiness_report.json` | read_only_evidence | review_evidence | manifested | `378e943b2ca70b09b02d784b5994ec372ede7224e44fa6e9e912e06d902d612c` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/phase3_sidecar_external_interface_freeze_report.json` | read_only_evidence | review_evidence | manifested | `180346dee4a33225a88970da5d1ca6bb2e5da43cc0c5e3b86891e18ac194f07c` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/phase3_sidecar_interface_compliance_report.json` | read_only_evidence | review_evidence | manifested | `8dc6023b0f16a52e19d9fcafbea6d6bd96ee6f5bdff96faca2931f29b5edfd84` | keep_read_only_do_not_import_as_runtime_logic |
| `outputs/wrapper_deployable_report.json` | read_only_evidence | review_evidence | manifested | `2f214f2ab2ae8937f52e5f59575a9fef3caa664fdeb0c18c8c2e59723eb5cbd8` | keep_read_only_do_not_import_as_runtime_logic |
| `reports/phase2_final_experiment_report.md` | read_only_evidence | review_evidence | manifested | `6539938eb54ca71c6d01aea1290042cb1f96b5778a4d2a4ae532378ae72ce430` | keep_read_only_do_not_import_as_runtime_logic |
| `reports/phase2_hard_gate_deployable_frontier.md` | read_only_evidence | review_evidence | manifested | `66eca0d4dbee8d5efc16a5295b2c7d8626b29378a18e811107d2b4d1ac8c7137` | keep_read_only_do_not_import_as_runtime_logic |
| `reports/phase3_deployment_failure_modes_audit.md` | read_only_evidence | review_evidence | manifested | `0c7b3ce5555242b5e705aa62f2657cd0bf68ebffe6f0c7a6fdda9178e1d1cf97` | keep_read_only_do_not_import_as_runtime_logic |
| `reports/phase3_integration_readiness_audit.md` | read_only_evidence | review_evidence | manifested | `1241ce50fcae01ff76b993d369911fdc47fa87e5c6d1eeb79ec8047f9f178eac` | keep_read_only_do_not_import_as_runtime_logic |
| `reports/phase3_sidecar_external_interface_freeze_audit.md` | read_only_evidence | review_evidence | manifested | `9a2a52a5a9ce18b7df7b76db515dc8320dd383806b9c1be95f2347c696e976bc` | keep_read_only_do_not_import_as_runtime_logic |
| `reports/phase3_sidecar_interface_compliance_audit.md` | read_only_evidence | review_evidence | manifested | `a7e965ff7529b53a25b1e36c0d17c6c512564dbb90072f816285867e28a0d1ec` | keep_read_only_do_not_import_as_runtime_logic |
| `sidecar_original_project_adapter/bridge_outputs/minimal_export_contract_audit.json` | read_only_evidence | export_contract | manifested | `fabe6d0a5b31bd6829436315a8e3b70777cd734ed4dec729e956817160042840` | keep_read_only_do_not_import_as_runtime_logic |
| `sidecar_original_project_adapter/minimal_export_contract.md` | read_only_evidence | export_contract | manifested | `e7c8dc0530c6feb9dc7374fe57b2cbd0bc97e506833f456832fabf8d98e16c91` | keep_read_only_do_not_import_as_runtime_logic |
| `sidecar_original_project_adapter/minimal_export_contract_audit.py` | read_only_evidence | export_contract | manifested | `c2eda54e982db67b0256ee81053835f66d7e58e6b4da7ac5f221a312f80a7380` | keep_read_only_do_not_import_as_runtime_logic |

## Blocked Actions

- `do_not_integrate_review_evidence_as_runtime_logic`
- `do_not_copy_relaxed_or_what_if_artifacts`
- `do_not_use_strict_external_labels_for_threshold_selection`
- `do_not_relax_strict_external_fallback_from_handoff_bundle`
- `do_not_change_original_project_code_from_integrity_audit`
- `do_not_relax_strict_external_fallback_from_artifact_manifest`

## Interpretation

This is the single handoff index for the staged hard-gate package. It keeps integration
candidates, read-only evidence, generated support files, and blocked artifacts separated
for review. It does not authorize original-project writes or strict_external relaxation.