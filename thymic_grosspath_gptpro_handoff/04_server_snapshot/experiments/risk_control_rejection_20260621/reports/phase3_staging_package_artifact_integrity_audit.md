# Phase 3 Staging Package Artifact Integrity Audit

## Result

- passed: True
- file_n: 34
- integration_allowed_file_n: 8
- read_only_file_n: 26
- generated_manifest_file_n: 2
- generated_cache_file_n: 6
- unexpected_unmanifested_file_n: 0
- missing_manifest_file_n: 0
- forbidden_path_present_n: 0
- relaxed_artifacts_present: False
- main_project_write_required: False
- original_project_code_modified: False

## Inventory

| Target | Class | Origin | Integration Allowed | Read-only | Bytes | SHA256 |
| --- | --- | --- | --- | --- | ---: | --- |
| `README.md` | package_documentation | manifested | False | True | 2670 | `ece3b41dbb2b04249484009d6fdb0aac6c28e134bf7a7f4f9db0bde6328564b0` |
| `configs/final_policy_contract.json` | runtime_contract | manifested | True | False | 3067 | `37f4f2d2449145a87006c5252fd91964cacfee596157259f3df88659ff2146ac` |
| `integration/__pycache__/risk_control_deployable_wrapper.cpython-311.pyc` | unmanifested | generated_cache | False | True | 7500 | `76f7d90b42c2fc2e19ddbc7e4ae8f5cfae480f3ce592ce3352a6b8cd7b5b88e7` |
| `integration/__pycache__/risk_control_gate_entrypoint.cpython-311.pyc` | unmanifested | generated_cache | False | True | 7408 | `1b5ad3159893f3f46e7a26edc07456ffea9a0824a2b6b4467601fa2570e1112f` |
| `integration/risk_control_deployable_wrapper.py` | deployable_code | manifested | True | False | 4394 | `3e9ff8a93d1a3125e6ec87a93786b92a5470faa61429f8e0aa6bdef5b2d6e676` |
| `integration/risk_control_gate_entrypoint.py` | deployable_code | manifested | True | False | 3946 | `a0ac7b5693ad9076a102d3e735d6cb3b17fae584aa8aa9785827777ae52f2789` |
| `outputs/handoff_file_classification_manifest.csv` | unmanifested | generated_manifest | False | True | 5700 | `cd3ab4741eb0876b68e7b32001d79f1ce5231c154ed002e60c68d25c2a819b80` |
| `outputs/handoff_file_classification_manifest.json` | unmanifested | generated_manifest | False | True | 11449 | `fc425343a9485c0cb4467516cc3f547cbd371fb35ee05a5f16b2e226d2eee54c` |
| `outputs/phase2_hard_gate_deployable_frontier_report.json` | review_evidence | manifested | False | True | 885 | `6af182da17497513f5d0d6d1bc55e4cb59990d079cc9b8b11f56235f5144e33e` |
| `outputs/phase3_allowed_integration_items.csv` | review_evidence | manifested | False | True | 258 | `057ec2dc36301aba29697f0fd6b47a4929e3bb7cb3929f5d4eb4338c5a2b5caa` |
| `outputs/phase3_deployment_failure_modes_cases.csv` | review_evidence | manifested | False | True | 789 | `034b00b862c2409038e04077b2e82814ec7681af7d4c2fbf96c6d4262f692307` |
| `outputs/phase3_deployment_failure_modes_report.json` | review_evidence | manifested | False | True | 572 | `7d92898234fe6bfd4341f8aa83659db02c132406478b85a170acfff711a3ecd5` |
| `outputs/phase3_integration_readiness_report.json` | review_evidence | manifested | False | True | 379 | `378e943b2ca70b09b02d784b5994ec372ede7224e44fa6e9e912e06d902d612c` |
| `outputs/phase3_sidecar_external_interface_freeze_report.json` | review_evidence | manifested | False | True | 776 | `180346dee4a33225a88970da5d1ca6bb2e5da43cc0c5e3b86891e18ac194f07c` |
| `outputs/phase3_sidecar_interface_compliance_report.json` | review_evidence | manifested | False | True | 794 | `8dc6023b0f16a52e19d9fcafbea6d6bd96ee6f5bdff96faca2931f29b5edfd84` |
| `outputs/wrapper_deployable_report.json` | review_evidence | manifested | False | True | 869 | `2f214f2ab2ae8937f52e5f59575a9fef3caa664fdeb0c18c8c2e59723eb5cbd8` |
| `reports/phase2_final_experiment_report.md` | review_evidence | manifested | False | True | 33309 | `6539938eb54ca71c6d01aea1290042cb1f96b5778a4d2a4ae532378ae72ce430` |
| `reports/phase2_hard_gate_deployable_frontier.md` | review_evidence | manifested | False | True | 5271 | `66eca0d4dbee8d5efc16a5295b2c7d8626b29378a18e811107d2b4d1ac8c7137` |
| `reports/phase3_deployment_failure_modes_audit.md` | review_evidence | manifested | False | True | 1361 | `0c7b3ce5555242b5e705aa62f2657cd0bf68ebffe6f0c7a6fdda9178e1d1cf97` |
| `reports/phase3_integration_readiness_audit.md` | review_evidence | manifested | False | True | 1795 | `1241ce50fcae01ff76b993d369911fdc47fa87e5c6d1eeb79ec8047f9f178eac` |
| `reports/phase3_sidecar_external_interface_freeze_audit.md` | review_evidence | manifested | False | True | 2419 | `9a2a52a5a9ce18b7df7b76db515dc8320dd383806b9c1be95f2347c696e976bc` |
| `reports/phase3_sidecar_interface_compliance_audit.md` | review_evidence | manifested | False | True | 978 | `a7e965ff7529b53a25b1e36c0d17c6c512564dbb90072f816285867e28a0d1ec` |
| `scripts/__pycache__/audit_rejection_layer.cpython-311.pyc` | unmanifested | generated_cache | False | True | 28744 | `fa940f0b0a2f38ee4710be9d75765a1f1c1ad4bab6335e642cb232800ec71050` |
| `scripts/__pycache__/final_policy_runner.cpython-311.pyc` | unmanifested | generated_cache | False | True | 14734 | `3f12b31f334a8d0a8771ac4c3526b091c883ced94b483fb4278ecc1214fbdffc` |
| `scripts/__pycache__/policy_contract_validator.cpython-311.pyc` | unmanifested | generated_cache | False | True | 10239 | `2e51f1f21c373a5d05cf228407789c91a8ffa57acc5854d921fa900967f50e0f` |
| `scripts/__pycache__/release_rejected_audit.cpython-311.pyc` | unmanifested | generated_cache | False | True | 19329 | `d77088b8c29bc40145b884f0c02cd7c5a2ced3312f052db46bf050ad206d154c` |
| `scripts/audit_rejection_layer.py` | package_support_code | manifested | True | False | 17429 | `af7a9a2741f0d7a9c3780b81d6dcc1ad244f6a98c16a7c21bc356506ff8aac7a` |
| `scripts/final_policy_runner.py` | package_support_code | manifested | True | False | 9087 | `2cdb5314d751aa350061d898f9a8acceeb7c33366ddfc321b7417ff5b6d51c89` |
| `scripts/policy_contract_validator.py` | package_support_code | manifested | True | False | 5724 | `a9590abce521e29c54a0b9ff5d8f87420f7bf8a69f6c4b5cbfac991414d93b6b` |
| `scripts/release_rejected_audit.py` | package_support_code | manifested | True | False | 13815 | `9a80ab362bb1160fb65c5c17d84288e6c8804775299ea08440db4158b228b770` |
| `sidecar_original_project_adapter/bridge_outputs/minimal_export_contract_audit.json` | export_contract | manifested | False | True | 2432 | `fabe6d0a5b31bd6829436315a8e3b70777cd734ed4dec729e956817160042840` |
| `sidecar_original_project_adapter/minimal_export_contract.md` | export_contract | manifested | False | True | 1489 | `e7c8dc0530c6feb9dc7374fe57b2cbd0bc97e506833f456832fabf8d98e16c91` |
| `sidecar_original_project_adapter/minimal_export_contract_audit.py` | export_contract | manifested | False | True | 9869 | `c2eda54e982db67b0256ee81053835f66d7e58e6b4da7ac5f221a312f80a7380` |
| `templates/risk_control_v1_runtime_feature_template.csv` | runtime_contract | manifested | True | False | 283 | `3e52d6983a61fdedc79bf1015ccb5f2cf08a101c802ff2add336454d29e462c2` |

## Blocked Actions

- `do_not_integrate_review_evidence_as_runtime_logic`
- `do_not_copy_relaxed_or_what_if_artifacts`
- `do_not_change_original_project_code_from_integrity_audit`
- `do_not_relax_strict_external_fallback_from_artifact_manifest`

## Interpretation

This audit locks the staged handoff package as a content-addressed artifact list. It is
a no-write audit for the original project and blocks relaxed/what-if paths from package
handoff.