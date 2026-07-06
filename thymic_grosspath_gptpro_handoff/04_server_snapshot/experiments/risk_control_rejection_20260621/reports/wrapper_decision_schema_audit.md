# Wrapper Decision Schema Audit

## Purpose

This audit checks deployment-facing decision CSVs emitted by the wrapper and integration
paths. Each export must carry the same required deployment decision columns and must not
include audit labels, fold metadata, or error-derived selector fields.

## Results

| Export | Passed | Rows | Missing Required | Forbidden Present |
| --- | --- | ---: | --- | --- |
| integration_deployable_decisions | True | 699 | - | - |
| wrapper_deployable_decisions | True | 699 | - | - |
| runtime_template_wrapper_dry_run | True | 1 | - | - |
| synthetic_deployment_scenario | True | 4 | - | - |