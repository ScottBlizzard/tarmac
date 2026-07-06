# Phase 3 Export-hook Patch Blueprint

## Result

- passed: True
- recommended_strategy: `review_blueprint_before_any_main_project_write`
- candidate_reviewed_n: 6
- blueprint_ready_n: 4
- manual_review_required_n: 2
- top_ready_path: `thymic_surimage/train.py`
- top_ready_function: `save_standard_outputs`
- main_project_write_performed: False
- strict_external_relaxation_allowed: False
- strict_external_labels_used_for_selection: False
- original_project_code_modified: False

## Blueprint Rows

| Rank | Status | Path | Function | Insert After Line | Action |
| ---: | --- | --- | --- | ---: | --- |
| 1 | `blueprint_ready` | `thymic_surimage/train.py` | `save_standard_outputs` | 395 | after writing case_predictions_mean, call an approved sidecar-safe exporter that materializes only risk_control_v1 runtime-required columns |
| 2 | `blueprint_ready` | `thymic_surimage/strict_train.py` | `save_level_outputs` | 573 | after writing case_predictions_mean, call an approved sidecar-safe exporter that materializes only risk_control_v1 runtime-required columns |
| 3 | `manual_review_required` | `thymic_baseline/aggregate_cv_run.py` | `` | 0 | manual review required before any patch blueprint can be trusted |
| 4 | `manual_review_required` | `thymic_baseline/train.py` | `` | 0 | manual review required before any patch blueprint can be trusted |
| 5 | `blueprint_ready` | `thymic_surimage/ablation_train.py` | `save_fine_outputs` | 163 | after writing case_predictions_mean, call an approved sidecar-safe exporter that materializes only risk_control_v1 runtime-required columns |
| 6 | `blueprint_ready` | `thymic_surimage/evaluate_strict_pipeline.py` | `main` | 135 | after writing case_predictions_mean, call an approved sidecar-safe exporter that materializes only risk_control_v1 runtime-required columns |

## Interpretation

This is a patch blueprint, not a patch. It records candidate insertion points for
a future stable runtime-feature export hook and keeps strict_external relaxation
and label-derived selector fields out of scope.