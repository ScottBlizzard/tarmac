# Phase 3 Export-hook Candidate Prioritization

## Result

- passed: True
- recommended_strategy: `review_core_package_export_candidates_before_any_main_project_patch`
- candidate_n: 70
- shortlist_n: 6
- top_candidate_path: `thymic_surimage/train.py`
- main_project_write_performed: False
- strict_external_relaxation_allowed: False
- strict_external_labels_used_for_selection: False
- original_project_code_modified: False

## Preferred Candidates

- `thymic_surimage/train.py`
- `thymic_surimage/strict_train.py`
- `thymic_baseline/aggregate_cv_run.py`
- `thymic_baseline/train.py`
- `thymic_surimage/ablation_train.py`
- `thymic_surimage/evaluate_strict_pipeline.py`

## Top Ranked Rows

| Rank | Score | Class | Path |
| ---: | ---: | --- | --- |
| 1 | 148 | `core_package_export` | `thymic_surimage/train.py` |
| 2 | 146 | `core_package_export` | `thymic_surimage/strict_train.py` |
| 3 | 130 | `core_package_export` | `thymic_baseline/aggregate_cv_run.py` |
| 4 | 130 | `core_package_export` | `thymic_baseline/train.py` |
| 5 | 110 | `core_package_export` | `thymic_surimage/ablation_train.py` |
| 6 | 110 | `core_package_export` | `thymic_surimage/evaluate_strict_pipeline.py` |
| 7 | 110 | `core_package_export` | `thymic_surimage/evaluate_strict_views.py` |
| 8 | 110 | `core_package_export` | `thymic_surimage/foreground_multicrop.py` |
| 9 | 35 | `historical_script_export` | `run_task7_concat_curriculum_probe.py` |
| 10 | 35 | `historical_script_export` | `scripts/derive_task7_from_task6_predictions.py` |
| 11 | 35 | `historical_script_export` | `scripts/evaluate_foldwise_binary_threshold.py` |
| 12 | 35 | `historical_script_export` | `scripts/evaluate_foldwise_weighted_fine_views.py` |
| 13 | 35 | `historical_script_export` | `scripts/evaluate_hierarchical_gate.py` |
| 14 | 35 | `historical_script_export` | `scripts/evaluate_task4_dino_plip_classblend.py` |
| 15 | 35 | `historical_script_export` | `scripts/evaluate_taskB_foldwise_view_router.py` |

## Interpretation

This audit narrows the broad export-hook candidate set produced by the gap audit.
It prefers maintained package-level training/export modules over dated one-off scripts.
It is still a read-only planning artifact and does not authorize any main-project patch.