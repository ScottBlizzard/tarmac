# Thymic Gross Pathology AI: GPTPro Handoff Package

Updated: 2026-07-12

This package is a text-first handoff for GPTPro to rethink the project from a broad, exploratory perspective. It intentionally includes many reports, scripts, configs, and result summaries so GPTPro can inspect both successful and failed directions.

## Important Privacy Boundary

This package excludes raw medical images, image review folders, large ZIP packages, model checkpoints, and full prediction-level image/case output tables where avoidable. It is intended for method design and experiment planning, not for direct model training.

## Folder Map

| Folder | Contents | Notes |
| --- | --- | --- |
| `01_local_reports_md_csv/` | Local Markdown/CSV reports from `D:\影响分析\汇报` | Main narrative history and doctor-facing reports. |
| `02_local_scripts_py/` | Local Python scripts from `D:\影响分析\scripts` | Earlier Task1-Task7, curriculum, third-batch, external, and risk-control code. |
| `03_local_result_summaries/` | Local result summaries copied with original partial structure | Some deep paths were truncated by Windows path limits; use the flat folder too. |
| `03_local_result_summaries_flat/` | Flattened local summary/result/metrics files | Safer complete local result-summary copy. |
| `04_server_snapshot/` | Server-side text/code snapshot from `/workspace/thymic_project` | Includes latest reports, scripts, v195+ sidecar, configs, and summary metrics. |
| `05_20260706_base_model_expansion/` | Earlier base-model expansion evidence and scripts | Four-domain audit, first external sweeps, subtype-aux ablation, and the plan that preceded the completed 2026-07-11 wave. |
| `06_20260711_base_model_capability/` | Completed capability expansion plus the 2026-07-12 interpretation correction | Read this first for the historical 92% audit, genuine coarse-to-fine plan, runs 206-369, candidate lock, and fresh-external protocol. |
| `GPTPRO_PROMPT.md` | Ready-to-copy prompt for GPTPro | Start here if only one file can be read first. |
| `MANIFEST.csv` | File manifest generated after packaging | Use to search filenames quickly. |

## Suggested Reading Order

Start with the completed 2026-07-11 update:

1. `GPTPRO_PROMPT.md`
2. `06_20260711_base_model_capability/GPTPRO_PROMPT_20260712_POST_F1_F2.md`
3. `06_20260711_base_model_capability/reports/Task7_WaveC_D1_E1_F1_F2_Results_20260712.md`
4. `06_20260711_base_model_capability/reports/PHYSICIAN_ROI_ANNOTATION_LOCK_WORKFLOW_20260712.md`
5. `06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
6. `06_20260711_base_model_capability/reports/Task7_Visual_Capability_and_Genuine_Coarse_to_Fine_Reframing_20260712.md`
7. `06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`
8. `06_20260711_base_model_capability/scripts/phase2_fresh_external_candidate_lock_20260711.csv`
9. `06_20260711_base_model_capability/reports/FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`

Then read the 2026-07-06 update for earlier context:

1. `GPTPRO_PROMPT.md`
2. `05_20260706_base_model_expansion/README_20260706_BASE_MODEL_EXPANSION.md`
3. `05_20260706_base_model_expansion/reports/Task7_Base_Model_Expansion_Ledger_20260706.md`
4. `05_20260706_base_model_expansion/reports/AI Pathology Model Improvement.md`
5. `05_20260706_base_model_expansion/server_metrics/experiments/base_model_expansion_20260706/outputs/`

Then use the older package history as background.

1. `GPTPRO_PROMPT.md`
2. `01_local_reports_md_csv/项目阶段性工作详报.md`
3. `01_local_reports_md_csv/Task7项目完成度评估.md`
4. `01_local_reports_md_csv/三批数据集外部泛化拒识增强方案报告_.md`
5. `01_local_reports_md_csv/Task7模型升级负结果阶段汇总_2026-05-16.md`
6. `01_local_reports_md_csv/2026-05-18_Task7课程学习与经验标签阶段汇报_医生版.md`
7. `01_local_reports_md_csv/2026-05-25_Task7第三批与严格外部集后续尝试阶段汇报_医生版.md`
8. `01_local_reports_md_csv/2026-05-26_GrossPath-RC_v2域泛化诊断与训练清单.md`
9. `04_server_snapshot/task_plan.md`, `progress.md`, `findings.md`
10. `04_server_snapshot/experiments/risk_control_rejection_20260621/reports/`
11. `04_server_snapshot/experiments/risk_control_rejection_20260621/v195_plus_sidecar/`
12. Result summaries in `04_server_snapshot/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/`

## Current Project Focus

The project is thymic gross pathology image AI. The clinically central task is low-risk vs high-risk thymic epithelial tumor classification:

- Low risk: A / AB / B1
- High risk: B2 / B3 / thymic carcinoma

The immediate scientific bottleneck is not selective release itself, but poor external generalization of the base forced-classification model. The risk-control workflow is useful, but it should not replace a stronger base model.

## Key Dataset Counts

Current three-domain modeled/audited set:

- `old_data`: 285 cases/images, low 144, high 141
- `third_batch`: 306 cases/images, low 224, high 82
- `strict_external`: 108 cases/images, low 61, high 47

Additional newer external set:

- Raw upload: 166 files
- Case-level deduplicated: 162 cases
- Low 77, high 85
- Class counts: A 22, AB 29, B1 26, B2 28, B3 29, TC 28
- Processing/QC finished; qkvb-family model inference has now been run in the 2026-07-06 update.

As of 2026-07-11, both external cohorts have been inspected and are consumed audit sets. Phase-2 candidates C1/C2 require a new label-blinded external cohort; neither existing external cohort may be reused for selection or confirmatory claims.

## Key Baseline Situation

The old-data numbers must be separated by method layer:

- Early direct visual mainline: approximately 76.5% Acc/BAcc.
- Candidate 41 image-model fusion: approximately 83.5% Acc/BAcc.
- No.64 behavior-level two-stage reviewer: approximately 92.6% Acc/BAcc.
- Later `base162`, which inherits the No.64/adaptation/meta-stack lineage: approximately 92.3% Acc/BAcc on old data.

The 92% result is a full-coverage automatic output, but it is not evidence of a 92% single/base visual model. Its incremental gain mainly came from modeling probabilities, confidence, disagreement, and source-specific error behavior. On the third-batch strict holdout, `base162` BAcc was 0.7300 with high-risk recall 0.5263; on the historical strict external evaluable cohort, BAcc was 0.6220.

The current image-grounded C1/C2 candidates reach approximately 0.74-0.75 BAcc under unified 591-case OOF/source-LODO evaluation. They are not directly numerically comparable to the old 285-case protocol and are not externally confirmed.

Representative selective workflow status:

- v195/v195+ can reduce automatic errors under a review/reject workflow, but the review burden remains substantial and this is not the same as a strong full-coverage base model.

## What GPTPro Should Optimize For

The user accepts either a stronger direct visual model or a genuine image-grounded multi-model system. In a valid coarse-to-fine system, stage 2 must re-read raw images, high-resolution patches, ROIs, or additional views and learn visual evidence that stage 1 did not adequately use. Confidence/disagreement may be a router or safety signal but cannot be the source of the diagnostic claim.

Do not redirect the project toward release coverage, rejection rules, probability-only correction, ordinary backbone swaps, or another internal threshold/fusion sweep. Treat prior failures as negative evidence and propose experiments with a changed visual information source or causal assumption.
