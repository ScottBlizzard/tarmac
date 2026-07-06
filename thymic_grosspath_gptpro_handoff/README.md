# Thymic Gross Pathology AI: GPTPro Handoff Package

Date: 2026-07-06

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
| `GPTPRO_PROMPT.md` | Ready-to-copy prompt for GPTPro | Start here if only one file can be read first. |
| `MANIFEST.csv` | File manifest generated after packaging | Use to search filenames quickly. |

## Suggested Reading Order

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
- Processing/QC finished; model inference was not yet run at packaging time.

## Key Baseline Situation

Representative forced-classification status from the reports:

- Old data: about 92.3% accuracy / balanced accuracy in the strongest internal workflow-era model.
- Third batch: about 83.0% accuracy, balanced accuracy about 76.8%, with high-risk recall still weak.
- Strict external: about 64.8% accuracy, balanced accuracy about 62.8%; high-risk recall about 46.8%.

Representative selective workflow status:

- v195/v195+ can reduce automatic errors under a review/reject workflow, but the review burden remains substantial and this is not the same as a strong full-coverage base model.

## What GPTPro Should Optimize For

The user explicitly wants broad, outward-looking method ideation and concrete experiments. Do not prematurely narrow to existing DINO/probe/risk-control variants. Treat previous failures as useful negative evidence, then propose genuinely new base-model generalization directions.

