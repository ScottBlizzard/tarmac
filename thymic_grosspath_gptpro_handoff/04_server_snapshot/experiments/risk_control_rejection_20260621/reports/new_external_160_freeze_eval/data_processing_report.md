# New External 160 Data Processing Report

## Scope
This report covers data processing only. No model inference, model selection, threshold tuning, retraining, or v195/v195+ evaluation was run.

## Input
- Raw directory: `/workspace/最新外部数据集/胸腺瘤/`
- Raw files: 166
- File extensions: all `.jpg`
- File signatures:
  - JPEG: 115
  - PNG content with `.jpg` extension: 51

## Outputs
- Raw manifest: `experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/new_external_160_raw_manifest.csv`
- Case manifest: `experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/new_external_160_case_manifest.csv`
- Duplicate groups: `experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/new_external_160_duplicate_groups.csv`
- Staging images: `experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/staging_dedup_images/`
- QC report JSON: `experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/new_external_160_manifest_qc_report.json`

## Processed Counts
- Raw file rows: 166
- Case-level rows after grouping by final numeric identifier: 162
- Duplicate identifier groups: 4
- Extra duplicate files: 4
- Staging files: 162
- Missing staging paths: 0

## Case-Level Distribution

| Label | Cases |
|---|---:|
| A | 22 |
| AB | 29 |
| B1 | 26 |
| B2 | 28 |
| B3 | 29 |
| TC | 28 |

| Task7 group | Cases |
|---|---:|
| low_risk_group | 77 |
| high_risk_group | 85 |

## Duplicate Groups

| Identifier | Case label used | Task7 group | Notes |
|---|---|---|---|
| 1016232 | B3 | high_risk_group | Two visually same images; detailed diagnosis contains mostly B2 plus small B3 component. |
| 1027127 | AB | low_risk_group | Two byte-identical files with different detail level in filename. |
| 1099577 | TC | high_risk_group | Two byte-identical files; detailed diagnosis contains B3 plus squamous carcinoma component. |
| 1136278 | TC | high_risk_group | Two byte-identical files with different detail level in filename. |

## Notes
- Staging images are copied, not symlinked, because symlink creation was unavailable in this environment.
- Staging filenames use the historical Chinese label tokens (`A型`, `AB型`, `B1型`, `B2型`, `B3型`, `胸腺鳞状细胞癌`) for later compatibility.
- Mixed diagnoses are mapped to the highest-risk subtype for the subtype column. For Task7 low/high labels, all mixed B2/B3/TC cases remain high-risk either way.
- The raw uploaded data was not modified.
