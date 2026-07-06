# New External 160 Freeze Evaluation Findings

## Uploaded Data
- Directory exists: `/workspace/最新外部数据集/胸腺瘤/`
- Raw file count: 166 files.
- All filenames end with `.jpg`.
- Magic-number check found 115 true JPEG files and 51 PNG files with `.jpg` extension.
- File naming pattern appears to encode diagnosis text before the final hyphen and a numeric identifier after it.
- Repeated final numeric identifiers: `1016232`, `1027127`, `1099577`, `1136278`.
- Three repeated groups are byte-identical duplicates; `1016232` is visually the same case/image but not byte/pixel identical.

## Data Processing Outputs
- Script: `experiments/risk_control_rejection_20260621/scripts/new_external_160_manifest_qc.py`
- Output root: `experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/`
- Raw manifest: `new_external_160_raw_manifest.csv`, 166 rows.
- Case manifest: `new_external_160_case_manifest.csv`, 162 rows.
- Duplicate-group table: `new_external_160_duplicate_groups.csv`, 8 rows across 4 identifier groups.
- Staging image directory: `staging_dedup_images/`, 162 files.
- Verified every case manifest row has an existing staging image path.
- Staging filenames use Chinese labels such as `A型`, `AB型`, `B1型`, `B2型`, `B3型`, and `胸腺鳞状细胞癌` so they remain compatible with the historical external-folder parser if model testing is later resumed.

## Case-Level Label Distribution
- A: 22
- AB: 29
- B1: 26
- B2: 28
- B3: 29
- TC: 28
- Task7 low-risk group: 77
- Task7 high-risk group: 85

## Label Mapping Notes
- The current case manifest maps mixed diagnoses to the highest-risk component for subtype reporting.
- For Task7 low/high evaluation this does not change the low/high label for mixed high-risk cases.
- Examples:
  - `1016232`: detailed filename says `B2型胸腺瘤（约占90%）伴少量B3型胸腺瘤（约占10%）`; case-level subtype is currently `B3`, Task7 is high-risk.
  - `1099577`: detailed filename says `B3型胸腺瘤（约占80%）伴有鳞状细胞癌（约占20%）`; case-level subtype is currently `TC`, Task7 is high-risk.

## Historical External Inference Path
- `scripts/run_task7_external_thymoma_carcinoma_folder_20260522.py` supports a free-form external image folder.
- It derives Task7 low/high labels from filenames, extracts DINO features, selects policies from old OOF only, and applies them to external images.
- This path is suitable as a first frozen external-evaluation path if only input and output paths are changed.

## v195/v195+ Interface Finding
- The existing v195+ sidecar reconstructs decisions from historical case tables:
  - `outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv`
  - `outputs/grosspath_rc_v173_image_only_review_corrector_20260527/v173_corrector_case_outputs.csv`
- It does not directly accept raw new images. The new dataset must first be converted into equivalent model/probability/risk-feature inputs before strict v195/v195+ policy evaluation can be valid.
- Per user instruction on 2026-06-23, model testing is paused for now; only data processing has been performed after that instruction.
