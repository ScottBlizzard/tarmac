# New External 160 Freeze Evaluation Progress

## 2026-06-23

- Confirmed user instruction: the new external dataset must be used only for frozen evaluation, not reverse tuning.
- Confirmed raw upload directory and initial counts: 166 files under `/workspace/最新外部数据集/胸腺瘤/`.
- Confirmed 51 files are PNG content with `.jpg` extension; this is an ingestion/QC issue, not a model result.
- Confirmed duplicate-looking entries should be treated as duplicate case/image records rather than independent cases.
- Read historical external inference scripts and v195+ sidecar runner to identify the needed input interface.
- Created this experiment-local plan directory.
- Added `new_external_160_manifest_qc.py` under the experiment scripts directory.
- Ran data processing only. No model inference, no threshold tuning, no training.
- Initial staging run exposed stale files from a prior naming format; fixed the script to clear staging before rebuilding.
- Re-ran and verified:
  - raw manifest rows: 166
  - case manifest rows: 162
  - staging files: 162
  - missing staging paths: 0
