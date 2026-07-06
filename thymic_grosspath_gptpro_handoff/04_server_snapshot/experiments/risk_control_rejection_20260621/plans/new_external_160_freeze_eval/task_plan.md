# New External 160 Freeze Evaluation Plan

## Goal
Evaluate the newly uploaded external thymic gross-image dataset with previously trained/frozen project workflows, without using the new external data for tuning, threshold selection, rule selection, retraining, or model choice.

## Data Boundary
- Raw input directory: `/workspace/最新外部数据集/胸腺瘤/`
- Experiment-only output root: `experiments/risk_control_rejection_20260621/outputs/new_external_160_freeze_eval/`
- Experiment-only report root: `experiments/risk_control_rejection_20260621/reports/new_external_160_freeze_eval/`
- Original project code and original raw data must not be modified.

## Frozen-Evaluation Discipline
1. New external labels may be used only for final auditing and metrics.
2. New external predictions must not drive threshold changes, candidate selection, rule changes, or model selection.
3. Input normalization, manifest creation, label parsing, duplicate grouping, and format compatibility are allowed because they are data-ingestion steps.
4. Any later improvement inspired by these results must be treated as a new development cycle and validated on another untouched external dataset.

## Phases

| Phase | Status | Purpose |
|---|---|---|
| 1. Plan and interface audit | complete | Locate the historical frozen external inference path and define a no-tuning execution route. |
| 2. New external manifest/QC | complete | Build a case-level manifest from uploaded raw images, map labels to Task7 low/high, record duplicate/format issues. |
| 3. Frozen model inference | pending | Run the existing trained/frozen image-feature workflow on the new external manifest using copied/experiment-local wrappers. |
| 4. Risk-control evaluation | pending | Apply v195 comparator and v195+ sidecar where inputs are available; otherwise explicitly report interface gaps. |
| 5. Report | pending | Summarize raw accuracy, selective auto/review behavior, errors, high-risk FN, and limitations. |

## Current Plan
1. Create an external manifest script under the experiment folder.
2. Use the historical Task7 external folder pipeline as the first runnable model path because it already supports free-form external folders.
3. Keep output in the experiment folder by passing `--output-dir`.
4. If the historical v195/v195+ sidecar cannot be applied directly to newly inferred images, do not fake equivalence; report the missing feature interface and evaluate the available frozen Task7 prediction path first.

## Errors Encountered

| Time UTC | Error | Attempt | Resolution |
|---|---|---|---|
| 2026-06-23 11:03 | Default `python` lacks pandas | CSV inspection | Use `conda run -n thymic_baseline python` for pandas-dependent scripts. |
| 2026-06-23 11:03 | `conda run ... python -c` multiline escaping caused `SyntaxError` | Inline CSV inspection | Use script files or simpler one-line commands. |
| 2026-06-23 11:10 | Staging directory had 324 files after a second manifest run | Re-running manifest after changing staging filename format | Added staging directory cleanup before rebuilding; verified staging file count is now 162. |
