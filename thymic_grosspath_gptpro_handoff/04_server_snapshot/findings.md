# Project Findings

## Coarse Scan Findings

- Working directory: `/workspace/thymic_project`.
- The directory is not a git repository, so version history, branch status, and ownership of changes cannot be assessed from local git metadata.
- Initial file listing suggests this is a research/ML project around thymic pathology or thymic image classification, not a conventional web application.
- Major visible areas include `scripts/`, `outputs/`, `logs/`, `datasets/`, `datasets_full/`, `paper/`, `third_party/`, `reports/`, and `thymic_baseline/`.
- There are many dated experiment scripts and logs, especially around task 5/6/7, gross pathology, DINO/DINOv2/DINOv3, PLIP/BiomedCLIP, routing/cascade/release policy experiments, and external generalization.
- Root storage profile: `outputs/` is about 187G, `datasets/` about 7.6G, `third_party/` about 2.1G, `cache/` about 2.1G, `artifacts/` about 1.6G. This is output-heavy and experiment-heavy.
- `thymic_baseline/README.md` presents a runnable baseline scaffold for the first thymic gross-image paper, including patient-level 5-fold split generation, whole/crop/whole+crop variants, single/dual branch classifiers, and image/case-level evaluation.
- `datasets_full/README.md` says the full thymic gross-image data directory is currently an empty skeleton, not formal newly added data.
- `third_party/README.md` documents external research snapshots and licensing notes; GPLv3 code is marked as unsuitable for direct copying.
- `reports/ThymicGross/00_报告导航.md` identifies the intended reading path as Round3 best results, Round3 strong-model report, second-cycle report, and baseline report.
- Visible core tasks include original Task1-4, later Task5 three-class, Task6 six-class, and Task7 low-risk vs high-risk including TC.
- Task7 is clinically framed as low risk `A/AB/B1` versus high risk `B2/B3/TC`; early derived result on 120 cases reported AUC 0.7956 and BAcc 0.7333.
- A later Task5/6/7 report compares old 120-case experiments with new 285-case batch1+batch2 experiments. Task5 improved slightly, while Task6 and Task7 declined, attributed mainly to distribution shift, class imbalance, single-image-per-case selection, incomplete experience labels, and unchanged training strategy.
- A long 2026-05-27 record describes mature Task7 cross-domain safety workflows: v50/v75/v79/v82 style two-stage review, quality gating, unlabeled batch audit, and low-risk/high-risk guard policies. It explicitly warns against tuning on strict external truth and distinguishes exploratory external-exposed findings from formal model selection evidence.
- `thymic_baseline/config.py` defines canonical task configs and class orders. `thymic_baseline/train.py` is a CLI training entry point with CV support, timm-style pretrained backbone handling, class weighting, sampler environment variables, and smoke-test batch limits.
- Top-level `scripts/` has 223 files; outputs selected by reports rather than script naming alone must be treated as canonical.

## Open Questions

- What is the intended final deliverable: paper, reproducible training pipeline, inference tool, benchmark report, or deployable model package?
- Which scripts and outputs are canonical versus exploratory/obsolete?
- Whether local data and model artifacts required for reproduction are present.
- Whether the paper claims match the latest experiment outputs.
- Which workflow is currently intended as the final paper/product candidate: early `thymic_baseline`, Round3 best single models, Task5/6/7 merged-data models, or the later grosspath risk-control framework.
- Whether “strict external” evaluation data is present locally and whether any reported external results were used only for validation or also leaked into model/threshold selection.

## Coarse Interpretation

- The project appears to have evolved through at least four phases:
  1. A reproducible first-paper baseline scaffold and early manuscript around 160 cases.
  2. A Round3 strong-representation phase centered on DINOv2 frozen probes and Task1-4 improvements.
  3. A Task5/6/7 phase using 120-case and later 285-case data, where merged data exposed distribution-shift and class-imbalance issues.
  4. A late Task7 risk-control/selective-diagnosis framework with unlabeled shift audit, stable release, high-safety fallback, residual-risk boundaries, and Wilson confidence intervals.
- `paper/main.tex` still reflects the early baseline manuscript state, not the later v185/v195/v198 framework results.
- Later outputs indicate the likely current headline has moved to a risk-controlled, shift-adaptive selective diagnosis framework rather than a raw classifier.
- The most mature late-stage narrative appears around v185/v192/v194/v198:
  - v185: recommended unlabeled shift-adaptive workflow, all-domain BAcc 99.81%, review/reject 58.94%, FN=1, FP=0.
  - v192: explicitly records residual boundary; remaining automatic error is case 2516531 and non-leaky sentinels failed to rescue it.
  - v194: has a drafted Results section and claim-evidence map.
  - v198: efficiency-upgraded workflow v195 increases all-domain automatic decisions from 287 to 447 and reduces review/reject from 58.94% to 36.05%, with one observed auto error.
- A major handover risk is that the repository contains multiple plausible "final" states. The final accepted narrative cannot be inferred from file recency alone; it must be confirmed by artifact lineage and claim consistency.

## Deep Assessment Findings

### Baseline Engineering Layer

- `thymic_baseline` is a runnable Python package, not just loose scripts.
- Registry loading enforces required columns, fills optional `task_l5_label`/`task_l6_label`/`task_l7_label`, supports `include_main_study`, and uses `case_id` as fallback `patient_id`.
- Split generation uses `StratifiedKFold` on `split_stratification_class`; training then rotates train/val/test by using the held-out fold as test and the next fold as validation.
- Image expansion indexes image files recursively and fails on duplicate filenames or missing image files, which is good for leakage/ambiguity control.
- Evaluation writes both image-level and case-level outputs. Case aggregation supports `mean`, `max_prob`, and `majority_vote`; main reports generally use case-level mean.
- The baseline model layer uses timm backbones with project-local Hugging Face cache defaults under `artifacts/huggingface`.
- The data layer has later augmentation profiles controlled by environment variables: `THYMIC_AUG_PROFILE=external_mimic/domain_robust/style_light`, plus sampler controls in `train.py`. This indicates the baseline package has been extended for later domain-robust experiments.
- Crop handling supports optional `THYMIC_CROP_CACHE_DIR` and uses a background-distance specimen crop with center fallback.
- Server and PowerShell runners include dependency checks, split generation, pretrained weight warmup, skip/resume behavior, and CV aggregation.
- Potential script-level risk to verify later: the PowerShell smoke path appears to append duplicate `--pretrained-mode` and `--device` arguments after already adding them.

### Task5/6/7 Data Preparation Layer

- `scripts/prepare_task56_registry.py` derives Task5/6/7 labels from the original registry:
  - Task5: `A/AB -> A_AB`, `B1/B2/B3 -> B123`, `TC -> TC`.
  - Task6: `A`, `AB`, `B1`, `B2`, `B3`, `TC`.
  - Task7: `A/AB/B1 -> low_risk_group`, `B2/B3/TC -> high_risk_group`.
- `scripts/prepare_batch1_batch2_task567_inputs.py` creates the later 285-case frozen dataset from `datasets/thymic_gross_images` and `datasets/thymic_gross_images_batch2`.
- The batch1+batch2 frozen input intentionally materializes one training image per case:
  - single-image cases use the first image;
  - multi-image cases use the second image.
- Training images are copied into `outputs/batch1_batch2_task567_20260514/frozen_inputs/selected_images` with unique prefixed names.
- Combined case ids are source-prefixed as `batch1_*` and `batch2_*` to avoid cross-batch id collisions.
- Batch1+batch2 fold assignment is custom round-robin by Task6 label with fold-size balancing, not the original `StratifiedKFold` generator.
- Experience labels from batch1 and batch2 are merged into `combined_experience_label_soft.csv`; the script permits case-level fallback when the selected image does not have an exact experience-label row.
- `scripts/run_batch1_batch2_task567_7models.sh` runs seven model families for each of Task5/6/7: two SE-ResNeXt50 variants, three DINOv2 single backbones, DINOv2 vits+vitb concat, and DINOv2 vits+vitb with experience auxiliary labels.
- `scripts/generate_batch1_batch2_task67_reports.py` is hardcoded to summarize the Task5/6/7 7-model runs and write report artifacts under `reports/ThymicGross/batch1_batch2_task67_reports_20260514`.

### Frozen-Feature Probe Layer

- `scripts/run_dinov2_frozen_probe.py` is the generic Round3 frozen DINOv2 probe.
- It extracts DINOv2 features with feature modes `cls`, `patch_mean`, or `cls_patchmean`; `whole_plus_crop` concatenates feature vectors from both views.
- Probe choices include logistic regression, MLP, and LDA. Logistic regression selects `C` on the validation fold, then evaluates the held-out test fold.
- The probe can fit at image level or case-feature level; outputs still include image-level and case-level predictions/metrics.
- `scripts/run_task56_dinov2_probe.py` adapts the same probe machinery to Task5/6/7 with a local `TASK56` config.
- `scripts/run_task56_dinov2_concat_probe.py` concatenates two DINO backbones and explicitly checks labels, case ids, and image names for identical order before concatenation.
- `scripts/run_task56_dino_experience_aux.py` adds an MLP with a main classification head and auxiliary heads for experience-label fields. Missing auxiliary labels use `ignore_index=-100`; each fold logs auxiliary coverage.
- PLIP and BiomedCLIP scripts follow the same broad frozen-feature + logistic-regression + case-aggregation pattern and appear to be comparison branches rather than the dominant final line.

### Data Asset Snapshot

- Present image directories include:
  - `datasets/thymic_gross_images`: 168 `.JPG` files.
  - `datasets/thymic_gross_images_batch2`: 174 `.JPG` files.
  - `datasets/third_batch_306_20260521` plus transformed variants.
  - `datasets/external_thymoma_carcinoma_20260522` plus transformed variants.
- `outputs/batch1_batch2_task567_20260514/frozen_inputs/selected_images` contains 285 `.JPG` files, matching the frozen 285-case Task5/6/7 setup.
- The default `python` on this machine does not have `pandas`; project execution likely depends on activating the documented `thymic_baseline` conda environment.
- Current `datasets/thymic_case_registry_seed_from_folders.csv` has 157 rows, not 160: 20 each for benign, MNT, A, AB, B1, B2, B3, but only 17 thymic carcinoma cases.
- Current `outputs/task567_registry.csv` therefore has Task5/6/7 counts based on 117 included cases after excluding benign and MNT: Task7 low=60, high=57.
- The batch1+batch2 frozen 285-case dataset has counts:
  - source: batch1=117, batch2=168.
  - Task6: A=44, AB=50, B1=50, B2=60, B3=24, TC=57.
  - Task7: low=144, high=141.
  - folds: exactly 57 cases per fold.
  - missing experience labels: 81 rows, all from batch1.
  - excluded rows: 40, all batch1 benign hyperplasia or micronodular thymoma.
- Preflight status for the 285-case frozen input is `pass`.
- Later safety-framework tables use domains `old_data` n=285, `third_batch` n=306, `strict_external` n=108, all-domain n=699.
- The single residual automatic error recorded in v192 is third-batch case `2516531`, label `TC`, high-to-low false negative.

### Late Task7 Risk-Control Framework

- v118 is the conservative high-safety baseline: fixed two-signal scorecard `wholecrop_prob <= 0.625` and `prob_mean_core <= 0.775`, all-domain BAcc 99.81%, review/control 79.97%, FN=1, FP=0.
- The v118 output directory exists, but no matching `scripts/run_grosspath_rc_v118...py` script is present. Later scripts consume `outputs/grosspath_rc_v118_global_two_signal_scorecard_20260527/v118_global_two_signal_cases.csv`, so v118 is a critical reproducibility gap unless generated elsewhere outside the current script set.
- v156/v158 establish the unlabeled severe-shift gate as evidence, not a fully general deployment implementation:
  - v156 maps old, third, and strict external batch audits from v77, then only triggers concept-direction correction for domains classified as `severe_shift`.
  - v158 stress-tests nine no-label gate variants; the reports say 7/9 variants cleanly isolate strict external in the current data, while also warning that over-sensitive or too-strict gates can be unsafe.
  - This supports the mechanism, but the current v185 deployment script still hardcodes `strict_external` as the severe domain rather than computing it from incoming unlabeled batch metrics.
- v161 reduces review burden by selecting internal zero-error release rules from the v118 review pool. It keeps all-domain BAcc 99.81% while reducing review/reject from about 80% to about 57.5%.
- v173 trains image/tabular review-pool correctors, but its own write-up indicates the best selected operating points mostly release already-correct cases; they do not reliably rescue base-model errors. This is why later work treats them as safe-release/review-compression candidates, not true auto-correction.
- v182 stabilizes the v178 image-agreement release by requiring one fixed rule to satisfy all-fold train-zero and held-out-zero constraints. It becomes the conservative efficiency workflow: all-domain BAcc 99.81%, review/reject 56.08%, FN=1, FP=0.
- v185 combines v182 and v118 into the main shift-adaptive selective workflow: within-internal-shift batches use v182; strict-external-like severe-shift batches fall back to v118. Its reported all-domain BAcc is 99.81%, review/reject 58.94%, FN=1, FP=0.
- v187 computes automatic-decision Wilson confidence intervals for v118/v182/v185. It correctly distinguishes zero additional release errors from the one residual automatic-decision error.
- v187 has a denominator bug in per-scope `review_or_reject_rate`: inside each scope loop it records `(df[review_col] & mask).mean()`, which divides by the full 699-case table instead of the scoped subset. The all-domain row is correct; scoped review rates in `v187_auto_decision_error_ci.csv` are understated. v198 inherits prior v185 rows from v187, so v198 per-scope v185 review rates are also wrong, while the all-domain headline remains correct.
- v195/v198 form a later efficiency-upgraded line. v195 scans action modes but the selected current result is `agreement_release_only`, not true automatic correction. It increases all-domain automatic decisions from 287 to 447 and drops review/reject to 36.05%, with one observed automatic/system error.
- v196 nested validation is less favorable than the fixed v195 selected result: all-domain nested-plus-locked-external review/reject is about 30.61%, but with 3 action errors and BAcc around 99.40%. This means v195 should be framed carefully as a selected efficiency-upgraded workflow, not an already prospectively locked deployment threshold.
- v200 is a cautious later attempt at directional flipping. It allows flips only when same-domain training support has a no-harm rescue rule; strict external gets no flip because there is no same-domain label support. Its own interpretation says v195 remains the main review-compression module.
- v201 improves the directional-flip branch into the first small non-harmful automatic correction candidate: 2 all-domain flips, 2 rescues, 0 hurt/action errors, all-domain BAcc 99.81%, review/reject 58.66%. Its own write-up still treats it as complementary to v195, not as the main module.
- v202 appears to be the latest packaging layer. It defines a doctor-reviewed clean primary cohort with five exclusion-list cases, two of which are present in the current Task7 main flow (`2307206`, `2205101`). Its main v195 clean-cohort result is n=697, BAcc 99.81%, Acc 99.86%, FN=1, FP=0, review/reject 36.01%, auto errors=1. Full-cohort sensitivity remains n=699, BAcc 99.81%, review/reject 36.05%.

### Environment and Verification Readiness

- The workspace is not a git repository, so there is no commit history, branch state, or clean/dirty worktree signal available locally.
- There is no root-level locked dependency file such as `requirements.txt`, `environment.yml`, or `pyproject.toml` for the project itself. Environment setup is script-based through `scripts/server_create_conda_env.sh`.
- Current machine has conda envs `thymic_baseline` and `thymic_plip`. In `thymic_baseline`, core imports are currently available: torch 2.5.1+cu124, torchvision 0.20.1+cu121, timm 1.0.26, sklearn 1.8.0, cv2 4.13.0, pandas 3.0.2, numpy 2.4.5, PIL 12.2.0.
- GPU is available: NVIDIA GeForce RTX 4090 with 49140 MiB memory.
- Project-local tests are essentially absent. The only discovered pytest files are under `third_party/condor_pytorch`; no obvious test suite exists for `thymic_baseline` or the risk-control scripts.
- Repository size is about 201G; `outputs/` dominates. This makes brute-force scans and whole-tree operations slow and increases the need for targeted commands.
- Runbooks exist for the first-paper baseline, but the server runbook is mojibake-encoded in this environment. Commands remain readable, but documentation quality is degraded.
