# Progress Log

## 2026-06-20

- User clarified that a coarse project scan should happen first, then a more targeted plan should be produced.
- Ran initial file discovery over root and common project metadata.
- Confirmed current path is `/workspace/thymic_project`.
- Confirmed local directory is not a git repository.
- Identified broad project shape: ML/research workspace with scripts, outputs, logs, datasets, paper assets, and vendored third-party code.
- Read coarse entry docs: `thymic_baseline/README.md`, `datasets_full/README.md`, `third_party/README.md`, and `reports/ThymicGross/00_报告导航.md`.
- Sampled key reports for Task7 definition, batch1+batch2 Task5/6/7 result drift, and the 2026-05-27 cross-domain safety workflow line.
- Counted major areas: 223 root-level script files under `scripts/`, 795 output files within two levels of `outputs/`, 49 root log files under `logs/`, and 12 files under `paper/`.
- Read Round3 best-results report, Round3 strong-model report, baseline-stage report, paper manuscript start, and late framework artifacts v186/v192/v194/v198.
- Converted the coarse scan into a targeted deep-assessment plan and marked coarse scan plus targeted planning complete.
- User clarified the eventual goal is code modification after full project reading, but current phase must not modify project code.
- Started deep assessment phase. Research notes may be updated, but project implementation files will not be changed.
- Read core `thymic_baseline` modules: registry, data, models, metrics, cropping, split generation, training, CV aggregation, run summarization, and first-paper runner scripts.
- Read Task5/6/7 preparation and reporting scripts: derived registry generation, batch1+batch2 frozen input creation, 7-model runner, and report generator.
- Read Round3 and Task56 frozen-feature probe scripts for DINOv2, DINO concat, experience auxiliary heads, PLIP, and BiomedCLIP.
- Started data asset inspection. Confirmed key image roots exist and counted files. Default `python` lacks pandas, so further CSV stats will use standard library or the project conda environment if available.
- Completed standard-library CSV/JSON stats for key registries, frozen Task5/6/7 inputs, final safety tables, and residual-error artifacts.
- Read late Task7 risk-control scripts and artifacts around v156/v158/v161/v173/v182/v185/v187/v195/v196/v198/v200.
- Confirmed v156/v158 support the unlabeled severe-shift gate as current-split evidence, while v185 currently hardcodes strict external fallback behavior.
- Confirmed v195's selected action is agreement release / review compression, not automatic flipping correction.
- Found a concrete metric bug: v187 scoped `review_or_reject_rate` divides by all 699 cases rather than by each scope; v198 inherits this for prior v185 scoped rows.
- Confirmed v118 is a key high-safety baseline output but no matching v118 generation script is present in `scripts/`.
- Read v201/v202, the latest discovered risk-control outputs. v201 is a small stable supported-domain flip candidate; v202 packages clean-cohort exclusion analysis around v195 as the main result.
- Checked environment/runbook state: conda env exists, core dependencies import under `thymic_baseline`, one RTX 4090 is available, but root dependency locking and project tests are absent.
