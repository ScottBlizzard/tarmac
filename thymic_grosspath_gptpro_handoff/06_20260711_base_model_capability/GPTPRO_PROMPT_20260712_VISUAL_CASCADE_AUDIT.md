# GPTPro Prompt: Audit and Redesign the Image-Grounded Capability Mainline

You are a critical senior research lead in medical AI, computational pathology, domain generalization, small-data learning, and clinical validation.

Inspect this repository carefully:

Repository: https://github.com/ScottBlizzard/tarmac

Package: `thymic_grosspath_gptpro_handoff/`

Read these files first:

1. `06_20260711_base_model_capability/reports/Task7_Visual_Capability_and_Genuine_Coarse_to_Fine_Reframing_20260712.md`
2. `06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
3. `06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`
4. `06_20260711_base_model_capability/reports/FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`
5. `01_local_reports_md_csv/2026-05-21_Task7旧数据二阶段模型阶段精华_医生版.md`
6. `01_local_reports_md_csv/2026-05-26_GrossPath-RC_v1冻结式工作流实验报告.md`
7. Relevant implementations, especially the No.64 reviewer, old/third meta-stack, and 2026-07-11 dense-token experiments.

Use older folders to verify provenance and avoid proposing repeats. Do not merely summarize the package.

## Scientific Objective

Task7 classifies thymic gross pathology images at 100% coverage:

- Low risk: A / AB / B1.
- High risk: B2 / B3 / thymic carcinoma.

The goal is not necessarily a single network. Multi-model systems are acceptable only when their gain comes from complementary image evidence learned by visual models. A coarse-to-fine cascade is acceptable when the second model re-reads raw images, high-resolution regions, patches, tumor/cut-surface ROIs, or additional views and learns a genuinely finer visual decision function.

Probability-only correction is not counted as base visual capability. Confidence, disagreement, calibration, error-prediction scores, rejection, and selective review may be downstream routing or safety tools, but they cannot substitute for image-grounded diagnostic improvement.

## Critical Historical Correction

The project previously blurred several different results:

- Early single visual model on old_data (285 cases): about 0.765 Acc/BAcc.
- Candidate 41 multi-model base fusion: 0.8351 Acc, 0.8348 BAcc.
- No.64 two-stage automatic reviewer: 0.9263 Acc/BAcc.
- Later `base162`: 0.9228 Acc and 0.9227 BAcc on old_data.

The 92% results must not be casually described as a 92% single/base visual model.

Code audit shows that the best No.64 configuration used:

- Router: `best41_wrong_model_visible_extra`.
- Corrector: `model_non_easy_extra`.
- Winning corrector feature set: `model`, not DINO or raw-image features.
- Main inputs: model probabilities, predicted labels, confidence/review scores, disagreement-related behavior, image count, and selection rule.
- Old-data result: 61/285 routed, 26 observed rescues, 0 observed harms.

`base162` then inherited the No.64/adaptation/meta-stack lineage and used a final `0.8 * run108 + 0.2 * run104` blend. It emits a prediction for every case, but its information lineage remains behavior-level meta-correction rather than a newly learned fine visual representation.

Its transfer was poor:

- Third batch all: Acc 0.8301, BAcc 0.7718, high-risk recall 0.6463.
- Third-batch strict holdout: Acc 0.8675, BAcc 0.7300, high-risk recall 0.5263.
- Strict external evaluable cohort: Acc 0.6381, BAcc 0.6220, high-risk recall 0.4681.

This is consistent with a model that learned source-specific error signatures and calibration patterns rather than transferable fine morphology.

## Current Image-Grounded Candidates

Internal development contains 591 cases: batch1 117, batch2 168, and third_batch 306.

The current locked image-grounded candidates are:

`C1`: SigLIP-L at 512 px, six deterministic views from one primary image, all dense patch tokens, gated pooling.

- OOF BAcc/AUC: 0.7477/0.8240.
- Source-LODO BAcc/AUC: 0.7397/0.8072.

`C2`: equal average of AIMv2 MixStyle and C1.

- OOF BAcc/AUC: 0.7514/0.8377.
- Source-LODO BAcc/AUC: 0.7441/0.8108.

These are closer to the intended base visual capability but are not a breakthrough and have not been confirmed on a genuinely fresh external cohort.

Both historical external cohorts, 108 and 162 cases, have already been inspected. They are consumed audit sets and cannot be used for further model selection, threshold tuning, fusion tuning, or confirmation.

## Your Task

Answer as a skeptical research lead. The team can begin experiments without waiting for your reply; your role is an independent design and leakage audit.

### 1. Audit the historical correction

- Is it scientifically correct to reclassify No.64 and base162 as behavior-level meta-correction rather than base visual capability?
- Inspect the implementation for direct leakage, non-nested feature generation, target-derived difficulty strata, reuse of OOF predictions, route-score leakage, hyperparameter/model-selection optimism, and threshold-selection bias.
- Separate what is valid cross-fitting from what remains optimistically selected on the same 285 cases.
- State whether any portion of the 92% result can defensibly support a claim of learned fine visual morphology.

### 2. Define an operational taxonomy

Give strict criteria that separate:

- a single visual classifier;
- a legitimate image-grounded ensemble;
- a genuine coarse-to-fine visual cascade;
- a behavior-only meta-corrector;
- a selective/rejection workflow.

Specify which outputs belong in the main full-coverage model table and which must be reported only as safety/workflow analyses.

### 3. Redesign the next experiments

Produce an executable plan for two parallel routes:

**Route A: stronger direct image classifier.**

**Route B: genuine coarse-to-fine visual cascade**, where the second model sees image evidence not adequately used by stage 1.

For each experiment, specify:

- changed assumption;
- exact input information;
- architecture and training target;
- patient-level split and nested-selection logic;
- source-LODO protocol;
- minimum ablations;
- compute-conscious pilot;
- success threshold and stopping rule.

Do not recommend ordinary backbone swaps, another small loss-weight search, another confidence threshold sweep, another internal fusion sweep, or any method already exhausted unless you identify a truly changed information source or causal assumption.

### 4. Prove whether stage 2 learned new visual knowledge

Design a same-routed-case experiment comparing:

- M0: stage 1 retained;
- M1: probabilities/confidence/disagreement only;
- M2: stage-2 image only;
- M3: stage-2 image plus stage-1 outputs;
- M4: matched random routing plus stage-2 image.

Define the primary conditional estimand, paired statistical test, rescue/harm accounting, B1/B2 analysis, and the result pattern required to conclude that stage 2 learned transferable visual evidence.

### 5. Judge the proposed first visual specialist

The provisional plan uses C1 as stage 1 and a high-resolution patch/ROI specialist as stage 2. Critique and improve:

- patch extraction and scale;
- trainable versus deterministic ROI;
- global/local context preservation;
- MIL versus cross-attention;
- boundary-case training with matched controls;
- class/subtype auxiliary supervision;
- prevention of background, source, and acquisition shortcuts;
- how to handle the fact that only 17/591 internal cases have two original images.

### 6. Set advancement and external-validation rules

- Define internal OOF and source-LODO advancement criteria relative to C1/C2.
- Define how many candidate families may be compared before a fresh lock.
- State how to avoid a new meta-selection layer overfitting the 591 cases.
- Review the fresh external blind-test protocol and propose changes if needed.
- Give a practical center/subtype/sample-size target, especially for high-risk sensitivity and B1/B2 precision.

### 7. Give a prioritized decision tree

Provide the first 3-5 experiments in execution order, with explicit go/no-go outcomes. Separate:

- experiments that can start now with existing images and cached models;
- experiments requiring physician ROI/view adjudication;
- experiments requiring new multi-center or multi-view data.

## Required Output Structure

1. `Verdict on the 92% Historical Result`
2. `Operational Taxonomy of Acceptable and Unacceptable Multi-Model Systems`
3. `Leakage and Selection-Bias Audit`
4. `Route A: Direct Visual Model Plan`
5. `Route B: Genuine Coarse-to-Fine Visual Cascade Plan`
6. `Same-Routed-Case Causal Ablation`
7. `Advancement, Stopping, and Fresh-External Rules`
8. `First Five Experiments and Go/No-Go Decisions`

Be explicit about uncertainty. Distinguish verified facts, code-based inference, and recommendations. Do not redirect the project toward release coverage or confidence-only correction.
