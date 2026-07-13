# Independent Expert Briefing: Improving Cross-Source Visual Capability After a Sequential-Expert NO-GO

Repository: https://github.com/ScottBlizzard/tarmac

## Your Role

Treat this as your first contact with the project. You have not seen any prior conversation, report, model, or decision. Act as an independent senior research lead in medical imaging and machine learning.

This document is intended to be self-contained. Use the repository only to verify details, inspect code, and challenge the summary. Do not assume that internal names such as C1, C2, OOF, or LODO have any meaning until they are defined below.

We need an evidence-based decision, not encouragement and not another broad hyperparameter sweep. Separate verified facts, causal inferences, unresolved questions, and recommendations.

## Project in One Paragraph

The project classifies gross surgical photographs of thymic epithelial tumors. The available label is one of six pathological subtypes: A, AB, B1, B2, B3, or thymic carcinoma (TC). The current primary task groups A/AB/B1 as low risk and B2/B3/TC as high risk. The intended paper is for a medicine-engineering journal, so cross-source generalization is foundational. The immediate objective is a stronger image-only classifier that emits a low/high prediction for every case. Selective rejection, physician review, source-specific calibration, confidence correction, and output stacking may be downstream safety tools, but they do not count as improvement in base visual capability.

## Clinical and Modeling Task

- Input at present: one selected primary gross photograph per case.
- The locked six-view representation creates deterministic crops from that same photograph; it does not add a second acquisition view or new clinical information.
- Binary target:
  - low risk: A, AB, B1;
  - high risk: B2, B3, TC.
- Primary operating requirement: 100% case coverage at a fixed direct decision rule.
- Primary metrics: balanced accuracy (BAcc), ROC AUC, high-risk sensitivity, and low-risk specificity.
- Safety requirement: an apparent BAcc gain is unacceptable if it is mainly a shift from high-risk sensitivity toward low-risk specificity.
- Multi-model systems are allowed only when each useful stage reads image evidence and demonstrates complementary visual knowledge. A stage that only manipulates confidence, disagreement, or previous model outputs is not considered a stronger visual model.
- Six-class prediction is a later secondary objective. It is not the current optimization target because the six subtype counts are highly imbalanced and the binary boundary is not yet robust.

## Data Available

### Internal development cohort

There are 591 cases from three acquisition batches/sources:

| Source | Cases |
| --- | ---: |
| batch1 | 117 |
| batch2 | 168 |
| third_batch | 306 |
| Total | 591 |

Subtype distribution:

| Subtype | Cases | Binary risk |
| --- | ---: | --- |
| A | 44 | Low |
| AB | 262 | Low |
| B1 | 62 | Low |
| B2 | 89 | High |
| B3 | 24 | High |
| TC | 110 | High |
| Total | 591 | 368 low / 223 high |

The imbalance is important: AB dominates the low-risk group, TC is relatively common in the high-risk group, and A/B3 are small.

### External cohorts

- Historical strict external cohort: 108 cases.
- Newer external cohort: 162 deduplicated cases.

Both external cohorts were already inspected during earlier development. They are consumed audit sets. They must not be reused for model selection, training, threshold selection, calibration, or a new external-generalization claim. A future publication claim requires a genuinely fresh, label-blinded external cohort.

### Physician feature table

Physicians previously recorded visible gross characteristics for 589 of the 591 internal cases. This table can support retrospective error analysis, subtype characterization, and hypothesis generation. It must not be used as a deployment input, a hidden target, or a substitute for image learning. Do not request a new 120-case physician annotation exercise unless you first establish a precise hypothesis for which the existing table is insufficient.

## Evaluation Protocol

Two internal protocols are mandatory:

1. **Five-fold out-of-fold evaluation (five-fold OOF).** Each case is predicted by a model that did not train on that case. Sources are mixed across folds. This estimates ordinary internal generalization.
2. **Source leave-one-domain-out evaluation (source-LODO).** Each of batch1, batch2, and third_batch is held out in turn. The model trains only on the other sources. This is the main internal proxy for domain shift and is harder than mixed-source OOF.

Within each outer fold, model fitting, early stopping, specialist-miss mining, and route-threshold selection use only the outer training side. The held-out fold is used once for evaluation. Case-level splitting is mandatory.

Definitions:

- BAcc = mean of high-risk sensitivity and low-risk specificity.
- Sensitivity = recall for B2/B3/TC.
- Specificity = recall for A/AB/B1.
- OOF and LODO results always refer to predictions covering all 591 cases across their held-out folds.

No result should be advanced merely because accuracy rises under class imbalance, AUC rises without fixed-threshold improvement, or one source/subtype improves while the clinically important boundary deteriorates elsewhere.

## Historical Result That Must Not Be Misinterpreted

An earlier workflow reported approximately 92% internal accuracy. That number is not evidence that a visual classifier learned 92% accurate morphology.

The audit established the following lineage:

- An early direct image model was approximately 76.5% Acc/BAcc on 285 old-data cases.
- A later candidate reached approximately 83.5% by stacking outputs from previously selected corrected systems.
- The approximately 92% descendants were behavior-level reviewers/meta-correctors. Their inputs included previous predictions, probabilities, confidence, disagreement, and in one winning path a `difficulty` feature derived from model correctness and true-class probabilities.
- A later 92% descendant also consumed a nominal holdout during meta-stacking and fold reassignment.
- This lineage did not transfer: one later system achieved only 0.7300 BAcc with 0.5263 high-risk recall on the third-batch strict holdout and 0.6220 BAcc on the historical strict external evaluable cohort.

Therefore, do not use the historical 92% number as a baseline, target, or proof that confidence routing works. The leakage-clean baseline is the direct image evidence described next.

## Current Leakage-Clean Visual Baselines

### C1: strongest locked single visual model

For this briefing, **C1** means:

- pretrained SigLIP-L image encoder at 512 px;
- six deterministic views from the same photograph: whole image, foreground crop, and four crop quadrants;
- all dense patch tokens rather than only a pooled embedding;
- a low-capacity gated pooling classifier;
- direct low/high output at 100% coverage.

| Protocol | Acc | BAcc | AUC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Five-fold OOF | 0.7496 | 0.7477 | 0.8240 | 0.7399 | 0.7554 |
| Source-LODO | 0.7462 | 0.7397 | 0.8072 | 0.7130 | 0.7663 |

### C2: simple fixed visual ensemble

For this briefing, **C2** means an equal-probability fusion of C1 and a separately trained AIMv2 MixStyle image model. Both components read images. C2 is the current comparison baseline, not a behavior-level corrector.

| Protocol | Acc | BAcc | AUC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Five-fold OOF | 0.7597 | 0.7514 | 0.8377 | 0.7175 | 0.7853 |
| Source-LODO | 0.7462 | 0.7441 | 0.8108 | 0.7354 | 0.7527 |

C2 source-LODO subtype risk accuracy was B1 0.5000 and B2 0.6629. This B1/B2 boundary is the recurring central failure.

## What Has Already Been Tried

The repository contains a complete run ledger rather than only selected positive results. Broad tested families include:

- alternative pretrained backbones and scales: DINOv2, DINOv3, EVA02, ConvNeXt, SwinV2, ViTamin, SigLIP-B/L/SO400M, and AIMv2;
- pooled-token versus dense-token representations and multiple view-pooling structures;
- conservative and expanded LoRA fine-tuning;
- class weighting, focal-style objectives, SAM, source-balanced training, GroupDRO, REx, DANN, class-conditional alignment, and MixStyle variants;
- supervised contrastive learning, VICReg initialization, source/subtype positives, and B1/B2 hard negatives;
- binary, six-class auxiliary, ordinal, concept, prototype, boundary-expert, high-risk-sentinel, and mixture-of-experts heads;
- conventional preprocessing, white balance, contrast enhancement, sharpening, and low-quality-image variants;
- native-resolution tiles, fixed spatial pyramids, label-trained attention ROI, deterministic anatomy ROI, random bags, and bag-consistency training;
- confidence routing, image specialist routing, matched-random routing controls, and fixed probability ensembles.

These tests do not prove that every conceivable method in each family is impossible. They do show that another low-level change of backbone, threshold, loss weight, sample count, or fusion weight is not scientifically justified unless it changes a specific failed assumption.

The only clear representation-level gain in the broad search was C1's dense six-view SigLIP-L representation. Many later methods improved ranking AUC or one subtype while moving fixed-threshold errors across sources and between B1 and B2.

## Most Recent Hypothesis: Sequential AB and TC Specialists

### Rationale

The six-class counts suggested a genuine image-grounded hierarchy:

1. detect the abundant low-risk AB subtype;
2. after removing most AB cases, detect high-risk TC against the remaining subtypes;
3. classify all unresolved cases using a low/high fallback trained mainly on A/B1 versus B2/B3, while retaining some AB and TC examples so specialist misses remain classifiable.

This was intended to differ from confidence correction: every stage independently read the frozen dense image tokens.

### Locked implementation

1. Train an AB-versus-rest visual expert.
2. Train a TC expert against A/B1/B2/B3 plus AB cases missed by training-side cross-fitted AB experts.
3. Train a binary fallback on all A/B1/B2/B3 plus 40 dynamically sampled AB and 40 TC cases per epoch. Half of the sampled AB/TC anchors were enriched from training-side expert misses.
4. Route AB-only cases to low risk and TC-only cases to high risk. Send conflicts and non-routes to the fallback.
5. Choose route thresholds from outer-training-side validation predictions only, targeting 98% low-risk purity for AB routing and 95% high-risk purity for TC routing.
6. Use the C1 six-view dense-token representation for all independent heads.

### Result

| Protocol | Model | BAcc | AUC | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| Five-fold OOF | C2 baseline | 0.7514 | 0.8377 | 0.7175 | 0.7853 |
| Five-fold OOF | Sequential system | 0.7195 | 0.8118 | 0.6592 | 0.7799 |
| Source-LODO | C2 baseline | 0.7441 | 0.8108 | 0.7354 | 0.7527 |
| Source-LODO | Sequential system | 0.7278 | 0.7895 | 0.6457 | 0.8098 |

Paired source/risk-stratified bootstrap versus C2 showed:

- source-LODO delta BAcc: -0.0154, 95% CI [-0.0474, 0.0175];
- source-LODO delta AUC: -0.0208, 95% CI [-0.0438, 0.0019];
- source-LODO delta sensitivity: -0.0882, 95% CI [-0.1435, -0.0359];
- source-LODO delta specificity: +0.0574, 95% CI [0.0217, 0.0951].

The system shifted the operating behavior toward low risk rather than learning a stronger boundary.

Source-LODO subtype risk accuracy, sequential system versus C2:

| Subtype | Sequential | C2 | Delta |
| --- | ---: | ---: | ---: |
| A | 0.7273 | 0.6591 | +0.0682 |
| AB | 0.8626 | 0.8282 | +0.0344 |
| B1 | 0.6452 | 0.5000 | +0.1452 |
| B2 | 0.5281 | 0.6629 | -0.1348 |
| B3 | 0.6250 | 0.7083 | -0.0833 |
| TC | 0.7455 | 0.8000 | -0.0545 |

### Mechanism audit

- AB expert BAcc/AUC: 0.7811/0.8505 in five-fold OOF, but only 0.6178/0.6280 in source-LODO. Its AB signal did not transfer across sources.
- TC expert BAcc/AUC: 0.7939/0.8691 in five-fold OOF and 0.7473/0.8450 in source-LODO. Ranking survived better, but no source-held fold met the training-side purity gate, so TC routed zero source-LODO cases.
- Routing was small and net positive: it changed nine five-fold final labels (eight rescued, one harmed) and three source-LODO labels (all three rescued).
- The fallback was the main failure. On the core A/B1 versus B2/B3 cases, fallback BAcc was 0.5765 five-fold and 0.5998 source-LODO, below C2's 0.6072 and 0.6193.

### Formal decision

The preregistered gates required improvement over C2 without sacrificing high-risk safety or source robustness. Only the B1 gate passed. OOF BAcc, source-LODO BAcc, source-LODO sensitivity, minimum-source BAcc, and B2 accuracy failed. The sequential implementation is therefore **NO-GO**.

This decision closes seed repeats, 30/40/50 anchor-count sweeps, route-threshold searches, and fusion-weight tuning for this implementation. It does **not** by itself prove that every possible image-grounded hierarchy is impossible. Determining the correct scope of inference is part of your task.

## Resource and Privacy Constraints

- Patient images, case-level predictions, dense feature arrays, and model weights stay on the server and are not in GitHub.
- The repository contains code, experiment definitions, aggregate reports, and non-patient metadata only.
- A complete 591-case C1 six-view dense-token feature bank is available on the server, so a new head-level experiment can reuse it.
- The server has one approximately 48 GB GPU, but disk capacity is tight.
- Do not design a plan that depends on reusing the consumed 108- or 162-case external labels.

## What We Need From You

### 1. Independent audit

Restate the scientific problem in your own words and identify any hidden assumptions, leakage risks, evaluation defects, or unsupported interpretations. Explicitly distinguish what is verified from what is inferred.

### 2. Causal diagnosis

Explain why the current models repeatedly trade B1 against B2 and why the sequential design improved low-risk subtypes while harming B2/B3/TC. Address at least these competing possibilities:

- inadequate sample size or subtype imbalance;
- source-specific acquisition shortcuts;
- insufficient visual information in one gross photograph;
- representation failure despite adequate information;
- label noise or genuine morphological continuity at the B1/B2 boundary;
- an architectural or optimization assumption not actually tested.

State which explanations are supported, contradicted, or unresolved by the evidence.

### 3. Broad hypothesis map, then a narrow execution plan

First map the remaining scientifically distinct directions. For each direction, state what new information or learning assumption it introduces and which prior negative result it escapes.

Then rank **at most two immediate primary experiments** on the existing 591 images. An experiment is eligible only if it is structurally different from another backbone/loss/threshold/sampler/fusion sweep.

For each selected experiment specify:

- exact image inputs and representation;
- model and loss;
- whether training is end-to-end or feature-based;
- nested five-fold and source-LODO procedure;
- leakage controls and matched baselines;
- primary and subtype/source metrics;
- expected mechanism of improvement;
- likely failure mode;
- a hard preregistered GO/NO-GO rule;
- estimated compute and storage cost.

### 4. Minimum new-data design if the current input is information-limited

If no existing-image experiment is compelling, say so directly. Propose the smallest scientifically adequate acquisition plan, including:

- standardized gross views per case, such as whole specimen, cut surface, capsule/interface, and close-up views;
- priority subtypes, especially B1/B2 and rare A/B3;
- target independent cases per subtype and per center;
- number and diversity of centers;
- acquisition standardization and quality-control fields;
- how to keep a genuinely untouched external test;
- whether the existing 589-case physician feature table can guide acquisition without becoming a model input.

### 5. Publication-level interpretation

State what can currently be claimed in a medicine-engineering paper, what cannot be claimed, and what minimum evidence would make the generalization story publishable.

## Required Response Format

Use these sections:

1. **Executive verdict**
2. **Verified facts vs. inferences**
3. **Failure-mechanism diagnosis**
4. **Remaining hypothesis map**
5. **Primary experiment 1**
6. **Primary experiment 2** (only if justified)
7. **Minimum new-data plan**
8. **Publication claim boundary**
9. **Immediate ordered action list**

Do not redirect the project toward release coverage, rejection optimization, manual-review workload, source-aware thresholding, confidence-only correction, or another physician annotation task. The priority is genuine image-grounded capability and cross-source generalization.

## Repository Reading Order

After reading this briefing, verify it against:

1. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
2. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H1_SEQUENTIAL_AB_TC_FALLBACK_RESULTS_20260713.md`
3. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`
4. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_WaveC_D1_E1_F1_F2_Results_20260712.md`
5. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_Visual_Capability_and_Genuine_Coarse_to_Fine_Reframing_20260712.md`
6. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/AI Pathology Model Improvement.md`
7. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/scripts/run_task7_sequential_ab_tc_fallback_20260713.py`
8. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/scripts/analyze_task7_sequential_ab_tc_fallback_20260713.py`

Challenge any mismatch you find between this briefing, the aggregate reports, and the executable code.
