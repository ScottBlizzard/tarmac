# GPT Pro Request: Final Audit of the Cross-Representation Oracle Gap

Repository: https://github.com/ScottBlizzard/tarmac

Branch: `main`

Evidence commit at the time of this request:
`ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8`

Date: 2026-07-14

## Your role

Act as an independent senior medical-imaging machine-learning research lead,
methodologist, and experiment auditor. Assume you have never seen this project
before. Do not rely on prior chat context. Read the required repository files
before reaching a verdict.

This is a terminal decision audit, not an open-ended request for another list
of methods. You must choose between two outcomes:

1. identify exactly one scientifically defensible, image-grounded experiment
   that tests whether cross-representation complementarity can become a real
   full-coverage classifier improvement; or
2. conclude that the apparent oracle gap is not actionable with the existing
   fixed data and that current-cohort classifier development should stop.

Do not assume that an experiment must exist. Conversely, do not recommend
stopping without first auditing whether direct feature-level fusion across the
locked SigLIP/AIM and PE representation families was already tested under a
fully nested source-LODO protocol.

## Governing question

Six representative source-LODO image models have a non-deployable
"any-model-correct" oracle balanced accuracy of 0.9006, while ordinary
probability averaging reaches only 0.7602 and choosing the most confident
model reaches only 0.7462.

The exact question is:

> Does this oracle gap support one genuinely new, image-feature-level direct
> disease classifier, or does it merely show retrospective error diversity
> that no transferable selector can identify from the available photographs?

The project accepts multi-model systems only when every component obtains
disease-relevant evidence from the image. A system that mainly learns which
probability pattern is usually correct is not an improvement in visual
capability.

## Required reading, in order

Read these files before answering:

1. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/BASE_MODEL_CAPABILITY_PLATEAU_ROOT_CAUSE_ANALYSIS_20260714.md`
2. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
3. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/scripts/analyze_task7_capability_plateau_20260714.py`
4. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`
5. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_GPTPro_Plan_Execution_A1_B1_Results_20260712.md`
6. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_WaveC_D1_E1_F1_F2_Results_20260712.md`
7. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H1_SEQUENTIAL_AB_TC_FALLBACK_RESULTS_20260713.md`
8. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H2_CANONICAL_SPATIAL_RELATIONAL_RESULTS_20260713.md`
9. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H3_REPRESENTATION_RENEWAL_RESULTS_20260713.md`
10. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H4_QUALITY_DOMAIN_RANDOMIZATION_RESULTS_20260713.md`
11. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H5_SECOND_ORDER_TEXTURE_RESULTS_20260713.md`
12. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H6_NUISANCE_ANCHORED_CSD_RESULTS_20260714.md`
13. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H7_PE_EMBEDDING_LISA_RESULTS_20260714.md`
14. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/FREQUENCY_SOURCE_VS_RISK_AUDIT_RESULTS_20260713.md`
15. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/PE_PART_STABILITY_AUDIT_RESULTS_20260713.md`
16. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/MULTI_IMAGE_AVAILABILITY_AUDIT_20260713.md`
17. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/GPTPRO_RESPONSE_20260713_AFTER_H3_H5_NO_GO_BLANK.md`
18. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/GPTPRO_RESPONSE_20260714_FIXED_DATA_LITERATURE_SEARCH_BLANK.md`
19. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/AI Pathology Model Improvement.md`, especially section 9.

Inspect the relevant implementation scripts whenever a report claims that a
fusion, specialist, router, or cross-representation mechanism was tested. Do
not infer coverage from experiment names alone.

## Clinical task

Input: one previously selected gross photograph per internal case.

Primary endpoint, Task7:

- low risk: WHO A, AB, B1;
- high risk: WHO B2, B3, thymic carcinoma (TC).

Required output:

- direct image-only low/high-risk prediction;
- 100% case coverage;
- fixed threshold 0.5 for the locked primary analysis;
- no rejection or physician-review deferral used to inflate the primary
  metric.

Selective review may remain a downstream safety layer, but it is not the
objective of this request.

## Fixed data boundary

Internal development cohort: 591 unique cases.

| Acquisition batch | A | AB | B1 | B2 | B3 | TC | Total | High-risk rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batch1 | 20 | 20 | 20 | 20 | 20 | 17 | 117 | 0.487 |
| batch2 | 24 | 30 | 30 | 40 | 4 | 40 | 168 | 0.500 |
| third batch | 0 | 212 | 12 | 29 | 0 | 53 | 306 | 0.268 |

Totals:

- low risk: 368;
- high risk: 223;
- B1: 62;
- B2: 89;
- B3: 24.

Source and risk are associated: uncorrected Cramer's V 0.234,
chi-square p = 9.48e-8.

Source and six-class subtype are strongly associated: uncorrected Cramer's V
0.455, chi-square p = 7.03e-47.

These are acquisition batches from one project, not independent hospitals.
Source-LODO is an internal acquisition-shift proxy, not multicenter external
validation.

Only 17 old-domain cases have a second image. The remaining 574/591 cases have
one image, and every third-batch case is single-image. Six deterministic crops
from one photograph are not six independent views.

No new hospital, patient, image view, label, unlabeled image pool, or physician
annotation will become available. Do not make new data collection the answer.

Previously inspected stress-test cohorts:

- historical strict external: 108 cases;
- newer external: 162 deduplicated cases.

Both have been consumed by earlier project decisions. They may be used only as
transparent retrospective stress tests after an internal decision. They must
not select a model, fusion, threshold, seed, or hyperparameter and cannot be
presented as fresh confirmation.

## Compute and transfer boundary

- Server GPU: NVIDIA RTX 4090.
- Downloading checkpoints, packages, or datasets is billable and forbidden.
- Small source-code uploads are allowed.
- Use only existing cached checkpoints, installed libraries, or lightweight
  modules trained from scratch.
- Images, feature arrays, predictions, weights, and case identifiers remain on
  the server and never enter GitHub.
- Avoid unnecessary reconstruction of multi-gigabyte feature banks.
- Read `reports/LOCAL_ASSET_AUDIT_AFTER_GPTPRO_20260713.md` before requiring an
  asset.

## Historical result correction

The historical 92% corrected workflow is not evidence of a 92% visual
classifier.

- An early visual baseline was approximately 0.765 BAcc on 285 old-data cases.
- Candidate 41 reached approximately 0.835 BAcc but stacked outputs from prior
  corrected systems rather than learning new image evidence.
- No.64 reached approximately 0.926 BAcc using a `difficulty` feature derived
  from correctness and true-class probabilities. That feature requires the
  true label and is direct leakage.
- `base162` consumed the nominal third-batch holdout during meta-stacking and
  reached approximately 0.923 BAcc on old data, then fell to approximately
  0.730 BAcc on the strict third-batch test and 0.622 on historical strict
  external data.

Therefore, do not use behavior-level correction, confidence gating, or the old
92% result as a precedent for the requested experiment.

## Current direct-model evidence

All numbers below use 100% coverage and threshold 0.5.

| Model | Representation or mechanism | Source-LODO BAcc | AUC | Sens | Spec | B1 | B2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | SigLIP-L 512, six-view dense tokens, gated pooling | 0.7397 | 0.8072 | 0.7130 | 0.7663 | 0.4194 | 0.6292 |
| C2 | fixed C1 + AIMv2 MixStyle probability average | 0.7441 | 0.8108 | 0.7354 | 0.7527 | 0.5000 | 0.6629 |
| H3 | PE-Spatial dense tokens | **0.7539** | 0.7984 | 0.6816 | 0.8261 | 0.6452 | 0.5843 |
| H5 | PE second-order texture | 0.7422 | 0.8133 | 0.6502 | 0.8342 | 0.5968 | 0.4719 |
| H6 | nuisance-anchored CSD on PE | 0.7454 | 0.8226 | 0.6592 | 0.8315 | 0.6774 | 0.4944 |
| H7 | PE embedding LISA | 0.7382 | **0.8352** | 0.6368 | 0.8397 | 0.6935 | 0.4831 |

H3 mixed-source five-fold BAcc is 0.8003, but its source-LODO BAcc is 0.7539.
Its B1 improvement relative to C2 is accompanied by a B2 decline. H5-H7
continue the same pattern: specificity and B1 can rise while sensitivity and
B2 fall.

## Completed method coverage

The project has already tested, with negative or insufficient results:

- DINOv2, DINOv3, EVA02, ConvNeXt-DINOv3, ViTamin, SigLIP-B/L/SO400M,
  SigLIP2 variants, MedSigLIP, C-RADIO, AIMv2, and PE-Spatial;
- frozen probes, last-block or broader tuning, conservative and expanded LoRA,
  VICReg initialization, and supervised contrastive learning;
- whole image, foreground crop, deterministic quadrants, native-resolution
  tiles, label-trained attention ROI, label-free anatomy ROI, random bags,
  spatial pyramids, and explicit grid relations;
- mean, gated, summary-statistic, fixed probability, and large internal
  equal-weight fusion searches;
- binary, six-class, ordinal, subtype auxiliary, concept auxiliary, prototype,
  retrieval, B1/B2 boundary, and high-risk sentinel objectives;
- class weighting, focal/asymmetric losses, hard negatives, source-balanced
  sampling, SAM, MixStyle, DANN, REx, GroupDRO, alignment, quality
  randomization, CSD, and LISA;
- AB/TC sequential experts, image-only cascade specialists, behavior fusion,
  MoE-like variants, confidence routing, and rejection workflows.

Do not propose a nearby repeat without identifying the exact assumption or
information path that changes.

## New case-level complementarity audit

The final audit aligns the locked C1, C2, H3, H5, H6, and H7 source-LODO
predictions case by case.

### Deployable and oracle combinations

| Model set and rule | BAcc | Sens | Spec |
| --- | ---: | ---: | ---: |
| all six, mean probability at 0.5 | 0.7602 | 0.6861 | 0.8342 |
| all six, majority vote | 0.7602 | 0.6861 | 0.8342 |
| all six, choose highest-confidence model | 0.7462 | 0.6637 | 0.8288 |
| all six, pooled oracle threshold on mean probability | 0.7656 | 0.6861 | 0.8451 |
| all six, any-model-correct oracle | **0.9006** | 0.8610 | 0.9402 |
| C2+H3+H5+H6+H7, any-model-correct oracle | 0.8925 | 0.8475 | 0.9375 |
| H3+H5+H6+H7, any-model-correct oracle | 0.8462 | 0.7713 | 0.9212 |

The fixed six-model mean improves BAcc over H3 by only 0.0063. A 20,000-draw
paired, risk-stratified bootstrap gives a 95% CI of [-0.0176, 0.0307]. The
mean has the same B1 accuracy as H3, 40/62, but worse B2 accuracy, 47/89 versus
52/89.

The high oracle is therefore not itself a deployable gain.

### Error dependence

- H6 versus H7 probability Pearson correlation: 0.926;
- H6 versus H7 prediction agreement: 0.915;
- H6 versus H7 error Jaccard: 0.695;
- H3 versus H6 error Jaccard: 0.655;
- H3 versus H7 error Jaccard: 0.627;
- C1 versus C2 probability Pearson correlation: 0.909;
- C1 versus C2 error Jaccard: 0.685.

The PE family is highly correlated. Most of the oracle expansion occurs when
representation families are combined, but even then the confidence of the
correct family is not reliably higher.

### Persistent errors

Across all six models:

- all six correct: 334/591;
- at least five models wrong: 83/591;
- all six wrong: 53/591.

| Subtype | n | At least 5/6 wrong | All 6 wrong |
| --- | ---: | ---: | ---: |
| A | 44 | 5 | 3 |
| AB | 262 | 17 | 8 |
| B1 | 62 | **15** | **11** |
| B2 | 89 | **25** | **19** |
| B3 | 24 | 5 | 2 |
| TC | 110 | 16 | 10 |

### Source-dependent boundary failures

B2 accuracy:

| Source | n | C1 | C2 | H3 | H5 | H6 | H7 | Six-model mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batch1 | 20 | 0.95 | 0.95 | 0.65 | 0.70 | 0.50 | 0.55 | 0.65 |
| batch2 | 40 | 0.55 | 0.55 | 0.65 | 0.60 | 0.675 | 0.60 | 0.625 |
| third batch | 29 | 0.517 | 0.621 | 0.448 | 0.138 | 0.241 | 0.276 | 0.310 |

B1 accuracy:

| Source | n | C1 | C2 | H3 | H5 | H6 | H7 | Six-model mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batch1 | 20 | 0.20 | 0.50 | 0.55 | 0.55 | 0.60 | 0.65 | 0.50 |
| batch2 | 30 | 0.567 | 0.567 | 0.733 | 0.567 | 0.667 | 0.70 | 0.70 |
| third batch | 12 | 0.417 | 0.333 | 0.583 | 0.75 | 0.833 | 0.75 | 0.75 |

The same subtype is not equally difficult across acquisition batches. A
feature-level fusion could therefore improve average performance by learning
source style instead of invariant disease evidence. Your audit must directly
address this risk.

### Threshold diagnostic

Even after retrospectively choosing the pooled threshold on the same 591
source-LODO cases, the best individual result is H3 at 0.7730 BAcc. Thresholds
selected on the other two sources and transferred to the held source yield a
combined H3 BAcc of 0.7613. Calibration is not the primary bottleneck.

### Retrospective gross-description concepts

The parsed gross-description concept table covers 589/591 cases. After Fisher
tests and BH correction, no concept is significantly associated with being
wrong in at least five of six models. The concepts may explain hypotheses but
do not currently define a stable router. They are not deployment-time image
evidence and must not be used as fusion inputs.

## Primary audit task: was the relevant mechanism already tested?

Before proposing an experiment, audit the repository and answer each question
with file and line references:

1. Were locked C1 SigLIP-L case embeddings and H3 PE-Spatial case embeddings
   ever concatenated or jointly attended by a low-capacity head trained
   directly on the Task7 disease label?
2. Was such a cross-family model evaluated with outer source-LODO, with all
   training, normalization, checkpoint selection, and architecture selection
   restricted to the other sources?
3. Did any earlier `fusion` experiment combine image feature tensors, or did it
   only combine probabilities, logits, predictions, confidence, or behavior
   features?
4. Did an earlier MoE or cascade give each expert genuinely different image
   representations, or did it reuse one representation and change only the
   head or routed population?
5. Is the apparent all-six oracle mostly caused by unstable thresholds and
   source-specific errors, or is there evidence of conditional feature
   complementarity that a direct disease head could learn?
6. Are the required frozen feature banks still available, or can they be
   regenerated from cached checkpoints without a forbidden download and
   without exceeding the storage boundary?

Do not call an experiment new until this audit is complete.

## Forbidden model inputs and procedures

The final classifier, router, gate, attention module, or fusion head must not
receive any of the following:

- base-model probabilities, logits, margins, confidence, entropy, predicted
  labels, disagreement, fold variance, or correctness;
- any `difficulty` or reliability target derived from the true label;
- source, batch, hospital, domain, split membership, image count, or
  third-batch flags;
- retrospective gross-description text or parsed physician concepts;
- external-cohort labels or features;
- test-source normalization, calibration, threshold fitting, adaptation, or
  transductive preprocessing;
- a rejection option in the primary endpoint.

Do not propose:

- a learned selector whose target is "which model was correct";
- stacking or calibration on the six source-LODO output columns;
- another probability weight, threshold, seed, augmentation-strength, or
  fusion-subset search;
- a large architecture search over multiple fusion mechanisms;
- another nearby PE pooling, covariance, CSD, LISA, MixStyle, or DG variant;
- synthetic images whose pathology preservation cannot be verified;
- downloading a new checkpoint or dataset;
- using the consumed external cohorts for selection.

## What may qualify as image-grounded cross-representation fusion

An experiment is potentially eligible only if all learned inputs are direct
image-derived tensors from frozen or training-fold-fitted image encoders and
the learned output is supervised directly by the Task7 disease label.

Examples of the relevant class of mechanism include a tightly constrained
low-capacity disease head over aligned frozen C1 and H3 case embeddings, or a
single fixed token-level cross-family fusion whose only objective is Task7.
These examples are not recommendations. You must determine whether they are
new, identifiable, technically feasible, and statistically defensible after
the repository audit.

An image-only gate is not automatically valid. If it can solve the task mainly
by recognizing acquisition batch, it has not converted the oracle gap into
invariant disease evidence. Any proposed mechanism must include a control that
distinguishes direct complementary morphology from source recognition.

## Decision branch A: requirements for exactly one experiment

If and only if one experiment remains defensible, provide a complete locked
preregistration. It must include all of the following.

### A1. Changed assumption

State in one paragraph:

- what new information path or learning mechanism is tested;
- why C1/C2/H3-H7, F2, the internal equal-weight fusion searches, the B1
  cascade, and prior MoE variants did not already test it;
- why a positive result would represent direct image capability rather than
  source-conditioned behavior correction;
- what exact negative result would close the hypothesis.

### A2. Immutable model specification

Specify exactly:

- input views and resolution;
- frozen feature tensors and their dimensions;
- whether each encoder is frozen;
- pooling, normalization, projection, fusion, and classifier equations;
- all trainable layers and total trainable parameter count;
- initialization;
- loss and class balancing;
- optimizer, learning rate, weight decay, batch size, epochs, patience, and
  seed;
- one immutable configuration, not a grid.

If any choice cannot be fixed from existing evidence, count that uncertainty
against running the experiment rather than opening a search.

### A3. Leakage-safe source-LODO

Source-LODO must run first and is the primary decision point.

For each held acquisition source:

- exclude that source from all supervised training;
- exclude it from learned normalization, feature selection, checkpoint
  selection, architecture selection, loss weighting, and thresholding;
- perform any validation or early stopping only within the remaining sources;
- ensure case alignment and one prediction per patient;
- apply the frozen pipeline once to the held source;
- retain 100% coverage and threshold 0.5.

Label-free inference through an externally pretrained frozen encoder is
permitted, but no target-source statistic may change the learned decision
function.

Five-fold OOF may run only if the locked source-LODO advancement gates pass.
A confirmation seed may run only if both primary source-LODO and secondary
five-fold gates pass.

### A4. Required controls

At minimum include:

- locked C1 predictions on the exact same cases;
- locked C2 predictions on the exact same cases;
- locked H3 predictions on the exact same cases;
- each proposed representation branch with the same new head capacity;
- the exact proposed direct fusion;
- a control capable of showing whether the fusion gain is explained by source
  recognition rather than conditional disease evidence.

Do not use the any-model-correct oracle as a comparator for advancement.

### A5. Required metrics

Report:

- accuracy, balanced accuracy, AUC, sensitivity, specificity, TN, FP, FN, TP;
- B1 and B2 counts and accuracy;
- all six subtype counts and accuracy;
- each held-source BAcc, sensitivity, and specificity;
- source-by-subtype counts, especially third-batch B2;
- same-case rescue and harm relative to H3 and C2;
- paired bootstrap deltas with confidence intervals;
- minimum-source BAcc;
- trainable parameter count, runtime, and peak storage.

### A6. Advancement gates

Define exact numerical gates before any run. At minimum, the primary
source-LODO gate must require:

- BAcc improvement of at least 0.02 over H3;
- positive BAcc direction in at least two of three held sources;
- no held source BAcc decline greater than 0.02;
- no clinically material sensitivity decline relative to C2;
- no continuation of the pattern in which B1 improves while B2 materially
  declines;
- a prespecified improvement in third-batch B2 counts;
- paired uncertainty that does not support a trivial or purely unstable gain;
- evidence from the source-control ablation consistent with complementary
  disease evidence.

You may strengthen or refine these gates, but do not weaken them merely because
the cohort is small. Explain the exact B1, B2, and third-B2 count requirements.

### A7. Execution plan

Give:

- exact repository files to add or modify;
- exact server commands;
- exact immutable input paths;
- expected runtime and storage;
- recovery behavior after interruption;
- aggregate output files allowed into GitHub;
- case-level outputs that must remain server-only;
- the hard stopping rule.

Only one primary configuration is allowed. Do not include a backup method,
conditional family, or post-failure search.

## Decision branch B: requirements if no experiment remains

If the repository audit shows that the relevant mechanism was already tested,
is not identifiable from source confounding, lacks the required feature assets,
or cannot be evaluated independently after repeated reuse of the 591 cases,
state `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT` prominently.

Then provide:

1. the precise reason the 0.9006 oracle is not an actionable training target;
2. which existing experiment most closely tested the apparent opportunity;
3. why an image-feature fusion would still be expected to learn source or move
   B1/B2 errors;
4. the model or models that should remain frozen as the honest baseline;
5. a manuscript strategy for a medical-engineering journal under the fixed
   evidence ceiling;
6. claims that are supported, claims that must be weakened, and claims that
   must be abandoned;
7. whether the project should retain Task7, change the clinical endpoint, or
   present the work as a feasibility/domain-shift study;
8. whether the consumed 108- and 162-case cohorts should be reported as
   retrospective stress tests and exactly how they must be labeled;
9. the minimum analysis and figure package needed for a defensible paper;
10. a clear statement that no further seed, threshold, fusion, or architecture
    search should follow.

Do not substitute a generic recommendation to collect more data. New data are
not available.

## Server artifact map

Authoritative internal registry:

`/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv`

Locked source-LODO prediction files:

### C1

`/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711/oof_predictions.csv`

### C2

`/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv`

### H3

`/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo/oof_predictions.csv`

### H5

`/workspace/thymic_project/experiments/h5_second_order_texture_20260713/primary_seed20260713/source_lodo/oof_predictions.csv`

### H6

`/workspace/thymic_project/experiments/h6_nuisance_anchored_csd_20260714/source_lodo/nuisance_csd/oof_predictions.csv`

### H7

`/workspace/thymic_project/experiments/h7_pe_embedding_lisa_20260714/source_lodo/oof_predictions.csv`

Plateau-audit aggregate outputs:

`/workspace/thymic_project/experiments/task7_capability_plateau_audit_20260714`

Gross-description concept table:

`/workspace/thymic_project/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv`

The plateau audit initially referenced an H3 compact summary control under the
name C1. That error was detected and corrected before the final report. The
formal results in commit `ff81fe4` use the actual locked run 348 C1 file shown
above. Do not reintroduce the compact-control path.

## Required response format

Use this exact top-level structure:

1. **Executive verdict**
2. **Repository audit of cross-representation fusion coverage**
3. **What the 0.9006 oracle does and does not prove**
4. **Source-confounding and identifiability assessment**
5. **Decision: one locked experiment or stop**
6. **Complete branch-A preregistration or complete branch-B manuscript plan**
7. **Evidence and literature table**
8. **Exact next actions and hard stopping rule**

For every material repository claim, cite the exact file and line range. For
every method or medical-imaging claim, cite primary literature or official
documentation. Distinguish direct evidence, inference, and speculation.

Do not provide both a primary experiment and a menu of backups. Do not hedge
with "try several and see." The purpose of this request is to make one final,
auditable decision.
