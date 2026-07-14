# GPT Pro Request: Fixed-Data Literature Search and Executable Capability Plan

Repository: https://github.com/ScottBlizzard/tarmac

Date: 2026-07-14

## Your role

Act as a senior medical-imaging machine-learning research lead, literature
reviewer, and experiment designer. Assume you have never seen this project
before. Read the repository files listed below before recommending an
experiment.

This request changes one governing premise from the previous review:

> The available image cohorts are fixed. No new hospital, patient, view, label,
> or unlabeled gross-pathology image pool will become available for this
> project. We still must make the strongest technically honest attempt to
> improve the 100%-coverage image classifier with the data already present.

Do not make new data collection, multicenter recruitment, or stopping all model
development your primary answer. You may state the resulting evidence ceiling,
but you must still identify the best bounded engineering experiment that can be
run under the fixed-data constraint.

## Clinical task

Input: one gross photograph selected previously for each internal case.

Primary endpoint, Task7:

- low risk: WHO A, AB, B1;
- high risk: WHO B2, B3, thymic carcinoma (TC).

Required output:

- direct image-only low/high-risk prediction;
- 100% case coverage;
- fixed threshold 0.5;
- no physician-review rejection used to inflate the primary metric.

A multi-model or multi-stage solution is permitted only when its components
learn complementary visual evidence. Probability correction, confidence gates,
disagreement routing, and label-derived behavior features do not count as
improved visual capability.

## Non-negotiable data boundary

Internal development cohort: 591 unique cases.

- batch 1: 117;
- batch 2: 168;
- third batch: 306;
- low risk: 368;
- high risk: 223.

The three batches are acquisition sources from the available project data, not
independent multicenter confirmation. There are 608 accessible internal image
paths because only 17 old-data cases have a second image; 574/591 cases have
exactly one image. Broad multi-view learning is therefore not a realistic main
mechanism.

Previously inspected stress-test cohorts:

- historical strict external: 108 cases;
- newer external: 162 deduplicated cases.

Both external cohorts have already been inspected. They may be used only as
transparent retrospective stress tests after an internal source-LODO decision.
They cannot be presented as fresh, untouched confirmation and must not drive
model, threshold, fusion, or hyperparameter selection.

No new patient data can be acquired. Do not defer the answer to a future data
package.

## Compute and transfer boundary

- Server GPU: NVIDIA RTX 4090, approximately 48 GB VRAM.
- Root disk currently has approximately 41 GB free.
- Downloading new checkpoints, packages, or datasets is billable and forbidden.
- Small source-code uploads from the local repository are allowed.
- Use only existing local checkpoints/code, standard installed libraries, or a
  lightweight method trained from scratch.
- Avoid rebuilding multi-gigabyte dense banks when streaming or compact
  summaries are sufficient.
- Images, case-level predictions, feature arrays, and weights remain on the
  server and never enter GitHub.

Read `reports/LOCAL_ASSET_AUDIT_AFTER_GPTPRO_20260713.md` for the verified asset
boundary. In particular, PE-Spatial is cached locally; SAM2 code and compatible
weights are not.

## Current best direct-model capability

All numbers below use threshold 0.5 and 100% coverage.

### H3 PE-Spatial dense-token model

Mixed-source five-fold OOF:

- accuracy: 0.7986;
- balanced accuracy: 0.8003;
- AUC: 0.8700;
- sensitivity: 0.8072;
- specificity: 0.7935;
- B1 accuracy: 0.6774;
- B2 accuracy: 0.6966.

Internal source leave-one-domain-out (source-LODO):

- balanced accuracy: 0.7539;
- AUC: 0.7984;
- sensitivity: 0.6816;
- specificity: 0.8261;
- B1 accuracy: 0.6452;
- B2 accuracy: 0.5843;
- held-source BAcc: 0.7434 / 0.7381 / 0.7440.

H3 is the strongest mixed-source direct model, but it becomes conservative
under source shift and loses high-risk/B2 sensitivity.

### C2 locked visual reference

Mixed-source five-fold OOF:

- balanced accuracy: 0.7514;
- AUC: 0.8377;
- sensitivity: 0.7175;
- specificity: 0.7853.

Source-LODO:

- balanced accuracy: 0.7441;
- AUC: 0.8108;
- sensitivity: 0.7354;
- specificity: 0.7527;
- B1 accuracy: 0.5000;
- B2 accuracy: 0.6629.

C2 is weaker in mixed-source BAcc but more balanced for high-risk/B2 under
source shift.

### H5 second-order PE texture model

- five-fold BAcc/AUC: 0.7969/0.8395;
- source-LODO BAcc/AUC: 0.7422/0.8133;
- source-LODO sensitivity/specificity: 0.6502/0.8342;
- source-LODO B1/B2 accuracy: 0.5968/0.4719.

It improved mixed-source boundary behavior but failed under held-source shift.

## Newly completed acquisition-bias diagnostics

### Fixed Haar frequency audit

After training-fold residualization for risk, fixed frequency features predicted
the three acquisition sources:

- BAcc 0.4388 versus chance 0.3333;
- stratified permutation p=0.0099.

After training-fold residualization for source, risk prediction was weak:

- BAcc 0.5381 versus chance 0.5;
- permutation p=0.0792.

All 156/156 fixed frequency features had larger source partial eta-squared than
risk partial eta-squared. Median partial eta-squared was 0.05617 for source and
0.00037 for risk.

### Label-free PE part stability audit

Per-image three-cluster PE parts were stable and nondegenerate:

- median adjusted Rand: photometric 0.9708, blur/JPEG 0.9607, flip 0.9191;
- median normalized cluster entropy: 0.8258;
- median largest-cluster fraction: 0.6144;
- median cluster-versus-specimen-mask NMI: 0.0878.

However, stability was source-sensitive. Blur/JPEG token-cosine partial
eta-squared was 0.0846 for source and 0.0010 for risk. The audit did not justify
a current-cohort part classifier or new physician annotation burden.

These diagnostics indicate that the images contain real learnable local
structure, but acquisition nuisance is easier to learn than stable risk.

## Major completed negative or insufficient families

Do not recommend a renamed repeat without identifying the exact failed
assumption that changes. The repository has already tested or audited:

- DINOv2, DINOv3 variants, EVA02, AIMv2, ConvNeXt/ConvNeXtV2, SwinV2,
  ViTamin-L, SigLIP-B/L/SO400M, SigLIP2, RADIO, and PE-Spatial;
- pooled features, dense tokens, gated pooling, spatial pyramids, true-grid
  relational models, hierarchical MIL, and global-to-local cross-attention;
- original-resolution tiles, fixed quadrants, random local bags, attention ROI,
  label-free specimen/anatomy ROI, and physician-ROI infrastructure;
- conservative and broader LoRA, source-balanced LoRA, LP-style initialization,
  frozen-head and partial-fine-tuning variants;
- VICReg initialization, supervised contrastive learning, cross-source
  positives, subtype positives, and B1/B2 hard negatives;
- DANN, GroupDRO, REx, class-conditional alignment, CORAL-like alignment,
  MixStyle variants, source balancing, and quality/domain randomization;
- automatic contrast, gray-world/white-balance, unsharp, blur, JPEG,
  photometric, and crop consistency interventions;
- six-class and subtype auxiliary learning, ordinal targets, concept auxiliary
  learning, boundary experts, prototype/retrieval models, high-risk sentinels,
  mixture-of-experts, and sequential AB/TC/fallback experts;
- covariance/second-order texture tokens and frequency/source diagnostics;
- fixed image-only ensembles, extensive internal probability fusion, confidence
  routing, thresholding, and corrected-system output stacking.

The old approximately 92% result is not a pure visual model. It came from
behavior-level correction/output stacking, included label-derived leakage in a
winning router, and transferred poorly. Do not use it as a baseline visual
capability claim.

## What we need you to do

### 1. Perform a real literature search

Search primary papers, official code repositories, and authoritative technical
documentation available through 2026-07-14. Focus on methods relevant to:

- small medical-imaging datasets;
- severe acquisition/source confounding;
- multi-source domain generalization with only three source groups;
- preserving class signal while suppressing source nuisance;
- foundation-model adaptation without new checkpoints;
- high-risk sensitivity and adjacent-class boundaries;
- rigorous model selection when no fresh cohort exists.

Return a literature table with at least 12 directly relevant primary sources.
For each source include:

- full method/paper name and year;
- DOI, arXiv, or official project URL;
- core mechanism;
- why it might change a failed assumption here;
- overlap with completed project work;
- required code/checkpoint/data assets;
- whether it is executable without a new download.

Do not pad the table with generic surveys or methods that clearly duplicate the
completed work.

### 2. Build an explicit elimination matrix

For every plausible method family found, classify it as:

- genuinely new mechanism;
- meaningful modification of a failed mechanism;
- already tested in substance;
- impossible under local assets;
- statistically unsuitable for 591 cases/three sources.

Explain the classification. This matrix must prevent another long sequence of
renamed repeats.

### 3. Rank at most three executable experiments

Rank no more than three experiments that are materially different from the
completed work. At least one must be executable now with cached assets. For each
candidate specify:

- changed causal or visual assumption;
- why current negative evidence does not already close it;
- exact input representation;
- model and optimization mechanism;
- expected effect on source nuisance, sensitivity, B1, and B2;
- compute, runtime, and storage estimate;
- local asset dependencies;
- most informative negative result;
- principal leakage or adaptive-overfitting risk.

Do not rank a method solely because it is recent or easy to run.

### 4. Select one primary experiment and make it executable

Provide one locked primary experiment, not a broad sweep. Include enough detail
for another coding agent to implement it without inventing choices:

- exact architecture and tensor flow;
- frozen/trainable components and parameter counts;
- losses with formulas and fixed weights;
- sampling and source handling;
- optimizer, learning rates, weight decay, schedule, epochs, batch size, AMP,
  gradient clipping, and seed;
- five-fold and source-LODO split handling;
- training-only preprocessing and nuisance estimation;
- checkpoint selection rule using training/validation data only;
- fixed threshold 0.5 and 100% coverage;
- required ablation/control, limited to what identifies the mechanism;
- server command plan, output tree, expected disk use, and recovery behavior;
- aggregate metrics and privacy boundary.

Use source-LODO as the primary capability test and ordinary five-fold OOF as a
secondary measure. Compare against both C2 and H3. The experiment should aim to
retain H3's B1/mixed-source gain while recovering C2's sensitivity and B2
behavior.

Propose locked, power-aware advancement gates. At minimum discuss the existing
reference values:

- source-LODO BAcc: H3 0.7539, C2 0.7441;
- sensitivity: H3 0.6816, C2 0.7354;
- specificity: H3 0.8261, C2 0.7527;
- B1: H3 0.6452, C2 0.5000;
- B2: H3 0.5843, C2 0.6629;
- held-source BAcc: H3 0.7434/0.7381/0.7440.

Do not use post-hoc threshold search, test-source calibration, confidence
rejection, or external labels to pass a gate.

### 5. Give one backup only

Provide one backup experiment that should run only if the primary fails for a
specific mechanistic reason. State the trigger. Do not provide a queue of minor
variants.

### 6. Be explicit about the evidence ceiling

Separate:

- an engineering improvement on fixed data;
- internal cross-batch robustness evidence;
- retrospective performance on consumed external cohorts;
- a claim that would require a fresh cohort and therefore cannot be made.

The lack of fresh data limits publication claims, but it is not permission to
avoid proposing the strongest executable current-data experiment.

## Answers that are not acceptable

Do not answer primarily with:

- collect more data;
- stop all model development;
- use selective prediction or send most cases to physicians;
- tune confidence thresholds or routing percentages;
- try more backbones, more seeds, more fusion weights, or more augmentation
  strengths;
- use SAM2/Hiera without first respecting the verified missing-asset boundary;
- generic DANN/GroupDRO/REx/MixStyle/LoRA/ROI/MIL advice without a changed
  mechanism and comparison to completed implementations;
- a large menu of ideas without one exact primary experiment.

If you conclude that no current-data experiment is scientifically confirmatory,
say so briefly, then still give the best bounded exploratory engineering
experiment and the exact limitations of its interpretation.

## Required reading order

Use the repository link above and read these files in order:

1. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
2. `.../reports/AI Pathology Model Improvement.md`
3. `.../reports/Task7_Base_Model_Capability_Experiments_20260711.md`
4. `.../reports/H3_REPRESENTATION_RENEWAL_RESULTS_20260713.md`
5. `.../reports/H4_QUALITY_DOMAIN_RANDOMIZATION_RESULTS_20260713.md`
6. `.../reports/H5_SECOND_ORDER_TEXTURE_RESULTS_20260713.md`
7. `.../reports/FREQUENCY_SOURCE_VS_RISK_AUDIT_RESULTS_20260713.md`
8. `.../reports/PE_PART_STABILITY_AUDIT_RESULTS_20260713.md`
9. `.../reports/MULTI_IMAGE_AVAILABILITY_AUDIT_20260713.md`
10. `.../reports/LOCAL_ASSET_AUDIT_AFTER_GPTPRO_20260713.md`
11. `.../reports/CURRENT_COHORT_MODEL_FREEZE_DECISION_20260713.md`
12. `.../GPTPRO_RESPONSE_20260713_AFTER_H3_H5_NO_GO_BLANK.md`

Treat the prior freeze document as the previous scientific decision, not as an
instruction to refuse this request. The operational premise is now fixed-data
exploratory improvement with strict source-LODO control.

Inspect relevant scripts and aggregate metrics when a proposed mechanism may
overlap existing work. Do not infer implementation details from filenames
alone.

## Required final structure

Return the answer in this order:

1. Fixed-data decision in five sentences or fewer.
2. Literature evidence table.
3. Method-family elimination matrix.
4. Ranked top three executable experiments.
5. Exact preregistration for the primary experiment.
6. One conditional backup experiment.
7. Advancement gates and stopping rule.
8. Evidence ceiling and manuscript-safe wording.
9. Missing repository fact, if any, that materially blocks implementation.

Be decisive. We need an executable research decision, not another brainstorm.
