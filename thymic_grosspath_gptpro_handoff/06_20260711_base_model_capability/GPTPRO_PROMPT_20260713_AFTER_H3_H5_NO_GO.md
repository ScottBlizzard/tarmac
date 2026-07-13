# Independent Expert Request: What Remains After H3-H5 Failed Cross-Domain Gates?

Repository: https://github.com/ScottBlizzard/tarmac

Please treat this as a completely new project. Do not assume any context outside
this prompt and the repository. Read the governing README and the H3-H5 reports
and code before proposing another experiment.

## Primary objective

Improve the intrinsic visual capability of a direct, image-only, 100%-coverage
classifier for thymic gross pathology. This is not a selective-release,
confidence-routing, rejection, or physician-review optimization problem.

Task7 is binary risk classification:

- low risk: A, AB, B1;
- high risk: B2, B3, thymic carcinoma (TC).

Multi-model systems are acceptable only when models learn complementary image
evidence. Probability correction, behavior-level stacking, confidence gates,
or threshold search do not count as stronger visual capability.

## Available internal development data

- 591 cases from one hospital and three acquisition batches:
  - batch1: 117;
  - batch2: 168;
  - third_batch: 306.
- Six subtypes:
  - A 44;
  - AB 262;
  - B1 62;
  - B2 89;
  - B3 24;
  - TC 110.
- One uniformly accessible selected photograph per case.
- Existing physician morphology table: 589/591 cases, explanatory audit only,
  not a model input.
- Evaluation:
  - locked five-fold OOF;
  - locked source-LODO, holding out each acquisition batch;
  - threshold 0.5;
  - 100% coverage;
  - paired source-by-risk-stratified bootstrap.

Source-LODO is only an internal batch-shift proxy. It is not multicenter
external validation.

## Consumed external data firewall

Historical 108-case and 162-case external cohorts have already been inspected.
They cannot be reused for training, model selection, thresholding, adaptation,
or another nominal blind test. A new external claim requires genuinely fresh,
label-blinded data.

## Corrected historical interpretation

The old 92% result was not a 92% visual classifier. It inherited output
stacking, true-label-derived difficulty leakage, and holdout reuse. The
leakage-clean visual baseline was around the mid-70% BAcc range. Do not use the
historical 92% magnitude as a current target or baseline.

## Locked references

### C2

Equal probability average of two fixed visual models, no rejection:

- five-fold OOF: BAcc 0.7514, AUC 0.8377, sensitivity 0.7175,
  specificity 0.7853;
- source-LODO: BAcc 0.7441, AUC 0.8108, sensitivity 0.7354,
  specificity 0.7527;
- LODO B1/B2 risk accuracy: 0.5000/0.6629;
- source BAcc: batch1 0.7890, batch2 0.7083, third_batch 0.7337.

### H3 PE-Spatial direct model

Frozen `facebook/PE-Spatial-L14-448`, native-aspect six views, up to 1,024
dense tokens per view, masked gated direct head:

- five-fold OOF: BAcc 0.8003, AUC 0.8700, sensitivity 0.8072,
  specificity 0.7935;
- source-LODO: BAcc 0.7539, AUC 0.7984, sensitivity 0.6816,
  specificity 0.8261;
- LODO B1/B2: 0.6452/0.5843;
- source BAcc: batch1 0.7434, batch2 0.7381, third_batch 0.7440;
- paired LODO BAcc delta versus C2: +0.0098, 95% CI
  [-0.0253, +0.0457].

This is the strongest mixed-source direct classifier, but it failed the
cross-domain gates through reduced high-risk sensitivity, B2 accuracy, and
batch1 transfer.

## Newly completed H4: quality-domain randomization

H4 used the exact frozen H3 PE bank. It trained on paired clean and realistic
degraded views using:

`0.5 * (CE_clean + CE_augmented) + 0.10 * JS(clean, augmented)`

Degradations included fixed-range downsampling, blur, JPEG, photometric, and
white-balance perturbations. Main validation and inference used clean images.

Results:

- five-fold OOF: BAcc 0.8009, AUC 0.8850, sensitivity 0.7758,
  specificity 0.8261;
- source-LODO: BAcc 0.7254, AUC 0.8089, sensitivity 0.6682,
  specificity 0.7826;
- LODO B1/B2: 0.6452/0.5281;
- source BAcc: batch1 0.7276, batch2 0.7500, third_batch 0.6823;
- H4 minus H3 PE LODO BAcc: -0.0285, 95% CI
  [-0.0510, -0.0064].

Decision: formal NO-GO. Do not search perturbation severity, consistency
weight, quality routing, threshold, seed, or source calibration.

## Newly completed H5: true second-order texture

H5 used a direct two-branch head on the frozen H3 PE tokens:

1. first-order masked gated branch;
2. trainable 1,024-to-64 projection, per-view 64 x 64 masked covariance,
   upper-triangle signed-square-root/L2 texture token, six-view texture pooling;
3. feature fusion before one low/high classifier.

It had 537,668 trainable parameters and no source, confidence, physician, or
quality inputs.

Results:

- five-fold OOF: BAcc 0.7969, AUC 0.8395, sensitivity 0.7623,
  specificity 0.8315;
- five-fold B1/B2: 0.6613/0.7079;
- source-LODO: BAcc 0.7422, AUC 0.8133, sensitivity 0.6502,
  specificity 0.8342;
- LODO B1/B2: 0.5968/0.4719;
- source BAcc: batch1 0.7355, batch2 0.7143, third_batch 0.7115;
- H5 minus C2 LODO BAcc: -0.0018, 95% CI
  [-0.0392, +0.0356];
- H5 minus C2 LODO sensitivity: -0.0852, 95% CI
  [-0.1435, -0.0269].

Decision: formal NO-GO. The apparent mixed-source B1/B2 improvement did not
transfer; B2 collapsed under source shift. Do not search covariance dimension,
normalization, fusion, loss, sampler, threshold, or seed.

## Multi-image availability audit

All 591 selected cached images exist. However:

- none of the 591 registered original `source_case_folder` paths is currently
  accessible on the server;
- among 285 cases with a recorded original image count, 268 have one image and
  17 have two;
- all 306 third-batch original image counts are missing.

A complete all-image case-bag experiment cannot start until original folders
are recovered or remounted. No patient image, case mapping, feature bank,
prediction row, or weight is stored in GitHub.

## Broad negative evidence already completed

Do not recommend a low-level repeat without explaining which failed assumption
changes. Completed negative or insufficient families include:

- DINOv2/v3, EVA02, ConvNeXt, Swin, ViTamin, AIMv2, SigLIP B/L/SO400M,
  SigLIP2 Base/Large/So400M, RADIOv2.5-L, MedSigLIP;
- frozen, LoRA, broader LoRA, supervised contrastive, VICReg, subtype and
  boundary auxiliary objectives;
- DANN, GroupDRO, REx, class-conditional alignment, MixStyle, SAM optimizer;
- mean, statistics, gated, spatial-pyramid, ROI, attention ROI, anatomy ROI,
  hierarchical MIL, cross-attention, random-bag consistency;
- sequential AB/TC/fallback specialists and other six-class/ordinal/boundary
  experts;
- generic preprocessing, automatic contrast/white balance/unsharp, quality
  consistency, and low-rank second-order covariance;
- probability-weight and threshold searches.

The repeated pattern is improved OOF/AUC/specificity or one boundary subtype,
followed by worse held-source sensitivity, B2, or another acquisition batch.

## Hard constraints

1. Do not reopen the consumed 108/162 external labels.
2. Do not use physician morphology fields as training inputs.
3. Do not optimize rejection, release fraction, or physician workload.
4. Do not propose threshold, confidence, probability-correction, or
   behavior-level stacking as visual improvement.
5. Do not sweep a closed H3-H5 family.
6. Do not rely on downloading new large checkpoints to the server; outbound
   server transfer is billable. State explicitly if a proposal requires a new
   model or data download.
7. Preserve fixed 0.5 threshold, 100% coverage, five-fold OOF, source-LODO,
   per-source metrics, B1/B2 metrics, and paired bootstrap.
8. Any current-data experiment must be preregistered as exploratory and must
   include a control that isolates its claimed new visual information.

## Exact questions

Please provide a critical decision, not a broad idea list.

1. Given the complete evidence, is there one scientifically defensible
   current-data experiment left that has a materially different mechanism and
   nontrivial expected information gain?
2. Specifically audit these remaining research ideas: unsupervised part
   discovery, selective high-resolution subtoken refinement, RGB plus
   frequency/wavelet auxiliary tokens, PE intermediate-layer extraction,
   SAM2/Hiera-SLCA, LP-FT to SAFT, and gross-pathology self-supervised
   pretraining. Identify which are already invalidated, conditionally valid, or
   blocked by missing data/checkpoints.
3. If exactly one experiment remains valid, give an executable preregistration:
   data firewall, fold-wise construction, architecture, fixed parameters,
   mandatory controls, storage estimate, expected runtime, success gates, and
   stopping rule. It must not use full-OOF results to select variants.
4. If no current-data experiment is defensible, say so directly. Specify the
   minimum new-data package needed to reopen model development: image views,
   per-case completeness, center count, acquisition metadata, approximate case
   count, subtype balance, and untouched external-test design.
5. Distinguish clearly between:
   - a real visual-capability hypothesis;
   - a useful engineering audit;
   - a methodologically invalid continuation;
   - work that is impossible until original files or new data are available.

The desired output is a single ranked decision with one primary next action,
not encouragement to run many variants.
