I reviewed the specified reports and the key implementation chain: the early curriculum models, Candidate 41, the No.64 router/corrector, the old/third meta-stack, the `base162`/GrossPath-RC materialization, and the July 11 dense-token extraction/training/selection code. This is a static code-and-provenance audit; I did not rerun training because the image stores and cached model artifacts are not contained in the repository.

I use three evidence labels below:

* **Verified:** directly supported by code or retained result files.
* **Code-based inference:** a methodological consequence of the implementation.
* **Recommendation:** proposed next-step design.

# 1. Verdict on the 92% Historical Result

## Bottom-line verdict

**Yes: No.64 and `base162` should be reclassified as behavior-level meta-correction, not base visual capability.** More strongly, the reported 92.63% No.64 result is not a leakage-clean estimate of a prospective automatic reviewer, because the winning router receives a target-derived difficulty variable that cannot exist at deployment without knowing the true label.

The historical ladder should be described approximately as follows:

| Result               | Defensible classification                                                               |
| -------------------- | --------------------------------------------------------------------------------------- |
| Approximately 0.765  | Early image-grounded visual baseline                                                    |
| Candidate 41, 0.8351 | Selected ensemble of corrected model-output systems; not a clean direct visual ensemble |
| No.64, 0.9263        | Exploratory behavior-level router/corrector on the same 285 cases                       |
| `base162`, 0.9228    | Full-coverage behavior-level meta-stack inheriting the No.64/adaptation lineage         |
| C1/C2                | Current image-grounded candidates, not yet freshly externally confirmed                 |

The repository reports the 0.7649, 0.8351, 0.9263, and 0.9228 figures, and identifies the No.64 winner as `best41_wrong_model_visible_extra` plus `model_non_easy_extra`, with 61 routed cases, 26 apparent rescues, and zero apparent harms.

One additional naming correction is warranted. The May report describes the 0.7649 result as a fold-wise blend of Stage2 and Stage3 visual models, not unambiguously a single set of network weights. It is still an image-grounded result, but “early visual baseline” is safer than “single model.”

## Candidate 41 is also less “base visual” than its historical label suggests

Candidate 41’s winning `candidates_only` stack does not reread images. It consumes predictions, probabilities, confidence, vote disagreement, and dispersion from four previously selected corrected-candidate systems. The stacker itself evaluates multiple model families and feature sets and then sorts them on the same pooled OOF result.

The upstream corrected-candidate family routes cases using learned review/hardness scores and then applies fold-wise automatic correctors. It scans score definitions, target recalls, feature sets, model types, and training scopes.

Therefore, under the strict taxonomy below:

> Candidate 41 is an **image-derived behavior/output ensemble**, not a clean ensemble whose improvement can be attributed directly to complementary visual representations.

Its 0.8351 result may still be a useful historical system comparator. It should not be used as proof that visual representation learning reached 83.5%.

## Can any part of the 92% result support learned fine visual morphology?

**No part of the incremental gain from Candidate 41 to No.64 can defensibly support that claim.**

Reasons:

1. The winning No.64 corrector used the `model` feature set rather than DINO or raw-image features.
2. Its inputs are mainly model outputs, margins, predictions, review scores, image count, and selection behavior.
3. The winning router explicitly models Candidate 41 errors.
4. The image-embedding alternatives did not supply the winning gain.
5. There is no same-routed-case experiment showing an image-only second reader outperforming the retained stage-1 prediction.
6. There is no fresh external replication of the 26-rescue/zero-harm pattern.

The underlying visual models obviously contained image information. That establishes coarse image signal. It does **not** causally attribute the additional 26 corrected cases, or the approximately 9.1 percentage-point increase, to newly learned morphology.

The strongest defensible historical wording is:

> “On the 285-case old-data cohort, an extensively selected, cross-fitted model-behavior reviewer achieved an exploratory OOF balanced accuracy of 0.926. The pipeline included target-derived routing information and was not evaluated with full pipeline-level nesting; therefore the magnitude is likely optimistic and is not evidence of a 92% visual classifier.”

For `base162`, the code materializes `0.2 × run104 + 0.8 × run108` and a 0.595 decision threshold.  Its sharp degradation on third-batch and external subgroups is consistent with source-specific behavioral signatures, although that mechanism is an inference rather than directly proven.

# 2. Operational Taxonomy of Acceptable and Unacceptable Multi-Model Systems

The decisive tests are not the number of networks or whether every case receives an output. They are:

1. **What information enters the diagnostic branch?**
2. **Does a later stage reread pixels or genuinely underused image regions?**
3. **Is it trained to diagnose the disease, or to predict and flip another model’s errors?**
4. **Is the full pipeline nested and evaluated at 100% coverage?**
5. **Can the gain be reproduced under source shift and fresh external testing?**

| Category                                  | Operational definition                                                                                                                                                                                                   | Required proof                                                                                                  | Reporting location                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Single visual classifier**              | One image architecture receives only current-case images or deterministic image-derived tensors and directly predicts Task7.                                                                                             | Patient-level OOF, source-LODO, fixed threshold, fresh external.                                                | Main full-coverage visual table.                |
| **Same-architecture CV ensemble**         | Several fold or seed instances of one visual architecture are averaged.                                                                                                                                                  | Same as above; must be named as an ensemble, not a single network.                                              | Main visual table.                              |
| **Legitimate image-grounded ensemble**    | Every member independently reads image evidence and predicts disease; fusion is fixed in advance or selected entirely inside the training partition.                                                                     | Member ablations, paired gain, nested fusion selection, source-LODO and fresh external.                         | Main visual table.                              |
| **Genuine coarse-to-fine visual cascade** | Stage 2 reads raw pixels, smaller native-resolution tiles, an automatically selected tumor/cut-surface ROI, or an additional original view that stage 1 did not adequately process. Stage 2 has a direct disease target. | Same-routed M2/M3 experiment, positive image-only conditional gain, source-LODO and fresh external replication. | Main visual table if final coverage is 100%.    |
| **Behavior-only meta-corrector**          | Diagnostic output is produced mainly from probabilities, predicted labels, margins, disagreement, error scores, source, image count, or routing behavior.                                                                | May be valid as an operational system, but cannot establish visual capability.                                  | Separate post-processing/meta-correction table. |
| **Selective/rejection workflow**          | Some cases are withheld, rejected, sent for repeat imaging, or referred to a person.                                                                                                                                     | Risk-coverage curve, review burden, residual FN/FP, and clinical workflow analysis.                             | Safety/workflow table only.                     |

## Important boundary conditions

A later network is **not** coarse-to-fine merely because it is called “stage 2.”

It does not qualify if it:

* repools the same C1 cached tokens;
* consumes C1 probabilities plus another confidence score;
* uses a second classifier on the same globally resized six views without demonstrating new image information;
* predicts “stage 1 wrong” and flips the answer;
* uses a manually supplied ROI at test time without disclosing human assistance.

A second stage using the same original photograph can qualify only when it accesses materially different image information—for example, 20–30% specimen-relative crops extracted from the original pixels, rather than C1’s large overlapping 70% quadrants.

C1 itself should be labeled precisely. Its lock file describes a fivefold-head ensemble probability rather than one deployed set of weights. C2 is a fixed equal average of two image-model families. Both are acceptable in the main visual table because their diagnostic members are image-grounded and their fusion is fixed; neither should be called a literal single network.

## Required table separation

### Main full-coverage visual table

Include only:

* C1;
* C2;
* new direct visual classifiers;
* genuine full-coverage visual cascades that pass the M0–M4 test.

Required columns should include:

* exact image inputs;
* number of original images;
* number and scale of derived views;
* whether deployment averages folds/seeds;
* BAcc, AUC, sensitivity, specificity, FN, FP;
* B1 accuracy and B2 accuracy;
* minimum-source BAcc;
* coverage, fixed at 100%;
* internal OOF, source-LODO, and fresh external results.

### Separate meta/safety tables

Report separately:

* Candidate 41;
* No.64;
* `base162`;
* calibration and confidence correction;
* error prediction;
* rejection or referral;
* oracle routing;
* difficulty-stratified or retrospectively selected subsets.

A full-coverage output does not automatically make a behavior meta-corrector a base visual model.

# 3. Leakage and Selection-Bias Audit

## 3.1 No.64: direct label-derived feature leakage

The most important verified finding is the use of `difficulty` inside the winning router’s supposedly model-visible feature matrix.

The curriculum report states that the difficulty strata were derived from prior models’ correctness, agreement, and true-class probability. They are therefore label-dependent retrospective descriptors, not prospective image features.

The route-score implementation then one-hot encodes both `selection_rule` and `difficulty` as inputs to `model_visible`.  The winning router was `best41_wrong_model_visible_extra`.

**Judgment:** this is direct target-derived feature leakage at the routing stage. OOF fitting of the router does not repair the fact that a test patient’s difficulty category itself was constructed using that patient’s true label and model correctness.

The clean rerun must:

* completely remove `difficulty`, `difficulty_fine`, `hard_core`, true-class probabilities, and any derivatives from inference features;
* permit them only as training labels or analysis strata;
* recompute any training-only stratum inside the outer training partition.

## 3.2 What No.64 did correctly

Several local steps are legitimate cross-fitting:

* router and corrector predictions for a held-out fold are generated by models that do not directly fit that fold;
* the corrector threshold is estimated by an inner cross-fitting procedure;
* the route budget for one outer fold is selected using the other folds rather than that outer fold’s outcomes.

These precautions are real and should not be dismissed.

They are nevertheless insufficient because the **entire stacked pipeline** is not nested.

## 3.3 Non-nested generation of upstream OOF features

The easiest way to see the problem is with a fold example.

Suppose fold 1 is the final No.64 test fold. A training case in fold 2 has an upstream OOF prediction generated by a model trained on folds 1, 3, 4, and 5. Thus that training-case feature already depends on fold 1 data. No.64 then uses such features from fold 2 to select the router, corrector, and route budget applied to fold 1.

The fold-1 patient was excluded from its own upstream prediction, but fold 1 was **not isolated from the training features used to construct the final fold-1 pipeline**.

This affects:

* Candidate 41 training features;
* router OOF scores;
* corrector OOF scores;
* threshold/budget selection over those scores;
* target-derived difficulty strata generated before the final outer loop.

This is not literal same-row train/test contamination, but it is non-nested stacking. The remedy is an outermost loop inside which **all** lower-level OOF predictions are regenerated using only the outer training patients.

## 3.4 Target-derived corrector scope

No.64’s corrector search includes scopes such as `non_easy`, defined using `difficulty_fine`.

Using labels to select training examples is not intrinsically invalid. It becomes invalid for unbiased evaluation when the strata were globally generated and not recomputed inside the outer training partition.

Therefore:

* `difficulty_fine` as an inference feature is leakage;
* `difficulty_fine` as a globally fixed training scope is non-nested target-conditioned design;
* a training-only hard-mining variable could be acceptable if regenerated exclusively inside each outer training fold.

## 3.5 No.64 model-selection optimism

The No.64 implementation evaluates roughly:

* 60 corrector configurations;
* 4 route scores per corrector;
* approximately 12 route budgets;
* plus upstream router-target, feature, and model searches.

That is about 2,880 No.64 route/corrector/budget combinations before accounting for upstream Candidate 41 and router searches. The final winner is then selected by pooled OOF performance on the same 285 patients.

The 26-rescue/zero-harm result is therefore a selected extreme, not a prespecified estimate. Zero observed harm after this search should be regarded as a strong winner’s-curse warning.

### Clean interpretation

* The per-case cross-fitted predictions show that behavior-based correction may contain signal.
* The selected 0.9263 magnitude is not an unbiased estimate.
* A fully nested reconstruction is needed to estimate how much signal remains.
* Even a successful nested reconstruction would remain a behavior meta-corrector, not a visual cascade.

## 3.6 Candidate 41 audit

Candidate 41 has two different validity levels:

**Locally valid:** the final logistic or tree stacker uses fold-excluded fitting for each test fold.

**Not fully nested:** the four candidate outputs are themselves selected OOF corrected systems. Training-case candidate predictions can have been generated by models that included the stacker’s outer test fold. Candidate family and stacker family are then selected on the same 285-case OOF pool.

Its best retained result is a logistic model with `candidates_only`, BAcc 0.8348.

I would retain it as a historical behavior-system benchmark, but not as the reference “base image ensemble.”

## 3.7 `base162` audit

The old/third meta-stack has several additional sources of optimism.

### The nominal third-batch holdout was consumed

The implementation concatenates `adapt72` and `holdout234`, assigns new stratified folds across the entire third batch, and places all old and third cases into one development frame.

It then computes model selection scores from old and all-third OOF metrics and explicitly preserves an old-data 0.92 guard. The configuration, weight mode, objective, and threshold are selected from the same pooled OOF predictions.

**Code-based inference:** for this meta-stack stage, `holdout234` ceased to be a strict holdout. Its subgroup metric is an internal OOF subgroup result, not independent validation.

### Threshold-selection bias

For every candidate meta-model, `choose_threshold` searches thresholds on the full OOF vector and then reports performance on that same OOF vector.

That is optimistic threshold evaluation. It is different from C1/C2’s fixed 0.5 evaluation.

### Source and behavior features

The probability feature set includes:

* selected base probability, label and routed status;
* old/adaptation candidate probabilities;
* margins and logits;
* candidate means, ranges, and differences;
* optional extra OOF probabilities;
* domain and `third_split` dummy variables.

These are useful adaptation signals, but they make the system especially capable of learning source-specific calibration and error signatures.

### Transductive preprocessing in DINO branches

When DINO branches are enabled, scaling and PCA are fitted on the complete combined feature matrix before CV.  This is unsupervised rather than label leakage, but it is still transductive preprocessing and should be fitted inside each training fold.

### Final blend and external transfer

The final `base162` probability is the fixed 0.8/0.2 run108/run104 blend, with threshold 0.595.  Its poor third-batch and external behavior is consistent with the hypothesis that the meta-stack learned domain-conditioned error signatures rather than invariant fine morphology. The external failure supports that interpretation but does not uniquely prove the mechanism.

## 3.8 Current C1/C2 audit

The current dense-token implementation is substantially cleaner:

* one feature-bank row per case;
* deterministic views;
* locked fold correction;
* separate training, validation, and test partitions;
* early stopping on validation;
* fixed 0.5 test threshold;
* explicit source-LODO that excludes the held source from training.

I found no analogous direct label-derived inference feature in the C1 path.

Its remaining uncertainty is **selection**, not obvious direct leakage:

* many backbones, pooling heads, losses, view systems, and fusions were explored;
* OOF and source-LODO have both become development criteria;
* the exhaustive equal-fusion search was highly unstable under meta-cross-selection.

Therefore C1/C2 are valid locked image-grounded candidates, but their internal metrics remain selected estimates. The fresh external test is essential.

# 4. Route A: Direct Visual Model Plan

## A1. Native-detail hierarchical direct classifier

### Changed assumption

C1’s bottleneck may not be the backbone. It may be the effective visual scale and the loss of explicit global-local relationships.

C1’s four quadrant crops each cover approximately 70% of the specimen and overlap heavily. Every view is resized to 512 pixels.   Its `gated` head then flattens tokens from all views into one bag, using a view embedding but no explicit original-image coordinate or scale relationship.

This means C1 is multiview, but not yet a true native-detail hierarchy.

### Exact input information

Use the one predeclared primary original image and extract, from original pixels:

* one 512-pixel whole-image view;
* one 512-pixel specimen-box view;
* four medium tiles, each covering 50% of the specimen box;
* eight local tiles, each covering 25% of the specimen box;
* normalized tile center coordinates, tile scale, and tissue coverage.

Candidate tiles should be generated by a fixed overlapping grid. Keep tiles with at least 60% specimen tissue, then select a fixed number by deterministic spatial coverage—not by label, model error, confidence, or external performance.

This is materially different from repeating C1’s 70% quadrants. A 25% specimen tile provides approximately 2.8 times the linear magnification of a 70% quadrant before the crop is resized to 512.

Do not input:

* source;
* batch;
* image count;
* selection rule;
* stage-1 probability;
* confidence;
* difficulty.

### Architecture and training target

First-pass architecture:

1. Frozen C1 SigLIP-L encoder shared across all views.
2. Per-view dense-token gated pooling to a 256-dimensional tile representation.
3. Add learned scale and normalized coordinate embeddings.
4. Two-layer, four-head global-to-local cross-attention, with the global specimen token as context.
5. Direct binary Task7 head.

Primary loss: direct binary cross-entropy with a fixed source×risk balanced sampler.

Do not initially add subtype, ordinal, concept, error-prediction, or confidence losses. Previous B1/B2 experts, ordinal losses, heavy auxiliary supervision, DANN, and simple ROI variants did not create a robust breakthrough.

### Patient-level splitting and nested selection

Group all images by `original_case_id`.

For each outer fold:

1. Hold out the complete patient fold.
2. Within the outer training patients, run threefold inner CV.
3. Compare at most two prespecified variants:

   * hierarchical gated MIL;
   * two-layer global-to-local cross-attention.
4. Select architecture and median best epoch using inner BAcc at threshold 0.5.
5. Retrain on the entire outer training set for the selected epoch count.
6. Evaluate the outer test fold once.

The only publishable “selected pipeline” OOF estimate should be produced by this outer loop. Fixed-family OOF estimates can also be reported, but not substituted after seeing which is highest.

### Source-LODO

For each held source:

* exclude the source from encoder-head training, tile-selection learning, normalization, epoch selection, and architecture selection;
* perform inner CV using only the remaining sources;
* apply the complete fixed pipeline once to the held source.

Because the encoder is frozen and externally pretrained, its original pretraining is acceptable. Any data-derived tile selector, normalizer, ROI detector, or classifier must be trained only on the remaining sources.

### Minimum ablations

The minimum causal ablation is:

| Ablation                                          | Purpose                                                 |
| ------------------------------------------------- | ------------------------------------------------------- |
| C1 six views with the new aggregator              | Tests architecture without new information              |
| Native tiles with simple gated MIL                | Tests new scale without cross-attention                 |
| Native tiles plus coordinates and cross-attention | Full A1                                                 |
| Same number of matched random tissue tiles        | Tests whether any extra crop count suffices             |
| Local tiles without the global specimen token     | Tests whether global context prevents texture shortcuts |

A gain that appears with the new aggregator on C1’s old views is an architecture gain, not evidence that high-resolution information mattered. A gain that also appears with random tiles is not evidence of learned ROI selection.

### Compute-conscious pilot

* Generate the native tile bank once with the frozen C1 encoder.
* Run one engineering fold only to verify memory and convergence; do not tune scientific parameters from it.
* Run one full fivefold A1-MIL model.
* Only if fivefold point ΔBAcc versus C1 is at least +0.02 and no fold deteriorates by more than 0.03, run cross-attention, source-LODO, and a second seed.
* Do not start LoRA or full encoder fine-tuning until the frozen-feature experiment demonstrates new-image-information gain.

### Success and stopping

Advance A1 only if all are met:

* OOF ΔBAcc versus C1 at least +0.03;
* paired patient bootstrap lower 95% bound above zero;
* source-LODO ΔBAcc at least +0.02;
* positive LODO difference in at least two of three sources;
* no source BAcc falls by more than 0.02;
* high-risk sensitivity is no worse than C1 by more than 0.02;
* neither B1 nor B2 deteriorates by more than 0.05;
* random-tile control does not explain the gain.

Stop the family after the two prespecified aggregation variants if it fails. Do not open a grid over tile count, tile scale, attention depth, losses, and thresholds.

## A2. Physician-anchored cut-surface/ROI direct classifier

### Changed assumption

The discriminative information may be concentrated in the cut surface, tumor-capsule interface, or a small heterogeneous region that deterministic specimen grids cannot identify.

This is a changed information source because the project’s prior “ROI” work was predominantly specimen crop, tight crop, masking, color adjustment, or fixed quadrants—not physician-validated tumor/cut-surface localization. The previous ROI4 and five-view experiments produced modest or unstable gains rather than a transferable breakthrough.

### Annotation pilot

Select 120 patients:

* approximately 20 per subtype A, AB, B1, B2, B3, and carcinoma;
* balanced across the three internal sources;
* include both C1-correct and C1-incorrect cases within each subtype where available.

Two gross-pathology physicians, blinded to:

* model outputs;
* Task7 label;
* final histologic subtype,

should independently mark:

* specimen extent;
* cut surface;
* probable tumor;
* capsule/interface;
* hemorrhage/necrosis/artifact;
* up to three regions they would inspect at higher magnification;
* “no visually diagnostic ROI” when applicable.

This prevents the ROI itself from becoming a retrospective label annotation.

### Architecture and targets

First evaluate a **manual-ROI oracle** using held-out physician ROIs:

* global specimen view;
* physician ROI at two scales;
* same low-capacity hierarchical visual head as A1;
* direct Task7 target.

Manual test-time ROI results are an upper-bound analysis only and must not enter the main automatic model table.

Only if the oracle is positive should an automatic ROI detector be trained. The detector target should be anatomical/visual ROI, not Task7 error or risk label. The diagnostic network then rereads the detector-selected raw pixels.

### Go/no-go

Proceed to automatic ROI only if:

* manual-ROI image model produces routed conditional ΔBAcc of at least +0.08 versus C1;
* B1 and B2 each have nonnegative net rescue;
* physicians show meaningful region agreement, such as at least 70% top-region hit agreement;
* the gain is not reproduced by matched random ROIs.

The automatic system must retain at least 70% of the manual-oracle gain and provide a deterministic specimen-grid fallback when the detector fails.

If the manual ROI oracle is negative, stop ROI-model development. That result would suggest the limitation is image information content rather than localization.

## A3. Standardized original multiview direct model

This is the strongest changed-information route but requires new data.

Collect three predeclared original views:

1. whole specimen;
2. close cut-surface view;
3. capsule/interface or most suspicious region view.

Use a set model with view-type and missing-view embeddings. Do not silently select the most favorable prediction.

The existing 17 dual-image internal cases are useful only for descriptive sensitivity analysis. Their two-image average appeared promising, but the sample is too small, single-source, and protocol-inconsistent for model development.

Do not train a current multi-image MIL model on those 17 cases.

# 5. Route B: Genuine Coarse-to-Fine Visual Cascade Plan

## B1. C1 followed by a native-detail image specialist

### Stage 1

Use locked C1 unchanged for the primary mechanism experiment.

C1 is preferable to C2 initially because it gives a clean single-architecture visual stage and avoids confounding the causal ablation with an already fused stage 1. C2 should be used once as a locked sensitivity analysis after stage-2 visual gain is demonstrated.

C1’s main current weakness is the B1/B2 boundary: its locked OOF B1 and B2 accuracies are approximately 0.44 and 0.64, and its source-LODO B1 and B2 accuracies approximately 0.42 and 0.63.

### Router

Use a single prespecified behavior router for the first experiment:

> Route the 40% of cases with the smallest absolute C1 logit margin, using the 40th-percentile threshold estimated on the outer training patients.

This is deliberately simple. It avoids another learned error-router search and makes the stage-2 evidence test interpretable.

A fixed 30% route can be a predeclared secondary sensitivity analysis, but it must not replace 40% after results are seen.

### Stage-2 input

Stage 2 receives:

* the original primary image;
* global specimen view;
* four 50% specimen-relative tiles;
* eight 25% specimen-relative tiles;
* coordinates and scale.

M2 must not receive C1 probability, confidence, fold variance, route score, source, or difficulty.

### Architecture and target

Use the A1 image reader:

* frozen SigLIP-L initially;
* hierarchical gated MIL as primary;
* global-to-local cross-attention as the one architecture ablation;
* direct Task7 disease target.

Do not train it to predict “C1 wrong” or “flip C1.”

### Training population

Do not train only on observed C1 errors.

Within each outer training fold, generate C1 predictions by inner cross-fitting and construct training batches containing:

* 50% routed cases;
* 25% matched controls with similar C1 margin, source, and risk class;
* 25% randomly sampled full-cohort controls.

Use subtype matching when possible so that the specialist does not learn “routed means B1/B2” or a changed class prior.

Observed stage-1 error can be used for analysis, but not as an inference input or sole training inclusion rule.

### Final full-coverage rule

* nonrouted: retain C1;
* routed: use M2 for the image-only cascade result;
* M3, a nested fusion of stage-2 image embedding and stage-1 outputs, is a secondary full-coverage result.

All patients receive one prediction.

### Success threshold

The cascade advances only if:

* routed conditional image-only ΔBAcc, M2 minus M0, is at least +0.06;
* the paired 95% interval excludes zero;
* raw rescues exceed harms;
* full-cohort cascade ΔBAcc is at least +0.02;
* source-LODO full-cohort ΔBAcc is at least +0.015;
* at least two held sources show positive net rescue;
* neither held source nor B1/B2 subgroup shows a material collapse;
* M2 and M3 outperform the behavior-only M1 control.

A 0.06 gain over 40% of patients corresponds roughly to a 0.024 full-cohort improvement before other interactions, which is large enough to matter.

### Stop condition

Stop this cascade family if:

* M2 does not beat M0 on the same routed patients;
* only M1 improves;
* M3 improves but M2 does not and M3 does not beat M1;
* random routing performs similarly;
* source-LODO reverses the gain.

In any of those cases, the result is behavior correction or generic second-model averaging, not demonstrated fine visual knowledge.

## B2. Automatic ROI stage-2 specialist

This is B1 with an automatically predicted physician-anchored ROI replacing or augmenting the deterministic local tiles.

Required evidence sequence:

1. manual ROI oracle is positive;
2. automatic ROI detector is trained entirely inside each outer training fold;
3. stage 2 rereads raw ROI pixels;
4. M2 image-only gain remains positive;
5. source-LODO and fresh external replicate.

If physician ROIs are supplied at deployment, report it as human-assisted image analysis, not an automatic visual cascade.

## B3. True new-view stage-2 specialist

The cleanest future cascade is:

* stage 1: primary whole/cut-surface image;
* stage 2: separately acquired close-up cut-surface and interface views.

This directly satisfies the “new visual evidence” criterion. It also avoids debating whether another crop of the same photograph is sufficiently novel.

It requires standardized prospective multiview acquisition and cannot be supported by the present 17 dual-image cases.

## Specific judgment of the provisional specialist

### Patch extraction and scale

Do not reuse only whole/crop/70%-quadrant views.

Use at least two local scales:

* medium: 45–55% of specimen width/height;
* local: 20–30%.

Extract from the original-resolution image and only then resize the tile for the encoder. Preserve coordinates and tile scale.

### Deterministic versus trainable ROI

Start deterministic.

A weakly supervised attention ROI trained on 591 labels can easily identify source, ruler, specimen preparation, hemorrhage, or camera signatures. A trainable ROI should be introduced only after a manual-ROI oracle demonstrates that a stable relevant region exists.

### Global/local context

Never make the specialist local-only. Gross features such as capsule, border, shape, attachment, and lesion-to-specimen relationship can be lost in small tiles.

Every local representation should be conditioned on:

* global specimen representation;
* tile coordinates;
* scale;
* foreground mask.

### MIL versus cross-attention

Use hierarchical gated MIL first because it has lower capacity and variance.

Run one cross-attention variant because the local-global relationship may be diagnostically important. Do not conduct a broad transformer-depth/head search.

Attention maps are not localization proof. They require random-tile controls and comparison with physician ROIs.

### Boundary-case training with matched controls

Enrich B1/B2 and routed cases, but always include:

* A/AB controls that look complex;
* B3/carcinoma controls that look deceptively homogeneous;
* matched source and image-quality controls;
* both stage-1-correct and stage-1-wrong examples.

Otherwise, the specialist can learn route membership or subtype prevalence rather than morphology.

### Auxiliary subtype supervision

The primary target remains binary risk.

A six-class auxiliary head can be introduced only after the image-information pilot is positive:

* fixed low weight, for example 0.1;
* no ordinal assumption;
* one prespecified ablation;
* no new search over subtype losses.

Prior boundary experts, subtype-derived risk, and ordinal formulations did not improve the mainline robustly.

### Shortcut prevention

Mandatory controls should include:

* background-only;
* border/ruler/text-only;
* specimen with background replaced by a constant;
* color-histogram-only probe;
* matched random tiles;
* source classifier trained on learned embeddings;
* performance by source, camera, background, fixation, and image adequacy;
* source-balanced minibatches;
* full source-LODO.

The previous background-only qkvb result was near random, which is reassuring for that model, but it does not rule out specimen-preparation or acquisition shortcuts in a new patch specialist.

### Handling the 17 multiview cases

* Keep the current mainline one-primary-image.
* Do not use the 17 cases to select view fusion or train a bag model.
* Retain them as a locked descriptive sensitivity analysis.
* Begin a standardized multiview collection prospectively.
* Split every future view by patient, never by image.

# 6. Same-Routed-Case Causal Ablation

## Fixed routed population

For every outer fold, define routing from inner-cross-fitted stage-1 outputs before any stage-2 result is examined.

All M0–M4 comparisons use the identical routed patients for that outer test fold, except M4’s matched-random route experiment.

## Model definitions

| Model  | Inputs                                                                                   | Purpose                                                           |
| ------ | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **M0** | Original C1 prediction                                                                   | Retained stage-1 comparator                                       |
| **M1** | C1 probability, logit, margin, predicted label, fold-head SD, view-level SD/disagreement | Behavior-only falsification control                               |
| **M2** | Stage-2 global plus native local image evidence only                                     | Tests whether stage 2 learned new visual evidence                 |
| **M3** | Stage-2 image embedding plus the M1 behavior vector                                      | Tests complementary conditional fusion                            |
| **M4** | M2 applied under matched random routing of equal size                                    | Tests whether the selected route enriches cases helped by stage 2 |

M1 should be a single prespecified regularized logistic model, not a model grid.

M3’s fusion head must be trained inside the outer training data. It cannot be fitted on pooled OOF predictions and then evaluated on those same predictions.

## Primary conditional estimand

For the routed cases, define:

[
\tau_{\text{visual}}
====================

\frac{1}{2}
\sum_{y \in {0,1}}
E\left[
I(\widehat{Y}_{M2}=Y)
---------------------

I(\widehat{Y}_{M0}=Y)
\mid R=1, Y=y
\right].
]

This is exactly:

> **conditional routed balanced accuracy of M2 minus conditional routed balanced accuracy of M0.**

It is the primary test of newly learned visual knowledge.

Secondary estimands:

[
\tau_{3-1}=BAcc_R(M3)-BAcc_R(M1)
]

and

[
\tau_{3-2}=BAcc_R(M3)-BAcc_R(M2).
]

The first tests whether image evidence adds information beyond behavior outputs. The second tests whether stage-1 outputs add useful context to the stage-2 image reader.

## Paired statistical analysis

Use:

1. **Patient-level paired bootstrap**, stratified by source, outer fold, and true risk class, for the primary BAcc difference and confidence interval.
2. **Paired permutation test**, swapping M0/M2 predictions within patients, for the primary null.
3. **Exact McNemar test** on rescue versus harm discordances overall.
4. Separate exact McNemar tables for low-risk and high-risk cases.
5. Source-specific effects without pooling away a negative source.

Do not use an unpaired test; all predictions concern the same patients.

## Rescue and harm accounting

For model (M_k):

* rescue: M0 wrong, (M_k) correct;
* harm: M0 correct, (M_k) wrong;
* unchanged correct;
* unchanged wrong.

Report:

* raw rescue and harm counts;
* net rescue;
* rescue-to-harm ratio;
* class-balanced net rescue;
* source-specific rescue and harm;
* B1 and B2 rescue and harm separately.

A high-risk rescue and a low-risk rescue should not be allowed to obscure substantial high-risk harm. Therefore retain the full FN/FP accounting.

## B1/B2 analysis

For the B1/B2 boundary subset report:

* B1 specificity, equivalently one minus B1 false-positive rate;
* B2 sensitivity;
* boundary balanced accuracy;
* B1 rescues and harms;
* B2 rescues and harms;
* exact confidence intervals.

Because current subgroup counts are small, avoid declaring significance from asymptotic subtype tests. Emphasize counts, direction, and replication under source-LODO.

Do not tune the route or decision threshold to improve B1/B2 after viewing these results.

## M4 matched random routing

Generate 1,000 random routes with the same size as the real routed set, matched on:

* source;
* C1 predicted class;
* C1 margin decile;
* image-quality bin.

Do not match on true label because a deployable router does not know it.

Apply the same fixed M2 predictions and compute (\tau_{\text{visual}}) and net rescue for each random route. The observed router should exceed at least the 95th percentile of the random-route distribution to support targeted complementarity.

## Evidence pattern required for a transferable visual specialist

The strong conclusion requires all of the following:

1. M2 beats M0 on the same routed cases.
2. M3 beats M1.
3. M2 has positive net rescue, not merely better AUC.
4. M1 alone cannot reproduce most of the gain.
5. Actual routing outperforms M4 matched random routing.
6. The direction is positive in at least two of three LODO sources.
7. No source has substantial negative net rescue.
8. B1 and B2 do not move in opposing clinically unacceptable directions.
9. The direction replicates in the fresh external cohort.

Interpretation of weaker patterns:

* **Only M1 improves:** behavior meta-correction.
* **M3 improves but M2 and M3−M1 do not:** probability-conditioned correction, not visual proof.
* **M2 improves but M4 is equal:** stage 2 may be a generally stronger direct model; use it as Route A rather than claiming a targeted cascade.
* **Internal M2 gain disappears in LODO:** likely source-specific visual or acquisition learning.
* **Fresh external failure:** no transferable fine-morphology claim.

# 7. Advancement, Stopping, and Fresh-External Rules

## 7.1 Internal advancement relative to C1/C2

### Direct model

Primary comparator: C1.

Secondary comparator: C2.

Advance only if:

* OOF ΔBAcc versus C1 ≥ +0.03;
* OOF paired 95% CI excludes zero;
* source-LODO ΔBAcc versus C1 ≥ +0.02;
* positive source effect in at least two of three sources;
* no source drops by more than 0.02;
* sensitivity and specificity each remain at least 0.70 internally;
* B1 and B2 are not both worse;
* AUC does not fall by more than 0.01;
* result replicates across a second fixed seed.

If the direct model is more than 0.01 worse than C2 in both OOF and LODO, it should not replace C2 even if it beats C1.

### Cascade

In addition to the full-cohort criteria:

* (\tau_{\text{visual}}) ≥ +0.06 on routed cases;
* conditional paired CI excludes zero;
* M3−M1 ≥ +0.03;
* actual routing beats matched random routing;
* full-cohort OOF ΔBAcc ≥ +0.02;
* full-cohort LODO ΔBAcc ≥ +0.015.

### Source minimum

C1’s locked LODO minimum-source BAcc is approximately 0.702. A new model should not advance with a minimum-source result below 0.68 unless the difference is explained by very small denominators and replicated elsewhere. C2’s locked minimum-source value is approximately 0.708.

## 7.2 Candidate-family budget

Permit one preregistered wave of at most **three changed-information families**, with no more than two implementation variants each:

1. deterministic native multiscale tiles;
2. physician-anchored automatic ROI;
3. true additional original views.

MIL versus cross-attention counts as two variants within family 1, not two new families.

Do not add ordinary backbone swaps, minor loss weights, or threshold variants to this budget.

After this wave:

* lock at most one new candidate;
* keep C1 and C2 as benchmarks;
* if no family passes, stop reusing the 591 patients for architecture search and prioritize new data.

## 7.3 Avoiding a new meta-selection layer

Use hard gates, not a weighted composite selection score.

A predeclared tie-break should be:

1. passes both OOF and LODO gates;
2. higher LODO BAcc;
3. higher minimum-source BAcc;
4. fewer learned components;
5. direct model preferred over cascade when performance differs by less than 0.01.

Do not:

* search fusion weights among the winners;
* average several near-winners;
* tune a route budget from pooled OOF;
* choose the best seed;
* choose a threshold other than the locked primary 0.5;
* use C1/C2 errors from the consumed external sets.

Report all six allowed variants, including failures.

## 7.4 Review of the fresh external protocol

The current protocol has several strong safeguards:

* the 108- and 162-case cohorts are explicitly quarantined;
* C1/C2 probabilities, weights, and 0.5 thresholds are locked;
* prediction hashes precede label release;
* full coverage is primary;
* center and subtype performance are required;
* any model modified after unblinding requires another fresh cohort.

I recommend the following additions.

### Candidate hierarchy

At fresh evaluation, permit:

* C1;
* C2;
* at most one newly locked candidate.

The historical fusion may remain descriptive, but should not consume the primary hypothesis.

Primary hypotheses:

1. absolute performance of the new candidate;
2. paired new-candidate versus C1 difference.

C2 comparison is secondary. If two superiority comparisons are treated as confirmatory, apply a simple multiplicity correction such as Holm.

### Cohort construction

Use a consecutive, prevalence-preserving primary cohort. This is essential for valid precision/PPV estimates.

A separate B1/B2-enriched cohort may be collected to increase boundary precision, but:

* it cannot provide population PPV without reweighting;
* it must be clearly labeled secondary.

### Reference standard

Predefine:

* handling of mixed thymoma;
* highest-grade versus dominant-component rule;
* carcinoma mapping;
* repeat specimens;
* neoadjuvant treatment;
* equivocal B1/B2 diagnoses.

Have at least two pathologists verify B1/B2 and mixed/borderline cases, blinded to model results.

### Quality and exclusions

* Image adequacy should be assigned before model execution by an independent person.
* Primary analysis should retain all intended-use images, with deterministic fallback for model failure.
* Do not create a favorable “readable” primary subset after seeing outcomes.
* Report an all-eligible result and prespecified adequacy strata.

### Center-level inference

The current protocol proposes source-and-label-stratified bootstrap. With multicenter data, add:

* center-specific point estimates;
* hierarchical bootstrap that resamples centers and then patients;
* heterogeneity reporting;
* exact accounting of center contribution.

“No center-level collapse” should be operationalized, for example:

* no adequately sized center with BAcc below 0.65;
* no adequately sized center with high-risk sensitivity below 0.60.

## 7.5 Practical sample-size targets

The present protocol’s minimum of 200 and preference for 300 is a reasonable floor, but it needs composition requirements.

### Minimum decision-grade cohort

* at least 300 consecutive patients;
* at least 3 centers;
* at least 120 high-risk and 120 low-risk cases;
* at least 60 B1 and 60 B2 cases pooled;
* no center contributing more than 50%;
* at least 20–25 high-risk cases per center.

At a true sensitivity of 0.75, 120 high-risk cases give an approximate 95% binomial interval half-width of 0.08. This is acceptable for an initial independent decision, but not highly precise.

### Preferred confirmatory cohort

* 450–600 consecutive patients;
* at least 4 centers;
* at least 200 high-risk and 200 low-risk cases;
* at least 80–100 B1 and 80–100 B2 cases;
* no center contributing more than 40%.

At 200 high-risk cases and sensitivity near 0.75, the approximate interval half-width is about 0.06. With 80–100 cases per boundary subtype, accuracy intervals near 0.65–0.70 have a half-width of roughly 0.09–0.10.

### Boundary reporting

For B1/B2 report:

* B1 false-positive rate;
* B2 false-negative rate;
* B1 specificity;
* B2 sensitivity;
* B1/B2 boundary BAcc;
* PPV among predicted high-risk cases in the consecutive primary cohort.

“Precision within a true subtype” is not directly defined in the usual classification sense, so B1 specificity and B2 sensitivity are the more interpretable subtype endpoints.

## 7.6 Fresh external success rules

A strong confirmation requires:

* overall BAcc ≥0.75;
* high-risk sensitivity ≥0.70;
* low-risk specificity ≥0.70;
* lower 95% confidence bound for high-risk sensitivity at least approximately 0.65;
* no center-level collapse;
* candidate versus C1 paired ΔBAcc point estimate at least +0.02;
* superiority wording only if the paired interval excludes zero;
* positive or nonnegative B1 and B2 direction.

Classify outcomes as:

* **Confirmed:** meets the strong gate and center replication.
* **Promising but inconclusive:** BAcc 0.70–0.75, or paired interval crosses zero.
* **Failed transfer:** BAcc below 0.70, sensitivity or specificity below 0.65, or material center collapse.

No threshold adjustment, calibration, routing change, or fusion tuning is permitted after unblinding.

# 8. First Five Experiments and Go/No-Go Decisions

| Order | Experiment                                                              | Can start with                               | Go decision                                                                                                   | No-go decision                                                                                                                                  |
| ----- | ----------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | Leakage-safe, fully nested No.64 reconstruction                         | Existing predictions/code                    | Retain only as a behavior-meta comparator if nested gain over Candidate 41 is ≥+0.03 and rescues exceed harms | Retire the 92% magnitude if gain falls below +0.02, difficulty removal collapses routing, or harms emerge. Outcome never becomes a visual claim |
| **2** | A1 native-detail direct classifier                                      | Existing primary images and C1 encoder       | OOF +0.03, paired CI >0, LODO +0.02, no source collapse                                                       | Stop native-tile family after MIL/cross-attention pair; do not tune scales/losses repeatedly                                                    |
| **3** | B1 fixed-route M0–M4 cascade                                            | Same new tile bank plus cached C1            | M2−M0 ≥+0.06 routed, M3−M1 ≥+0.03, full OOF +0.02, positive LODO                                              | If only M1 improves, classify as behavior correction and stop visual-cascade claim                                                              |
| **4** | Blinded physician ROI sufficiency/oracle study                          | Physician adjudication on 120 internal cases | Manual-ROI M2 gain ≥+0.08, B1/B2 nonnegative, meaningful reader agreement; then train automatic ROI           | If manual ROI is not better than deterministic or random tiles, stop ROI development                                                            |
| **5** | Standardized multiview acquisition and one fresh multicenter blind test | New multiview and multicenter data           | Lock one model; confirm at BAcc ≥0.75, sens/spec ≥0.70, stable centers                                        | Below BAcc 0.70, center collapse, or M2 visual gain failing to replicate ends the claim; revised models require another cohort                  |

## Execution grouping

### Can start immediately

* Experiment 1: strict nested historical audit.
* Experiment 2: native-detail feature-bank extraction and direct model.
* Experiment 3: fixed-route M0–M4 experiment after the tile bank is available.

Experiments 1 and 2 can run in parallel. Experiment 1 is claim hygiene and must not delay visual work.

### Requires physician ROI/view adjudication

* Experiment 4.
* Include the blinded visual-sufficiency component before training an ROI detector.

### Requires new data

* Standardized original multiview development.
* Fresh multicenter blind confirmation.
* A proper external B1/B2 boundary cohort.

The decisive branch is simple:

> **If M2 cannot beat M0 on the same routed cases, stop calling the system coarse-to-fine. If a manual high-resolution ROI cannot improve the result either, stop optimizing the same single photograph and collect genuinely new views or centers.**
