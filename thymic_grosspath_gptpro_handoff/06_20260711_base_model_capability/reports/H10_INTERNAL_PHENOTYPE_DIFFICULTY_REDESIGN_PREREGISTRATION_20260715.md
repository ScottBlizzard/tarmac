# H10 Internal Phenotype-Difficulty Redesign Preregistration

Experiment ID: `H10_INTERNAL_PHENOTYPE_DIFFICULTY_REDESIGN_20260715`

Date locked: 2026-07-15

## Why H9 does not answer the internal-capability question

H9 used source-LODO evaluation and defined online difficulty from the current model's
training-set true-class probability. That experiment was a valid source-shift stress
test, but it mixed two different questions:

1. whether the image model can classify cases from the same institutional development
   population; and
2. whether a source-specific decision rule transfers when an entire acquisition batch
   is held out.

The 285 old cases and 306 third-batch cases are therefore treated as one 591-case
internal development cohort for H10. Historical batch labels remain available only for
post-hoc audit. They must not enter fold construction, sampling weights, model inputs,
checkpoint selection, or the primary decision.

The strict 108-case external stress set remains untouched and is not used in H10.

## Physician-data audit locked before training

The hospital gross-findings workbook can be matched to 589/591 internal cases. Only
concepts parsed from `gross_text` / `肉眼所见` may be used. `diagnosis_text` is forbidden
because it directly contains the target diagnosis.

The initial audit showed that a one-dimensional physician high/low-risk score is too
coarse:

- gross-concept score AUC for the binary target: 0.6621;
- canonical concept pattern: 151 cases;
- mixed high- and low-risk concepts: 359 cases;
- concept-discordant pattern: 67 cases;
- uninformative pattern: 12 cases;
- missing gross record: 2 cases.

Under locked source-LODO H3 predictions, accuracy was 80.1% in canonical cases, 77.4%
in mixed cases, and 68.7% in discordant cases. This supports phenotype-aware analysis,
but does not justify using the gross-concept score as an inference feature or as a
single easy/medium/hard label.

## Stage 1: true internal H3 baselines

### Split

- Cohort: all 591 internal cases.
- Unit: one case, with 591 unique case IDs and 591 unique original pathology IDs.
- Five folds: deterministic shuffled `StratifiedKFold`, stratified only by six-class
  subtype (`A`, `AB`, `B1`, `B2`, `B3`, `TC`).
- Split seed: 20260715.
- Fold sizes and each subtype count must differ by at most one across folds.
- Each outer fold uses the next fold cyclically for validation and the other three
  folds for training.

### Representation and head

- Frozen encoder: `facebook/PE-Spatial-L14-448`.
- Checkpoint SHA-256:
  `47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`.
- Six dense views: whole, specimen crop, and four specimen quadrants.
- Locked H3 masked gated pooling head: 151,107 trainable parameters.
- Full-coverage binary prediction at threshold 0.5.
- Optimizer/training lock: AdamW, learning rate 3e-4, weight decay 1e-4,
  batch size 8, dropout 0.10, up to 80 epochs, patience 12.

### Sampling candidates

No candidate may use acquisition batch or source.

1. `INTERNAL_NATURAL`: every training case appears exactly once per epoch in a
   deterministic random order.
2. `INTERNAL_RISK_BALANCED`: weighted replacement sampling with equal total mass for
   low- and high-risk cases.
3. `INTERNAL_SUBTYPE_TEMPERED`: equal total mass for low/high risk; within each risk
   group, subtype mass is proportional to the square root of subtype frequency. This
   moderately increases rare-subtype exposure without forcing six equal subtype masses.

### Primary and secondary endpoints

Primary endpoint: pooled fivefold binary balanced accuracy.

Secondary endpoints:

- accuracy, AUC, sensitivity, specificity, TP/TN/FP/FN;
- correct counts for every subtype, especially B1 and B2;
- fold stability;
- batch/source-stratified metrics for audit only.

A sampling method is called a clear internal winner only if it exceeds the next-best
method by at least 0.010 balanced accuracy and the lower bound of a 20,000-replicate
subtype-stratified paired bootstrap confidence interval is above zero. Otherwise the
decision is `NO_CLEAR_INTERNAL_SAMPLER_WINNER`.

## Stage 2: redesigned difficulty system

The pooled Stage-1 OOF predictions are diagnostic only. They may describe the cohort,
but they must not be copied directly into a later outer-fold training table.

Difficulty is represented by intersecting axes rather than a single confidence tier:

1. **Physician phenotype axis**
   - canonical: gross concepts point in the same direction as the binary label;
   - mixed: both low- and high-risk concepts are recorded;
   - discordant/mimic: concepts point in the opposite direction;
   - sparse/missing: no usable directional concept pattern.
2. **Model learnability axis**
   - stable: all internal cross-fitted teachers are correct;
   - boundary: at least one but not all teachers are correct;
   - persistent failure: all teachers are wrong.
3. **Subtype-boundary axis**
   - special attention to B1/B2 and to AB/TC phenotype overlap;
   - subtype remains an audit/stratification label, never an inference-time oracle.
4. **Image adequacy axis**
   - clarity, specimen coverage, and cut-surface/view labels are used only where they
     are actually available; missing quality labels are never imputed as good quality.

The resulting diagnostic roles are:

- `canonical_anchor`;
- `stable_noncanonical`;
- `learnable_boundary`;
- `persistent_canonical_failure` (physician pattern is canonical but the image model
  fails, suggesting representation/view failure);
- `persistent_mimic_failure` (mixed or discordant phenotype and persistent model
  failure);
- `persistent_sparse_or_missing`.

Any Stage-2 curriculum or pair-learning experiment must recreate model learnability
inside each outer training partition through nested OOF prediction. Outer validation
and test labels or predictions are forbidden from training weights and sample roles.

## Interpretation boundary

H10 Stage 1 estimates same-institution internal discrimination. It does not replace
source-LODO or strict external evaluation and cannot establish external generalization.
The purpose is to stop conflating internal model capability with batch-transfer stress,
then build a clinically meaningful training taxonomy without confidence leakage.
