# H10 Stage 2 Nested Phenotype Curriculum Preregistration

Experiment ID: `H10_STAGE2_NESTED_PHENOTYPE_CURRICULUM_20260715`

Date locked: 2026-07-15

## Rationale

H10 Stage 1 established a source-free same-institution baseline and showed why a
single easy/medium/hard score is inadequate. Across natural, risk-balanced, and
subtype-tempered internal OOF models, 408/591 cases were always correct, 107 were
model-dependent, and 76 were always wrong. The persistent failures included two
different mechanisms:

- 63 mixed/discordant physician-phenotype mimics;
- 12 canonical physician patterns that all image models still missed;
- one sparse/missing case.

B1/B2 accounted for 41/76 persistent failures. A curriculum should therefore avoid
treating all hard cases as one population.

## Leakage firewall

The Stage-1 591-case OOF roles are diagnostic and forbidden as Stage-2 training
weights. For every outer fold:

1. outer test and validation cases are removed;
2. only the three outer-training folds are used to generate difficulty roles;
3. the outer-training cases are split into three subtype-stratified inner folds;
4. natural, risk-balanced, and subtype-tempered H3 teachers are trained on two inner
   folds for exactly 12 epochs and predict the remaining inner fold;
5. each outer-training case therefore receives three genuinely out-of-fold teacher
   predictions produced without outer validation/test images or labels;
6. physician concepts are joined only for outer-training cases;
7. the final outer model sees dense image tokens and the binary target only. Physician
   concepts and difficulty roles are not model inputs and are unavailable at inference.

The hospital `diagnosis_text` field remains forbidden. Only concepts parsed from the
raw physician gross-observation field may define the physician phenotype.

## Fixed training roles

The intersection of physician phenotype and three-teacher correctness defines:

- `canonical_anchor`: canonical phenotype and 3/3 teachers correct;
- `stable_noncanonical`: noncanonical phenotype and 3/3 teachers correct;
- `learnable_boundary`: 1/3 or 2/3 teachers correct;
- `persistent_canonical_failure`: 0/3 correct with canonical phenotype;
- `persistent_mimic_failure`: 0/3 correct with mixed/discordant phenotype;
- `persistent_sparse_or_missing`: remaining 0/3 cases.

Roles are fixed before final outer training. They are never updated from in-sample
training confidence.

## Curriculum

All outer-training cases retain positive sampling probability in every epoch. Base
weights balance low/high risk without using source or batch. Role factors are:

| Epochs | Anchor | Stable noncanonical | Learnable boundary | Persistent canonical | Persistent mimic | Sparse/missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-8, anchor warmup | 1.50 | 1.00 | 0.75 | 0.75 | 0.50 | 0.50 |
| 9-16, boundary bridge | 1.00 | 0.75 | 1.50 | 1.00 | 0.75 | 0.50 |
| 17+, targeted replay | 0.75 | 0.50 | 1.50 | 1.50 | 1.00 | 0.50 |

Weights are multiplied by the source-free risk-balanced base weights, clipped to
0.2-5.0 times the median, and normalized to mean one. Checkpoints from epochs 1-23
are ineligible. Selection by outer-validation BAcc begins at epoch 24, after the model
has completed both introductory phases and at least eight targeted-replay epochs.

## Locked model and split

- Cohort: the same 591 internal cases.
- Split: H10 subtype-only split SHA-256
  `f13030d7467c907851ed89abe14e385e024fcdddb6ad4be26433bc7994a10beb`.
- Source/batch is forbidden from fold assignment, teacher sampling, final sampling,
  model inputs, losses, and checkpoint selection.
- Frozen encoder: PE-Spatial-L14-448, locked checkpoint and six dense views.
- Head: locked H3 masked gated pool, 151,107 trainable parameters.
- Final optimizer: AdamW, learning rate 3e-4, weight decay 1e-4, batch size 8,
  dropout 0.10, up to 80 epochs, patience 12, minimum 24 epochs.
- Primary seed: 20260713.
- Threshold: 0.5; coverage: 100%.
- Comparator: H10 `INTERNAL_RISK_BALANCED` on the identical split and seed.
- Strict external data remain untouched.

## Advancement gates

All gates must pass:

1. pooled BAcc gain versus risk-balanced baseline at least +0.0100 and the lower bound
   of a 20,000-replicate subtype-stratified paired bootstrap CI above zero;
2. ordinary accuracy at least 0.7986;
3. TP at least 167 and TN at least 299;
4. B1 at least 40/62 and B2 at least 59/89;
5. combined B1+B2 net correct gain at least +8, with neither subtype declining;
6. no historical-source audit BAcc decline worse than -0.0200 and minimum audit BAcc
   at least 0.7419;
7. threshold 0.5 and complete 591-case coverage.

Passing only AUC, TC, AB, or a single boundary subtype is not sufficient. If any gate
fails, the current nested curriculum is a no-go and no confirmation or external run is
opened.

## Interpretation boundary

This experiment tests whether a physician-phenotype-aware, leak-free curriculum can
improve same-institution forced classification. Even a positive result would remain an
internal development result and would require a separately locked external test.
