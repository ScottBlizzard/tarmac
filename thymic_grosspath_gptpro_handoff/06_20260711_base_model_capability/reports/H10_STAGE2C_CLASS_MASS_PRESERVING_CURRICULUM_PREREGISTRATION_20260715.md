# H10 Stage 2C Class-Mass-Preserving Curriculum Preregistration

Experiment ID: `H10_STAGE2C_CLASS_MASS_PRESERVING_CURRICULUM_20260715`

Date locked: 2026-07-15

## Rationale

Stage 2B removed hard replay and improved BAcc from 0.7498 to 0.7780, but remained
below the source-free risk-balanced baseline at 0.7888. It retained the baseline's
167 true positives but lost eight true negatives. Its AUC rose from 0.8430 to 0.8664.

The registered role factors changed effective class mass after risk balancing. During
the boundary bridge, high-risk mass averaged 0.5204 and reached 0.5343 in one fold.
Validation-derived threshold adjustment failed and is not an acceptable remedy.
Stage 2C tests one mechanism-specific correction at training time.

## Locked design

Everything from Stage 2B remains fixed:

- all 591 same-institution cases are one internal cohort;
- subtype-only fivefold split SHA-256
  `f13030d7467c907851ed89abe14e385e024fcdddb6ad4be26433bc7994a10beb`;
- identical frozen PE-Spatial-L14-448 six-view bank and H3 head;
- exact leak-free nested role files copied with SHA-256 verification;
- epochs 1-8 anchor warmup and epochs 9-16 boundary bridge;
- no targeted hard replay;
- all 16 epochs run and any epoch may be selected by outer-validation BAcc;
- fixed threshold 0.5 and 100% test coverage;
- source/batch forbidden from training and selection;
- physician concepts and roles unavailable at inference.

## Only changed factor

After role multiplication and the registered median clipping, weights are normalized
separately inside low- and high-risk training cases. Each risk class contributes
exactly 50% of total sampling mass in every epoch. Relative role weights within each
risk class remain unchanged, and every case retains positive probability.

No threshold, fusion rule, model structure, feature, or test-time operation changes.

## Decision rule

The same primary comparator and gates as Stage 2B apply. In addition, all recorded
epoch-level low/high sampling masses must equal 0.5 within numerical tolerance. If
Stage 2C does not beat the H10 risk-balanced baseline, phenotype-role scheduling is
closed as a direct per-case sampling mechanism; no additional role-factor search is
allowed on these folds.
