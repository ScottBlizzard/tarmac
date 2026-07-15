# H10 Stage 2B Early Stop Before Hard Replay Preregistration

Experiment ID: `H10_STAGE2B_EARLY_STOP_BEFORE_HARD_REPLAY_20260715`

Date locked: 2026-07-15

## Question

H10 Stage 2 showed a significant BAcc loss after forcing every final model through
targeted replay. Its validation curves usually peaked during the anchor or boundary
phase, while training loss approached zero. Stage 2B tests the user's hypothesis that
more exposure to difficult cases is not necessarily useful in this imbalanced cohort.

This is a post-Stage-2 exploratory ablation. It may reject the hard-replay mechanism,
but it cannot by itself provide independent confirmation of a positive method.

## Locked design

- Cohort: all 591 same-institution cases, with no batch/domain partition in training.
- Split: the unchanged subtype-only fivefold split, SHA-256
  `f13030d7467c907851ed89abe14e385e024fcdddb6ad4be26433bc7994a10beb`.
- Features and model: the same frozen PE-Spatial-L14-448 six-view bank and 151,107
  parameter H3 head used by H10.
- Roles: exact server-only nested OOF roles generated for H10 Stage 2. Each role was
  generated only from its outer-training partition; outer validation/test cases did
  not participate. Role files must pass byte-for-byte SHA-256 verification after the
  server-local copy.
- Source/batch remains forbidden from sampling, loss, checkpoint selection, and model
  inputs. Historical source labels are used only for the final audit.
- Physician concepts and roles remain training controls, not inference inputs.

## Only changed factor

The first 16 epochs of the registered H10 curriculum are retained:

- epochs 1-8: anchor warmup;
- epochs 9-16: boundary bridge;
- no targeted hard replay.

All cases keep positive sampling probability. The best outer-validation BAcc from any
of epochs 1-16 is selected. Every fold still runs all 16 epochs; seed, optimizer,
batch size, fixed threshold 0.5, and 100% coverage are unchanged.

## Comparator and gates

Primary comparator: H10 source-free risk-balanced baseline on the identical split and
seed. A positive exploratory signal requires all of:

1. pooled BAcc gain at least +0.0100 and subtype-stratified paired-bootstrap 95% CI
   lower bound above zero;
2. accuracy at least 0.7986, TP at least 167, and TN at least 299;
3. B1 and B2 both non-decreasing, with combined net gain at least five cases;
4. at least three of five outer folds have positive test BAcc delta;
5. no historical-source audit BAcc decline below -0.0200 and minimum source BAcc at
   least 0.7419;
6. complete 591-case evaluation at threshold 0.5.

If these gates fail, phenotype-based easy/boundary scheduling is closed as a direct
per-case sampling mechanism. The roles may still be retained for error stratification.
