# H3 Representation Renewal Results (2026-07-13)

## Decision

H3 is a **NO-GO under the preregistered cross-domain gates**.

The PE-Spatial dense-token model produced a real direct-classification gain in
mixed-source five-fold evaluation: balanced accuracy increased from the locked
C2 reference of 0.7514 to 0.8003 at threshold 0.5 and 100% coverage. This is not
a rejection or confidence-gating result. However, source-LODO balanced accuracy
was only 0.7539 versus C2's 0.7441, with a paired delta of +0.0098 and 95% CI
[-0.0253, +0.0457]. Source-LODO sensitivity fell from 0.7354 to 0.6816 and B2
accuracy fell from 0.6629 to 0.5843. The improvement therefore did not transfer
reliably across acquisition batches.

The second H3A winner, SigLIP2-Base-NaFlex, failed both OOF and source-LODO
advancement criteria. Because neither frozen H3B candidate passed all gates,
the preregistered multi-layer, texture, spatial re-embedding, and SAFT branches
were not opened.

## Locked evaluation

- Cohort: 591 internal-development cases.
- Sources: batch1 117, batch2 168, third_batch 306.
- Endpoint: A/AB/B1 low risk versus B2/B3/TC high risk.
- Five-fold: test fold k, validation fold k+1, remaining folds for training.
- Source-LODO: each acquisition batch held out in turn.
- Threshold: 0.5.
- Coverage: 100%.
- Primary seed: 20260713.
- Source-LODO is an internal batch-robustness proxy, not multicenter external
  validation.

## H3A frozen representation screen

Candidate ranking used validation predictions only. The final rank was
PE-Spatial, SigLIP2-Base-NaFlex, SigLIP2-Large-512, SigLIP2-So400m-NaFlex,
MedSigLIP-448, and C-RADIOv2.5-L. The top two advanced to H3B.

| Candidate | 5-fold val BAcc | 5-fold OOF BAcc | LODO val BAcc | LODO BAcc |
| --- | ---: | ---: | ---: | ---: |
| PE-Spatial-L14-448 | 0.7525 | 0.6967 | 0.7495 | 0.6687 |
| SigLIP2-Base-NaFlex | 0.7364 | 0.6812 | 0.7430 | 0.6353 |
| SigLIP2-Large-P16-512 | 0.7248 | 0.6599 | 0.7194 | 0.6137 |
| SigLIP2-So400m-NaFlex | 0.7244 | 0.6811 | 0.7432 | 0.6740 |
| MedSigLIP-448 | 0.7081 | 0.6456 | 0.7122 | 0.6352 |
| C-RADIOv2.5-L | 0.7039 | 0.6424 | 0.7273 | 0.6443 |
| Existing C1 compact reference | 0.7196 | 0.6809 | 0.7383 | 0.6451 |

The compact screen was deliberately lossy: each of six views was reduced to
mean, standard deviation, and maximum statistics. It selected encoders for the
full spatial-token confirmation; it was not used as the final capability claim.

## H3B full dense-token confirmation

Each view retained up to 1,024 valid spatial tokens. Padded tokens were masked
before softmax pooling. Full-bank audits confirmed finite values, zero padding,
and exact agreement between mask counts and spatial grid sizes.

| Model | OOF BAcc | OOF AUC | OOF Sens | OOF Spec | LODO BAcc | LODO AUC | LODO Sens | LODO Spec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 locked reference | 0.7514 | 0.8377 | 0.7175 | 0.7853 | 0.7441 | 0.8108 | 0.7354 | 0.7527 |
| PE-Spatial dense | **0.8003** | **0.8700** | **0.8072** | **0.7935** | **0.7539** | 0.7984 | 0.6816 | **0.8261** |
| SigLIP2-Base-NaFlex dense | 0.7535 | 0.8186 | 0.7489 | 0.7582 | 0.7161 | 0.7769 | 0.6278 | 0.8043 |

### PE-Spatial details

- Five-fold subtype accuracy: B1 0.6774, B2 0.6966.
- Five-fold source BAcc: batch1 0.8118, batch2 0.7619, third_batch 0.8032.
- Source-LODO subtype accuracy: B1 0.6452, B2 0.5843.
- Source-LODO source BAcc: batch1 0.7434, batch2 0.7381, third_batch 0.7440.
- Source-LODO source deltas versus C2: batch1 -0.0456, batch2 +0.0298,
  third_batch +0.0103.
- B1 delta versus C2: +0.1452.
- B2 delta versus C2: -0.0787.
- Mean B1/B2 delta: +0.0333, but the required no-subtype-decline condition
  failed.
- Paired source/risk-stratified bootstrap, 20,000 replicates: delta LODO BAcc
  +0.0098, 95% CI [-0.0253, +0.0457].

PE-Spatial learned a substantially stronger mixed-source boundary and improved
two of three held-out sources, but it did not preserve batch1 or high-risk
sensitivity. Its gain cannot be described as a robust external improvement.

### SigLIP2-Base-NaFlex details

- Five-fold subtype accuracy: B1 0.4839, B2 0.6629.
- Five-fold source BAcc: batch1 0.7776, batch2 0.7024, third_batch 0.7532.
- Source-LODO subtype accuracy: B1 0.5161, B2 0.5618.
- Source-LODO source BAcc: batch1 0.7162, batch2 0.6845, third_batch 0.6798.
- All three held-out sources declined versus C2.
- Paired source/risk-stratified bootstrap, 20,000 replicates: delta LODO BAcc
  -0.0280, 95% CI [-0.0662, +0.0103].

The dense-token mechanism alone was insufficient. PE-Spatial's spatially
aligned pretraining, not merely retention of more tokens, appears to account
for most of the five-fold gain.

## Locked gates

| Gate | PE-Spatial | SigLIP2-Base |
| --- | --- | --- |
| OOF BAcc >= 0.7664 | Pass | Fail |
| Source-LODO BAcc >= 0.7641 | Fail | Fail |
| Source-LODO sensitivity >= 0.7354 | Fail | Fail |
| Source-LODO specificity >= 0.7527 | Pass | Pass |
| At least two held-out sources improve | Pass | Fail |
| No held-out source declines by more than 0.02 | Fail | Fail |
| B1 accuracy >= 0.5000 | Pass | Pass |
| B2 accuracy >= 0.6629 | Fail | Fail |
| Mean B1/B2 improves >= 0.03 and neither declines | Fail | Fail |
| Bootstrap LODO delta BAcc CI lower > 0 | Fail | Fail |
| Confirmation seed directionally positive | Not eligible | Not eligible |
| Threshold 0.5 and coverage 100% | Pass | Pass |

## Interpretation

1. A stronger frozen representation plus full local tokens can materially raise
   direct full-coverage classification. The 0.8003 OOF result is the strongest
   non-rejection result in this experiment series.
2. That gain remains acquisition-batch dependent. PE-Spatial improved batch2
   and third_batch in source-LODO but regressed batch1 by 0.0456.
3. The central boundary did not improve symmetrically. PE-Spatial rescued B1
   but harmed B2, moving source-LODO behavior toward low-risk predictions.
4. Post-hoc threshold tuning, confidence routing, or confirmation-seed fishing
   would not repair the failed hypothesis. The next experiment must change the
   domain-robustness assumption and be preregistered as a new phase.

## Reproducibility and storage

All patient images, dense feature banks, fold predictions, and checkpoints
remain on the server. Only aggregate metrics, code, hashes, and reports are in
the repository.

PE-Spatial dense-bank hashes before regenerable-bank cleanup:

- dense features: `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`
- valid mask: `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`
- spatial shapes: `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`
- bank config: `f6e4ef8309e4c50e262e89d6d9c70098cf0f329d97a79f14ecb8922ff0a5dc50`

SigLIP2-Base-NaFlex dense-bank hashes before regenerable-bank cleanup:

- dense features: `4c8751822928210ed2d785e8c0886923f232130abe2508cdae42b5b7ea73fcf9`
- valid mask: `8321a0506427d5c1d3be5dddcd156a9e13e7653ab91a105e5f5ec23ee708b439`
- spatial shapes: `75c6d1961f2d22e842b2b5c69053c111b35af90064268c8ed59b9aa1c77e5e2a`
- bank config: `5702798ea6b7c3cbff8287cfd376e1e7f6f5a6c8d6edbcc8a0b8a07ce9c0424b`

Server aggregate outputs:

- `/workspace/thymic_project/experiments/h3_representation_renewal_20260713/aggregate`
- `/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs`
- `/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_gate_aggregate`

## Next phase

H3 representation optimization is closed for this cohort. H4 must be a new,
locked domain-robustness experiment centered on the observed failure: batch1
regression and high-risk/B2 under-calling under source shift. It must compare
against C2 and PE-Spatial under the same five-fold and source-LODO protocols,
retain threshold 0.5 and 100% coverage, and forbid source-specific test-time
calibration.
