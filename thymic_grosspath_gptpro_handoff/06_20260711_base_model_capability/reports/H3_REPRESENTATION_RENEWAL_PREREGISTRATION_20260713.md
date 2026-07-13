# H3 Representation Renewal Preregistration

Date: 2026-07-13

## 1. Purpose

H2 rejected one narrow hypothesis: final-layer dense tokens from the existing
square-resized SigLIP-L@512 encoder do not become more transferable merely by
replacing gated pooling with a coordinate-aware relational head.

H3 tests a different hypothesis:

> A materially different pretrained representation and information-preserving
> input geometry can improve direct, fixed-threshold, full-coverage risk
> classification before any workflow, routing, or confidence correction.

H3 must not change the binary endpoint, case splits, threshold, source labels,
or evaluation population.

## 2. Immutable cohort and endpoint

- 591 internal-development cases.
- Sources: batch1 (117), batch2 (168), third_batch (306).
- Subtypes: A 44, AB 262, B1 62, B2 89, B3 24, TC 110.
- Binary endpoint: A/AB/B1 are low risk; B2/B3/TC are high risk.
- Five-fold protocol: test fold `k`, validation fold `k+1`, remaining folds for
  training.
- Source-LODO protocol: one complete source is held out, validation uses the
  predetermined next master fold among the remaining sources.
- Decision threshold: 0.5.
- Coverage: 100%.
- Primary seed: 20260713. Confirmation seed 20260714 is allowed only after a
  positive primary result.

The three sources are acquisition batches from one project, not independent
hospitals. Source-LODO measures batch robustness and must not be described as
multicenter external validation.

## 3. Frozen candidate set

The existing SigLIP-L@512 bank is the reference representation. The new
candidate set is fixed before outcome inspection:

1. `google/siglip2-so400m-patch16-naflex`
2. `nvidia/C-RADIOv2-L` (RADIOv2.5; ModelScope mirror:
   `nv-community/C-RADIOv2-L`)
3. `facebook/PE-Spatial-L14-448`
4. `google/medsiglip-448`
5. `google/siglip2-large-patch16-512`
6. `google/siglip2-base-patch16-naflex`

`SigLIP2-L NaFlex` is not listed because no official checkpoint exists. SAM2
plus SLCA is a custom architecture rather than a pure encoder replacement and
is deferred until this representation screen is complete. Exact revisions,
weight hashes, licenses, preprocessing settings, and output dimensions must be
recorded at extraction time.

Availability correction recorded before inspecting any new-candidate outcome:
the initial draft used the nonexistent shorthand `nvidia/RADIO-L`. The frozen
candidate intended by the survey is the official RADIOv2.5 Large checkpoint
`nvidia/C-RADIOv2-L`; the ModelScope namespace above is only its network-accessible
mirror and does not change the candidate.

## 4. H3A: low-cost nested representation screen

H3A is an engineering and training-side selection stage, not the final efficacy
claim.

For each of the six views (`whole`, `crop`, `crop_q0`-`crop_q3`), the frozen
dense patch sequence is reduced without labels to three fixed statistic tokens:

1. masked mean;
2. masked standard deviation;
3. masked channel-wise maximum.

All candidates use the same trainable head:

- per-token LayerNorm;
- projection to 128 dimensions;
- learned view and statistic embeddings;
- gated attention pooling over 18 tokens;
- LayerNorm and one two-class linear output.

No subtype loss, class-specific expert, focal loss, threshold search, source
adversary, probability fusion, or rejection workflow is allowed.

Candidate ranking uses validation predictions only. For each outer fold, the
encoder with the highest validation balanced accuracy is selected. Aggregate
ranking uses mean five-fold validation balanced accuracy, then mean LODO
validation balanced accuracy, then the smaller encoder as the deterministic
tie-breaker. Outer test predictions cannot change the shortlist.

At most two encoders advance to H3B.

## 5. H3B: full dense-token confirmation

For each H3A winner, preserve the model's supported geometry and cache the full
dense sequence plus a valid-token mask. The existing matched gated head is used
with masked softmax; padded NaFlex tokens cannot participate in pooling.

H3B compares:

- existing SigLIP-L@512 square-resize matched gated reference;
- winning encoder with its official preprocessing;
- where supported, the same model family under fixed-square and native-aspect
  preprocessing to separate representation and geometry effects.

Only after an encoder passes H3B may multi-layer fusion, second-order texture,
or spatial re-embedding be tested.

## 6. H3B success gates

The reference is C2: OOF BAcc 0.7514, LODO BAcc 0.7441, LODO sensitivity
0.7354, LODO specificity 0.7527, B1 accuracy 0.5000, and B2 accuracy 0.6629.

All conditions are required:

1. OOF BAcc >= 0.7664.
2. Source-LODO BAcc >= 0.7641.
3. Source-LODO sensitivity >= 0.7354.
4. Source-LODO specificity >= 0.7527.
5. At least two held-out sources improve.
6. No held-out source declines by more than 0.02.
7. B1 accuracy >= 0.5000.
8. B2 accuracy >= 0.6629.
9. Mean B1/B2 accuracy improves by at least 0.03 and neither subtype declines.
10. Paired source-stratified bootstrap 95% CI lower bound for LODO delta BAcc
    is greater than zero.
11. The confirmation seed remains directionally positive.
12. Threshold remains 0.5 with 100% coverage.

## 7. Conditional continuation

- If H3B passes: test fixed early/middle/final layer fusion, then a separately
  ablated low-rank second-order texture token. Spatial re-embedding is allowed
  only on the new winning representation and must retain a coordinate
  permutation control.
- If frozen H3B passes but adaptation is needed: run linear probe first, then
  SAFT using training-fold gradients only.
- If no H3B candidate passes: close representation optimization on the current
  591-case single-photo cohort and prioritize standardized prospective
  multiview, multicenter data.

## 8. Storage and data-transfer rule

Patient images, feature banks, model weights, and case-level predictions remain
on the server. Only code, manifests, hashes, aggregate metrics, and compact
reports may be transferred to the local repository. One large dense feature
bank is retained at a time; completed regenerable banks may be removed after
their hashes and aggregate outputs are recorded.
