# H5 Low-Rank Second-Order Texture Preregistration (2026-07-13)

## Status and rationale

H5 is a new user-directed exploratory experiment. It does not reopen or alter
the H3 or H4 decisions.

The research report identified true second-order texture representation as an
untested mechanism. Prior heads used weighted means, standard deviations, or
spatial relations. Those operations do not explicitly encode pairwise feature
channel co-occurrence. H5 tests one fixed low-rank covariance texture branch on
the strongest frozen PE-Spatial token bank. It remains a direct image-reading
binary classifier; it does not use confidence routing, rejection, physician
features, source labels at inference, or test-time calibration.

This is the last allowed experiment on the retained single-image PE bank. A
failure closes covariance-dimension, normalization, fusion, seed, loss, and
threshold searches on this representation.

## Immutable cohort and evaluation

- Cases: the same 591 internal-development cases.
- Sources: batch1 117, batch2 168, third_batch 306.
- Low risk: A/AB/B1; high risk: B2/B3/TC.
- Five-fold and source-LODO partitions are unchanged.
- Threshold: 0.5.
- Coverage: 100%.
- Primary seed: 20260713.
- Confirmation seed 20260714 is allowed only if every primary gate passes.
- Source-LODO remains an internal acquisition-batch stress test, not external
  multicenter validation.

## Frozen visual representation

- Encoder: `facebook/PE-Spatial-L14-448`.
- Views: whole image, foreground crop, and four crop quadrants.
- Native-aspect dense grid with up to 1,024 valid tokens per view.
- Feature dimension: 1,024.
- Encoder and input features remain frozen.
- Required clean feature hash:
  `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`.

## Single allowed model

Candidate: `pe_spatial_lowrank_covariance_v1`.

The model has two visual branches trained jointly:

1. First-order branch: the H3 masked 128-dimensional projection and gated
   pooling over all valid tokens and views.
2. Second-order branch:
   - LayerNorm the frozen PE tokens;
   - project each token from 1,024 to 64 dimensions;
   - center valid tokens independently within each view;
   - compute the 64 x 64 masked sample covariance for each view;
   - retain the upper triangle;
   - apply elementwise signed square-root and L2 normalization;
   - project the covariance vector to a 128-dimensional texture token;
   - gated-pool the six texture tokens.
3. Concatenate the first-order and texture case vectors, apply one fixed
   128-dimensional fusion layer, and output one low/high probability.

No source embedding, subtype auxiliary loss, boundary expert, quality feature,
probability fusion, routing rule, or abstention is allowed.

## Locked training

- Hidden dimension: 128.
- Attention dimension: 64.
- Covariance projection dimension: 64.
- Dropout: 0.10.
- AdamW learning rate: 3e-4.
- Weight decay: 1e-4.
- Maximum epochs: 80.
- Early-stopping patience: 12.
- Batch size: 4.
- Gradient clipping: 5.0.
- Training sampler: the same source-by-risk sampler as H3/H4.
- Loss: binary cross-entropy only.
- Checkpoint selection: validation BAcc only.

One single-epoch fold smoke test is allowed for shape, finite-value, masking,
gradient, and output-file validation. Smoke results cannot change any locked
choice.

### Pre-scientific numerical amendment

The first engineering smoke produced a finite loss but a non-finite FP16
gradient norm on its first backward pass; it emitted no scientific prediction
set. Before any primary run, the implementation was changed to BF16 autocast
with unscaled gradients (FP32 fallback where BF16 is unavailable), and the
signed square-root used a `1e-6` lower clamp. Model structure, dimensions,
sampler, loss, optimizer, learning rate, stopping rule, seed, and success gates
were unchanged. The repeated one-epoch smoke then completed with finite loss,
gradients, metrics, and output files. Its metrics remain engineering-only.

## References

- C2: OOF BAcc 0.7514; LODO BAcc 0.7441, sensitivity 0.7354,
  specificity 0.7527; B1 0.5000; B2 0.6629.
- H3 PE: OOF BAcc 0.8003; LODO BAcc 0.7539, sensitivity 0.6816,
  specificity 0.8261; B1 0.6452; B2 0.5843.
- H4 quality consistency: OOF BAcc 0.8009; LODO BAcc 0.7254,
  sensitivity 0.6682, specificity 0.7826; B1 0.6452; B2 0.5281.

## Primary success gates

All conditions are required:

1. Five-fold OOF BAcc >= 0.7903.
2. Five-fold sensitivity >= 0.7772 and specificity >= 0.7635.
3. Source-LODO BAcc >= 0.7641.
4. Source-LODO sensitivity >= 0.7354.
5. Source-LODO specificity >= 0.7527.
6. Source-LODO B1 accuracy >= 0.6000.
7. Source-LODO B2 accuracy >= 0.6629.
8. At least two held-out sources improve versus C2.
9. No held-out source declines by more than 0.02 versus C2.
10. Source-LODO BAcc and sensitivity both exceed H3 PE.
11. Source-LODO B2 accuracy exceeds H3 PE and does not trade away B1.
12. Paired source/risk-stratified bootstrap versus C2 has a 95% CI lower
    bound above zero for source-LODO BAcc.
13. A conditional confirmation seed remains directionally positive for OOF
    BAcc, source-LODO BAcc, source-LODO sensitivity, and B2 accuracy.
14. Threshold remains 0.5 with 100% coverage.

## Stopping rule

If the primary candidate fails any deterministic gate, do not run another
covariance dimension, matrix-square-root variant, shrinkage coefficient,
fusion architecture, sampler, loss, threshold, or confirmation seed. Close H5,
hash and delete the regenerable PE bank, and move the mainline to genuinely new
image information: standardized prospective multi-view photography, additional
unselected case images where available, or new multicenter data.

The historical 108-case and 162-case external cohorts remain consumed audit
sets and cannot be reopened for model selection.
