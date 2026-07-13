# Task7 Frequency Source-versus-Risk Audit Preregistration (2026-07-13)

## Role and boundary

This is a diagnostic audit requested after the H3-H5 no-go decision. It cannot
nominate, tune, or reopen a Task7 classifier. Its purpose is to test whether
fixed frequency information is more strongly associated with acquisition
source than with low/high-risk morphology in the current photographs.

## Cohort

- Exactly 591 internal cases and one previously selected photograph per case.
- Acquisition sources: batch 1, batch 2, and third batch.
- Task7 risk labels are used only for this diagnostic comparison.
- No strict-external or new-external image is used.

## Locked representation

For each photograph, analyze two deterministic views:

1. the whole image;
2. the existing label-free specimen crop.

Each view is resized to 512 x 512 with bilinear interpolation. Convert RGB to
fixed Y, Cb, and Cr channels. Apply a manually implemented orthonormal 2D Haar
decomposition for five levels. At each level and channel, retain log energy for
horizontal, vertical, diagonal, and combined detail coefficients, relative
detail energy, and the fine-to-coarse energy ratio. There is no learned image
encoder and no feature or wavelet-family search.

## Primary analysis

Use fixed L2 logistic diagnostic probes with 5-fold cross-validation repeated
20 times:

- predict the three acquisition sources after training-fold residualization of
  frequency features for risk;
- predict binary risk after training-fold residualization of frequency features
  for acquisition source.

Folds are stratified jointly by target and confound. Preprocessing is fit on
training folds only. Evaluate balanced accuracy. Report normalized above-chance
separability, using chance 1/3 for source and 1/2 for risk.

Run 100 one-sided null permutations. Source labels are permuted within risk
strata; risk labels are permuted within source strata.

For every locked frequency feature, also report partial eta-squared for source
controlling risk and for risk controlling source.

## Locked interpretation rule

Declare frequency information source-dominant only if both conditions hold:

1. normalized source separability minus normalized risk separability is at
   least 0.10;
2. median source partial eta-squared exceeds median risk partial eta-squared.

Regardless of result, this audit does not authorize a frequency classifier on
the repeatedly inspected 591-case cohort. A source-dominant result informs the
camera, compression, focus, and resizing controls in the prospective capture
protocol. A non-source-dominant result is hypothesis-generating only and still
requires genuinely new confirmation data.

## Privacy and storage

Case-level frequency features, metadata, and any diagnostic predictions remain
on the server. GitHub receives only this preregistration, executable code, and
aggregate feature-level/result summaries. No image or model download is needed.
