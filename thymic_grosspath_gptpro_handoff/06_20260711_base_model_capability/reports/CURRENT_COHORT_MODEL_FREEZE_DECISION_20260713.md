# Current-Cohort Model Freeze Decision (2026-07-13)

## Decision

Freeze nomination of new Task7 classifiers from the repeatedly inspected
591-case internal cohort. This is a scientific model-selection boundary, not a
claim that no code can be written or no transform can be imagined.

The current photographs remain valid for reproducibility, retrospective error
analysis, engineering audits, and hypothesis generation. They are no longer an
independent basis for claiming that another selected model improves
generalizable image-reading capability.

## Evidence supporting the boundary

1. H3 PE-Spatial increased mixed-source OOF BAcc to 0.8003 but failed the
   cross-domain advancement gates.
2. H4 quality consistency reduced source-LODO BAcc to 0.7254.
3. H5 texture modeling retained mixed-source performance but reduced
   source-LODO BAcc to 0.7422 and B2 accuracy to 0.4719.
4. Fixed frequency features significantly predicted acquisition source after
   controlling risk, while risk prediction after controlling source was weak
   and not significant under the locked permutation audit.
5. Label-free PE parts were coherent and perturbation-stable but formally
   source-sensitive; source effects on stability were much larger than risk
   effects.
6. Only 17/591 internal cases have a second photograph, all in the historical
   old-data domain. The remaining 574 cases provide no added view.
7. The 108-case and 162-case external cohorts have already been inspected and
   are consumed audit sets, not fresh confirmatory tests.

## Frozen work

Do not use the current cohort to nominate another model through:

- backbone, PE-layer, pooling, ROI, part, tile, texture, frequency, or wavelet
  search;
- LP-FT, SAFT, LoRA, loss, sampler, seed, or threshold search;
- confidence/disagreement routing presented as improved visual capability;
- fusion-weight or specialist-route optimization;
- reuse of the consumed external cohorts for selection or confirmation.

Bug fixes and exact reproducibility reruns are allowed, but they cannot silently
change the locked model family or advancement criteria.

## Work that remains valid

- Retrospective, clearly labeled error and bias audits.
- Data-manifest, image-integrity, and acquisition-quality engineering.
- Manuscript reporting of locked positive and negative results.
- Prospective four-view multicenter collection and completeness monitoring.
- A future gross-pathology SSL program using a large center-diverse development
  pool that excludes every sealed external-test center.

The existing physician feature table remains explanatory only. The PE part
audit does not justify a new physician annotation request on the current
cohort.

## Reopening gates

Model development reopens only after the new-data package satisfies the locked
protocol in `FOUR_VIEW_MULTICENTER_ACQUISITION_PROTOCOL_20260713.md`, including:

- at least four development hospitals;
- four fixed image slots with at least 95% overall and 90% per-center
  completeness;
- a prespecified primary single image for paired single-view versus four-view
  comparison;
- external-test hospitals sealed before model fitting;
- labels, metadata, missingness, code, and prediction custody governed before
  test release.

The historical C2 model remains the balanced internal reference. H3 PE-Spatial
remains the strongest mixed-source exploratory representation. Neither is a
confirmed multicenter clinical model.
