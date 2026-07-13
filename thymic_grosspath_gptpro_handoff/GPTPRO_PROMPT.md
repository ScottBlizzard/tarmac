# Current GPTPro Prompt

Repository: https://github.com/ScottBlizzard/tarmac

Use the complete current prompt at:

`thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/GPTPRO_PROMPT_20260713_AFTER_SEQUENTIAL_NO_GO.md`

Read the latest result ledger first:

`thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H1_SEQUENTIAL_AB_TC_FALLBACK_RESULTS_20260713.md`

Critical correction:

- The early direct visual model was approximately 76.5% Acc/BAcc on the 285-case old-data cohort.
- Candidate 41 image-model fusion reached approximately 83.5%.
- The approximately 92% No.64/`base162` results were behavior-level two-stage/meta-correction descendants, not evidence that a single/base visual model learned 92% fine morphology.
- The behavior-level gain did not transfer to the third-batch strict holdout or historical strict external cohort.

The project accepts either a stronger direct visual model or a genuine image-grounded coarse-to-fine multi-model system. A valid stage-2 specialist must re-read image evidence and demonstrate incremental value over a probability/confidence-only correction baseline on nested OOF, source-LODO, and eventually a genuinely fresh label-blinded external cohort.

New evidence: the preregistered sequential AB expert, TC expert, and six-subtype-covered binary fallback also failed. Versus C2, it reduced five-fold BAcc from 0.7514 to 0.7195 and source-LODO BAcc from 0.7441 to 0.7278. Source-LODO sensitivity fell from 0.7354 to 0.6457. The AB expert collapsed across sources, the TC expert could not satisfy the fixed purity route gate, and the fallback again traded B1 gains for B2/B3/TC losses. The route itself changed very few decisions and was net-positive; the fallback was the main failure.

The existing physician feature table already covers 589 of 591 internal cases and may be used for retrospective explanation only. Do not require a new 120-case physician annotation exercise unless a precise new hypothesis makes that information indispensable. Recommend at most two structurally new image-grounded experiments, or state plainly that the next rational step is standardized additional views and genuinely new multicenter data.

Do not redirect the project toward automatic release coverage, rejection rules, or confidence-only correction.
