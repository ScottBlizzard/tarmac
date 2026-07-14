# Fixed-Data Exploratory Reopen Decision (2026-07-14)

## Constraint clarification

The project cannot acquire a new hospital, patient cohort, image view, label,
or unlabeled gross-pathology pool. The prospective four-view multicenter
protocol remains a statement of ideal future evidence, but it is not an
executable dependency for the current project.

The 2026-07-13 freeze is therefore narrowed:

- confirmatory generalization claims remain frozen;
- unconstrained model, seed, threshold, fusion, and routing searches remain
  frozen;
- one bounded, literature-justified exploratory visual-capability experiment
  may reopen after explicit comparison with completed mechanisms.

## Reopened objective

Improve the direct image-only Task7 classifier under the fixed-data constraint,
while being explicit that the result is exploratory internal cross-batch
evidence rather than fresh multicenter confirmation.

The reopened experiment must:

- retain threshold 0.5 and 100% coverage;
- use source-LODO as the primary model-selection test;
- use ordinary five-fold OOF only as a secondary result;
- compare against both C2 and H3 PE-Spatial;
- target H3's principal failure: high-risk/B2 under-calling under source shift;
- learn image evidence rather than confidence correction;
- avoid external labels, post-hoc thresholding, and test-source calibration;
- use one locked primary configuration and only the ablation needed to identify
  its mechanism;
- stop if its prespecified cross-batch gates fail.

The historical 108-case and 162-case external sets may be reported only as
consumed retrospective stress tests after the internal decision. They cannot
select a model or support an untouched-external claim.

## Literature decision request

`GPTPRO_PROMPT_20260714_FIXED_DATA_LITERATURE_SEARCH.md` asks an independent GPT
Pro review to search primary literature, eliminate substantive duplicates of
completed work, rank at most three executable mechanisms, and provide one exact
primary experiment plus one conditional backup.

This reopening does not restore statistical independence to the 591 cases. It
restores a practical engineering path under the user's fixed-data constraint,
with the evidence ceiling stated rather than ignored.

## Closeout (2026-07-14)

The single bounded experiment was completed as H6 nuisance-anchored CSD. It
failed both the source-LODO advancement gates and the five-fold retention
gates; the conditional Fishr trigger was absent. The exploratory reopening is
therefore closed without a confirmation seed, external stress-test rerun, or
follow-up hyperparameter/method search. See
[H6_NUISANCE_ANCHORED_CSD_RESULTS_20260714.md](H6_NUISANCE_ANCHORED_CSD_RESULTS_20260714.md).
