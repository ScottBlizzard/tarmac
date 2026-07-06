# v157 Framework Claim Status After v156

## One-sentence Story

The current project should be written as a risk-controlled, cross-domain gross pathology diagnosis framework, not as a single-model accuracy race.

## Module Status

| Stage | Module | Current status | Evidence | Writing boundary |
|---|---|---|---|---|
| Stage 1 | Primary image classifier / selective auto-pass backbone | main workflow backbone | Previous v50/v79/v118-v119 family establishes the high-safety workflow and two-signal risk card on old+third+strict external. | The remaining innovation must come from risk control and cross-domain handling, not from claiming a single stronger classifier. |
| Stage 2 | Direction-aware router | candidate supported by point-estimate gain | v148 strict external image-direction router BAcc 80.6% vs low-conf 78.1%, Delta +2.49 pp; concept-direction review Delta +1.12 pp. | External sample size is small; significance is trend-level. |
| Stage 3 | Image-distilled concept / pseudo-severe automatic correction | severe-shift-only candidate | v155 concept-direction rule: flips 35, rescued 31, hurt 4, BAcc 77.1%, Delta BAcc +23.88 pp; ordinary-domain boundary: old_data: flips 59, net -17, rescued 21, hurt 38, Delta BAcc -6.10 pp; third_batch: flips 60, net -18, rescued 21, hurt 39, Delta BAcc -10.59 pp | Unsafe as a global auto-flip rule; needs a reliable severe-shift gate. |
| Stage 4 | Unlabeled shift gate for safe correction | core safety boundary evidence | v156 internal max shift index 0.969, strict external 2.324; gated strict-external Delta BAcc +23.88 pp; false old/third gate -6.10 pp/-10.59 pp. | Current separation is shown on existing batches; more external batches are required to calibrate a deployable threshold. |
| Stage 5 | Reject / clinical review fallback | required safety layer | v151/v153 show safe all-domain auto-flip collapses to zero or near-zero coverage; v156 shows false severe-shift triggers are harmful. | Review burden must be reported with accuracy; high accuracy without review-rate accounting is incomplete. |

## Claim Boundaries

| Claim | Status | Evidence | Boundary |
|---|---|---|---|
| The framework is more than a single classifier. | supported as framework design | It contains primary classifier, direction-aware router, concept-distilled correction, unlabeled shift gate, and rejection layer. | Must still quantify each module separately and avoid implying every module is fully locked. |
| Concept information is useful. | supported with nuance | v141 direct fusion is nearly negative, v142/v144/v154-v156 show concept value appears in routing/correction rather than direct classification. | Write as structured intermediate supervision / risk signal, not as simple multimodal fusion. |
| Automatic correction can improve severe external shift. | candidate | v156 gated severe-shift Delta BAcc +23.88 pp. | Cannot be claimed as deployable without severe-shift gate validation and prospective external validation. |
| Unconditional auto-correction is unsafe. | supported | v156 no-gate all-domain Delta BAcc -3.72 pp. | This is a strength of the risk-control framing: it explains why naive post-processing is rejected. |

## Next Experimental Priority

1. Calibrate the no-label severe-shift gate on old+third and any future non-test external-like data.
2. Keep the strict external set as validation only; do not select thresholds using its labels.
3. For every future result, report auto-pass rate, auto-correct rate, review/reject rate, Acc, BAcc, F1, FN, FP, rescued cases, and hurt cases.
4. If another severe-shift development batch becomes available, validate whether the v154/v156 concept-direction correction remains beneficial before moving it from candidate to locked module.