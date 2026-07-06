# v194 Results Section Draft

## Mini-outline

- Main result: a risk-controlled, shift-adaptive selective diagnosis framework reaches high balanced accuracy while explicitly controlling automatic decisions.
- Deployment policy: unlabeled severe-shift auditing switches the system from the standard efficiency workflow to a high-safety fallback.
- Stable release: fixed image-agreement release improves review efficiency only when constrained by fold-wise stability; greedy release is rejected by audit.
- Negative evidence: automatic flipping and residual FN sentinels are not reliable enough to replace review under the current sample size.
- Safety boundary: automatic decision error rates require Wilson confidence intervals and prospective sample-size planning.

## Results Draft

**Paragraph role: opening / main result.**  
The final Task7 system should be interpreted as a risk-controlled selective diagnosis framework rather than a standalone binary classifier. Under the recommended v185 unlabeled shift-adaptive workflow, the system achieved an all-domain balanced accuracy of 99.81% with a review/reject rate of 58.94%, leaving FN=1 and FP=0. This operating point preserved the same all-domain balanced accuracy as both the fixed v182 efficiency workflow and the fixed v118 high-safety fallback, while avoiding the high review burden of the fixed high-safety mode (79.97%).

**Paragraph role: deployment mechanism.**  
The unlabeled shift-adaptive policy provides the deployment-time risk-control mechanism. For within-internal-shift batches, the system uses the fixed v182 stable release workflow, which keeps all-domain BAcc at 99.81% and reduces review/reject to 56.08%. When the severe-shift gate flags a batch as strict-external-like, the system falls back to v118 high-safety review; on the current strict external split this raises the review/reject rate from 68.52% to 87.04%. This result supports the system-level claim that risk control is conditional on the unlabeled batch state, not only on per-case confidence.

**Paragraph role: ablation / stable release.**  
The safe-release module was only retained after stability auditing. The fixed v182 stable release rule kept BAcc at 99.81% with zero released errors and a review/reject rate of 56.08%. In contrast, the greedy per-fold image-agreement audit lowered the review/reject rate to 48.21%, but released 3 errors and reduced BAcc to 99.40%. This contrast is important because it shows that lower review burden alone is not accepted as improvement; release rules must satisfy stability constraints.

**Paragraph role: comparator evidence.**  
Direction-aware routing provided mechanism-level evidence beyond confidence-only selective prediction. At a comparable 30% review budget on the strict external split, the image-direction router reached BAcc 80.59%, compared with 78.10% for the low-confidence selective-review baseline. The absolute performance of these 30% review policies was below the final safety-oriented workflow, but the comparison supports the claim that error direction and shift state contain useful information not captured by confidence alone.

**Paragraph role: negative correction evidence.**  
Automatic correction by direct flipping was not supported. The aggressive image-only corrector reduced review/reject to 16.31%, but retained automatic errors and was therefore treated as a safety-boundary candidate rather than a recommended workflow. The disagreement-flip policy produced severe harm (BAcc 68.62%, FN=87, FP=131), and the error-enriched flip-risk strategy also failed to transfer to the current residual-error distribution (BAcc 87.65%). These negative results justify keeping uncertain cases in review/rejection rather than forcing automatic correction.

**Paragraph role: residual error boundary.**  
The final residual automatic error was case 2516531 (TC), a high-risk-to-low-risk false negative. A simple fold-wise probability sentinel added 2 reviews but rescued 0 FN cases, and a learned DINO/probability FN-risk sentinel added 27 reviews but also rescued 0 FN cases. Although the best learned FN-risk model reached internal low-risk AUROC 0.838, it did not stably trigger review for the residual FN under fold-wise evaluation. This supports treating the remaining FN as a residual risk boundary rather than post-hoc tuning a case-specific rule.

**Paragraph role: statistical safety boundary.**  
Automatic decision safety was reported with confidence intervals rather than only with point estimates. The v185 adaptive workflow made 287 all-domain automatic decisions with 1 error, corresponding to an observed automatic error rate of 0.35% and a Wilson95 upper bound of 1.95%. On strict external cases, the adaptive workflow made only 14 automatic decisions; even with zero observed error, the Wilson95 upper bound remained 21.53%. A prospective validation set would need 73 zero-error automatic decisions to bound the Wilson95 upper limit below 5%, or 110 automatic decisions if one error is allowed.

## Self-review Checklist

- Clarity: The section states that the contribution is a risk-controlled workflow, not a raw classifier.
- Flow: Results move from main operating point to mechanism, ablation, comparator, negative evidence, residual boundary, and statistical boundary.
- Terminology: The terms v185 adaptive workflow, fixed v182 efficiency workflow, and fixed v118 fallback are used consistently.
- Unsupported claims: The draft does not claim mature automatic flipping or complete clinical safety.
- Missing evidence: Larger prospective external validation remains required, especially for strict external automatic-decision confidence intervals.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| The main contribution is a risk-controlled selective diagnosis framework. | v185 BAcc 99.81%, review/reject 58.94%, FN=1, FP=0. | supported |
| Stable release improves efficiency without increasing current released errors. | fixed v182 review/reject 56.08%, released errors 0; greedy v180 releases 3 errors. | supported as current-split/stability-audited candidate |
| Direction-aware routing improves over confidence-only routing under strict external shift. | strict external BAcc 80.59% vs 78.10% at comparable 30% review. | trend-level support |
| Automatic flipping is ready to replace review. | v174 and v175 both fail with substantial harm or poor transfer. | not supported |
| The final FN can be fixed by a simple non-leaky sentinel. | v190/v191 rescue 0 FN under nested evaluation. | not supported |
| Current automatic-decision safety needs confidence intervals. | v185 all-domain Wilson95 upper 1.95%; strict external upper 21.53%. | supported |
