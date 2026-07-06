# v186 Final Framework Table

## Recommended Deployment Workflow

- Recommended framework: v185 unlabeled shift-adaptive workflow, all-domain BAcc 99.81%, review/reject 58.94%, FN=1, FP=0.
- Standard efficiency workflow: fixed v182, BAcc 99.81%, review/reject 56.08%.
- High-safety fallback: fixed v118, BAcc 99.81%, review/reject 79.97%.

## Strict External Behavior

- On strict external, adaptive policy falls back to v118: review/reject 87.04%; fixed v182 would review 68.52%; fixed v118 reviews 87.04%.
- Current strict external BAcc is 100.00% for the adaptive branch. This should be described as current-split validation, not definitive external generalization.

## Gate Definition

- If an unlabeled batch is flagged as severe external shift, the system falls back from v182 stable release to v118 high-safety review.

## Paper Boundary

The main computational claim should be a risk-controlled, shift-adaptive selective diagnosis framework. Automatic flipping remains negative evidence; the framework's strength is stable release plus deployment-time rejection tightening.