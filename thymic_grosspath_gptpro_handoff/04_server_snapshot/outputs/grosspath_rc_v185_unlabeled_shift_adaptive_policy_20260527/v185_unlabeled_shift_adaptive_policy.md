# v185 Unlabeled Shift-adaptive Policy

## Rule

Use v182 stable fixed release on within-internal-shift batches. If the unlabeled severe-shift gate flags a batch as strict external-like, fall back to v118 high-safety review for that batch.

## Result

- Fixed v182: all-domain BAcc 99.81%, review 56.08%.
- Adaptive v182->v118: all-domain BAcc 99.81%, review 58.94%.
- Fixed v118: all-domain BAcc 99.81%, review 79.97%.
- Strict external branch: adaptive uses v118, review 87.04% vs fixed v182 review 68.52%; both BAcc 100.00% in the current split.

## Boundary

This experiment validates the deployment-control logic rather than proving that v118 is always superior on severe-shift data. In the current strict external split, both v182 and v118 are correct; the adaptive policy is more conservative when the unlabeled gate detects severe shift.