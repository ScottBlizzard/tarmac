# v94 Paper Main Assets

## Main Message

The current strongest paper storyline is a risk-controlled, batch-adaptive selective diagnosis framework rather than a single classifier. The primary deployable workflow is Batch-adaptive main; Fixed v79-light is the primary high-safety fixed workflow; Quality+direction uniform90 is a high-review upper-bound candidate.

## Claim Tiers

| claim | evidence | status |
| --- | --- | --- |
| Basic cross-domain workflow can be selected without external tuning | v93 strict-external-held-out selection chooses Fixed v50; held-out BAcc 97.30%. | main |
| High-safety workflow can also be selected without external tuning | v93 strict-external-held-out high-safety selection chooses Fixed v79-light; held-out BAcc 99.18%, FN=0. | main |
| Unlabeled batch-adaptive workflow is a deployment compromise | v91 Batch-adaptive main: all-domain control 74.11%, BAcc 98.24%; strict external BAcc 99.18%. | main |
| High-review upper-bound can nearly eliminate all-domain errors | v91 Quality+direction uniform90: all-domain BAcc 99.88%, FN=0, FP=1, control 89.84%. | candidate |
| Upper-bound improvement is promising but not primary | v92 vs v79-light: delta BAcc 1.04%, 95% CI 0.24% to 2.01%, McNemar p=0.070. | candidate, cautious |

## Main Result Table

| workflow | claim_tier | all_control_rate | all_bacc | all_fn | all_fp | strict_external_control_rate | strict_external_bacc | strict_external_fn | strict_external_fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure auto | Baseline | 0.00% | 82.16% | 51 | 72 | 0.00% | 70.11% | 15 | 17 |
| Fixed v50 | Stable baseline workflow | 72.68% | 97.94% | 8 | 5 | 73.15% | 97.30% | 1 | 2 |
| Batch-adaptive main | Primary deployable workflow | 74.11% | 98.24% | 7 | 4 | 82.41% | 99.18% | 0 | 1 |
| Fixed v79-light | Primary high-safety fixed workflow | 78.97% | 98.84% | 5 | 2 | 82.41% | 99.18% | 0 | 1 |
| Fixed v79-strict | High-specificity candidate | 82.55% | 99.07% | 5 | 0 | 87.04% | 100.00% | 0 | 0 |
| Quality+direction uniform90 | High-review upper-bound candidate | 89.84% | 99.88% | 0 | 1 | 89.81% | 99.18% | 0 | 1 |