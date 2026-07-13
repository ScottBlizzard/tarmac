# Task7 Frequency Source-versus-Risk Audit Results (2026-07-13)

## Conclusion

Fixed Haar-frequency information is substantially more associated with
acquisition source than with Task7 risk after controlling the counterpart. The
strict preregistered binary source-dominance rule was not met because the
normalized cross-validated separability gap was 0.0820 rather than at least
0.10. That formal result is retained unchanged.

The supporting evidence is nevertheless directionally uniform: all 156 locked
frequency features had a larger source partial effect than risk partial effect,
source prediction was significant under stratified permutation, and risk
prediction was not. This audit provides no basis for a frequency classifier.

## Locked data and representation

- 591 internal cases, one selected photograph per case.
- Sources: batch 1, 117; batch 2, 168; third batch, 306.
- Risk: low, 368; high, 223.
- Deterministic whole-image and label-free specimen-crop views.
- Five-level Haar detail energies on Y, Cb, and Cr channels.
- 156 fixed features; no feature selection or wavelet-family search.

## Cross-validated diagnostic probes

| Diagnostic target | Controlled factor | BAcc | Chance | Normalized above chance | Stratified permutation p |
|---|---|---:|---:|---:|---:|
| Acquisition source | Risk | 0.4388 | 0.3333 | 0.1582 | 0.0099 |
| Task7 risk | Source | 0.5381 | 0.5000 | 0.0762 | 0.0792 |

Repeated-CV mean ranges across the 20 repeats were 0.4022-0.4739 for source and
0.5156-0.5564 for risk. The normalized source-minus-risk gap was 0.0820.

## Partial effects

| Statistic | Source controlling risk | Risk controlling source |
|---|---:|---:|
| Median partial eta-squared | 0.05617 | 0.00037 |
| 90th percentile | 0.08864 | 0.00298 |
| Maximum | 0.10665 | 0.01052 |

All 156/156 features had source partial eta-squared greater than risk partial
eta-squared. The strongest source effects were whole-image Cb/Cr detail energies
at levels 1-3. This pattern is consistent with color, compression, background,
focus, or device-era differences. It does not establish which acquisition
factor is causal.

## Preregistered decision

The locked rule required both:

1. normalized source-minus-risk separability at least 0.10;
2. median source partial eta-squared greater than median risk partial
   eta-squared.

Condition 1 failed narrowly; condition 2 passed strongly. Therefore the formal
binary rule is **not met**. It must not be changed after seeing the result.

## Project implication

Do not build or tune an RGB-frequency/wavelet risk classifier on this cohort.
The fixed audit indicates that this family has a high probability of exploiting
acquisition signatures. The prospective protocol should prioritize consistent
background, color reference, white balance, illumination, compression history,
native-file retention, and device-era metadata.

This result also helps explain H5: additional texture information can improve a
mixed-source split while failing when an acquisition source is held out.

## Reproducibility and privacy

Server-only outputs:

- `/workspace/thymic_project/experiments/frequency_source_vs_risk_audit_20260713/frequency_features.float32.npy`
- `/workspace/thymic_project/experiments/frequency_source_vs_risk_audit_20260713/frequency_feature_metadata.csv`
- `/workspace/thymic_project/experiments/frequency_source_vs_risk_audit_20260713/frequency_feature_effects.csv`
- `/workspace/thymic_project/experiments/frequency_source_vs_risk_audit_20260713/frequency_source_vs_risk_summary.json`

No image, case-level feature, metadata row, or prediction was downloaded. The
repository contains only code, preregistration, and aggregate interpretation.

- Executed script SHA-256:
  `b2f473ad67f14f4d566c01da4274c37e157e8c06c78f137481c61b0c1aeb7ac4`
- Aggregate summary SHA-256:
  `4b6be0f3471e65e48293912ca7ba47b937259fbc6e6785fcc7c8210de4598289`
