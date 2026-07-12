# Task7 A1 Native-Detail Pilot Results

All values use 100% coverage and threshold 0.5. Fixed families were reported separately; no pooled-OOF winner replacement was performed. Medium/local slots below 60% tissue coverage were masked from aggregation.

## Overall metrics

| Model | Split | BAcc | AUC | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| locked_c1 | fivefold | 0.7477 | 0.8240 | 0.7399 | 0.7554 |
| c1_hier_mil | fivefold | 0.7472 | 0.8152 | 0.7444 | 0.7500 |
| native_hier_mil | fivefold | 0.7435 | 0.7931 | 0.7668 | 0.7201 |
| native_cross_attention | fivefold | 0.6899 | 0.7429 | 0.7085 | 0.6712 |
| locked_c1 | source_lodo | 0.7397 | 0.8072 | 0.7130 | 0.7663 |
| c1_hier_mil | source_lodo | 0.7235 | 0.7952 | 0.6861 | 0.7609 |
| native_hier_mil | source_lodo | 0.6897 | 0.7413 | 0.6457 | 0.7337 |
| native_cross_attention | source_lodo | 0.6755 | 0.7205 | 0.5874 | 0.7636 |

## Paired comparisons

| Split | Baseline | Candidate | Delta BAcc [95% CI] | Delta AUC [95% CI] | Rescue/Harm |
| --- | --- | --- | ---: | ---: | ---: |
| fivefold | c1_hier_mil | native_hier_mil | -0.0037 [-0.0415, 0.0351] | -0.0221 [-0.0573, 0.0123] | 61/67 |
| fivefold | c1_hier_mil | native_cross_attention | -0.0573 [-0.0982, -0.0165] | -0.0723 [-0.1124, -0.0331] | 54/91 |
| fivefold | locked_c1 | native_hier_mil | -0.0042 [-0.0442, 0.0367] | -0.0309 [-0.0668, 0.0034] | 59/66 |
| fivefold | locked_c1 | native_cross_attention | -0.0578 [-0.0979, -0.0173] | -0.0811 [-0.1178, -0.0443] | 58/96 |
| source_lodo | c1_hier_mil | native_hier_mil | -0.0338 [-0.0721, 0.0048] | -0.0539 [-0.0886, -0.0206] | 56/75 |
| source_lodo | c1_hier_mil | native_cross_attention | -0.0480 [-0.0871, -0.0085] | -0.0747 [-0.1158, -0.0339] | 57/78 |
| source_lodo | locked_c1 | native_hier_mil | -0.0499 [-0.0887, -0.0120] | -0.0659 [-0.1005, -0.0315] | 53/80 |
| source_lodo | locked_c1 | native_cross_attention | -0.0641 [-0.1038, -0.0232] | -0.0867 [-0.1253, -0.0473] | 61/90 |

## Decision

- `native_hier_mil`: **NO-GO**; positive LODO sources 0/3.
- `native_cross_attention`: **NO-GO**; positive LODO sources 0/3.

The fixed-grid native-tile family is closed after its two prespecified architectures. This does not show that local morphology is absent; it shows that deterministic specimen-grid localization does not provide transferable incremental evidence.
