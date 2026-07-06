# Task7 Paper-ready Summary Pack (v86)

## Primary Fixed Workflows
| domain | workflow | policy_id | control_rate | balanced_accuracy | balanced_accuracy_wilson_95ci | sensitivity | sensitivity_wilson_95ci | specificity | specificity_wilson_95ci | fn | fp | remaining_errors | claim_tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_data | Standard risk-control workflow (v50) | v50_main | 72.63% | 99.65% | 96.76%-99.94% | 100.00% | 97.35%-100.00% | 99.31% | 96.17%-99.88% | 0 | 1 | 1 | main |
| old_data | High-risk miss protection (v75) | v75_quality_lowconf | 74.74% | 99.65% | 96.76%-99.94% | 100.00% | 97.35%-100.00% | 99.31% | 96.17%-99.88% | 0 | 1 | 1 | candidate |
| old_data | Bidirectional light guard (v79-light) | v79_light_lowrisk_guard | 77.19% | 99.65% | 96.76%-99.94% | 100.00% | 97.35%-100.00% | 99.31% | 96.17%-99.88% | 0 | 1 | 1 | main |
| old_data | Bidirectional strict guard (v79-strict) | v79_strict_lowrisk_guard | 82.11% | 100.00% | 97.37%-100.00% | 100.00% | 97.35%-100.00% | 100.00% | 97.40%-100.00% | 0 | 0 | 0 | candidate |
| third_batch | Standard risk-control workflow (v50) | v50_main | 72.55% | 95.29% | 90.11%-97.78% | 91.46% | 83.41%-95.80% | 99.11% | 96.80%-99.75% | 7 | 2 | 9 | main |
| third_batch | High-risk miss protection (v75) | v75_quality_lowconf | 76.80% | 96.50% | 91.66%-98.56% | 93.90% | 86.51%-97.37% | 99.11% | 96.80%-99.75% | 5 | 2 | 7 | candidate |
| third_batch | Bidirectional light guard (v79-light) | v79_light_lowrisk_guard | 79.41% | 96.95% | 92.41%-98.68% | 93.90% | 86.51%-97.37% | 100.00% | 98.31%-100.00% | 5 | 0 | 5 | main |
| third_batch | Bidirectional strict guard (v79-strict) | v79_strict_lowrisk_guard | 81.37% | 96.95% | 92.41%-98.68% | 93.90% | 86.51%-97.37% | 100.00% | 98.31%-100.00% | 5 | 0 | 5 | candidate |
| strict_external | Standard risk-control workflow (v50) | v50_main | 73.15% | 97.30% | 88.85%-99.36% | 97.87% | 88.89%-99.62% | 96.72% | 88.81%-99.10% | 1 | 2 | 3 | main |
| strict_external | High-risk miss protection (v75) | v75_quality_lowconf | 79.63% | 98.36% | 90.63%-99.55% | 100.00% | 92.44%-100.00% | 96.72% | 88.81%-99.10% | 0 | 2 | 2 | candidate |
| strict_external | Bidirectional light guard (v79-light) | v79_light_lowrisk_guard | 82.41% | 99.18% | 91.86%-99.86% | 100.00% | 92.44%-100.00% | 98.36% | 91.28%-99.71% | 0 | 1 | 1 | main |
| strict_external | Bidirectional strict guard (v79-strict) | v79_strict_lowrisk_guard | 87.04% | 100.00% | 93.26%-100.00% | 100.00% | 92.44%-100.00% | 100.00% | 94.08%-100.00% | 0 | 0 | 0 | candidate |

## Strict External Focus
| domain | workflow | policy_id | control_rate | balanced_accuracy | balanced_accuracy_wilson_95ci | sensitivity | sensitivity_wilson_95ci | specificity | specificity_wilson_95ci | fn | fp | remaining_errors | claim_tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_external | Standard risk-control workflow (v50) | v50_main | 73.15% | 97.30% | 88.85%-99.36% | 97.87% | 88.89%-99.62% | 96.72% | 88.81%-99.10% | 1 | 2 | 3 | main |
| strict_external | High-risk miss protection (v75) | v75_quality_lowconf | 79.63% | 98.36% | 90.63%-99.55% | 100.00% | 92.44%-100.00% | 96.72% | 88.81%-99.10% | 0 | 2 | 2 | candidate |
| strict_external | Bidirectional light guard (v79-light) | v79_light_lowrisk_guard | 82.41% | 99.18% | 91.86%-99.86% | 100.00% | 92.44%-100.00% | 98.36% | 91.28%-99.71% | 0 | 1 | 1 | main |
| strict_external | Bidirectional strict guard (v79-strict) | v79_strict_lowrisk_guard | 87.04% | 100.00% | 93.26%-100.00% | 100.00% | 92.44%-100.00% | 100.00% | 94.08%-100.00% | 0 | 0 | 0 | candidate |

## Unlabeled Batch Audit
| audit_name | reference | target | domain_auc_cv | quality_proxy_mean | batch_shift_index | shift_category | recommended_policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pseudo_old_to_third | old_data | third_batch | 0.5169705309024195 | 0.0219499023231179 | 0.5232008994137931 | within_internal_shift | v50_main |
| pseudo_third_to_old | third_batch | old_data | 0.5317165462676299 | 0.0110613962923415 | 0.9694861165212963 | within_internal_shift | v50_main |
| strict_external_vs_dev | development_all | strict_external | 0.9945008460236888 | 0.3060960602052423 | 2.3241836209922586 | severe_shift | v75_quality_lowconf |

## Adaptive Workflows
| workflow | policy_id | overall_control_rate | overall_balanced_accuracy | overall_sensitivity | overall_specificity | fn | fp | remaining_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard risk-control workflow (v50) | v50_main | 72.68% | 97.94% | 97.04% | 98.83% | 8 | 5 | 13 |
| High-risk miss protection (v75) | v75_quality_lowconf | 76.39% | 98.49% | 98.15% | 98.83% | 5 | 5 | 10 |
| Bidirectional light guard (v79-light) | v79_light_lowrisk_guard | 78.97% | 98.84% | 98.15% | 99.53% | 5 | 2 | 7 |
| Bidirectional strict guard (v79-strict) | v79_strict_lowrisk_guard | 82.55% | 99.07% | 98.15% | 100.00% | 5 | 0 | 5 |
| Adaptive v50->v75 | adaptive_v75_on_shift | 73.68% | 98.12% | 97.41% | 98.83% | 7 | 5 | 12 |
| Adaptive v50->v79-light | adaptive_light_on_shift | 74.11% | 98.24% | 97.41% | 99.07% | 7 | 4 | 11 |
| Adaptive v50->v79-strict | adaptive_strict_on_shift | 74.82% | 98.35% | 97.41% | 99.30% | 7 | 3 | 10 |

## Module Ablation
| domain | module_label | delta_control_rate | delta_bacc | delta_sensitivity | delta_specificity | delta_fn | delta_fp | delta_remaining_error_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_data | High-risk miss protection | +2.11 pp | +0.00 pp | +0.00 pp | +0.00 pp | 0 | 0 | 0 |
| old_data | Light low-risk overcall guard | +2.46 pp | +0.00 pp | +0.00 pp | +0.00 pp | 0 | 0 | 0 |
| old_data | Strict low-risk overcall guard | +4.91 pp | +0.35 pp | +0.00 pp | +0.69 pp | 0 | -1 | -1 |
| old_data | Full fixed safety workflow | +9.47 pp | +0.35 pp | +0.00 pp | +0.69 pp | 0 | -1 | -1 |
| third_batch | High-risk miss protection | +4.25 pp | +1.22 pp | +2.44 pp | +0.00 pp | -2 | 0 | -2 |
| third_batch | Light low-risk overcall guard | +2.61 pp | +0.45 pp | +0.00 pp | +0.89 pp | 0 | -2 | -2 |
| third_batch | Strict low-risk overcall guard | +1.96 pp | +0.00 pp | +0.00 pp | +0.00 pp | 0 | 0 | 0 |
| third_batch | Full fixed safety workflow | +8.82 pp | +1.67 pp | +2.44 pp | +0.89 pp | -2 | -2 | -4 |
| strict_external | High-risk miss protection | +6.48 pp | +1.06 pp | +2.13 pp | +0.00 pp | -1 | 0 | -1 |
| strict_external | Light low-risk overcall guard | +2.78 pp | +0.82 pp | +0.00 pp | +1.64 pp | 0 | -1 | -1 |
| strict_external | Strict low-risk overcall guard | +4.63 pp | +0.82 pp | +0.00 pp | +1.64 pp | 0 | -1 | -1 |
| strict_external | Full fixed safety workflow | +13.89 pp | +2.70 pp | +2.13 pp | +3.28 pp | -1 | -2 | -3 |

## Paired Delta Statistics
| domain | comparison | delta_bacc | delta_bacc_bootstrap_95ci | delta_fn | delta_fp | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- |
| old_data | Light low-risk guard vs v75 | +0.00 pp | +0.00 to +0.00 pp | 0 | 0 | 1.000 |
| old_data | Light full workflow vs v50 | +0.00 pp | +0.00 to +0.00 pp | 0 | 0 | 1.000 |
| old_data | Strict full workflow vs v50 | +0.35 pp | +0.00 to +1.04 pp | 0 | -1 | 1.000 |
| third_batch | Light low-risk guard vs v75 | +0.45 pp | +0.00 to +1.12 pp | 0 | -2 | 0.500 |
| third_batch | Light full workflow vs v50 | +1.67 pp | +0.22 to +3.72 pp | -2 | -2 | 0.125 |
| third_batch | Strict full workflow vs v50 | +1.67 pp | +0.22 to +3.72 pp | -2 | -2 | 0.125 |
| strict_external | Light low-risk guard vs v75 | +0.82 pp | +0.00 to +2.46 pp | 0 | -1 | 1.000 |
| strict_external | Light full workflow vs v50 | +1.88 pp | +0.00 to +4.83 pp | -1 | -1 | 0.500 |
| strict_external | Strict full workflow vs v50 | +2.70 pp | +0.00 to +6.14 pp | -1 | -2 | 0.250 |

## Decision Utility Named Scenarios
| scenario | rank | policy | policy_label | utility | gain_vs_v50 | control_rate | bacc | fn | fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_low_review_cost | 1 | v79_strict_lowrisk_guard | Fixed v79 strict | -0.0522746781115879 | 0.0266380543633762 | 0.8254649499284692 | 0.9907407407407408 | 5 | 0 |
| balanced_medium_review_cost | 1 | v79_light_lowrisk_guard | Fixed v79 light | -0.1175965665236051 | 0.0194563662374821 | 0.7896995708154506 | 0.9884097384097384 | 5 | 2 |
| high_sensitivity_low_review_cost | 1 | v79_strict_lowrisk_guard | Fixed v79 strict | -0.1595708154506437 | 0.0910157367668097 | 0.8254649499284692 | 0.9907407407407408 | 5 | 0 |
| high_sensitivity_medium_review_cost | 1 | v79_light_lowrisk_guard | Fixed v79 light | -0.2248927038626609 | 0.0838340486409156 | 0.7896995708154506 | 0.9884097384097384 | 5 | 2 |
| avoid_overcall_low_review_cost | 1 | v79_strict_lowrisk_guard | Fixed v79 strict | -0.0522746781115879 | 0.0552503576537911 | 0.8254649499284692 | 0.9907407407407408 | 5 | 0 |
| avoid_overcall_medium_review_cost | 1 | v79_strict_lowrisk_guard | Fixed v79 strict | -0.1183118741058655 | 0.0473533619456366 | 0.8254649499284692 | 0.9907407407407408 | 5 | 0 |
| review_expensive | 1 | v75_quality_lowconf | Fixed v75 | -0.2314735336194563 | 0.0354792560801144 | 0.7639484978540773 | 0.9849132349132348 | 5 | 5 |

## Claim Boundary
- Main reportable workflow: v50 baseline and v79-light bidirectional guard.
- High-safety candidate: v79-strict, because strict external reaches 100% but still needs more external batches.
- Deployment-oriented framework: v82 adaptive routing, because it uses unlabeled batch shift audit before choosing the safety tier.
- Exploratory-only results remain excluded from the primary table if thresholds were chosen after inspecting strict external performance.