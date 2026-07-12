# Task7 B1 Fixed-Route M0-M4 Results

Stage 1 is locked C1. The router uses the outer-training 40th percentile of absolute C1 logit margin. M2 receives image tokens and tile metadata only; M3 is fit from inner-crossfit M2 predictions.

## Overall full-coverage metrics

| Split | Model | BAcc | AUC | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| fivefold | m0_c1 | 0.7477 | 0.8240 | 0.7399 | 0.7554 |
| fivefold | m1_behavior | 0.7473 | 0.8212 | 0.7309 | 0.7636 |
| fivefold | m2_image | 0.7537 | 0.7964 | 0.7085 | 0.7989 |
| fivefold | m3_fusion | 0.7599 | 0.8333 | 0.7399 | 0.7799 |
| source_lodo | m0_c1 | 0.7397 | 0.8072 | 0.7130 | 0.7663 |
| source_lodo | m1_behavior | 0.7365 | 0.8048 | 0.7040 | 0.7690 |
| source_lodo | m2_image | 0.6993 | 0.7180 | 0.6323 | 0.7663 |
| source_lodo | m3_fusion | 0.7187 | 0.8050 | 0.6547 | 0.7826 |

## Same-routed-case tests

| Split | Comparison | Delta BAcc [95% CI] | Rescue/Harm |
| --- | --- | ---: | ---: |
| fivefold | M2-M0 | 0.0165 [-0.0658, 0.1002] | 53/44 |
| fivefold | M3-M1 | 0.0305 [-0.0230, 0.0821] | 24/16 |
| source_lodo | M2-M0 | -0.0928 [-0.1686, -0.0182] | 45/63 |
| source_lodo | M3-M1 | -0.0415 [-0.1067, 0.0242] | 31/37 |

## Router stability

- `fivefold`: fold 1 22/123 (17.9%), fold 2 42/122 (34.4%), fold 3 75/118 (63.6%), fold 4 44/116 (37.9%), fold 5 62/112 (55.4%).
- `source_lodo`: batch1 34/117 (29.1%), batch2 53/168 (31.5%), third batch 175/306 (57.2%).

The actual five-fold route produced +0.0060 full-cohort BAcc, versus +0.0240 for the mean of 1,000 matched random routes and +0.0362 at their 95th percentile. Under source LODO, the actual route produced -0.0404 versus a matched-random mean of -0.0133. The actual route did not exceed matched random routing in either analysis.

## Boundary effects

| Split | Subtype | M0 accuracy | M2 accuracy | Delta |
| --- | --- | ---: | ---: | ---: |
| fivefold | B1 | 0.4355 | 0.6452 | +0.2097 |
| fivefold | B2 | 0.6404 | 0.6067 | -0.0337 |
| source_lodo | B1 | 0.4194 | 0.5484 | +0.1290 |
| source_lodo | B2 | 0.6292 | 0.5056 | -0.1236 |

## Decision

**NO-GO.** Only 1/3 held sources had a positive full-cohort M2 change. The routed image-only gain was small and uncertain in five-fold OOF, reversed significantly under source LODO, and traded B2 performance for B1. The fixed-grid confidence-routed cascade is closed.
