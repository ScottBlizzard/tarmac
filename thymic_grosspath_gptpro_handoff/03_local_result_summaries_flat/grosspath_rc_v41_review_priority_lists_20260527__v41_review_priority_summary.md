# 2026-05-27 v41 外部集复核优先级清单

目的：把 v37/v40 的 rank-normalized risk controller 转成可操作的病例优先级。这里的清单不用于训练，只用于解释风险门控排在前面的病例是否确实集中在关键边界错例。

## Top-K 错例富集

| target_dev_bacc | top_k | top_k_rate | p2_wrong_n | p2_wrong_precision | fn_high_to_low_n | fp_low_to_high_n | ab_n | b2_or_b2b3_n | borderline_quality_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9500 | 5 | 0.0463 | 2 | 0.4000 | 0 | 2 | 0 | 2 | 0 |
| 0.9500 | 10 | 0.0926 | 3 | 0.3000 | 1 | 2 | 2 | 5 | 0 |
| 0.9500 | 20 | 0.1852 | 8 | 0.4000 | 3 | 5 | 5 | 8 | 7 |
| 0.9500 | 30 | 0.2778 | 14 | 0.4667 | 4 | 10 | 11 | 10 | 16 |
| 0.9500 | 40 | 0.3704 | 18 | 0.4500 | 6 | 12 | 17 | 13 | 22 |
| 0.9500 | 50 | 0.4630 | 22 | 0.4400 | 9 | 13 | 21 | 16 | 25 |
| 0.9500 | 60 | 0.5556 | 24 | 0.4000 | 10 | 14 | 25 | 20 | 33 |
| 0.9500 | 108 | 1.0000 | 32 | 0.2963 | 15 | 17 | 39 | 31 | 66 |
| 0.9700 | 5 | 0.0463 | 2 | 0.4000 | 0 | 2 | 0 | 2 | 0 |
| 0.9700 | 10 | 0.0926 | 3 | 0.3000 | 1 | 2 | 2 | 5 | 0 |
| 0.9700 | 20 | 0.1852 | 8 | 0.4000 | 3 | 5 | 5 | 8 | 7 |
| 0.9700 | 30 | 0.2778 | 14 | 0.4667 | 4 | 10 | 11 | 10 | 16 |
| 0.9700 | 40 | 0.3704 | 18 | 0.4500 | 6 | 12 | 17 | 13 | 22 |
| 0.9700 | 50 | 0.4630 | 22 | 0.4400 | 9 | 13 | 21 | 16 | 25 |
| 0.9700 | 60 | 0.5556 | 24 | 0.4000 | 10 | 14 | 25 | 20 | 33 |
| 0.9700 | 108 | 1.0000 | 32 | 0.2963 | 15 | 17 | 39 | 31 | 66 |

## 95%目标复核池前 15 例

| priority_rank | original_case_id | image_name | auto_prediction | quality_status | quality_score | hard_risk_rank_pct_external | review_reason | task_l6_label | task_l7_label | truth_label_group | p2_wrong | p2_error_direction | route_bucket | main_prob | robust_prob | prob_mean_core | hard_risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725547 | B2型胸腺瘤--1725547.jpg | 高危 | pass | 100.0000 | 0.0093 | 主模型与鲁棒分支分歧较大 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3004 | 0.6155 | 0.6833 | 0.0000 |
| 2 | 1711115 | A型胸腺瘤--1711115.jpg | 高危 | pass | 100.0000 | 0.0185 | 主模型与鲁棒分支分歧较大 | A | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.3507 | 0.6338 | 0.6922 | 0.0000 |
| 3 | 1726670 | 伴有淋巴样间质的微结节型胸腺瘤--1726670.jpg | 低危 | pass | 100.0000 | 0.0278 | 综合风险排序靠前 | MNT_assumed_low | low_risk_group | 低危 | 0 | correct | review_on_p2_correct | 0.3340 | 0.4913 | 0.5396 | 0.0000 |
| 4 | 1906481 | A型胸腺瘤--1906481.jpg | 高危 | pass | 100.0000 | 0.0370 | 主模型与鲁棒分支分歧较大 | A | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.3271 | 0.6464 | 0.7153 | 0.0000 |
| 5 | 1905040 | 胸腺瘤（B2型）--1905040.jpg | 高危 | pass | 100.0000 | 0.0463 | 主模型与鲁棒分支分歧较大 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3331 | 0.6068 | 0.6682 | 0.0000 |
| 6 | 1904148 | B2型胸腺瘤--1904148.jpg | 高危 | pass | 100.0000 | 0.0556 | 主模型与鲁棒分支分歧较大 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3222 | 0.6202 | 0.6827 | 0.0000 |
| 7 | 1605273 | B2型胸腺瘤--1605273.jpg | 高危 | pass | 100.0000 | 0.0648 | 综合风险排序靠前 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.6642 | 0.8277 | 0.8615 | 0.0000 |
| 8 | 1821773 | AB型胸腺瘤--1821773.jpg | 低危 | pass | 100.0000 | 0.0741 | 综合风险排序靠前 | AB | low_risk_group | 低危 | 0 | correct | review_on_p2_correct | 0.3129 | 0.5050 | 0.5424 | 0.0000 |
| 9 | 1805322 | AB型胸腺瘤--1805322.jpg | 低危 | pass | 100.0000 | 0.0833 | 综合风险排序靠前 | AB | low_risk_group | 低危 | 0 | correct | review_on_p2_correct | 0.2552 | 0.3599 | 0.3705 | 0.0000 |
| 10 | 1709499 | B2型胸腺瘤--1709499.jpg | 低危 | pass | 100.0000 | 0.0926 | 综合风险排序靠前 | B2 | high_risk_group | 高危 | 1 | FN_high_to_low | captured_p2_error | 0.1547 | 0.2107 | 0.2419 | 0.0000 |
| 11 | 1700034 | AB型胸腺瘤--1700034.jpg | 高危 | borderline | 96.0000 | 0.1019 | 质量/构图边界 | AB | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.5938 | 0.5850 | 0.5767 | 0.0000 |
| 12 | 1603320 | B2型胸腺瘤--1603320.jpg | 高危 | borderline | 86.0000 | 0.1111 | 质量/构图边界 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.5107 | 0.6649 | 0.7045 | 0.0000 |
| 13 | 1824525 | B3型胸腺瘤--1824525.jpg | 高危 | pass | 100.0000 | 0.1204 | 主模型与鲁棒分支分歧较大 | B3 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3276 | 0.6351 | 0.6970 | 0.0000 |
| 14 | 1724549 | B1型胸腺瘤--1724549.jpg | 高危 | pass | 100.0000 | 0.1296 | 主模型与鲁棒分支分歧较大 | B1 | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.5174 | 0.7509 | 0.7999 | 0.0000 |
| 15 | 1826764 | AB型胸腺瘤--1826764.jpg | 高危 | borderline | 92.0000 | 0.1389 | 质量/构图边界 | AB | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.8135 | 0.8548 | 0.8644 | 0.0000 |

## 97%目标复核池前 15 例

| priority_rank | original_case_id | image_name | auto_prediction | quality_status | quality_score | hard_risk_rank_pct_external | review_reason | task_l6_label | task_l7_label | truth_label_group | p2_wrong | p2_error_direction | route_bucket | main_prob | robust_prob | prob_mean_core | hard_risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725547 | B2型胸腺瘤--1725547.jpg | 高危 | pass | 100.0000 | 0.0093 | 主模型与鲁棒分支分歧较大 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3004 | 0.6155 | 0.6833 | 0.0000 |
| 2 | 1711115 | A型胸腺瘤--1711115.jpg | 高危 | pass | 100.0000 | 0.0185 | 主模型与鲁棒分支分歧较大 | A | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.3507 | 0.6338 | 0.6922 | 0.0000 |
| 3 | 1726670 | 伴有淋巴样间质的微结节型胸腺瘤--1726670.jpg | 低危 | pass | 100.0000 | 0.0278 | 综合风险排序靠前 | MNT_assumed_low | low_risk_group | 低危 | 0 | correct | review_on_p2_correct | 0.3340 | 0.4913 | 0.5396 | 0.0000 |
| 4 | 1906481 | A型胸腺瘤--1906481.jpg | 高危 | pass | 100.0000 | 0.0370 | 主模型与鲁棒分支分歧较大 | A | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.3271 | 0.6464 | 0.7153 | 0.0000 |
| 5 | 1905040 | 胸腺瘤（B2型）--1905040.jpg | 高危 | pass | 100.0000 | 0.0463 | 主模型与鲁棒分支分歧较大 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3331 | 0.6068 | 0.6682 | 0.0000 |
| 6 | 1904148 | B2型胸腺瘤--1904148.jpg | 高危 | pass | 100.0000 | 0.0556 | 主模型与鲁棒分支分歧较大 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3222 | 0.6202 | 0.6827 | 0.0000 |
| 7 | 1605273 | B2型胸腺瘤--1605273.jpg | 高危 | pass | 100.0000 | 0.0648 | 综合风险排序靠前 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.6642 | 0.8277 | 0.8615 | 0.0000 |
| 8 | 1821773 | AB型胸腺瘤--1821773.jpg | 低危 | pass | 100.0000 | 0.0741 | 综合风险排序靠前 | AB | low_risk_group | 低危 | 0 | correct | review_on_p2_correct | 0.3129 | 0.5050 | 0.5424 | 0.0000 |
| 9 | 1805322 | AB型胸腺瘤--1805322.jpg | 低危 | pass | 100.0000 | 0.0833 | 综合风险排序靠前 | AB | low_risk_group | 低危 | 0 | correct | review_on_p2_correct | 0.2552 | 0.3599 | 0.3705 | 0.0000 |
| 10 | 1709499 | B2型胸腺瘤--1709499.jpg | 低危 | pass | 100.0000 | 0.0926 | 综合风险排序靠前 | B2 | high_risk_group | 高危 | 1 | FN_high_to_low | captured_p2_error | 0.1547 | 0.2107 | 0.2419 | 0.0000 |
| 11 | 1700034 | AB型胸腺瘤--1700034.jpg | 高危 | borderline | 96.0000 | 0.1019 | 质量/构图边界 | AB | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.5938 | 0.5850 | 0.5767 | 0.0000 |
| 12 | 1603320 | B2型胸腺瘤--1603320.jpg | 高危 | borderline | 86.0000 | 0.1111 | 质量/构图边界 | B2 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.5107 | 0.6649 | 0.7045 | 0.0000 |
| 13 | 1824525 | B3型胸腺瘤--1824525.jpg | 高危 | pass | 100.0000 | 0.1204 | 主模型与鲁棒分支分歧较大 | B3 | high_risk_group | 高危 | 0 | correct | review_on_p2_correct | 0.3276 | 0.6351 | 0.6970 | 0.0000 |
| 14 | 1724549 | B1型胸腺瘤--1724549.jpg | 高危 | pass | 100.0000 | 0.1296 | 主模型与鲁棒分支分歧较大 | B1 | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.5174 | 0.7509 | 0.7999 | 0.0000 |
| 15 | 1826764 | AB型胸腺瘤--1826764.jpg | 高危 | borderline | 92.0000 | 0.1389 | 质量/构图边界 | AB | low_risk_group | 低危 | 1 | FP_low_to_high | captured_p2_error | 0.8135 | 0.8548 | 0.8644 | 0.0000 |

## 当前判断

风险排序前列并不是随机病例，主要富集在 AB、B2/B2-B3 以及质量/构图边界病例。该清单可以直接转成医生复核优先级：先看排序靠前的病例，判断是拍照/取材问题、真实边界病例，还是模型应学习的稳定模式。