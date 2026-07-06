# GrossPath-RC v1 冻结式工作流实验

日期：2026-05-26

## 本轮目的

v1 的目标是把 v0 的分散结果收敛成更正式的冻结式 workflow。我们只用旧数据+第三批开发集训练或选择策略，然后把同一套策略直接套到第三批 holdout 和严格外部集，不根据外部集分数反向调参。

本轮新增两件事：

1. 三路模型概率 stacking：用 `prob162_blend`、`DINOv3/VitL`、`QKVB` 的 OOF 概率训练轻量 stacking 分类头。
2. 可靠性 guard：用开发集 OOF 训练一个“这次主模型是否可靠”的评分器，再用开发集选择自动放行阈值。

当前 v1 自动选择的主分类器是 `base162`。可靠性 guard 的 OOF 正确性识别 AUC：`guard_plain=0.6367`，`guard_balanced=0.6467`。

## 强制分类模型对照

| group | model | threshold_from_dev | n | acc | bacc | auc | sens_high | spec_low | fn | fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev_all_old_plus_third | base162 | 0.5950 | 591 | 0.8748 | 0.8632 | 0.9011 | 0.8161 | 0.9103 | 41 | 33 |
| dev_all_old_plus_third | mean_core | 0.5795 | 591 | 0.8088 | 0.7855 | 0.8600 | 0.6906 | 0.8804 | 69 | 44 |
| dev_all_old_plus_third | stack_plain | 0.4550 | 591 | 0.8748 | 0.8632 | 0.8984 | 0.8161 | 0.9103 | 41 | 33 |
| dev_all_old_plus_third | stack_balanced | 0.5750 | 591 | 0.8748 | 0.8632 | 0.8975 | 0.8161 | 0.9103 | 41 | 33 |
| old | base162 | 0.5950 | 285 | 0.9228 | 0.9227 | 0.9489 | 0.9149 | 0.9306 | 12 | 10 |
| old | mean_core | 0.5795 | 285 | 0.8140 | 0.8135 | 0.8738 | 0.7660 | 0.8611 | 33 | 20 |
| old | stack_plain | 0.4550 | 285 | 0.9228 | 0.9227 | 0.9430 | 0.9149 | 0.9306 | 12 | 10 |
| old | stack_balanced | 0.5750 | 285 | 0.9228 | 0.9227 | 0.9431 | 0.9149 | 0.9306 | 12 | 10 |
| third_holdout234 | base162 | 0.5950 | 234 | 0.8675 | 0.7300 | 0.7303 | 0.5263 | 0.9337 | 18 | 13 |
| third_holdout234 | mean_core | 0.5795 | 234 | 0.8504 | 0.6986 | 0.7767 | 0.4737 | 0.9235 | 20 | 15 |
| third_holdout234 | stack_plain | 0.4550 | 234 | 0.8675 | 0.7300 | 0.7317 | 0.5263 | 0.9337 | 18 | 13 |
| third_holdout234 | stack_balanced | 0.5750 | 234 | 0.8675 | 0.7300 | 0.7260 | 0.5263 | 0.9337 | 18 | 13 |
| external_strict | base162 | 0.5950 | 105 | 0.6381 | 0.6220 | 0.6016 | 0.4681 | 0.7759 | 25 | 13 |
| external_strict | mean_core | 0.5795 | 105 | 0.6762 | 0.6766 | 0.6662 | 0.6809 | 0.6724 | 15 | 19 |
| external_strict | stack_plain | 0.4550 | 105 | 0.6190 | 0.6047 | 0.5976 | 0.4681 | 0.7414 | 25 | 15 |
| external_strict | stack_balanced | 0.5750 | 105 | 0.6286 | 0.6134 | 0.5965 | 0.4681 | 0.7586 | 25 | 14 |
| external_readable_auto | base162 | 0.5950 | 79 | 0.6582 | 0.6337 | 0.6245 | 0.4848 | 0.7826 | 17 | 10 |
| external_readable_auto | mean_core | 0.5795 | 79 | 0.6962 | 0.7006 | 0.6851 | 0.7273 | 0.6739 | 9 | 15 |
| external_readable_auto | stack_plain | 0.4550 | 79 | 0.6329 | 0.6120 | 0.6192 | 0.4848 | 0.7391 | 17 | 12 |
| external_readable_auto | stack_balanced | 0.5750 | 79 | 0.6456 | 0.6229 | 0.6186 | 0.4848 | 0.7609 | 17 | 11 |

## 开发集选择出的 guard 阈值

| spec_name | score_name | threshold | target_acc | max_auto_low_high_miss | selection_reason | auto_n | auto_coverage | auto_accuracy | auto_balanced_accuracy | auto_low_high_miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guard_plain_acc90_miss10 | guard_plain | 0.9130 | 0.9000 | 0.1000 | target_met | 268 | 0.4535 | 0.9179 | 0.9069 | 0.1000 |
| guard_plain_acc92_miss10 | guard_plain | 0.9231 | 0.9200 | 0.1000 | target_met | 231 | 0.3909 | 0.9221 | 0.9154 | 0.0929 |
| guard_plain_acc95_miss10 | guard_plain | 0.9261 | 0.9500 | 0.1000 | fallback_best_available | 221 | 0.3739 | 0.9321 | 0.9257 | 0.0821 |
| guard_balanced_acc90_miss10 | guard_balanced | 0.5157 | 0.9000 | 0.1000 | target_met | 366 | 0.6193 | 0.9153 | 0.8977 | 0.0980 |
| guard_margin_agree_acc90_miss10 | guard_margin_agree | 0.2877 | 0.9000 | 0.1000 | target_met | 508 | 0.8596 | 0.9016 | 0.8864 | 0.0923 |

## 开发集和第三批 holdout workflow

| group | policy | total_n | auto_n | review_n | retake_n | auto_coverage | auto_accuracy | auto_balanced_accuracy | auto_sensitivity_high | auto_specificity_low | auto_low_high_miss_rate | auto_high_ppv | review_error_rate_main | high_auto_low_missed | high_review_or_retake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev_all_old_plus_third | forced_main_all | 591 | 591 | 0 | 0 | 1.0000 | 0.8748 | 0.8632 | 0.8161 | 0.9103 | 0.1090 | 0.8465 |  | 41 | 0 |
| dev_all_old_plus_third | consensus_all3 | 591 | 381 | 210 | 0 | 0.6447 | 0.9055 | 0.8851 | 0.8217 | 0.9484 | 0.0878 | 0.8908 | 0.1810 | 23 | 94 |
| dev_all_old_plus_third | v1_guard_plain_acc90_miss10 | 591 | 268 | 323 | 0 | 0.4535 | 0.9179 | 0.9069 | 0.8455 | 0.9684 | 0.1000 | 0.9490 | 0.1610 | 17 | 113 |
| dev_all_old_plus_third | v1_guard_plain_acc92_miss10 | 591 | 231 | 360 | 0 | 0.3909 | 0.9221 | 0.9154 | 0.8687 | 0.9621 | 0.0929 | 0.9451 | 0.1556 | 13 | 124 |
| dev_all_old_plus_third | v1_guard_balanced_acc90_miss10 | 591 | 366 | 225 | 0 | 0.6193 | 0.9153 | 0.8977 | 0.8261 | 0.9693 | 0.0980 | 0.9421 | 0.1911 | 24 | 85 |
| dev_all_old_plus_third | v1_guard_margin_agree_acc90_miss10 | 591 | 508 | 83 | 0 | 0.8596 | 0.9016 | 0.8864 | 0.8315 | 0.9414 | 0.0923 | 0.8895 | 0.2892 | 31 | 39 |
| third_holdout234 | forced_main_all | 234 | 234 | 0 | 0 | 1.0000 | 0.8675 | 0.7300 | 0.5263 | 0.9337 | 0.0896 | 0.6061 |  | 18 | 0 |
| third_holdout234 | consensus_all3 | 234 | 176 | 58 | 0 | 0.7521 | 0.9091 | 0.7445 | 0.5217 | 0.9673 | 0.0692 | 0.7059 | 0.2586 | 11 | 15 |
| third_holdout234 | v1_guard_plain_acc90_miss10 | 234 | 117 | 117 | 0 | 0.5000 | 0.9060 | 0.6071 | 0.2143 | 1.0000 | 0.0965 | 1.0000 | 0.1709 | 11 | 24 |
| third_holdout234 | v1_guard_plain_acc92_miss10 | 234 | 101 | 133 | 0 | 0.4316 | 0.9010 | 0.6154 | 0.2308 | 1.0000 | 0.1020 | 1.0000 | 0.1579 | 10 | 25 |
| third_holdout234 | v1_guard_balanced_acc90_miss10 | 234 | 160 | 74 | 0 | 0.6838 | 0.9125 | 0.6111 | 0.2222 | 1.0000 | 0.0897 | 1.0000 | 0.2297 | 14 | 20 |
| third_holdout234 | v1_guard_margin_agree_acc90_miss10 | 234 | 204 | 30 | 0 | 0.8718 | 0.8873 | 0.6994 | 0.4333 | 0.9655 | 0.0919 | 0.6842 | 0.2667 | 17 | 8 |

## 外部集冻结评估 workflow

| group | policy | total_n | auto_n | review_n | retake_n | auto_coverage | auto_accuracy | auto_balanced_accuracy | auto_sensitivity_high | auto_specificity_low | auto_low_high_miss_rate | auto_high_ppv | review_error_rate_main | high_auto_low_missed | high_review_or_retake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external_strict | forced_main_all | 105 | 105 | 0 | 0 | 1.0000 | 0.6381 | 0.6220 | 0.4681 | 0.7759 | 0.3571 | 0.6286 |  | 25 | 0 |
| external_strict | consensus_all3 | 105 | 57 | 48 | 0 | 0.5429 | 0.7368 | 0.7377 | 0.7857 | 0.6897 | 0.2308 | 0.7097 | 0.4792 | 6 | 19 |
| external_strict | v1_guard_plain_acc90_miss10 | 105 | 26 | 79 | 0 | 0.2476 | 0.5769 | 0.5196 | 0.3333 | 0.7059 | 0.3333 | 0.3750 | 0.3418 | 6 | 38 |
| external_strict | v1_guard_plain_acc92_miss10 | 105 | 20 | 85 | 0 | 0.1905 | 0.5500 | 0.5208 | 0.3750 | 0.6667 | 0.3846 | 0.4286 | 0.3412 | 5 | 39 |
| external_strict | v1_guard_balanced_acc90_miss10 | 105 | 37 | 68 | 0 | 0.3524 | 0.5946 | 0.5288 | 0.3077 | 0.7500 | 0.3333 | 0.4000 | 0.3382 | 9 | 34 |
| external_strict | quality_v1_guard_plain_acc90_miss10 | 105 | 20 | 66 | 19 | 0.1905 | 0.6000 | 0.5238 | 0.3333 | 0.7143 | 0.2857 | 0.3333 | 0.3333 | 4 | 41 |
| external_strict | quality_v1_guard_balanced_acc90_miss10 | 105 | 28 | 58 | 19 | 0.2667 | 0.6071 | 0.5351 | 0.3333 | 0.7368 | 0.3000 | 0.3750 | 0.3276 | 6 | 38 |
| external_readable_auto | forced_main_all | 79 | 79 | 0 | 0 | 1.0000 | 0.6582 | 0.6337 | 0.4848 | 0.7826 | 0.3208 | 0.6154 |  | 17 | 0 |
| external_readable_auto | consensus_all3 | 79 | 41 | 38 | 0 | 0.5190 | 0.7805 | 0.7847 | 0.8421 | 0.7273 | 0.1579 | 0.7273 | 0.4737 | 3 | 14 |
| external_readable_auto | v1_guard_plain_acc90_miss10 | 79 | 21 | 58 | 0 | 0.2658 | 0.6190 | 0.5333 | 0.3333 | 0.7333 | 0.2667 | 0.3333 | 0.3276 | 4 | 27 |
| external_readable_auto | v1_guard_plain_acc92_miss10 | 79 | 16 | 63 | 0 | 0.2025 | 0.6250 | 0.5636 | 0.4000 | 0.7273 | 0.2727 | 0.4000 | 0.3333 | 3 | 28 |
| external_readable_auto | v1_guard_balanced_acc90_miss10 | 79 | 29 | 50 | 0 | 0.3671 | 0.6207 | 0.5417 | 0.3333 | 0.7500 | 0.2857 | 0.3750 | 0.3200 | 6 | 24 |
| external_readable_auto | quality_v1_guard_plain_acc90_miss10 | 79 | 21 | 58 | 0 | 0.2658 | 0.6190 | 0.5333 | 0.3333 | 0.7333 | 0.2667 | 0.3333 | 0.3276 | 4 | 27 |
| external_readable_auto | quality_v1_guard_balanced_acc90_miss10 | 79 | 29 | 50 | 0 | 0.3671 | 0.6207 | 0.5417 | 0.3333 | 0.7500 | 0.2857 | 0.3750 | 0.3200 | 6 | 24 |

## 阶段判断

1. stacking 分类头没有超过 base162。它在开发集和第三批 holdout 上与 base162 基本同分，在外部严格集反而略差，所以这一版不能作为主结果。
2. mean_core 在外部严格集的强制分类比 base162 更平衡，但它在开发集和第三批 holdout 明显下降。这个现象提示 DINOv3/QKVB 分支对外部高危更敏感，但不能直接替代主模型。
3. v1 guard 在开发集内能筛出 90% 以上准确率的自动子集，但冻结到外部集后明显失效。严格外部集上 `v1_guard_plain_acc90_miss10` 只有 0.5769 Acc，低于简单 `consensus_all3` 的 0.7368 Acc。
4. 当前正式可用的外部风险控制信号仍然是多模型一致性，而不是域内训练出来的可靠性评分器。也就是说，v1 guard 是一次有价值的否证实验，不应包装成提分结果。
5. 外部集的质量门控 policy 只使用图像质量判断，不使用真值；它适合写成“安全工作流”，不适合被包装成全量诊断准确率。

## 下一步

下一步不应该继续盲目调阈值，而应该做两个更硬的模块：一是图像质量/视角/尺度的域泛化训练，二是少数核心概念的图像蒸馏，特别是边界、包膜、结节/分叶、囊变坏死和主体尺度。v1 的结果说明，外部泛化问题不是靠域内置信度校准就能解决的。
