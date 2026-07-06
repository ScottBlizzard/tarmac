# GrossPath-RC v0 完整 workflow policy 实验

## 实验目标

把第一轮结果收敛成正式工作流：强制分类、模型一致性自动放行、质量门控、复核、重拍。

## 开发集策略对照

| group | policy | total_n | auto_n | review_n | retake_n | auto_coverage | auto_accuracy | auto_balanced_accuracy | auto_sensitivity_high | auto_specificity_low | auto_low_high_miss_rate | auto_high_ppv | review_error_rate_base162 | high_auto_low_missed | high_review_or_retake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev_all_old_plus_third | auto_if_all3_agree | 591 | 381 | 210 | 0 | 0.6447 | 0.9055 | 0.8851 | 0.8217 | 0.9484 | 0.0878 | 0.8908 | 0.1810 | 23 | 94 |
| dev_all_old_plus_third | auto_if_162_107_agree | 591 | 445 | 146 | 0 | 0.7530 | 0.8989 | 0.8883 | 0.8452 | 0.9314 | 0.0915 | 0.8820 | 0.1986 | 26 | 55 |
| dev_all_old_plus_third | auto_if_162_103_agree | 591 | 435 | 156 | 0 | 0.7360 | 0.8897 | 0.8663 | 0.7919 | 0.9406 | 0.1033 | 0.8741 | 0.1667 | 31 | 74 |
| dev_all_old_plus_third | forced_162_all | 591 | 591 | 0 | 0 | 1.0000 | 0.8748 | 0.8632 | 0.8161 | 0.9103 | 0.1090 | 0.8465 |  | 41 | 0 |
| dev_all_old_plus_third | forced_dinov3_all | 591 | 591 | 0 | 0 | 1.0000 | 0.6988 | 0.6742 | 0.5740 | 0.7745 | 0.2500 | 0.6066 |  | 95 | 0 |
| old | auto_if_162_107_agree | 285 | 190 | 95 | 0 | 0.6667 | 0.9474 | 0.9478 | 0.9406 | 0.9551 | 0.0659 | 0.9596 | 0.1263 | 6 | 40 |
| old | auto_if_all3_agree | 285 | 160 | 125 | 0 | 0.5614 | 0.9375 | 0.9375 | 0.9250 | 0.9500 | 0.0732 | 0.9487 | 0.0960 | 6 | 61 |
| old | auto_if_162_103_agree | 285 | 189 | 96 | 0 | 0.6632 | 0.9312 | 0.9311 | 0.9043 | 0.9579 | 0.0900 | 0.9551 | 0.0938 | 9 | 47 |
| old | forced_162_all | 285 | 285 | 0 | 0 | 1.0000 | 0.9228 | 0.9227 | 0.9149 | 0.9306 | 0.0822 | 0.9281 |  | 12 | 0 |
| old | forced_dinov3_all | 285 | 285 | 0 | 0 | 1.0000 | 0.6491 | 0.6489 | 0.6241 | 0.6736 | 0.3533 | 0.6519 |  | 53 | 0 |
| third_all | auto_if_all3_agree | 306 | 221 | 85 | 0 | 0.7222 | 0.8824 | 0.8004 | 0.6531 | 0.9477 | 0.0944 | 0.7805 | 0.3059 | 17 | 33 |
| third_all | auto_if_162_107_agree | 306 | 255 | 51 | 0 | 0.8333 | 0.8627 | 0.8109 | 0.7015 | 0.9202 | 0.1036 | 0.7581 | 0.3333 | 20 | 15 |
| third_all | auto_if_162_103_agree | 306 | 246 | 60 | 0 | 0.8039 | 0.8577 | 0.7660 | 0.6000 | 0.9319 | 0.1100 | 0.7174 | 0.2833 | 22 | 27 |
| third_all | forced_162_all | 306 | 306 | 0 | 0 | 1.0000 | 0.8301 | 0.7718 | 0.6463 | 0.8973 | 0.1261 | 0.6974 |  | 29 | 0 |
| third_all | forced_dinov3_all | 306 | 306 | 0 | 0 | 1.0000 | 0.7451 | 0.6635 | 0.4878 | 0.8393 | 0.1826 | 0.5263 |  | 42 | 0 |
| third_holdout234 | auto_if_all3_agree | 234 | 176 | 58 | 0 | 0.7521 | 0.9091 | 0.7445 | 0.5217 | 0.9673 | 0.0692 | 0.7059 | 0.2586 | 11 | 15 |
| third_holdout234 | auto_if_162_107_agree | 234 | 196 | 38 | 0 | 0.8376 | 0.8980 | 0.7759 | 0.6000 | 0.9518 | 0.0706 | 0.6923 | 0.2895 | 12 | 8 |
| third_holdout234 | auto_if_162_103_agree | 234 | 196 | 38 | 0 | 0.8376 | 0.8929 | 0.7200 | 0.4815 | 0.9586 | 0.0795 | 0.6500 | 0.2632 | 14 | 11 |
| third_holdout234 | forced_162_all | 234 | 234 | 0 | 0 | 1.0000 | 0.8675 | 0.7300 | 0.5263 | 0.9337 | 0.0896 | 0.6061 |  | 18 | 0 |
| third_holdout234 | forced_dinov3_all | 234 | 234 | 0 | 0 | 1.0000 | 0.7906 | 0.6523 | 0.4474 | 0.8571 | 0.1111 | 0.3778 |  | 21 | 0 |

## 外部严格集策略对照

| group | policy | total_n | auto_n | review_n | retake_n | auto_coverage | auto_accuracy | auto_balanced_accuracy | auto_sensitivity_high | auto_specificity_low | auto_low_high_miss_rate | auto_high_ppv | review_error_rate_base162 | high_auto_low_missed | high_review_or_retake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external_strict | quality_readable_162_103_agree | 105 | 45 | 41 | 19 | 0.4286 | 0.7778 | 0.7800 | 0.8000 | 0.7600 | 0.1739 | 0.7273 | 0.4878 | 4 | 27 |
| external_strict | quality_readable_all3_agree | 105 | 40 | 46 | 19 | 0.3810 | 0.7750 | 0.7782 | 0.8421 | 0.7143 | 0.1667 | 0.7273 | 0.4565 | 3 | 28 |
| external_strict | auto_if_all3_agree | 105 | 57 | 48 | 0 | 0.5429 | 0.7368 | 0.7377 | 0.7857 | 0.6897 | 0.2308 | 0.7097 | 0.4792 | 6 | 19 |
| external_strict | auto_if_162_103_agree | 105 | 65 | 40 | 0 | 0.6190 | 0.7231 | 0.7225 | 0.7097 | 0.7353 | 0.2647 | 0.7097 | 0.5000 | 9 | 16 |
| external_strict | auto_if_162_107_agree | 105 | 62 | 43 | 0 | 0.5905 | 0.6774 | 0.6792 | 0.7333 | 0.6250 | 0.2857 | 0.6471 | 0.4186 | 8 | 17 |
| external_strict | quality_readable_force_162 | 105 | 78 | 8 | 19 | 0.7429 | 0.6538 | 0.6313 | 0.4848 | 0.7778 | 0.3269 | 0.6154 | 0.3750 | 17 | 14 |
| external_strict | forced_162_all | 105 | 105 | 0 | 0 | 1.0000 | 0.6381 | 0.6220 | 0.4681 | 0.7759 | 0.3571 | 0.6286 |  | 25 | 0 |
| external_strict | forced_dinov3_all | 105 | 105 | 0 | 0 | 1.0000 | 0.6381 | 0.6543 | 0.8085 | 0.5000 | 0.2368 | 0.5672 |  | 9 | 0 |

## 外部 readable_auto 策略对照

| group | policy | total_n | auto_n | review_n | retake_n | auto_coverage | auto_accuracy | auto_balanced_accuracy | auto_sensitivity_high | auto_specificity_low | auto_low_high_miss_rate | auto_high_ppv | review_error_rate_base162 | high_auto_low_missed | high_review_or_retake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external_readable_auto | auto_if_162_103_agree | 79 | 46 | 33 | 0 | 0.5823 | 0.7826 | 0.7846 | 0.8000 | 0.7692 | 0.1667 | 0.7273 | 0.5152 | 4 | 13 |
| external_readable_auto | quality_readable_162_103_agree | 79 | 46 | 33 | 0 | 0.5823 | 0.7826 | 0.7846 | 0.8000 | 0.7692 | 0.1667 | 0.7273 | 0.5152 | 4 | 13 |
| external_readable_auto | auto_if_all3_agree | 79 | 41 | 38 | 0 | 0.5190 | 0.7805 | 0.7847 | 0.8421 | 0.7273 | 0.1579 | 0.7273 | 0.4737 | 3 | 14 |
| external_readable_auto | quality_readable_all3_agree | 79 | 41 | 38 | 0 | 0.5190 | 0.7805 | 0.7847 | 0.8421 | 0.7273 | 0.1579 | 0.7273 | 0.4737 | 3 | 14 |
| external_readable_auto | auto_if_162_107_agree | 79 | 46 | 33 | 0 | 0.5823 | 0.6957 | 0.7010 | 0.7619 | 0.6400 | 0.2381 | 0.6400 | 0.3939 | 5 | 12 |
| external_readable_auto | forced_dinov3_all | 79 | 79 | 0 | 0 | 1.0000 | 0.6709 | 0.7003 | 0.8788 | 0.5217 | 0.1429 | 0.5686 |  | 4 | 0 |
| external_readable_auto | forced_162_all | 79 | 79 | 0 | 0 | 1.0000 | 0.6582 | 0.6337 | 0.4848 | 0.7826 | 0.3208 | 0.6154 |  | 17 | 0 |
| external_readable_auto | quality_readable_force_162 | 79 | 79 | 0 | 0 | 1.0000 | 0.6582 | 0.6337 | 0.4848 | 0.7826 | 0.3208 | 0.6154 |  | 17 | 0 |

## 阶段结论

1. `forced_162_all` 仍是全量强制分类基线，但外部严格集 Acc 只有约 0.638。
2. 模型一致性是当前最稳定的风险控制信号，尤其是 `auto_if_all3_agree` 和 `auto_if_162_103_agree`。
3. 在外部 readable_auto 子集，`auto_if_all3_agree` / `auto_if_162_103_agree` 的自动准确率约 0.78，覆盖率约 0.52-0.58。
4. 质量门控 + consensus 是更接近临床工作流的策略，但会进一步降低全体覆盖率；它适合写成安全 workflow，而不是刷全量 accuracy。
5. 当前最适合继续强化的主线是 selective prediction / defer-to-review，而不是概念直接提分。