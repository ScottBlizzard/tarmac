# GrossPath-RC v0 实验小结

输出目录：`D:\影响分析\outputs\grosspath_rc_v0_20260526`

## 数据覆盖

- 结构化肉眼所见病例数：1104
- 开发集病例数：591，概念匹配率：0.997
- 外部压力测试病例数：108，概念匹配率：0.000
- `prob_mean_core` 阈值只在开发集选择：0.495，开发集 BAcc：0.7855

## 强制分类最佳结果

| model | n | threshold | accuracy | balanced_accuracy | f1 | auc | sensitivity_high | specificity_low | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prob162_blend__old | 285 | 0.5950 | 0.9228 | 0.9227 | 0.9214 | 0.9489 | 0.9149 | 0.9306 | 134 | 10 | 12 | 129 |
| prob162_blend__dev_all_old_plus_third | 591 | 0.5950 | 0.8748 | 0.8632 | 0.8311 | 0.9011 | 0.8161 | 0.9103 | 335 | 33 | 41 | 182 |
| prob162_blend__concept_matched | 589 | 0.5950 | 0.8744 | 0.8627 | 0.8303 | 0.9006 | 0.8153 | 0.9101 | 334 | 33 | 41 | 181 |
| prob_mean_core__old | 285 | 0.4950 | 0.7860 | 0.7865 | 0.7946 | 0.8738 | 0.8369 | 0.7361 | 106 | 38 | 23 | 118 |
| prob_mean_core__dev_all_old_plus_third | 591 | 0.4950 | 0.7868 | 0.7855 | 0.7342 | 0.8600 | 0.7803 | 0.7908 | 291 | 77 | 49 | 174 |
| prob_mean_core__concept_matched | 589 | 0.4950 | 0.7861 | 0.7847 | 0.7331 | 0.8596 | 0.7793 | 0.7902 | 290 | 77 | 49 | 173 |
| prob162_blend__third_all | 306 | 0.5950 | 0.8301 | 0.7718 | 0.6709 | 0.8137 | 0.6463 | 0.8973 | 201 | 23 | 29 | 53 |
| prob_mean_core__third_all | 306 | 0.4950 | 0.7876 | 0.7544 | 0.6328 | 0.8183 | 0.6829 | 0.8259 | 185 | 39 | 26 | 56 |

## 外部压力测试强制分类最佳结果

| model | n | threshold | accuracy | balanced_accuracy | f1 | auc | sensitivity_high | specificity_low | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prob103_vitl__external_readable_auto | 79 | 0.5000 | 0.6709 | 0.7003 | 0.6905 | 0.6864 | 0.8788 | 0.5217 | 24 | 22 | 4 | 29 |
| prob103_vitl__external_all | 108 | 0.5000 | 0.6389 | 0.6584 | 0.6609 | 0.6732 | 0.8085 | 0.5082 | 31 | 30 | 9 | 38 |
| prob103_vitl__external_strict | 105 | 0.5000 | 0.6381 | 0.6543 | 0.6667 | 0.6651 | 0.8085 | 0.5000 | 29 | 29 | 9 | 38 |
| prob162_blend__external_readable_auto | 79 | 0.5950 | 0.6582 | 0.6337 | 0.5424 | 0.6245 | 0.4848 | 0.7826 | 36 | 10 | 17 | 16 |
| prob162_blend__external_all | 108 | 0.5950 | 0.6481 | 0.6275 | 0.5366 | 0.6052 | 0.4681 | 0.7869 | 48 | 13 | 25 | 22 |
| prob162_blend__external_strict | 105 | 0.5950 | 0.6381 | 0.6220 | 0.5366 | 0.6016 | 0.4681 | 0.7759 | 45 | 13 | 25 | 22 |
| prob107_qkvb__external_readable_auto | 79 | 0.5000 | 0.5696 | 0.6090 | 0.6222 | 0.6370 | 0.8485 | 0.3696 | 17 | 29 | 5 | 28 |
| prob107_qkvb__external_all | 108 | 0.5000 | 0.5741 | 0.6034 | 0.6290 | 0.6446 | 0.8298 | 0.3770 | 23 | 38 | 8 | 39 |

## 概念信号与模型融合

| model | n | threshold | accuracy | balanced_accuracy | f1 | auc | sensitivity_high | specificity_low | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| concept_oracle | 591 | 0.6200 | 0.7056 | 0.6708 | 0.5756 | 0.7155 | 0.5291 | 0.8125 | 299 | 69 | 105 | 118 |
| model_probs | 591 | 0.5750 | 0.8731 | 0.8619 | 0.8292 | 0.8975 | 0.8161 | 0.9076 | 334 | 34 | 41 | 182 |
| model_plus_concepts_oracle | 591 | 0.5950 | 0.8663 | 0.8529 | 0.8184 | 0.9021 | 0.7982 | 0.9076 | 334 | 34 | 45 | 178 |

## 错误/复核识别

| router | n | error_rate | error_auc | error_ap | top20pct_error_enrichment |
| --- | --- | --- | --- | --- | --- |
| uncertainty_disagreement | 591 | 0.1252 | 0.6467 | 0.2188 | 1.8274 |
| uncertainty_disagreement_plus_concepts | 591 | 0.1252 | 0.5425 | 0.1577 | 1.3536 |

## 外部压力测试上的错误风险识别

| router | external_error_rate | external_error_auc | external_error_ap |
| --- | --- | --- | --- |
| uncertainty_disagreement | 0.3519 | 0.4515 | 0.3221 |
| uncertainty_disagreement_plus_concepts | 0.3519 | 0.4650 | 0.3321 |

## Consensus 自动放行策略

开发集：

| group | policy | coverage | auto_n | review_n | auto_accuracy | auto_bacc | auto_sensitivity_high | auto_specificity_low | review_error_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old | auto_if_162_107_agree | 0.6667 | 190 | 95 | 0.9474 | 0.9478 | 0.9406 | 0.9551 | 0.1263 | 85 | 4 | 6 | 95 |
| old | auto_if_all3_agree | 0.5614 | 160 | 125 | 0.9375 | 0.9375 | 0.9250 | 0.9500 | 0.0960 | 76 | 4 | 6 | 74 |
| old | auto_if_162_103_agree | 0.6632 | 189 | 96 | 0.9312 | 0.9311 | 0.9043 | 0.9579 | 0.0938 | 91 | 4 | 9 | 85 |
| old | auto_if_103_107_agree | 0.7930 | 226 | 59 | 0.9159 | 0.9159 | 0.9151 | 0.9167 | 0.0508 | 110 | 10 | 9 | 97 |
| dev_all_old_plus_third | auto_if_162_107_agree | 0.7530 | 445 | 146 | 0.8989 | 0.8883 | 0.8452 | 0.9314 | 0.1986 | 258 | 19 | 26 | 142 |
| concept_matched | auto_if_162_107_agree | 0.7521 | 443 | 146 | 0.8984 | 0.8877 | 0.8443 | 0.9312 | 0.1986 | 257 | 19 | 26 | 141 |
| dev_all_old_plus_third | auto_if_all3_agree | 0.6447 | 381 | 210 | 0.9055 | 0.8851 | 0.8217 | 0.9484 | 0.1810 | 239 | 13 | 23 | 106 |
| concept_matched | auto_if_all3_agree | 0.6452 | 380 | 209 | 0.9053 | 0.8850 | 0.8217 | 0.9482 | 0.1818 | 238 | 13 | 23 | 106 |
| dev_all_old_plus_third | auto_if_103_107_agree | 0.8003 | 473 | 118 | 0.8879 | 0.8713 | 0.8171 | 0.9256 | 0.1780 | 286 | 23 | 30 | 134 |
| concept_matched | auto_if_103_107_agree | 0.8014 | 472 | 117 | 0.8877 | 0.8712 | 0.8171 | 0.9253 | 0.1795 | 285 | 23 | 30 | 134 |

外部压力测试：

| group | policy | coverage | auto_n | review_n | auto_accuracy | auto_bacc | auto_sensitivity_high | auto_specificity_low | review_error_rate | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external_readable_auto | auto_if_all3_agree | 0.5190 | 41 | 38 | 0.7805 | 0.7847 | 0.8421 | 0.7273 | 0.4737 | 16 | 6 | 3 | 16 |
| external_readable_auto | auto_if_162_103_agree | 0.5823 | 46 | 33 | 0.7826 | 0.7846 | 0.8000 | 0.7692 | 0.5152 | 20 | 6 | 4 | 16 |
| external_all | auto_if_all3_agree | 0.5463 | 59 | 49 | 0.7458 | 0.7477 | 0.7857 | 0.7097 | 0.4694 | 22 | 9 | 6 | 22 |
| external_strict | auto_if_all3_agree | 0.5429 | 57 | 48 | 0.7368 | 0.7377 | 0.7857 | 0.6897 | 0.4792 | 20 | 9 | 6 | 22 |
| external_all | auto_if_162_103_agree | 0.6204 | 67 | 41 | 0.7313 | 0.7298 | 0.7097 | 0.7500 | 0.4878 | 27 | 9 | 9 | 22 |
| external_strict | auto_if_162_103_agree | 0.6190 | 65 | 40 | 0.7231 | 0.7225 | 0.7097 | 0.7353 | 0.5000 | 25 | 9 | 9 | 22 |
| external_readable_auto | auto_if_162_107_agree | 0.5823 | 46 | 33 | 0.6957 | 0.7010 | 0.7619 | 0.6400 | 0.3939 | 16 | 9 | 5 | 16 |
| external_all | auto_if_162_107_agree | 0.5926 | 64 | 44 | 0.6875 | 0.6902 | 0.7333 | 0.6471 | 0.4091 | 22 | 12 | 8 | 22 |
| external_strict | auto_if_162_107_agree | 0.5905 | 62 | 43 | 0.6774 | 0.6792 | 0.7333 | 0.6250 | 0.4186 | 20 | 12 | 8 | 22 |
| external_readable_auto | auto_if_103_107_agree | 0.8734 | 69 | 10 | 0.6957 | 0.6769 | 0.5333 | 0.8205 | 0.6000 | 32 | 7 | 14 | 16 |

## 说明

- `concept_oracle` 使用医生肉眼所见结构化概念，属于上限/机制验证，不是部署时直接输入。
- 外部集结果只作为已暴露压力测试，不用于阈值回调。
- v0 的重点是判断概念、分歧、不确定性是否能支撑风险控制主线。