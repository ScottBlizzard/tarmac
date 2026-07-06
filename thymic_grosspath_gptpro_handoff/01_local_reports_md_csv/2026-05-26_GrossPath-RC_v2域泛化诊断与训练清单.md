# GrossPath-RC v2 域泛化诊断与训练清单

日期：2026-05-26

## 本轮定位

v1 已经证明：单纯 stacking 和域内可靠性 guard 不能解决严格外部泛化。v2 先不调外部集分数，而是只用旧数据+第三批开发数据，把错误和图像质量、视角、尺度、核心概念联系起来，形成下一轮训练的可执行清单。

## 主模型表现总览

| group | n | acc | bacc | tn | fp | fn | tp | fn_rate_high | fp_rate_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev_all_old_plus_third | 591 | 0.8748 | 0.8632 | 335 | 33 | 41 | 182 | 0.1839 | 0.0897 |
| old | 285 | 0.9228 | 0.9227 | 134 | 10 | 12 | 129 | 0.0851 | 0.0694 |
| third_all | 306 | 0.8301 | 0.7718 | 201 | 23 | 29 | 53 | 0.3537 | 0.1027 |
| third_holdout234 | 234 | 0.8675 | 0.7300 | 183 | 13 | 18 | 20 | 0.4737 | 0.0663 |
| external_strict | 105 | 0.6381 | 0.6220 | 45 | 13 | 25 | 22 | 0.5319 | 0.2241 |
| external_readable_auto | 79 | 0.6582 | 0.6337 | 36 | 10 | 17 | 16 | 0.5152 | 0.2174 |

## 视角标签覆盖情况

当前可用的逐病例视角标签主要来自 2026-05-15 的多图/高危漏诊回看工作包，覆盖 58/591 例。它可以作为视角辅助头的种子标签，但还不能直接支持全量 cut/outer/mixed 分层结论。

| view_type_final | view_label_source | n |
| --- | --- | --- |
| unlabeled_third | missing | 306 |
| unlabeled_old | missing | 227 |
| mixed | p2_highrisk_fn | 21 |
| cut_surface | p2_highrisk_fn | 17 |
| cut_surface | p1_multiview | 14 |
| mixed | p1_multiview | 2 |
| outer_surface | p2_highrisk_fn | 2 |
| outer_surface | p1_multiview | 1 |
| unclear | p2_highrisk_fn | 1 |

## 开发集错误富集子群

下面表格按相对错误率排序。`relative_wrong_rate > 1` 表示这个子群比开发集平均更容易错。

| analysis | feature | value | wrong_rate | relative_wrong_rate | n | acc | bacc | tn | fp | fn | tp | fn_rate_high | fp_rate_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | third_split | adapt72 | 0.2917 | 2.3294 | 72 | 0.7083 | 0.6964 | 18 | 10 | 11 | 33 | 0.2500 | 0.3571 |
| development | brightness_mean_quartile | Q3 | 0.2286 | 1.8255 | 105 | 0.7714 | 0.7619 | 51 | 12 | 12 | 30 | 0.2857 | 0.1905 |
| development | task_l6_label | B2 | 0.2247 | 1.7947 | 89 | 0.7753 |  | 0 | 0 | 20 | 69 | 0.2247 |  |
| development | red_tissue_ratio_quartile | Q1_low | 0.1981 | 1.5822 | 106 | 0.8019 | 0.7330 | 69 | 10 | 11 | 16 | 0.4074 | 0.1266 |
| development | saturation_mean_quartile | Q3 | 0.1905 | 1.5212 | 105 | 0.8095 | 0.7913 | 58 | 10 | 10 | 27 | 0.2703 | 0.1471 |
| development | task_l7_label | high_risk_group | 0.1839 | 1.4684 | 223 | 0.8161 |  | 0 | 0 | 41 | 182 | 0.1839 |  |
| development | core_agree_all | 0 | 0.1810 | 1.4452 | 210 | 0.8190 | 0.8180 | 96 | 20 | 18 | 76 | 0.1915 | 0.1724 |
| development | core_agree_count | 2 | 0.1810 | 1.4452 | 210 | 0.8190 | 0.8180 | 96 | 20 | 18 | 76 | 0.1915 | 0.1724 |
| development | file_kb_quartile | Q3 | 0.1810 | 1.4452 | 105 | 0.8190 | 0.7685 | 66 | 8 | 11 | 20 | 0.3548 | 0.1081 |
| development | border_blue_ratio_quartile | Q2 | 0.1792 | 1.4315 | 106 | 0.8208 | 0.8155 | 59 | 12 | 7 | 28 | 0.2000 | 0.1690 |
| development | task_l6_label | TC | 0.1727 | 1.3795 | 110 | 0.8273 |  | 0 | 0 | 19 | 91 | 0.1727 |  |
| development | subject_area_proxy_quartile | Q3 | 0.1714 | 1.3691 | 105 | 0.8286 | 0.7832 | 64 | 5 | 13 | 23 | 0.3611 | 0.0725 |
| development | domain | third | 0.1699 | 1.3572 | 306 | 0.8301 | 0.7718 | 201 | 23 | 29 | 53 | 0.3537 | 0.1027 |
| development | view_type_final | unlabeled_third | 0.1699 | 1.3572 | 306 | 0.8301 | 0.7718 | 201 | 23 | 29 | 53 | 0.3537 | 0.1027 |
| development | file_kb_quartile | Q2 | 0.1698 | 1.3562 | 106 | 0.8302 | 0.8202 | 55 | 8 | 10 | 33 | 0.2326 | 0.1270 |
| development | border_blue_ratio_quartile | Q1_low | 0.1698 | 1.3562 | 106 | 0.8302 | 0.7972 | 63 | 7 | 11 | 25 | 0.3056 | 0.1000 |
| development | contrast_std_quartile | Q2 | 0.1604 | 1.2809 | 106 | 0.8396 | 0.8228 | 60 | 8 | 9 | 29 | 0.2368 | 0.1176 |
| development | saturation_mean_quartile | Q1_low | 0.1509 | 1.2055 | 106 | 0.8491 | 0.8339 | 61 | 8 | 8 | 29 | 0.2162 | 0.1159 |
| development | saturation_mean_quartile | Q2 | 0.1509 | 1.2055 | 106 | 0.8491 | 0.8059 | 70 | 8 | 8 | 20 | 0.2857 | 0.1026 |
| development | subject_area_proxy_quartile | Q1_low | 0.1509 | 1.2055 | 106 | 0.8491 | 0.8277 | 62 | 7 | 9 | 28 | 0.2432 | 0.1014 |
| development | contrast_std_quartile | Q1_low | 0.1415 | 1.1302 | 106 | 0.8585 | 0.7968 | 74 | 7 | 8 | 17 | 0.3200 | 0.0864 |
| development | contrast_std_quartile | Q4_high | 0.1415 | 1.1302 | 106 | 0.8585 | 0.8321 | 64 | 6 | 9 | 27 | 0.2500 | 0.0857 |
| development | red_tissue_ratio_quartile | Q2 | 0.1415 | 1.1302 | 106 | 0.8585 | 0.8366 | 66 | 8 | 7 | 25 | 0.2188 | 0.1081 |
| development | third_split | holdout234 | 0.1325 | 1.0580 | 234 | 0.8675 | 0.7300 | 183 | 13 | 18 | 20 | 0.4737 | 0.0663 |
| development | file_kb_quartile | Q1_low | 0.1321 | 1.0548 | 106 | 0.8679 | 0.8373 | 69 | 7 | 7 | 23 | 0.2333 | 0.0921 |

## 概念相关错误富集

这些结果只来自开发集已有肉眼所见/经验概念，不使用外部集真值调参。

| concept | value | n | acc | bacc | wrong_rate | relative_wrong_rate | fn | fp | fn_rate_high | fp_rate_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hemorrhage | present | 12 | 0.8333 | 0.8125 | 0.1667 | 1.3311 | 1 | 1 | 0.2500 | 0.1250 |
| cut_surface_mentioned | absent | 31 | 0.8387 | 0.7690 | 0.1613 | 1.2881 | 3 | 2 | 0.3750 | 0.0870 |
| surface_mentioned | present | 69 | 0.8551 | 0.8381 | 0.1449 | 1.1575 | 6 | 4 | 0.2308 | 0.0930 |
| gross_highrisk_score | Q3 | 142 | 0.8592 | 0.8496 | 0.1408 | 1.1249 | 14 | 6 | 0.2258 | 0.0750 |
| tumor_max_dim_mm | Q3 | 129 | 0.8605 | 0.8456 | 0.1395 | 1.1144 | 8 | 10 | 0.1951 | 0.1136 |
| boundary_unclear | present | 131 | 0.8626 | 0.8626 | 0.1374 | 1.0974 | 11 | 7 | 0.1375 | 0.1373 |
| boundary_clear | absent | 313 | 0.8626 | 0.8587 | 0.1374 | 1.0972 | 22 | 21 | 0.1667 | 0.1160 |
| tumor_max_dim_mm | Q4_high | 146 | 0.8630 | 0.8518 | 0.1370 | 1.0940 | 11 | 9 | 0.1964 | 0.1000 |
| nodular_lobulated | present | 327 | 0.8654 | 0.8433 | 0.1346 | 1.0746 | 22 | 22 | 0.2157 | 0.0978 |
| texture_soft | present | 195 | 0.8667 | 0.8434 | 0.1333 | 1.0649 | 14 | 12 | 0.2222 | 0.0909 |
| tumor_max_dim_mm | Q1_low | 161 | 0.8696 | 0.8571 | 0.1304 | 1.0417 | 14 | 7 | 0.2121 | 0.0737 |
| texture_medium | present | 428 | 0.8715 | 0.8620 | 0.1285 | 1.0263 | 29 | 26 | 0.1779 | 0.0981 |
| capsule_any | present | 430 | 0.8721 | 0.8501 | 0.1279 | 1.0215 | 27 | 28 | 0.2061 | 0.0936 |
| cystic_change | absent | 549 | 0.8725 | 0.8588 | 0.1275 | 1.0183 | 40 | 30 | 0.1951 | 0.0872 |
| texture_tough | absent | 518 | 0.8726 | 0.8569 | 0.1274 | 1.0176 | 36 | 30 | 0.1967 | 0.0896 |
| capsule_absent | absent | 559 | 0.8730 | 0.8594 | 0.1270 | 1.0144 | 38 | 33 | 0.1891 | 0.0922 |
| capsule_complete | present | 268 | 0.8731 | 0.8264 | 0.1269 | 1.0132 | 13 | 21 | 0.2500 | 0.0972 |
| gross_highrisk_score | Q2 | 150 | 0.8733 | 0.8543 | 0.1267 | 1.0116 | 7 | 12 | 0.1842 | 0.1071 |
| gross_conflict_score | present | 359 | 0.8747 | 0.8518 | 0.1253 | 1.0011 | 25 | 20 | 0.2137 | 0.0826 |
| necrosis | absent | 583 | 0.8748 | 0.8627 | 0.1252 | 1.0000 | 41 | 32 | 0.1864 | 0.0882 |
| capsule_involved | absent | 591 | 0.8748 | 0.8632 | 0.1252 | 1.0000 | 41 | 33 | 0.1839 | 0.0897 |
| gross_conflict_score | absent | 232 | 0.8750 | 0.8729 | 0.1250 | 0.9983 | 16 | 13 | 0.1509 | 0.1032 |
| hemorrhage | absent | 579 | 0.8756 | 0.8642 | 0.1244 | 0.9931 | 40 | 32 | 0.1826 | 0.0889 |
| capsule_complete | absent | 323 | 0.8762 | 0.8787 | 0.1238 | 0.9890 | 28 | 12 | 0.1637 | 0.0789 |
| cut_surface_mentioned | present | 560 | 0.8768 | 0.8667 | 0.1232 | 0.9840 | 38 | 31 | 0.1767 | 0.0899 |

## 外部集仅作诊断参考

外部表只用于描述分布和风险，不用于选择训练阈值。

| analysis | feature | value | wrong_rate | relative_wrong_rate | n | acc | bacc | tn | fp | fn | tp | fn_rate_high | fp_rate_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external_diagnostic_only | task_l6_label | B2 | 0.6429 | 1.8271 | 28 | 0.3571 |  | 0 | 0 | 18 | 10 | 0.6429 |  |
| external_diagnostic_only | lap_var_quartile | Q3 | 0.5769 | 1.6397 | 26 | 0.4231 | 0.4107 | 8 | 6 | 9 | 3 | 0.7500 | 0.4286 |
| external_diagnostic_only | task_l7_label | high_risk_group | 0.5319 | 1.5118 | 47 | 0.4681 |  | 0 | 0 | 25 | 22 | 0.5319 |  |
| external_diagnostic_only | core_agree_all | 0 | 0.4694 | 1.3340 | 49 | 0.5306 | 0.4333 | 26 | 4 | 19 | 0 | 1.0000 | 0.1333 |
| external_diagnostic_only | core_agree_count | 2 | 0.4694 | 1.3340 | 49 | 0.5306 | 0.4333 | 26 | 4 | 19 | 0 | 1.0000 | 0.1333 |
| external_diagnostic_only | fg_ratio_quartile | Q3 | 0.4615 | 1.3117 | 26 | 0.5385 | 0.4938 | 11 | 5 | 7 | 3 | 0.7000 | 0.3125 |
| external_diagnostic_only | fg_ratio_quartile | Q2 | 0.4444 | 1.2632 | 27 | 0.5556 | 0.5577 | 8 | 5 | 7 | 7 | 0.5000 | 0.3846 |
| external_diagnostic_only | manual_quality_status_v1 | reject_retake | 0.3810 | 1.0827 | 21 | 0.6190 | 0.5481 | 11 | 2 | 6 | 2 | 0.7500 | 0.1538 |
| external_diagnostic_only | manual_quality_status_v1 | borderline_review | 0.3750 | 1.0658 | 8 | 0.6250 | 0.5833 | 1 | 1 | 2 | 4 | 0.3333 | 0.5000 |
| external_diagnostic_only | bbox_area_ratio_quartile | Q1_low | 0.3704 | 1.0526 | 27 | 0.6296 | 0.6500 | 10 | 2 | 8 | 7 | 0.5333 | 0.1667 |
| external_diagnostic_only | bbox_area_ratio_quartile | Q2 | 0.3704 | 1.0526 | 27 | 0.6296 | 0.6417 | 9 | 3 | 7 | 8 | 0.4667 | 0.2500 |
| external_diagnostic_only | bbox_area_ratio_quartile | Q4_high | 0.3704 | 1.0526 | 27 | 0.6296 | 0.5556 | 14 | 4 | 6 | 3 | 0.6667 | 0.2222 |
| external_diagnostic_only | source_folder | 胸腺癌 | 0.3519 | 1.0000 | 108 | 0.6481 | 0.6275 | 48 | 13 | 25 | 22 | 0.5319 | 0.2131 |
| external_diagnostic_only | manual_quality_status_v1 | pass_readable | 0.3418 | 0.9714 | 79 | 0.6582 | 0.6337 | 36 | 10 | 17 | 16 | 0.5152 | 0.2174 |
| external_diagnostic_only | fg_ratio_quartile | Q1_low | 0.3333 | 0.9474 | 27 | 0.6667 | 0.6731 | 11 | 2 | 7 | 7 | 0.5000 | 0.1538 |
| external_diagnostic_only | bbox_area_ratio_quartile | Q3 | 0.3077 | 0.8745 | 26 | 0.6923 | 0.6090 | 15 | 4 | 4 | 3 | 0.5714 | 0.2105 |
| external_diagnostic_only | task_l6_label | B3 | 0.3000 | 0.8526 | 10 | 0.7000 |  | 0 | 0 | 3 | 7 | 0.3000 |  |
| external_diagnostic_only | lap_var_quartile | Q1_low | 0.2963 | 0.8421 | 27 | 0.7037 | 0.5833 | 17 | 1 | 7 | 2 | 0.7778 | 0.0556 |
| external_diagnostic_only | lap_var_quartile | Q4_high | 0.2963 | 0.8421 | 27 | 0.7037 | 0.7083 | 9 | 3 | 5 | 10 | 0.3333 | 0.2500 |
| external_diagnostic_only | task_l6_label | AB | 0.2821 | 0.8016 | 39 | 0.7179 |  | 28 | 11 | 0 | 0 |  | 0.2821 |

## 生成的训练清单

1. `v2_training_safe_core_cases.csv`：高一致性、主模型正确、边距较大的稳定样本，共 214 例。可作为稳定原型/teacher anchor。
2. `v2_training_hard_error_cases.csv`：开发集主模型错误样本，共 74 例。用于 hard mining、复核器训练、概念冲突回看。
3. `v2_training_domain_focus_cases.csv`：第三批、视角特殊、主体尺度异常等域泛化重点样本，共 384 例。用于颜色/尺度/背景增强和域泛化训练。

## v2 后续训练建议

1. 先做不依赖外部集的增强实验：颜色温度、亮度、饱和度、背景蓝板扰动、主体面积随机缩放、whole+crop 多尺度一致性。
2. 加入 view/quality 辅助头：视角不一定要求医生级精确，但要让 backbone 显式知道 cut、outer、mixed、主体过小、背景异常这些域因素。
3. 核心概念只保留少数高价值项：边界、包膜、结节/分叶、囊变/坏死/出血、主体尺度。概念数量少但要能被图像预测，否则不要进入主模型。
4. 训练目标从单一 CE 改成多目标：主任务分类 + 视角/质量辅助 + 概念辅助 + 多增强一致性。这样比继续堆阈值更可能提升外部泛化。
5. 外部严格集只能在模型冻结后评估一次；如果要分析外部失败原因，只能作为事后报告，不能反向调策略。
