# Task7 DINO 双 Backbone 两阶段升级计划

时间：2026-05-16

## 目标

把当前最强但不可训练的统一基线：

- `DINOv2 vits14 + vitb14 concat`

改造成一个真正可训练的双 backbone 模型，然后验证：

1. `full-view` 先训练，是否能学到比单 backbone 更强的统一表征；
2. 在此基础上再做 `cut` 子集精修，是否能超过当前 `cut` 子集统一基线；
3. 是否比现有单 backbone 两阶段 `full -> cut refine` 更稳。

## 背景判断

目前已有结论：

- 普通 `timm` backbone（`ConvNeXt / Swin / EfficientNet`）无论 `cut-only` 还是 `full-view`，都没打过当前 `DINO` 主线；
- 单 backbone `DINO` 两阶段 `full -> cut refine`，AUC 略优，但 `Accuracy / BAcc / F1` 还没超过 `cut` 子集统一基线；
- 当前最强统一基线本质上是“冻结双 backbone 特征 + 轻量分类器”，并不是一个能直接继续 fine-tune 的模型。

所以这次升级的核心是：

- 把“最强统一基线的结构”本身变成可训练模型；
- 再测试 `full-view -> cut refine` 是否终于能把 `cut` 路线抬起来。

## 模型结构

### Stage 1：双 backbone full-view 训练

- backbone A：`dinov2_vits14`
- backbone B：`dinov2_vitb14`
- 特征：两个 backbone 的 `CLS` token 直接拼接
- head：`MLP`
- tune scope：`last2_blocks`

### Stage 2：双 backbone cut refine

- 用 Stage 1 对应 fold 的 `best_model.pt` 初始化
- 数据切到 `cut_oriented_task7_registry.csv`
- 继续训练，学习率更小，epoch 更短

## 训练策略

### Stage 1

- task：`Task7 low/high`
- input：`whole`
- image size：`392`
- batch size：`4`
- head lr：`1e-3`
- backbone lr：`1e-5`
- epochs：`30`
- patience：`8`

### Stage 2

- task：`Task7 low/high`
- input：`whole`
- image size：`392`
- batch size：`4`
- head lr：`5e-4`
- backbone lr：`5e-6`
- epochs：`20`
- patience：`6`

## 评估口径

### 第一步

先跑 `fold1 smoke`，同时看：

- `Stage1 full-view`
- `Stage2 full -> cut refine`

对比对象：

- 当前 `Task7 full-view DINO concat` 基线
- 当前 `cut` 子集统一基线
- 当前单 backbone 两阶段 `full -> cut refine`

### 第二步

如果 `fold1` 至少满足下面两条中的一条，就扩完整 `5-fold`：

1. `Stage2` 在 `cut` 子集上的 `Accuracy / BAcc / F1` 任一项超过当前基线；
2. `Stage2` 的 `AUC` 明显更高，且 `Accuracy / BAcc / F1` 基本追平。

## 成功标准

这条线的成功标准不是单纯 “AUC 变高”，而是：

- `cut` 子集上的 `Accuracy / BAcc / F1` 至少有一项实质超过当前统一基线；
- 或者在几乎不掉 `Accuracy` 的前提下，显著减少高危误判。

## 停损标准

满足任一条件就降级：

1. `fold1 smoke` 明显弱于当前 `cut` 子集基线；
2. 比单 backbone 两阶段也没有改进；
3. 显存或训练稳定性明显不可接受。

## 当前执行顺序

1. 实现 `run_dinov2_dual_finetune.py`
2. 跑 `fold1 smoke`
3. 判断是否扩完整 `5-fold`
