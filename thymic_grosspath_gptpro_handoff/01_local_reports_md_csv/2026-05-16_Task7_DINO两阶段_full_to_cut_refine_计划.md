# Task7 DINO 两阶段 full-to-cut refine 计划

时间：2026-05-16

## 一、背景

当前新数据上：

- `DINO 全学 concat` 仍然是整体最强主线
- `cut-only` 各种 specialist、换 backbone、路由分治都没有稳定超过基线
- 但从医学直觉上，切面图仍然是最值得深挖的一条信息线

因此下一步不再走“纯 cut 从头训练”，而改成两阶段：

1. 先用全量主图做 `Task7 full-view` 微调，学一个更贴合胸腺 gross 图的新 backbone
2. 再把这个 backbone 迁到 `cut-oriented` 子集继续精修，让模型在切面判别上进一步适配

## 二、核心假设

如果直接从 DINO 预训练起点训 `cut-only`，样本太少，容易不稳。

但如果先用：

- `combined_task567_registry.csv`
- `Task7`
- `whole`

做一轮 full-view 任务适配，再转到：

- `cut_oriented_task7_registry.csv`
- 相同 fold
- 相同 backbone/head

继续训练，可能同时保留：

- 全局胸腺 gross 表征稳定性
- 切面特异性判别能力

## 三、实验设计

### Stage 1：full-view smoke

已完成：

- 任务：`task7_lowrisk_vs_highrisk_tc`
- 输入：`whole`
- registry：`combined_task567_registry.csv`
- split：`combined_5fold_assignments.csv`
- backbone：`dinov2_vitb14`
- tune scope：`last2_blocks`
- head：`mlp`
- image size：`392`

输出目录：

- `outputs/batch1_batch2_task567_20260514/task7_dino_upgrade_runs/01_vitb14_fullview_last2_mlp_smoke_fold1`

### Stage 2：cut refine smoke

在 Stage 1 `fold1` 最优权重基础上继续训练：

- 初始化 checkpoint：
  - `.../01_vitb14_fullview_last2_mlp_smoke_fold1/fold_1/best_model.pt`
- 输入 registry：
  - `cut_oriented_task7_registry.csv`
- split：
  - `combined_5fold_assignments.csv`
- 任务：
  - `task7_lowrisk_vs_highrisk_tc`
- 输入：
  - `whole`
- backbone/head 配置保持一致：
  - `dinov2_vitb14`
  - `last2_blocks`
  - `mlp`

## 四、实施细节

为支持两阶段方案，需要在 `run_dinov2_finetune.py` 中加入：

- `--init-checkpoint`

用于把 Stage 1 的完整模型权重加载到 Stage 2，再继续 fine-tune。

## 五、评价口径

Stage 2 主要与同一 `cut-oriented` 子集上的统一基线对比：

- `Accuracy`
- `Balanced Accuracy`
- `AUC`
- `F1`

当前 cut 子集统一基线：

- `Accuracy = 0.8000`
- `Balanced Accuracy = 0.8235`
- `AUC = 0.8869`
- `F1 = 0.8125`

## 六、判定标准

如果 Stage 2 `fold1 smoke`：

- `Accuracy / Balanced Accuracy / F1` 明显高于 cut 子集统一基线中的至少两项
- 且 `AUC` 不明显下降

则扩成完整 `5-fold`。

否则：

- 这条 full-to-cut refine 暂不继续
- 回到其他 DINO 升级路线，如联合监督或 LoRA/adapter

