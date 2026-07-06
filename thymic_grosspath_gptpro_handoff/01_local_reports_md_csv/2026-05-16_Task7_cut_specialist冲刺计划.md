# Task7 cut specialist 冲刺计划

## 一、当前判断

围绕新数据的 `Task7` 主线，我们已经确认：

- `outer_surface` 数量很少，不是当前整体错误的主要来源。
- `cut_surface` 数量充足，但准确率仍然不高，说明问题不只是“有没有切面图”，而是模型在切面图上的判别能力还不够强。
- `mixed` 里最难的是 `balanced_mixed`，但即使把 `mixed` 做了拆分，当前轻量 specialist 版本也没有超过统一基线。

因此下一阶段的重点不再是继续打磨轻量路由，而是：

1. 先把 `cut specialist` 真正做强。
2. 先看它在 `cut-oriented` 子集上能不能明显超过当前统一模型。
3. 如果这一步成立，再考虑把它接回完整 `Task7` 主线。

## 二、主目标

### 主目标

- 在 `cut-oriented` 图像子集上提升 `Task7` 二分类能力。
- 判断“更强的 cut specialist”能否明显超过当前统一 `Task7 concat` 基线。

### 次目标

- 判断 cut specialist 是否值得继续扩展到完整 `Task7`。
- 为后续 `Task6 + Task7` 联合训练做准备。

## 三、数据定义

### 1. cut-oriented 训练子集

当前使用：

- `cut_surface`
- `cut_heavy`

不纳入：

- `outer_surface`
- `outer_heavy`
- `balanced_mixed`
- `unclear`

### 2. 当前 cut-oriented 子集规模

基于现有自动视图标签：

- 总病例数：`188`
- 其中：
  - `cut_surface = 140`
  - `cut_heavy = 48`
- `Task7` 分布：
  - `low_risk_group = 95`
  - `high_risk_group = 93`

这是一个比较干净、而且高低危近似平衡的子集，适合先检验模型能力上限。

## 四、实验路线

### P0. 训练入口修复

先解决 stronger smoke 的训练入口问题：

- 当前 `run_dinov2_finetune.py` 在 `image_size > 400` 时会因为 `Resize(400,400) + RandomCrop(image_size)` 报错。
- 先修成：
  - 训练时 `Resize(max(400, image_size))`
  - 再做 `RandomCrop(image_size)`

目标：

- 保持旧实验不受影响。
- 让 `392 / 448 / 518` 这类更高分辨率可以直接进入训练。

### P1. stronger cut specialist smoke

先做真正的 stronger smoke：

- 任务：`Task7`
- 数据：`cut-oriented registry`
- backbone：`DINOv2 vitb14`
- 输入：`whole`
- 训练方式：`partial fine-tuning`
- tune scope：优先 `last2_blocks`
- head：`mlp`
- selection metric：`accuracy`

建议起始配置：

- `image_size = 392`
- `epochs = 25`
- `batch_size = 8`
- `label_smoothing = 0.05`
- `patience = 8`

### P2. stronger cut specialist 对照

把 stronger smoke 和以下对象对照：

1. 当前统一 `Task7 concat` 基线在同一 `cut-oriented` 子集上的表现
2. 之前轻量 specialist 版本在同一子集上的表现

主看：

- `accuracy`
- `balanced accuracy`
- `AUC`
- `F1`

### P3. full 5-fold

如果 `fold1 smoke` 明显优于当前统一基线，再扩成完整 `5-fold`。

通过标准：

- `accuracy` 明显高于统一基线
- `balanced accuracy` 不下降
- `AUC` 保持提升或至少不掉

### P4. 第二轮模型增强

如果第一版 stronger smoke 仍不够，再依次试：

1. `image_size` 从 `392` 提到 `448`
2. `vitl14` 替代 `vitb14`
3. `head_type` 从 `mlp` 再调深一点
4. `tune_scope` 从 `last2_blocks` 扩到 `full`

## 五、优先级

当前优先级固定为：

1. 修训练入口
2. 跑 `cut specialist fold1 smoke`
3. 做同口径对照
4. 决定是否上 `5-fold`

当前不优先：

- 再继续打磨轻量 view routing
- 继续把主要精力放在 `outer specialist`
- 先做复杂联合训练

## 六、判定逻辑

### 如果 stronger smoke 仍然不强

说明当前问题可能已经不只是视图拆分，而是：

- `Task7` 的类别边界本身很难
- 仅靠 partial fine-tuning 还不够
- 需要进一步引入更强监督，或回到 `Task6 + Task7` 联动

### 如果 stronger smoke 明显变强

说明当前判断成立：

- 切面图确实需要一套更强的专门模型
- 统一模型虽然稳，但没有把切面图的信息上限真正打出来

## 七、当前执行顺序

现在按下面顺序直接执行：

1. 修 `run_dinov2_finetune.py / data.py` 的高分辨率训练入口
2. 用 `cut_oriented_task7_registry.csv` 跑 `fold1 smoke`
3. 计算同一子集上的统一基线指标
4. 决定是否上完整 `5-fold`
