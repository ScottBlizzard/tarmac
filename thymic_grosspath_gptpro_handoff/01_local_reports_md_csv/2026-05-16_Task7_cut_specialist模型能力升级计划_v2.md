# Task7 cut specialist 模型能力升级计划 v2

一、背景

前一轮结果说明两件事：

1. `cut_surface + cut_heavy` 这批图本身并非没有信息，但当前 stronger cut specialist 还没有稳定超过统一基线。
2. 真正方案 B、`Task6 -> Task7` 联动版等路线在完整 `5-fold` 上都没有站住。

因此当前阶段不再优先改路由，而是直接验证一个更核心的假设：

> 不是切面图无效，而是当前模型能力还不够，尚未把切面图中的细粒度结构信息充分学出来。

二、目标

在同一 `cut-oriented` 子集上，直接提升 cut specialist 的模型能力，优先观察：

- `Accuracy`
- `Balanced Accuracy`
- `AUC`
- `F1`

先做 `fold1 smoke`，只要有明确正信号，再扩到完整 `5-fold`。

三、固定数据范围

- 数据：`cut_surface + cut_heavy`
- 病例数：`188`
- 任务：`Task7 low risk vs high risk`
- 输入图像：`selected_images`

四、本轮不再改动的东西

- 不再继续拆 `mixed`
- 不再继续调 `view-aware routing`
- 不再继续改 `outer specialist`

本轮唯一主轴：提升 `cut` 这条支路本身的模型能力。

五、要试的能力升级项

P0-1. 更高分辨率

- 现有 stronger 版用 `image_size = 392`
- 切面图更可能受益于更细的内部纹理与异质性信息
- 本轮提升到 `518`

P0-2. 更大 backbone

- 现有 stronger 版主试的是 `DINOv2 vitb14`
- 本轮加入 `DINOv2 vitl14`

P0-3. 保持可比

- 仍然使用：
  - `last2_blocks`
  - `mlp head`
  - `label_smoothing = 0.05`
  - `25 epochs`
  - `batch_size = 8`

六、执行顺序

1. `vitb14 + 518 + last2 + mlp` 跑 `fold1 smoke`
2. `vitl14 + 518 + last2 + mlp` 跑 `fold1 smoke`
3. 和当前同一 cut 子集上的统一基线比较
4. 如果其中一条在 `accuracy / bacc / f1` 上明显优于基线，再扩成完整 `5-fold`

七、成功标准

至少满足以下一条，才升完整 `5-fold`：

1. `accuracy` 明显高于 cut 子集统一基线
2. `balanced_accuracy` 与 `f1` 同时不弱于基线，且 `auc` 更高

八、停损标准

如果两个 `smoke` 都没有明显超过基线，则：

- 当前阶段暂停 cut specialist 单线继续加码
- 转向：
  - 蓝板标准化
  - 尺寸信息融合
  - 或其他结构化先验

九、当前判断

这一步是对“cut 图如果模型够强，应该能比全学更好”这一假设的直接验证。  
如果这一步仍然不成立，就说明当前主问题不只是模型能力大小，而是：

- cut 图内部的类别边界本身依然高度重叠
- mixed / outer 提供的辅助信息并非纯干扰
