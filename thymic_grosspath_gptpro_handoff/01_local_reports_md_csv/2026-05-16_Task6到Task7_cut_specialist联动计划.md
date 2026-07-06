# Task6到Task7 cut specialist 联动计划

一、目的

前一版 stronger cut specialist 直接训练 `Task7`，在单折 `fold1` 上有正向信号，但完整 `5-fold` 没有超过统一基线。现阶段更合理的判断是：在切面信息更强的图上，真正缺的不是简单二分类边界，而是更细的高危内部结构监督。

因此这一版改成：

- 先在 `cut-oriented` 子集上训练更强的 `Task6` 六分类 cut specialist
- 再把 `Task6` 的六分类结果折叠成 `Task7`
- 和同一 `cut-oriented` 子集上的直接 `Task7` 基线，以及前一版直接 `Task7` cut specialist 对照

二、核心假设

1. `cut_surface + cut_heavy` 这批图的信息更接近切面判读逻辑
2. 对这批图来说，`Task6` 的细监督比直接 `Task7` 更有利
3. 即使 `Task6` 具体亚型分错，只要能更多地保住 `B2 / B3 / TC` 留在高危组，折叠回 `Task7` 后整体结果就可能优于直接二分类

三、数据范围

- 数据源：`batch1_batch2_task567_20260514`
- 输入 registry：`cut_oriented_task7_registry.csv`
- 视图范围：
  - `cut_surface`
  - `cut_heavy`
- 病例数：`188`
- 类别范围：
  - `Task6`: `A / AB / B1 / B2 / B3 / TC`
  - `Task7`: `A / AB / B1` 归为低危，`B2 / B3 / TC` 归为高危

四、模型与训练设置

第一版直接沿用前一版 stronger cut specialist 的成功配置，只改任务定义：

- backbone：`DINOv2 vitb14`
- tune scope：`last2_blocks`
- head：`mlp`
- image size：`392`
- epochs：`25`
- batch size：`8`
- patience：`8`
- label smoothing：`0.05`
- selection metric：`Task6 primary metric = macro_f1`

五、执行顺序

P0. 单折 smoke

- 先跑 `fold1`
- 看 `Task6` 本身：
  - `macro_f1`
  - `macro_auc`
- 再把 `Task6` 预测折叠成 `Task7`
- 和同一 `fold1` 的：
  - 统一 `Task7 concat` 基线
  - 直接 `Task7` stronger cut specialist
  做并排比较

P1. 如果单折正向，再扩到完整 `5-fold`

- 聚合完整 `Task6` OOF
- 折叠出完整 `Task7`
- 和 cut-oriented 子集上的统一 `Task7 concat` 基线对照

六、成功标准

这一版至少满足下面之一，才值得升主线：

1. 折叠后的 `Task7 accuracy / balanced accuracy / f1` 超过同一 cut 子集上的统一基线
2. 即使总体 `accuracy` 只是持平，但高危召回明显更好

七、停损标准

满足以下任一条，就不继续在这版配置上扩 full run：

1. `fold1` 折叠回 `Task7` 后仍明显低于统一基线
2. `Task6` 本身 `macro_f1` 很弱，说明 stronger cut specialist 在六分类上也没学稳
3. 折叠后只提升 `auc`，但 `accuracy / f1` 无明显改善

八、后续分叉

如果这版成立：

- 继续做完整 `5-fold`
- 再考虑 `Task6 + Task7` 联合头

如果这版不成立：

- `cut specialist` 主线暂缓
- 转到蓝板标准化 / 尺寸信息 / 结构化病例特征
