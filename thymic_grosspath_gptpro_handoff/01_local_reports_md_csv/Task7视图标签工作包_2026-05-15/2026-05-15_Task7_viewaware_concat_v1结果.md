Task7 视图感知轻量判别器 v1 结果

实验目的
- 在不重训 backbone 的前提下，复用当前 `Task7` 最强 DINO 特征
- 先做一版最省成本的 `view-aware` 轻量分类：
  - `cut_surface` 走 `cut` 头
  - `outer_surface` 走 `outer` 头
  - `mixed` 同时参考 `cut + outer + global`
  - `unclear` 走 `global`
- 看整体正确率能否高于当前 `Task7 concat` 基线

标签来源
- `P2_high_risk_fn`：直接使用已标好的 `view_type_round1`
- `P1_multiview`：根据 `cut_surface_degree / outer_surface_degree / mixed_context` 规则推成 seed 标签
- 合并后 seed 数量：
  - `cut_surface = 29`
  - `mixed = 25`
  - `outer_surface = 3`
  - `unclear = 1`

实现方式
- backbone：`DINOv2 vits14 + vitb14 concat`
- 不重训 backbone，只提 frozen features
- 先用 seed 标签训练一个轻量 `view-type` 分类器，给全量 `285` 张主图打伪标签
- 再按伪标签做 Task7 路由分类

结果

当前 `view-aware v1`：
- `Accuracy = 0.6667`
- `Balanced Accuracy = 0.6668`
- `AUC = 0.7205`
- `F1 = 0.6690`
- `Sensitivity = 0.6809`
- `Specificity = 0.6528`

当前 `Task7 concat` 基线：
- `Accuracy = 0.7158`
- `Balanced Accuracy = 0.7157`
- `AUC = 0.7358`
- `F1 = 0.7117`
- `Sensitivity = 0.7092`
- `Specificity = 0.7222`

差值（view-aware v1 - concat 基线）
- `Accuracy`: `-0.0491`
- `Balanced Accuracy`: `-0.0489`
- `AUC`: `-0.0153`
- `F1`: `-0.0427`
- `Sensitivity`: `-0.0283`
- `Specificity`: `-0.0694`

当前判断
- 这条线第一版已经验证完毕，结果不如当前 `Task7 concat` 基线。
- 视图类型这个方向本身未必无效，但第一版实现明显有两个限制：
  1. `outer_surface` seed 太少，仅 `3` 张，单独训练 `outer` 头非常不稳
  2. `mixed` 的融合规则还是手工设定，缺少更稳定的学习式 gate
- 因此当前不建议继续沿着这版“直接三路由 + 极少 outer seed”深挖。

更合理的后续方向
- 如果继续保留视图线，优先级应降到次要验证项
- 后续更值得做的是：
  - 只把 `view_type` 当额外结构化特征接到后端
  - 或者先继续补更可靠的 `outer_surface` 标注，再看是否值得重启这条线
