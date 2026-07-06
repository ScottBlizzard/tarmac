# Task7 真正方案 B（cut specialist + outer-oriented specialist + balanced mixed 融合）Smoke 结果

这次不再使用共享多头，也不是后处理式路由，而是按真正的方案 B 做：

- `global`：全量训练一个全局分类器
- `cut specialist`：使用 `cut_surface + cut_heavy + balanced_mixed(软权重)` 训练
- `outer-oriented specialist`：使用 `outer_surface + outer_heavy + balanced_mixed(软权重)` 训练
- `balanced_mixed`：推理时按 `cut/outer` 概率加权融合，并引入 `global` 兜底

输入特征：
- `DINOv2 vits14 + vitb14 concat` frozen features

视图拆分逻辑：
- `cut_surface = 140`
- `outer_surface = 7`
- `cut_heavy = 48`
- `outer_heavy = 12`
- `balanced_mixed = 76`
- `unclear = 2`

## fold1 smoke

当前 `Task7 concat` 基线 `fold1`：
- `AUC = 0.7352`
- `Accuracy = 0.6842`
- `Balanced Accuracy = 0.6835`
- `F1 = 0.7000`

### 方案 B smoke（默认阈值 0.5）
- `AUC = 0.7611`
- `Accuracy = 0.6316`
- `Balanced Accuracy = 0.6312`
- `F1 = 0.6441`
- `TN = 17`
- `FP = 11`
- `FN = 10`
- `TP = 19`

### 方案 B smoke（按验证集选阈值）
- 最优验证阈值：`0.37`
- `Test Accuracy = 0.6667`
- `Test Balanced Accuracy = 0.6650`
- `TN = 16`
- `FP = 12`
- `FN = 7`
- `TP = 22`

## 结论

这版真正的方案 B：

- `AUC` 确实高于当前基线，说明排序能力有信号
- 但无论默认阈值还是验证集调阈值，`Accuracy / Balanced Accuracy / F1` 都没有超过当前 `Task7 concat` 基线
- 所以它目前还不值得直接扩成完整 `5-fold`

当前判断：

- 视图分治并不是完全无效
- 但只靠现有自动视图标签和当前这批 `outer` 样本量，仍不足以把整体命中率拉上去
- 如果后面继续押视图路线，更值得先补：
  - 更可靠的全量视图标签
  - 更多 `outer-oriented` 训练样本
  - 或者把视图信息作为辅助条件，而不是主路由
