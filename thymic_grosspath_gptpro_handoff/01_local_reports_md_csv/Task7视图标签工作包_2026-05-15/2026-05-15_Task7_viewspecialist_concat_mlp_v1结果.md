# Task7 视图特异性正式训练 v1 结果

本轮实验不再使用后处理式路由或逻辑回归 probe，而是改成正式的多头 MLP：

- 共享一套 `DINOv2 vits14 + vitb14 concat` frozen 特征
- `global_head`
- `cut_head`
- `outer_head`
- 通过视图概率做软融合
- `mixed` 同时参考 `cut/outer/global`

## 1. fold1 smoke

相对当前 `Task7 concat` 基线，`fold1` 单折结果是正向的：

- 基线 `fold1`
  - `AUC = 0.7352`
  - `Accuracy = 0.6842`
  - `Balanced Accuracy = 0.6835`
  - `F1 = 0.7000`

- `view-specialist MLP v1` `fold1`
  - `AUC = 0.7592`
  - `Accuracy = 0.7193`
  - `Balanced Accuracy = 0.7186`
  - `F1 = 0.7333`

因此这版值得继续扩成完整 `5-fold`。

## 2. 完整 5-fold 结果

`full_mlp_v1` 最终 `OOF case-level mean`：

- `AUC = 0.7426`
- `Accuracy = 0.6982`
- `Balanced Accuracy = 0.6982`
- `F1 = 0.6950`
- `Sensitivity = 0.6950`
- `Specificity = 0.7014`

当前 `Task7 concat` 基线：

- `AUC = 0.7358`
- `Accuracy = 0.7158`
- `Balanced Accuracy = 0.7157`
- `F1 = 0.7117`

## 3. 结果解读

这版结果说明：

- 视图特异性正式训练并非完全无效
- 它把排序能力继续往上推了一点：`AUC 0.7358 -> 0.7426`
- 但当前版本的分类边界仍然不够稳，导致：
  - `Accuracy 0.7158 -> 0.6982`
  - `F1 0.7117 -> 0.6950`

也就是说：

- 这条线已经比之前的“弱路由 v1”更像样
- 但仍未超过当前主基线
- 当前主要问题不再是模型完全学不到，而更像是：
  - 视图标签噪声
  - `mixed` 占比太高
  - `outer` 样本太少
  - 最终二分类阈值与边界仍不理想

## 4. 阈值检查

针对 `full_mlp_v1` 又做了一次按折验证集选阈值的检查。

结果：

- 阈值优化后
  - `Accuracy = 0.7018`
  - `Balanced Accuracy = 0.7021`
  - `Sensitivity = 0.7376`
  - `Specificity = 0.6667`

说明：

- 阈值调整能把结果略微往上拉一点
- 但仍未超过当前 `Task7 concat` 基线

## 5. 当前判断

这条“cut specialist + outer specialist + mixed 双模式融合”的正式训练版：

- 比之前的轻量版本更强
- 单折可以出现明显正增益
- 但完整 `5-fold` 还没有打过当前 `Task7 concat`

当前不建议把它升成新的主结果。

如果后面还要继续压这条线，更值得的方向不是继续小修参数，而是：

- 补更可靠的全量视图标签
- 对 `mixed` 再细分为 `cut-heavy / outer-heavy`
- 或进一步做真正的 partial fine-tuning，而不只是 frozen features 上的多头训练
