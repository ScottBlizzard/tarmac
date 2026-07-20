# H17-0 历史边界概念可辨识性结果

日期：2026-07-21

状态：`NO-GO HISTORICAL TEXT SUPERVISION`

主线：提高 591 例全覆盖、固定阈值 `0.5` 的低危/高危直接图像分类能力。

## 1. 结论

H17-0 没有通过预注册门槛。

历史医生记录中的 C1“包膜完整/边界清楚”和 C2“包膜中断/边界不清”在当前单张照片中存在弱而稳定的视觉相关性，但不足以成为可靠的空间形态监督：

- 最佳固定空间角色 probe 的 OOF AUC 为 `0.7173`；
- OOF BAcc 为 `0.6728`；
- 外周边界单独 probe 的 AUC 只有 `0.6511`；
- batch1 的空间 probe AUC 只有 `0.6161`；
- 真标签相对 50 次来源内置乱的 AUC 差为 `+0.2087`；
- 来源字段单独预测概念的 AUC 只有 `0.5213`；
- 真正的可学习 attention probe AUC 为 `0.6947`、BAcc 为 `0.6447`；
- attention 没有选择性聚焦 outer-ring，outer-ring 平均权重仅 `0.0529`，低于 16 区域均匀分配的 `0.0625`；
- C1 和 C2 在 445 个已知病例上严格互补，实际上只有一个独立边界概念维度。

因此：

1. 不能把历史 C1/C2 直接加入风险模型；
2. 不能把 C1/C2 包装成两个独立概念；
3. 不能开展基于历史自由文字的大规模概念风险训练；
4. 按预注册进入 20 例 morphology-only 可见性 mini-pilot；
5. 严格外部数据继续封存。

## 2. 数据与防泄漏

- 病例：锁定的 591 例内部队列；
- 概念已知病例：445；
- 未知病例：146，完全从概念 loss 和指标中排除；
- C1：356 阳性、89 阴性；
- C2：89 阳性、356 阴性；
- C1/C2 已知掩码完全一致；
- C2 在已知病例上严格等于 `1-C1`；
- 五折：沿用 H10/H16 的 `master_fold_id`；
- 标准化、PCA、C 选择均在外层训练/验证折内完成；
- 风险标签未读取；
- 六亚型未读取；
- 旧模型输出、置信度和错误身份未读取；
- 严格外部图像和标签未读取。

## 3. 固定特征 probe

### 3.1 候选

| 候选 | 输入 |
|---|---|
| P0-global | 冻结 PE-Spatial specimen 区域特征及描述量 |
| P1-outer | 冻结 outer-ring 区域特征及描述量 |
| P1-spatial | specimen、core、outer-ring、outer-core 差值和固定空间关系 |
| P3-source-only | 仅来源 one-hot，作为来源捷径对照 |

所有视觉候选均使用训练折内 StandardScaler、PCA 和 class-balanced logistic probe。阈值固定为 `0.5`。

### 3.2 总体结果

| 候选 | Accuracy | BAcc | AUC |
|---|---:|---:|---:|
| P0-global | 0.7011 | 0.6657 | 0.7091 |
| P1-outer | 0.6022 | 0.5829 | 0.6511 |
| P1-spatial | 0.7191 | 0.6728 | 0.7173 |
| P3-source-only | 0.6449 | 0.5337 | 0.5213 |

P1-spatial 相对 P0-global 的 AUC 只提高 `0.0082`。真正的 outer-ring 单独使用时明显更差，说明弱信号主要不是稳定的外周包膜语义。

### 3.3 五折稳定性

P1-spatial 五折 AUC：

| Fold | AUC |
|---|---:|
| 1 | 0.7200 |
| 2 | 0.7266 |
| 3 | 0.7250 |
| 4 | 0.7261 |
| 5 | 0.7320 |

五折数值稳定，说明 `0.7173` 不是单个坏折造成；但稳定地低于晋级目标仍然是 no-go。

### 3.4 来源结果

| 来源 | P0-global AUC | P1-outer AUC | P1-spatial AUC |
|---|---:|---:|---:|
| batch1 | 0.6648 | 0.5584 | 0.6161 |
| batch2 | 0.7261 | 0.6368 | 0.7658 |
| third_batch | 0.7143 | 0.6848 | 0.7123 |

P1-spatial 在 batch2 明显较强，却在 batch1 失败。该概念不能被视为跨来源稳定视觉语义。

### 3.5 置乱和来源对照

50 次在每个来源内部置乱训练标签：

| 指标 | 值 |
|---|---:|
| mean AUC | 0.5086 |
| median AUC | 0.5167 |
| 95% quantile AUC | 0.5596 |
| max AUC | 0.5752 |
| 真标签 P1 AUC | 0.7173 |
| 真标签减置乱均值 | 0.2087 |

这证明 P1 的弱信号不是纯随机拟合。

但来源 one-hot 对概念的 AUC 只有 `0.5213`，P1 相对来源对照高 `0.1960`。因此，失败也不能简单解释为“模型只认来源”。更合理的解释是：

1. 当前照片确实含有部分边界相关外观；
2. 历史文字可能来自触诊、其他切面或完整标本观察；
3. 记录中的边界概念和当前单图的可见边界并不一一对应；
4. 当前单图中的可见信号强度不足以形成稳定监督。

## 4. Attention probe

为严格执行原计划，又运行了真正的 16 区域 attention pooling：

- backbone 和区域特征冻结；
- 只训练区域 projection、role embedding、attention 和二分类头；
- 五折训练/验证/测试隔离；
- class-balanced BCE；
- 阈值固定 `0.5`；
- 风险、亚型、来源和旧模型输出均不作为输入。

### 4.1 结果

| 指标 | Attention probe |
|---|---:|
| Accuracy | 0.6674 |
| BAcc | 0.6447 |
| AUC | 0.6947 |
| Sensitivity | 0.6826 |
| Specificity | 0.6067 |

五折 AUC：

`0.7158 / 0.7254 / 0.7065 / 0.7191 / 0.6970`

来源 AUC：

| 来源 | AUC |
|---|---:|
| batch1 | 0.5682 |
| batch2 | 0.7619 |
| third_batch | 0.6939 |

### 4.2 Attention 机制

| 区域组 | 实际平均 attention | 16 区域均匀基线 |
|---|---:|---:|
| specimen + core + outer-ring | 0.1884 | 0.1875 |
| outer-ring | 0.0529 | 0.0625 |
| 9 个颜色/梯度候选合计 | 0.5579 | 0.5625 |

attention 基本停留在均匀分配，没有学会选择性强化 outer-ring。不能声称模型学到了包膜语义。

## 5. 预注册门槛

固定 P1-spatial：

| 门槛 | 结果 |
|---|---|
| macro AUC `>=0.75` | 失败，0.7173 |
| macro BAcc `>=0.68` | 失败，0.6728 |
| 至少 4/5 折 AUC `>0.70` | 通过，5/5 |
| 三来源 AUC 均 `>=0.65` | 失败，batch1 0.6161 |
| 真标签减置乱 AUC `>=0.10` | 通过，+0.2087 |
| 相对 source-only AUC `>=+0.05` | 通过，+0.1960 |
| outer-ring AUC `>=0.70` | 失败，0.6511 |
| spatial 不比 global 低超过 0.02 | 通过 |

Attention 版本同样未通过 AUC、BAcc 和三来源门槛。

最终决策：

`FAIL_HISTORICAL_TEXT_PROBE_RUN_20_CASE_VISIBILITY_MINIPILOT`

## 6. 对主线的含义

H17-0 没有训练风险模型，因此本轮不能直接提高 H10 的风险分类指标。它完成的是必要的错误路线排除：

- 历史自由文字不能直接转成当前照片的空间语义标签；
- “包膜完整”和“包膜不清”不能被当作两个独立概念扩充监督量；
- 继续增强 attention 容量不能自动修复标签与照片不对应；
- 弱概念信号存在，但主要不是可靠 outer-ring 语义；
- 下一步必须使用当前照片本身的 morphology-only 标注验证可见性。

## 7. 服务器产物

固定线性 probe：

`/workspace/thymic_project/experiments/h17_spatial_morphology_risk_20260720/h17_0`

Attention probe：

`/workspace/thymic_project/experiments/h17_spatial_morphology_risk_20260720/h17_0_attention`

代码：

- `run_task7_h17_historical_concept_probe_20260721.py`
- `run_task7_h17_attention_concept_probe_20260721.py`

两套输出均包含 OOF 预测、逐折/来源指标、配置、门槛 JSON 和 `RUN.status`。
