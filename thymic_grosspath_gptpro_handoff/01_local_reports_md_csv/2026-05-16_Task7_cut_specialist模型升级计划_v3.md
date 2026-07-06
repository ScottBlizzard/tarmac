# Task7 cut specialist 模型升级计划 v3

时间：2026-05-16

## 一、为什么要进入 v3

前两轮 cut 主线已经把一件事说明白了：

- 只靠视图路由、mixed 拆分、或简单 specialist 头，不能稳定超过当前统一基线。
- 纯 cut 图并不“天然容易”，说明问题已经不只是选图，而是模型本身对切面内部形态的判别能力还不够。
- 继续在同一套 DINO 路线上抠分辨率、抠轻量头，边际收益已经很低。

因此 v3 的核心目标不是再调小参数，而是正式回答：

> 换模型家族以后，cut-only 这条线能不能真正变强。

## 二、当前基准

### 1. Task7 全局主基线（新数据）

- 模型：`DINOv2 vits14 + vitb14 concat`
- 结果：
  - Accuracy = 0.7158
  - Balanced Accuracy = 0.7157
  - AUC = 0.7358
  - F1 = 0.7117

### 2. cut-oriented 子集基线（188 例）

- 同一模型在 cut 子集上：
  - Accuracy = 0.7500
  - Balanced Accuracy = 0.7507
  - AUC = 0.7839
  - F1 = 0.7638

### 3. 当前 strongest cut specialist（DINO 路线）

- 配置：`vitb14 + 392 + last2 + mlp`
- 单折 smoke 是正向的，但完整 5-fold 没有打过 cut 子集基线。

这说明：

- cut-only 方向不是完全没信号；
- 但当前 DINO specialist 版本还没有真正把 cut 图的优势吃透。

## 三、v3 的核心假设

### 假设 1

`cut_surface` 和 `cut-heavy mixed` 的图，主要取决于内部结构、异质性、坏死/出血、囊变、分叶切面等局部形态。

这类信号未必是 DINOv2 当前最擅长的。

### 假设 2

医学预训练或更偏局部纹理的视觉 backbone，在 cut-only 这条线上，可能比 DINO 更适合。

### 假设 3

如果新 backbone 在 cut-only fold1 smoke 上都打不过：

- cut 子集统一基线；
- 或至少打不过当前 strongest DINO cut smoke；

那就不值得继续扩成完整 5-fold。

## 四、v3 先试哪几条

### P0：切换模型家族，先做 cut-only smoke

第一批只做两条：

1. `BiomedCLIP cut-only`
2. `PLIP cut-only`

统一口径：

- 任务：`Task7`
- 数据：`cut_oriented_task7_registry.csv`
- 折：先跑 `fold1 smoke`
- 输入：`whole`
- 先不做额外路由，只回答“换 backbone 家族后，cut-only 本身能不能更强”

### P1：如果某条 smoke 明显正向，再扩完整 5-fold

判断标准：

- Accuracy 至少不低于 cut 子集统一基线太多；
- AUC 或 F1 有明确提升；
- 或者高危召回明显更好。

### P2：只有在 P1 成立时，才继续做更重训练

比如：

- deeper partial fine-tune
- LoRA / adapter
- `Task6 + Task7` 联合监督

## 五、为什么先试 BiomedCLIP 和 PLIP

### BiomedCLIP

- 是医学视觉-语言预训练模型；
- 虽然 previous whole 主线不一定最强，但 cut 图更像“局部形态 + 医学表征”的场景；
- 先用 frozen probe smoke，成本低，能快速判断值不值得继续。

### PLIP

- 同样是医学/病理方向预训练模型；
- 在 gross image 上未必天然占优，但它和 DINO 的关注点不同；
- cut-only 恰好是最值得试它的地方。

## 六、本轮不再优先做什么

本轮先降级这些方向：

- 继续调分辨率
- 继续调视图路由细节
- 继续拆 mixed 权重
- 继续在同一 DINO specialist 架构上小修

原因：这些方向都已经看到了边际收益变小。

## 七、执行顺序

### Step 1

`BiomedCLIP cut-only fold1 smoke`

### Step 2

`PLIP cut-only fold1 smoke`

### Step 3

把两者和以下两个参照做并排比较：

- cut 子集统一基线
- strongest DINO cut smoke

### Step 4

只把最有希望的一条扩成完整 5-fold。

## 八、成功标准

如果出现以下任一情况，就认为 v3 值得继续：

- Accuracy / F1 明显超过当前 DINO cut smoke；
- AUC 明显高于 DINO cut smoke，且 Accuracy 不掉太多；
- 高危召回明显改善，并且总体命中不崩。

## 九、失败标准

如果 `BiomedCLIP` 和 `PLIP` 两条在 fold1 上都没有正信号：

- 不再继续扩大 cut-only smoke 到 5-fold；
- 说明当前 cut-only 主线的瓶颈更深，不只是换一个 backbone 家族就能解决；
- 那时再转向结构化特征融合或新的监督方式。

## 十、本轮预期产物

- `BiomedCLIP cut-only smoke` 结果
- `PLIP cut-only smoke` 结果
- 与 DINO cut 基线的并排对照表
- 明确下一步只押哪一条
