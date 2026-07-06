# Task7 项目完成度评估

## 使用说明

本文用于内部判断“胸腺大体病理图像 Task7 低危/高危风险控制诊断框架”的当前项目完成度，并辅助评估其分别面向医学期刊和计算机期刊时的证据成熟度。

本文不是投稿论文正文，也不重新计算实验指标。内容基于现有阶段性工作详报、2026-05-28 完整成果与论文完成度报告、clean/full cohort 分析、Task7 第三批与 strict external 阶段汇报、GrossPath-RC v0-v2 工作流报告及相关负结果记录整理。

最重要的口径边界：

- 当前最强结果属于 `selective diagnosis / risk-controlled workflow`，不是全量病例完全自动诊断。
- 第三批数据已经参与观察、适配和策略选择，应称为“扩展开发队列”或“适配队列”，不能称为 `strict external`。
- `strict external` 主要用于冻结评估或外部压力测试，其强制分类下降是项目转向风险控制工作流的重要依据。
- 所有 99% 或 100% 的高分结果必须同时说明自动覆盖率、复核/拒识比例、自动错误数和 Wilson95 上界。

## 相较上一版完成度总结报告的完善优化

相较于 2026-05-28 完成度总结报告，本文主要做了以下完善：

- 从摘要式结论扩展为可审查的证据矩阵，便于逐项判断项目完成度。
- 新增 1-6 项核心资料整理：数据队列、clean/full cohort、Task7 主结果、strict external、v195 workflow 和负结果。
- 明确区分第三批扩展开发队列与 `strict external`，避免把适配数据误写成严格外部验证。
- 强化 v195 高分的解释边界：属于 `selective diagnosis / risk-controlled workflow`，不是全量自动诊断。
- 补充 hard case 分层与医生复核状态判断：已完成分层和复核清单，但核心 hard 逐例医生复核尚未完整闭环。
- 增加医学方向和计算机方向对标论文矩阵，并标注官方学术来源。
- 将负结果整理为论文证据，用于支撑方法边界、失败模式和后续补强方向。

## 简短目录

1. 数据集与队列总表
2. clean cohort / full cohort 形成依据
3. Task7 最终主结果表
4. strict external 强制分类结果
5. v195 workflow 结果表
6. 负结果与不采用原因
7. 对标论文矩阵：医学方向与计算机方向
8. 汇总判断与后续整理优先级

---

## 一、数据集与队列总表

### 已有证据

| 队列 | 样本量 / 规模 | 低危/高危或类别构成 | 来源 / 角色 | 是否参与训练或调参 | 推荐报告叫法 | 主要注意点 |
| --- | ---: | --- | --- | --- | --- | --- |
| 早期 Task1-Task6 数据 | 160 例，172 张图像 | 覆盖 Task1-Task6 多种任务 | 任务体系建立、DINO 强表征验证、早期多分类与二分类实验 | 是 | 早期开发队列 | 不能与后续 Task7、strict external 或 v195 workflow 直接横向比较 |
| Task7 旧数据 / 新合并数据 | 285 例 | 低危/高危基本均衡，旧阶段记录为低危 144 例、高危 141 例 | Task7 低危/高危二分类开发、OOF 评估、课程学习、hard case 分层 | 是 | 内部开发队列 / OOF 队列 | OOF 结果不等于外部验证结果 |
| 第三批数据 | 306 例 | 低危 224 例、高危 82 例；低危以 AB 为主，高危以 B2/TC 为主 | 新增同体系数据，用于观察分布变化、适配和扩展开发验证 | 是，已经参与观察、适配和策略选择 | 扩展开发队列 / 适配队列 / 时间外开发队列 | 不能再称为 `strict external` 或完全干净外部测试集 |
| 第三批 adapt72 | 72 例 | 从第三批中抽取，偏向高危和错误高发样本 | 小样本适配训练和策略选择 | 是 | 第三批适配集 | 不是最终外部测试 |
| 第三批 holdout234 | 234 例 | 第三批剩余病例，低危占多数 | 适配后 holdout 评估 | 原则上用于评估；若使用其标签筛选策略，则只能算参考分析 | 第三批 holdout | 使用 holdout 标签得到的高分只能写作参考上限 |
| strict external | 108 例；部分分析中严格可评估约 105 例 | 低危/高危约 61/47 | 外部压力测试或冻结评估 | 不应用于训练、调参或策略选择 | 严格外部集 / 外部压力测试集 | 样本量较小，强制分类表现下降，后期 workflow 结果必须报告复核/拒识比例 |
| clean cohort | n=697 | 后期主分析队列，低危/高危 429/268 | 医生确认特殊组织学或任务边界后形成 | 用于后期主分析 | clean primary cohort / 主分析队列 | 不是按模型错误临时删除 |
| full cohort | n=699 | 完整可追溯队列，低危/高危 429/270 | clean cohort 的敏感性分析对照 | 用于敏感性分析 | full cohort sensitivity analysis | 需要与 clean cohort 同时报告，证明结论对剔除不敏感 |
| v195 自动释放子集 | clean cohort 中自动覆盖 63.99%；strict external 中自动覆盖 48.15% | 随策略和队列变化 | risk-controlled workflow 自动放行部分 | 属于 workflow 评估口径 | selective diagnosis 自动释放子集 | 不是全量病例；必须同步报告复核/拒识和自动错误 |

### 可写结论

- 项目已经形成比较清楚的数据分层：旧数据负责内部开发和 OOF 评估，第三批负责扩展开发和适配验证，`strict external` 负责外部压力测试，clean/full cohort 负责后期主分析与敏感性分析。
- 当前数据治理已经能支撑一篇回顾性医学 AI 研究的基本队列描述，但外部验证样本量和前瞻验证仍不足。
- 第三批数据的引入证明旧数据结果不能直接外推，也说明模型面对图像构图、亮度、主体大小、类别比例变化时存在可测量域偏移。

### 不能过度表述

- 不能把第三批写成 `strict external`，因为第三批已经参与适配和策略选择。
- 不能把 OOF 结果写成真实外部验证。
- 不能把 clean cohort 写成“删除错例后的提分队列”；它的依据应是医生确认的任务边界或特殊组织学原因。
- 不能把 selective diagnosis 自动释放子集的高分解释成全量病例完全自动诊断成功。

### 完成度判断

| 维度 | 当前完成度 | 判断 |
| --- | --- | --- |
| 医学期刊 | 基本具备，但需要补强 | 已有队列分层、外部压力测试、医生复核和 clean/full sensitivity；缺少前瞻验证、多中心大样本外部验证和医生读片对照实验 |
| 计算机期刊 | 中等，需要方法化表达 | 已有跨域队列、适配队列、选择性诊断和风险控制证据；需要更明确的问题定义、统一算法流程、可复现实验协议和消融表 |

### 待补资料

- 各队列的病例纳入排除流程图。
- 每个队列的医院来源、时间范围、拍摄方式、每例图像数和最终选图规则。
- strict external 是否完全未参与策略选择的冻结说明。
- clean cohort exclusion list 的医生确认依据和版本锁定记录。

---

## 二、clean cohort / full cohort 形成依据

### 已有证据

clean cohort 的形成基于医生复核意见和固定 exclusion list。原始数据不物理删除；主分析使用 clean primary cohort，完整队列作为 full-cohort sensitivity analysis。

已记录的 exclusion list 包括：

| 病例 | 记录原因 | 与当前 Task7 主流程关系 |
| --- | --- | --- |
| 2404716 | MEC | 不在当前 Task7 主流程中 |
| 2307206 | 淋巴上皮癌 | 在当前 Task7 主流程中 |
| 2205101 | 淋巴上皮癌 | 在当前 Task7 主流程中 |
| 2203278 | 微结节型 TC | 不在当前 Task7 主流程中 |
| 2113767 | 肠型腺癌 | 不在当前 Task7 主流程中 |

当前 v195 主结果：

| 队列 | n | Acc | BAcc | FN | FP | 复核/拒识 | 自动错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full cohort | 699 | 99.86% | 99.81% | 1 | 0 | 36.05% | 1 |
| clean cohort | 697 | 99.86% | 99.81% | 1 | 0 | 36.01% | 1 |
| clean old_data | 283 | 100.00% | 100.00% | 0 | 0 | 26.15% | 0 |

这些指标均属于 `selective diagnosis / risk-controlled workflow` 口径，不是全量自动诊断口径。

### 可写结论

- clean cohort 的主要作用是让任务边界更干净，而不是人为提高分数。
- clean cohort 与 full cohort 结果基本一致，说明当前主结论对特殊病例剔除不敏感。
- 主文适合报告 clean cohort，补充材料适合报告 full cohort sensitivity analysis。

### 不能过度表述

- 不能写成“剔除后模型明显提升”，因为现有结果显示 clean 与 full 几乎一致。
- 不能在没有医生确认说明的情况下，把 exclusion list 写成纯工程筛选。
- 不能只报告 clean cohort，不报告 full cohort sensitivity，否则容易被质疑选择性剔除。

### 完成度判断

| 维度 | 当前完成度 | 判断 |
| --- | --- | --- |
| 医学期刊 | 较好 | clean/full 双口径和医生确认 exclusion list 是重要加分项；仍需补齐纳入排除流程图和医生确认签名或版本记录 |
| 计算机期刊 | 辅助价值较高 | 可作为数据治理和任务定义边界证据，但不是主要算法贡献 |

### 待补资料

- exclusion list 的正式医生复核表或会议纪要。
- 每个剔除病例的病理亚型、剔除理由、是否影响 Task7 标签定义。
- clean/full cohort 结果在补充材料中的固定表格版本。

---

## 三、Task7 最终主结果表

### 已有证据

Task7 指低危/高危二分类：低危包括 A、AB、B1；高危包括 B2、B3、胸腺癌。该任务是当前最适合作为报告主线的医学问题，因为它比 WHO 五分类或六分类更贴近风险分层和高危漏判控制。

早中期强制分类 / OOF 代表结果：

| 阶段 | 口径 | 代表方案 | 主要结果 | 阶段判断 |
| --- | --- | --- | --- | --- |
| Task7 基线修补 | 旧数据 285 例 OOF | `DINOv2 vits14+vitb14 concat` + logreg / case MLP / image MLP 融合 | AUC 0.7454，Acc 0.7368，BAcc 0.7370，F1 0.7387 | 证明双 DINO 强特征和轻量 probe 是稳定基础 |
| Task7 probe 优化 | 旧数据 285 例 OOF | case-level MLP seed2027 | AUC 0.7620，Acc/BAcc 约 0.7298 | probe 优化有帮助，但未解决核心难例 |
| 课程学习 | 旧数据 285 例 OOF | Stage2 + Stage3 fold-wise fusion | AUC 0.8117，Acc/BAcc 0.7649，F1 0.7633 | 早中期正式主结果，说明难例分层有效 |
| 经验标签候选 | 旧数据 285 例 OOF | 31 例高可信核心经验标签候选融合 | Acc 0.7684，BAcc 0.7685；细网格复查降到 0.7614 | 有信号但不够稳定，不能作为成熟突破 |
| 二阶段自动复核候选 | 旧数据 OOF/候选二阶段 | `review_score_hybrid_max_allrange_fn` 等 | overall Acc 0.8070，BAcc 0.8065，AUC 0.8314，F1 0.7955 | 候选结果，不等同成熟自动纠偏系统 |

后期主流程 v195 workflow：

| 队列 | n | Acc | BAcc | F1 | sensitivity | specificity | FN/FP | 自动覆盖 | 复核/拒识 | 自动错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean cohort | 697 | 99.86% | 99.81% | 99.81% | 99.63% | 100.00% | 1/0 | 63.99% | 36.01% | 1 |
| full cohort | 699 | 99.86% | 99.81% | 99.81% | 99.63% | 100.00% | 1/0 | 63.95% | 36.05% | 1 |

注意：后期主流程结果属于 `selective diagnosis / risk-controlled workflow`。复核/拒识病例没有被强行自动诊断，因此不能把该结果写成全量病例完全自动诊断。

### 可写结论

- Task7 从“单模型二分类”逐步演进为“风险可控选择性诊断工作流”。
- 早中期结果证明胸腺大体图中存在可学习的低危/高危信号，但全量强制分类未达到理想稳定性。
- hard case 分层说明模型瓶颈集中于核心 hard 病例，而不是所有病例都识别困难。
- v195 workflow 在允许复核/拒识的临床工作流口径下，能够以较低自动错误率自动释放约三分之二 clean cohort 病例。

### 不能过度表述

- 不能把 v195 的 99% 以上指标写成“模型对全部病例自动诊断准确率 99%”。
- 不能把二阶段 0.807 候选结果写成成熟自动复核系统。
- 不能把经验标签 0.7684 写成稳定突破，因为细网格复查下降。
- 不能忽略 FN；Task7 中 FN 是高危被判成低危，医学风险高于 FP。

### 完成度判断

| 维度 | 当前完成度 | 判断 |
| --- | --- | --- |
| 医学期刊 | 较强的回顾性工作流证据 | 已能支撑“风险可控辅助判读”主线；若要冲更高层级，需医生对照实验、前瞻验证和更多外部队列 |
| 计算机期刊 | 应用证据较强，方法贡献需压缩提炼 | 需要把课程学习、hard case 分层、v195 规则和域偏移门控整合成一个清楚算法框架，而不是多版本实验堆叠 |

### 待补资料

- Task7 最终推荐主结果表的固定版本。
- 早中期 OOF、第三批、strict external、v195 workflow 的口径对照图。
- hard case 分层定义和病例数的正式说明。
- 医生最关心的 FN 清单和复核结论。

---

## 四、strict external 强制分类结果

### 已有证据

strict external 是项目中最关键的外部压力测试口径。该队列约 108 例，部分分析中严格可评估约 105 例，原则上不用于训练、调参或策略选择。

强制分类代表结果：

| 口径 | n | Acc | BAcc | AUC | 高危召回 | 低危特异度 | FN/FP | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict external 全部病例 | 108 | 0.6481 | 0.6275 | 0.6083 | 0.4681 | 0.7869 | 25/13 | 明显低于旧数据和第三批 |
| strict external 严格可评估 | 105 | 0.6381 | 0.6220 | 0.6038 | 0.4681 | 0.7759 | 25/13 | 强制分类不足，是外部泛化边界证据 |
| v1 base162 forced classification | 105 | 0.6381 | 0.6220 | 0.6016 | 0.4681 | 0.7759 | 25/13 | 与阶段性详报一致 |
| v1 mean_core forced classification | 105 | 0.6762 | 0.6766 | 0.6662 | 0.6809 | 0.6724 | 15/19 | 高危更敏感，但旧数据和第三批下降，不能替代主模型 |

### 可写结论

- strict external 强制分类表现下降，说明模型尚未解决跨拍摄体系、跨数据来源、跨构图风格的稳定泛化问题。
- 这不是单纯阈值问题，而是域偏移问题：图像质量、主体大小、切面/外观比例、拍摄背景和病变呈现方式都可能影响模型。
- strict external 的下降是后期从“全量强制分类器”转向 `risk-controlled workflow` 的核心依据。

### 不能过度表述

- 不能用旧数据或第三批的高分掩盖 strict external 强制分类下降。
- 不能把 strict external 后期 workflow 的高点估计单独写成“外部验证 100%”。
- 不能只报 strict external 的 Acc；必须同时报 BAcc、高危召回、FN/FP 和复核/拒识比例。

### 完成度判断

| 维度 | 当前完成度 | 判断 |
| --- | --- | --- |
| 医学期刊 | 有外部压力测试证据，但外部强制分类不足 | 诚实呈现会增强可信度；若想形成强外部验证论文，需要更多外部样本和医生对照 |
| 计算机期刊 | 域泛化问题成立 | 可作为方法论文问题动机，但现阶段需要更系统的 domain shift 分析和跨域消融 |

### 待补资料

- strict external 的来源、时间范围、图像采集条件和病例构成。
- strict external 是否完全冻结评估的流程说明。
- strict external 错例按图像质量、主体大小、切面/外观、特殊组织学等因素的分层分析。
- 强制分类与 workflow 结果的并列表，避免读者混淆。

---

## 五、v195 workflow 结果表

### 已有证据

v195 是当前阶段主流程。它采用 `selective diagnosis / risk-controlled workflow` 口径：系统只自动释放有把握的病例，其余病例进入复核、拒识或人工确认流程。

核心指标：

| 队列 | n | 低危/高危 | Acc | BAcc | F1 | sensitivity | specificity | FN/FP | 自动覆盖 | 复核/拒识 | 自动错误 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧数据 clean | 283 | 144/139 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0/0 | 73.85% | 26.15% | 0 |
| 第三批适配队列 | 306 | 224/82 | 99.67% | 99.39% | 99.39% | 98.78% | 100.00% | 1/0 | 60.46% | 39.54% | 1 |
| strict external | 108 | 61/47 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0/0 | 48.15% | 51.85% | 0 |
| clean cohort 总体 | 697 | 429/268 | 99.86% | 99.81% | 99.81% | 99.63% | 100.00% | 1/0 | 63.99% | 36.01% | 1 |
| full cohort sensitivity | 699 | 429/270 | 99.86% | 99.81% | 99.81% | 99.63% | 100.00% | 1/0 | 63.95% | 36.05% | 1 |

自动错误风险：

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| v195 全域自动判读 | 447 例 | 自动释放或自动判读病例数 |
| 自动错误 | 1 例 | 自动释放部分出现的错误数 |
| 自动错误 Wilson95 上界 | 1.26% | 对自动错误率 95% 置信区间上界的估计 |
| 相比 v185 | 自动判读 287 -> 447；复核/拒识 58.94% -> 约 36.05% | 提高自动覆盖，同时保持自动错误为 1 |

### 可写结论

- v195 是当前最适合作为主流程的风险控制版本。
- 相比更保守版本，v195 明显提高自动释放效率；相比更激进版本，v195 保持较低自动错误。
- strict external 的 workflow 点估计较高，但自动覆盖只有 48.15%，复核/拒识 51.85%，说明外部域仍需要严格风险控制。
- Wilson95 上界可以帮助避免只报“0 错误”或“1 错误”带来的过度乐观解释。

### 不能过度表述

- 不能写“strict external 外部验证 100% 准确”，必须写成“在 v195 selective diagnosis workflow 口径下，自动释放子集点估计无错误，但复核/拒识比例 51.85%”。
- 不能省略自动覆盖率，否则高分会被误解为全量自动诊断。
- 不能把 v195 等同于最终部署系统；它仍需要前瞻验证、医生工作流验证和更多外部批次验证。

### 完成度判断

| 维度 | 当前完成度 | 判断 |
| --- | --- | --- |
| 医学期刊 | 风险控制工作流证据较强 | 已具备“辅助判读 / 自动释放 + 复核”的回顾性证据；缺少真实医生工作流实验 |
| 计算机期刊 | 有方法潜力 | 需要把 v195 从经验规则整理为可复现算法，包括输入、评分、释放规则、拒识规则和风险估计 |

### 待补资料

- v195 的完整规则定义和冻结版本号。
- v195 与 v118、v161、v180、v182、v185 的统一对照表。
- Wilson95 的计算方法、分母和置信区间说明。
- 医生复核流程：复核病例由谁处理、如何处理、是否改变最终诊断。

---

## 六、负结果与不采用原因

### 已有证据

负结果是当前项目完成度评估的重要组成部分。它们说明团队不是只保留高分，而是系统排除了不稳定、不可部署或不符合医学安全的方向。

#### 1. 输入策略类

| 尝试 | 目的 | 结果现象 | 不采用原因 | 论文价值 |
| --- | --- | --- | --- | --- |
| cut-only / cut specialist | 让模型更关注肿瘤切面主体 | 有 AUC 信号，如 DINO cut specialist AUC 0.7544；full-to-cut AUC 0.7848，但 Acc/BAcc/F1 不稳定 | 单独 cut 不能稳定超过统一基线 | 说明切面重要但不能孤立使用 |
| full-to-cut refine | 先用全图再用切面细化 | AUC 有提升，分类指标未同步稳定提升 | 最终分类边界不够稳 | 可作为局部信号探索 |
| whole / crop / whole+crop / flip TTA | 适配第三批输入分布 | 第三批上不同输入差异明显，但 TTA 平均和部分归一化下降 | 简单输入变换不能解决域偏移 | 支撑“外部下降不是单一色彩或裁剪问题” |
| 蓝板背景归一化 / 图像统计量 | 减轻背景与拍摄差异 | 第三批表现未稳定提升 | 不能单独作为主方法 | 支撑质量门控和域偏移分析 |

#### 2. 模型替换类

| 尝试 | 目的 | 结果现象 | 不采用原因 | 论文价值 |
| --- | --- | --- | --- | --- |
| ConvNeXt / Swin / EfficientNet | 替换 DINO backbone | 某些 fold 或局部指标有信号，但整体分类边界不稳 | 不如 DINO frozen feature 稳定 | 支撑 DINO 强表征作为基础选择 |
| BiomedCLIP / PLIP cut-only | 使用医学图文预训练模型 | BiomedCLIP cut-only AUC 0.6402；PLIP 单模弱于 DINO | 单模不足，融合收益有限 | 说明通用医学预训练不一定适合胸腺大体图 |
| DINO 微调 / last-block / head-only | 提高第三批适配能力 | 可提高局部高危召回，但旧数据表现明显下降 | 破坏旧数据稳定性，泛化不稳 | 支撑“冻结特征 + 风险控制”优于盲目微调 |
| DINOv3 / QKVB / TTA 特征组合 | 增强外部敏感性 | DINOv3 对外部高危更敏感，但旧数据和第三批下降 | 不能直接替代主模型 | 可作为分歧信号或候选敏感分支 |

#### 3. 经验标签类

| 尝试 | 目的 | 结果现象 | 不采用原因 | 论文价值 |
| --- | --- | --- | --- | --- |
| 43 例 strong label | 用少量高质量经验线索辅助训练 | 有信号但未超过主结果 | 覆盖太窄 | 说明医生经验标签方向可行 |
| 204 例 weak+strong 扩展标签 | 扩大经验标签覆盖 | AUC 有一定信号，但 weak label 干扰 easy/medium 学习 | 噪声过大，分类边界不稳 | 支撑“经验标签必须少而精” |
| 31 例高可信核心集 | 用少而精的正反对照标签辅助 Stage3 | 0.01 网格 Acc 0.7684，但 0.005 复查降到 0.7614 | 不够稳定，不能作为正式突破 | 可作为经验标签候选正证据 |
| 核心 hard clean3 / hard 加权 | 尝试救核心 hard | fold1 明显下降或扰动边界 | 核心 hard 不适合直接当普通训练样本加权 | 支撑核心 hard 需要医生复核和单独机制 |

#### 4. 融合 / guard 类

| 尝试 | 目的 | 结果现象 | 不采用原因 | 论文价值 |
| --- | --- | --- | --- | --- |
| stacking | 融合多个模型概率 | 旧数据和第三批基本同分，strict external 反而略差 | 没有稳定超过 base162 | 作为否证实验，说明复杂融合不一定泛化 |
| mean_core | 改善 strict external 高危敏感性 | strict external BAcc 0.6766，高于 base162；但旧数据和第三批 holdout 下降 | 不能替代主模型 | 可作为外部敏感分支，不作主线 |
| v1 guard | 训练可靠性评分器筛自动子集 | 域内能筛出高准确子集，冻结到外部后失效 | 域内置信度校准不能直接解决外部泛化 | 支撑多模型一致性比 guard 更稳 |
| 质量门控 | 筛查不可读、重拍或边界病例 | 可辅助安全工作流 | 不是分类模型性能提升 | 适合写作安全流程，不适合写成全量诊断准确率 |

#### 5. 自动纠偏候选

| 尝试 | 目的 | 结果现象 | 不采用原因 | 论文价值 |
| --- | --- | --- | --- | --- |
| v199 / v200 | 自动翻转或纠偏错误病例 | 有误伤记录 | 安全性不足 | 作为失败边界 |
| v201_supported_domain_flip | 在支持域中做自动纠偏 | 第三批救回 2 例，误伤 0 例 | 样本量太小，只是候选增强模块 | 可写成正信号，但不能写成成熟自动纠偏 |

### 可写结论

- 当前负结果支持一个清楚判断：项目主要价值不在“不断堆更复杂模型”，而在识别风险边界并建立可拒识、可复核、可审计的工作流。
- 输入变换、模型替换、简单微调和 stacking 都没有稳定解决外部泛化。
- 医生经验标签有价值，但需要高质量、少而精、正反对照明确；不能把弱标签大规模灌入训练。
- 自动纠偏目前只能作为候选模块，不能作为成熟部署能力。

### 不能过度表述

- 不能把局部高 AUC 当作主模型改进，必须同时看 Acc、BAcc、FN/FP 和外部表现。
- 不能把失败实验删掉，否则报告会显得只挑高分。
- 不能把 v201 写成正式纠偏模块。
- 不能把 guard 写成外部可靠性解决方案，因为现有证据显示它在 strict external 上失效。

### 完成度判断

| 维度 | 当前完成度 | 判断 |
| --- | --- | --- |
| 医学期刊 | 有助于增强可信度 | 负结果能证明团队重视安全边界和高危漏判，不盲目追分 |
| 计算机期刊 | 很重要但需结构化 | 需要把负结果整理为消融和失败模式分析，而不是散落在阶段报告中 |

### 待补资料

- 所有负结果的统一表格，包含同一数据口径、同一指标集和是否采用。
- 失败案例图示：输入策略失败、域偏移失败、核心 hard 失败、guard 失败。
- 与 v195 主流程对照的消融表：去掉一致性、去掉质量门控、去掉偏移门控、改变释放阈值等。
- v201 的更多样本验证和误伤风险统计。

---

## 七、对标论文矩阵：医学方向与计算机方向

本节用于判断当前项目分别接近哪些医学期刊和计算机期刊论文水准。来源优先采用 PubMed、PMC、ScienceDirect、Nature、RSNA、AAAI、OpenReview 等官方学术网站或出版社页面。若官方页面未明确报告某项信息，则标注为“官方页面未确认”，不作推断。

### 7.1 医学方向对标论文

| 论文 | 期刊 | 数据量 | 是否多中心 | 是否外部验证 | 是否前瞻验证 | 任务 | 模型 | 指标 | 是否开源 | 主要贡献 | 我们项目能否达到 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [An interpretable machine learning approach using nnU-Net-based radiomics for preoperative risk stratification of thymic epithelial tumors: a multicenter study](https://pubmed.ncbi.nlm.nih.gov/41668011/) | 官方 PubMed 页面显示为胸腺上皮肿瘤风险分层多中心研究；具体期刊名需以 PubMed 页面为准 | 764 例 | 是 | 是，训练/内部验证/两个外部验证队列 | 否，回顾性 | TET 术前风险分层 | nnU-Net 分割 + radiomics + machine learning | AUC、校准、决策曲线等 | 官方摘要页未确认 | 同病种、同风险分层、多中心外部验证，是医学方向最直接高标准对标 | 目前任务接近，但数据规模、多中心外部验证和正式临床证据不足 |
| [Machine learning-based radiomics analysis in enhancing CT for predicting pathological subtypes and WHO staging of thymic epithelial tumors: a multicenter study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12163439/) | 官方 PMC 页面；题名为多中心 TET CECT radiomics 研究 | 257 例，中心一 181 例、中心二 76 例 | 是，两个中心 | 是，中心一训练、中心二测试 | 否，回顾性 | 预测 TET 病理亚型和 WHO 分期 | CECT radiomics + 多种机器学习模型 | AUC、Accuracy、混淆矩阵等 | 官方页面未确认 | 对标 Task5/Task6 多分类和 WHO 分期任务 | 我们有 Task5/Task6 早期结果，但主线已转 Task7；多中心 CT 证据强于当前大体图证据 |
| [CT-Based Radiomics Signatures for Predicting the Risk Categorization of Thymic Epithelial Tumors](https://pubmed.ncbi.nlm.nih.gov/33718203/) | Frontiers in Oncology | 190 例；低危 83、高危 58、胸腺癌 49 | 否，单中心为主 | 否，内部验证集 | 否，回顾性 | 低危胸腺瘤、高危胸腺瘤、胸腺癌三分类 | 3D CT radiomics + clinical / semantic features + logistic regression | AUC、ACC | 官方页面未确认 | 建立 TET 风险三分类的 radiomics 基线 | 我们 Task7 低危/高危更简单，但大体图新颖；缺 CT radiomics 这类标准化输入 |
| [Contrast-enhanced CT-based radiomics model for differentiating risk subgroups of thymic epithelial tumors](https://pubmed.ncbi.nlm.nih.gov/35249531/) | BMC Medical Imaging | 164 例；训练 130，测试 34 | 否，单中心 / 时间划分 | 否，时间测试集但非外部中心 | 否，回顾性 | TET 风险亚组分类：传统 LRT/HRT/TC 与改良 LRT*/HRT*/TC | CECT radiomics + LASSO + logistic regression | AUC、Accuracy、校准、决策曲线 | 官方页面未确认 | 证明 TET 风险分层中任务折叠方式会影响表现 | 我们项目同样需要解释为何采用低危/高危折叠；当前 workflow 证据更完整，但样本标准化弱 |
| [Diagnostic performance of radiomics model for preoperative risk categorization in thymic epithelial tumors: a systematic review and meta-analysis](https://pubmed.ncbi.nlm.nih.gov/37644397/) | BMC Medical Imaging | 13 项研究，2134 例 | 是，系统综述覆盖多研究 | 不适用，综述本身不做外部验证 | 否 | TET 术前风险分层 radiomics 诊断性能综述 | meta-analysis | pooled AUC、sensitivity、specificity、QUADAS-2、RQS | 不适用 | 提供 TET radiomics 领域证据基线：高/低危 pooled AUC 约 0.855 | 可作为背景和讨论依据；我们是大体图路线，需强调区别而非直接比拼 CT radiomics |
| [Machine learning-based radiomic computed tomography phenotyping of thymic epithelial tumors: Predicting pathological and survival outcomes](https://www.sciencedirect.com/science/article/pii/S0022522322007978) | The Journal of Thoracic and Cardiovascular Surgery | 124 例 | 官方页面未确认多中心 | 否，bootstrap 内部验证 | 否，回顾性 | 预测 WHO 类型、TNM 分期、OS/PFS | CT radiomics + random forest / random survival forest | AUC、integrated AUC、Brier score | 官方页面未确认 | 将 TET AI 从分类推进到病理和生存结局预测 | 我们目前主要是风险分层 workflow，尚未达到预后预测层级 |
| [Development and Validation of Contrast-Enhanced CT-Based Deep Transfer Learning and Combined Clinical-Radiomics Model to Discriminate Thymomas and Thymic Cysts: A Multicenter Study](https://www.sciencedirect.com/science/article/abs/pii/S1076633223005536) | Academic Radiology | 官方摘要显示中心一训练 137、内部验证 59、中心二外部验证 68；初筛 398 | 是，两个中心 | 是，独立外部验证队列 | 否，回顾性 | 胸腺瘤 vs 胸腺囊肿鉴别 | deep transfer learning + clinical-radiomics | AUC、sensitivity、specificity、accuracy 等 | 官方页面未确认 | 展示多中心、外部验证、临床-radiomics 融合的标准写法 | 我们有风险控制框架优势，但缺这种清晰外部验证设计 |
| [Assessment of deep learning assistance for the pathological diagnosis of gastric cancer](https://www.nature.com/articles/s41379-022-01073-z) | Modern Pathology | 110 WSI reader study；16 名病理医生来自 12 家医院；模型训练含 2123 标注 WSI，测试还包括 3212 真实世界 WSI 和 2 个外部中心 1582 WSI | 是，医生来自 12 家医院，算法也有外部中心测试 | 是，算法有外部中心测试；reader study 为模拟临床评估 | 否，回顾性/模拟 reader study | 胃癌病理诊断的 AI 辅助效果评估 | DeepLab v3 病理 DL 系统 + 病理医生 MRMC reader study | ROC-AUC、sensitivity、specificity、阅片时间 | 部分开源：官方页面给出 GitHub 数据示例、R 代码和核心组件 | 对标“AI 不是替代医生，而是辅助医生”的医学工作流验证 | 我们项目若补医生读片对照，可接近其研究设计；目前还缺 reader study |
| [External Validation of Deep Learning Algorithms for Radiologic Diagnosis: A Systematic Review](https://pubs.rsna.org/doi/pdf/10.1148/ryai.210064) | Radiology: Artificial Intelligence | 系统综述，非单一数据集 | 是，综述层面 | 关注外部验证 | 否 | 总结医学影像 DL 外部验证问题 | Systematic review | 外部验证比例、性能下降等综述指标 | 不适用 | 证明外部验证不足和外部掉分是医学 AI 普遍问题 | 可用于解释 strict external 强制分类下降的合理性和重要性 |

### 7.2 计算机方向对标论文

| 论文 | 期刊 / 会议 | 数据量 | 是否多中心 | 是否外部验证 | 是否前瞻验证 | 任务 | 模型 | 指标 | 是否开源 | 主要贡献 | 我们项目能否达到 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [DINOv2: Learning Robust Visual Features without Supervision](https://openreview.net/forum?id=a68SUt6zFt) | TMLR / OpenReview | 大规模 curated image dataset；训练 1B 参数 ViT 并蒸馏小模型 | 非医学多中心概念 | 多 benchmark 泛化评估 | 否 | 通用自监督视觉特征学习 | DINOv2 self-supervised ViT | 多个 image-level / pixel-level benchmark | 是，官方页面给出 GitHub | 你们项目 DINOv2 frozen feature 的方法来源 | 不能作为本项目创新，只能作为 backbone 来源；需要证明 GrossPath-RC 的任务贡献 |
| [A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) | Nature | 170,000+ WSI，10 亿级 pathology image tiles | 是，真实世界 Providence 病理数据和公开 TCGA 等 | 多任务外部/公开数据评估 | 否 | 病理 WSI foundation model | Prov-GigaPath：DINOv2 tile pretraining + slide-level MAE / LongNet | 多个病理任务指标 | 是，官方页面给出模型权重和源码链接 | 病理 foundation model 高标准对标 | 我们数据规模远小；可对标特征使用和病理 foundation model 叙事，不能对标预训练规模 |
| [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) | Nature Medicine | 1.17M+ image-caption pairs；14 个 benchmark | 跨多来源病理数据 | 多 benchmark 迁移评估 | 否 | 病理图像-文本基础模型 | CONCH visual-language foundation model | 分类、分割、captioning、image-text retrieval 等 | 是，官方页面给出 HuggingFace 权重和 GitHub 代码 | 对标医生经验标签、概念描述、图文监督方向 | 我们当前没有大规模图文预训练；可借鉴其“经验标签/概念线索”未来方向 |
| [A multimodal whole-slide foundation model for pathology](https://www.nature.com/articles/s41591-025-03982-3) | Nature Medicine | 官方页面显示使用多模态 whole-slide 数据；具体规模见论文和补充材料 | 是，多来源数据 | 多任务评估 | 否 | 多模态 WSI foundation model | TITAN / TITAN_V | 诊断、检索、报告相关指标等 | 是，官方页面给出 GitHub 代码和权重 | 对标多模态病理 AI 的顶级方法贡献 | 当前项目远未达到，但可作为计算机方向长期标杆 |
| [Estimating diagnostic uncertainty in artificial intelligence assisted pathology using conformal prediction](https://www.nature.com/articles/s41467-022-34945-8) | Nature Communications | 病理 AI 数据来自前列腺病理相关数据；官方页面提供匿名 demo 和来源说明 | 官方页面涉及公开/合作数据；多中心信息需读全文确认 | 有外部/独立数据分析，具体以论文为准 | 否 | 病理 AI 诊断不确定性估计 | conformal prediction + 既有病理 DL 模型 | coverage、set size、诊断不确定性相关指标 | 部分开源：官方页面给出 GitHub 和 Zenodo；底层 DL 训练代码因隐私/合作未开放 | 与 v195 的复核/拒识、自动释放风险控制高度相关 | 我们可以在思想上接近，但需要把 v195 规则形式化为 conformal / selective prediction 风格 |
| [Fair Conformal Predictors for Applications in Medical Imaging](https://ojs.aaai.org/index.php/AAAI/article/view/21459) | AAAI Conference on Artificial Intelligence | dermatology photography dataset；另含 radiologist field interviews | 否，主要方法验证 | 否 | 否 | 医学影像不确定性和公平覆盖 | conformal prediction / equalized coverage | coverage、set size、subgroup fairness 等 | 官方页面未确认代码 | 对标“拒识/不确定性 + 临床用户需求”的计算机方法化表达 | 我们可借鉴其把医生需求转成不确定性方法；当前缺公平性和形式化覆盖指标 |
| [The limits of fair medical imaging AI in real-world generalization](https://www.nature.com/articles/s41591-024-03113-4) | Nature Medicine | 6 个全球 CXR 数据集，并扩展到 dermatology / ophthalmology；训练 3456 个模型 | 是，多数据集 | 是，OOD / external test settings | 否 | 医学影像公平性、shortcut、分布偏移泛化 | ERM、GroupDRO、DANN、CDANN、reweighting 等 | AUROC、FNR/FPR gap、fairness gap、OOD performance | 官方页面未确认代码 | 对标 strict external 域偏移与模型 shortcut 问题 | 我们有 strict external 下降证据，但缺如此系统的跨域/公平性实验 |
| [Applying Conformal Prediction to a Deep Learning Model for Intracranial Hemorrhage Detection to Improve Trustworthiness](https://pubs.rsna.org/doi/10.1148/ryai.240032) | Radiology: Artificial Intelligence | CT 颅内出血数据；具体样本量需看全文 | 官方页面未确认 | 官方页面未确认 | 否，回顾性 | ICH 检测可信度和困难切片识别 | DL + Mondrian conformal prediction | detection performance、uncertainty / challenging section accuracy | 官方页面未确认 | 医学影像中将 conformal prediction 用于提升可信度的近似对标 | 我们 v195 可转写为“自动释放 + 不确定复核”的可信 AI，但需补形式化指标 |
| [A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging](https://pubs.rsna.org/doi/full/10.1148/ryai.220232) | Radiology: Artificial Intelligence | 综述/方法指南，不是单一实验数据 | 不适用 | 不适用 | 不适用 | 医学影像 AI 交叉验证规范 | CV 方法综述；holdout、k-fold、nested CV 等 | prediction error、overoptimism、model selection bias | 是，官方页面给出示例代码仓库 | 可作为你们 OOF、fold-wise threshold、patient-level split 写法的规范依据 | 完全可达到；需要在报告中明确 patient-level split、fold-wise 规则和避免泄漏 |
| [Pathologist-level interpretable whole-slide cancer diagnosis with deep learning](https://www.nature.com/articles/s42256-019-0052-1) | Nature Machine Intelligence | WSI 癌症诊断数据；具体规模需看全文 | 官方页面未确认 | 官方页面未确认 | 否 | 可解释 WSI 癌症诊断 | interpretable deep learning WSI model | AUC、accuracy、pathologist-level comparison 等 | 官方页面未确认 | 对标“可解释 + 病理医生水平”叙事 | 我们目前没有 WSI、医生水平对照和强解释模块；只能作为远期对标 |

### 7.3 对照后的项目水准判断

| 对标维度 | 对标论文显示的高标准 | 当前项目状态 | 差距 / 下一步 |
| --- | --- | --- | --- |
| 同病种医学证据 | TET CT/radiomics 论文多强调多中心、外部验证、训练/验证/测试清晰划分 | 我们有大体图新颖性、Task7 workflow、strict external 压力测试 | 需要更大样本、多中心外部验证、正式队列流程图 |
| 临床辅助价值 | 胃癌病理 AI reader study 用 16 名病理医生验证 AI 辅助价值 | 我们目前有医生复核材料，但没有医生读片对照实验 | 应设计有/无 AI 的医生读片或报告复核实验 |
| 外部泛化 | 医学 AI 外部验证综述和 Nature Medicine 域泛化论文都强调外部掉分和 OOD 评估 | 我们已正面记录 strict external 强制分类下降 | 需要把外部下降从现象提升为系统域偏移分析 |
| 风险控制 / 拒识 | conformal prediction 论文用 coverage、set size、不确定性指标形式化拒识 | 我们已有 v195 自动覆盖、复核/拒识、自动错误和 Wilson95 | 需要将 v195 规则形式化，补充消融和覆盖-错误 tradeoff 曲线 |
| 计算机方法创新 | DINOv2、Prov-GigaPath、CONCH、TITAN 属于基础模型级贡献 | 我们主要使用现成强表征和规则式 workflow | 不宜主张 foundation model 创新；应主张 GrossPath-RC 工作流框架和小样本大体图应用 |
| 可复现性 | 高水平论文常提供代码、权重、数据子集或完整方法细节 | 我们目前是阶段报告集合 | 需要整理特征提取、阈值选择、v195 规则、Wilson95 计算和负结果表 |

---

## 汇总判断

### 面向医学期刊

当前项目已经接近一项较完整的回顾性医学 AI 工作流研究。它具备明确医学任务、队列分层、医生复核材料、clean/full sensitivity、strict external 压力测试和风险控制主流程。

当前最稳妥的论文主张是：

> 胸腺大体病理图像中存在可学习的低危/高危风险信号；在强制全量分类跨域泛化不足的情况下，基于多模型一致性、质量/偏移风险控制和复核/拒识机制的选择性诊断工作流，能够在自动释放部分保持较低错误率，同时把不确定病例交给医生复核。

暂不建议主张：

> 模型已经实现跨中心全量自动诊断，且准确率达到 99% 或 100%。

### 面向计算机期刊

当前项目的计算机侧潜力不在单纯“使用 DINO 做分类”，而在 GrossPath-RC 方向：小样本医学大体图、域偏移、选择性诊断、风险控制、自动释放和复核机制。

若要提升计算机期刊完成度，需要补强：

- 统一算法定义和伪代码。
- 严格消融实验。
- 跨域失败模式分析。
- 与普通 selective prediction、conformal risk control、domain generalization 方法的对照。
- 可复现的数据切分、特征提取、阈值选择和 Wilson95 计算流程。

