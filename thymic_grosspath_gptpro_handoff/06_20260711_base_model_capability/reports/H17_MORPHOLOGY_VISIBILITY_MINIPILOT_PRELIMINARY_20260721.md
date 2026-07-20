# H17 20 例形态可见性 Mini-Pilot 初步结果

日期：2026-07-21

状态：`PENDING INDEPENDENT SECOND READER`

性质：研究阶段 morphology-only 可见性审计，不是临床诊断。

## 1. 当前完成情况

H17-0 历史文字 probe 失败后，已按预注册生成 20 例 morphology-only mini-pilot：

- 不使用旧模型错误；
- 不使用旧模型置信度；
- 不使用 persistent-hard 身份；
- 低危/高危各 10 例，仅用于私有抽样平衡；
- batch1/batch2/third_batch 为 7/7/6；
- 五折各 4 例；
- 六亚型均覆盖；
- annotator 不看风险、亚型、来源或模型输出；
- packet 中没有风险判断和置信度字段；
- 严格外部未读取。

服务器 packet：

`/workspace/thymic_project/experiments/h17_spatial_morphology_risk_20260720/morphology_visibility_minipilot20_20260721`

## 2. GPT 第一遍盲态可见性审计

本地 20 张图已复制为匿名 `MVxxx` 文件。GPT 只标记当前照片中形态是否可判断，不判断低危/高危。

该结果只能作为第一遍非临床审计，不能替代两名独立医学读者。

| 概念 | 可判断数 | 可判断率 | 初步门槛 |
|---|---:|---:|---|
| M1 包膜/边界完整清楚 | 9/20 | 45% | 失败 |
| M2 包膜中断/边界不清 | 12/20 | 60% | 失败 |
| M3 苍白均质切面 | 15/20 | 75% | 通过 |
| M4 异质斑驳切面 | 17/20 | 85% | 通过 |
| M5 出血/坏死/囊变 | 16/20 | 80% | 通过 |
| M6 分叶/结节/纤维隔 | 17/20 | 85% | 通过 |
| M7 邻近组织连续 | 6/20 | 30% | 失败 |
| M8 内部异常是否局限 | 5/20 | 25% | 失败 |
| M9 干扰伪影 | 20/20 | 100% | 负控制 |
| M10 有效主切面 | 19/20 | 95% | 质量控制 |

全局形态可判断：

- `yes`：13/20；
- `partial`：4/20；
- `no`：3/20。

## 3. 最关键的结果

至少四个非伪影概念达到 70% 可见性：

- M3 苍白/均质；
- M4 异质/斑驳；
- M5 出血/坏死/囊变；
- M6 分叶/结节/纤维隔。

但 H16 最缺失的空间风险方向概念没有达到可见性门槛：

- M1/M2 包膜边界；
- M7 邻近组织连续；
- M8 内部异常相对边界的位置。

这说明：

1. 单张照片较容易展示颜色、切面纹理、出血和分叶；
2. 决定“内部异常是否到达边界、是否跨界、是否联系邻近组织”的信息经常没有被拍到；
3. 模型持续依赖颜色和局部显著区域，是因为这些信息最稳定可见；
4. 仅增加 M3-M6 监督可能让模型更准确描述大体外观，但不一定能纠正风险方向；
5. 若独立读者也确认 M1/M2/M7/M8 可见性低，标准化多视图采集应上升为主线，而不是继续训练同一单图。

## 4. 尚未通过的门槛

当前不能宣布 mini-pilot 通过，因为还缺：

- 独立第二读者；
- Gwet AC1/Cohen kappa；
- Z1/Z2/Z3 ROI IoU；
- 对 `not_visible_or_uncertain` 使用的一致性；
- M3-M6 的跨读者重复性。

服务器校验器当前正式状态：

`PENDING_INDEPENDENT_MORPHOLOGY_ANNOTATION`

## 5. 已完成工程

已生成：

- 20 张匿名图像；
- 两份 morphology-only reader 表；
- M1-M10、R1-R6 和 Z0-Z5 schema；
- secure case key；
- 文件 SHA256；
- 固定抽样配置；
- 自动 validator；
- GPT 非临床第一遍 CSV；
- 四页盲态 contact sheet。

本地 GPT 初步审计：

`D:\影响分析\artifacts\h17_morphology_visibility_minipilot20_20260721\GPT_PRELIMINARY_MORPHOLOGY_VISIBILITY.csv`

## 6. 下一步决策

只需完成 20 例的独立 morphology-only 表，不需要医生重新判断风险或复核模型错误。

若至少四个 M1-M8 概念同时满足：

- 两名 reader 各自可判断率 `>=70%`；
- Gwet AC1 或 Cohen kappa `>=0.60`；
- 对应区域 ROI 一致性通过；

则进入 60 例 pilot 和 G0 形态风险信息量 oracle。

若最终只有 M3-M6 通过，而 M1/M2/M7/M8 继续失败，则不应立即训练风险模型。应先验证 M3-M6 对 H10 是否有独立增量；增量不足时，转向标准化多视图采集。
