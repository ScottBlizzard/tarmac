# Prompt for GPTPro

你是一个顶级医学 AI / 计算病理 / 机器学习研究负责人。请完整阅读我给你的这个 GitHub 仓库中的交接包：

`thymic_grosspath_gptpro_handoff/`

你的任务不是保守总结，而是从非常开放的角度重新思考：如何提升胸腺大体图像低危/高危二分类基础模型的跨域泛化能力。

## 项目背景

这是一个胸腺大体病理图像 AI 项目。核心临床任务是 Task7：

- 低危：A / AB / B1
- 高危：B2 / B3 / 胸腺癌

当前项目已经做了很多工作：SuRImage/CNN baseline、DINOv2/DINOv3 frozen features、MLP/logreg probes、cut-only、whole/crop、多图策略、视图分治、课程学习、经验标签、第三批适配、严格外部测试、risk-controlled selective workflow、v195/v195+ 拒识/复核工作流等。

现在我希望你暂时不要把主线继续局限在“提高自动放行比例”或“调拒识规则”上。那些 workflow 可以作为后续安全层，但现在真正的主线是：

**基础强制分类模型本身必须更强，尤其是严格外部和新外部数据上的泛化能力必须提升。**

## 当前数据边界

请按以下数据边界理解，不要混用：

1. `old_data`
   - 285 例/图
   - 低危 144，高危 141
   - 内部开发/OOF 主体

2. `third_batch`
   - 306 例/图
   - 低危 224，高危 82
   - 同体系新增数据，已经参与观察、适配、策略选择
   - 不能再称为 strict external

3. `strict_external`
   - 108 例/图
   - 低危 61，高危 47
   - 冻结外部压力测试
   - 原则上不能用于训练、调参、模型选择

4. `new_external_160`
   - 原始上传 166 文件
   - case-level 去重后 162 例
   - 低危 77，高危 85
   - A 22, AB 29, B1 26, B2 28, B3 29, TC 28
   - 已完成处理/QC，但还没有正式跑模型推理

## 已知核心指标

请从交接包中核对细节。先给你一个定位：

- 旧数据上最强域内模型/流程可以到约 92.3% Acc/BAcc。
- 第三批经过适配后整体约 83.0% Acc，BAcc 约 76.8%，但高危召回仍弱。
- 严格外部强制分类约 64.8% Acc，BAcc 约 62.8%，高危召回约 46.8%。
- v195/v195+ 工作流能通过自动放行/复核控制错误，但这不是 full-coverage base classifier 的胜利。

## 你要优先阅读的文件

请先读：

1. `README.md`
2. `01_local_reports_md_csv/项目阶段性工作详报.md`
3. `01_local_reports_md_csv/Task7项目完成度评估.md`
4. `01_local_reports_md_csv/三批数据集外部泛化拒识增强方案报告_.md`
5. `01_local_reports_md_csv/Task7模型升级负结果阶段汇总_2026-05-16.md`
6. `01_local_reports_md_csv/2026-05-18_Task7课程学习与经验标签阶段汇报_医生版.md`
7. `01_local_reports_md_csv/2026-05-25_Task7第三批与严格外部集后续尝试阶段汇报_医生版.md`
8. `01_local_reports_md_csv/2026-05-26_GrossPath-RC_v2域泛化诊断与训练清单.md`
9. `04_server_snapshot/task_plan.md`
10. `04_server_snapshot/progress.md`
11. `04_server_snapshot/findings.md`
12. `04_server_snapshot/experiments/risk_control_rejection_20260621/reports/`
13. `04_server_snapshot/experiments/risk_control_rejection_20260621/v195_plus_sidecar/`
14. `04_server_snapshot/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/` 中的 summary/metrics 表

然后再按需看：

- `02_local_scripts_py/`
- `04_server_snapshot/scripts/`
- `03_local_result_summaries_flat/`

## 请特别注意已经尝试过的方向

下面这些不是禁止你再想，而是请你不要低水平重复。你可以提出更高级版本，但要说明为什么不是旧实验的简单复刻。

已经试过且多数不稳定/未解决外部泛化的方向包括：

- SuRImage / SE-ResNeXt / CNN / CAM crop
- WHO 六分类先做再合并二分类
- DINOv2 frozen feature + logreg/MLP
- DINOv2/DINOv3 微调，包括 last-block、head-only、full fine-tune、QKVB、TTA
- cut-only、whole-only、whole+crop、full-to-cut refine
- view-aware routing、cut/outer/mixed specialist
- PLIP、BiomedCLIP、ConvNeXt、Swin、EfficientNet、EVA02、SigLIP、ViTamin、AIMv2 等横向替换或融合
- 弱经验标签、大规模 hard 样本加权、核心 hard 直接加入训练
- stacking、guard、domain-internal reliability selector
- selective release / rejection / auto-review workflow

## 我希望你做的事

请用中文输出一个研究负责人级别的方案，不要只做摘要。请完成以下内容：

1. 先复盘项目真实问题
   - 用你自己的话判断：现在到底是数据问题、模型能力问题、域偏移问题、标注边界问题、输入视角问题，还是多因素叠加？
   - 哪些历史结论你同意，哪些你认为可能误判？

2. 重新打开思维空间
   - 不要过早收窄到 DINO + probe。
   - 允许大胆发散，包括但不限于：更强视觉基础模型、病理/医学 foundation model、SAM/MedSAM/YOLO/DETR/grounding 模型辅助 ROI、multi-instance learning、多图病例建模、自监督/对比学习、无监督域适应、test-time adaptation、source-free adaptation、domain adversarial、style randomization、颜色/背景/尺度增强、合成数据、扩散模型增强、主动学习、医生弱监督、概念瓶颈、多任务学习、ordinal/clinical-risk loss、B2/B3/TC 专门建模、open-set/OOD 分解、外部域模拟等。

3. 给出一份大范围实验矩阵
   - 至少 20 个候选方向。
   - 每个方向写：核心想法、为什么可能解决外部泛化、与历史尝试的区别、所需数据/代码、预期风险、成功标准。

4. 给出优先级排序
   - 从 20+ 个方向中选出最值得先跑的 8-12 个。
   - 不要只按“最稳”排序，也要考虑可能带来突破的高风险方向。

5. 设计严格评估协议
   - 必须区分 old_data、third_batch、strict_external、new_external_160。
   - strict_external 和 new_external_160 不得用于训练/调参/模型选择。
   - 必须报告 Acc、BAcc、AUC、高危召回、低危特异度、FN、FP。
   - 任何 workflow/rejection 结果必须和 full-coverage forced classification 分开报告。
   - 给出如何避免 patient leakage、multi-image leakage、threshold leakage。

6. 给出可执行计划
   - 第一周跑什么，第二周跑什么，第三周跑什么。
   - 需要改哪些脚本或新增哪些脚本。
   - 该从哪些现有文件/结果表开始复用。
   - 哪些旧实验可以直接作为 baseline，不需要重跑。

7. 给出论文策略
   - 如果基础模型外部泛化能提升到可接受水平，医工交叉论文主线怎么写？
   - 如果仍然提升有限，如何把它写成“外部泛化挑战 + 风险控制 + 数据治理”的扎实论文？
   - 哪些结果绝对不能夸大？

## 输出格式

请按以下结构输出：

1. `项目真实瓶颈判断`
2. `历史实验复盘与可能误判点`
3. `广义方法空间：20+ 候选实验`
4. `优先级最高的 8-12 个实验`
5. `严格评估协议`
6. `三周执行计划`
7. `论文主线建议`
8. `最需要医生/同伴补充的信息`

请尽量具体，不要停留在“可以试试某模型”。如果建议换模型基底，请说清楚候选模型家族、输入策略、训练方式、融合方式、评价方式、和为什么它不是已失败 backbone sweep 的重复。

