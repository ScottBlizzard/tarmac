# Phase 1 实验报告：Risk-Controlled Rejection Layer

日期：2026-06-21

## 1. 实验目的

本次实验按照既定方案 `docs/superpowers/specs/2026-06-21-risk-controlled-rejection-layer-design.md` 开始 Phase 1。

目标不是降低 strict external 拒识率，也不是优化外部集分数，而是建立一个隔离、可复现、可审计的风险控制实验框架：

- 统一生成病例级风险特征表。
- 复刻 v195 当前主结果，作为对照基线。
- 生成初步 coverage-risk 曲线，用于观察不同风险排序信号下的自动判读风险。
- 保持 strict external 为冻结审计集，不用其标签进行阈值或特征选择。

## 2. 隔离原则

本次实验全部放在新目录下：

```text
experiments/risk_control_rejection_20260621/
```

遵守以下约束：

- 没有修改原项目既有代码文件。
- 没有覆盖原项目既有输出文件。
- 原始 `scripts/`、`outputs/`、`reports/`、`datasets/` 仅作为只读输入。
- 新脚本、新测试、新输出和新报告全部写入本实验目录。

## 3. 新增文件

### 3.1 计划文件

```text
experiments/risk_control_rejection_20260621/plans/phase1_implementation_plan.md
```

作用：

- 固定本轮 Phase 1 的执行步骤。
- 明确输入文件、输出文件和验证标准。

### 3.2 新脚本

```text
experiments/risk_control_rejection_20260621/scripts/build_case_risk_features.py
experiments/risk_control_rejection_20260621/scripts/audit_rejection_layer.py
```

`build_case_risk_features.py` 的作用：

- 读取既有 v201 case-level 表和 v77 batch shift 审计表。
- 生成统一病例级风险特征表。
- 输出 batch shift 特征表。

`audit_rejection_layer.py` 的作用：

- 读取新生成的 `case_risk_features.csv`。
- 复刻 v195 selective diagnosis 行为。
- 生成初步风险排序 coverage-risk 曲线。
- 输出 Phase 1 审计汇总和报告。

### 3.3 新测试

```text
experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py
```

测试内容：

- Wilson 区间计算。
- 风险均值排序函数。
- 二分类 entropy 计算。

## 4. 只读输入来源

本次实验读取了以下既有项目产物：

```text
outputs/grosspath_rc_v201_stable_supported_domain_flip_20260527/v201_stable_supported_domain_flip_cases.csv
outputs/grosspath_rc_v77_batch_shift_audit_policy_switch_20260527/v77_unlabeled_batch_shift_audit.csv
outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv
outputs/grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527/v195_selected_candidate_action_cases.csv
```

这些文件没有被修改。

## 5. 生成输出

本次实验生成以下输出：

```text
experiments/risk_control_rejection_20260621/outputs/case_risk_features.csv
experiments/risk_control_rejection_20260621/outputs/batch_shift_features.csv
experiments/risk_control_rejection_20260621/outputs/case_risk_features_report.json
experiments/risk_control_rejection_20260621/outputs/v195_reconstructed_case_outputs.csv
experiments/risk_control_rejection_20260621/outputs/v195_baseline_reconstruction.csv
experiments/risk_control_rejection_20260621/outputs/coverage_risk_curves.csv
experiments/risk_control_rejection_20260621/outputs/phase1_audit_summary.csv
experiments/risk_control_rejection_20260621/outputs/phase1_audit_report.json
```

## 6. 实验结果

### 6.1 病例级风险特征表

生成的 `case_risk_features.csv` 共 699 行、31 列。

域分布：

| Domain | n |
|---|---:|
| old_data | 285 |
| third_batch | 306 |
| strict_external | 108 |

核心特征包括：

- `prob_mean_core`
- `base_confidence`
- `base_uncertainty`
- `base_entropy`
- `model_disagreement_rate`
- `corrector_confidence_mean`
- `corrector_prob_high_mean`
- `router_risk_mean`
- `router_risk_max`
- `batch_shift_index`
- `domain_auc_cv`
- `mean_outside_ref_05_95_rate`
- `quality_proxy_mean`

其中 batch shift 特征显示：

| Domain | batch_shift_index | domain_auc_cv | Gate |
|---|---:|---:|---|
| old_data | 0.9695 | 0.5317 | normal_or_known_shift |
| third_batch | 0.5232 | 0.5170 | normal_or_known_shift |
| strict_external | 2.3242 | 0.9945 | severe_unknown_shift |

这与既有 v77/v156 的结论一致：strict external 是明显 severe shift。

### 6.2 v195 基线复刻

本次脚本成功复刻 v195 当前主结果。

| Scope | n | Auto | Review/Reject | Auto Error | Auto FN High | Auto FP Low | BAcc | Review Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old_data | 285 | 210 | 75 | 0 | 0 | 0 | 100.00% | 26.32% |
| third_batch | 306 | 185 | 121 | 1 | 1 | 0 | 99.39% | 39.54% |
| strict_external | 108 | 52 | 56 | 0 | 0 | 0 | 100.00% | 51.85% |
| all_domains | 699 | 447 | 252 | 1 | 1 | 0 | 99.81% | 36.05% |

关键校验点：

- all-domain：447 自动判读，252 拒识，auto error=1。
- strict external：52 自动判读，56 拒识，auto error=0。
- 这些数字与既有 v195/v202 主线一致。

### 6.3 strict external Wilson 边界

strict external 下 v195 为：

```text
auto error = 0 / 52
Wilson 95% upper = 6.88%
```

这说明虽然当前 strict external 观测自动错误为 0，但由于样本量较小，置信上界仍然较宽。因此后续不能把“0 错误”写成已充分证明的临床安全率，只能写成当前冻结审计观察。

### 6.4 初步 coverage-risk 曲线

本次生成了 112 行 coverage-risk 曲线，覆盖：

- 4 种风险排序分数：
  - `uncertainty_score`
  - `disagreement_score`
  - `router_score`
  - `hybrid_score`
- 7 个目标 review rate：
  - 20%、30%、40%、50%、60%、70%、80%
- 4 个 scope：
  - old_data
  - third_batch
  - strict_external
  - all_domains

这些排序只是审计基线，不是部署模型。

all-domain 主要观察：

| Risk Score | Review Rate | Auto | Review | Auto Error | Auto FN High | BAcc |
|---|---:|---:|---:|---:|---:|---:|
| disagreement_score | 70% | 210 | 489 | 1 | 1 | 99.81% |
| router_score | 80% | 140 | 559 | 1 | 1 | 99.81% |
| hybrid_score | 80% | 140 | 559 | 3 | 3 | 99.44% |
| uncertainty_score | 80% | 140 | 559 | 3 | 3 | 99.44% |

初步解读：

- 简单风险排序信号可以在较高 review rate 下接近 v195 的 auto error 水平。
- 但在 30%-60% review rate 下，auto error 明显高于 v195。
- 这说明当前这些简单排序信号不能直接替代 v195，也不能作为降低 strict external 拒识率的依据。

### 6.5 strict external 风险排序观察

strict external 上，在 50%-80% review rate 的简单排序审计中，当前表面上仍保持 auto error=0。

但这只是冻结审计观察，不能作为调参依据，也不能写成 coverage 提升结论。原因：

- strict external 只有 108 例。
- 当前所有排序分数都来自已知历史产物。
- 如果后续根据 strict external 表现调整特征或阈值，会形成间接过拟合。

## 7. 验证

执行了以下验证命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python -m unittest experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py -v
```

结果：

```text
Ran 3 tests in 1.052s
OK
```

随后执行输出完整性校验：

```text
features_shape (699, 31)
v195_all_auto_review_error 447 252 1
v195_strict_auto_review_error 52 56 0
curve_rows 112
```

验证结论：

- 新脚本能正常运行。
- 输出文件齐全。
- v195 复刻指标正确。
- coverage-risk 曲线生成成功。

## 8. 当前结论

本次实验完成了 Phase 1 的第一个最小闭环：

1. 成功建立隔离实验目录。
2. 成功生成统一病例级风险特征表。
3. 成功复刻 v195 主结果。
4. 成功生成初步 coverage-risk 曲线。
5. 验证了简单风险排序不能直接替代 v195。

最重要的结论是：

> 当前阶段不应追求降低 strict external 拒识率。更合理的方向是继续完善统一风险特征接口、加强 old+third 内部 nested audit，并把 strict external 保持为冻结审计集。

## 9. 下一步建议

下一步不建议直接调 strict external 阈值。

建议继续做 Phase 1 的第二步：

1. 加入更严格的 old+third nested audit。
2. 将 `confirmed_error`、`confirmed_safe`、`unlabeled_or_uncertain` 三类标签显式写入特征表。
3. 建立风险排序的内部选择规则，只在 old+third 上选择。
4. 生成 per-domain 和 per-fold 的 coverage-risk 稳定性表。
5. strict external 继续只做冻结报告，不参与任何选择。

