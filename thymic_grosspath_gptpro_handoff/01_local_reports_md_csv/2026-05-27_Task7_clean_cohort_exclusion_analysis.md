# Task7 Clean Cohort Exclusion Analysis

## Cohort Definition

根据医生复核意见，我们建立固定 exclusion list。原始数据不物理删除；主分析使用 clean primary cohort，完整队列作为 full-cohort sensitivity analysis。

## Exclusion List Status

- 2404716：MEC，不在当前 Task7 主流程中。
- 2307206：淋巴上皮癌，在当前 Task7 主流程中。
- 2205101：淋巴上皮癌，在当前 Task7 主流程中。
- 2203278：微结节型TC，不在当前 Task7 主流程中。
- 2113767：肠型腺癌，不在当前 Task7 主流程中。

## Main v195 Result

- Full cohort: n=699, BAcc 99.81%, Acc 99.86%, FN=1, FP=0, review/reject 36.05%, auto errors 1.
- Clean cohort: n=697, BAcc 99.81%, Acc 99.86%, FN=1, FP=0, review/reject 36.01%, auto errors 1.
- Clean old_data: n=283, BAcc 100.00%, review/reject 26.15%.

## Interpretation

剔除医生指出的特殊组织学/混杂病例后，当前主结果没有变差。由于当前主流程中实际命中的 exclusion 只有 2205101 和 2307206，且二者原本都被模型判为高危正确，clean cohort 的主要作用是让队列定义更干净，而不是人为提高分数。

论文写法建议：主文报告 clean cohort；补充材料报告 full cohort sensitivity analysis，说明结论对这些特殊病例剔除不敏感。