# v131 Final Results Pack with Leave-one-FP-out Evidence

本包在 v129 基础上补入 v130 留一 FP 错例验证，作为当前最完整的 Results 入口。

## v130 稳定性补充

- min-review：held-out FP 抓回 2/3，平均控制率 79.83%，最差 FP=1。
- capped stable-envelope：held-out FP 抓回 3/3，平均控制率 80.59%，最差 FP=0。

## 当前写作边界

v118/v119 仍是主写的高安全候选；v128/v130 作为稳定性补充，说明带上限稳定包络能够在内部留域和留一错例场景中保持 FP=0。由于 held-out FP 只有 3 个，该证据不能替代真正多中心前瞻验证。