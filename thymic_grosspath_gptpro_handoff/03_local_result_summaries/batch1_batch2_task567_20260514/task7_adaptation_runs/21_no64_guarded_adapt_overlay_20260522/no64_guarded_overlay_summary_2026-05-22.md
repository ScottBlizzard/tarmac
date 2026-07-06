# No64 guarded third adaptation overlay

Old main is fixed to No.64. Reconstructed old OOF is ACC 0.9263 / BACC 0.9263. Third metrics here use the 234-case adapt72 holdout only.

|Policy|Old ACC|Old BACC|Third ACC|Third BACC|TN/FP/FN/TP|Note|
|---|---:|---:|---:|---:|---|---|
|third_base_on_adapt72_holdout|0.9263|0.9263|0.7436|0.7090|149/47/13/25|No.64 old protected; third uses old-only 64-style proxy base on the 234-case adapt72 holdout|
|best_acc_guard92|0.9263|0.9262|0.8718|0.7113|186/10/20/18|adapt_r2_c0.0003, t=0.62, route=base_low_margin, old_budget=2|
|best_bacc_guard92|0.9228|0.9227|0.8419|0.7359|175/21/16/22|adapt_r4_c0.0003, t=0.58, route=base_low_margin, old_budget=2|
|tp_preserved_guard92|0.9228|0.9227|0.7521|0.7142|151/45/13/25|adapt_r2_c0.0003, t=0.57, route=candidate_range, old_budget=2|

Interpretation: the large ACC gain mainly comes from reducing false positives; it can reduce high-risk TP. If TP must be preserved, only a small gain remains.