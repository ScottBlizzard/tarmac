from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V37_DIR = ROOT / "outputs" / "grosspath_rc_v37_rank_normalized_risk_20260527"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v41_review_priority_lists_20260527"


def risk_reason(row: pd.Series) -> str:
    p2 = int(row["p2_pred"])
    main_prob = float(row.get("main_prob", np.nan))
    robust_prob = float(row.get("robust_prob", np.nan))
    core_prob = float(row.get("prob_mean_core", np.nan))
    reasons = []
    if p2 == 0 and robust_prob >= 0.57:
        reasons.append("主模型低危但鲁棒分支偏高危")
    if p2 == 1 and core_prob <= 0.50:
        reasons.append("自动高危但核心平均概率偏低")
    if abs(main_prob - robust_prob) >= 0.20:
        reasons.append("主模型与鲁棒分支分歧较大")
    if row.get("quality_status") == "borderline":
        reasons.append("质量/构图边界")
    if not reasons:
        reasons.append("综合风险排序靠前")
    return "；".join(reasons)


def public_pred_label(x: int) -> str:
    return "高危" if int(x) == 1 else "低危"


def make_lists(cases: pd.DataFrame, target: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = cases.loc[cases["target_dev_bacc"].eq(target) & cases["review_flag"].eq(1)].copy()
    sub = sub.sort_values("hard_risk_score", ascending=False).reset_index(drop=True)
    sub.insert(0, "priority_rank", np.arange(1, len(sub) + 1))
    sub["auto_prediction"] = sub["p2_pred"].map(public_pred_label)
    sub["review_reason"] = sub.apply(risk_reason, axis=1)
    sub["truth_label_group"] = sub["label_idx"].map(public_pred_label)

    doctor_cols = [
        "priority_rank",
        "original_case_id",
        "image_name",
        "auto_prediction",
        "quality_status",
        "quality_score",
        "hard_risk_rank_pct_external",
        "review_reason",
    ]
    analysis_cols = doctor_cols + [
        "task_l6_label",
        "task_l7_label",
        "truth_label_group",
        "p2_wrong",
        "p2_error_direction",
        "route_bucket",
        "main_prob",
        "robust_prob",
        "prob_mean_core",
        "hard_risk_score",
    ]
    return sub[doctor_cols], sub[analysis_cols]


def topk_summary(cases: pd.DataFrame, target: float) -> pd.DataFrame:
    sub = cases.loc[cases["target_dev_bacc"].eq(target)].copy().sort_values("hard_risk_score", ascending=False).reset_index(drop=True)
    rows = []
    for k in [5, 10, 20, 30, 40, 50, 60, len(sub)]:
        k = min(k, len(sub))
        top = sub.iloc[:k]
        wrong = top["p2_wrong"].astype(int).eq(1)
        rows.append(
            {
                "target_dev_bacc": target,
                "top_k": k,
                "top_k_rate": k / len(sub),
                "p2_wrong_n": int(wrong.sum()),
                "p2_wrong_precision": float(wrong.mean()),
                "fn_high_to_low_n": int(top["p2_error_direction"].eq("FN_high_to_low").sum()),
                "fp_low_to_high_n": int(top["p2_error_direction"].eq("FP_low_to_high").sum()),
                "ab_n": int(top["task_l6_label"].eq("AB").sum()),
                "b2_or_b2b3_n": int(top["task_l6_label"].isin(["B2", "B2_B3_mixed"]).sum()),
                "borderline_quality_n": int(top["quality_status"].eq("borderline").sum()),
            }
        )
    return pd.DataFrame(rows)


def write_md(summary: pd.DataFrame, analysis_95: pd.DataFrame, analysis_97: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        view = df.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
            else:
                view[col] = view[col].astype(str)
        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy(dtype=str)]
        return "\n".join([header, sep] + rows)

    md = [
        "# 2026-05-27 v41 外部集复核优先级清单",
        "",
        "目的：把 v37/v40 的 rank-normalized risk controller 转成可操作的病例优先级。这里的清单不用于训练，只用于解释风险门控排在前面的病例是否确实集中在关键边界错例。",
        "",
        "## Top-K 错例富集",
        "",
        md_table(summary),
        "",
        "## 95%目标复核池前 15 例",
        "",
        md_table(analysis_95.head(15)),
        "",
        "## 97%目标复核池前 15 例",
        "",
        md_table(analysis_97.head(15)),
        "",
        "## 当前判断",
        "",
        "风险排序前列并不是随机病例，主要富集在 AB、B2/B2-B3 以及质量/构图边界病例。该清单可以直接转成医生复核优先级：先看排序靠前的病例，判断是拍照/取材问题、真实边界病例，还是模型应学习的稳定模式。",
    ]
    (OUT_DIR / "v41_review_priority_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(V37_DIR / "v37_rank_controller_case_routes_external.csv")
    doctor_95, analysis_95 = make_lists(cases, 0.95)
    doctor_97, analysis_97 = make_lists(cases, 0.97)
    summary = pd.concat([topk_summary(cases, 0.95), topk_summary(cases, 0.97)], ignore_index=True)

    doctor_95.to_csv(OUT_DIR / "v41_doctor_blinded_priority_target95.csv", index=False, encoding="utf-8-sig")
    doctor_97.to_csv(OUT_DIR / "v41_doctor_blinded_priority_target97.csv", index=False, encoding="utf-8-sig")
    analysis_95.to_csv(OUT_DIR / "v41_internal_priority_with_truth_target95.csv", index=False, encoding="utf-8-sig")
    analysis_97.to_csv(OUT_DIR / "v41_internal_priority_with_truth_target97.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v41_topk_error_enrichment_summary.csv", index=False, encoding="utf-8-sig")
    write_md(summary, analysis_95, analysis_97)

    print(summary.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
