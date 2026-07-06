from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V27 = ROOT / "outputs" / "grosspath_rc_v27_unified_workflow_20260527" / "v27_unified_workflow_metrics.csv"
V37 = ROOT / "outputs" / "grosspath_rc_v37_rank_normalized_risk_20260527" / "v37_rank_controller_selected_targets_external_eval.csv"
V40 = ROOT / "outputs" / "grosspath_rc_v40_risk_ranker_baselines_20260527" / "v40_ranker_efficiency_summary.csv"
V42 = ROOT / "outputs" / "grosspath_rc_v42_dev_selection_bootstrap_20260527" / "v42_bootstrap_summary.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v43_final_strategy_table_20260527"


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v27 = pd.read_csv(V27)
    v37 = pd.read_csv(V37)
    v40 = pd.read_csv(V40)
    v42 = pd.read_csv(V42)

    rows = []
    for policy, label, note in [
        ("P0_main_auto_all", "P0 主模型纯自动", "纯自动参考。"),
        ("P2_safety_switch_auto", "P2 安全切换纯自动", "无复核负担的最佳纯自动基线。"),
        ("P7_logistic_recall90_quality_or_safety_review", "P7 高召回质量/安全复核", "上一阶段高安全策略，复核量较大。"),
    ]:
        r = v27.loc[v27["policy"].eq(policy)].iloc[0]
        rows.append(
            {
                "strategy": label,
                "type": "baseline_or_previous",
                "external_review_rate_pct": pct(r["risk_control_rate"]),
                "external_accuracy_pct": pct(r["workflow_accuracy"]),
                "external_bacc_pct": pct(r["workflow_balanced_accuracy"]),
                "external_sensitivity_pct": pct(r["workflow_sensitivity"]),
                "external_specificity_pct": pct(r["workflow_specificity"]),
                "external_fn": int(r["workflow_fn"]),
                "external_fp": int(r["workflow_fp"]),
                "bootstrap_bacc_ci95_pct": "",
                "minimum_review_for_bacc90_pct": "",
                "minimum_review_for_bacc93_pct": "",
                "note": note,
            }
        )

    for target, label in [(0.95, "v37 rank controller dev95"), (0.97, "v37 rank controller dev97")]:
        r = v37.loc[v37["target_dev_bacc"].eq(target)].iloc[0]
        b = v42.loc[v42["target_dev_bacc"].eq(target)].iloc[0]
        rows.append(
            {
                "strategy": label,
                "type": "proposed_rank_controller",
                "external_review_rate_pct": pct(r["external_review_rate"]),
                "external_accuracy_pct": pct(r["external_accuracy"]),
                "external_bacc_pct": pct(r["external_balanced_accuracy"]),
                "external_sensitivity_pct": pct(r["external_sensitivity"]),
                "external_specificity_pct": pct(r["external_specificity"]),
                "external_fn": int(r["external_fn"]),
                "external_fp": int(r["external_fp"]),
                "bootstrap_bacc_ci95_pct": f"{pct(b['external_bacc_ci025'])}-{pct(b['external_bacc_ci975'])}",
                "minimum_review_for_bacc90_pct": "",
                "minimum_review_for_bacc93_pct": "",
                "note": "开发集目标选择复核比例，外部按风险排名比例执行。",
            }
        )

    learned = v40.loc[v40["ranker"].eq("learned_hard_gate")].iloc[0]
    rows.append(
        {
            "strategy": "learned hard gate efficiency",
            "type": "ranker_efficiency",
            "external_review_rate_pct": "",
            "external_accuracy_pct": "",
            "external_bacc_pct": "",
            "external_sensitivity_pct": "",
            "external_specificity_pct": "",
            "external_fn": "",
            "external_fp": "",
            "bootstrap_bacc_ci95_pct": "",
            "minimum_review_for_bacc90_pct": pct(learned["min_review_for_bacc90"]),
            "minimum_review_for_bacc93_pct": pct(learned["min_review_for_bacc93"]),
            "note": "达到同等外部BAcc所需复核比例最低，优于低置信度/分歧基线。",
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v43_final_strategy_table.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v43 最终策略汇总表",
        "",
        "这张表用于后续医生汇报和论文主表草稿。P0/P2 是纯自动基线，P7 是上一阶段高复核策略，v37 是当前推荐的 rank-normalized risk controller。",
        "",
        "| strategy | type | review % | Acc % | BAcc % | Sens % | Spec % | FN | FP | bootstrap BAcc 95%CI | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in out.iterrows():
        md.append(
            f"| {r['strategy']} | {r['type']} | {r['external_review_rate_pct']} | {r['external_accuracy_pct']} | {r['external_bacc_pct']} | "
            f"{r['external_sensitivity_pct']} | {r['external_specificity_pct']} | {r['external_fn']} | {r['external_fp']} | {r['bootstrap_bacc_ci95_pct']} | {r['note']} |"
        )
    (OUT_DIR / "v43_final_strategy_table.md").write_text("\n".join(md), encoding="utf-8")

    print(out.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
