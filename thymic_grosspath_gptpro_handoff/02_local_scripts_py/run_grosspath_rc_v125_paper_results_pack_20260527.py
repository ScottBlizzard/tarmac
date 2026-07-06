from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v125_paper_results_pack_20260527"

V121_PRIMARY = ROOT / "outputs" / "grosspath_rc_v121_paper_ready_scorecard_pack_20260527" / "v121_primary_evidence_table.csv"
V121_CLAIMS = ROOT / "outputs" / "grosspath_rc_v121_paper_ready_scorecard_pack_20260527" / "v121_claim_tiers.csv"
V122_ABLATION = ROOT / "outputs" / "grosspath_rc_v122_two_signal_ablation_robustness_20260527" / "v122_ablation_workflow_summary.csv"
V122_ROBUST = ROOT / "outputs" / "grosspath_rc_v122_two_signal_ablation_robustness_20260527" / "v122_robustness_summary.csv"
V123_PAIR = ROOT / "outputs" / "grosspath_rc_v123_paired_stats_confidence_20260527" / "v123_paired_delta_stats.csv"
V123_CI = ROOT / "outputs" / "grosspath_rc_v123_paired_stats_confidence_20260527" / "v123_metric_wilson_ci.csv"
V124_FRONTIER = ROOT / "outputs" / "grosspath_rc_v124_review_efficiency_frontier_20260527" / "v124_min_budget_to_capture_all_fp.csv"

FIGURES = [
    ROOT / "outputs" / "grosspath_rc_v119_bidirectional_two_signal_map_20260527" / "figures" / "v119_bidirectional_two_signal_map.pdf",
    ROOT / "outputs" / "grosspath_rc_v119_bidirectional_two_signal_map_20260527" / "figures" / "v119_bidirectional_two_signal_map.png",
    ROOT / "outputs" / "grosspath_rc_v122_two_signal_ablation_robustness_20260527" / "figures" / "v122_two_signal_remaining_error_heatmap.pdf",
    ROOT / "outputs" / "grosspath_rc_v122_two_signal_ablation_robustness_20260527" / "figures" / "v122_two_signal_remaining_error_heatmap.png",
    ROOT / "outputs" / "grosspath_rc_v124_review_efficiency_frontier_20260527" / "figures" / "v124_review_efficiency_frontier.pdf",
    ROOT / "outputs" / "grosspath_rc_v124_review_efficiency_frontier_20260527" / "figures" / "v124_review_efficiency_frontier.png",
]


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in [
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "best_control_rate",
            "best_balanced_accuracy",
            "base_bacc",
            "new_bacc",
            "delta_bacc",
            "delta_bacc_ci_low",
            "delta_bacc_ci_high",
            "sensitivity_ci_low",
            "sensitivity_ci_high",
            "specificity_ci_low",
            "specificity_ci_high",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
        ]:
            out[col] = out[col].map(pct)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    primary = pd.read_csv(V121_PRIMARY)
    claims = pd.read_csv(V121_CLAIMS)
    ablation = pd.read_csv(V122_ABLATION)
    robust = pd.read_csv(V122_ROBUST)
    paired = pd.read_csv(V123_PAIR)
    ci = pd.read_csv(V123_CI)
    frontier = pd.read_csv(V124_FRONTIER)

    main_primary = primary[
        primary["workflow"].isin(
            [
                "v91 main locked",
                "v118/v119 global bidirectional two-signal scorecard",
                "v120 post-hoc crop rescue upper bound",
            ]
        )
    ].copy()

    ablation_focus = ablation[
        ablation["workflow"].isin(
            [
                "v118_fixed_two_signal",
                "selected_wholecrop_only",
                "selected_core_only",
                "selected_two_signal",
            ]
        )
    ].copy()
    paired_focus = paired[paired["scope"].isin(["all_domains", "third_batch", "strict_external"])].copy()
    ci_focus = ci[
        ci["workflow"].eq("v118_global_two_signal") & ci["scope"].isin(["all_domains", "third_batch", "strict_external"])
    ].copy()

    main_primary.to_csv(OUT_DIR / "v125_main_result_table.csv", index=False, encoding="utf-8-sig")
    format_metrics(main_primary).to_csv(OUT_DIR / "v125_main_result_table_formatted.csv", index=False, encoding="utf-8-sig")
    ablation_focus.to_csv(OUT_DIR / "v125_two_signal_ablation_table.csv", index=False, encoding="utf-8-sig")
    format_metrics(ablation_focus).to_csv(OUT_DIR / "v125_two_signal_ablation_table_formatted.csv", index=False, encoding="utf-8-sig")
    frontier.to_csv(OUT_DIR / "v125_review_efficiency_frontier_table.csv", index=False, encoding="utf-8-sig")
    format_metrics(frontier).to_csv(OUT_DIR / "v125_review_efficiency_frontier_table_formatted.csv", index=False, encoding="utf-8-sig")
    paired_focus.to_csv(OUT_DIR / "v125_paired_delta_table.csv", index=False, encoding="utf-8-sig")
    format_metrics(paired_focus).to_csv(OUT_DIR / "v125_paired_delta_table_formatted.csv", index=False, encoding="utf-8-sig")
    ci_focus.to_csv(OUT_DIR / "v125_wilson_ci_table.csv", index=False, encoding="utf-8-sig")
    format_metrics(ci_focus).to_csv(OUT_DIR / "v125_wilson_ci_table_formatted.csv", index=False, encoding="utf-8-sig")
    claims.to_csv(OUT_DIR / "v125_claim_tiers.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(OUT_DIR / "v125_two_signal_robustness_summary.csv", index=False, encoding="utf-8-sig")

    for src in FIGURES:
        if src.exists():
            shutil.copy2(src, fig_dir / src.name)

    v118 = main_primary[main_primary["workflow"].eq("v118/v119 global bidirectional two-signal scorecard")].iloc[0]
    v91 = main_primary[main_primary["workflow"].eq("v91 main locked")].iloc[0]
    v120 = main_primary[main_primary["workflow"].eq("v120 post-hoc crop rescue upper bound")].iloc[0]
    two = frontier[frontier["mode"].eq("two_signal")].iloc[0]
    whole = frontier[frontier["mode"].eq("wholecrop_only")].iloc[0]
    core = frontier[frontier["mode"].eq("core_only")].iloc[0]
    pair = paired[paired["scope"].eq("all_domains")].iloc[0]
    robust_row = robust.iloc[0]

    lines = [
        "# v125 Paper Results Pack",
        "",
        "本包用于论文 Results / 医生汇报的固定结果入口，重点是把性能、方法消融、效率前沿和证据边界放在一起。",
        "",
        "## 主结果",
        "",
        f"- 锁定主流程 v91：all-domain BAcc {pct(v91['balanced_accuracy'])}，控制率 {pct(v91['control_rate'])}，FN={int(v91['fn'])}，FP={int(v91['fp'])}。",
        f"- 当前高安全候选 v118/v119：all-domain BAcc {pct(v118['balanced_accuracy'])}，控制率 {pct(v118['control_rate'])}，FN={int(v118['fn'])}，FP={int(v118['fp'])}。",
        f"- 后验上限 v120：all-domain BAcc {pct(v120['balanced_accuracy'])}，控制率 {pct(v120['control_rate'])}，FN={int(v120['fn'])}，FP={int(v120['fp'])}，但嵌套验证未通过。",
        "",
        "## 方法证据",
        "",
        f"- two-signal FP guard 抓回全部 3 个 FP 只需 {int(two['min_extra_review_budget'])} 个额外复核；wholecrop-only/core-only 分别需要 {int(whole['min_extra_review_budget'])}/{int(core['min_extra_review_budget'])} 个。",
        f"- 控制率 <=80.5% 且 FP=0 的 two-signal 安全阈值组合共有 {int(robust_row['two_signal_safe_rules_control_le_80_5_n'])} 组，说明不是单点偶然。",
        f"- 同病例配对：v118 相比 v111 救回 {int(pair['base_wrong_new_correct'])} 个错误、误伤 {int(pair['base_correct_new_wrong'])} 个，ΔBAcc {pct(pair['delta_bacc'])}，bootstrap CI {pct(pair['delta_bacc_ci_low'])} 到 {pct(pair['delta_bacc_ci_high'])}。",
        "",
        "## 推荐写法",
        "",
        "主文可把 v118/v119 写成“风险可控的双向两信号评分卡”，强调复核效率和同病例错误减少；v120 只作为上限/消融，不能作为正式部署流程。",
    ]
    (OUT_DIR / "v125_paper_results_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")

    print("Wrote", OUT_DIR)
    print(format_metrics(main_primary).to_string(index=False))
    print()
    print(format_metrics(frontier).to_string(index=False))


if __name__ == "__main__":
    main()
