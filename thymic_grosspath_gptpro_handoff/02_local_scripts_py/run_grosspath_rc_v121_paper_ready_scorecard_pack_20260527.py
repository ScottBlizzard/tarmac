from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v121_paper_ready_scorecard_pack_20260527"

V116 = ROOT / "outputs" / "grosspath_rc_v116_evidence_tier_summary_20260527" / "v116_evidence_tier_summary.csv"
V118 = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_summary.csv"
V119_CONTRIB = ROOT / "outputs" / "grosspath_rc_v119_bidirectional_two_signal_map_20260527" / "v119_guard_contribution.csv"
V119_FIG_PDF = ROOT / "outputs" / "grosspath_rc_v119_bidirectional_two_signal_map_20260527" / "figures" / "v119_bidirectional_two_signal_map.pdf"
V119_FIG_PNG = ROOT / "outputs" / "grosspath_rc_v119_bidirectional_two_signal_map_20260527" / "figures" / "v119_bidirectional_two_signal_map.png"
V120 = ROOT / "outputs" / "grosspath_rc_v120_residual_fn_crop_rescue_validation_20260527" / "v120_residual_fn_crop_rescue_summary.csv"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    v116 = pd.read_csv(V116)
    v118 = pd.read_csv(V118)
    v120 = pd.read_csv(V120)
    contrib = pd.read_csv(V119_CONTRIB)

    primary_rows = []
    for label in ["v91 main locked", "v79 strict fixed", "v111 branch high-safety", "v115 nested stable candidate"]:
        row = v116[(v116["workflow_label"].eq(label)) & (v116["scope"].eq("all_domains"))].iloc[0].to_dict()
        primary_rows.append(
            {
                "workflow": label,
                "evidence_tier": row["evidence_tier"],
                "scope": "all_domains",
                "control_rate": row["control_rate"],
                "balanced_accuracy": row["balanced_accuracy"],
                "remaining_error_n": int(row["remaining_error_n"]),
                "fn": int(row["fn"]),
                "fp": int(row["fp"]),
            }
        )
    v118_all = v118[(v118["workflow"].eq("global_two_signal_scorecard")) & (v118["scope"].eq("all_domains"))].iloc[0]
    primary_rows.append(
        {
            "workflow": "v118/v119 global bidirectional two-signal scorecard",
            "evidence_tier": "deployable candidate after nested support",
            "scope": "all_domains",
            "control_rate": float(v118_all["control_rate"]),
            "balanced_accuracy": float(v118_all["balanced_accuracy"]),
            "remaining_error_n": int(v118_all["remaining_error_n"]),
            "fn": int(v118_all["fn"]),
            "fp": int(v118_all["fp"]),
        }
    )
    v120_post = v120[
        (v120["workflow"].eq("posthoc_crop_rescue_upper_bound")) & (v120["scope"].eq("all_domains"))
    ].iloc[0]
    primary_rows.append(
        {
            "workflow": "v120 post-hoc crop rescue upper bound",
            "evidence_tier": "post-hoc upper bound only",
            "scope": "all_domains",
            "control_rate": float(v120_post["control_rate"]),
            "balanced_accuracy": float(v120_post["balanced_accuracy"]),
            "remaining_error_n": int(v120_post["remaining_error_n"]),
            "fn": int(v120_post["fn"]),
            "fp": int(v120_post["fp"]),
        }
    )
    primary = pd.DataFrame(primary_rows)

    domain_rows = []
    for scope in ["old_data", "third_batch", "strict_external"]:
        row = v118[(v118["workflow"].eq("global_two_signal_scorecard")) & (v118["scope"].eq(scope))].iloc[0]
        domain_rows.append(row.to_dict())
    domain = pd.DataFrame(domain_rows)

    claim_tiers = pd.DataFrame(
        [
            {
                "tier": "locked primary",
                "workflow": "v91 main locked",
                "claim": "主流程证据最干净，适合做正式基线主结论",
                "boundary": "性能低于后续高安全候选，但选择边界更干净",
            },
            {
                "tier": "deployable high-safety candidate",
                "workflow": "v118/v119 global bidirectional two-signal scorecard",
                "claim": "统一两信号双向评分卡可将 all-domain BAcc 提至 99.81%，FP 清零",
                "boundary": "同一数据内开发，仍需前瞻外部验证",
            },
            {
                "tier": "post-hoc upper bound",
                "workflow": "v120 post-hoc crop rescue upper bound",
                "claim": "crop 信号理论上可救回最后一个 FN，使当前三域 0 错误",
                "boundary": "嵌套验证失败，不能作为正式流程",
            },
        ]
    )

    primary.to_csv(OUT_DIR / "v121_primary_evidence_table.csv", index=False, encoding="utf-8-sig")
    format_metrics(primary).to_csv(OUT_DIR / "v121_primary_evidence_table_formatted.csv", index=False, encoding="utf-8-sig")
    domain.to_csv(OUT_DIR / "v121_v118_domain_breakdown.csv", index=False, encoding="utf-8-sig")
    format_metrics(domain).to_csv(OUT_DIR / "v121_v118_domain_breakdown_formatted.csv", index=False, encoding="utf-8-sig")
    contrib.to_csv(OUT_DIR / "v121_guard_contribution.csv", index=False, encoding="utf-8-sig")
    claim_tiers.to_csv(OUT_DIR / "v121_claim_tiers.csv", index=False, encoding="utf-8-sig")

    for src in [V119_FIG_PDF, V119_FIG_PNG]:
        if src.exists():
            shutil.copy2(src, fig_dir / src.name)

    v118_line = format_metrics(pd.DataFrame([v118_all.to_dict()])).iloc[0]
    post_line = format_metrics(pd.DataFrame([v120_post.to_dict()])).iloc[0]
    lines = [
        "# v121 Paper-ready Scorecard Pack",
        "",
        "这份汇总包用于后续论文/汇报写作，核心是把不同证据等级分开：",
        "",
        f"- 锁定主流程：v91 all-domain BAcc 98.24%，控制率 74.11%。",
        f"- 当前最适合写成高安全候选的方法：v118/v119 双向两信号评分卡，all-domain BAcc {v118_line['balanced_accuracy']}，控制率 {v118_line['control_rate']}，FN={int(v118_all['fn'])}，FP={int(v118_all['fp'])}。",
        f"- 后验上限：v120 crop rescue，all-domain BAcc {post_line['balanced_accuracy']}，控制率 {post_line['control_rate']}，FN={int(v120_post['fn'])}，FP={int(v120_post['fp'])}，但嵌套验证未通过。",
        "",
        "建议写作顺序：先报 v91 主流程，再报 v118/v119 作为方法改进和高安全候选，最后把 v120 放在上限/消融段落，不作为正式部署流程。",
    ]
    (OUT_DIR / "v121_paper_ready_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(format_metrics(primary).to_string(index=False))
    print()
    print(claim_tiers.to_string(index=False))


if __name__ == "__main__":
    main()
