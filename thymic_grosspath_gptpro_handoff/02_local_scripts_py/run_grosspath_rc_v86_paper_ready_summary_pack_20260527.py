from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v86_paper_ready_summary_pack_20260527"
NOTE = ROOT / "reports" / "notes" / "grosspath_rc_v72_v74_crossdomain_quality_strategy_20260527.md"

V80 = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527" / "v80_tiered_lowrisk_guard_summary.csv"
V81 = ROOT / "outputs" / "grosspath_rc_v81_tiered_workflow_bootstrap_ci_20260527" / "v81_tiered_workflow_bootstrap_ci_summary.csv"
V82 = ROOT / "outputs" / "grosspath_rc_v82_unlabeled_adaptive_workflow_20260527" / "v82_fixed_and_adaptive_summary.csv"
V83 = ROOT / "outputs" / "grosspath_rc_v83_workflow_decision_utility_20260527" / "v83_utility_named_scenario_rankings.csv"
V84 = ROOT / "outputs" / "grosspath_rc_v84_module_ablation_contribution_20260527" / "v84_fixed_workflow_module_delta_by_domain.csv"
V85 = ROOT / "outputs" / "grosspath_rc_v85_paired_delta_significance_20260527" / "v85_paired_delta_ci_summary.csv"
V77 = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527" / "v77_unlabeled_batch_shift_audit.csv"


POLICY_LABELS = {
    "v50_main": "Standard risk-control workflow (v50)",
    "v75_quality_lowconf": "High-risk miss protection (v75)",
    "v79_light_lowrisk_guard": "Bidirectional light guard (v79-light)",
    "v79_strict_lowrisk_guard": "Bidirectional strict guard (v79-strict)",
    "adaptive_v75_on_shift": "Adaptive v50->v75",
    "adaptive_light_on_shift": "Adaptive v50->v79-light",
    "adaptive_strict_on_shift": "Adaptive v50->v79-strict",
}


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_main_table(v80: pd.DataFrame, v81: pd.DataFrame) -> pd.DataFrame:
    ci_cols = [
        "domain",
        "policy",
        "balanced_accuracy_wilson_low",
        "balanced_accuracy_wilson_high",
        "sensitivity_wilson_low",
        "sensitivity_wilson_high",
        "specificity_wilson_low",
        "specificity_wilson_high",
    ]
    merged = v80.merge(v81[ci_cols], left_on=["split", "policy"], right_on=["domain", "policy"], how="left")
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "domain": r["split"],
                "workflow": POLICY_LABELS.get(r["policy"], r["policy"]),
                "policy_id": r["policy"],
                "control_rate": pct(float(r["control_rate"])),
                "balanced_accuracy": pct(float(r["balanced_accuracy"])),
                "balanced_accuracy_wilson_95ci": f"{pct(float(r['balanced_accuracy_wilson_low']))}-{pct(float(r['balanced_accuracy_wilson_high']))}",
                "sensitivity": pct(float(r["sensitivity"])),
                "sensitivity_wilson_95ci": f"{pct(float(r['sensitivity_wilson_low']))}-{pct(float(r['sensitivity_wilson_high']))}",
                "specificity": pct(float(r["specificity"])),
                "specificity_wilson_95ci": f"{pct(float(r['specificity_wilson_low']))}-{pct(float(r['specificity_wilson_high']))}",
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "remaining_errors": int(r["remaining_error_n"]),
                "claim_tier": "main" if r["policy"] in ["v50_main", "v79_light_lowrisk_guard"] else "candidate",
            }
        )
    return pd.DataFrame(rows)


def build_external_table(main_table: pd.DataFrame) -> pd.DataFrame:
    return main_table.loc[main_table["domain"].eq("strict_external")].copy()


def build_adaptive_table(v82: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in v82.loc[v82["scope"].eq("all_domains")].iterrows():
        rows.append(
            {
                "workflow": POLICY_LABELS.get(r["policy"], r["policy"]),
                "policy_id": r["policy"],
                "overall_control_rate": pct(float(r["control_rate"])),
                "overall_balanced_accuracy": pct(float(r["balanced_accuracy"])),
                "overall_sensitivity": pct(float(r["sensitivity"])),
                "overall_specificity": pct(float(r["specificity"])),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "remaining_errors": int(r["remaining_error_n"]),
            }
        )
    return pd.DataFrame(rows)


def build_module_table(v84: pd.DataFrame) -> pd.DataFrame:
    out = v84.copy()
    for col in ["delta_control_rate", "delta_bacc", "delta_sensitivity", "delta_specificity"]:
        out[col] = out[col].map(lambda x: f"{float(x) * 100:+.2f} pp")
    return out[
        [
            "domain",
            "module_label",
            "delta_control_rate",
            "delta_bacc",
            "delta_sensitivity",
            "delta_specificity",
            "delta_fn",
            "delta_fp",
            "delta_remaining_error_n",
        ]
    ]


def build_stats_table(v85: pd.DataFrame) -> pd.DataFrame:
    focus = v85.loc[
        v85["pair_id"].isin(["v79_light_vs_v50", "v79_strict_vs_v50", "v79_light_vs_v75"])
    ].copy()
    rows = []
    for _, r in focus.iterrows():
        rows.append(
            {
                "domain": r["domain"],
                "comparison": r["comparison"],
                "delta_bacc": f"{float(r['delta_balanced_accuracy']) * 100:+.2f} pp",
                "delta_bacc_bootstrap_95ci": f"{float(r['delta_balanced_accuracy_ci025']) * 100:+.2f} to {float(r['delta_balanced_accuracy_ci975']) * 100:+.2f} pp",
                "delta_fn": int(r["delta_fn"]),
                "delta_fp": int(r["delta_fp"]),
                "mcnemar_p": f"{float(r['mcnemar_p']):.3f}",
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    main: pd.DataFrame,
    external: pd.DataFrame,
    adaptive: pd.DataFrame,
    audit: pd.DataFrame,
    module: pd.DataFrame,
    stats: pd.DataFrame,
    utility: pd.DataFrame,
) -> Path:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        text_df = df.copy()
        for col in text_df.columns:
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(text_df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in text_df.to_numpy(dtype=str)]
        return "\n".join([header, sep] + rows)

    md = OUT_DIR / "v86_paper_ready_summary.md"
    lines = [
        "# Task7 Paper-ready Summary Pack (v86)",
        "",
        "## Primary Fixed Workflows",
        md_table(main),
        "",
        "## Strict External Focus",
        md_table(external),
        "",
        "## Unlabeled Batch Audit",
        md_table(
            audit[
                [
                    "audit_name",
                    "reference",
                    "target",
                    "domain_auc_cv",
                    "quality_proxy_mean",
                    "batch_shift_index",
                    "shift_category",
                    "recommended_policy",
                ]
            ]
        ),
        "",
        "## Adaptive Workflows",
        md_table(adaptive),
        "",
        "## Module Ablation",
        md_table(module),
        "",
        "## Paired Delta Statistics",
        md_table(stats),
        "",
        "## Decision Utility Named Scenarios",
        md_table(utility),
        "",
        "## Claim Boundary",
        "- Main reportable workflow: v50 baseline and v79-light bidirectional guard.",
        "- High-safety candidate: v79-strict, because strict external reaches 100% but still needs more external batches.",
        "- Deployment-oriented framework: v82 adaptive routing, because it uses unlabeled batch shift audit before choosing the safety tier.",
        "- Exploratory-only results remain excluded from the primary table if thresholds were chosen after inspecting strict external performance.",
    ]
    md.write_text("\n".join(lines), encoding="utf-8-sig")
    return md


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v80 = pd.read_csv(V80)
    v81 = pd.read_csv(V81)
    v82 = pd.read_csv(V82)
    v83 = pd.read_csv(V83)
    v84 = pd.read_csv(V84)
    v85 = pd.read_csv(V85)
    v77 = pd.read_csv(V77)

    main_table = build_main_table(v80, v81)
    external_table = build_external_table(main_table)
    adaptive_table = build_adaptive_table(v82)
    module_table = build_module_table(v84)
    stats_table = build_stats_table(v85)
    utility_focus = v83.loc[v83["rank"].eq(1)].copy()

    main_table.to_csv(OUT_DIR / "v86_primary_fixed_workflow_table.csv", index=False, encoding="utf-8-sig")
    external_table.to_csv(OUT_DIR / "v86_strict_external_focus_table.csv", index=False, encoding="utf-8-sig")
    adaptive_table.to_csv(OUT_DIR / "v86_adaptive_workflow_table.csv", index=False, encoding="utf-8-sig")
    module_table.to_csv(OUT_DIR / "v86_module_ablation_table.csv", index=False, encoding="utf-8-sig")
    stats_table.to_csv(OUT_DIR / "v86_paired_delta_stats_table.csv", index=False, encoding="utf-8-sig")
    utility_focus.to_csv(OUT_DIR / "v86_decision_utility_best_scenarios.csv", index=False, encoding="utf-8-sig")
    md = write_markdown(main_table, external_table, adaptive_table, v77, module_table, stats_table, utility_focus)

    print(f"Saved paper-ready summary pack to: {OUT_DIR}")
    print(f"Markdown: {md}")
    print("\nStrict external focus:")
    print(external_table.to_string(index=False))


if __name__ == "__main__":
    main()
