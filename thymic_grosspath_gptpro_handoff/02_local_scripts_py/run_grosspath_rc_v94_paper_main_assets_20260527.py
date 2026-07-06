from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
V91 = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_summary.csv"
V92 = ROOT / "outputs" / "grosspath_rc_v92_integrated_paired_stats_20260527" / "v92_integrated_paired_stats_summary.csv"
V93 = ROOT / "outputs" / "grosspath_rc_v93_leave_domain_out_selection_20260527" / "v93_leave_domain_out_selection.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v94_paper_main_assets_20260527"
FIG_DIR = OUT_DIR / "figures"


MAIN_POLICIES = [
    "pure_auto",
    "v50_main",
    "adaptive_v50_to_v79_light",
    "v79_light_lowrisk_guard",
    "v79_strict_lowrisk_guard",
    "quality_direction_uniform90",
]

CLAIM_TIER = {
    "pure_auto": ("Baseline", "Shows why direct automatic diagnosis is insufficient across domains."),
    "v50_main": ("Stable baseline workflow", "Lowest-cost workflow satisfying basic cross-domain constraints in v93."),
    "adaptive_v50_to_v79_light": ("Primary deployable workflow", "Unlabeled batch audit keeps internal review lower and escalates severe-shift batches."),
    "v79_light_lowrisk_guard": ("Primary high-safety fixed workflow", "Selected by v93 high-safety constraints when strict external is held out."),
    "v79_strict_lowrisk_guard": ("High-specificity candidate", "Removes low-risk overcall at higher review rate; needs further external validation."),
    "quality_direction_uniform90": ("High-review upper-bound candidate", "Best current all-domain error reduction, but review/control rate is near 90%."),
}


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in MAIN_POLICIES:
        all_row = summary.loc[summary["scope"].eq("all_domains") & summary["policy"].eq(policy)].iloc[0]
        ext_row = summary.loc[summary["scope"].eq("strict_external") & summary["policy"].eq(policy)].iloc[0]
        tier, interpretation = CLAIM_TIER[policy]
        rows.append(
            {
                "policy": policy,
                "workflow": all_row["policy_label"],
                "claim_tier": tier,
                "all_control_rate": pct(all_row["control_rate"]),
                "all_bacc": pct(all_row["balanced_accuracy"]),
                "all_sensitivity": pct(all_row["sensitivity"]),
                "all_specificity": pct(all_row["specificity"]),
                "all_fn": int(all_row["fn"]),
                "all_fp": int(all_row["fp"]),
                "all_remaining_errors": int(all_row["remaining_error_n"]),
                "strict_external_control_rate": pct(ext_row["control_rate"]),
                "strict_external_bacc": pct(ext_row["balanced_accuracy"]),
                "strict_external_sensitivity": pct(ext_row["sensitivity"]),
                "strict_external_specificity": pct(ext_row["specificity"]),
                "strict_external_fn": int(ext_row["fn"]),
                "strict_external_fp": int(ext_row["fp"]),
                "strict_external_remaining_errors": int(ext_row["remaining_error_n"]),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_claim_table(v92: pd.DataFrame, v93: pd.DataFrame) -> pd.DataFrame:
    strict_basic = v93.loc[v93["heldout_domain"].eq("strict_external") & v93["rule"].eq("basic_cross_domain_min_review")].iloc[0]
    strict_high = v93.loc[v93["heldout_domain"].eq("strict_external") & v93["rule"].eq("high_safety_balanced_min_review")].iloc[0]
    qd_vs_light = v92.loc[
        v92["scope"].eq("all_domains")
        & v92["candidate"].eq("quality_direction_uniform90")
        & v92["baseline"].eq("v79_light_lowrisk_guard")
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "claim": "Basic cross-domain workflow can be selected without external tuning",
                "evidence": f"v93 strict-external-held-out selection chooses {strict_basic['selected_policy_label']}; held-out BAcc {pct(strict_basic['heldout_bacc'])}.",
                "status": "main",
            },
            {
                "claim": "High-safety workflow can also be selected without external tuning",
                "evidence": f"v93 strict-external-held-out high-safety selection chooses {strict_high['selected_policy_label']}; held-out BAcc {pct(strict_high['heldout_bacc'])}, FN={int(strict_high['heldout_fn'])}.",
                "status": "main",
            },
            {
                "claim": "Unlabeled batch-adaptive workflow is a deployment compromise",
                "evidence": "v91 Batch-adaptive main: all-domain control 74.11%, BAcc 98.24%; strict external BAcc 99.18%.",
                "status": "main",
            },
            {
                "claim": "High-review upper-bound can nearly eliminate all-domain errors",
                "evidence": "v91 Quality+direction uniform90: all-domain BAcc 99.88%, FN=0, FP=1, control 89.84%.",
                "status": "candidate",
            },
            {
                "claim": "Upper-bound improvement is promising but not primary",
                "evidence": f"v92 vs v79-light: delta BAcc {pct(qd_vs_light['delta_bacc'])}, 95% CI {pct(qd_vs_light['delta_bacc_ci_low'])} to {pct(qd_vs_light['delta_bacc_ci_high'])}, McNemar p={qd_vs_light['mcnemar_p']:.3f}.",
                "status": "candidate, cautious",
            },
        ]
    )


def draw_box(ax, xy, width, height, text, face="#E8EEF5", edge="#557A95", fontsize=8.5):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def make_main_figure(summary: pd.DataFrame, v93: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("A. Batch-adaptive selective diagnosis framework", loc="left", fontsize=11, fontweight="bold")
    draw_box(ax, (0.05, 0.68), 0.28, 0.18, "Gross pathology image\n+ base classifier", "#F2F4F7")
    draw_box(ax, (0.39, 0.68), 0.25, 0.18, "Unlabeled batch\nshift audit", "#E8EEF5")
    draw_box(ax, (0.72, 0.76), 0.22, 0.12, "Internal-like:\nlow-review v50", "#DFF0D8", "#5A8F5A")
    draw_box(ax, (0.72, 0.56), 0.22, 0.12, "Severe shift:\nescalate v79-light", "#FDEBD0", "#B9770E")
    draw_box(ax, (0.39, 0.30), 0.25, 0.16, "Error-risk ranking\nand selective review", "#EAF2F8")
    draw_box(ax, (0.72, 0.26), 0.22, 0.12, "High-review\nupper-bound option", "#F9EAF5", "#B03A8E")
    for start, end in [
        ((0.33, 0.77), (0.39, 0.77)),
        ((0.64, 0.77), (0.72, 0.82)),
        ((0.64, 0.77), (0.72, 0.62)),
        ((0.515, 0.68), (0.515, 0.46)),
        ((0.64, 0.38), (0.72, 0.32)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.set_title("B. Overall trade-off", loc="left", fontsize=11, fontweight="bold")
    plot = summary.loc[summary["scope"].eq("all_domains") & summary["policy"].isin(MAIN_POLICIES)].copy()
    colors = ["#666666", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#e377c2"]
    for color, (_, row) in zip(colors, plot.iterrows()):
        ax.scatter(row["control_rate"] * 100, row["balanced_accuracy"] * 100, s=90, color=color, label=row["policy_label"])
    ax.set_xlabel("Review/control rate (%)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[1, 0]
    ax.set_title("C. Strict external residual errors", loc="left", fontsize=11, fontweight="bold")
    ext = summary.loc[summary["scope"].eq("strict_external") & summary["policy"].isin(MAIN_POLICIES[1:])].copy()
    ax.bar(ext["policy_label"], ext["remaining_error_n"], color=colors[1:])
    ax.set_ylabel("Remaining errors")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.set_title("D. Strict external held-out selection", loc="left", fontsize=11, fontweight="bold")
    strict_sel = v93.loc[v93["heldout_domain"].eq("strict_external") & v93["selected"].eq(True)].copy()
    strict_sel = strict_sel.loc[
        strict_sel["rule"].isin(["basic_cross_domain_min_review", "high_safety_balanced_min_review", "zero_error_if_possible"])
    ]
    ax.scatter(strict_sel["heldout_control_rate"] * 100, strict_sel["heldout_bacc"] * 100, s=85, color="#1f77b4")
    for _, row in strict_sel.iterrows():
        ax.annotate(row["selected_policy_label"], (row["heldout_control_rate"] * 100, row["heldout_bacc"] * 100), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Held-out control rate (%)")
    ax.set_ylabel("Held-out BAcc (%)")
    ax.set_xlim(72.0, 93.5)
    ax.set_ylim(97.15, 99.35)
    ax.grid(alpha=0.25)

    fig.suptitle("Risk-controlled gross pathology diagnosis: main evidence map", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "v94_main_evidence_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v94_main_evidence_map.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary_md(main: pd.DataFrame, claims: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = [str(row[c]).replace("\n", "<br>") for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = ["# v94 Paper Main Assets", ""]
    lines.append("## Main Message")
    lines.append("")
    lines.append(
        "The current strongest paper storyline is a risk-controlled, batch-adaptive selective diagnosis framework rather than a single classifier. "
        "The primary deployable workflow is Batch-adaptive main; Fixed v79-light is the primary high-safety fixed workflow; Quality+direction uniform90 is a high-review upper-bound candidate."
    )
    lines.append("")
    lines.append("## Claim Tiers")
    lines.append("")
    lines.append(md_table(claims))
    lines.append("")
    lines.append("## Main Result Table")
    lines.append("")
    lines.append(md_table(main[["workflow", "claim_tier", "all_control_rate", "all_bacc", "all_fn", "all_fp", "strict_external_control_rate", "strict_external_bacc", "strict_external_fn", "strict_external_fp"]]))
    (OUT_DIR / "v94_paper_main_assets_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(V91)
    v92 = pd.read_csv(ROOT / "outputs" / "grosspath_rc_v92_integrated_paired_stats_20260527" / "v92_integrated_paired_stats_summary.csv")
    v93 = pd.read_csv(V93)
    main_table = build_main_table(summary)
    claim_table = build_claim_table(v92, v93)
    main_table.to_csv(OUT_DIR / "v94_main_result_table.csv", index=False, encoding="utf-8-sig")
    claim_table.to_csv(OUT_DIR / "v94_claim_tier_table.csv", index=False, encoding="utf-8-sig")
    make_main_figure(summary, v93)
    write_summary_md(main_table, claim_table)

    print("v94 main result table:")
    print(
        main_table[
            [
                "workflow",
                "claim_tier",
                "all_control_rate",
                "all_bacc",
                "all_fn",
                "all_fp",
                "strict_external_control_rate",
                "strict_external_bacc",
                "strict_external_fn",
                "strict_external_fp",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
