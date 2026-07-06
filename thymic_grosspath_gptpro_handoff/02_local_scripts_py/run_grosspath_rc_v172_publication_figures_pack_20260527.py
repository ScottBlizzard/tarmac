from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v172_publication_figures_pack_20260527"
FIG_DIR = OUT_DIR / "figures"
V170_MAIN = ROOT / "outputs" / "grosspath_rc_v170_paper_evidence_tier_pack_20260527" / "v170_main_operating_table_tiered.csv"
V170_MODULE = ROOT / "outputs" / "grosspath_rc_v170_paper_evidence_tier_pack_20260527" / "v170_module_tier_table.csv"
V162_POOL = ROOT / "outputs" / "grosspath_rc_v162_frontier_after_safe_release_20260527" / "v162_operating_point_pool_with_safe_release.csv"
V164_ABLATION = ROOT / "outputs" / "grosspath_rc_v164_stable_safe_release_strategy_compare_20260527" / "v164_all_domain_strategy_focus.csv"
V158_GATE = ROOT / "outputs" / "grosspath_rc_v158_unlabeled_gate_threshold_stability_20260527" / "v158_gate_component_stability.csv"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def short_point(name: str) -> str:
    if name == "locked_high_safety_two_signal_scorecard":
        return "High-safety\nscorecard"
    if name == "v161_safe_release_scorecard":
        return "Safe-release\nscorecard"
    if name.startswith("baseline_low_conf_dev_selected"):
        return "Low-conf\nreview"
    if name.startswith("dev_stable_router_all_domains"):
        return "Dev-stable\nrouter"
    if name.startswith("shift_aware_image_directional_candidate"):
        return "Image-direction\nrouter"
    if name == "severe_shift_gated_concept_direction_autocorrect":
        return "Severe-shift\ncorrection"
    return name.replace("_", "\n")


def export_tables() -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(V170_MAIN)
    pool = pd.read_csv(V162_POOL)
    ablation = pd.read_csv(V164_ABLATION)
    modules = pd.read_csv(V170_MODULE)
    gate = pd.read_csv(V158_GATE)

    main.to_csv(OUT_DIR / "v172_source_main_operating_table.csv", index=False, encoding="utf-8-sig")
    pool.to_csv(OUT_DIR / "v172_source_operating_frontier_pool.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(OUT_DIR / "v172_source_safe_release_ablation.csv", index=False, encoding="utf-8-sig")
    modules.to_csv(OUT_DIR / "v172_source_module_tiers.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(OUT_DIR / "v172_source_gate_stability.csv", index=False, encoding="utf-8-sig")
    return {
        "main": OUT_DIR / "v172_source_main_operating_table.csv",
        "frontier": OUT_DIR / "v172_source_operating_frontier_pool.csv",
        "ablation": OUT_DIR / "v172_source_safe_release_ablation.csv",
        "modules": OUT_DIR / "v172_source_module_tiers.csv",
        "gate": OUT_DIR / "v172_source_gate_stability.csv",
    }


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        return plt, FancyArrowPatch, FancyBboxPatch, None
    except Exception as exc:  # pragma: no cover - server may lack matplotlib.
        return None, None, None, exc


def draw_pipeline(plt, FancyArrowPatch, FancyBboxPatch) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_axis_off()
    boxes = [
        ("Input gross\npathology image", 0.03, 0.55, 0.14, 0.26, "#E6F6FF"),
        ("Primary image\nclassifier", 0.23, 0.55, 0.14, 0.26, "#D9E2EC"),
        ("Two-signal\nrisk scorecard", 0.43, 0.55, 0.14, 0.26, "#D9EAD3"),
        ("Safe-release\ncandidate", 0.63, 0.72, 0.14, 0.20, "#FFF3BF"),
        ("Review / reject\nfallback", 0.63, 0.42, 0.14, 0.20, "#FFE3E3"),
        ("Output with\nrisk boundary", 0.83, 0.55, 0.14, 0.26, "#EDE7F6"),
        ("Unlabeled shift gate\n+ concept correction\n(candidate branch)", 0.43, 0.12, 0.34, 0.20, "#F3F4F6"),
    ]
    for text, x, y, w, h, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                linewidth=1.0,
                edgecolor="#334E68",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="#102A43")
    arrows = [
        ((0.17, 0.68), (0.23, 0.68)),
        ((0.37, 0.68), (0.43, 0.68)),
        ((0.57, 0.68), (0.63, 0.80)),
        ((0.57, 0.68), (0.63, 0.52)),
        ((0.77, 0.82), (0.83, 0.72)),
        ((0.77, 0.52), (0.83, 0.62)),
        ((0.50, 0.55), (0.50, 0.32)),
        ((0.67, 0.32), (0.67, 0.42)),
    ]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#334E68"))
    ax.text(0.43, 0.96, "Risk-controlled cross-domain Task7 workflow", fontsize=14, weight="bold", ha="center", color="#102A43")
    ax.text(0.705, 0.66, "release if\nstable", fontsize=8.5, ha="center", color="#52606D")
    ax.text(0.705, 0.36, "otherwise\nreview", fontsize=8.5, ha="center", color="#52606D")
    fig.tight_layout()
    outputs = [FIG_DIR / "v172_fig1_pipeline.png", FIG_DIR / "v172_fig1_pipeline.pdf"]
    for path in outputs:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def draw_frontier(plt) -> list[Path]:
    pool = pd.read_csv(V162_POOL)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    families = [
        ("baseline_low_conf_dev_selected", "#7B8794", "Low-confidence review"),
        ("dev_stable_router_all_domains", "#2F80ED", "Dev-stable router"),
        ("shift_aware_image_directional_candidate", "#00A676", "Image-direction router"),
    ]
    for prefix, color, label in families:
        sub = pool.loc[pool["operating_point"].str.startswith(prefix)].sort_values("review_rate")
        if not sub.empty:
            ax.plot(sub["review_rate"] * 100, sub["all_bacc"] * 100, marker="o", color=color, linewidth=1.6, markersize=4, label=label)
    special = pool.loc[pool["operating_point"].isin(["locked_high_safety_two_signal_scorecard", "v161_safe_release_scorecard"])]
    for _, row in special.iterrows():
        color = "#C0392B" if row["operating_point"] == "v161_safe_release_scorecard" else "#F2994A"
        ax.scatter(row["review_rate"] * 100, row["all_bacc"] * 100, s=110, color=color, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(row["review_rate"] * 100 + 1.2, row["all_bacc"] * 100 - 0.6, short_point(row["operating_point"]), fontsize=8.5, color="#102A43")
    ax.set_xlabel("Review / reject rate (%)")
    ax.set_ylabel("All-domain balanced accuracy (%)")
    ax.set_title("Safety-efficiency operating frontier")
    ax.set_xlim(-2, 85)
    ax.set_ylim(72, 101)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    outputs = [FIG_DIR / "v172_fig2_safety_efficiency_frontier.png", FIG_DIR / "v172_fig2_safety_efficiency_frontier.pdf"]
    for path in outputs:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def draw_ablation(plt) -> list[Path]:
    ab = pd.read_csv(V164_ABLATION)
    order = [
        "old_only_selected",
        "third_only_selected",
        "leave_domain_intersection",
        "pooled_old_third_zero_error_v161",
        "balanced_zero_error",
    ]
    labels = ["Old-only", "Third-only", "Intersection", "Old+third\nzero-error", "Balanced\nzero-error"]
    sub = ab.set_index("strategy").loc[order].reset_index()
    colors = ["#D64545" if x > 0 else "#2F80ED" for x in sub["released_error_n"]]
    fig, ax1 = plt.subplots(figsize=(7.4, 5.2))
    ax2 = ax1.twinx()
    x = range(len(sub))
    ax1.bar(x, sub["review_rate"] * 100, color="#BCCCDC", width=0.62, label="Review/reject rate")
    ax2.plot(x, sub["released_error_n"], color="#D64545", marker="o", linewidth=2, label="Released errors")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Review / reject rate (%)")
    ax2.set_ylabel("Released errors")
    ax1.set_title("Safe-release strategy ablation")
    ax1.set_ylim(30, 65)
    ax2.set_ylim(-0.2, max(6, int(sub["released_error_n"].max()) + 1))
    for i, row in sub.iterrows():
        ax1.text(i, row["review_rate"] * 100 + 0.8, f"{row['review_rate'] * 100:.1f}%", ha="center", fontsize=8)
        ax2.text(i, row["released_error_n"] + 0.25, str(int(row["released_error_n"])), ha="center", fontsize=8, color=colors[i])
    ax1.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    outputs = [FIG_DIR / "v172_fig3_safe_release_ablation.png", FIG_DIR / "v172_fig3_safe_release_ablation.pdf"]
    for path in outputs:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = export_tables()
    plt, FancyArrowPatch, FancyBboxPatch, import_error = try_import_matplotlib()
    figure_paths: list[Path] = []
    matplotlib_available = plt is not None
    if matplotlib_available:
        figure_paths.extend(draw_pipeline(plt, FancyArrowPatch, FancyBboxPatch))
        figure_paths.extend(draw_frontier(plt))
        figure_paths.extend(draw_ablation(plt))
    index_rows = [
        {
            "figure": "Figure 1",
            "file_stem": "v172_fig1_pipeline",
            "purpose": "Workflow schematic showing classifier, risk scorecard, safe-release, review fallback, and candidate severe-shift correction branch.",
            "status": "generated" if matplotlib_available else "skipped_no_matplotlib",
        },
        {
            "figure": "Figure 2",
            "file_stem": "v172_fig2_safety_efficiency_frontier",
            "purpose": "Review-rate versus all-domain BAcc frontier comparing confidence baseline, direction routers, high-safety scorecard, and safe-release.",
            "status": "generated" if matplotlib_available else "skipped_no_matplotlib",
        },
        {
            "figure": "Figure 3",
            "file_stem": "v172_fig3_safe_release_ablation",
            "purpose": "Safe-release ablation showing why single-domain old-only release is unsafe and multi-domain constraints are needed.",
            "status": "generated" if matplotlib_available else "skipped_no_matplotlib",
        },
    ]
    pd.DataFrame(index_rows).to_csv(OUT_DIR / "v172_figure_index.csv", index=False, encoding="utf-8-sig")
    md = [
        "# v172 Publication Figures Pack",
        "",
        f"Matplotlib available: {matplotlib_available}.",
        "",
        "## Figures",
    ]
    for row in index_rows:
        md.append(f"- {row['figure']}: {row['file_stem']} ({row['status']}). {row['purpose']}")
    md += [
        "",
        "## Source Tables",
    ]
    for name, path in tables.items():
        md.append(f"- {name}: {path.name}")
    if import_error is not None:
        md += ["", f"Matplotlib import error: `{import_error!r}`"]
    (OUT_DIR / "v172_publication_figures_pack.md").write_text("\n".join(md), encoding="utf-8")
    report = {
        "matplotlib_available": matplotlib_available,
        "figure_count": int(len(figure_paths)),
        "table_count": int(len(tables)),
        "figure_files": [str(p) for p in figure_paths],
        "import_error": repr(import_error) if import_error else None,
    }
    (OUT_DIR / "v172_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v172] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
