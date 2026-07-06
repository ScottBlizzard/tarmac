from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v61_pareto_utility_analysis_20260527"
FIG_DIR = OUT_DIR / "figures"

V51 = ROOT / "outputs" / "grosspath_rc_v51_workflow_validation_20260527"
V52 = ROOT / "outputs" / "grosspath_rc_v52_quality_retake_overlay_20260527"
V54 = ROOT / "outputs" / "grosspath_rc_v54_constrained_policy_search_20260527"
V58 = ROOT / "outputs" / "grosspath_rc_v58_fp_buffer_after_v54_20260527"
V59 = ROOT / "outputs" / "grosspath_rc_v59_lowrisk_boundary_specialist_20260527"
V55 = ROOT / "outputs" / "grosspath_rc_v55_final_tier_table_20260527"


FINAL_POLICY_MAP = {
    "P2_pure_auto": "P2 纯自动",
    "v37_balanced_dev97": "v37 均衡档",
    "v54_low_control_highsens": "v54 省复核高敏感",
    "v59_specialist_logistic_addon04": "v59 低危边界折中",
    "v50_sens98_spec90": "v50 主推高安全",
    "v52_quality_score_le82_overlay": "v52 质量≤82",
    "v52_quality_score_le88_overlay": "v52 质量≤88",
}


def configure_matplotlib_font() -> None:
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def metric_row(
    source: str,
    policy_id: str,
    family: str,
    review_rate: float,
    accuracy: float,
    bacc: float,
    sensitivity: float,
    specificity: float,
    fn: int,
    fp: int,
    selection_status: str,
) -> dict[str, object]:
    return {
        "source": source,
        "policy_id": policy_id,
        "display_name": FINAL_POLICY_MAP.get(policy_id, policy_id),
        "family": family,
        "review_or_control_rate": float(review_rate),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bacc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fn": int(fn),
        "fp": int(fp),
        "selection_status": selection_status,
        "is_final_tier": int(policy_id in FINAL_POLICY_MAP),
    }


def load_candidate_universe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    v51 = pd.read_csv(V51 / "v51_tiered_workflow_summary.csv")
    for _, r in v51.loc[v51["split"].eq("external")].iterrows():
        if r["policy"] in ["P2_pure_auto", "v37_balanced_dev97", "v48_direction_dev97", "v48_fn_high_safety", "v50_sens98_spec90"]:
            rows.append(
                metric_row(
                    "v51_fixed_dev_selected",
                    str(r["policy"]),
                    str(r["type"]),
                    r["review_rate"],
                    r["accuracy"],
                    r["balanced_accuracy"],
                    r["sensitivity"],
                    r["specificity"],
                    r["fn"],
                    r["fp"],
                    "development_selected",
                )
            )

    v54 = pd.read_csv(V54 / "v54_policy_grid_metrics.csv")
    for _, r in v54.iterrows():
        rows.append(
            metric_row(
                "v54_candidate_grid",
                str(r["policy"]),
                str(r["family"]),
                r["external_review_rate"],
                r["external_accuracy"],
                r["external_balanced_accuracy"],
                r["external_sensitivity"],
                r["external_specificity"],
                r["external_fn"],
                r["external_fp"],
                "external_descriptive_candidate",
            )
        )

    # Add explicit named v54 key policy id.
    v54_key = pd.read_csv(V54 / "v54_key_policy_bootstrap_ci.csv")
    for _, r in v54_key.iterrows():
        rows.append(
            metric_row(
                "v54_key_policy",
                str(r["policy_name"]),
                "development_selected_key",
                r["review_rate"],
                r["accuracy"],
                r["balanced_accuracy"],
                r["sensitivity"],
                r["specificity"],
                r["fn"],
                r["fp"],
                "development_selected",
            )
        )

    v58_path = V58 / "v58_fp_buffer_grid.csv"
    if v58_path.exists():
        v58 = pd.read_csv(v58_path)
        for _, r in v58.iterrows():
            rows.append(
                metric_row(
                    "v58_fp_buffer_grid",
                    f"v58::{r['addon_name']}::{float(r['addon_rate']):.3f}",
                    "fp_buffer_candidate",
                    r["external_review_rate"],
                    r["external_accuracy"],
                    r["external_balanced_accuracy"],
                    r["external_sensitivity"],
                    r["external_specificity"],
                    r["external_fn"],
                    r["external_fp"],
                    "external_descriptive_candidate",
                )
            )

    v59 = pd.read_csv(V59 / "v59_specialist_fp_buffer_grid.csv")
    for _, r in v59.iterrows():
        rows.append(
            metric_row(
                "v59_specialist_grid",
                f"v59::{r['model_name']}::{float(r['addon_rate']):.3f}",
                "lowrisk_specialist_candidate",
                r["external_review_rate"],
                r["external_accuracy"],
                r["external_balanced_accuracy"],
                r["external_sensitivity"],
                r["external_specificity"],
                r["external_fn"],
                r["external_fp"],
                "external_descriptive_candidate",
            )
        )
    v59_ci = pd.read_csv(V59 / "v59_selected_policy_bootstrap_ci.csv")
    for _, r in v59_ci.iterrows():
        rows.append(
            metric_row(
                "v59_selected_policy",
                str(r["policy_name"]),
                "lowrisk_specialist_development_selected",
                r["review_rate"],
                r["accuracy"],
                r["balanced_accuracy"],
                r["sensitivity"],
                r["specificity"],
                r["fn"],
                r["fp"],
                "development_selected",
            )
        )

    v52 = pd.read_csv(V52 / "v52_quality_retake_overlay_summary.csv")
    quality_name = {
        "v50_plus_quality_score_le82": "v52_quality_score_le82_overlay",
        "v50_plus_quality_score_le88": "v52_quality_score_le88_overlay",
    }
    for _, r in v52.iterrows():
        pid = quality_name.get(str(r["policy"]), str(r["policy"]))
        rows.append(
            metric_row(
                "v52_quality_overlay",
                pid,
                "quality_retake_overlay",
                r["total_control_rate"],
                r["accuracy"],
                r["balanced_accuracy"],
                r["sensitivity"],
                r["specificity"],
                r["fn"],
                r["fp"],
                "quality_overlay_exploratory",
            )
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(["policy_id", "review_or_control_rate", "balanced_accuracy", "fn", "fp"]).reset_index(drop=True)
    return out


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    pts = df.copy().sort_values(["review_or_control_rate", "balanced_accuracy"], ascending=[True, False])
    keep = []
    best_bacc = -np.inf
    best_sens = -np.inf
    for idx, r in pts.iterrows():
        bacc = float(r["balanced_accuracy"])
        sens = float(r["sensitivity"])
        if bacc > best_bacc + 1e-12 or (abs(bacc - best_bacc) <= 1e-12 and sens > best_sens + 1e-12):
            keep.append(idx)
            best_bacc = max(best_bacc, bacc)
            best_sens = max(best_sens, sens)
    return pts.loc[keep].reset_index(drop=True)


def compute_utility_table(final: pd.DataFrame, n_cases: int = 108) -> pd.DataFrame:
    rows = []
    fn_weights = [2, 5, 10, 20, 50]
    fp_weights = [1, 2]
    review_weights = [0.02, 0.05, 0.10, 0.20, 0.50]
    for fp_w in fp_weights:
        for fn_w in fn_weights:
            for review_w in review_weights:
                tmp = final.copy()
                tmp["expected_cost"] = (
                    fn_w * tmp["fn"].astype(float)
                    + fp_w * tmp["fp"].astype(float)
                    + review_w * tmp["review_or_control_rate"].astype(float) * n_cases
                )
                best = tmp.sort_values(["expected_cost", "review_or_control_rate", "fp"]).iloc[0]
                rows.append(
                    {
                        "fn_weight": fn_w,
                        "fp_weight": fp_w,
                        "review_weight": review_w,
                        "best_policy_id": best["policy_id"],
                        "best_policy_name": best["display_name"],
                        "expected_cost": float(best["expected_cost"]),
                        "review_or_control_rate": float(best["review_or_control_rate"]),
                        "bacc": float(best["balanced_accuracy"]),
                        "fn": int(best["fn"]),
                        "fp": int(best["fp"]),
                    }
                )
    return pd.DataFrame(rows)


def make_plots(candidates: pd.DataFrame, frontier: pd.DataFrame, final: pd.DataFrame, utility: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    cloud = candidates.loc[candidates["selection_status"].ne("development_selected")]
    ax.scatter(
        cloud["review_or_control_rate"] * 100,
        cloud["balanced_accuracy"] * 100,
        s=18,
        color="#b0b7c3",
        alpha=0.35,
        label="candidate policies (descriptive)",
    )
    ax.plot(
        frontier["review_or_control_rate"] * 100,
        frontier["balanced_accuracy"] * 100,
        color="#1f618d",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="external descriptive Pareto frontier",
    )
    final_plot = final.sort_values("review_or_control_rate")
    ax.scatter(
        final_plot["review_or_control_rate"] * 100,
        final_plot["balanced_accuracy"] * 100,
        s=78,
        color="#c0392b",
        edgecolor="white",
        linewidth=0.7,
        label="predefined final tiers",
    )
    for _, r in final_plot.iterrows():
        ax.text(r["review_or_control_rate"] * 100 + 0.5, r["balanced_accuracy"] * 100 + 0.15, r["display_name"], fontsize=7)
    ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review / control rate (%)")
    ax.set_ylabel("External balanced accuracy (%)")
    ax.set_title("Safety-resource Pareto analysis")
    ax.set_xlim(-2, 94)
    ax.set_ylim(68, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v61_pareto_frontier.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v61_pareto_frontier.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    heat = utility.loc[utility["fp_weight"].eq(1)].copy()
    pivot = heat.pivot_table(index="fn_weight", columns="review_weight", values="best_policy_name", aggfunc="first")
    policies = list(pd.unique(heat["best_policy_name"]))
    colors = plt.cm.Set2(np.linspace(0, 1, len(policies)))
    color_map = {p: colors[i] for i, p in enumerate(policies)}
    mat = np.zeros((len(pivot.index), len(pivot.columns), 4))
    for i, idx in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            mat[i, j] = color_map[pivot.loc[idx, col]]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    for i, idx in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            ax.text(j, i, pivot.loc[idx, col].replace(" ", "\n"), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Review/control cost per case")
    ax.set_ylabel("FN cost weight (FP weight = 1)")
    ax.set_title("Utility-optimal final tier under clinical cost assumptions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v61_utility_policy_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v61_utility_policy_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_universe()
    frontier = pareto_frontier(candidates)
    # Use final tiers from v55 so the output mirrors the doctor-facing strategy table.
    v55 = pd.read_csv(V55 / "v55_final_tier_strategy_table.csv")
    final_ids = {
        "P2_pure_auto",
        "v37_balanced_dev97",
        "v54_low_control_highsens",
        "v59_specialist_logistic_addon04",
        "v50_sens98_spec90",
        "v52_quality_score_le82_overlay",
        "v52_quality_score_le88_overlay",
    }
    final = candidates.loc[candidates["policy_id"].isin(final_ids)].copy()
    # De-duplicate final tiers by keeping the most explicit selected source.
    priority = {
        "development_selected": 0,
        "quality_overlay_exploratory": 1,
        "external_descriptive_candidate": 2,
    }
    final["_priority"] = final["selection_status"].map(priority).fillna(9)
    final = final.sort_values(["policy_id", "_priority"]).drop_duplicates("policy_id").drop(columns=["_priority"])
    final["tier"] = final["policy_id"].map(v55.set_index("source")["tier"]).fillna("")
    utility = compute_utility_table(final)

    candidates.to_csv(OUT_DIR / "v61_candidate_policy_universe.csv", index=False, encoding="utf-8-sig")
    frontier.to_csv(OUT_DIR / "v61_external_descriptive_pareto_frontier.csv", index=False, encoding="utf-8-sig")
    final.sort_values("review_or_control_rate").to_csv(OUT_DIR / "v61_final_tiers_for_utility.csv", index=False, encoding="utf-8-sig")
    utility.to_csv(OUT_DIR / "v61_final_tier_utility_grid.csv", index=False, encoding="utf-8-sig")
    make_plots(candidates, frontier, final, utility)

    print("Final tiers:")
    print(
        final.sort_values("review_or_control_rate")[
            [
                "tier",
                "display_name",
                "review_or_control_rate",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "selection_status",
            ]
        ].to_string(index=False)
    )
    print("\nUtility winners:")
    print(
        utility.groupby(["fp_weight", "best_policy_name"]).size().reset_index(name="n_cost_settings").sort_values(["fp_weight", "n_cost_settings"], ascending=[True, False]).to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
