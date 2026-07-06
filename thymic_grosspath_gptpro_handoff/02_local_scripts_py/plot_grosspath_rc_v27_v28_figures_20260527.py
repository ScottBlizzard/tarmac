from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V27_DIR = ROOT / "outputs" / "grosspath_rc_v27_unified_workflow_20260527"
V28_DIR = ROOT / "outputs" / "grosspath_rc_v28_decision_utility_20260527"
FIG_DIR = V28_DIR / "figures"


POLICY_ORDER = [
    "P0_main_auto_all",
    "P1_robust_auto_all",
    "P2_safety_switch_auto",
    "P3_qscore92_review_else_safety",
    "P4_rf_quality_review_else_safety",
    "P5_extra_trees_quality_review_else_safety",
    "P6_manual_quality_or_safety_review_upper",
    "P7_logistic_recall90_quality_or_safety_review",
]

POLICY_LABEL = {
    "P0_main_auto_all": "P0 Main",
    "P1_robust_auto_all": "P1 Robust",
    "P2_safety_switch_auto": "P2 Safety switch",
    "P3_qscore92_review_else_safety": "P3 Q92 review",
    "P4_rf_quality_review_else_safety": "P4 RF quality",
    "P5_extra_trees_quality_review_else_safety": "P5 ET quality",
    "P6_manual_quality_or_safety_review_upper": "P6 Manual quality",
    "P7_logistic_recall90_quality_or_safety_review": "P7 High safety",
}

EVIDENCE_COLOR = {
    "strict_auto_baseline": "#566573",
    "strict_auto_rule": "#2e86c1",
    "automatic_quality_heuristic": "#117a65",
    "quality_cv_proof_of_concept": "#d68910",
    "manual_quality_upper_bound": "#884ea0",
    "quality_cv_high_safety": "#c0392b",
}

PLOT_OFFSET = {
    "P1_robust_auto_all": (-0.75, 0.18),
    "P2_safety_switch_auto": (0.75, -0.18),
}


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["policy"] = pd.Categorical(out["policy"], POLICY_ORDER, ordered=True)
    return out.sort_values("policy").reset_index(drop=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for _, row in metrics.iterrows():
        color = EVIDENCE_COLOR.get(row["evidence_level"], "#333333")
        dx, dy = PLOT_OFFSET.get(str(row["policy"]), (0.0, 0.0))
        display_x = row["risk_control_rate"] * 100 + dx
        display_y = row["workflow_balanced_accuracy"] * 100 + dy
        ax.scatter(
            display_x,
            display_y,
            s=80 + row["auto_output_rate"] * 180,
            color=color,
            edgecolor="white",
            linewidth=1.4,
            alpha=0.92,
        )
        policy = str(row["policy"])
        if policy == "P2_safety_switch_auto":
            continue
        label = "P1/P2 Robust/Safety" if policy == "P1_robust_auto_all" else POLICY_LABEL[policy]
        ax.text(display_x + 1.1, display_y + 0.25, label, fontsize=9, color="#1f2d3d")

    ax.axhline(90, color="#b03a2e", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(50, color="#7d6608", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Risk-control burden: review + retake (%)")
    ax.set_ylabel("Workflow balanced accuracy (%)")
    ax.set_title("External-set performance vs risk-control burden")
    ax.set_xlim(-3, 82)
    ax.set_ylim(60, 95.5)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    legend_handles = []
    for evidence, color in EVIDENCE_COLOR.items():
        if evidence in set(metrics["evidence_level"]):
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=evidence,
                )
            )
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        title="Evidence level",
        title_fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, "v27_policy_tradeoff_external")


def plot_composition(metrics: pd.DataFrame) -> None:
    x = np.arange(len(metrics))
    auto = metrics["auto_output_rate"].to_numpy() * 100
    review = metrics["review_rate"].to_numpy() * 100
    retake = metrics["retake_rate"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x, auto, label="Auto output", color="#2e86c1")
    ax.bar(x, review, bottom=auto, label="Review", color="#f5b041")
    ax.bar(x, retake, bottom=auto + review, label="Retake", color="#c0392b")

    for i, row in metrics.iterrows():
        ax.text(
            i,
            102.0,
            f"BAcc {row['workflow_balanced_accuracy'] * 100:.1f}\nFN {int(row['workflow_fn'])}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f2d3d",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABEL[str(p)] for p in metrics["policy"]], rotation=28, ha="right")
    ax.set_ylabel("Case proportion (%)")
    ax.set_ylim(0, 116)
    ax.set_title("Workflow composition and residual false negatives")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, "v27_workflow_composition_external")


def plot_errors(metrics: pd.DataFrame) -> None:
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - width / 2, metrics["workflow_fn"], width, label="False negative", color="#c0392b")
    ax.bar(x + width / 2, metrics["workflow_fp"], width, label="False positive", color="#2471a3")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABEL[str(p)] for p in metrics["policy"]], rotation=28, ha="right")
    ax.set_ylabel("Number of cases")
    ax.set_title("External-set error profile after each workflow policy")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, "v27_error_profile_external")


def plot_utility_heatmap(utility: pd.DataFrame) -> None:
    utility = utility.copy()
    utility["policy"] = pd.Categorical(utility["policy"], POLICY_ORDER, ordered=True)
    pivot = utility.pivot_table(
        index="scenario",
        columns="policy",
        values="normalized_cost",
        aggfunc="mean",
        observed=False,
    )
    pivot = pivot[[p for p in POLICY_ORDER if p in pivot.columns]]
    pivot = pivot.sort_index()

    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([POLICY_LABEL[str(p)] for p in pivot.columns], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(s).replace("_", " ") for s in pivot.index])
    ax.set_title("Decision utility by clinical scenario (lower is better)")

    for i in range(values.shape[0]):
        row_min = np.nanmin(values[i])
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > np.nanpercentile(values, 65) else "#1f2d3d"
            weight = "bold" if abs(val - row_min) < 1e-9 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized cost")
    fig.tight_layout()
    save_figure(fig, "v28_decision_utility_heatmap")


def write_summary(metrics: pd.DataFrame, utility: pd.DataFrame) -> None:
    cols = [
        "policy",
        "evidence_level",
        "auto_output_rate",
        "risk_control_rate",
        "workflow_accuracy",
        "workflow_balanced_accuracy",
        "workflow_sensitivity",
        "workflow_specificity",
        "workflow_fn",
        "workflow_fp",
    ]
    summary = metrics[cols].copy()
    for col in [
        "auto_output_rate",
        "risk_control_rate",
        "workflow_accuracy",
        "workflow_balanced_accuracy",
        "workflow_sensitivity",
        "workflow_specificity",
    ]:
        summary[col] = (summary[col] * 100).round(2)
    summary.to_csv(V28_DIR / "v27_v28_policy_summary_for_figures.csv", index=False, encoding="utf-8-sig")

    best = (
        utility.sort_values(["scenario", "normalized_cost"])
        .groupby("scenario", as_index=False)
        .first()[
            [
                "scenario",
                "policy",
                "normalized_cost",
                "workflow_balanced_accuracy",
                "workflow_fn",
                "workflow_fp",
                "risk_control_rate",
            ]
        ]
        .copy()
    )
    for col in ["workflow_balanced_accuracy", "risk_control_rate"]:
        best[col] = (best[col] * 100).round(2)
    best["normalized_cost"] = best["normalized_cost"].round(4)
    best.to_csv(V28_DIR / "v28_best_policy_by_scenario_for_figures.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    metrics = pd.read_csv(V27_DIR / "v27_unified_workflow_metrics.csv")
    metrics = ordered(metrics)
    utility = pd.read_csv(V28_DIR / "v28_decision_utility_by_policy.csv")

    plot_tradeoff(metrics)
    plot_composition(metrics)
    plot_errors(metrics)
    plot_utility_heatmap(utility)
    write_summary(metrics, utility)

    print(f"Saved figures to: {FIG_DIR}")
    print(f"Saved summaries to: {V28_DIR}")


if __name__ == "__main__":
    main()
