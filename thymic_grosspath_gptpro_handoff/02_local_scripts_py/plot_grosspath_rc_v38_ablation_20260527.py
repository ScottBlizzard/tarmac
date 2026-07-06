from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v38_risk_gate_feature_ablation_20260527"
FIG_DIR = OUT_DIR / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(OUT_DIR / "v38_feature_ablation_target_transfer_metrics.csv")
    focus = metrics.loc[metrics["target_dev_bacc"].isin([0.95, 0.97])].copy()
    focus["label"] = focus["feature_set"] + "\n" + focus["selection"].str.replace("_", " ")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, target in zip(axes, [0.95, 0.97]):
        sub = focus.loc[focus["target_dev_bacc"].eq(target)].copy()
        colors = sub["selection"].map({"rank_budget": "#117a65", "absolute_threshold": "#c0392b"}).fillna("#566573")
        ax.scatter(
            sub["external_review_rate"] * 100,
            sub["external_balanced_accuracy"] * 100,
            s=72,
            c=colors,
            edgecolor="white",
            linewidth=1.0,
        )
        for _, row in sub.iterrows():
            ax.text(
                row["external_review_rate"] * 100 + 1.0,
                row["external_balanced_accuracy"] * 100,
                row["feature_set"].replace("_", " "),
                fontsize=7,
                color="#1f2d3d",
            )
        ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.65)
        ax.axvline(60, color="#7d6608", linestyle=":", linewidth=1, alpha=0.65)
        ax.set_title(f"Dev target BAcc {int(target * 100)}%")
        ax.set_xlabel("External review rate (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(-3, 101)
        ax.set_ylim(68, 100)
    axes[0].set_ylabel("External balanced accuracy (%)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#117a65", markersize=8, label="Rank budget"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b", markersize=8, label="Absolute threshold"),
    ]
    axes[1].legend(handles=handles, loc="lower right")
    fig.suptitle("Risk-gate feature ablation: performance vs review burden", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v38_feature_ablation_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v38_feature_ablation_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
