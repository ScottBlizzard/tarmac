from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v35_dev_targeted_review_budget_20260527"
FIG_DIR = OUT_DIR / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curve = pd.read_csv(OUT_DIR / "v35_review_budget_curve.csv")
    selected = pd.read_csv(OUT_DIR / "v35_dev_target_selected_budgets_external_eval.csv")

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(curve["dev_review_rate"] * 100, curve["dev_balanced_accuracy"] * 100, label="Development OOF workflow BAcc", color="#2471a3", linewidth=2.0)
    ax.plot(curve["external_review_rate"] * 100, curve["external_balanced_accuracy"] * 100, label="External workflow BAcc", color="#c0392b", linewidth=2.0)
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(93, color="#7d6608", linestyle=":", linewidth=1.0, alpha=0.7)
    for _, row in selected.iterrows():
        if row["target_dev_bacc"] in [0.90, 0.95, 0.97]:
            ax.scatter(row["external_review_rate"] * 100, row["external_balanced_accuracy"] * 100, color="#c0392b", s=52, zorder=3)
            ax.text(
                row["external_review_rate"] * 100 + 1.0,
                row["external_balanced_accuracy"] * 100 - 0.7,
                f"dev target {row['target_dev_bacc'] * 100:.0f}%",
                fontsize=8,
                color="#7b241c",
            )
    ax.set_xlabel("Review / risk-control rate (%)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Dev-targeted review budget transfers to external set")
    ax.set_xlim(-2, 82)
    ax.set_ylim(68, 99)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v35_dev_targeted_budget_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v35_dev_targeted_budget_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
