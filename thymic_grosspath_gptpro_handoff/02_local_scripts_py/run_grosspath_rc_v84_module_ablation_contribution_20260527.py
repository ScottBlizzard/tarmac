from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V80 = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527" / "v80_tiered_lowrisk_guard_summary.csv"
V82 = ROOT / "outputs" / "grosspath_rc_v82_unlabeled_adaptive_workflow_20260527" / "v82_fixed_and_adaptive_summary.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v84_module_ablation_contribution_20260527"
FIG_DIR = OUT_DIR / "figures"


PAIRS = [
    ("highrisk_safety_v75_minus_v50", "v50_main", "v75_quality_lowconf", "High-risk miss protection"),
    ("light_lowrisk_guard_minus_v75", "v75_quality_lowconf", "v79_light_lowrisk_guard", "Light low-risk overcall guard"),
    ("strict_lowrisk_guard_minus_light", "v79_light_lowrisk_guard", "v79_strict_lowrisk_guard", "Strict low-risk overcall guard"),
    ("strict_full_minus_v50", "v50_main", "v79_strict_lowrisk_guard", "Full fixed safety workflow"),
]


def delta_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in ["old_data", "third_batch", "strict_external"]:
        sub = summary.loc[summary["split"].eq(domain)].set_index("policy")
        for module, before, after, label in PAIRS:
            b = sub.loc[before]
            a = sub.loc[after]
            rows.append(
                {
                    "domain": domain,
                    "module": module,
                    "module_label": label,
                    "before_policy": before,
                    "after_policy": after,
                    "delta_control_rate": float(a["control_rate"] - b["control_rate"]),
                    "delta_bacc": float(a["balanced_accuracy"] - b["balanced_accuracy"]),
                    "delta_sensitivity": float(a["sensitivity"] - b["sensitivity"]),
                    "delta_specificity": float(a["specificity"] - b["specificity"]),
                    "delta_fn": int(a["fn"] - b["fn"]),
                    "delta_fp": int(a["fp"] - b["fp"]),
                    "delta_remaining_error_n": int(a["remaining_error_n"] - b["remaining_error_n"]),
                }
            )
    return pd.DataFrame(rows)


def adaptive_delta(v82: pd.DataFrame) -> pd.DataFrame:
    all_scope = v82.loc[v82["scope"].eq("all_domains")].set_index("policy")
    pairs = [
        ("adaptive_light_minus_fixed_v50", "v50_main", "adaptive_light_on_shift", "Adaptive light vs fixed v50"),
        ("adaptive_strict_minus_fixed_v50", "v50_main", "adaptive_strict_on_shift", "Adaptive strict vs fixed v50"),
        ("fixed_light_minus_adaptive_light", "adaptive_light_on_shift", "v79_light_lowrisk_guard", "Fixed light vs adaptive light"),
        ("fixed_strict_minus_adaptive_strict", "adaptive_strict_on_shift", "v79_strict_lowrisk_guard", "Fixed strict vs adaptive strict"),
    ]
    rows = []
    for module, before, after, label in pairs:
        b = all_scope.loc[before]
        a = all_scope.loc[after]
        rows.append(
            {
                "scope": "all_domains",
                "comparison": module,
                "comparison_label": label,
                "before_policy": before,
                "after_policy": after,
                "delta_control_rate": float(a["control_rate"] - b["control_rate"]),
                "delta_bacc": float(a["balanced_accuracy"] - b["balanced_accuracy"]),
                "delta_sensitivity": float(a["sensitivity"] - b["sensitivity"]),
                "delta_specificity": float(a["specificity"] - b["specificity"]),
                "delta_fn": int(a["fn"] - b["fn"]),
                "delta_fp": int(a["fp"] - b["fp"]),
                "delta_remaining_error_n": int(a["remaining_error_n"] - b["remaining_error_n"]),
            }
        )
    return pd.DataFrame(rows)


def make_plot(delta: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    domains = ["old_data", "third_batch", "strict_external"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    colors = {
        "highrisk_safety_v75_minus_v50": "#117a65",
        "light_lowrisk_guard_minus_v75": "#d68910",
        "strict_lowrisk_guard_minus_light": "#c0392b",
        "strict_full_minus_v50": "#884ea0",
    }
    for ax, domain in zip(axes, domains):
        sub = delta.loc[delta["domain"].eq(domain)]
        for _, row in sub.iterrows():
            ax.scatter(row["delta_control_rate"] * 100, row["delta_bacc"] * 100, s=90, color=colors[row["module"]], edgecolor="white")
            ax.text(row["delta_control_rate"] * 100 + 0.1, row["delta_bacc"] * 100, row["module_label"], fontsize=7)
        ax.axhline(0, color="#566573", linewidth=1)
        ax.axvline(0, color="#566573", linewidth=1)
        ax.set_title(domain)
        ax.set_xlabel("Delta control rate (pp)")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Delta balanced accuracy (pp)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v84_module_delta_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v84_module_delta_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v80 = pd.read_csv(V80)
    v82 = pd.read_csv(V82)
    delta = delta_rows(v80)
    adelta = adaptive_delta(v82)
    delta.to_csv(OUT_DIR / "v84_fixed_workflow_module_delta_by_domain.csv", index=False, encoding="utf-8-sig")
    adelta.to_csv(OUT_DIR / "v84_adaptive_vs_fixed_delta.csv", index=False, encoding="utf-8-sig")
    make_plot(delta)

    print("Fixed workflow module deltas:")
    print(
        delta[
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
        ].to_string(index=False)
    )
    print("\nAdaptive/fixed deltas:")
    print(adelta.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
