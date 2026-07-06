from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "grosspath_rc_v82_unlabeled_adaptive_workflow_20260527" / "v82_fixed_and_adaptive_summary.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v83_workflow_decision_utility_20260527"
FIG_DIR = OUT_DIR / "figures"


FN_WEIGHTS = [2, 5, 10, 20, 40]
FP_WEIGHTS = [1, 2, 5]
CONTROL_COSTS = [0.00, 0.02, 0.05, 0.10, 0.20]
POLICY_ORDER = [
    "v50_main",
    "v75_quality_lowconf",
    "v79_light_lowrisk_guard",
    "v79_strict_lowrisk_guard",
    "adaptive_v75_on_shift",
    "adaptive_light_on_shift",
    "adaptive_strict_on_shift",
]
POLICY_LABELS = {
    "v50_main": "Fixed v50",
    "v75_quality_lowconf": "Fixed v75",
    "v79_light_lowrisk_guard": "Fixed v79 light",
    "v79_strict_lowrisk_guard": "Fixed v79 strict",
    "adaptive_v75_on_shift": "Adaptive v75",
    "adaptive_light_on_shift": "Adaptive v79 light",
    "adaptive_strict_on_shift": "Adaptive v79 strict",
}


def utility(row: pd.Series, fn_w: float, fp_w: float, control_c: float) -> float:
    n = float(row["n"])
    cost = fn_w * float(row["fn"]) + fp_w * float(row["fp"]) + control_c * float(row["control_n"])
    return -cost / n


def run_grid(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["scope"].eq("all_domains") & summary["policy"].isin(POLICY_ORDER)].copy()
    rows = []
    for fn_w in FN_WEIGHTS:
        for fp_w in FP_WEIGHTS:
            for control_c in CONTROL_COSTS:
                tmp = base.copy()
                tmp["fn_weight"] = fn_w
                tmp["fp_weight"] = fp_w
                tmp["control_cost"] = control_c
                tmp["utility"] = tmp.apply(lambda r: utility(r, fn_w, fp_w, control_c), axis=1)
                baseline_u = float(tmp.loc[tmp["policy"].eq("v50_main"), "utility"].iloc[0])
                tmp["utility_gain_vs_v50"] = tmp["utility"] - baseline_u
                rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def scenario_recommendations(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fn_w, fp_w, control_c), sub in grid.groupby(["fn_weight", "fp_weight", "control_cost"], sort=False):
        sub = sub.sort_values(["utility", "balanced_accuracy"], ascending=[False, False])
        best = sub.iloc[0]
        rows.append(
            {
                "fn_weight": fn_w,
                "fp_weight": fp_w,
                "control_cost": control_c,
                "best_policy": best["policy"],
                "best_policy_label": POLICY_LABELS.get(best["policy"], best["policy"]),
                "best_utility": best["utility"],
                "utility_gain_vs_v50": best["utility_gain_vs_v50"],
                "best_control_rate": best["control_rate"],
                "best_bacc": best["balanced_accuracy"],
                "best_fn": best["fn"],
                "best_fp": best["fp"],
            }
        )
    return pd.DataFrame(rows)


def compact_scenarios(grid: pd.DataFrame) -> pd.DataFrame:
    named = [
        ("balanced_low_review_cost", 5, 1, 0.02),
        ("balanced_medium_review_cost", 5, 1, 0.10),
        ("high_sensitivity_low_review_cost", 20, 1, 0.02),
        ("high_sensitivity_medium_review_cost", 20, 1, 0.10),
        ("avoid_overcall_low_review_cost", 5, 5, 0.02),
        ("avoid_overcall_medium_review_cost", 5, 5, 0.10),
        ("review_expensive", 10, 1, 0.20),
    ]
    rows = []
    for name, fn_w, fp_w, control_c in named:
        sub = grid.loc[grid["fn_weight"].eq(fn_w) & grid["fp_weight"].eq(fp_w) & grid["control_cost"].eq(control_c)].copy()
        sub = sub.sort_values("utility", ascending=False)
        for rank, (_, row) in enumerate(sub.head(4).iterrows(), start=1):
            rows.append(
                {
                    "scenario": name,
                    "rank": rank,
                    "policy": row["policy"],
                    "policy_label": POLICY_LABELS.get(row["policy"], row["policy"]),
                    "utility": row["utility"],
                    "gain_vs_v50": row["utility_gain_vs_v50"],
                    "control_rate": row["control_rate"],
                    "bacc": row["balanced_accuracy"],
                    "fn": row["fn"],
                    "fp": row["fp"],
                }
            )
    return pd.DataFrame(rows)


def make_plot(reco: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sub = reco.loc[reco["fp_weight"].eq(1)].copy()
    for control_c in [0.02, 0.10, 0.20]:
        pivot = sub.loc[sub["control_cost"].eq(control_c)].pivot(index="fn_weight", columns="fp_weight", values="best_policy_label")
        # Keep a text table figure; policy regions are easier to read with labels here.
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        ax.axis("off")
        cell_text = [[fn, sub.loc[(sub["control_cost"].eq(control_c)) & (sub["fn_weight"].eq(fn)) & (sub["fp_weight"].eq(1)), "best_policy_label"].iloc[0]] for fn in FN_WEIGHTS]
        table = ax.table(cellText=cell_text, colLabels=["FN cost", f"Best policy (FP cost=1, review cost={control_c})"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax.set_title(f"Decision utility winner: review cost={control_c}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"v83_utility_winners_review_cost_{str(control_c).replace('.', 'p')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    grid = run_grid(summary)
    reco = scenario_recommendations(grid)
    compact = compact_scenarios(grid)
    grid.to_csv(OUT_DIR / "v83_utility_grid_all_policies.csv", index=False, encoding="utf-8-sig")
    reco.to_csv(OUT_DIR / "v83_utility_best_policy_by_cost_scenario.csv", index=False, encoding="utf-8-sig")
    compact.to_csv(OUT_DIR / "v83_utility_named_scenario_rankings.csv", index=False, encoding="utf-8-sig")
    make_plot(reco)

    print("Named scenario rankings:")
    print(compact.to_string(index=False))
    print("\nBest policy counts across full utility grid:")
    print(reco["best_policy_label"].value_counts().to_string())
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
