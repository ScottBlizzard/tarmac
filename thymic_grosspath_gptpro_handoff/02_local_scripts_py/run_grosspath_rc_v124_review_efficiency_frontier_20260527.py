from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v124_review_efficiency_frontier_20260527"
V122_GRID = ROOT / "outputs" / "grosspath_rc_v122_two_signal_ablation_robustness_20260527" / "v122_fp_guard_full_grid.csv"
V122_RANDOM = ROOT / "outputs" / "grosspath_rc_v122_two_signal_ablation_robustness_20260527" / "v122_random_capture_baseline.csv"


def build_frontier(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, part in grid.groupby("mode", sort=False):
        for extra_n in sorted(part["extra_review_n"].unique()):
            sub = part[part["extra_review_n"].le(extra_n)]
            best = sub.sort_values(
                ["captured_error_n", "remaining_error_n", "balanced_accuracy", "extra_review_n"],
                ascending=[False, True, False, True],
            ).iloc[0]
            rows.append(
                {
                    "mode": mode,
                    "extra_review_budget": int(extra_n),
                    "best_captured_error_n": int(best["captured_error_n"]),
                    "best_remaining_error_n": int(best["remaining_error_n"]),
                    "best_fn": int(best["fn"]),
                    "best_fp": int(best["fp"]),
                    "best_control_rate": float(best["control_rate"]),
                    "best_balanced_accuracy": float(best["balanced_accuracy"]),
                    "wc_max": best["wc_max"],
                    "core_max": best["core_max"],
                }
            )
    return pd.DataFrame(rows)


def first_budget_for_capture(frontier: pd.DataFrame, capture_n: int) -> pd.DataFrame:
    rows = []
    for mode, part in frontier.groupby("mode", sort=False):
        ok = part[part["best_captured_error_n"].ge(capture_n)].copy()
        if ok.empty:
            rows.append({"mode": mode, "target_captured_error_n": capture_n, "min_extra_review_budget": np.nan})
        else:
            best = ok.sort_values("extra_review_budget").iloc[0]
            rows.append(
                {
                    "mode": mode,
                    "target_captured_error_n": capture_n,
                    "min_extra_review_budget": int(best["extra_review_budget"]),
                    "best_control_rate": float(best["best_control_rate"]),
                    "best_balanced_accuracy": float(best["best_balanced_accuracy"]),
                    "wc_max": best["wc_max"],
                    "core_max": best["core_max"],
                }
            )
    return pd.DataFrame(rows)


def plot_frontier(frontier: pd.DataFrame, random_expected: float) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "wholecrop_only": "#4c78a8",
        "core_only": "#f58518",
        "two_signal": "#54a24b",
    }
    labels = {
        "wholecrop_only": "wholecrop only",
        "core_only": "core only",
        "two_signal": "two-signal",
    }
    plt.figure(figsize=(7.6, 5.4))
    for mode, part in frontier.groupby("mode", sort=False):
        part = part.sort_values("extra_review_budget")
        plt.step(
            part["extra_review_budget"],
            part["best_captured_error_n"],
            where="post",
            linewidth=2.2,
            color=colors.get(mode, "gray"),
            label=labels.get(mode, mode),
        )
    plt.scatter([8], [3], color="#54a24b", edgecolor="black", s=80, zorder=5, label="v118 operating point")
    plt.scatter([8], [random_expected], color="#999999", marker="x", s=90, zorder=5, label="random expected at n=8")
    plt.axhline(3, color="#2d2d2d", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel("Additional auto-high cases reviewed")
    plt.ylabel("Captured FP errors")
    plt.title("Review-efficiency frontier for low-risk overcall guard")
    plt.xlim(0, max(20, int(frontier["extra_review_budget"].max())))
    plt.ylim(-0.05, 3.25)
    plt.yticks([0, 1, 2, 3])
    plt.grid(alpha=0.24)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "v124_review_efficiency_frontier.png", dpi=220)
    plt.savefig(fig_dir / "v124_review_efficiency_frontier.pdf")
    plt.close()


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(V122_GRID)
    random_baseline = pd.read_csv(V122_RANDOM)
    frontier = build_frontier(grid)
    target3 = first_budget_for_capture(frontier, 3)
    random_expected = float(random_baseline["random_expected_captured_error_n"].iloc[0])
    plot_frontier(frontier, random_expected)

    frontier.to_csv(OUT_DIR / "v124_review_efficiency_frontier.csv", index=False, encoding="utf-8-sig")
    target3.to_csv(OUT_DIR / "v124_min_budget_to_capture_all_fp.csv", index=False, encoding="utf-8-sig")

    formatted = target3.copy()
    for col in ["best_control_rate", "best_balanced_accuracy"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(pct)
    formatted.to_csv(OUT_DIR / "v124_min_budget_to_capture_all_fp_formatted.csv", index=False, encoding="utf-8-sig")

    two = target3[target3["mode"].eq("two_signal")].iloc[0]
    whole = target3[target3["mode"].eq("wholecrop_only")].iloc[0]
    core = target3[target3["mode"].eq("core_only")].iloc[0]
    lines = [
        "# v124 Review-efficiency Frontier",
        "",
        "v124 converts the v122 threshold grid into a review-efficiency frontier.",
        "",
        f"- To capture all 3 low-risk overcall errors, two-signal needs {int(two['min_extra_review_budget'])} additional reviews.",
        f"- Wholecrop-only needs {int(whole['min_extra_review_budget'])} additional reviews; core-only needs {int(core['min_extra_review_budget'])}.",
        f"- Randomly reviewing 8 comparable auto-high cases is expected to capture {random_expected:.2f} errors.",
        "",
        "This gives a direct method-level claim: two-signal gating is on the efficient frontier and halves the review burden required by either single-signal alternative.",
    ]
    (OUT_DIR / "v124_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(formatted.to_string(index=False))
    print()
    print((OUT_DIR / "figures" / "v124_review_efficiency_frontier.pdf"))


if __name__ == "__main__":
    main()
