from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v119_bidirectional_two_signal_map_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"
V118_SUMMARY = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_summary.csv"

FN_WC_MIN = 0.25
FN_CORE_MAX = 0.325
FP_WC_MAX = 0.625
FP_CORE_MAX = 0.775


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def load_cases() -> pd.DataFrame:
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["review_or_control", "v111_extra_review", "v111_review_or_control", "v118_extra_review", "v118_review_or_control"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    return df


def guard_contribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    guards = [
        (
            "FN guard",
            f"auto-low internal & wholecrop_prob>={FN_WC_MIN:.3f} & prob_mean_core<={FN_CORE_MAX:.3f}",
            df["v111_extra_review"],
        ),
        (
            "FP guard",
            f"auto-high internal & wholecrop_prob<={FP_WC_MAX:.3f} & prob_mean_core<={FP_CORE_MAX:.3f}",
            df["v118_extra_review"],
        ),
    ]
    for name, rule, flag in guards:
        rows.append(
            {
                "guard": name,
                "rule": rule,
                "extra_review_n": int(flag.sum()),
                "captured_error_n": int((flag & df["final_correct"].eq(0)).sum()),
                "clean_review_n": int((flag & df["final_correct"].eq(1)).sum()),
                "captured_cases": "; ".join(
                    df.loc[flag & df["final_correct"].eq(0), "original_case_id"].astype(str).tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_bidirectional_map(df: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    internal = df["domain"].isin(["old_data", "third_batch"])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True, sharey=True)

    base_auto_low = internal & (~df["review_or_control"]) & df["final_pred"].eq(0)
    low = df[base_auto_low].copy()
    low["group"] = "clean auto-low"
    low.loc[low["v111_extra_review"] & low["final_correct"].eq(0), "group"] = "rescued FN"
    for group, color, size in [("clean auto-low", "#91b7db", 26), ("rescued FN", "#c83737", 52)]:
        part = low[low["group"].eq(group)]
        axes[0].scatter(
            part["wholecrop_prob"],
            part["prob_mean_core"],
            s=size,
            c=color,
            label=f"{group} (n={len(part)})",
            edgecolors="black" if group == "rescued FN" else "none",
            linewidths=0.8,
            alpha=0.9,
        )
    axes[0].axvline(FN_WC_MIN, color="#2d2d2d", linestyle="--", linewidth=1.2)
    axes[0].axhline(FN_CORE_MAX, color="#2d2d2d", linestyle="--", linewidth=1.2)
    axes[0].fill_between([FN_WC_MIN, 1], [0, 0], [FN_CORE_MAX, FN_CORE_MAX], color="#c83737", alpha=0.08)
    axes[0].set_title("High-risk miss guard")
    axes[0].set_xlabel("Whole-crop high-risk probability")
    axes[0].set_ylabel("Core-model mean high-risk probability")
    axes[0].legend(frameon=False, loc="upper right")

    base_auto_high = internal & (~df["v111_review_or_control"]) & df["final_pred"].eq(1)
    high = df[base_auto_high].copy()
    high["group"] = "clean auto-high"
    high.loc[high["v118_extra_review"] & high["final_correct"].eq(0), "group"] = "rescued FP"
    for group, color, size in [("clean auto-high", "#91b7db", 26), ("rescued FP", "#c83737", 52)]:
        part = high[high["group"].eq(group)]
        axes[1].scatter(
            part["wholecrop_prob"],
            part["prob_mean_core"],
            s=size,
            c=color,
            label=f"{group} (n={len(part)})",
            edgecolors="black" if group == "rescued FP" else "none",
            linewidths=0.8,
            alpha=0.9,
        )
    axes[1].axvline(FP_WC_MAX, color="#2d2d2d", linestyle="--", linewidth=1.2)
    axes[1].axhline(FP_CORE_MAX, color="#2d2d2d", linestyle="--", linewidth=1.2)
    axes[1].fill_between([0, FP_WC_MAX], [0, 0], [FP_CORE_MAX, FP_CORE_MAX], color="#c83737", alpha=0.08)
    axes[1].set_title("Low-risk overcall guard")
    axes[1].set_xlabel("Whole-crop high-risk probability")
    axes[1].legend(frameon=False, loc="lower right")

    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.22)
    fig.suptitle("Bidirectional two-signal risk scorecard", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "v119_bidirectional_two_signal_map.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "v119_bidirectional_two_signal_map.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    summary = pd.read_csv(V118_SUMMARY)
    contrib = guard_contribution(df)
    plot_bidirectional_map(df)

    contrib.to_csv(OUT_DIR / "v119_guard_contribution.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v119_final_scorecard_summary.csv", index=False, encoding="utf-8-sig")
    formatted = summary.copy()
    for col in formatted.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            formatted[col] = formatted[col].map(pct)
    formatted.to_csv(OUT_DIR / "v119_final_scorecard_summary_formatted.csv", index=False, encoding="utf-8-sig")

    final_all = summary[
        (summary["workflow"].eq("global_two_signal_scorecard")) & (summary["scope"].eq("all_domains"))
    ].iloc[0]
    lines = [
        "# v119 Bidirectional Two-signal Map",
        "",
        "v119 packages v109 and v118 into one interpretable bidirectional scorecard.",
        "",
        f"- Final all-domain: BAcc {pct(final_all['balanced_accuracy'])}, control {pct(final_all['control_rate'])}, FN={int(final_all['fn'])}, FP={int(final_all['fp'])}.",
        f"- FN guard captured {int(contrib.loc[contrib['guard'].eq('FN guard'), 'captured_error_n'].iloc[0])} errors with {int(contrib.loc[contrib['guard'].eq('FN guard'), 'extra_review_n'].iloc[0])} extra reviews.",
        f"- FP guard captured {int(contrib.loc[contrib['guard'].eq('FP guard'), 'captured_error_n'].iloc[0])} errors with {int(contrib.loc[contrib['guard'].eq('FP guard'), 'extra_review_n'].iloc[0])} extra reviews.",
        "",
        "The figure provides a method-level view: automatic low-risk and automatic high-risk outputs are checked by opposite quadrants of the same two signal axes.",
    ]
    (OUT_DIR / "v119_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(contrib.to_string(index=False))
    print()
    print(formatted[formatted["scope"].isin(["all_domains", "third_batch", "strict_external"])].to_string(index=False))


if __name__ == "__main__":
    main()
