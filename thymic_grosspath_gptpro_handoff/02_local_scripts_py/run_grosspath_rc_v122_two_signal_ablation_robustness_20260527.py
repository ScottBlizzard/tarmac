from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v122_two_signal_ablation_robustness_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"

V118_WC_MAX = 0.625
V118_CORE_MAX = 0.775


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def metrics(df: pd.DataFrame, review: pd.Series | np.ndarray) -> dict[str, float | int]:
    review = pd.Series(review, index=df.index).astype(bool)
    wrong = df["final_correct"].eq(0)
    rem = (~review) & wrong
    fn = int((rem & df["label_idx"].eq(1) & df["final_pred"].eq(0)).sum())
    fp = int((rem & df["label_idx"].eq(0) & df["final_pred"].eq(1)).sum())
    pos = int(df["label_idx"].eq(1).sum())
    neg = int(df["label_idx"].eq(0).sum())
    sens = (pos - fn) / pos if pos else np.nan
    spec = (neg - fp) / neg if neg else np.nan
    return {
        "n": int(len(df)),
        "control_rate": float(review.mean()),
        "remaining_error_n": int(rem.sum()),
        "fn": fn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    return out


def load_cases() -> pd.DataFrame:
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["review_or_control", "v111_review_or_control", "v118_review_or_control", "v118_extra_review"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    return df


def fp_guard(df: pd.DataFrame, base_review: pd.Series, mode: str, wc_max: float | None = None, core_max: float | None = None) -> pd.Series:
    auto_high = df["domain"].isin(["old_data", "third_batch"]) & (~base_review.astype(bool)) & df["final_pred"].eq(1)
    flag = auto_high.copy()
    if mode in ["wholecrop_only", "two_signal"]:
        flag &= pd.to_numeric(df["wholecrop_prob"], errors="coerce").le(float(wc_max))
    if mode in ["core_only", "two_signal"]:
        flag &= pd.to_numeric(df["prob_mean_core"], errors="coerce").le(float(core_max))
    return flag


def evaluate_workflow(df: pd.DataFrame, workflow: str, review: pd.Series, extra: pd.Series) -> dict[str, float | int | str]:
    m = metrics(df, review)
    return {
        "workflow": workflow,
        "extra_review_n": int(extra.sum()),
        "captured_error_n": int((extra & df["final_correct"].eq(0)).sum()),
        "clean_review_n": int((extra & df["final_correct"].eq(1)).sum()),
        **m,
    }


def scan_single_and_two_signal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df["v111_review_or_control"].astype(bool)
    rows = []
    for mode in ["wholecrop_only", "core_only", "two_signal"]:
        wc_values = [np.nan] if mode == "core_only" else np.round(np.arange(0.400, 0.901, 0.025), 3)
        core_values = [np.nan] if mode == "wholecrop_only" else np.round(np.arange(0.650, 0.901, 0.025), 3)
        for wc_max in wc_values:
            for core_max in core_values:
                extra = fp_guard(df, base, mode, None if pd.isna(wc_max) else float(wc_max), None if pd.isna(core_max) else float(core_max))
                review = base | extra
                rows.append(
                    {
                        "mode": mode,
                        "wc_max": "" if pd.isna(wc_max) else float(wc_max),
                        "core_max": "" if pd.isna(core_max) else float(core_max),
                        "extra_review_n": int(extra.sum()),
                        "captured_error_n": int((extra & df["final_correct"].eq(0)).sum()),
                        "clean_review_n": int((extra & df["final_correct"].eq(1)).sum()),
                        **metrics(df, review),
                    }
                )
    scan = pd.DataFrame(rows)
    selected_rows = []
    for mode, part in scan.groupby("mode", sort=False):
        candidates = part[(part["fp"].eq(0)) & (part["fn"].le(1))].copy()
        if candidates.empty:
            candidates = part.copy()
        selected = candidates.sort_values(
            ["extra_review_n", "clean_review_n", "remaining_error_n", "balanced_accuracy"],
            ascending=[True, True, True, False],
        ).iloc[0]
        selected_rows.append(selected)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    return scan, selected


def random_capture_baseline(df: pd.DataFrame, extra_n: int, seed: int = 20260527, n_iter: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = df["v111_review_or_control"].astype(bool)
    pool = df.index[df["domain"].isin(["old_data", "third_batch"]) & (~base) & df["final_pred"].eq(1)].to_numpy()
    is_error = df.loc[pool, "final_correct"].eq(0).to_numpy()
    rows = []
    for _ in range(n_iter):
        chosen = rng.choice(len(pool), size=min(extra_n, len(pool)), replace=False)
        captured = int(is_error[chosen].sum())
        rows.append(captured)
    arr = np.asarray(rows)
    return pd.DataFrame(
        [
            {
                "pool_n": int(len(pool)),
                "pool_error_n": int(is_error.sum()),
                "random_extra_review_n": int(extra_n),
                "random_expected_captured_error_n": float(arr.mean()),
                "random_capture_p95": float(np.quantile(arr, 0.95)),
                "prob_capture_at_least_v118": float((arr >= 3).mean()),
            }
        ]
    )


def plot_heatmap(scan: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    two = scan[scan["mode"].eq("two_signal")].copy()
    two["wc_max"] = pd.to_numeric(two["wc_max"])
    two["core_max"] = pd.to_numeric(two["core_max"])
    pivot = two.pivot(index="core_max", columns="wc_max", values="remaining_error_n").sort_index(ascending=False)
    plt.figure(figsize=(8.2, 5.8))
    plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r", vmin=0, vmax=max(4, int(two["remaining_error_n"].max())))
    plt.colorbar(label="Remaining errors")
    plt.xticks(range(len(pivot.columns)), [f"{x:.3f}" for x in pivot.columns], rotation=90, fontsize=7)
    plt.yticks(range(len(pivot.index)), [f"{x:.3f}" for x in pivot.index], fontsize=8)
    plt.xlabel("wholecrop_prob max")
    plt.ylabel("prob_mean_core max")
    plt.title("Two-signal FP guard robustness grid")
    if V118_WC_MAX in list(pivot.columns) and V118_CORE_MAX in list(pivot.index):
        x = list(pivot.columns).index(V118_WC_MAX)
        y = list(pivot.index).index(V118_CORE_MAX)
        plt.scatter([x], [y], marker="x", s=90, c="red", linewidths=2, label="v118")
        plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "v122_two_signal_remaining_error_heatmap.png", dpi=220)
    plt.savefig(fig_dir / "v122_two_signal_remaining_error_heatmap.pdf")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    base = df["v111_review_or_control"].astype(bool)
    v118_extra = df["v118_extra_review"].astype(bool)
    v118_review = df["v118_review_or_control"].astype(bool)

    scan, selected = scan_single_and_two_signal(df)
    v111_row = evaluate_workflow(df, "v111_base", base, pd.Series(False, index=df.index))
    v118_row = evaluate_workflow(df, "v118_fixed_two_signal", v118_review, v118_extra)
    selected_workflows = []
    for row in selected.itertuples(index=False):
        mode = str(row.mode)
        wc = None if row.wc_max == "" else float(row.wc_max)
        core = None if row.core_max == "" else float(row.core_max)
        extra = fp_guard(df, base, mode, wc, core)
        selected_workflows.append(evaluate_workflow(df, f"selected_{mode}", base | extra, extra) | {"wc_max": row.wc_max, "core_max": row.core_max})

    workflow = pd.DataFrame([v111_row, v118_row] + selected_workflows)
    rand = random_capture_baseline(df, int(v118_extra.sum()))

    robust = scan[
        scan["mode"].eq("two_signal")
        & scan["fp"].eq(0)
        & scan["fn"].le(1)
        & scan["control_rate"].le(0.805)
    ].copy()
    robustness_summary = pd.DataFrame(
        [
            {
                "two_signal_safe_rules_control_le_80_5_n": int(len(robust)),
                "wc_min": float(pd.to_numeric(robust["wc_max"]).min()) if len(robust) else np.nan,
                "wc_max": float(pd.to_numeric(robust["wc_max"]).max()) if len(robust) else np.nan,
                "core_min": float(pd.to_numeric(robust["core_max"]).min()) if len(robust) else np.nan,
                "core_max": float(pd.to_numeric(robust["core_max"]).max()) if len(robust) else np.nan,
            }
        ]
    )

    scan.to_csv(OUT_DIR / "v122_fp_guard_full_grid.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v122_selected_single_vs_two_signal_rules.csv", index=False, encoding="utf-8-sig")
    workflow.to_csv(OUT_DIR / "v122_ablation_workflow_summary.csv", index=False, encoding="utf-8-sig")
    format_table(workflow).to_csv(OUT_DIR / "v122_ablation_workflow_summary_formatted.csv", index=False, encoding="utf-8-sig")
    rand.to_csv(OUT_DIR / "v122_random_capture_baseline.csv", index=False, encoding="utf-8-sig")
    robustness_summary.to_csv(OUT_DIR / "v122_robustness_summary.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(OUT_DIR / "v122_two_signal_safe_plateau_rules.csv", index=False, encoding="utf-8-sig")
    plot_heatmap(scan)

    v118 = workflow[workflow["workflow"].eq("v118_fixed_two_signal")].iloc[0]
    whole = workflow[workflow["workflow"].eq("selected_wholecrop_only")].iloc[0]
    core = workflow[workflow["workflow"].eq("selected_core_only")].iloc[0]
    two = workflow[workflow["workflow"].eq("selected_two_signal")].iloc[0]
    lines = [
        "# v122 Two-signal Ablation and Robustness",
        "",
        "v122 tests whether the low-risk FP guard really needs two signals rather than a single threshold.",
        "",
        f"- v118 fixed two-signal: BAcc {pct(v118['balanced_accuracy'])}, control {pct(v118['control_rate'])}, FN={int(v118['fn'])}, FP={int(v118['fp'])}, extra_review={int(v118['extra_review_n'])}.",
        f"- Best wholecrop-only guard: BAcc {pct(whole['balanced_accuracy'])}, control {pct(whole['control_rate'])}, FN={int(whole['fn'])}, FP={int(whole['fp'])}, extra_review={int(whole['extra_review_n'])}.",
        f"- Best core-only guard: BAcc {pct(core['balanced_accuracy'])}, control {pct(core['control_rate'])}, FN={int(core['fn'])}, FP={int(core['fp'])}, extra_review={int(core['extra_review_n'])}.",
        f"- Best scanned two-signal guard: BAcc {pct(two['balanced_accuracy'])}, control {pct(two['control_rate'])}, FN={int(two['fn'])}, FP={int(two['fp'])}, extra_review={int(two['extra_review_n'])}.",
        f"- Safe two-signal plateau under 80.5% control includes {int(robustness_summary.iloc[0]['two_signal_safe_rules_control_le_80_5_n'])} threshold pairs.",
        f"- Randomly reviewing {int(v118['extra_review_n'])} auto-high cases would capture {rand.iloc[0]['random_expected_captured_error_n']:.2f} FP errors on average; probability of capturing at least the v118 count is {rand.iloc[0]['prob_capture_at_least_v118']:.4f}.",
        "",
        "This supports writing the FP guard as a robust two-signal directional scorecard rather than a fragile single-threshold rule.",
    ]
    (OUT_DIR / "v122_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(format_table(workflow).to_string(index=False))
    print()
    print(selected.to_string(index=False))
    print()
    print(rand.to_string(index=False))
    print()
    print(robustness_summary.to_string(index=False))


if __name__ == "__main__":
    main()
