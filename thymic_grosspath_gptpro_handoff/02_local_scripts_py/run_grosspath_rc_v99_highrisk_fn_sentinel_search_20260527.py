from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v99_highrisk_fn_sentinel_search_20260527"
FIG_DIR = OUT_DIR / "figures"

BASE_POLICY = "adaptive_v50_to_v79_light"
REFERENCE_POLICY = "v79_light_lowrisk_guard"
MAX_ALL_CONTROL = 0.80


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["final_wrong"] = out["final_correct"].eq(0)
    out["main_robust_gap"] = (out["main_prob"] - out["robust_prob"]).abs()
    return out


def selective_metrics(df: pd.DataFrame, review: pd.Series | np.ndarray) -> dict[str, float | int]:
    review = pd.Series(review, index=df.index).astype(bool)
    remaining_wrong = (~review) & df["final_wrong"]
    fn = int((remaining_wrong & df["label_idx"].eq(1) & df["final_pred"].eq(0)).sum())
    fp = int((remaining_wrong & df["label_idx"].eq(0) & df["final_pred"].eq(1)).sum())
    pos = int(df["label_idx"].eq(1).sum())
    neg = int(df["label_idx"].eq(0).sum())
    sens = (pos - fn) / pos if pos else np.nan
    spec = (neg - fp) / neg if neg else np.nan
    return {
        "n": len(df),
        "control_n": int(review.sum()),
        "control_rate": float(review.mean()),
        "auto_n": int((~review).sum()),
        "remaining_error_n": int(remaining_wrong.sum()),
        "fn": fn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def candidate_extra_review(df: pd.DataFrame, feature: str, threshold: float) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    if feature == "main_prob":
        signal = df["main_prob"]
    elif feature == "robust_prob":
        signal = df["robust_prob"]
    elif feature == "prob_mean_core":
        signal = df["prob_mean_core"]
    elif feature == "main_robust_gap":
        signal = df["main_robust_gap"]
    else:
        raise ValueError(feature)
    return low_auto & signal.ge(threshold)


def evaluate_rule(df: pd.DataFrame, feature: str, threshold: float) -> dict[str, object]:
    extra = candidate_extra_review(df, feature, threshold)
    review = df["review_or_control"].astype(bool) | extra
    all_metrics = selective_metrics(df, review)
    row: dict[str, object] = {
        "feature": feature,
        "threshold": threshold,
        "extra_review_n": int(extra.sum()),
        "extra_captured_error_n": int((extra & df["final_wrong"]).sum()),
    }
    row.update({f"all_{k}": v for k, v in all_metrics.items()})
    for domain, sub in df.groupby("domain", sort=False):
        metrics = selective_metrics(sub, review.loc[sub.index])
        row.update({f"{domain}_{k}": v for k, v in metrics.items()})
    return row


def search_rules(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = {
        "main_prob": np.arange(0.15, 0.451, 0.025),
        "robust_prob": np.arange(0.15, 0.451, 0.025),
        "prob_mean_core": np.arange(0.15, 0.451, 0.025),
        "main_robust_gap": np.arange(0.05, 0.201, 0.025),
    }
    for feature, values in thresholds.items():
        for threshold in values:
            rows.append(evaluate_rule(df, feature, float(round(threshold, 3))))
    out = pd.DataFrame(rows)
    return out.sort_values(["all_balanced_accuracy", "third_batch_balanced_accuracy", "all_control_rate"], ascending=[False, False, True]).reset_index(drop=True)


def selected_rule(search: pd.DataFrame) -> pd.Series:
    feasible = search.loc[
        search["all_control_rate"].le(MAX_ALL_CONTROL)
        & search["strict_external_fn"].eq(0)
        & search["strict_external_fp"].le(1)
    ].copy()
    if feasible.empty:
        feasible = search.copy()
    return feasible.sort_values(
        ["all_balanced_accuracy", "third_batch_balanced_accuracy", "all_control_rate"],
        ascending=[False, False, True],
    ).iloc[0]


def workflow_rows(df: pd.DataFrame, label: str, review: pd.Series) -> list[dict[str, object]]:
    rows = []
    scopes = [("all_domains", df)] + list(df.groupby("domain", sort=False))
    for scope, sub in scopes:
        metrics = selective_metrics(sub, review.loc[sub.index])
        rows.append({"workflow": label, "scope": scope, **metrics})
    return rows


def build_comparison(routes: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    base = add_features(routes.loc[routes["policy"].eq(BASE_POLICY)].copy())
    ref = add_features(routes.loc[routes["policy"].eq(REFERENCE_POLICY)].copy())
    extra = candidate_extra_review(base, str(selected["feature"]), float(selected["threshold"]))
    sentinel_review = base["review_or_control"].astype(bool) | extra
    rows = []
    rows.extend(workflow_rows(base, "Batch-adaptive main", base["review_or_control"].astype(bool)))
    rows.extend(workflow_rows(ref, "Fixed v79-light", ref["review_or_control"].astype(bool)))
    rows.extend(workflow_rows(base, f"v99 FN sentinel ({selected['feature']} >= {selected['threshold']:.3f})", sentinel_review))
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["balanced_accuracy", "sensitivity", "specificity"]:
            out[col] = out[col].map(pct)
    for col in out.select_dtypes(include=[float]).columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


def make_plots(search: pd.DataFrame, comparison: pd.DataFrame, selected: pd.Series) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    feasible = search["all_control_rate"].le(MAX_ALL_CONTROL)
    ax.scatter(
        search.loc[~feasible, "all_control_rate"] * 100,
        search.loc[~feasible, "all_balanced_accuracy"] * 100,
        s=40,
        color="#B0BEC5",
        label="over control budget",
    )
    sc = ax.scatter(
        search.loc[feasible, "all_control_rate"] * 100,
        search.loc[feasible, "all_balanced_accuracy"] * 100,
        c=search.loc[feasible, "third_batch_fn"],
        cmap="Reds_r",
        s=60,
        edgecolor="white",
        linewidth=0.6,
        label="feasible",
    )
    ax.scatter(
        selected["all_control_rate"] * 100,
        selected["all_balanced_accuracy"] * 100,
        s=150,
        marker="*",
        color="#1B5E20",
        edgecolor="white",
        linewidth=0.8,
        label="selected",
    )
    ax.axvline(MAX_ALL_CONTROL * 100, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("All-domain control rate (%)")
    ax.set_ylabel("All-domain balanced accuracy (%)")
    ax.set_title("High-risk FN sentinel search")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Third-batch FN")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v99_fn_sentinel_search.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v99_fn_sentinel_search.pdf", bbox_inches="tight")
    plt.close(fig)

    focus = comparison.loc[comparison["scope"].isin(["all_domains", "third_batch", "strict_external"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for workflow, sub in focus.groupby("workflow", sort=False):
        axes[0].plot(sub["scope"], sub["balanced_accuracy"] * 100, marker="o", label=workflow)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_title("BAcc comparison")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    third = comparison.loc[comparison["scope"].eq("third_batch")].copy()
    x = np.arange(len(third))
    axes[1].bar(x - 0.18, third["fn"], width=0.36, label="FN", color="#C62828")
    axes[1].bar(x + 0.18, third["fp"], width=0.36, label="FP", color="#1565C0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(third["workflow"], rotation=25, ha="right")
    axes[1].set_ylabel("Third-batch residual errors")
    axes[1].set_title("Third-batch error direction")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v99_fn_sentinel_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v99_fn_sentinel_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(selected: pd.Series, comparison: pd.DataFrame) -> None:
    sentinel = comparison.loc[comparison["workflow"].str.startswith("v99") & comparison["scope"].eq("all_domains")].iloc[0]
    sentinel_third = comparison.loc[comparison["workflow"].str.startswith("v99") & comparison["scope"].eq("third_batch")].iloc[0]
    base_third = comparison.loc[(comparison["workflow"] == "Batch-adaptive main") & (comparison["scope"] == "third_batch")].iloc[0]
    fixed = comparison.loc[(comparison["workflow"] == "Fixed v79-light") & (comparison["scope"] == "all_domains")].iloc[0]
    lines = [
        "# v99 High-risk FN Sentinel Search",
        "",
        "## Selected Exploratory Rule",
        "",
        f"- Rule: among auto-passed low-risk predictions, send cases to review if `{selected['feature']}` >= {selected['threshold']:.3f}.",
        f"- All-domain control rate: {pct(sentinel['control_rate'])}; all-domain BAcc: {pct(sentinel['balanced_accuracy'])}; FN/FP: {int(sentinel['fn'])}/{int(sentinel['fp'])}.",
        f"- Third batch improves from BAcc {pct(base_third['balanced_accuracy'])}, FN={int(base_third['fn'])}, FP={int(base_third['fp'])} to BAcc {pct(sentinel_third['balanced_accuracy'])}, FN={int(sentinel_third['fn'])}, FP={int(sentinel_third['fp'])}.",
        f"- Fixed v79-light all-domain BAcc is {pct(fixed['balanced_accuracy'])} at control {pct(fixed['control_rate'])}; the sentinel reaches similar control but shifts the error profile toward fewer FN and more FP.",
        "",
        "## Boundary",
        "",
        "This is an exploratory internal safety rule derived after residual-error analysis. It is useful as a high-risk-protection candidate, but it should be validated with nested selection or a new held-out batch before being treated as a final locked workflow.",
        "",
    ]
    (OUT_DIR / "v99_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES)
    base = add_features(routes.loc[routes["policy"].eq(BASE_POLICY)].copy())
    search = search_rules(base)
    selected = selected_rule(search)
    comparison = build_comparison(routes, selected)

    search.to_csv(OUT_DIR / "v99_fn_sentinel_search.csv", index=False, encoding="utf-8-sig")
    search.head(30).to_csv(OUT_DIR / "v99_fn_sentinel_top30.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "v99_workflow_comparison.csv", index=False, encoding="utf-8-sig")
    format_table(search.head(30)).to_csv(OUT_DIR / "v99_fn_sentinel_top30_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(comparison).to_csv(OUT_DIR / "v99_workflow_comparison_formatted.csv", index=False, encoding="utf-8-sig")
    make_plots(search, comparison, selected)
    write_summary(selected, comparison)

    print("Wrote", OUT_DIR)
    print("Selected:", selected[["feature", "threshold", "extra_review_n", "all_control_rate", "all_balanced_accuracy", "third_batch_fn", "third_batch_fp", "strict_external_fn", "strict_external_fp"]].to_string())
    print()
    print(format_table(comparison.loc[comparison["scope"].isin(["all_domains", "third_batch", "strict_external"])]).to_string(index=False))


if __name__ == "__main__":
    main()
