from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527" / "v80_tiered_lowrisk_guard_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v87_triage_efficiency_curve_20260527"
FIG_DIR = OUT_DIR / "figures"

POLICY_ORDER = [
    "P2_pure_auto",
    "v50_main",
    "v75_quality_lowconf",
    "v79_light_lowrisk_guard",
    "v79_strict_lowrisk_guard",
]

POLICY_LABELS = {
    "P2_pure_auto": "Pure auto",
    "v50_main": "v50",
    "v75_quality_lowconf": "v75",
    "v79_light_lowrisk_guard": "v79-light",
    "v79_strict_lowrisk_guard": "v79-strict",
}

DOMAIN_LABELS = {
    "old_data": "Old data",
    "third_batch": "Third batch",
    "strict_external": "Strict external",
}

DOMAIN_ORDER = ["old_data", "third_batch", "strict_external"]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else np.nan


def confusion_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    # Balanced accuracy is not meaningful when the evaluated subset contains
    # only one true class, which can happen after strict selective review.
    bacc = (sens + spec) / 2 if (tp + fn) and (tn + fp) else np.nan
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_n": tp + fn,
        "negative_n": tn + fp,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "sensitivity": sens,
        "specificity": spec,
    }


def summarize_policy(df: pd.DataFrame, domain: str, policy: str) -> dict[str, object]:
    n = len(df)
    if policy == "P2_pure_auto":
        review = pd.Series(False, index=df.index)
        final_pred = df["p2_pred"].astype(int)
    else:
        review = df["review_or_control"].astype(int).eq(1)
        # Reviewed cases are treated as corrected by the manual/secondary stage.
        final_pred = df["p2_pred"].astype(int).copy()
        final_pred.loc[review] = df.loc[review, "label_idx"].astype(int)

    auto = ~review
    p2_wrong = df["p2_wrong"].astype(int).eq(1)
    pure_auto_wrong_n = int(p2_wrong.sum())
    pure_auto_fn_n = int(((df["label_idx"].astype(int) == 1) & (df["p2_pred"].astype(int) == 0)).sum())
    pure_auto_fp_n = int(((df["label_idx"].astype(int) == 0) & (df["p2_pred"].astype(int) == 1)).sum())

    captured_wrong_n = int((review & p2_wrong).sum())
    captured_fn_n = int((review & (df["label_idx"].astype(int) == 1) & (df["p2_pred"].astype(int) == 0)).sum())
    captured_fp_n = int((review & (df["label_idx"].astype(int) == 0) & (df["p2_pred"].astype(int) == 1)).sum())

    final = confusion_metrics(df["label_idx"], final_pred)
    auto_metrics = confusion_metrics(df.loc[auto, "label_idx"], df.loc[auto, "p2_pred"]) if int(auto.sum()) else {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
    }

    control_n = int(review.sum())
    auto_n = int(auto.sum())
    remaining_error_n = int((auto & p2_wrong).sum())

    return {
        "domain": domain,
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "n": n,
        "control_n": control_n,
        "control_rate": safe_div(control_n, n),
        "auto_n": auto_n,
        "auto_rate": safe_div(auto_n, n),
        "pure_auto_wrong_n": pure_auto_wrong_n,
        "pure_auto_fn_n": pure_auto_fn_n,
        "pure_auto_fp_n": pure_auto_fp_n,
        "pure_auto_error_rate": safe_div(pure_auto_wrong_n, n),
        "captured_wrong_n": captured_wrong_n,
        "captured_fn_n": captured_fn_n,
        "captured_fp_n": captured_fp_n,
        "remaining_error_n": remaining_error_n,
        "remaining_fn_n": int(auto_metrics["fn"]),
        "remaining_fp_n": int(auto_metrics["fp"]),
        "error_capture_rate": safe_div(captured_wrong_n, pure_auto_wrong_n),
        "fn_capture_rate": safe_div(captured_fn_n, pure_auto_fn_n),
        "fp_capture_rate": safe_div(captured_fp_n, pure_auto_fp_n),
        "review_yield": safe_div(captured_wrong_n, control_n),
        "review_yield_enrichment": safe_div(safe_div(captured_wrong_n, control_n), safe_div(pure_auto_wrong_n, n)),
        "controls_per_captured_error": safe_div(control_n, captured_wrong_n),
        "final_accuracy": final["accuracy"],
        "final_balanced_accuracy": final["balanced_accuracy"],
        "final_sensitivity": final["sensitivity"],
        "final_specificity": final["specificity"],
        "auto_subset_accuracy": auto_metrics["accuracy"],
        "auto_subset_balanced_accuracy": auto_metrics["balanced_accuracy"],
        "auto_subset_positive_n": int(auto_metrics["positive_n"]),
        "auto_subset_negative_n": int(auto_metrics["negative_n"]),
        "auto_subset_sensitivity": auto_metrics["sensitivity"],
        "auto_subset_specificity": auto_metrics["specificity"],
    }


def build_summary(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, domain_df in routes.groupby("domain", sort=False):
        base = domain_df.loc[domain_df["policy"].eq("v50_main")].copy()
        rows.append(summarize_policy(base, domain, "P2_pure_auto"))
        for policy in POLICY_ORDER[1:]:
            sub = domain_df.loc[domain_df["policy"].eq(policy)].copy()
            rows.append(summarize_policy(sub, domain, policy))
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], categories=POLICY_ORDER, ordered=True)
    out["domain"] = pd.Categorical(out["domain"], categories=DOMAIN_ORDER, ordered=True)
    return out.sort_values(["domain", "policy"]).reset_index(drop=True)


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def fmt_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "domain",
        "policy_label",
        "control_rate",
        "auto_subset_accuracy",
        "auto_subset_balanced_accuracy",
        "final_balanced_accuracy",
        "remaining_error_n",
        "remaining_fn_n",
        "remaining_fp_n",
        "error_capture_rate",
        "review_yield",
        "controls_per_captured_error",
    ]
    out = summary[cols].copy()
    out["domain"] = out["domain"].map(DOMAIN_LABELS)
    for col in [
        "control_rate",
        "auto_subset_accuracy",
        "auto_subset_balanced_accuracy",
        "final_balanced_accuracy",
        "error_capture_rate",
        "review_yield",
    ]:
        out[col] = out[col].map(pct)
    out["controls_per_captured_error"] = out["controls_per_captured_error"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["auto_subset_balanced_accuracy"] = out["auto_subset_balanced_accuracy"].replace("", "NA")
    return out


def make_figures(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = summary.copy()
    plot_df["domain_label"] = plot_df["domain"].map(DOMAIN_LABELS)
    colors_by_policy = {
        "P2_pure_auto": "#666666",
        "v50_main": "#1f77b4",
        "v75_quality_lowconf": "#2ca02c",
        "v79_light_lowrisk_guard": "#ff7f0e",
        "v79_strict_lowrisk_guard": "#d62728",
    }
    markers_by_policy = {
        "P2_pure_auto": "o",
        "v50_main": "s",
        "v75_quality_lowconf": "^",
        "v79_light_lowrisk_guard": "D",
        "v79_strict_lowrisk_guard": "P",
    }

    metrics = [
        ("control_rate", "final_balanced_accuracy", "Final BAcc", "v87_control_vs_final_bacc"),
        ("control_rate", "auto_subset_balanced_accuracy", "Auto-passed subset BAcc", "v87_control_vs_autopass_bacc"),
        ("control_rate", "error_capture_rate", "Original error capture rate", "v87_control_vs_error_capture"),
        ("control_rate", "remaining_error_n", "Remaining errors after workflow", "v87_control_vs_remaining_errors"),
    ]
    for x_col, y_col, y_label, name in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), sharey=False)
        for ax, domain in zip(axes, DOMAIN_ORDER):
            sub = plot_df.loc[plot_df["domain"].eq(domain)].sort_values("policy")
            ax.plot(
                sub[x_col] * 100,
                sub[y_col] * (100 if "rate" in y_col or "accuracy" in y_col else 1),
                color="#B8C7D9",
                linewidth=1.4,
                zorder=1,
            )
            for _, row in sub.iterrows():
                y = row[y_col] * (100 if "rate" in y_col or "accuracy" in y_col else 1)
                ax.scatter(
                    row[x_col] * 100,
                    y,
                    s=50,
                    marker=markers_by_policy[row["policy"]],
                    color=colors_by_policy[row["policy"]],
                    label=row["policy_label"],
                    zorder=2,
                )
            ax.set_title(DOMAIN_LABELS.get(domain, domain))
            ax.set_xlabel("Review/control rate (%)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(y_label + (" (%)" if "rate" in y_col or "accuracy" in y_col else ""))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(POLICY_ORDER), frameon=False, bbox_to_anchor=(0.5, 1.05))
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def write_key_messages(summary: pd.DataFrame) -> None:
    strict = summary.loc[summary["domain"].eq("strict_external")]
    third = summary.loc[summary["domain"].eq("third_batch")]
    light_strict = strict.loc[strict["policy"].eq("v79_light_lowrisk_guard")].iloc[0]
    strict_strict = strict.loc[strict["policy"].eq("v79_strict_lowrisk_guard")].iloc[0]
    v50_strict = strict.loc[strict["policy"].eq("v50_main")].iloc[0]
    light_third = third.loc[third["policy"].eq("v79_light_lowrisk_guard")].iloc[0]
    text = f"""# v87 复核效率与放行安全性分析

## 核心发现

- 严格外部集上，v50 在 {pct(v50_strict['control_rate'])} 控制率下把原始自动错误从 {int(v50_strict['pure_auto_wrong_n'])} 例压到 {int(v50_strict['remaining_error_n'])} 例，错误捕获率为 {pct(v50_strict['error_capture_rate'])}。
- 严格外部集上，v79-light 在 {pct(light_strict['control_rate'])} 控制率下只剩 {int(light_strict['remaining_error_n'])} 个错误，放行子集准确率为 {pct(light_strict['auto_subset_accuracy'])}，原始错误捕获率为 {pct(light_strict['error_capture_rate'])}。
- 严格外部集上，v79-strict 在 {pct(strict_strict['control_rate'])} 控制率下把剩余错误压到 0，原始错误捕获率为 {pct(strict_strict['error_capture_rate'])}。该结果仍应作为候选高安全版本，不能脱离样本量限制夸大。
- 第三批上，v79-light 在 {pct(light_third['control_rate'])} 控制率下剩余 {int(light_third['remaining_error_n'])} 个错误，原始错误捕获率为 {pct(light_third['error_capture_rate'])}，说明双向保护不是只对外部集有效。

## 论文表达价值

这一版补的是“复核是否值得”的证据。我们不仅报告最终 BAcc，还能说明：随着控制率提高，放行子集的安全性如何变化、原始错误有多少被复核池截获、FN/FP 各被截获多少。这能把当前流程从一个后处理阈值，写成风险控制型 selective diagnosis framework。
"""
    (OUT_DIR / "v87_key_messages.md").write_text(text, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES)
    summary = build_summary(routes)
    summary.to_csv(OUT_DIR / "v87_triage_efficiency_summary.csv", index=False, encoding="utf-8-sig")
    fmt_summary(summary).to_csv(OUT_DIR / "v87_triage_efficiency_summary_formatted.csv", index=False, encoding="utf-8-sig")
    make_figures(summary)
    write_key_messages(summary)

    print("v87 formatted summary:")
    print(fmt_summary(summary).to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
