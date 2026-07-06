from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v98_third_batch_residual_error_anatomy_20260527"
FIG_DIR = OUT_DIR / "figures"
POLICY = "adaptive_v50_to_v79_light"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["truth_group"] = np.where(out["label_idx"].eq(1), "high_risk", "low_risk")
    out["p2_pred_group"] = np.where(out["p2_pred"].eq(1), "high_risk", "low_risk")
    out["final_pred_group"] = np.where(out["final_pred"].eq(1), "high_risk", "low_risk")
    out["final_wrong"] = out["final_correct"].eq(0)
    out["final_error_direction"] = np.select(
        [
            out["final_wrong"] & out["label_idx"].eq(1) & out["final_pred"].eq(0),
            out["final_wrong"] & out["label_idx"].eq(0) & out["final_pred"].eq(1),
        ],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    out["abs_prob_margin_from_05"] = (out["prob_mean_core"] - 0.5).abs()
    out["main_robust_gap"] = (out["main_prob"] - out["robust_prob"]).abs()
    return out


def summarize_domain(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, sub in df.groupby("domain", sort=False):
        wrong = sub["final_wrong"]
        fn = ((sub["label_idx"] == 1) & (sub["final_pred"] == 0)).sum()
        fp = ((sub["label_idx"] == 0) & (sub["final_pred"] == 1)).sum()
        rows.append(
            {
                "domain": domain,
                "n": len(sub),
                "control_rate": sub["review_or_control"].mean(),
                "remaining_error_n": int(wrong.sum()),
                "remaining_error_rate": wrong.mean(),
                "fn": int(fn),
                "fp": int(fp),
                "median_core_prob_correct": sub.loc[~wrong, "prob_mean_core"].median(),
                "median_core_prob_wrong": sub.loc[wrong, "prob_mean_core"].median(),
                "median_abs_margin_correct": sub.loc[~wrong, "abs_prob_margin_from_05"].median(),
                "median_abs_margin_wrong": sub.loc[wrong, "abs_prob_margin_from_05"].median(),
                "median_main_robust_gap_correct": sub.loc[~wrong, "main_robust_gap"].median(),
                "median_main_robust_gap_wrong": sub.loc[wrong, "main_robust_gap"].median(),
            }
        )
    return pd.DataFrame(rows)


def group_error_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["domain", group_col], dropna=False):
        domain, group = keys
        wrong = sub["final_wrong"]
        rows.append(
            {
                "domain": domain,
                group_col: group,
                "n": len(sub),
                "remaining_error_n": int(wrong.sum()),
                "remaining_error_rate": wrong.mean(),
                "fn": int(((sub["label_idx"] == 1) & (sub["final_pred"] == 0)).sum()),
                "fp": int(((sub["label_idx"] == 0) & (sub["final_pred"] == 1)).sum()),
                "control_rate": sub["review_or_control"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "remaining_error_n", "n"], ascending=[True, False, False])


def third_case_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "original_case_id",
        "task_l6_label",
        "task_l7_label",
        "view_type_final",
        "image_name",
        "truth_group",
        "p2_pred_group",
        "final_pred_group",
        "final_error_direction",
        "review_or_control",
        "main_prob",
        "robust_prob",
        "prob_mean_core",
        "core_agree_count",
        "abs_prob_margin_from_05",
        "main_robust_gap",
        "quality_score",
        "quality_status",
    ]
    third = df.loc[df["domain"].eq("third_batch") & df["final_wrong"]].copy()
    return third[cols].sort_values(["final_error_direction", "task_l6_label", "original_case_id"]).reset_index(drop=True)


def make_plots(domain_summary: pd.DataFrame, l6: pd.DataFrame, third_cases: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(domain_summary))
    axes[0].bar(x, domain_summary["fn"], label="FN high->low", color="#C62828")
    axes[0].bar(x, domain_summary["fp"], bottom=domain_summary["fn"], label="FP low->high", color="#1565C0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domain_summary["domain"], rotation=20, ha="right")
    axes[0].set_ylabel("Remaining errors")
    axes[0].set_title("Residual errors by domain")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    third_l6 = l6.loc[l6["domain"].eq("third_batch") & (l6["remaining_error_n"] > 0)].copy()
    third_l6 = third_l6.sort_values("remaining_error_n", ascending=False)
    axes[1].bar(third_l6["task_l6_label"].astype(str), third_l6["remaining_error_n"], color="#6A4C93")
    axes[1].set_title("Third batch residual errors by L6 label")
    axes[1].set_ylabel("Remaining errors")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v98_residual_error_anatomy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v98_residual_error_anatomy.pdf", bbox_inches="tight")
    plt.close(fig)

    if not third_cases.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        colors = np.where(third_cases["final_error_direction"].eq("FN_high_to_low"), "#C62828", "#1565C0")
        ax.scatter(
            third_cases["prob_mean_core"],
            third_cases["main_robust_gap"],
            s=90,
            c=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axvline(0.5, color="#444444", linestyle="--", linewidth=1)
        for _, row in third_cases.iterrows():
            ax.text(row["prob_mean_core"] + 0.005, row["main_robust_gap"] + 0.005, str(row["original_case_id"]), fontsize=7)
        ax.set_xlabel("Core mean high-risk probability")
        ax.set_ylabel("Main vs robust probability gap")
        ax.set_title("Third batch residual cases")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "v98_third_batch_residual_case_scatter.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "v98_third_batch_residual_case_scatter.pdf", bbox_inches="tight")
        plt.close(fig)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["control_rate"]:
            out[col] = out[col].map(pct)
    for col in out.select_dtypes(include=[float]).columns:
        if col not in out.columns[out.columns.str.endswith("_rate")].tolist() and col != "control_rate":
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


def write_summary(domain_summary: pd.DataFrame, l6: pd.DataFrame, view: pd.DataFrame, cases: pd.DataFrame) -> None:
    third = domain_summary.loc[domain_summary["domain"].eq("third_batch")].iloc[0]
    top_l6 = l6.loc[l6["domain"].eq("third_batch") & (l6["remaining_error_n"] > 0)].sort_values("remaining_error_n", ascending=False)
    top_l6_text = "; ".join([f"{r.task_l6_label}: {int(r.remaining_error_n)}" for r in top_l6.itertuples()])
    view_err = view.loc[view["domain"].eq("third_batch") & (view["remaining_error_n"] > 0)].sort_values("remaining_error_n", ascending=False)
    view_text = "; ".join([f"{r.view_type_final}: {int(r.remaining_error_n)}/{int(r.n)}" for r in view_err.itertuples()])
    lines = [
        "# v98 Third Batch Residual Error Anatomy",
        "",
        "## Key Findings",
        "",
        f"- Under Batch-adaptive main, third batch has {int(third['remaining_error_n'])} residual errors: FN={int(third['fn'])}, FP={int(third['fp'])}. This is the main residual weakness after old data and strict external are controlled.",
        f"- Third-batch errors are concentrated by L6 label: {top_l6_text}.",
        f"- View-level distribution among wrong cases: {view_text}.",
        f"- Wrong third-batch cases have median core probability {third['median_core_prob_wrong']:.3f} and median main/robust gap {third['median_main_robust_gap_wrong']:.3f}, so part of the issue is not only image-domain shift but high-confidence boundary behavior inside the internal domain.",
        "",
        "## Files",
        "",
        "- v98_domain_residual_summary.csv",
        "- v98_error_by_l6.csv",
        "- v98_error_by_view.csv",
        "- v98_third_batch_residual_cases.csv",
        "- figures/v98_residual_error_anatomy.png/pdf",
        "- figures/v98_third_batch_residual_case_scatter.png/pdf",
        "",
    ]
    (OUT_DIR / "v98_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES)
    df = routes.loc[routes["policy"].eq(POLICY)].copy()
    df = add_labels(df)

    domain_summary = summarize_domain(df)
    l6 = group_error_table(df, "task_l6_label")
    view = group_error_table(df, "view_type_final")
    cases = third_case_table(df)

    domain_summary.to_csv(OUT_DIR / "v98_domain_residual_summary.csv", index=False, encoding="utf-8-sig")
    l6.to_csv(OUT_DIR / "v98_error_by_l6.csv", index=False, encoding="utf-8-sig")
    view.to_csv(OUT_DIR / "v98_error_by_view.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v98_third_batch_residual_cases.csv", index=False, encoding="utf-8-sig")

    format_table(domain_summary).to_csv(OUT_DIR / "v98_domain_residual_summary_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(l6).to_csv(OUT_DIR / "v98_error_by_l6_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(view).to_csv(OUT_DIR / "v98_error_by_view_formatted.csv", index=False, encoding="utf-8-sig")
    make_plots(domain_summary, l6, cases)
    write_summary(domain_summary, l6, view, cases)

    print("Wrote", OUT_DIR)
    print(format_table(domain_summary).to_string(index=False))
    print()
    print(format_table(l6.loc[l6["domain"].eq("third_batch") & (l6["remaining_error_n"] > 0)]).to_string(index=False))
    print()
    print(cases[["original_case_id", "task_l6_label", "view_type_final", "truth_group", "final_pred_group", "final_error_direction", "prob_mean_core", "main_robust_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
