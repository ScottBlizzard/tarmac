from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
V101_CASES = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527" / "v101_internal_cases_with_multimodel_features.csv"
V101_RULES = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527" / "v101_fold_selected_rules.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v103_multimodel_sentinel_mechanism_20260527"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 20260527


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def extra_review(df: pd.DataFrame, signal: str, threshold: float) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    return low_auto & pd.to_numeric(df[signal], errors="coerce").ge(threshold)


def reconstruct_nested(cases: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy().reset_index(drop=True)
    out["base_final_wrong"] = out["final_correct"].eq(0)
    out["base_fn"] = out["base_final_wrong"] & out["label_idx"].eq(1) & out["final_pred"].eq(0)
    out["base_fp"] = out["base_final_wrong"] & out["label_idx"].eq(0) & out["final_pred"].eq(1)
    out["nested_extra_review"] = False
    out["nested_rule_label"] = ""
    out["nested_signal"] = ""
    out["nested_threshold"] = np.nan

    y_strata = out["domain"].astype(str) + "_" + out["label_idx"].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rule_map = {int(r.fold): r for r in rules.itertuples()}
    for fold, (_, test_idx) in enumerate(cv.split(out, y_strata), start=1):
        rule = rule_map[fold]
        signal = str(rule.signal)
        threshold = float(rule.threshold)
        sub = out.iloc[test_idx]
        extra = extra_review(sub, signal, threshold)
        out.loc[test_idx, "nested_extra_review"] = extra.to_numpy()
        out.loc[test_idx, "nested_rule_label"] = f"{signal}>={threshold:.3f}"
        out.loc[test_idx, "nested_signal"] = signal
        out.loc[test_idx, "nested_threshold"] = threshold

    out["nested_review_or_control"] = out["review_or_control"].astype(bool) | out["nested_extra_review"].astype(bool)
    out["nested_remaining_wrong"] = (~out["nested_review_or_control"]) & out["base_final_wrong"]
    out["nested_rescued_error"] = out["nested_extra_review"] & out["base_final_wrong"]
    out["nested_extra_clean_review"] = out["nested_extra_review"] & (~out["base_final_wrong"])
    out["truth_group"] = np.where(out["label_idx"].eq(1), "high_risk", "low_risk")
    out["base_pred_group"] = np.where(out["final_pred"].eq(1), "high_risk", "low_risk")
    out["error_direction"] = np.select(
        [out["base_fn"], out["base_fp"]],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    return out


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["domain", "task_l6_label"], dropna=False):
        domain, label = keys
        rows.append(
            {
                "domain": domain,
                "task_l6_label": label,
                "n": len(sub),
                "base_error_n": int(sub["base_final_wrong"].sum()),
                "base_fn_n": int(sub["base_fn"].sum()),
                "base_fp_n": int(sub["base_fp"].sum()),
                "extra_review_n": int(sub["nested_extra_review"].sum()),
                "rescued_error_n": int(sub["nested_rescued_error"].sum()),
                "extra_clean_review_n": int(sub["nested_extra_clean_review"].sum()),
                "remaining_error_n": int(sub["nested_remaining_wrong"].sum()),
                "remaining_fn_n": int((sub["nested_remaining_wrong"] & sub["label_idx"].eq(1) & sub["final_pred"].eq(0)).sum()),
                "remaining_fp_n": int((sub["nested_remaining_wrong"] & sub["label_idx"].eq(0) & sub["final_pred"].eq(1)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "rescued_error_n", "base_error_n"], ascending=[True, False, False])


def case_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [
        "case_id",
        "original_case_id",
        "domain",
        "task_l6_label",
        "task_l7_label",
        "truth_group",
        "base_pred_group",
        "error_direction",
        "nested_rule_label",
        "main_prob",
        "robust_prob",
        "prob_mean_core",
        "selected_unified_prob",
        "selected_dinov3_prob",
        "mm_prob_p75",
        "mm_prob_max",
        "mm_vote_ge50",
        "mm_max_minus_core",
    ]
    rescued = df.loc[df["nested_rescued_error"], cols].sort_values(["domain", "task_l6_label", "original_case_id"]).reset_index(drop=True)
    extra_clean = df.loc[df["nested_extra_clean_review"], cols].sort_values(["domain", "task_l6_label", "original_case_id"]).reset_index(drop=True)
    remaining = df.loc[df["nested_remaining_wrong"], cols].sort_values(["domain", "task_l6_label", "original_case_id"]).reset_index(drop=True)
    return rescued, extra_clean, remaining


def make_plots(df: pd.DataFrame, group: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    third = df.loc[df["domain"].eq("third_batch")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = np.select(
        [
            third["nested_rescued_error"],
            third["nested_remaining_wrong"],
            third["nested_extra_clean_review"],
        ],
        ["#2E7D32", "#C62828", "#F9A825"],
        default="#B0BEC5",
    )
    axes[0].scatter(third["prob_mean_core"], third["selected_unified_prob"], c=colors, s=70, edgecolor="white", linewidth=0.5)
    axes[0].axvline(0.5, color="#666666", linestyle="--", linewidth=1)
    axes[0].axhline(0.25, color="#2E7D32", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Current core mean high-risk probability")
    axes[0].set_ylabel("Selected unified high-risk probability")
    axes[0].set_title("Third batch: current model vs multimodel evidence")
    axes[0].grid(alpha=0.25)

    g = group.loc[group["domain"].eq("third_batch") & ((group["rescued_error_n"] > 0) | (group["remaining_error_n"] > 0))].copy()
    x = np.arange(len(g))
    axes[1].bar(x - 0.2, g["rescued_error_n"], width=0.4, label="rescued", color="#2E7D32")
    axes[1].bar(x + 0.2, g["remaining_error_n"], width=0.4, label="remaining", color="#C62828")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g["task_l6_label"].astype(str), rotation=20)
    axes[1].set_ylabel("Cases")
    axes[1].set_title("Third batch rescue by L6 label")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v103_multimodel_sentinel_mechanism.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v103_multimodel_sentinel_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, rescued: pd.DataFrame, extra_clean: pd.DataFrame, remaining: pd.DataFrame, group: pd.DataFrame) -> None:
    third = df.loc[df["domain"].eq("third_batch")]
    third_rescued = rescued.loc[rescued["domain"].eq("third_batch")]
    third_remaining = remaining.loc[remaining["domain"].eq("third_batch")]
    third_clean = extra_clean.loc[extra_clean["domain"].eq("third_batch")]
    rescued_ids = ", ".join([str(x) for x in third_rescued["original_case_id"].tolist()])
    remaining_ids = ", ".join([str(x) for x in third_remaining["original_case_id"].tolist()])
    l6 = group.loc[group["domain"].eq("third_batch") & ((group["rescued_error_n"] > 0) | (group["remaining_error_n"] > 0))]
    l6_text = "; ".join(
        [
            f"{r.task_l6_label}: rescued {int(r.rescued_error_n)}, remaining {int(r.remaining_error_n)}"
            for r in l6.itertuples()
        ]
    )
    lines = [
        "# v103 Multimodel Sentinel Mechanism",
        "",
        "## Key Findings",
        "",
        f"- In third batch, v101 nested multimodel sentinel adds {int(third['nested_extra_review'].sum())} reviews: {len(third_rescued)} true residual errors captured and {len(third_clean)} extra clean reviews.",
        f"- Rescued third-batch error case IDs: {rescued_ids}.",
        f"- Remaining third-batch error case IDs: {remaining_ids}.",
        f"- L6 breakdown: {l6_text}.",
        "- Mechanistically, the rescued cases are current-workflow auto-low-risk cases where another model family still assigns a non-trivial high-risk probability. This supports the interpretation that v101 is model-family evidence aggregation rather than a simple confidence threshold.",
        "",
        "## Boundary",
        "",
        "This remains an internal mechanism analysis. It explains why v101 helps third batch, but external claim still requires exact feature-head extraction on strict external or a new prospective batch.",
        "",
    ]
    (OUT_DIR / "v103_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(V101_CASES)
    rules = pd.read_csv(V101_RULES)
    nested = reconstruct_nested(cases, rules)
    group = summarize_groups(nested)
    rescued, extra_clean, remaining = case_tables(nested)

    nested.to_csv(OUT_DIR / "v103_internal_cases_with_nested_multimodel_flags.csv", index=False, encoding="utf-8-sig")
    group.to_csv(OUT_DIR / "v103_group_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    rescued.to_csv(OUT_DIR / "v103_rescued_error_cases.csv", index=False, encoding="utf-8-sig")
    extra_clean.to_csv(OUT_DIR / "v103_extra_clean_review_cases.csv", index=False, encoding="utf-8-sig")
    remaining.to_csv(OUT_DIR / "v103_remaining_error_cases.csv", index=False, encoding="utf-8-sig")
    make_plots(nested, group)
    write_summary(nested, rescued, extra_clean, remaining, group)

    print("Wrote", OUT_DIR)
    print("Third rescued errors:")
    print(rescued.loc[rescued["domain"].eq("third_batch")][["original_case_id", "task_l6_label", "error_direction", "nested_rule_label", "prob_mean_core", "selected_unified_prob", "selected_dinov3_prob", "mm_prob_p75"]].to_string(index=False))
    print("\nThird remaining errors:")
    print(remaining.loc[remaining["domain"].eq("third_batch")][["original_case_id", "task_l6_label", "error_direction", "prob_mean_core", "selected_unified_prob", "selected_dinov3_prob", "mm_prob_p75"]].to_string(index=False))
    print("\nThird group summary:")
    print(group.loc[group["domain"].eq("third_batch") & ((group["rescued_error_n"] > 0) | (group["remaining_error_n"] > 0))].to_string(index=False))


if __name__ == "__main__":
    main()
