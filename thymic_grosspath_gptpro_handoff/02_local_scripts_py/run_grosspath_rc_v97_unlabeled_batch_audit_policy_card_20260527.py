from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V77 = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527" / "v77_unlabeled_batch_shift_audit.csv"
V95 = ROOT / "outputs" / "grosspath_rc_v95_selective_risk_confidence_bounds_20260527" / "v95_selective_risk_confidence_bounds.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v97_unlabeled_batch_audit_policy_card_20260527"
FIG_DIR = OUT_DIR / "figures"

SELECTED_POLICY = "adaptive_v50_to_v79_light"
SELECTED_LABEL = "Batch-adaptive main"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def component_for_shift(shift_category: str) -> str:
    if shift_category == "within_internal_shift":
        return "standard v50 component"
    if shift_category == "moderate_shift":
        return "safety-enhanced v79-light component"
    return "severe-shift v79-light component"


def action_for_shift(shift_category: str) -> str:
    if shift_category == "within_internal_shift":
        return "auto workflow allowed with standard review threshold"
    if shift_category == "moderate_shift":
        return "raise review threshold before diagnosis"
    return "raise review threshold and flag acquisition shift"


def build_policy_card(audit: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focus = audit.loc[audit["target"].isin(["old_data", "third_batch", "strict_external"])].copy()
    for _, row in focus.iterrows():
        target = row["target"]
        m = metrics.loc[(metrics["scope"] == target) & (metrics["policy"] == SELECTED_POLICY)].iloc[0]
        rows.append(
            {
                "target_domain": target,
                "reference_domain": row["reference"],
                "n_target": int(row["n_target"]),
                "domain_auc_cv": float(row["domain_auc_cv"]),
                "mean_abs_median_shift_iqr": float(row["mean_abs_median_shift_iqr"]),
                "mean_outside_ref_05_95_rate": float(row["mean_outside_ref_05_95_rate"]),
                "quality_proxy_mean": float(row["quality_proxy_mean"]),
                "quality_proxy_p90": float(row["quality_proxy_p90"]),
                "batch_shift_index": float(row["batch_shift_index"]),
                "shift_category": row["shift_category"],
                "selected_workflow": SELECTED_LABEL,
                "selected_component": component_for_shift(row["shift_category"]),
                "deployment_action": action_for_shift(row["shift_category"]),
                "control_rate": float(m["control_rate"]),
                "auto_rate": float(m["auto_rate"]),
                "auto_pass_error_risk": float(m["auto_pass_error_risk"]),
                "auto_pass_error_risk_wilson_high": float(m["auto_pass_error_risk_wilson_high"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "sensitivity": float(m["sensitivity"]),
                "specificity": float(m["specificity"]),
                "fn": int(m["fn"]),
                "fp": int(m["fp"]),
                "remaining_error_n": int(m["remaining_error_n"]),
            }
        )
    order = {"old_data": 0, "third_batch": 1, "strict_external": 2}
    out = pd.DataFrame(rows)
    out["order"] = out["target_domain"].map(order)
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def format_card(card: pd.DataFrame) -> pd.DataFrame:
    out = card.copy()
    for col in [
        "domain_auc_cv",
        "mean_outside_ref_05_95_rate",
        "quality_proxy_mean",
        "quality_proxy_p90",
        "control_rate",
        "auto_rate",
        "auto_pass_error_risk",
        "auto_pass_error_risk_wilson_high",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]:
        out[col] = out[col].map(pct)
    for col in ["mean_abs_median_shift_iqr", "batch_shift_index"]:
        out[col] = out[col].map(lambda x: f"{x:.3f}")
    return out


def make_plot(card: pd.DataFrame, audit: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    internal = audit.loc[audit["audit_name"].str.startswith("pseudo_")]
    auc_max = internal["domain_auc_cv"].max()
    q_max = internal["quality_proxy_mean"].max()

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.axvspan(0.5, auc_max, color="#E8F4EA", alpha=0.8, label="internal shift envelope")
    ax.axhspan(0, q_max, color="#E8F4EA", alpha=0.35)
    colors = {
        "within_internal_shift": "#2E7D32",
        "moderate_shift": "#F9A825",
        "severe_shift": "#C62828",
    }
    for _, row in card.iterrows():
        ax.scatter(
            row["domain_auc_cv"],
            row["quality_proxy_mean"],
            s=180 + 75 * row["batch_shift_index"],
            color=colors.get(row["shift_category"], "#666666"),
            edgecolor="white",
            linewidth=1.2,
        )
        ax.text(
            row["domain_auc_cv"] + 0.006,
            row["quality_proxy_mean"] + 0.004,
            f"{row['target_domain']}\n{row['selected_component'].split()[0]}",
            fontsize=9,
        )
    ax.set_xlabel("Unlabeled domain separability AUC")
    ax.set_ylabel("Mean acquisition/quality proxy risk")
    ax.set_title("Unlabeled batch audit and workflow selection")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v97_batch_audit_policy_card.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v97_batch_audit_policy_card.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(card: pd.DataFrame) -> None:
    strict = card.loc[card["target_domain"] == "strict_external"].iloc[0]
    old = card.loc[card["target_domain"] == "old_data"].iloc[0]
    third = card.loc[card["target_domain"] == "third_batch"].iloc[0]
    lines = [
        "# v97 Unlabeled Batch Audit to Workflow Card",
        "",
        "## Key Findings",
        "",
        f"- Old data and third batch remain inside the internal shift envelope, so the batch-adaptive workflow keeps the standard v50 component. Their observed BAcc values are {pct(old['balanced_accuracy'])} and {pct(third['balanced_accuracy'])}.",
        f"- Strict external is detected as severe shift without using labels: domain AUC {pct(strict['domain_auc_cv'])}, mean quality proxy {pct(strict['quality_proxy_mean'])}, batch shift index {strict['batch_shift_index']:.3f}.",
        f"- The severe-shift route selects the v79-light safety component. Strict external observed BAcc is {pct(strict['balanced_accuracy'])}, FN={int(strict['fn'])}, FP={int(strict['fp'])}, control rate {pct(strict['control_rate'])}.",
        "",
        "## Paper Use",
        "",
        "This table is suitable for the Methods/Results bridge: first audit an incoming hospital batch without labels, then select the operating point, and only afterwards report labeled performance as validation. It supports the claim that the framework is batch-adaptive rather than a fixed external-set-tuned classifier.",
        "",
    ]
    (OUT_DIR / "v97_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(V77)
    metrics = pd.read_csv(V95)
    card = build_policy_card(audit, metrics)
    card_fmt = format_card(card)

    card.to_csv(OUT_DIR / "v97_batch_audit_to_policy_card.csv", index=False, encoding="utf-8-sig")
    card_fmt.to_csv(OUT_DIR / "v97_batch_audit_to_policy_card_formatted.csv", index=False, encoding="utf-8-sig")
    make_plot(card, audit)
    write_summary(card)

    print("Wrote", OUT_DIR)
    print(card_fmt.to_string(index=False))


if __name__ == "__main__":
    main()
