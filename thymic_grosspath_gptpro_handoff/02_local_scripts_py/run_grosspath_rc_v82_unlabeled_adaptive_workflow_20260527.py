from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_ROUTES = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527" / "v80_tiered_lowrisk_guard_case_routes.csv"
AUDIT = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527" / "v77_unlabeled_batch_shift_audit.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v82_unlabeled_adaptive_workflow_20260527"
FIG_DIR = OUT_DIR / "figures"


POLICY_LABELS = {
    "v50_main": "Fixed v50",
    "v75_quality_lowconf": "Fixed v75",
    "v79_light_lowrisk_guard": "Fixed v79 light",
    "v79_strict_lowrisk_guard": "Fixed v79 strict",
    "adaptive_v75_on_shift": "Adaptive: v50 internal, v75 severe-shift",
    "adaptive_light_on_shift": "Adaptive: v50 internal, v79-light severe-shift",
    "adaptive_strict_on_shift": "Adaptive: v50 internal, v79-strict severe-shift",
}


def metrics(df: pd.DataFrame) -> dict[str, object]:
    y = df["label_idx"].to_numpy(int)
    pred = df["final_pred"].to_numpy(int)
    control = df["review_or_control"].to_numpy(int).astype(bool)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n": int(len(df)),
        "control_n": int(control.sum()),
        "control_rate": float(control.mean()),
        "auto_n": int((~control).sum()),
        "accuracy": float((tp + tn) / len(df)),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "remaining_error_n": int((pred != y).sum()),
    }


def audit_decision_table(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    domain_to_audit = {
        "old_data": "pseudo_third_to_old",
        "third_batch": "pseudo_old_to_third",
        "strict_external": "strict_external_vs_dev",
    }
    for domain, audit_name in domain_to_audit.items():
        row = audit.loc[audit["audit_name"].eq(audit_name)].iloc[0].to_dict()
        shift = str(row["shift_category"])
        rows.append(
            {
                "domain": domain,
                "audit_name": audit_name,
                "shift_category": shift,
                "domain_auc_cv": row["domain_auc_cv"],
                "quality_proxy_mean": row["quality_proxy_mean"],
                "batch_shift_index": row["batch_shift_index"],
                "selected_policy_adaptive_v75": "v75_quality_lowconf" if shift != "within_internal_shift" else "v50_main",
                "selected_policy_adaptive_light": "v79_light_lowrisk_guard" if shift != "within_internal_shift" else "v50_main",
                "selected_policy_adaptive_strict": "v79_strict_lowrisk_guard" if shift != "within_internal_shift" else "v50_main",
            }
        )
    return pd.DataFrame(rows)


def build_adaptive_cases(routes: pd.DataFrame, decisions: pd.DataFrame, adaptive_name: str, selected_col: str) -> pd.DataFrame:
    parts = []
    for _, row in decisions.iterrows():
        domain = row["domain"]
        selected = row[selected_col]
        sub = routes.loc[routes["domain"].eq(domain) & routes["policy"].eq(selected)].copy()
        sub["source_policy"] = selected
        sub["policy"] = adaptive_name
        sub["shift_category"] = row["shift_category"]
        sub["batch_shift_index"] = row["batch_shift_index"]
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def summarize_all(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, policy), sub in case_df.groupby(["scope", "policy"], sort=False):
        row = {"scope": scope, "policy": policy}
        row.update(metrics(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot = summary.loc[summary["scope"].eq("all_domains")].copy()
    order = [
        "v50_main",
        "v75_quality_lowconf",
        "v79_light_lowrisk_guard",
        "v79_strict_lowrisk_guard",
        "adaptive_v75_on_shift",
        "adaptive_light_on_shift",
        "adaptive_strict_on_shift",
    ]
    plot["order"] = plot["policy"].map({p: i for i, p in enumerate(order)})
    plot = plot.sort_values("order")
    colors = ["#7f8c8d", "#117a65", "#d68910", "#c0392b", "#76d7c4", "#f5b041", "#ec7063"]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for color, (_, row) in zip(colors, plot.iterrows()):
        ax.scatter(row["control_rate"] * 100, row["balanced_accuracy"] * 100, s=110, color=color, edgecolor="white")
        ax.text(row["control_rate"] * 100 + 0.2, row["balanced_accuracy"] * 100, POLICY_LABELS.get(row["policy"], row["policy"]), fontsize=8)
    ax.set_xlabel("Overall control rate (%)")
    ax.set_ylabel("Overall balanced accuracy (%)")
    ax.set_title("Fixed vs unlabeled batch-adaptive workflows")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v82_fixed_vs_adaptive_workflows.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v82_fixed_vs_adaptive_workflows.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(CASE_ROUTES)
    audit = pd.read_csv(AUDIT)
    decisions = audit_decision_table(audit)

    fixed = routes.copy()
    fixed["source_policy"] = fixed["policy"]
    fixed["shift_category"] = ""
    fixed["batch_shift_index"] = np.nan
    adaptive = [
        build_adaptive_cases(routes, decisions, "adaptive_v75_on_shift", "selected_policy_adaptive_v75"),
        build_adaptive_cases(routes, decisions, "adaptive_light_on_shift", "selected_policy_adaptive_light"),
        build_adaptive_cases(routes, decisions, "adaptive_strict_on_shift", "selected_policy_adaptive_strict"),
    ]
    all_cases = pd.concat([fixed] + adaptive, ignore_index=True)
    all_cases["scope"] = all_cases["domain"]
    pooled = all_cases.copy()
    pooled["scope"] = "all_domains"
    eval_cases = pd.concat([all_cases, pooled], ignore_index=True)
    summary = summarize_all(eval_cases)

    decisions.to_csv(OUT_DIR / "v82_unlabeled_batch_policy_decisions.csv", index=False, encoding="utf-8-sig")
    eval_cases.to_csv(OUT_DIR / "v82_fixed_and_adaptive_case_routes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v82_fixed_and_adaptive_summary.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)

    print("Unlabeled batch decisions:")
    print(decisions.to_string(index=False))
    print("\nSummary:")
    show = summary.loc[
        summary["scope"].isin(["all_domains", "strict_external"]),
        ["scope", "policy", "control_rate", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp", "remaining_error_n"],
    ].copy()
    show["policy_label"] = show["policy"].map(POLICY_LABELS).fillna(show["policy"])
    print(show[["scope", "policy_label", "control_rate", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp", "remaining_error_n"]].to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
