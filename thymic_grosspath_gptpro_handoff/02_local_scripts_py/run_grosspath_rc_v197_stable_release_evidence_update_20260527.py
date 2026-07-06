from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v197_stable_release_evidence_update_20260527"
V192_OPERATING = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527" / "v192_final_framework_operating_table.csv"
V195_SUMMARY = ROOT / "outputs" / "grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527" / "v195_selected_candidate_summary.csv"
V195_REPORT = ROOT / "outputs" / "grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527" / "v195_run_report.json"
V196_SUMMARY = ROOT / "outputs" / "grosspath_rc_v196_nested_adaptive_release_validation_20260527" / "v196_nested_release_summary.csv"
V196_SELECTION = ROOT / "outputs" / "grosspath_rc_v196_nested_adaptive_release_validation_20260527" / "v196_nested_fold_selected_rules.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def get_operating(df: pd.DataFrame, module: str, scope: str = "all_domains") -> pd.Series:
    return df.loc[df["module"].eq(module) & df["scope"].eq(scope)].iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    operating = pd.read_csv(V192_OPERATING)
    v195 = pd.read_csv(V195_SUMMARY)
    v196 = pd.read_csv(V196_SUMMARY)
    selections = pd.read_csv(V196_SELECTION)

    rows: list[dict[str, object]] = []
    for module, tier, interpretation in [
        (
            "v185 unlabeled shift-adaptive workflow",
            "previous recommended framework",
            "Risk-controlled baseline before second-stage release compression.",
        ),
        (
            "v182 fixed stable efficiency workflow",
            "previous efficiency workflow",
            "Stable image-agreement release before v195 high-confidence release.",
        ),
        (
            "v118 fixed high-safety fallback",
            "high-safety fallback",
            "Severe-shift fallback with higher review burden.",
        ),
    ]:
        r = get_operating(operating, module)
        rows.append(
            {
                "workflow": module,
                "tier": tier,
                "selection_protocol": "locked from prior evidence",
                "scope": "all_domains",
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "remaining_review_or_reject_rate": float(r["remaining_review_or_reject_rate"]),
                "auto_action_n": "",
                "action_error_n": "",
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "interpretation": interpretation,
            }
        )

    for scope, scope_label in [
        ("all_domains", "all_domains"),
        ("strict_external", "strict_external"),
    ]:
        r = v195.loc[v195["scope"].eq(scope)].iloc[0]
        rows.append(
            {
                "workflow": "v195 stable fixed high-confidence agreement release",
                "tier": "new candidate efficiency workflow",
                "selection_protocol": "fixed cross-fold stable zero-action-error rule",
                "scope": scope_label,
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "remaining_review_or_reject_rate": float(r["remaining_review_rate"]),
                "auto_action_n": int(r["auto_action_n"]),
                "action_error_n": int(r["action_error_n"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "interpretation": "Strong review compression with zero observed auto-action error; this is safe release, not true label flipping.",
            }
        )

    for scope, scope_label in [
        ("all_domains_nested_plus_locked_external", "all_domains"),
        ("strict_external_locked", "strict_external"),
    ]:
        r = v196.loc[v196["scope"].eq(scope)].iloc[0]
        rows.append(
            {
                "workflow": "v196 per-fold greedy agreement release audit",
                "tier": "negative stability audit",
                "selection_protocol": "each fold maximizes zero-train-error release",
                "scope": scope_label,
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "remaining_review_or_reject_rate": float(r["remaining_review_rate"]),
                "auto_action_n": int(r["auto_action_n"]),
                "action_error_n": int(r["action_error_n"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "interpretation": "Lower review burden but releases held-out errors; rejects greedy local rule selection.",
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "v197_stable_release_evidence_table.csv", index=False, encoding="utf-8-sig")

    v195_all = table.loc[
        table["workflow"].eq("v195 stable fixed high-confidence agreement release") & table["scope"].eq("all_domains")
    ].iloc[0]
    v196_all = table.loc[
        table["workflow"].eq("v196 per-fold greedy agreement release audit") & table["scope"].eq("all_domains")
    ].iloc[0]
    v185_all = table.loc[
        table["workflow"].eq("v185 unlabeled shift-adaptive workflow") & table["scope"].eq("all_domains")
    ].iloc[0]
    bad_folds = selections.loc[selections["heldout_action_error_n"].astype(int).gt(0)].copy()
    md = [
        "# v197 Stable Release Evidence Update",
        "",
        "## Main Message",
        "",
        (
            "The second-stage module should be written as a stable high-confidence agreement-release module, "
            "not as an automatic label-flipping corrector."
        ),
        "",
        "## Key Operating Points",
        "",
        (
            f"- v185 adaptive baseline: BAcc {pct(v185_all['balanced_accuracy'])}, "
            f"review/reject {pct(v185_all['remaining_review_or_reject_rate'])}."
        ),
        (
            f"- v195 stable fixed release: BAcc {pct(v195_all['balanced_accuracy'])}, "
            f"review/reject {pct(v195_all['remaining_review_or_reject_rate'])}, "
            f"auto actions {int(v195_all['auto_action_n'])}, action errors {int(v195_all['action_error_n'])}."
        ),
        (
            f"- v196 per-fold greedy audit: BAcc {pct(v196_all['balanced_accuracy'])}, "
            f"review/reject {pct(v196_all['remaining_review_or_reject_rate'])}, "
            f"auto actions {int(v196_all['auto_action_n'])}, action errors {int(v196_all['action_error_n'])}."
        ),
        "",
        "## Stability Interpretation",
        "",
        (
            "v195 reduces the all-domain review/reject rate by "
            f"{100 * (float(v185_all['remaining_review_or_reject_rate']) - float(v195_all['remaining_review_or_reject_rate'])):.2f} percentage points "
            "without introducing observed action errors. v196 shows why the selection must remain fixed and conservative: "
            "local fold-wise maximization releases held-out errors even when each training complement is error-free."
        ),
        "",
        "## Greedy Audit Errors",
        "",
    ]
    if bad_folds.empty:
        md.append("- No greedy held-out action errors were observed.")
    else:
        for _, r in bad_folds.iterrows():
            md.append(
                f"- Fold {int(r['heldout_fold'])}: candidate `{r['candidate_id']}` released "
                f"{int(r['heldout_action_error_n'])} held-out errors at threshold {float(r['threshold']):.3f}."
            )
    md += [
        "",
        "## Paper Claim Boundary",
        "",
        "- Supported: stable high-confidence agreement release compresses the review/reject pool.",
        "- Supported: greedy per-fold release maximization is unsafe and should be rejected.",
        "- Not supported: autonomous automatic correction by flipping low/high-risk predictions.",
    ]
    (OUT_DIR / "v197_stable_release_evidence_update.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "v195_all_domain_bacc": float(v195_all["balanced_accuracy"]),
        "v195_all_domain_review_rate": float(v195_all["remaining_review_or_reject_rate"]),
        "v195_action_error_n": int(v195_all["action_error_n"]),
        "v196_all_domain_bacc": float(v196_all["balanced_accuracy"]),
        "v196_all_domain_review_rate": float(v196_all["remaining_review_or_reject_rate"]),
        "v196_action_error_n": int(v196_all["action_error_n"]),
        "bad_greedy_fold_count": int(len(bad_folds)),
    }
    (OUT_DIR / "v197_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v197] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
