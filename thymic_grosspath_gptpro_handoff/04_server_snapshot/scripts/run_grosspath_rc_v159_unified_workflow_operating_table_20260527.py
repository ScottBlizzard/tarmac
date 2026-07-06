from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v159_unified_workflow_operating_table_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"
V147_SUMMARY = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_budget_summary.csv"
V156_SUMMARY = ROOT / "outputs" / "grosspath_rc_v156_severe_shift_gate_safety_boundary_20260527" / "v156_gate_scenario_summary.csv"
V158_COMPONENTS = ROOT / "outputs" / "grosspath_rc_v158_unlabeled_gate_threshold_stability_20260527" / "v158_gate_component_stability.csv"


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def pp(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):+.{digits}f} pp"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "tp": int(((y == 1) & (pred == 1)).sum()),
    }


def summarize_scorecard_cases(df: pd.DataFrame, scope: str, mask: pd.Series) -> dict[str, object]:
    sub = df.loc[mask].copy()
    review = as_bool(sub["v118_review_or_control"]).to_numpy()
    y = sub["label_idx"].astype(int).to_numpy()
    base_pred = sub["final_pred"].astype(int).to_numpy()
    prob = pd.to_numeric(sub["prob_mean_core"], errors="coerce").fillna(pd.to_numeric(sub["main_prob"], errors="coerce")).to_numpy(float)
    system_pred = base_pred.copy()
    system_pred[review] = y[review]
    base_wrong = base_pred != y
    captured = review & base_wrong
    clean_review = review & ~base_wrong
    m = metrics(y, system_pred, prob)
    row: dict[str, object] = {
        "evidence_line": "locked_high_safety_two_signal_scorecard",
        "workflow_type": "selective_review_main",
        "status": "main/high-safety candidate",
        "eval_domain": scope,
        "n": int(len(sub)),
        "auto_pass_n": int((~review).sum()),
        "auto_pass_rate": float((~review).mean()),
        "auto_correct_n": 0,
        "auto_correct_rate": 0.0,
        "review_or_reject_n": int(review.sum()),
        "review_or_reject_rate": float(review.mean()),
        "rescued_n": int(captured.sum()),
        "hurt_n": 0,
        "review_clean_n": int(clean_review.sum()),
        "base_error_n": int(base_wrong.sum()),
        "remaining_error_n": int((system_pred != y).sum()),
        "auc_source": "prob_mean_core",
        "note": "Review/control cases are assumed corrected; this is the current high-safety scorecard operating point.",
    }
    row.update(m)
    row.update(confusion_counts(y, system_pred))
    return row


def scorecard_rows() -> list[dict[str, object]]:
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    rows = []
    for scope in ["old_data", "third_batch", "strict_external"]:
        rows.append(summarize_scorecard_cases(df, scope, df["domain"].eq(scope)))
    rows.append(summarize_scorecard_cases(df, "all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])))
    return rows


def selective_review_rows() -> list[dict[str, object]]:
    s = pd.read_csv(V147_SUMMARY)
    s = s.loc[np.isclose(s["review_budget"].astype(float), 0.30)].copy()
    policies = [
        "baseline_low_conf_dev_selected",
        "dev_stable_router_all_domains",
        "shift_aware_image_directional_candidate",
        "shift_aware_concept_directional_candidate",
    ]
    rows = []
    for _, r in s.loc[s["policy"].isin(policies)].iterrows():
        row: dict[str, object] = {
            "evidence_line": str(r["policy"]),
            "workflow_type": "direction_aware_selective_review_30pct",
            "status": "router comparison / candidate" if str(r["policy"]) != "baseline_low_conf_dev_selected" else "baseline",
            "eval_domain": str(r["eval_domain"]).replace("all_three_domains", "all_domains"),
            "n": int(r["system_n"]),
            "auto_pass_n": int(r["auto_n"]),
            "auto_pass_rate": float(r["auto_rate"]),
            "auto_correct_n": 0,
            "auto_correct_rate": 0.0,
            "review_or_reject_n": int(r["review_n"]),
            "review_or_reject_rate": float(r["review_rate"]),
            "rescued_n": int(r["captured_errors"]),
            "hurt_n": 0,
            "review_clean_n": int(r["review_clean_n"]),
            "base_error_n": int(r["total_base_errors"]),
            "remaining_error_n": int(r["system_fn"] + r["system_fp"]),
            "accuracy": float(r["system_accuracy"]),
            "balanced_accuracy": float(r["system_balanced_accuracy"]),
            "f1": float(r["system_f1"]),
            "sensitivity_high": float(r["system_sensitivity_high"]),
            "specificity_low": float(r["system_specificity_low"]),
            "tn": int(r["system_tn"]),
            "fp": int(r["system_fp"]),
            "fn": int(r["system_fn"]),
            "tp": int(r["system_tp"]),
            "auc": float(r["system_auc"]),
            "auc_source": "base probability for selected router branch",
            "note": "30% review budget; reviewed cases are assumed corrected.",
        }
        rows.append(row)
    return rows


def autocorrect_rows() -> list[dict[str, object]]:
    s = pd.read_csv(V156_SUMMARY)
    s = s.loc[s["scenario"].eq("current_unlabeled_gate_only_severe_shift")].copy()
    rows = []
    for _, r in s.iterrows():
        n = int(r["n"])
        flip = int(r["flip_n"])
        row: dict[str, object] = {
            "evidence_line": "severe_shift_gated_concept_direction_autocorrect",
            "workflow_type": "auto_correction_candidate",
            "status": "candidate; severe-shift gated only",
            "eval_domain": str(r["eval_domain"]).replace("all_three_domains", "all_domains"),
            "n": n,
            "auto_pass_n": n - flip,
            "auto_pass_rate": float((n - flip) / max(1, n)),
            "auto_correct_n": flip,
            "auto_correct_rate": float(flip / max(1, n)),
            "review_or_reject_n": 0,
            "review_or_reject_rate": 0.0,
            "rescued_n": int(r["rescued_n"]),
            "hurt_n": int(r["hurt_n"]),
            "review_clean_n": 0,
            "base_error_n": int(r["base_errors"]),
            "remaining_error_n": int(r["final_errors"]),
            "accuracy": float(r["final_accuracy"]),
            "balanced_accuracy": float(r["final_balanced_accuracy"]),
            "f1": float(r["final_f1"]),
            "sensitivity_high": float(r["final_sensitivity_high"]),
            "specificity_low": float(r["final_specificity_low"]),
            "tn": int(r["final_tn"]),
            "fp": int(r["final_fp"]),
            "fn": int(r["final_fn"]),
            "tp": int(r["final_tp"]),
            "auc": float(r["final_auc"]),
            "auc_source": "prob_mean_core branch probability",
            "note": "No review is added here; this isolates the gated auto-correction module.",
        }
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = scorecard_rows() + selective_review_rows() + autocorrect_rows()
    table = pd.DataFrame(rows)
    ordered_cols = [
        "evidence_line",
        "workflow_type",
        "status",
        "eval_domain",
        "n",
        "auto_pass_n",
        "auto_pass_rate",
        "auto_correct_n",
        "auto_correct_rate",
        "review_or_reject_n",
        "review_or_reject_rate",
        "accuracy",
        "balanced_accuracy",
        "auc",
        "f1",
        "sensitivity_high",
        "specificity_low",
        "fn",
        "fp",
        "tn",
        "tp",
        "base_error_n",
        "remaining_error_n",
        "rescued_n",
        "hurt_n",
        "review_clean_n",
        "auc_source",
        "note",
    ]
    table = table[ordered_cols]
    table.to_csv(OUT_DIR / "v159_unified_workflow_operating_table.csv", index=False, encoding="utf-8-sig")

    focus = table.loc[
        table["eval_domain"].isin(["all_domains", "strict_external"])
        & table["evidence_line"].isin(
            [
                "locked_high_safety_two_signal_scorecard",
                "baseline_low_conf_dev_selected",
                "dev_stable_router_all_domains",
                "shift_aware_image_directional_candidate",
                "shift_aware_concept_directional_candidate",
                "severe_shift_gated_concept_direction_autocorrect",
            ]
        )
    ].copy()
    focus.to_csv(OUT_DIR / "v159_focus_operating_points.csv", index=False, encoding="utf-8-sig")

    gate = pd.read_csv(V158_COMPONENTS)
    clean_gate_count = int(gate["current_safety_status"].eq("clean_current_split").sum())
    gate_count = int(len(gate))
    scorecard_all = table.loc[
        table["evidence_line"].eq("locked_high_safety_two_signal_scorecard") & table["eval_domain"].eq("all_domains")
    ].iloc[0]
    autocorr_all = table.loc[
        table["evidence_line"].eq("severe_shift_gated_concept_direction_autocorrect") & table["eval_domain"].eq("all_domains")
    ].iloc[0]
    img_router_ext = table.loc[
        table["evidence_line"].eq("shift_aware_image_directional_candidate") & table["eval_domain"].eq("strict_external")
    ].iloc[0]
    lowconf_ext = table.loc[
        table["evidence_line"].eq("baseline_low_conf_dev_selected") & table["eval_domain"].eq("strict_external")
    ].iloc[0]

    md = [
        "# v159 Unified Workflow Operating Table",
        "",
        "## Main Readout",
        "",
        (
            f"- High-safety two-signal scorecard: all-domain BAcc {pct(scorecard_all['balanced_accuracy'])}, "
            f"auto-pass {pct(scorecard_all['auto_pass_rate'])}, review/reject {pct(scorecard_all['review_or_reject_rate'])}, "
            f"remaining FN={int(scorecard_all['fn'])}, FP={int(scorecard_all['fp'])}."
        ),
        (
            f"- 30% direction-aware image router on strict external: BAcc {pct(img_router_ext['balanced_accuracy'])} "
            f"vs low-conf {pct(lowconf_ext['balanced_accuracy'])}, difference {pp(img_router_ext['balanced_accuracy'] - lowconf_ext['balanced_accuracy'])}."
        ),
        (
            f"- Severe-shift gated concept auto-correction candidate: all-domain BAcc {pct(autocorr_all['balanced_accuracy'])}, "
            f"auto-correct {pct(autocorr_all['auto_correct_rate'])}, rescued {int(autocorr_all['rescued_n'])}, "
            f"hurt {int(autocorr_all['hurt_n'])}."
        ),
        (
            f"- Unlabeled gate stability from v158: {clean_gate_count}/{gate_count} gate variants cleanly trigger only the strict external batch in the current data."
        ),
        "",
        "## Interpretation",
        "",
        "These rows should not be collapsed into one headline number. The scorecard row is the current high-safety operating point, the 30% router rows test whether direction-aware review is better than confidence-only review, and the gated auto-correction row isolates the severe-shift correction module. This separation is important for a reviewer: it shows the framework has distinct safety mechanisms and also states which parts are locked versus candidate.",
    ]
    (OUT_DIR / "v159_unified_workflow_operating_table.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "rows": int(len(table)),
        "focus_rows": int(len(focus)),
        "scorecard_all_domain_bacc": float(scorecard_all["balanced_accuracy"]),
        "scorecard_all_domain_auto_pass_rate": float(scorecard_all["auto_pass_rate"]),
        "strict_external_image_router_delta_vs_lowconf": float(img_router_ext["balanced_accuracy"] - lowconf_ext["balanced_accuracy"]),
        "autocorrect_all_domain_bacc": float(autocorr_all["balanced_accuracy"]),
        "autocorrect_all_domain_auto_correct_rate": float(autocorr_all["auto_correct_rate"]),
        "clean_gate_variant_count": clean_gate_count,
        "gate_variant_count": gate_count,
    }
    (OUT_DIR / "v159_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v159] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
