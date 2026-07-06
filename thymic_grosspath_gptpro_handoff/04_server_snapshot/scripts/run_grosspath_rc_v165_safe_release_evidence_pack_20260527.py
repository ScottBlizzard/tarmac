from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v165_safe_release_evidence_pack_20260527"
V159 = ROOT / "outputs" / "grosspath_rc_v159_unified_workflow_operating_table_20260527" / "v159_unified_workflow_operating_table.csv"
V161_SUMMARY = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_summary.csv"
V161_CASES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_cases.csv"
V161_RULES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_selected_internal_zero_error_release_rules.csv"
V162_SELECTED = ROOT / "outputs" / "grosspath_rc_v162_frontier_after_safe_release_20260527" / "v162_constraint_selected_points.csv"
V164_FOCUS = ROOT / "outputs" / "grosspath_rc_v164_stable_safe_release_strategy_compare_20260527" / "v164_all_domain_strategy_focus.csv"
V164_RULES = ROOT / "outputs" / "grosspath_rc_v164_stable_safe_release_strategy_compare_20260527" / "v164_strategy_rules.csv"
V158_GATE = ROOT / "outputs" / "grosspath_rc_v158_unlabeled_gate_threshold_stability_20260527" / "v158_gate_component_stability.csv"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def table_row(
    row: pd.Series,
    evidence_line: str,
    source: str,
    status: str,
    review_col: str = "review_or_reject_rate",
    bacc_col: str = "balanced_accuracy",
) -> dict[str, object]:
    return {
        "evidence_line": evidence_line,
        "source": source,
        "status": status,
        "eval_domain": row.get("eval_domain", row.get("scope", "")),
        "n": int(row["n"]),
        "auto_pass_rate": float(row["auto_pass_rate"]),
        "review_or_reject_rate": float(row[review_col]),
        "auto_correct_rate": float(row.get("auto_correct_rate", 0.0)),
        "balanced_accuracy": float(row[bacc_col]),
        "accuracy": float(row.get("accuracy", row.get("all_acc", 0.0))),
        "f1": float(row.get("f1", row.get("all_f1", 0.0))),
        "fn": int(row["fn"]),
        "fp": int(row["fp"]),
        "note": row.get("note", ""),
    }


def build_main_table() -> pd.DataFrame:
    v159 = pd.read_csv(V159)
    v161 = pd.read_csv(V161_SUMMARY)
    rows = []
    selected_v159 = [
        ("baseline_low_conf_dev_selected", "30% low-confidence selective review baseline", "baseline"),
        ("dev_stable_router_all_domains", "30% dev-stable direction router", "router candidate"),
        ("shift_aware_image_directional_candidate", "30% image-direction router", "router candidate"),
        ("locked_high_safety_two_signal_scorecard", "v118 high-safety two-signal scorecard", "high-safety baseline"),
        ("severe_shift_gated_concept_direction_autocorrect", "severe-shift gated concept auto-correction", "auto-correction candidate"),
    ]
    for evidence, label, status in selected_v159:
        for domain in ["all_domains", "strict_external"]:
            sub = v159.loc[v159["evidence_line"].eq(evidence) & v159["eval_domain"].eq(domain)]
            if not sub.empty:
                rows.append(table_row(sub.iloc[0], label, "v159", status))
    for domain in ["all_domains", "old_data", "third_batch", "strict_external"]:
        sub = v161.loc[v161["workflow"].eq("v161_safe_release_scorecard") & v161["scope"].eq(domain)]
        if not sub.empty:
            rows.append(
                table_row(
                    sub.iloc[0],
                    "v161 safe-release high-safety scorecard",
                    "v161/v162/v164",
                    "current high-safety efficiency candidate",
                    review_col="review_rate",
                )
            )
    return pd.DataFrame(rows)


def remaining_errors() -> pd.DataFrame:
    df = pd.read_csv(V161_CASES, dtype={"case_id": str, "original_case_id": str})
    df["v161_final_review_or_reject"] = as_bool(df["v161_final_review_or_reject"])
    for col in ["label_idx", "final_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    auto = ~df["v161_final_review_or_reject"]
    err = auto & df["final_pred"].ne(df["label_idx"])
    cols = [
        "domain",
        "case_id",
        "original_case_id",
        "task_l6_label",
        "label_idx",
        "final_pred",
        "prob_mean_core",
        "wholecrop_prob",
        "main_prob",
        "robust_prob",
        "core_agree_count",
        "view_type_final",
        "image_name",
    ]
    return df.loc[err, [c for c in cols if c in df.columns]].copy()


def claim_map() -> pd.DataFrame:
    gate = pd.read_csv(V158_GATE)
    clean_gate = int(gate["current_safety_status"].eq("clean_current_split").sum())
    gate_n = int(len(gate))
    return pd.DataFrame(
        [
            {
                "claim": "The method should be framed as a risk-controlled workflow rather than a single classifier.",
                "evidence": "v159 separates selective review, severe-shift auto-correction, and high-safety scorecard operating points.",
                "status": "supported",
                "boundary": "Do not collapse candidate auto-correction and high-safety scorecard into one headline number.",
            },
            {
                "claim": "Safe-release reduces review burden while keeping the current high-safety error profile.",
                "evidence": "v161/v162: review/reject 79.97% -> 57.51%, auto-pass 20.03% -> 42.49%, all-domain BAcc 99.81%, FN=1, FP=0.",
                "status": "supported as current-data candidate",
                "boundary": "Rules are selected on old+third and externally observed; prospective validation is still required.",
            },
            {
                "claim": "Multi-domain internal constraints are necessary.",
                "evidence": "v164: old-only release gives third_batch 5 released errors, while old+third pooled/balanced zero-error has 0 released errors.",
                "status": "supported",
                "boundary": "Use old-only as negative ablation, not as deployable option.",
            },
            {
                "claim": "Unlabeled severe-shift gate is not a single fragile statistic.",
                "evidence": f"v158: {clean_gate}/{gate_n} gate variants cleanly trigger only strict external in the current data.",
                "status": "supported as current-batch evidence",
                "boundary": "Gate thresholds still need prospective calibration on future external batches.",
            },
            {
                "claim": "Direction-aware routing improves over confidence-only review under strict external shift.",
                "evidence": "v148/v159: strict external image-direction router BAcc 80.59% vs low-confidence 78.10% at ~30% review.",
                "status": "trend-level support",
                "boundary": "Point estimate improves but prior matched CI crossed zero; report cautiously.",
            },
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_table = build_main_table()
    main_table.to_csv(OUT_DIR / "v165_main_operating_results.csv", index=False, encoding="utf-8-sig")

    release_rules = pd.read_csv(V161_RULES)
    release_rules.to_csv(OUT_DIR / "v165_safe_release_rules.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(V164_FOCUS).to_csv(OUT_DIR / "v165_safe_release_ablation.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(V162_SELECTED).to_csv(OUT_DIR / "v165_frontier_constraint_selection.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(V164_RULES).to_csv(OUT_DIR / "v165_safe_release_strategy_rules.csv", index=False, encoding="utf-8-sig")

    rem = remaining_errors()
    rem.to_csv(OUT_DIR / "v165_remaining_auto_errors_after_safe_release.csv", index=False, encoding="utf-8-sig")
    claims = claim_map()
    claims.to_csv(OUT_DIR / "v165_claim_evidence_map.csv", index=False, encoding="utf-8-sig")

    all_v161 = main_table.loc[
        main_table["evidence_line"].eq("v161 safe-release high-safety scorecard")
        & main_table["eval_domain"].eq("all_domains")
    ].iloc[0]
    strict_v161 = main_table.loc[
        main_table["evidence_line"].eq("v161 safe-release high-safety scorecard")
        & main_table["eval_domain"].eq("strict_external")
    ].iloc[0]
    oldonly = pd.read_csv(V164_FOCUS).loc[lambda x: x["strategy"].eq("old_only_selected")].iloc[0]

    md = [
        "# v165 Safe-release Evidence Pack",
        "",
        "## Current Main Result",
        "",
        (
            f"The current high-safety efficiency candidate is v161 safe-release: all-domain BAcc "
            f"{pct(all_v161['balanced_accuracy'])}, review/reject {pct(all_v161['review_or_reject_rate'])}, "
            f"auto-pass {pct(all_v161['auto_pass_rate'])}, FN={int(all_v161['fn'])}, FP={int(all_v161['fp'])}."
        ),
        (
            f"On strict external, the same workflow gives BAcc {pct(strict_v161['balanced_accuracy'])}, "
            f"review/reject {pct(strict_v161['review_or_reject_rate'])}, FN={int(strict_v161['fn'])}, FP={int(strict_v161['fp'])}."
        ),
        "",
        "## Key Ablation",
        "",
        (
            f"Old-only safe-release is the negative control: it lowers all-domain review to "
            f"{pct(oldonly['review_rate'])}, but releases {int(oldonly['released_error_n'])} errors and leaves "
            f"FN={int(oldonly['fn'])}, FP={int(oldonly['fp'])}. This supports the multi-domain constraint design."
        ),
        "",
        "## Remaining Error",
        "",
        (
            "After v161, the remaining automatic error table contains "
            f"{len(rem)} case(s). This should be kept in the Results boundary rather than hidden."
        ),
        "",
        "## Files",
        "",
        "- v165_main_operating_results.csv",
        "- v165_safe_release_rules.csv",
        "- v165_safe_release_ablation.csv",
        "- v165_frontier_constraint_selection.csv",
        "- v165_remaining_auto_errors_after_safe_release.csv",
        "- v165_claim_evidence_map.csv",
    ]
    (OUT_DIR / "v165_safe_release_evidence_pack.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "main_result_bacc": float(all_v161["balanced_accuracy"]),
        "main_result_review_rate": float(all_v161["review_or_reject_rate"]),
        "main_result_auto_pass_rate": float(all_v161["auto_pass_rate"]),
        "main_result_fn": int(all_v161["fn"]),
        "main_result_fp": int(all_v161["fp"]),
        "strict_external_bacc": float(strict_v161["balanced_accuracy"]),
        "remaining_auto_errors": int(len(rem)),
        "claim_count": int(len(claims)),
    }
    (OUT_DIR / "v165_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v165] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
