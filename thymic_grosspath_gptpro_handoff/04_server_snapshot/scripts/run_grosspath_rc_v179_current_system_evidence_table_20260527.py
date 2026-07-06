from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v179_current_system_evidence_table_20260527"
V159 = ROOT / "outputs" / "grosspath_rc_v159_unified_workflow_operating_table_20260527" / "v159_unified_workflow_operating_table.csv"
V178 = ROOT / "outputs" / "grosspath_rc_v178_safe_release_union_with_image_agreement_20260527" / "v178_selected_workflow_summary.csv"
V177 = ROOT / "outputs" / "grosspath_rc_v177_autocorrection_feasibility_matrix_20260527" / "v177_autocorrection_feasibility_matrix.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def row_from_v159(df: pd.DataFrame, evidence: str, scope: str, label: str, tier: str, role: str) -> dict[str, object]:
    r = df.loc[df["evidence_line"].eq(evidence) & df["eval_domain"].eq(scope)].iloc[0]
    return {
        "module": label,
        "tier": tier,
        "role": role,
        "scope": scope,
        "auto_decision_rate": float(r["auto_pass_rate"]) + float(r.get("auto_correct_rate", 0.0)),
        "remaining_review_or_reject_rate": float(r["review_or_reject_rate"]),
        "auto_correct_or_release_rate": float(r.get("auto_correct_rate", 0.0)),
        "balanced_accuracy": float(r["balanced_accuracy"]),
        "accuracy": float(r["accuracy"]),
        "f1": float(r["f1"]),
        "auc": float(r["auc"]),
        "fn": int(r["fn"]),
        "fp": int(r["fp"]),
        "auto_action_error_n": "",
        "source": "v159",
        "writing_status": tier,
    }


def row_from_v178(df: pd.DataFrame, workflow: str, scope: str, label: str, tier: str, role: str) -> dict[str, object]:
    r = df.loc[df["workflow"].eq(workflow) & df["scope"].eq(scope)].iloc[0]
    return {
        "module": label,
        "tier": tier,
        "role": role,
        "scope": scope,
        "auto_decision_rate": 1.0 - float(r["final_review_rate"]),
        "remaining_review_or_reject_rate": float(r["final_review_rate"]),
        "auto_correct_or_release_rate": float(r["auto_release_rate"]),
        "balanced_accuracy": float(r["balanced_accuracy"]),
        "accuracy": float(r["accuracy"]),
        "f1": float(r["f1"]),
        "auc": float(r["auc"]),
        "fn": int(r["fn"]),
        "fp": int(r["fp"]),
        "auto_action_error_n": int(r["released_error_n"]),
        "source": "v178",
        "writing_status": tier,
    }


def row_from_v177(df: pd.DataFrame, module: str, scope: str, label: str, tier: str, role: str) -> dict[str, object]:
    r = df.loc[df["module"].eq(module)].iloc[0]
    return {
        "module": label,
        "tier": tier,
        "role": role,
        "scope": scope,
        "auto_decision_rate": 1.0 - float(r["remaining_review_rate"]),
        "remaining_review_or_reject_rate": float(r["remaining_review_rate"]),
        "auto_correct_or_release_rate": float(r["auto_action_rate"]),
        "balanced_accuracy": float(r["balanced_accuracy"]),
        "accuracy": "",
        "f1": "",
        "auc": "",
        "fn": int(r["fn"]),
        "fp": int(r["fp"]),
        "auto_action_error_n": "" if pd.isna(r["auto_error_n"]) else int(r["auto_error_n"]),
        "source": "v177",
        "writing_status": tier,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v159 = pd.read_csv(V159)
    v178 = pd.read_csv(V178)
    v177 = pd.read_csv(V177)
    v178_workflow = [w for w in v178["workflow"].unique() if w != "v161_safe_release"][0]

    rows = []
    for scope in ["old_data", "third_batch", "strict_external", "all_domains"]:
        rows.append(
            row_from_v159(
                v159,
                "locked_high_safety_two_signal_scorecard",
                scope,
                "v118 locked high-safety two-signal scorecard",
                "Tier 1 supported",
                "primary high-safety workflow",
            )
        )
        rows.append(
            row_from_v178(
                v178,
                "v161_safe_release",
                scope,
                "v161 safe-release scorecard",
                "Tier 2 efficiency candidate",
                "reduces review burden with current zero released errors",
            )
        )
        rows.append(
            row_from_v178(
                v178,
                v178_workflow,
                scope,
                "v178 safe-release + image agreement",
                "Tier 2 conservative refinement",
                "adds image agreement release on top of v161",
            )
        )
    for evidence, label, role in [
        ("baseline_low_conf_dev_selected", "30% low-confidence selective review", "selective prediction baseline"),
        ("shift_aware_image_directional_candidate", "30% direction-aware image router", "direction-aware router trend"),
        ("severe_shift_gated_concept_direction_autocorrect", "severe-shift gated concept correction", "conditional correction candidate"),
    ]:
        for scope in ["all_domains", "strict_external"]:
            rows.append(
                row_from_v159(
                    v159,
                    evidence,
                    scope,
                    label,
                    "Comparator / candidate",
                    role,
                )
            )
    for module, label, tier, role in [
        (
            "v173 image-only review corrector",
            "v173 aggressive image-only review corrector",
            "Candidate with safety boundary",
            "large review reduction but has current auto errors",
        ),
        (
            "v174 disagreement flip",
            "v174 disagreement flip",
            "Negative evidence",
            "automatic flipping is not currently reliable",
        ),
        (
            "v175 error-enriched flip-risk",
            "v175 error-enriched flip-risk",
            "Negative evidence",
            "historical candidate errors do not transfer to current residual errors",
        ),
    ]:
        rows.append(row_from_v177(v177, module, "all_domains", label, tier, role))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "v179_current_system_operating_table.csv", index=False, encoding="utf-8-sig")

    claims = pd.DataFrame(
        [
            {
                "claim": "Risk-controlled workflow is the main contribution, not a single raw classifier.",
                "supported_by": "v118/v161/v178 plus v159 router baselines",
                "status": "strong",
                "paper_boundary": "Frame as selective diagnosis / risk-controlled deployment.",
            },
            {
                "claim": "Two-signal safety scorecard provides the strongest current high-safety operating point.",
                "supported_by": "v118 all-domain BAcc 99.81%, FN=1, FP=0, review 79.97%",
                "status": "strong",
                "paper_boundary": "High review burden should be disclosed.",
            },
            {
                "claim": "Safe-release can reduce review burden without increasing current automatic errors.",
                "supported_by": "v161 review 57.51%, v178 review 56.08%, both released_error=0",
                "status": "candidate",
                "paper_boundary": "Needs prospective or stronger nested validation before deployment wording.",
            },
            {
                "claim": "Direction-aware router is better than confidence-only under strict external shift.",
                "supported_by": "v159 strict external 30% review BAcc 80.59% vs low-confidence 78.10%",
                "status": "trend",
                "paper_boundary": "Useful as mechanism evidence, not headline performance.",
            },
            {
                "claim": "Automatic flipping/correction is mature enough to replace review.",
                "supported_by": "v174/v175 negative evidence",
                "status": "not_supported",
                "paper_boundary": "Write as future work requiring more true residual-error samples or doctor labels.",
            },
        ]
    )
    claims.to_csv(OUT_DIR / "v179_claim_boundary_table.csv", index=False, encoding="utf-8-sig")

    all_rows = table.loc[table["scope"].eq("all_domains")].copy()
    v118 = all_rows.loc[all_rows["module"].eq("v118 locked high-safety two-signal scorecard")].iloc[0]
    v178r = all_rows.loc[all_rows["module"].eq("v178 safe-release + image agreement")].iloc[0]
    router = table.loc[
        table["module"].eq("30% direction-aware image router") & table["scope"].eq("strict_external")
    ].iloc[0]
    lowconf = table.loc[
        table["module"].eq("30% low-confidence selective review") & table["scope"].eq("strict_external")
    ].iloc[0]
    md = [
        "# v179 Current System Evidence Table",
        "",
        "## Current Recommended Main Line",
        "",
        (
            f"- Primary high-safety workflow: v118 scorecard, all-domain BAcc {pct(v118['balanced_accuracy'])}, "
            f"remaining review {pct(v118['remaining_review_or_reject_rate'])}, FN={int(v118['fn'])}, FP={int(v118['fp'])}."
        ),
        (
            f"- Conservative efficiency refinement: v178 union release, all-domain BAcc {pct(v178r['balanced_accuracy'])}, "
            f"remaining review {pct(v178r['remaining_review_or_reject_rate'])}, released errors {v178r['auto_action_error_n']}, "
            f"FN={int(v178r['fn'])}, FP={int(v178r['fp'])}."
        ),
        (
            f"- Direction-aware router trend: strict external BAcc {pct(router['balanced_accuracy'])} versus "
            f"{pct(lowconf['balanced_accuracy'])} for low-confidence selective review at ~30% review."
        ),
        "",
        "## Writing Boundary",
        "",
        "- Strong: risk-controlled workflow, two-signal safety scorecard, safe-release efficiency frontier.",
        "- Candidate: v178 image agreement refinement and v173 aggressive review reduction.",
        "- Negative: direct automatic flipping and error-enriched flip-risk are not reliable enough.",
    ]
    (OUT_DIR / "v179_current_system_evidence_table.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "row_count": int(len(table)),
        "claim_count": int(len(claims)),
        "v118_all_domain_bacc": float(v118["balanced_accuracy"]),
        "v118_review_rate": float(v118["remaining_review_or_reject_rate"]),
        "v178_all_domain_bacc": float(v178r["balanced_accuracy"]),
        "v178_review_rate": float(v178r["remaining_review_or_reject_rate"]),
        "strict_external_direction_router_bacc": float(router["balanced_accuracy"]),
        "strict_external_low_conf_bacc": float(lowconf["balanced_accuracy"]),
    }
    (OUT_DIR / "v179_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v179] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
