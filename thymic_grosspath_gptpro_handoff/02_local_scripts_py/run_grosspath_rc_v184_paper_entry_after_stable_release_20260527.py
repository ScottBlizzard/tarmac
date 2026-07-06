from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v184_paper_entry_after_stable_release_20260527"
V183_TABLE = ROOT / "outputs" / "grosspath_rc_v183_revised_system_after_stable_release_20260527" / "v183_revised_system_operating_table.csv"
V183_CLAIMS = ROOT / "outputs" / "grosspath_rc_v183_revised_system_after_stable_release_20260527" / "v183_revised_claim_table.csv"
V158_GATE = ROOT / "outputs" / "grosspath_rc_v158_unlabeled_gate_threshold_stability_20260527" / "v158_gate_component_stability.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(V183_TABLE)
    claims = pd.read_csv(V183_CLAIMS)
    gate = pd.read_csv(V158_GATE)

    priority = {
        "v118 locked high-safety two-signal scorecard": 1,
        "v161 safe-release scorecard": 2,
        "v182 stable fixed image-agreement release": 3,
        "v180 greedy per-fold image-agreement audit": 4,
        "30% low-confidence selective review": 5,
        "30% direction-aware image router": 6,
        "v173 aggressive image-only review corrector": 7,
        "v174 disagreement flip": 8,
        "v175 error-enriched flip-risk": 9,
    }
    table["paper_order"] = table["module"].map(priority).fillna(99).astype(int)
    main_scopes = ["all_domains", "old_data", "third_batch", "strict_external"]
    main_table = table.loc[table["scope"].isin(main_scopes)].sort_values(["paper_order", "scope"]).copy()
    main_table.to_csv(OUT_DIR / "v184_main_operating_table.csv", index=False, encoding="utf-8-sig")

    clean_gate_n = int(gate["current_safety_status"].eq("clean_current_split").sum())
    gate_n = int(len(gate))
    gate_summary = pd.DataFrame(
        [
            {
                "evidence": "unlabeled severe-shift gate stability",
                "clean_current_split_n": clean_gate_n,
                "total_gate_variants": gate_n,
                "false_internal_trigger_n": int(gate["false_internal_trigger"].astype(bool).sum()),
                "missed_strict_external_n": int(gate["missed_strict_external"].astype(bool).sum()),
                "interpretation": "Most gate variants trigger strict external without false internal trigger; keep as deployment risk audit, not as treatment-level proof.",
            }
        ]
    )
    gate_summary.to_csv(OUT_DIR / "v184_unlabeled_gate_summary.csv", index=False, encoding="utf-8-sig")

    all_rows = table.loc[table["scope"].eq("all_domains")].copy()
    strict_rows = table.loc[table["scope"].eq("strict_external")].copy()
    v118 = all_rows.loc[all_rows["module"].eq("v118 locked high-safety two-signal scorecard")].iloc[0]
    v161 = all_rows.loc[all_rows["module"].eq("v161 safe-release scorecard")].iloc[0]
    v182 = all_rows.loc[all_rows["module"].eq("v182 stable fixed image-agreement release")].iloc[0]
    v180 = all_rows.loc[all_rows["module"].eq("v180 greedy per-fold image-agreement audit")].iloc[0]
    lowconf_ext = strict_rows.loc[strict_rows["module"].eq("30% low-confidence selective review")].iloc[0]
    router_ext = strict_rows.loc[strict_rows["module"].eq("30% direction-aware image router")].iloc[0]

    figure_index = pd.DataFrame(
        [
            {
                "figure_or_table": "Main Table 1",
                "file": "v184_main_operating_table.csv",
                "purpose": "Operating points after stable release audit, including main workflow, efficiency refinement, comparators, and negative controls.",
            },
            {
                "figure_or_table": "Claim Table",
                "file": "v184_claim_boundary_table.csv",
                "purpose": "Supported, candidate, and unsupported claims after v180-v183 audits.",
            },
            {
                "figure_or_table": "Gate Summary",
                "file": "v184_unlabeled_gate_summary.csv",
                "purpose": "Unlabeled severe-shift gate stability evidence for deployment risk auditing.",
            },
        ]
    )
    figure_index.to_csv(OUT_DIR / "v184_paper_entry_index.csv", index=False, encoding="utf-8-sig")

    claims = pd.concat(
        [
            claims,
            pd.DataFrame(
                [
                    {
                        "claim": "Unlabeled severe-shift gate supports deployment-time risk auditing.",
                        "status": "candidate",
                        "evidence": f"{clean_gate_n}/{gate_n} gate variants cleanly trigger strict external only in current split.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    claims.to_csv(OUT_DIR / "v184_claim_boundary_table.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v184 Paper Entry After Stable Release",
        "",
        "## Current Main Result",
        "",
        (
            f"- v118 high-safety baseline: all-domain BAcc {pct(v118['balanced_accuracy'])}, review/reject "
            f"{pct(v118['remaining_review_or_reject_rate'])}, FN={int(v118['fn'])}, FP={int(v118['fp'])}."
        ),
        (
            f"- v182 stable fixed release: all-domain BAcc {pct(v182['balanced_accuracy'])}, review/reject "
            f"{pct(v182['remaining_review_or_reject_rate'])}, released errors {int(v182['released_error_n'])}, "
            f"FN={int(v182['fn'])}, FP={int(v182['fp'])}."
        ),
        (
            f"- v161 scorecard-only release remains the cleaner efficiency baseline: review/reject "
            f"{pct(v161['remaining_review_or_reject_rate'])}, released errors {int(v161['released_error_n'])}."
        ),
        "",
        "## Mechanism / Comparator Evidence",
        "",
        (
            f"- Direction-aware router under strict external shift: BAcc {pct(router_ext['balanced_accuracy'])} "
            f"vs {pct(lowconf_ext['balanced_accuracy'])} for low-confidence review at comparable review budget."
        ),
        (
            f"- Greedy per-fold release audit is intentionally negative: review/reject {pct(v180['remaining_review_or_reject_rate'])}, "
            f"but released errors {int(v180['released_error_n'])}."
        ),
        (
            f"- Unlabeled severe-shift gate: {clean_gate_n}/{gate_n} current variants are clean on the present split."
        ),
        "",
        "## Writing Boundary",
        "",
        "The paper entry should now use v118 as the safety baseline and v182 as the conservative efficiency refinement. Automatic flipping remains negative evidence and should not be described as deployment-ready.",
    ]
    (OUT_DIR / "v184_paper_entry_after_stable_release.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "main_table_rows": int(len(main_table)),
        "claim_count": int(len(claims)),
        "v118_bacc": float(v118["balanced_accuracy"]),
        "v118_review_rate": float(v118["remaining_review_or_reject_rate"]),
        "v182_bacc": float(v182["balanced_accuracy"]),
        "v182_review_rate": float(v182["remaining_review_or_reject_rate"]),
        "v182_released_errors": int(v182["released_error_n"]),
        "strict_external_router_bacc": float(router_ext["balanced_accuracy"]),
        "strict_external_lowconf_bacc": float(lowconf_ext["balanced_accuracy"]),
        "clean_gate_variants": clean_gate_n,
        "gate_variants": gate_n,
    }
    (OUT_DIR / "v184_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v184] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
