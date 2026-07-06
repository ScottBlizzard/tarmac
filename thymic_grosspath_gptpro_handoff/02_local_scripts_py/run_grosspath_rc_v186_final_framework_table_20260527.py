from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v186_final_framework_table_20260527"
V184_MAIN = ROOT / "outputs" / "grosspath_rc_v184_paper_entry_after_stable_release_20260527" / "v184_main_operating_table.csv"
V184_CLAIMS = ROOT / "outputs" / "grosspath_rc_v184_paper_entry_after_stable_release_20260527" / "v184_claim_boundary_table.csv"
V185_SUMMARY = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_summary.csv"
V185_GATE = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_gate_policy_definition.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def adaptive_rows(v185: pd.DataFrame) -> pd.DataFrame:
    rows = []
    label_map = {
        "fixed_v118_high_safety": ("v118 fixed high-safety fallback", "High-safety fallback", "all batches use strict high-safety scorecard"),
        "fixed_v182_stable_release": ("v182 fixed stable efficiency workflow", "Standard efficiency workflow", "all batches use stable fixed release"),
        "unlabeled_shift_adaptive_v182_to_v118": (
            "v185 unlabeled shift-adaptive workflow",
            "Recommended deployment framework",
            "within-internal batches use v182; severe-shift batches fall back to v118",
        ),
    }
    for _, r in v185.iterrows():
        module, tier, role = label_map[str(r["workflow"])]
        rows.append(
            {
                "module": module,
                "tier": tier,
                "role": role,
                "scope": r["scope"],
                "auto_decision_rate": float(r["auto_decision_rate"]),
                "remaining_review_or_reject_rate": float(r["review_or_reject_rate"]),
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "accuracy": float(r["accuracy"]),
                "f1": float(r["f1"]),
                "auc": float(r["auc"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "released_error_n": "",
                "source": "v185",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v184 = pd.read_csv(V184_MAIN)
    claims = pd.read_csv(V184_CLAIMS)
    v185 = pd.read_csv(V185_SUMMARY)
    gate = pd.read_csv(V185_GATE)
    adaptive = adaptive_rows(v185)

    keep_v184 = v184.loc[
        v184["module"].isin(
            [
                "v180 greedy per-fold image-agreement audit",
                "30% low-confidence selective review",
                "30% direction-aware image router",
                "v173 aggressive image-only review corrector",
                "v174 disagreement flip",
                "v175 error-enriched flip-risk",
            ]
        )
    ].copy()
    final_table = pd.concat([adaptive, keep_v184], ignore_index=True, sort=False)
    order = {
        "v185 unlabeled shift-adaptive workflow": 1,
        "v182 fixed stable efficiency workflow": 2,
        "v118 fixed high-safety fallback": 3,
        "v180 greedy per-fold image-agreement audit": 4,
        "30% low-confidence selective review": 5,
        "30% direction-aware image router": 6,
        "v173 aggressive image-only review corrector": 7,
        "v174 disagreement flip": 8,
        "v175 error-enriched flip-risk": 9,
    }
    final_table["paper_order"] = final_table["module"].map(order).fillna(99).astype(int)
    final_table = final_table.sort_values(["paper_order", "scope"]).copy()
    final_table.to_csv(OUT_DIR / "v186_final_framework_operating_table.csv", index=False, encoding="utf-8-sig")

    claims2 = pd.concat(
        [
            claims,
            pd.DataFrame(
                [
                    {
                        "claim": "Unlabeled shift-adaptive workflow provides deployment-time risk control.",
                        "status": "supported as framework logic",
                        "evidence": "v185 keeps BAcc 99.81%, review 58.94%, and falls back to v118 on severe-shift strict external batches.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    claims2.to_csv(OUT_DIR / "v186_final_framework_claim_table.csv", index=False, encoding="utf-8-sig")

    all_rows = final_table.loc[final_table["scope"].eq("all_domains")]
    strict_rows = final_table.loc[final_table["scope"].eq("strict_external")]
    adaptive_all = all_rows.loc[all_rows["module"].eq("v185 unlabeled shift-adaptive workflow")].iloc[0]
    fixed182_all = all_rows.loc[all_rows["module"].eq("v182 fixed stable efficiency workflow")].iloc[0]
    fixed118_all = all_rows.loc[all_rows["module"].eq("v118 fixed high-safety fallback")].iloc[0]
    adaptive_ext = strict_rows.loc[strict_rows["module"].eq("v185 unlabeled shift-adaptive workflow")].iloc[0]
    fixed182_ext = strict_rows.loc[strict_rows["module"].eq("v182 fixed stable efficiency workflow")].iloc[0]
    fixed118_ext = strict_rows.loc[strict_rows["module"].eq("v118 fixed high-safety fallback")].iloc[0]

    index = pd.DataFrame(
        [
            {
                "artifact": "v186_final_framework_operating_table.csv",
                "purpose": "Final current operating table after adding unlabeled shift-adaptive deployment workflow.",
            },
            {
                "artifact": "v186_final_framework_claim_table.csv",
                "purpose": "Claim boundary table including deployment-time risk-control claim.",
            },
            {
                "artifact": "v186_final_framework_summary.md",
                "purpose": "Short writing guide for paper/report use.",
            },
        ]
    )
    index.to_csv(OUT_DIR / "v186_artifact_index.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v186 Final Framework Table",
        "",
        "## Recommended Deployment Workflow",
        "",
        (
            f"- Recommended framework: v185 unlabeled shift-adaptive workflow, all-domain BAcc "
            f"{pct(adaptive_all['balanced_accuracy'])}, review/reject {pct(adaptive_all['remaining_review_or_reject_rate'])}, "
            f"FN={int(adaptive_all['fn'])}, FP={int(adaptive_all['fp'])}."
        ),
        (
            f"- Standard efficiency workflow: fixed v182, BAcc {pct(fixed182_all['balanced_accuracy'])}, "
            f"review/reject {pct(fixed182_all['remaining_review_or_reject_rate'])}."
        ),
        (
            f"- High-safety fallback: fixed v118, BAcc {pct(fixed118_all['balanced_accuracy'])}, "
            f"review/reject {pct(fixed118_all['remaining_review_or_reject_rate'])}."
        ),
        "",
        "## Strict External Behavior",
        "",
        (
            f"- On strict external, adaptive policy falls back to v118: review/reject "
            f"{pct(adaptive_ext['remaining_review_or_reject_rate'])}; fixed v182 would review "
            f"{pct(fixed182_ext['remaining_review_or_reject_rate'])}; fixed v118 reviews "
            f"{pct(fixed118_ext['remaining_review_or_reject_rate'])}."
        ),
        (
            f"- Current strict external BAcc is {pct(adaptive_ext['balanced_accuracy'])} for the adaptive branch. "
            "This should be described as current-split validation, not definitive external generalization."
        ),
        "",
        "## Gate Definition",
        "",
        f"- {gate.iloc[0]['policy_interpretation']}",
        "",
        "## Paper Boundary",
        "",
        "The main computational claim should be a risk-controlled, shift-adaptive selective diagnosis framework. Automatic flipping remains negative evidence; the framework's strength is stable release plus deployment-time rejection tightening.",
    ]
    (OUT_DIR / "v186_final_framework_summary.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "final_table_rows": int(len(final_table)),
        "claim_count": int(len(claims2)),
        "adaptive_all_bacc": float(adaptive_all["balanced_accuracy"]),
        "adaptive_all_review_rate": float(adaptive_all["remaining_review_or_reject_rate"]),
        "fixed_v182_review_rate": float(fixed182_all["remaining_review_or_reject_rate"]),
        "fixed_v118_review_rate": float(fixed118_all["remaining_review_or_reject_rate"]),
        "strict_external_adaptive_review_rate": float(adaptive_ext["remaining_review_or_reject_rate"]),
        "strict_external_fixed_v182_review_rate": float(fixed182_ext["remaining_review_or_reject_rate"]),
    }
    (OUT_DIR / "v186_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v186] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
