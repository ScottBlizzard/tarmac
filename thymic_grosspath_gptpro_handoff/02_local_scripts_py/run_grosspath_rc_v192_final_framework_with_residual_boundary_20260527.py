from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v192_final_framework_with_residual_boundary_20260527"
V186_TABLE = ROOT / "outputs" / "grosspath_rc_v186_final_framework_table_20260527" / "v186_final_framework_operating_table.csv"
V186_CLAIMS = ROOT / "outputs" / "grosspath_rc_v186_final_framework_table_20260527" / "v186_final_framework_claim_table.csv"
V189_ERRORS = ROOT / "outputs" / "grosspath_rc_v189_adaptive_error_review_anatomy_20260527" / "v189_adaptive_auto_residual_errors.csv"
V190_SUMMARY = ROOT / "outputs" / "grosspath_rc_v190_fn_sentinel_nested_scan_20260527" / "v190_fn_sentinel_workflow_summary.csv"
V190_REPORT = ROOT / "outputs" / "grosspath_rc_v190_fn_sentinel_nested_scan_20260527" / "v190_run_report.json"
V191_SUMMARY = ROOT / "outputs" / "grosspath_rc_v191_dino_fn_risk_sentinel_20260527" / "v191_fn_risk_sentinel_summary.csv"
V191_QUALITY = ROOT / "outputs" / "grosspath_rc_v191_dino_fn_risk_sentinel_20260527" / "v191_fn_risk_model_quality.csv"
V191_REPORT = ROOT / "outputs" / "grosspath_rc_v191_dino_fn_risk_sentinel_20260527" / "v191_run_report.json"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def row_from_sentinel(summary: pd.DataFrame, workflow: str, module: str, tier: str, role: str, source: str) -> dict[str, object]:
    r = summary.loc[summary["workflow"].eq(workflow) & summary["scope"].eq("all_domains")].iloc[0]
    return {
        "module": module,
        "tier": tier,
        "role": role,
        "scope": "all_domains",
        "auto_decision_rate": 1.0 - float(r["review_or_reject_rate"]),
        "remaining_review_or_reject_rate": float(r["review_or_reject_rate"]),
        "balanced_accuracy": float(r["balanced_accuracy"]),
        "accuracy": float(r["accuracy"]),
        "f1": float(r["f1"]),
        "auc": float(r["auc"]),
        "fn": int(r["fn"]),
        "fp": int(r["fp"]),
        "released_error_n": "",
        "source": source,
        "additional_sentinel_review_n": int(r["additional_sentinel_review_n"]),
        "sentinel_rescued_fn_n": int(r["sentinel_rescued_fn_n"]),
        "auto_error_n": int(r["auto_error_n"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(V186_TABLE)
    claims = pd.read_csv(V186_CLAIMS)
    residual = pd.read_csv(V189_ERRORS, dtype={"case_id": str, "original_case_id": str})
    v190 = pd.read_csv(V190_SUMMARY)
    v191 = pd.read_csv(V191_SUMMARY)
    v191q = pd.read_csv(V191_QUALITY)

    sentinel_rows = [
        row_from_sentinel(
            v190,
            "v190_nested_fn_sentinel",
            "v190 fold-wise probability FN sentinel",
            "Negative residual-boundary audit",
            "simple probability sentinel for final FN",
            "v190",
        ),
        row_from_sentinel(
            v191,
            "v191_tab_dino_logreg_c03",
            "v191 learned DINO FN-risk sentinel",
            "Negative residual-boundary audit",
            "learned DINO/probability FN-risk sentinel",
            "v191",
        ),
    ]
    table2 = pd.concat([table, pd.DataFrame(sentinel_rows)], ignore_index=True, sort=False)
    order = {
        "v185 unlabeled shift-adaptive workflow": 1,
        "v182 fixed stable efficiency workflow": 2,
        "v118 fixed high-safety fallback": 3,
        "v180 greedy per-fold image-agreement audit": 4,
        "v190 fold-wise probability FN sentinel": 5,
        "v191 learned DINO FN-risk sentinel": 6,
        "30% low-confidence selective review": 7,
        "30% direction-aware image router": 8,
        "v173 aggressive image-only review corrector": 9,
        "v174 disagreement flip": 10,
        "v175 error-enriched flip-risk": 11,
    }
    table2["paper_order"] = table2["module"].map(order).fillna(99).astype(int)
    table2 = table2.sort_values(["paper_order", "scope"]).copy()
    table2.to_csv(OUT_DIR / "v192_final_framework_operating_table.csv", index=False, encoding="utf-8-sig")

    case_id = str(residual.iloc[0]["original_case_id"]) if not residual.empty else ""
    case_label = str(residual.iloc[0]["task_l6_label"]) if not residual.empty else ""
    v190_all = v190.loc[v190["workflow"].eq("v190_nested_fn_sentinel") & v190["scope"].eq("all_domains")].iloc[0]
    v191_all = v191.loc[v191["workflow"].eq("v191_tab_dino_logreg_c03") & v191["scope"].eq("all_domains")].iloc[0]
    best_auc = v191q.sort_values("internal_lowrisk_auc", ascending=False).iloc[0]

    claims2 = pd.concat(
        [
            claims,
            pd.DataFrame(
                [
                    {
                        "claim": "The final residual automatic FN is not recoverable by non-leaky simple sentinels in the current sample.",
                        "status": "supported as residual-boundary evidence",
                        "evidence": f"v190 nested sentinel rescues {int(v190_all['sentinel_rescued_fn_n'])} FN and leaves auto_error={int(v190_all['auto_error_n'])}; full-fit post-hoc can catch {case_id}, so the difference is leakage-sensitive.",
                    },
                    {
                        "claim": "A learned DINO/probability FN-risk sentinel is not yet deployable.",
                        "status": "not supported",
                        "evidence": f"v191 best AUROC model reaches internal low-risk AUROC {best_auc['internal_lowrisk_auc']:.3f}, but nested sentinel still rescues {int(v191_all['sentinel_rescued_fn_n'])} FN and leaves auto_error={int(v191_all['auto_error_n'])}.",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    claims2.to_csv(OUT_DIR / "v192_final_framework_claim_table.csv", index=False, encoding="utf-8-sig")

    residual.to_csv(OUT_DIR / "v192_residual_auto_error_case.csv", index=False, encoding="utf-8-sig")
    v191q.to_csv(OUT_DIR / "v192_fn_risk_model_quality.csv", index=False, encoding="utf-8-sig")

    adaptive = table2.loc[
        table2["module"].eq("v185 unlabeled shift-adaptive workflow") & table2["scope"].eq("all_domains")
    ].iloc[0]
    md = [
        "# v192 Final Framework With Residual Boundary",
        "",
        "## Current Recommended Framework",
        "",
        (
            f"- v185 adaptive workflow remains the recommended deployment framework: BAcc "
            f"{pct(adaptive['balanced_accuracy'])}, review/reject {pct(adaptive['remaining_review_or_reject_rate'])}, "
            f"FN={int(adaptive['fn'])}, FP={int(adaptive['fp'])}."
        ),
        "",
        "## Residual Boundary",
        "",
        (
            f"- The only residual automatic error is {case_id} ({case_label}), a high-risk-to-low-risk FN."
        ),
        (
            f"- v190 simple fold-wise sentinel: additional review {int(v190_all['additional_sentinel_review_n'])}, "
            f"rescued FN {int(v190_all['sentinel_rescued_fn_n'])}, auto errors {int(v190_all['auto_error_n'])}."
        ),
        (
            f"- v191 DINO/probability sentinel: additional review {int(v191_all['additional_sentinel_review_n'])}, "
            f"rescued FN {int(v191_all['sentinel_rescued_fn_n'])}, auto errors {int(v191_all['auto_error_n'])}."
        ),
        "",
        "## Writing Boundary",
        "",
        "Do not post-hoc tune a sentinel around 2516531. The stronger claim is that the framework explicitly audits residual risk and refuses to promote leakage-sensitive fixes.",
    ]
    (OUT_DIR / "v192_final_framework_with_residual_boundary.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "final_table_rows": int(len(table2)),
        "claim_count": int(len(claims2)),
        "residual_case_id": case_id,
        "v190_auto_error_n": int(v190_all["auto_error_n"]),
        "v190_rescued_fn_n": int(v190_all["sentinel_rescued_fn_n"]),
        "v191_auto_error_n": int(v191_all["auto_error_n"]),
        "v191_rescued_fn_n": int(v191_all["sentinel_rescued_fn_n"]),
        "adaptive_bacc": float(adaptive["balanced_accuracy"]),
        "adaptive_review_rate": float(adaptive["remaining_review_or_reject_rate"]),
    }
    (OUT_DIR / "v192_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v192] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
