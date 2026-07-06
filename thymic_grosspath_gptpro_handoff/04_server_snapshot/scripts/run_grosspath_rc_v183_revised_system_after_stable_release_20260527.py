from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v183_revised_system_after_stable_release_20260527"
V179 = ROOT / "outputs" / "grosspath_rc_v179_current_system_evidence_table_20260527" / "v179_current_system_operating_table.csv"
V180 = ROOT / "outputs" / "grosspath_rc_v180_nested_v178_image_agreement_release_20260527" / "v180_nested_workflow_summary.csv"
V182 = ROOT / "outputs" / "grosspath_rc_v182_stable_fixed_image_agreement_release_20260527" / "v182_stable_fixed_workflow_summary.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def rows_from_summary(df: pd.DataFrame, workflow: str, module: str, tier: str, role: str, source: str) -> list[dict[str, object]]:
    out = []
    for _, r in df.loc[df["workflow"].eq(workflow)].iterrows():
        out.append(
            {
                "module": module,
                "tier": tier,
                "role": role,
                "scope": r["scope"],
                "auto_decision_rate": 1.0 - float(r["final_review_rate"]),
                "remaining_review_or_reject_rate": float(r["final_review_rate"]),
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "accuracy": float(r["accuracy"]),
                "f1": float(r["f1"]),
                "auc": float(r["auc"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "released_error_n": int(r["released_error_n"]),
                "source": source,
            }
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v179 = pd.read_csv(V179)
    v180 = pd.read_csv(V180)
    v182 = pd.read_csv(V182)

    base = v179.loc[
        v179["module"].isin(
            [
                "v118 locked high-safety two-signal scorecard",
                "30% low-confidence selective review",
                "30% direction-aware image router",
                "v173 aggressive image-only review corrector",
                "v174 disagreement flip",
                "v175 error-enriched flip-risk",
            ]
        )
    ].copy()
    base = base.rename(columns={"auto_action_error_n": "released_error_n"})
    if "released_error_n" in base:
        base["released_error_n"] = base["released_error_n"].fillna("")

    rows = []
    rows += rows_from_summary(
        v182,
        "v161_safe_release_fixed",
        "v161 safe-release scorecard",
        "Recommended efficiency candidate",
        "scorecard-only safe-release",
        "v182/v161",
    )
    rows += rows_from_summary(
        v182,
        "v182_stable_fixed_image_agreement_union",
        "v182 stable fixed image-agreement release",
        "Recommended conservative efficiency refinement",
        "fixed image-agreement release on top of v161",
        "v182",
    )
    rows += rows_from_summary(
        v180,
        "v180_nested_image_agreement_union",
        "v180 greedy per-fold image-agreement audit",
        "Negative stability audit",
        "shows why per-fold greedy release is unsafe",
        "v180",
    )
    revised = pd.concat([base, pd.DataFrame(rows)], ignore_index=True, sort=False)
    revised.to_csv(OUT_DIR / "v183_revised_system_operating_table.csv", index=False, encoding="utf-8-sig")

    all_rows = revised.loc[revised["scope"].eq("all_domains")]
    v118 = all_rows.loc[all_rows["module"].eq("v118 locked high-safety two-signal scorecard")].iloc[0]
    v161 = all_rows.loc[all_rows["module"].eq("v161 safe-release scorecard")].iloc[0]
    v182r = all_rows.loc[all_rows["module"].eq("v182 stable fixed image-agreement release")].iloc[0]
    v180r = all_rows.loc[all_rows["module"].eq("v180 greedy per-fold image-agreement audit")].iloc[0]

    claims = pd.DataFrame(
        [
            {
                "claim": "The safest locked operating point remains the two-signal scorecard.",
                "status": "supported",
                "evidence": f"v118 BAcc {pct(v118['balanced_accuracy'])}, review {pct(v118['remaining_review_or_reject_rate'])}, FN={int(v118['fn'])}, FP={int(v118['fp'])}.",
            },
            {
                "claim": "Safe-release can reduce review burden while preserving current safety.",
                "status": "supported as candidate",
                "evidence": f"v161 review {pct(v161['remaining_review_or_reject_rate'])}; v182 review {pct(v182r['remaining_review_or_reject_rate'])}, released errors {int(v182r['released_error_n'])}.",
            },
            {
                "claim": "Image-agreement release is useful only under fixed stability constraints.",
                "status": "supported",
                "evidence": f"v182 fixed stable rule has 0 released errors; v180 greedy per-fold audit has {int(v180r['released_error_n'])} released errors.",
            },
            {
                "claim": "Automatic flipping is deployable.",
                "status": "not supported",
                "evidence": "v174/v175 remain negative and are kept as boundary evidence.",
            },
        ]
    )
    claims.to_csv(OUT_DIR / "v183_revised_claim_table.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v183 Revised System After Stable Release",
        "",
        "## Recommended Operating Points",
        "",
        (
            f"- v118 high-safety baseline: BAcc {pct(v118['balanced_accuracy'])}, review "
            f"{pct(v118['remaining_review_or_reject_rate'])}, FN={int(v118['fn'])}, FP={int(v118['fp'])}."
        ),
        (
            f"- v182 conservative efficiency refinement: BAcc {pct(v182r['balanced_accuracy'])}, review "
            f"{pct(v182r['remaining_review_or_reject_rate'])}, released errors {int(v182r['released_error_n'])}, "
            f"FN={int(v182r['fn'])}, FP={int(v182r['fp'])}."
        ),
        (
            f"- v180 is kept as a negative stability audit: greedy per-fold release lowers review to "
            f"{pct(v180r['remaining_review_or_reject_rate'])} but causes {int(v180r['released_error_n'])} released errors."
        ),
        "",
        "## Paper Boundary",
        "",
        "Write the image-agreement component as a fixed, stability-constrained release refinement. Do not present greedy per-fold release or automatic flipping as deployment-ready.",
    ]
    (OUT_DIR / "v183_revised_system_after_stable_release.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "v118_bacc": float(v118["balanced_accuracy"]),
        "v118_review_rate": float(v118["remaining_review_or_reject_rate"]),
        "v182_bacc": float(v182r["balanced_accuracy"]),
        "v182_review_rate": float(v182r["remaining_review_or_reject_rate"]),
        "v182_released_errors": int(v182r["released_error_n"]),
        "v180_released_errors": int(v180r["released_error_n"]),
    }
    (OUT_DIR / "v183_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v183] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
