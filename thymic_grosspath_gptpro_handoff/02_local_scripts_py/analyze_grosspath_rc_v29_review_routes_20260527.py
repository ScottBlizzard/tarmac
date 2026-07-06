from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V27_DIR = ROOT / "outputs" / "grosspath_rc_v27_unified_workflow_20260527"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v29_review_route_analysis_20260527"

POLICIES = [
    "P3_qscore92_review_else_safety",
    "P4_rf_quality_review_else_safety",
    "P5_extra_trees_quality_review_else_safety",
    "P6_manual_quality_or_safety_review_upper",
    "P7_logistic_recall90_quality_or_safety_review",
]


def pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(V27_DIR / "v27_unified_case_routes_external.csv")

    # P2 is the current strict automatic safety-switch baseline. We use it as the
    # counterfactual reference for asking whether a review gate catches baseline errors.
    base = (
        routes.loc[routes["policy"].eq("P2_safety_switch_auto"), ["case_id", "auto_pred"]]
        .rename(columns={"auto_pred": "p2_pred"})
        .copy()
    )
    base["p2_pred"] = base["p2_pred"].astype(int)

    summary_rows = []
    route_rows = []
    case_rows = []

    for policy in POLICIES:
        sub = routes.loc[routes["policy"].eq(policy)].merge(base, on="case_id", how="left")
        sub["p2_wrong"] = sub["p2_pred"].ne(sub["label_idx"])
        sub["risk_control"] = sub["review_flag"].eq(1) | sub["retake_flag"].eq(1)
        sub["auto_output"] = sub["auto_output_flag"].eq(1)

        risk = sub["risk_control"]
        p2_wrong = sub["p2_wrong"]
        auto = sub["auto_output"]

        captured = int((risk & p2_wrong).sum())
        missed = int((auto & p2_wrong).sum())
        risk_correct = int((risk & ~p2_wrong).sum())
        total_p2_wrong = int(p2_wrong.sum())
        total_risk = int(risk.sum())

        summary_rows.append(
            {
                "policy": policy,
                "n": len(sub),
                "risk_control_n": total_risk,
                "risk_control_rate_pct": pct(total_risk / len(sub)),
                "p2_baseline_wrong_n": total_p2_wrong,
                "captured_p2_wrong_n": captured,
                "captured_p2_wrong_rate_pct": pct(captured / total_p2_wrong) if total_p2_wrong else 0.0,
                "missed_p2_wrong_n": missed,
                "risk_control_on_p2_correct_n": risk_correct,
                "risk_control_precision_vs_p2_error_pct": pct(captured / total_risk) if total_risk else 0.0,
                "review_n": int(sub["review_flag"].sum()),
                "retake_n": int(sub["retake_flag"].sum()),
                "quality_review_n": int(sub["quality_review_flag"].sum()),
                "safety_review_n": int(sub["safety_review_flag"].sum()),
                "both_quality_safety_n": int((sub["quality_review_flag"].eq(1) & sub["safety_review_flag"].eq(1)).sum()),
                "workflow_correct_n": int(sub["workflow_correct"].sum()),
                "workflow_accuracy_pct": pct(sub["workflow_correct"].mean()),
            }
        )

        grouped = (
            sub.groupby("route", dropna=False)
            .agg(
                n=("case_id", "size"),
                p2_wrong_n=("p2_wrong", "sum"),
                label_high_n=("label_idx", "sum"),
                workflow_wrong_n=("workflow_correct", lambda s: int((~s.astype(bool)).sum())),
            )
            .reset_index()
        )
        grouped["policy"] = policy
        grouped["p2_wrong_rate_pct"] = grouped.apply(
            lambda r: pct(r["p2_wrong_n"] / r["n"]) if r["n"] else 0.0,
            axis=1,
        )
        route_rows.append(grouped)

        def add_cases(mask: pd.Series, bucket: str) -> None:
            cols = [
                "policy",
                "case_id",
                "original_case_id",
                "source_folder",
                "task_l6_label",
                "task_l7_label",
                "image_name",
                "quality_status",
                "quality_score",
                "manual_quality_status_v1",
                "label_idx",
                "p2_pred",
                "main_prob",
                "main_pred",
                "robust_prob",
                "robust_pred",
                "prob_mean_core",
                "safety_trigger",
                "quality_review_flag",
                "safety_review_flag",
                "retake_flag",
                "route",
                "workflow_final_pred",
                "workflow_correct",
                "error_type",
            ]
            tmp = sub.loc[mask, cols].copy()
            tmp.insert(1, "bucket", bucket)
            case_rows.append(tmp)

        add_cases(risk & p2_wrong, "captured_p2_error_by_risk_control")
        add_cases(auto & p2_wrong, "missed_p2_error_auto_output")
        add_cases(risk & ~p2_wrong, "risk_control_on_p2_correct")

    summary = pd.DataFrame(summary_rows)
    by_route = pd.concat(route_rows, ignore_index=True)
    cases = pd.concat(case_rows, ignore_index=True)

    summary.to_csv(OUT_DIR / "v29_review_route_summary.csv", index=False, encoding="utf-8-sig")
    by_route.to_csv(OUT_DIR / "v29_review_route_by_route.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v29_review_route_case_buckets.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
