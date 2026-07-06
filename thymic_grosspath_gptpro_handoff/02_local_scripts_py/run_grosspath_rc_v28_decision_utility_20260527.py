from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V27_DIR = ROOT / "outputs" / "grosspath_rc_v27_unified_workflow_20260527"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v28_decision_utility_20260527"


SCENARIOS = [
    {
        "scenario": "balanced_clinical_cost",
        "fn_cost": 5.0,
        "fp_cost": 1.0,
        "review_cost": 0.20,
        "retake_cost": 0.50,
        "description": "漏诊成本较高，但允许一定复核工作量。",
    },
    {
        "scenario": "high_safety_low_review_cost",
        "fn_cost": 10.0,
        "fp_cost": 1.0,
        "review_cost": 0.15,
        "retake_cost": 0.50,
        "description": "高危漏诊优先，复核资源相对充足。",
    },
    {
        "scenario": "high_safety_high_review_cost",
        "fn_cost": 10.0,
        "fp_cost": 1.0,
        "review_cost": 0.75,
        "retake_cost": 1.00,
        "description": "高危漏诊成本高，但复核资源有限。",
    },
    {
        "scenario": "workflow_limited",
        "fn_cost": 5.0,
        "fp_cost": 1.0,
        "review_cost": 1.00,
        "retake_cost": 1.50,
        "description": "临床工作量约束强，复核成本高。",
    },
    {
        "scenario": "screening_safety_first",
        "fn_cost": 20.0,
        "fp_cost": 1.0,
        "review_cost": 0.25,
        "retake_cost": 0.75,
        "description": "筛查场景，尽量避免高危漏诊。",
    },
]


def is_pareto_efficient(frame: pd.DataFrame) -> pd.Series:
    # Maximize BAcc and minimize risk-control rate and FN.
    efficient = []
    for idx, row in frame.iterrows():
        dominated = False
        for jdx, other in frame.iterrows():
            if idx == jdx:
                continue
            no_worse = (
                other["workflow_balanced_accuracy"] >= row["workflow_balanced_accuracy"]
                and other["risk_control_rate"] <= row["risk_control_rate"]
                and other["workflow_fn"] <= row["workflow_fn"]
            )
            strictly_better = (
                other["workflow_balanced_accuracy"] > row["workflow_balanced_accuracy"]
                or other["risk_control_rate"] < row["risk_control_rate"]
                or other["workflow_fn"] < row["workflow_fn"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        efficient.append(not dominated)
    return pd.Series(efficient, index=frame.index)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(V27_DIR / "v27_unified_workflow_metrics.csv")
    n = float(metrics["n"].iloc[0])

    utility_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        base_cost = None
        for _, row in metrics.iterrows():
            total_cost = (
                float(row["workflow_fn"]) * scenario["fn_cost"]
                + float(row["workflow_fp"]) * scenario["fp_cost"]
                + float(row["review_n"]) * scenario["review_cost"]
                + float(row["retake_n"]) * scenario["retake_cost"]
            )
            normalized_cost = total_cost / n
            if row["policy"] == "P0_main_auto_all":
                base_cost = normalized_cost
            utility_rows.append(
                {
                    "scenario": scenario["scenario"],
                    "description": scenario["description"],
                    "policy": row["policy"],
                    "evidence_level": row["evidence_level"],
                    "fn_cost": scenario["fn_cost"],
                    "fp_cost": scenario["fp_cost"],
                    "review_cost": scenario["review_cost"],
                    "retake_cost": scenario["retake_cost"],
                    "normalized_cost": normalized_cost,
                    "workflow_accuracy": row["workflow_accuracy"],
                    "workflow_balanced_accuracy": row["workflow_balanced_accuracy"],
                    "workflow_sensitivity": row["workflow_sensitivity"],
                    "workflow_specificity": row["workflow_specificity"],
                    "workflow_fn": row["workflow_fn"],
                    "workflow_fp": row["workflow_fp"],
                    "risk_control_rate": row["risk_control_rate"],
                    "review_rate": row["review_rate"],
                    "retake_rate": row["retake_rate"],
                }
            )
        if base_cost is not None:
            for item in utility_rows:
                if item["scenario"] == scenario["scenario"]:
                    item["cost_reduction_vs_main"] = base_cost - float(item["normalized_cost"])
                    item["relative_cost_reduction_vs_main"] = (base_cost - float(item["normalized_cost"])) / base_cost if base_cost else np.nan

    utility_df = pd.DataFrame(utility_rows)
    utility_df.to_csv(OUT_DIR / "v28_decision_utility_by_policy.csv", index=False, encoding="utf-8-sig")

    best = (
        utility_df.sort_values(["scenario", "normalized_cost", "workflow_balanced_accuracy"], ascending=[True, True, False])
        .groupby("scenario", as_index=False)
        .head(3)
    )
    best.to_csv(OUT_DIR / "v28_decision_utility_top3_by_scenario.csv", index=False, encoding="utf-8-sig")

    frontier = metrics.copy()
    frontier["pareto_efficient"] = is_pareto_efficient(frontier)
    frontier.to_csv(OUT_DIR / "v28_pareto_frontier_accuracy_vs_burden.csv", index=False, encoding="utf-8-sig")

    print(f"[done] {OUT_DIR}")
    print("\nTop strategies by cost scenario:")
    print(
        best[
            [
                "scenario",
                "policy",
                "evidence_level",
                "normalized_cost",
                "cost_reduction_vs_main",
                "relative_cost_reduction_vs_main",
                "workflow_balanced_accuracy",
                "workflow_fn",
                "workflow_fp",
                "risk_control_rate",
            ]
        ].to_string(index=False)
    )
    print("\nPareto efficient policies:")
    print(
        frontier[frontier["pareto_efficient"]][
            [
                "policy",
                "evidence_level",
                "workflow_balanced_accuracy",
                "workflow_accuracy",
                "workflow_fn",
                "workflow_fp",
                "risk_control_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
