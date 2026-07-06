from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v55_final_tier_table_20260527"
V51 = ROOT / "outputs" / "grosspath_rc_v51_workflow_validation_20260527"
V52 = ROOT / "outputs" / "grosspath_rc_v52_quality_retake_overlay_20260527"
V54 = ROOT / "outputs" / "grosspath_rc_v54_constrained_policy_search_20260527"
V56 = ROOT / "outputs" / "grosspath_rc_v56_calibrated_policy_selection_20260527"
V59 = ROOT / "outputs" / "grosspath_rc_v59_lowrisk_boundary_specialist_20260527"


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v51 = pd.read_csv(V51 / "v51_tiered_workflow_summary.csv")
    v51_ext = v51.loc[v51["split"].eq("external")].set_index("policy")
    v52 = pd.read_csv(V52 / "v52_quality_retake_overlay_summary.csv").set_index("policy")
    v54 = pd.read_csv(V54 / "v54_key_policy_bootstrap_ci.csv").set_index("policy_name")
    v56 = pd.read_csv(V56 / "v56_calibration_split_selection_summary.csv").set_index("scenario")
    v59 = pd.read_csv(V59 / "v59_selected_policy_bootstrap_ci.csv").set_index("policy_name")

    rows = []

    def add_from_v51(policy: str, tier: str, role: str, claim_boundary: str) -> None:
        r = v51_ext.loc[policy]
        rows.append(
            {
                "tier": tier,
                "source": policy,
                "role": role,
                "review_or_control_rate_pct": pct(r["review_rate"]),
                "external_acc_pct": pct(r["accuracy"]),
                "external_bacc_pct": pct(r["balanced_accuracy"]),
                "external_sensitivity_pct": pct(r["sensitivity"]),
                "external_specificity_pct": pct(r["specificity"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "claim_boundary": claim_boundary,
            }
        )

    add_from_v51("P2_pure_auto", "0", "纯自动参考基线", "不作为部署主张")
    add_from_v51("v37_balanced_dev97", "1", "均衡风险控制档", "中等复核负担，均衡降低 FN/FP")

    r54 = v54.loc["v54_low_control_highsens"]
    r56 = v56.loc["empirical_sens985_spec95"]
    rows.append(
        {
            "tier": "2",
            "source": "v54_low_control_highsens",
            "role": "低控制高敏感性档",
            "review_or_control_rate_pct": pct(r54["review_rate"]),
            "external_acc_pct": pct(r54["accuracy"]),
            "external_bacc_pct": pct(r54["balanced_accuracy"]),
            "external_sensitivity_pct": pct(r54["sensitivity"]),
            "external_specificity_pct": pct(r54["specificity"]),
            "fn": int(r54["fn"]),
            "fp": int(r54["fp"]),
            "claim_boundary": f"省复核版本；1000次校准外部BAcc中位 {pct(r56['external_bacc_median'])}%",
        }
    )

    r59 = v59.loc["v59_specialist_logistic_addon04"]
    rows.append(
        {
            "tier": "2b",
            "source": "v59_specialist_logistic_addon04",
            "role": "低危边界专项折中档",
            "review_or_control_rate_pct": pct(r59["review_rate"]),
            "external_acc_pct": pct(r59["accuracy"]),
            "external_bacc_pct": pct(r59["balanced_accuracy"]),
            "external_sensitivity_pct": pct(r59["sensitivity"]),
            "external_specificity_pct": pct(r59["specificity"]),
            "fn": int(r59["fn"]),
            "fp": int(r59["fp"]),
            "claim_boundary": "可部署图像+模型特征专项复核；比v50省复核",
        }
    )

    add_from_v51("v48_direction_dev97", "3", "方向感知风险控制档", "方向感知主线，进一步降低高危漏诊")
    add_from_v51("v50_sens98_spec90", "4", "当前主推高安全档", "主结果：外部 BAcc 97.3%，FN 1")

    r52 = v52.loc["v50_plus_quality_score_le82"]
    rows.append(
        {
            "tier": "5a",
            "source": "v52_quality_score_le82_overlay",
            "role": "质量拒判候选档",
            "review_or_control_rate_pct": pct(r52["total_control_rate"]),
            "external_acc_pct": pct(r52["accuracy"]),
            "external_bacc_pct": pct(r52["balanced_accuracy"]),
            "external_sensitivity_pct": pct(r52["sensitivity"]),
            "external_specificity_pct": pct(r52["specificity"]),
            "fn": int(r52["fn"]),
            "fp": int(r52["fp"]),
            "claim_boundary": "探索性质量拒判，需前瞻验证",
        }
    )

    r52 = v52.loc["v50_plus_quality_score_le88"]
    rows.append(
        {
            "tier": "5b",
            "source": "v52_quality_score_le88_overlay",
            "role": "最高安全探索档",
            "review_or_control_rate_pct": pct(r52["total_control_rate"]),
            "external_acc_pct": pct(r52["accuracy"]),
            "external_bacc_pct": pct(r52["balanced_accuracy"]),
            "external_sensitivity_pct": pct(r52["sensitivity"]),
            "external_specificity_pct": pct(r52["specificity"]),
            "fn": int(r52["fn"]),
            "fp": int(r52["fp"]),
            "claim_boundary": "探索性，不作为纯自动模型能力",
        }
    )

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "v55_final_tier_strategy_table.csv", index=False, encoding="utf-8-sig")

    doctor = table[
        [
            "tier",
            "role",
            "review_or_control_rate_pct",
            "external_bacc_pct",
            "external_sensitivity_pct",
            "external_specificity_pct",
            "fn",
            "fp",
            "claim_boundary",
        ]
    ].copy()
    doctor.columns = ["档位", "定位", "复核/控制比例(%)", "外部BAcc(%)", "敏感性(%)", "特异性(%)", "FN", "FP", "边界"]
    doctor.to_csv(OUT_DIR / "v55_doctor_facing_tier_strategy_table.csv", index=False, encoding="utf-8-sig")

    print(doctor.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
