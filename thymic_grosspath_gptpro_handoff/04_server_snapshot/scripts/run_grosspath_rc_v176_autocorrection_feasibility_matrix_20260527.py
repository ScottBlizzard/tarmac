from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v176_autocorrection_feasibility_matrix_20260527"
V161_SUMMARY = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_summary.csv"
V173_SELECTED = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527" / "v173_selected_workflow_summary.csv"
V174_SELECTED = ROOT / "outputs" / "grosspath_rc_v174_disagree_flip_release_policy_20260527" / "v174_selected_flip_release_summary.csv"
V175_SELECTED = ROOT / "outputs" / "grosspath_rc_v175_error_enriched_flip_risk_20260527" / "v175_selected_flip_risk_summary.csv"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def pick_v173() -> pd.Series:
    df = pd.read_csv(V173_SELECTED)
    sub = df.loc[df["scope"].eq("all_domains")].copy()
    return sub.sort_values(["balanced_accuracy", "remaining_review_rate"], ascending=[False, True]).iloc[0]


def pick_v174(mode: str) -> pd.Series:
    df = pd.read_csv(V174_SELECTED)
    sub = df.loc[df["scope"].eq("all_domains") & df["mode"].eq(mode)].copy()
    if mode == "disagree_flip_only":
        return sub.sort_values(["rescued_n", "hurt_n", "balanced_accuracy"], ascending=[False, True, False]).iloc[0]
    return sub.sort_values(["auto_action_error_n", "remaining_review_rate", "balanced_accuracy"], ascending=[True, True, False]).iloc[0]


def pick_v175() -> pd.Series:
    df = pd.read_csv(V175_SELECTED)
    sub = df.loc[df["scope"].eq("all_domains")].copy()
    return sub.sort_values(["rescued_n", "hurt_n", "balanced_accuracy"], ascending=[False, True, False]).iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v161 = pd.read_csv(V161_SUMMARY)
    v118_all = v161.loc[v161["workflow"].eq("v118_high_safety_scorecard") & v161["scope"].eq("all_domains")].iloc[0]
    v161_all = v161.loc[v161["workflow"].eq("v161_safe_release_scorecard") & v161["scope"].eq("all_domains")].iloc[0]
    v173 = pick_v173()
    v174_release = pick_v174("agree_release_only")
    v174_flip = pick_v174("disagree_flip_only")
    v175 = pick_v175()

    rows = [
        {
            "module": "v118 high-safety scorecard",
            "action_type": "review/reject fallback",
            "selection_basis": "locked two-signal scorecard",
            "review_policy": "v118_review_or_control",
            "model": "two_signal_scorecard",
            "remaining_review_rate": float(v118_all["review_rate"]),
            "auto_action_rate": float(v118_all["auto_pass_rate"]),
            "balanced_accuracy": float(v118_all["balanced_accuracy"]),
            "fn": int(v118_all["fn"]),
            "fp": int(v118_all["fp"]),
            "rescued_n": None,
            "hurt_n": None,
            "auto_error_n": None,
            "claim_status": "Tier 1 high-safety baseline",
            "interpretation": "最稳的安全主线，但复核负担高。",
        },
        {
            "module": "v161 safe-release",
            "action_type": "release reviewed cases by scorecard agreement",
            "selection_basis": "old+third internal zero-error release, externally observed",
            "review_policy": "v161_final_review_or_reject",
            "model": "two_signal_release_rules",
            "remaining_review_rate": float(v161_all["review_rate"]),
            "auto_action_rate": float(v161_all["auto_pass_rate"]),
            "balanced_accuracy": float(v161_all["balanced_accuracy"]),
            "fn": int(v161_all["fn"]),
            "fp": int(v161_all["fp"]),
            "rescued_n": int(v161_all.get("newly_released_from_review_n", 0)),
            "hurt_n": int(v161_all.get("released_error_n", 0)),
            "auto_error_n": int(v161_all.get("released_error_n", 0)),
            "claim_status": "Tier 2 efficiency candidate",
            "interpretation": "能显著降复核，当前数据安全，但 nested/前瞻证据仍需保守。",
        },
        {
            "module": "v173 image-only review corrector",
            "action_type": "high-confidence image-based auto correction/release",
            "selection_basis": "internal OOF confidence threshold",
            "review_policy": str(v173["review_policy"]),
            "model": str(v173["model"]),
            "remaining_review_rate": float(v173["remaining_review_rate"]),
            "auto_action_rate": float(v173["auto_correct_rate"]),
            "balanced_accuracy": float(v173["balanced_accuracy"]),
            "fn": int(v173["fn"]),
            "fp": int(v173["fp"]),
            "rescued_n": int(v173["rescued_n"]),
            "hurt_n": int(v173["hurt_n"]),
            "auto_error_n": int(v173["auto_correct_error_n"]),
            "claim_status": "review-burden reduction candidate",
            "interpretation": "能大幅降复核，但主要释放原本正确病例，不能称为成熟自动翻转纠偏。",
        },
        {
            "module": "v174 agreement release",
            "action_type": "release only when corrector agrees with base",
            "selection_basis": "internal zero auto-error",
            "review_policy": str(v174_release["review_policy"]),
            "model": str(v174_release["model"]),
            "remaining_review_rate": float(v174_release["remaining_review_rate"]),
            "auto_action_rate": float(v174_release["auto_action_rate"]),
            "balanced_accuracy": float(v174_release["balanced_accuracy"]),
            "fn": int(v174_release["fn"]),
            "fp": int(v174_release["fp"]),
            "rescued_n": int(v174_release["rescued_n"]),
            "hurt_n": int(v174_release["hurt_n"]),
            "auto_error_n": int(v174_release["auto_action_error_n"]),
            "claim_status": "safe but inefficient",
            "interpretation": "零自动错误但释放量较小，证明 agreement release 比 flip 更安全。",
        },
        {
            "module": "v174 disagreement flip",
            "action_type": "flip when corrector disagrees with base",
            "selection_basis": "best internal net rescue",
            "review_policy": str(v174_flip["review_policy"]),
            "model": str(v174_flip["model"]),
            "remaining_review_rate": float(v174_flip["remaining_review_rate"]),
            "auto_action_rate": float(v174_flip["auto_action_rate"]),
            "balanced_accuracy": float(v174_flip["balanced_accuracy"]),
            "fn": int(v174_flip["fn"]),
            "fp": int(v174_flip["fp"]),
            "rescued_n": int(v174_flip["rescued_n"]),
            "hurt_n": int(v174_flip["hurt_n"]),
            "auto_error_n": int(v174_flip["auto_action_error_n"]),
            "claim_status": "not supported",
            "interpretation": "救回少量病例但误伤极多，不能作为自动纠偏方案。",
        },
        {
            "module": "v175 error-enriched flip-risk",
            "action_type": "flip-risk learned from historical candidate errors",
            "selection_basis": "old+third OOF enriched error training",
            "review_policy": str(v175["review_policy"]),
            "model": str(v175["model"]),
            "remaining_review_rate": float(v175["remaining_review_rate"]),
            "auto_action_rate": float(v175["auto_flip_rate"]),
            "balanced_accuracy": float(v175["balanced_accuracy"]),
            "fn": int(v175["fn"]),
            "fp": int(v175["fp"]),
            "rescued_n": int(v175["rescued_n"]),
            "hurt_n": int(v175["hurt_n"]),
            "auto_error_n": int(v175["auto_flip_error_n"]),
            "claim_status": "negative evidence",
            "interpretation": "历史弱模型错误无法迁移为当前强模型的可靠翻转器。",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "v176_autocorrection_feasibility_matrix.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v176 Autocorrection Feasibility Matrix",
        "",
        "## Core Conclusion",
        "",
        "目前证据支持的是“高安全放行 + 拒识/复核”框架，而不是“成熟自动翻转纠偏器”。v173 能显著降低复核率，但主要释放主模型本来就正确的病例；v174/v175 证明直接自动翻转会严重误伤。论文里应把自动翻转写成仍需更多错误样本或医生结构化标签的后续方向。",
        "",
        "## Main Rows",
        "",
    ]
    for row in rows:
        md.append(
            f"- {row['module']}: BAcc {pct(row['balanced_accuracy'])}, remaining review {pct(row['remaining_review_rate'])}, "
            f"rescued {row['rescued_n']}, hurt {row['hurt_n']}. {row['interpretation']}"
        )
    md += [
        "",
        "## Recommended Writing Boundary",
        "",
        "- 可以强写：direction/risk-aware review routing、two-signal safety scorecard、safe-release efficiency frontier、跨域风险边界。",
        "- 可以候选写：image-only review burden reduction / high-confidence release。",
        "- 不应强写：自动翻转纠偏已经成熟或可以替代医生复核。",
    ]
    (OUT_DIR / "v176_autocorrection_feasibility_matrix.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "row_count": int(len(table)),
        "strong_supported_modules": ["v118 high-safety scorecard", "v161 safe-release"],
        "candidate_modules": ["v173 image-only review corrector", "v174 agreement release"],
        "unsupported_modules": ["v174 disagreement flip", "v175 error-enriched flip-risk"],
    }
    (OUT_DIR / "v176_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v176] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
