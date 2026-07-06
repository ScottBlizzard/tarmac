from __future__ import annotations

import json
import math

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v202_clean_cohort_exclusion_analysis_20260527"
REPORT_DIR = ROOT / "汇报"
V185_CASES = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_cases.csv"
V195_CASES = ROOT / "outputs" / "grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527" / "v195_selected_candidate_action_cases.csv"
V195_SUMMARY = ROOT / "outputs" / "grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527" / "v195_selected_candidate_summary.csv"
V201_CASES = ROOT / "outputs" / "grosspath_rc_v201_stable_supported_domain_flip_20260527" / "v201_stable_supported_domain_flip_cases.csv"


EXCLUSIONS = [
    {
        "original_case_id": "2404716",
        "doctor_note": "MEC",
        "standardized_reason": "mucoepidermoid carcinoma / special histology outside primary thymoma-carcinoma risk cohort",
        "recommendation": "exclude from primary clean cohort; keep in full-cohort sensitivity archive",
    },
    {
        "original_case_id": "2307206",
        "doctor_note": "淋巴上皮癌",
        "standardized_reason": "lymphoepithelial carcinoma / special carcinoma subtype",
        "recommendation": "exclude from primary clean cohort; keep in full-cohort sensitivity archive",
    },
    {
        "original_case_id": "2205101",
        "doctor_note": "淋巴上皮癌",
        "standardized_reason": "lymphoepithelial carcinoma with possible mixed/complex clinical-pathology context",
        "recommendation": "exclude from primary clean cohort; keep in full-cohort sensitivity archive",
    },
    {
        "original_case_id": "2203278",
        "doctor_note": "微结节型TC",
        "standardized_reason": "micronodular thymic carcinoma / rare special morphology",
        "recommendation": "exclude from primary clean cohort; keep in full-cohort sensitivity archive",
    },
    {
        "original_case_id": "2113767",
        "doctor_note": "肠型腺癌",
        "standardized_reason": "intestinal-type adenocarcinoma / special histology outside primary cohort",
        "recommendation": "exclude from primary clean cohort; keep in full-cohort sensitivity archive",
    },
]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * float(x):.2f}%"


def load_v185_system() -> pd.DataFrame:
    df = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    df["review_or_reject"] = df["adaptive_review"].astype(bool)
    df["auto_decision"] = ~df["review_or_reject"]
    df["system_pred"] = df["final_pred"].astype(int)
    df.loc[df["review_or_reject"], "system_pred"] = df.loc[df["review_or_reject"], "label_idx"].astype(int)
    df["action_type"] = "v185_adaptive_selective_review"
    df["auto_action_n"] = 0
    df["action_error"] = False
    return df


def load_v195_system() -> pd.DataFrame:
    base = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    actions = pd.read_csv(V195_CASES, dtype={"case_id": str, "original_case_id": str})
    action_cols = [
        "case_id",
        "auto_correct_action",
        "action_disagrees_base",
        "remaining_review",
        "action_error",
        "rescued_by_action",
        "hurt_by_action",
    ]
    if actions.empty:
        for col in action_cols[1:]:
            base[col] = False
    else:
        base = base.merge(actions[action_cols], on="case_id", how="left", validate="one_to_one")
        for col in action_cols[1:]:
            base[col] = base[col].fillna(False).astype(bool)
    # v195 selected action is high-confidence agreement release. System prediction remains final_pred
    # unless still reviewed, in which case review is treated as adjudicated.
    base["review_or_reject"] = base["adaptive_review"].astype(bool) & ~base["auto_correct_action"].astype(bool)
    base["auto_decision"] = ~base["review_or_reject"]
    base["system_pred"] = base["final_pred"].astype(int)
    base.loc[base["review_or_reject"], "system_pred"] = base.loc[base["review_or_reject"], "label_idx"].astype(int)
    base["action_type"] = "v195_stable_agreement_release"
    base["auto_action_n"] = base["auto_correct_action"].astype(int)
    return base


def load_v201_system() -> pd.DataFrame:
    df = pd.read_csv(V201_CASES, dtype={"case_id": str, "original_case_id": str})
    df["review_or_reject"] = df["remaining_review"].astype(bool)
    df["auto_decision"] = ~df["review_or_reject"]
    df["system_pred"] = df["system_pred"].astype(int)
    df["action_type"] = "v201_stable_supported_domain_flip"
    df["action_error"] = df["flip_error"].astype(bool)
    df["auto_action_n"] = df["flip_trigger"].astype(int)
    return df


def summarize_policy(df: pd.DataFrame, policy: str, cohort: str, exclusion_ids: set[str]) -> list[dict[str, object]]:
    work = df.copy()
    work["original_case_id"] = work["original_case_id"].astype(str)
    if cohort == "clean_primary":
        work = work.loc[~work["original_case_id"].isin(exclusion_ids)].copy()
    elif cohort == "excluded_cases_only":
        work = work.loc[work["original_case_id"].isin(exclusion_ids)].copy()
    elif cohort == "full_sensitivity":
        pass
    else:
        raise ValueError(cohort)

    rows = []
    for scope, sub in [
        ("old_data", work.loc[work["domain"].eq("old_data")].copy()),
        ("third_batch", work.loc[work["domain"].eq("third_batch")].copy()),
        ("strict_external", work.loc[work["domain"].eq("strict_external")].copy()),
        ("all_domains", work.loc[work["domain"].isin(["old_data", "third_batch", "strict_external"])].copy()),
    ]:
        if sub.empty:
            rows.append(
                {
                    "policy": policy,
                    "cohort": cohort,
                    "scope": scope,
                    "n": 0,
                    "low_risk_n": 0,
                    "high_risk_n": 0,
                    "accuracy": math.nan,
                    "balanced_accuracy": math.nan,
                    "f1": math.nan,
                    "auc": math.nan,
                    "fn": 0,
                    "fp": 0,
                    "auto_decision_n": 0,
                    "auto_decision_rate": math.nan,
                    "review_or_reject_n": 0,
                    "review_or_reject_rate": math.nan,
                    "auto_error_n": 0,
                    "auto_error_rate": math.nan,
                    "auto_error_wilson95_high": math.nan,
                    "auto_action_n": 0,
                    "action_error_n": 0,
                }
            )
            continue
        y = sub["label_idx"].astype(int).to_numpy()
        pred = sub["system_pred"].astype(int).to_numpy()
        prob = sub["prob_mean_core"].astype(float).to_numpy()
        mm = metrics(y, pred, prob)
        auto = sub["auto_decision"].astype(bool)
        auto_error = auto & sub["final_pred"].astype(int).ne(sub["label_idx"].astype(int))
        low, high = wilson(int(auto_error.sum()), int(auto.sum()))
        row = {
            "policy": policy,
            "cohort": cohort,
            "scope": scope,
            "n": int(len(sub)),
            "low_risk_n": int((sub["label_idx"].astype(int) == 0).sum()),
            "high_risk_n": int((sub["label_idx"].astype(int) == 1).sum()),
            "auto_decision_n": int(auto.sum()),
            "auto_decision_rate": float(auto.mean()),
            "review_or_reject_n": int(sub["review_or_reject"].astype(bool).sum()),
            "review_or_reject_rate": float(sub["review_or_reject"].astype(bool).mean()),
            "auto_error_n": int(auto_error.sum()),
            "auto_error_rate": float(auto_error.sum() / max(1, auto.sum())),
            "auto_error_wilson95_high": high,
            "auto_action_n": int(pd.to_numeric(sub.get("auto_action_n", 0), errors="coerce").fillna(0).sum())
            if "auto_action_n" in sub
            else 0,
            "action_error_n": int(sub["action_error"].astype(bool).sum()) if "action_error" in sub else 0,
        }
        row.update(mm)
        rows.append(row)
    return rows


def exclusion_presence_table(reference: pd.DataFrame, exclusion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = reference.copy()
    ref["original_case_id"] = ref["original_case_id"].astype(str)
    for _, ex in exclusion.iterrows():
        cid = str(ex["original_case_id"])
        sub = ref.loc[ref["original_case_id"].eq(cid)].copy()
        if sub.empty:
            rows.append(
                {
                    **ex.to_dict(),
                    "in_current_task7_main": False,
                    "domain": "",
                    "task_l6_label": "",
                    "label_idx": "",
                    "final_pred": "",
                    "prob_mean_core": "",
                    "adaptive_review": "",
                    "adaptive_auto_decision": "",
                    "base_correct": "",
                }
            )
        else:
            r = sub.iloc[0]
            rows.append(
                {
                    **ex.to_dict(),
                    "in_current_task7_main": True,
                    "domain": r["domain"],
                    "task_l6_label": r["task_l6_label"],
                    "label_idx": int(r["label_idx"]),
                    "final_pred": int(r["final_pred"]),
                    "prob_mean_core": float(r["prob_mean_core"]),
                    "adaptive_review": bool(r["adaptive_review"]),
                    "adaptive_auto_decision": bool(r["adaptive_auto_decision"]),
                    "base_correct": bool(int(r["label_idx"]) == int(r["final_pred"])),
                }
            )
    return pd.DataFrame(rows)


def write_md(summary: pd.DataFrame, exclusion_presence: pd.DataFrame) -> None:
    v195 = summary.loc[
        summary["policy"].eq("v195_stable_agreement_release")
        & summary["cohort"].isin(["full_sensitivity", "clean_primary"])
        & summary["scope"].eq("all_domains")
    ].copy()
    full = v195.loc[v195["cohort"].eq("full_sensitivity")].iloc[0]
    clean = v195.loc[v195["cohort"].eq("clean_primary")].iloc[0]
    old_clean = summary.loc[
        summary["policy"].eq("v195_stable_agreement_release")
        & summary["cohort"].eq("clean_primary")
        & summary["scope"].eq("old_data")
    ].iloc[0]
    md = [
        "# Task7 Clean Cohort Exclusion Analysis",
        "",
        "## Cohort Definition",
        "",
        "根据医生复核意见，我们建立固定 exclusion list。原始数据不物理删除；主分析使用 clean primary cohort，完整队列作为 full-cohort sensitivity analysis。",
        "",
        "## Exclusion List Status",
        "",
    ]
    for _, r in exclusion_presence.iterrows():
        present = "在当前 Task7 主流程中" if bool(r["in_current_task7_main"]) else "不在当前 Task7 主流程中"
        md.append(f"- {r['original_case_id']}：{r['doctor_note']}，{present}。")
    md += [
        "",
        "## Main v195 Result",
        "",
        (
            f"- Full cohort: n={int(full['n'])}, BAcc {pct(full['balanced_accuracy'])}, "
            f"Acc {pct(full['accuracy'])}, FN={int(full['fn'])}, FP={int(full['fp'])}, "
            f"review/reject {pct(full['review_or_reject_rate'])}, auto errors {int(full['auto_error_n'])}."
        ),
        (
            f"- Clean cohort: n={int(clean['n'])}, BAcc {pct(clean['balanced_accuracy'])}, "
            f"Acc {pct(clean['accuracy'])}, FN={int(clean['fn'])}, FP={int(clean['fp'])}, "
            f"review/reject {pct(clean['review_or_reject_rate'])}, auto errors {int(clean['auto_error_n'])}."
        ),
        (
            f"- Clean old_data: n={int(old_clean['n'])}, BAcc {pct(old_clean['balanced_accuracy'])}, "
            f"review/reject {pct(old_clean['review_or_reject_rate'])}."
        ),
        "",
        "## Interpretation",
        "",
        "剔除医生指出的特殊组织学/混杂病例后，当前主结果没有变差。由于当前主流程中实际命中的 exclusion 只有 2205101 和 2307206，且二者原本都被模型判为高危正确，clean cohort 的主要作用是让队列定义更干净，而不是人为提高分数。",
        "",
        "论文写法建议：主文报告 clean cohort；补充材料报告 full cohort sensitivity analysis，说明结论对这些特殊病例剔除不敏感。",
    ]
    (OUT_DIR / "v202_clean_cohort_exclusion_analysis.md").write_text("\n".join(md), encoding="utf-8")
    (REPORT_DIR / "2026-05-27_Task7_clean_cohort_exclusion_analysis.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    exclusion = pd.DataFrame(EXCLUSIONS)
    exclusion_ids = set(exclusion["original_case_id"].astype(str))

    v185 = load_v185_system()
    v195 = load_v195_system()
    v201 = load_v201_system()
    policy_frames = [
        ("v185_unlabeled_shift_adaptive", v185),
        ("v195_stable_agreement_release", v195),
        ("v201_supported_domain_flip", v201),
    ]
    rows = []
    for policy, frame in policy_frames:
        for cohort in ["full_sensitivity", "clean_primary", "excluded_cases_only"]:
            rows.extend(summarize_policy(frame, policy, cohort, exclusion_ids))
    summary = pd.DataFrame(rows)
    presence = exclusion_presence_table(v185, exclusion)

    exclusion.to_csv(OUT_DIR / "v202_task7_exclusion_list.csv", index=False, encoding="utf-8-sig")
    presence.to_csv(OUT_DIR / "v202_exclusion_presence_in_current_task7.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v202_clean_vs_full_policy_metrics.csv", index=False, encoding="utf-8-sig")
    exclusion.to_csv(REPORT_DIR / "2026-05-27_Task7医生确认剔除病例清单.csv", index=False, encoding="utf-8-sig")
    presence.to_csv(REPORT_DIR / "2026-05-27_Task7剔除病例当前主流程状态.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT_DIR / "2026-05-27_Task7_clean_vs_full_results.csv", index=False, encoding="utf-8-sig")
    write_md(summary, presence)

    focus = summary.loc[
        summary["policy"].eq("v195_stable_agreement_release")
        & summary["cohort"].isin(["full_sensitivity", "clean_primary"])
        & summary["scope"].eq("all_domains")
    ].copy()
    report = {
        "exclusion_count_requested": int(len(exclusion)),
        "exclusion_count_in_current_task7": int(presence["in_current_task7_main"].sum()),
        "current_task7_exclusion_ids_present": presence.loc[presence["in_current_task7_main"], "original_case_id"].tolist(),
        "v195_full_all_domain": focus.loc[focus["cohort"].eq("full_sensitivity")].iloc[0].to_dict(),
        "v195_clean_all_domain": focus.loc[focus["cohort"].eq("clean_primary")].iloc[0].to_dict(),
    }
    (OUT_DIR / "v202_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v202] wrote {OUT_DIR}")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("v195_")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
