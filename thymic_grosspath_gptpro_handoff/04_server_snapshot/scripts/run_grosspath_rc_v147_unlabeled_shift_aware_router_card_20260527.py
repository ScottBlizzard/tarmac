from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import BUDGETS, ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527"
V145_LONG = ROOT / "outputs" / "grosspath_rc_v145_fusion_router_20260527" / "v145_fusion_router_scores_long.csv"
V146_CURVE = ROOT / "outputs" / "grosspath_rc_v146_domain_split_router_selection_20260527" / "v146_domain_split_router_budget_curve.csv"
V143_CASES = ROOT / "outputs" / "grosspath_rc_v143_image_feature_error_router_20260527" / "v143_image_feature_case_risks.csv"
V77_AUDIT = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527" / "v77_unlabeled_batch_shift_audit.csv"

TARGET_BUDGET = 0.30
DOMAINS = ["old_data", "third_batch", "strict_external"]


def load_scores() -> pd.DataFrame:
    scores = pd.read_csv(V145_LONG, dtype={"case_id": str})
    meta = pd.read_csv(V143_CASES, dtype={"case_id": str})
    meta = meta[["scope", "base_model", "case_id", "domain", "task_l6_label"]].drop_duplicates()
    out = scores.merge(meta, on=["scope", "base_model", "case_id"], how="left", validate="many_to_one")
    out["eval_domain"] = out["domain"].fillna(out["scope"])
    out.loc[out["scope"].eq("strict_external_locked"), "eval_domain"] = "strict_external"
    return out


def select_dev_stable_router(curve: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    b = curve.loc[np.isclose(curve["review_budget"].astype(float), TARGET_BUDGET)].copy()
    dev = b.loc[b["eval_domain"].isin(["old_data", "third_batch"])].copy()
    piv = dev.pivot_table(
        index=["base_model", "router"],
        columns="eval_domain",
        values="system_if_review_corrected_balanced_accuracy",
        aggfunc="first",
    ).dropna()
    piv["min_old_third_bacc"] = piv.min(axis=1)
    piv["mean_old_third_bacc"] = piv[["old_data", "third_batch"]].mean(axis=1)
    piv["old_third_gap"] = (piv["old_data"] - piv["third_batch"]).abs()
    ranked = piv.reset_index().sort_values(
        ["min_old_third_bacc", "mean_old_third_bacc", "old_third_gap"],
        ascending=[False, False, True],
    )
    top = ranked.iloc[0]
    return str(top["base_model"]), str(top["router"]), ranked


def select_low_conf_baseline(curve: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    b = curve.loc[np.isclose(curve["review_budget"].astype(float), TARGET_BUDGET)].copy()
    low = b.loc[b["router"].eq("single:low_conf") & b["eval_domain"].isin(["old_data", "third_batch"])].copy()
    piv = low.pivot_table(
        index=["base_model", "router"],
        columns="eval_domain",
        values="system_if_review_corrected_balanced_accuracy",
        aggfunc="first",
    ).dropna()
    piv["min_old_third_bacc"] = piv.min(axis=1)
    piv["mean_old_third_bacc"] = piv[["old_data", "third_batch"]].mean(axis=1)
    ranked = piv.reset_index().sort_values(["min_old_third_bacc", "mean_old_third_bacc"], ascending=False)
    top = ranked.iloc[0]
    return str(top["base_model"]), str(top["router"]), ranked


def audit_category_map() -> dict[str, str]:
    audit = pd.read_csv(V77_AUDIT)
    out: dict[str, str] = {}
    for _, row in audit.iterrows():
        target = str(row["target"])
        if target in DOMAINS:
            out[target] = str(row["shift_category"])
    return out


def policy_specs(dev_base: str, dev_router: str, low_base: str, low_router: str) -> dict[str, dict[str, tuple[str, str]]]:
    return {
        "baseline_low_conf_dev_selected": {d: (low_base, low_router) for d in DOMAINS},
        "dev_stable_router_all_domains": {d: (dev_base, dev_router) for d in DOMAINS},
        "shift_aware_image_directional_candidate": {
            "old_data": (dev_base, dev_router),
            "third_batch": (dev_base, dev_router),
            "strict_external": ("robust_prob", "single:v143_pca_directional"),
        },
        "shift_aware_concept_directional_candidate": {
            "old_data": (dev_base, dev_router),
            "third_batch": (dev_base, dev_router),
            "strict_external": ("prob_mean_core", "single:v144_concept_directional"),
        },
        "shift_aware_rank_fusion_candidate": {
            "old_data": (dev_base, dev_router),
            "third_batch": (dev_base, dev_router),
            "strict_external": ("prob_mean_core", "fusion:rank_mean_lowconf_v143_v144"),
        },
    }


def get_policy_cases(scores: pd.DataFrame, mapping: dict[str, tuple[str, str]]) -> pd.DataFrame:
    parts = []
    for domain, (base_model, router) in mapping.items():
        sub = scores.loc[
            scores["eval_domain"].eq(domain)
            & scores["base_model"].eq(base_model)
            & scores["router"].eq(router)
        ].copy()
        if sub.empty:
            raise ValueError(f"missing cases for {domain=} {base_model=} {router=}")
        sub["selected_base_model"] = base_model
        sub["selected_router"] = router
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def evaluate_policy_cases(policy_name: str, cases: pd.DataFrame, budget: float) -> tuple[list[dict[str, object]], pd.DataFrame]:
    all_case_rows = []
    summary_rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        sub = cases.loc[cases["eval_domain"].eq(domain)].copy()
        y = sub["label_idx"].astype(int).to_numpy()
        base_pred = sub["base_pred"].astype(int).to_numpy()
        base_prob = sub["base_prob"].to_numpy(float)
        risk = sub["risk_score"].to_numpy(float)
        base_wrong = base_pred != y
        order = np.argsort(-risk)
        review = np.zeros(len(sub), dtype=bool)
        k = int(round(float(budget) * len(sub)))
        if k > 0:
            review[order[:k]] = True
        final_pred = base_pred.copy()
        final_pred[review] = y[review]
        sub["policy"] = policy_name
        sub["review_budget"] = budget
        sub["review"] = review
        sub["auto"] = ~review
        sub["base_wrong"] = base_wrong
        sub["system_pred"] = final_pred
        all_case_rows.append(sub)
        row = {
            "policy": policy_name,
            "eval_domain": domain,
            "review_budget": float(budget),
            "selected_base_model": str(sub["selected_base_model"].iloc[0]),
            "selected_router": str(sub["selected_router"].iloc[0]),
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "auto_n": int((~review).sum()),
            "auto_rate": float((~review).mean()),
            "captured_errors": int((review & base_wrong).sum()),
            "total_base_errors": int(base_wrong.sum()),
            "error_capture_rate": float((review & base_wrong).sum() / max(1, base_wrong.sum())),
            "review_clean_n": int((review & ~base_wrong).sum()),
        }
        row.update({f"auto_{k2}": v for k2, v in metrics(y[~review], base_pred[~review], base_prob[~review]).items()})
        row.update({f"system_{k2}": v for k2, v in metrics(y, final_pred, base_prob).items()})
        summary_rows.append(row)
    all_cases = pd.concat(all_case_rows, ignore_index=True)
    y_all = all_cases["label_idx"].astype(int).to_numpy()
    pred_all = all_cases["base_pred"].astype(int).to_numpy()
    prob_all = all_cases["base_prob"].to_numpy(float)
    review_all = all_cases["review"].astype(bool).to_numpy()
    system_all = all_cases["system_pred"].astype(int).to_numpy()
    base_wrong_all = pred_all != y_all
    row_all = {
        "policy": policy_name,
        "eval_domain": "all_three_domains",
        "review_budget": float(budget),
        "selected_base_model": "domain_specific",
        "selected_router": "domain_specific",
        "review_n": int(review_all.sum()),
        "review_rate": float(review_all.mean()),
        "auto_n": int((~review_all).sum()),
        "auto_rate": float((~review_all).mean()),
        "captured_errors": int((review_all & base_wrong_all).sum()),
        "total_base_errors": int(base_wrong_all.sum()),
        "error_capture_rate": float((review_all & base_wrong_all).sum() / max(1, base_wrong_all.sum())),
        "review_clean_n": int((review_all & ~base_wrong_all).sum()),
    }
    row_all.update({f"auto_{k2}": v for k2, v in metrics(y_all[~review_all], pred_all[~review_all], prob_all[~review_all]).items()})
    row_all.update({f"system_{k2}": v for k2, v in metrics(y_all, system_all, prob_all).items()})
    summary_rows.append(row_all)
    return summary_rows, all_cases


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    curve = pd.read_csv(V146_CURVE)
    dev_base, dev_router, dev_rank = select_dev_stable_router(curve)
    low_base, low_router, low_rank = select_low_conf_baseline(curve)
    shift_categories = audit_category_map()
    specs = policy_specs(dev_base, dev_router, low_base, low_router)

    all_rows = []
    all_cases = []
    for policy_name, mapping in specs.items():
        cases = get_policy_cases(scores, mapping)
        for budget in BUDGETS:
            rows, case_rows = evaluate_policy_cases(policy_name, cases, float(budget))
            all_rows.extend(rows)
            all_cases.append(case_rows)

    summary = pd.DataFrame(all_rows).sort_values(["review_budget", "eval_domain", "system_balanced_accuracy"], ascending=[True, True, False])
    cases = pd.concat(all_cases, ignore_index=True)
    dev_rank.to_csv(OUT_DIR / "v147_internal_old_third_stable_router_rank.csv", index=False, encoding="utf-8-sig")
    low_rank.to_csv(OUT_DIR / "v147_internal_low_conf_baseline_rank.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v147_shift_aware_policy_budget_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v147_shift_aware_policy_cases.csv", index=False, encoding="utf-8-sig")
    card_rows = []
    for policy, mapping in specs.items():
        for domain, (base, router) in mapping.items():
            card_rows.append(
                {
                    "policy": policy,
                    "eval_domain": domain,
                    "unlabeled_shift_category": shift_categories.get(domain, ""),
                    "selected_base_model": base,
                    "selected_router": router,
                    "selection_note": (
                        "selected from old/third internal stability"
                        if router == dev_router and base == dev_base
                        else "baseline low confidence selected from old/third"
                        if router == low_router and base == low_base
                        else "method-prior severe-shift candidate; requires prospective validation"
                    ),
                }
            )
    card = pd.DataFrame(card_rows)
    card.to_csv(OUT_DIR / "v147_unlabeled_shift_policy_card.csv", index=False, encoding="utf-8-sig")
    report = {
        "target_budget": TARGET_BUDGET,
        "dev_stable_selected": {"base_model": dev_base, "router": dev_router},
        "low_conf_selected": {"base_model": low_base, "router": low_router},
        "shift_categories": shift_categories,
        "boundary": "Only old/third internal stability is used for data-driven router selection; severe-shift branches are marked as method-prior candidates and not final locked policies.",
    }
    (OUT_DIR / "v147_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v147] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
