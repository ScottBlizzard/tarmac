from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v148_router_card_comparative_evidence_20260527"
V147_CASES = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_cases.csv"
V147_SUMMARY = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_budget_summary.csv"

TARGET_BUDGET = 0.30
BASELINE_POLICY = "baseline_low_conf_dev_selected"
COMPARE_POLICIES = [
    "dev_stable_router_all_domains",
    "shift_aware_image_directional_candidate",
    "shift_aware_concept_directional_candidate",
    "shift_aware_rank_fusion_candidate",
]
DOMAINS = ["old_data", "third_batch", "strict_external", "all_three_domains"]


def system_bacc(y: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(balanced_accuracy_score(y, pred))


def bootstrap_diff(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 800) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260527)
    n = len(y)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = system_bacc(y[idx], pred_a[idx]) - system_bacc(y[idx], pred_b[idx])
    return float(np.nanmean(diffs)), float(np.nanquantile(diffs, 0.025)), float(np.nanquantile(diffs, 0.975))


def load_cases() -> pd.DataFrame:
    df = pd.read_csv(V147_CASES, dtype={"case_id": str})
    return df.loc[np.isclose(df["review_budget"].astype(float), TARGET_BUDGET)].copy()


def make_domain_cases(df: pd.DataFrame, policy: str, domain: str) -> pd.DataFrame:
    sub = df.loc[df["policy"].eq(policy)].copy()
    if domain != "all_three_domains":
        sub = sub.loc[sub["eval_domain"].eq(domain)].copy()
    return sub.sort_values(["eval_domain", "case_id"]).reset_index(drop=True)


def compare_policy(df: pd.DataFrame, policy: str, domain: str) -> dict[str, object]:
    base = make_domain_cases(df, BASELINE_POLICY, domain)
    comp = make_domain_cases(df, policy, domain)
    merged = base[
        ["eval_domain", "case_id", "label_idx", "system_pred", "review", "base_wrong"]
    ].merge(
        comp[["eval_domain", "case_id", "system_pred", "review", "base_wrong"]],
        on=["eval_domain", "case_id"],
        how="inner",
        suffixes=("_baseline", "_policy"),
        validate="one_to_one",
    )
    y = merged["label_idx"].astype(int).to_numpy()
    pred_base = merged["system_pred_baseline"].astype(int).to_numpy()
    pred_policy = merged["system_pred_policy"].astype(int).to_numpy()
    correct_base = pred_base == y
    correct_policy = pred_policy == y
    bmean, blo, bhi = bootstrap_diff(y, pred_policy, pred_base)
    review_base = merged["review_baseline"].astype(bool).to_numpy()
    review_policy = merged["review_policy"].astype(bool).to_numpy()
    return {
        "domain": domain,
        "policy": policy,
        "baseline_policy": BASELINE_POLICY,
        "n": int(len(merged)),
        "policy_system_accuracy": float(accuracy_score(y, pred_policy)),
        "baseline_system_accuracy": float(accuracy_score(y, pred_base)),
        "delta_accuracy": float(accuracy_score(y, pred_policy) - accuracy_score(y, pred_base)),
        "policy_system_bacc": system_bacc(y, pred_policy),
        "baseline_system_bacc": system_bacc(y, pred_base),
        "delta_bacc": float(system_bacc(y, pred_policy) - system_bacc(y, pred_base)),
        "bootstrap_delta_bacc_mean": bmean,
        "bootstrap_delta_bacc_ci_low": blo,
        "bootstrap_delta_bacc_ci_high": bhi,
        "policy_errors": int((~correct_policy).sum()),
        "baseline_errors": int((~correct_base).sum()),
        "errors_reduced": int((~correct_base).sum() - (~correct_policy).sum()),
        "policy_correct_baseline_wrong": int((correct_policy & ~correct_base).sum()),
        "policy_wrong_baseline_correct": int((~correct_policy & correct_base).sum()),
        "policy_review_n": int(review_policy.sum()),
        "baseline_review_n": int(review_base.sum()),
        "review_overlap_n": int((review_policy & review_base).sum()),
        "policy_extra_review_n": int((review_policy & ~review_base).sum()),
        "policy_less_review_n": int((~review_policy & review_base).sum()),
    }


def random_review_baseline(df: pd.DataFrame, baseline_summary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260527)
    rows = []
    for domain in DOMAINS:
        base_cases = make_domain_cases(df, BASELINE_POLICY, domain)
        y = base_cases["label_idx"].astype(int).to_numpy()
        pred = base_cases["base_pred"].astype(int).to_numpy()
        prob = base_cases["base_prob"].to_numpy(float)
        n = len(base_cases)
        review_n = int(round(TARGET_BUDGET * n))
        sims = []
        for _ in range(1000):
            review = np.zeros(n, dtype=bool)
            if review_n > 0:
                review[rng.choice(n, size=review_n, replace=False)] = True
            final = pred.copy()
            final[review] = y[review]
            m = metrics(y, final, prob)
            sims.append(
                {
                    "system_bacc": m["balanced_accuracy"],
                    "system_accuracy": m["accuracy"],
                    "captured_errors": int((review & (pred != y)).sum()),
                    "system_fn": m["fn"],
                    "system_fp": m["fp"],
                }
            )
        sims_df = pd.DataFrame(sims)
        rows.append(
            {
                "domain": domain,
                "n": int(n),
                "random_review_n": review_n,
                "random_system_bacc_mean": float(sims_df["system_bacc"].mean()),
                "random_system_bacc_p025": float(sims_df["system_bacc"].quantile(0.025)),
                "random_system_bacc_p975": float(sims_df["system_bacc"].quantile(0.975)),
                "random_captured_errors_mean": float(sims_df["captured_errors"].mean()),
                "random_fn_mean": float(sims_df["system_fn"].mean()),
                "random_fp_mean": float(sims_df["system_fp"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    rows = []
    for domain in DOMAINS:
        for policy in COMPARE_POLICIES:
            rows.append(compare_policy(cases, policy, domain))
    comp = pd.DataFrame(rows).sort_values(["domain", "delta_bacc"], ascending=[True, False])
    comp.to_csv(OUT_DIR / "v148_policy_vs_low_conf_paired_comparison.csv", index=False, encoding="utf-8-sig")

    summary = pd.read_csv(V147_SUMMARY)
    rand = random_review_baseline(cases, summary)
    rand.to_csv(OUT_DIR / "v148_random_review_baseline.csv", index=False, encoding="utf-8-sig")

    report = {
        "target_budget": TARGET_BUDGET,
        "baseline_policy": BASELINE_POLICY,
        "compared_policies": COMPARE_POLICIES,
        "main_readout": "paired system BAcc delta vs low-confidence baseline at matched per-domain review budget",
    }
    (OUT_DIR / "v148_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v148] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
