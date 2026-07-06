from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import (
    V118_CASES,
    as_bool,
    pct,
    rule_mask,
)


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v164_stable_safe_release_strategy_compare_20260527"
V161_RULES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_selected_internal_zero_error_release_rules.csv"
V163_RULES = ROOT / "outputs" / "grosspath_rc_v163_safe_release_leave_domain_validation_20260527" / "v163_leave_domain_selected_release_rules.csv"


def load_cases() -> pd.DataFrame:
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v118_review_or_control", "v111_review_or_control", "v118_extra_review"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct", "core_agree_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    for col in ["prob_mean_core", "wholecrop_prob", "main_prob", "robust_prob"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    return df


def rule_row(strategy: str, selection_basis: str, pred_side: int, core: float, wholecrop: float, agree: int) -> dict[str, object]:
    return {
        "strategy": strategy,
        "selection_basis": selection_basis,
        "pred_side": int(pred_side),
        "rule": (
            f"pred={pred_side}, "
            f"prob_mean_core {'<=' if pred_side == 0 else '>='} {core:.6f}, "
            f"wholecrop_prob {'<=' if pred_side == 0 else '>='} {wholecrop:.6f}, "
            f"core_agree_count>={agree}"
        ),
        "prob_mean_core_threshold": float(core),
        "wholecrop_prob_threshold": float(wholecrop),
        "core_agree_min": int(agree),
    }


def load_rule_strategy(strategy: str, source: pd.DataFrame, selection_basis: str) -> list[dict[str, object]]:
    rows = []
    for _, r in source.iterrows():
        rows.append(
            rule_row(
                strategy,
                selection_basis,
                int(r["pred_side"]),
                float(r["prob_mean_core_threshold"]),
                float(r["wholecrop_prob_threshold"]),
                int(r["core_agree_min"]),
            )
        )
    return rows


def intersection_strategy(v163: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for pred_side in [0, 1]:
        old = v163.loc[v163["split_name"].eq("train_old_apply_third_external") & v163["pred_side"].astype(int).eq(pred_side)].iloc[0]
        third = v163.loc[v163["split_name"].eq("train_third_apply_old_external") & v163["pred_side"].astype(int).eq(pred_side)].iloc[0]
        if pred_side == 0:
            core = min(float(old["prob_mean_core_threshold"]), float(third["prob_mean_core_threshold"]))
            wc = min(float(old["wholecrop_prob_threshold"]), float(third["wholecrop_prob_threshold"]))
        else:
            core = max(float(old["prob_mean_core_threshold"]), float(third["prob_mean_core_threshold"]))
            wc = max(float(old["wholecrop_prob_threshold"]), float(third["wholecrop_prob_threshold"]))
        agree = max(int(old["core_agree_min"]), int(third["core_agree_min"]))
        rows.append(rule_row("leave_domain_intersection", "intersection_of_old_only_and_third_only_zero_error_rules", pred_side, core, wc, agree))
    return rows


def candidate_thresholds(sub: pd.DataFrame, pred_side: int) -> tuple[list[float], list[float]]:
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    if pred_side == 1:
        quantiles = [1.0 - q for q in quantiles]
    core = sorted(set(float(x) for x in np.nanquantile(sub["prob_mean_core"].dropna(), quantiles)))
    wc = sorted(set(float(x) for x in np.nanquantile(sub["wholecrop_prob"].dropna(), quantiles)))
    return core, wc


def scan_zero_error_candidates(df: pd.DataFrame, pred_side: int) -> pd.DataFrame:
    internal = df["domain"].isin(["old_data", "third_batch"])
    sub = df.loc[df["v118_review_or_control"] & internal & df["final_pred"].eq(pred_side)].copy()
    core_grid, wc_grid = candidate_thresholds(sub, pred_side)
    rows = []
    for core, wc, agree in product(core_grid, wc_grid, [2, 3]):
        mask = rule_mask(df, pred_side, core, wc, agree)
        old = mask & df["domain"].eq("old_data")
        third = mask & df["domain"].eq("third_batch")
        ext = mask & df["domain"].eq("strict_external")
        old_err = int((old & df["base_wrong"]).sum())
        third_err = int((third & df["base_wrong"]).sum())
        if old_err or third_err:
            continue
        rows.append(
            {
                "pred_side": pred_side,
                "prob_mean_core_threshold": float(core),
                "wholecrop_prob_threshold": float(wc),
                "core_agree_min": int(agree),
                "old_release_n": int(old.sum()),
                "third_release_n": int(third.sum()),
                "internal_release_n": int(old.sum() + third.sum()),
                "min_internal_domain_release_n": int(min(old.sum(), third.sum())),
                "external_release_n": int(ext.sum()),
                "external_released_error_n": int((ext & df["base_wrong"]).sum()),
                "all_release_n": int(mask.sum()),
                "all_released_error_n": int((mask & df["base_wrong"]).sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["min_internal_domain_release_n", "internal_release_n", "external_release_n"],
        ascending=[False, False, False],
    )


def balanced_strategy(df: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    scans = []
    rows = []
    for pred_side in [0, 1]:
        scan = scan_zero_error_candidates(df, pred_side)
        scan.insert(0, "strategy", "balanced_zero_error")
        scans.append(scan)
        best = scan.iloc[0]
        rows.append(
            rule_row(
                "balanced_zero_error",
                "maximize_min_old_third_release_under_zero_error_in_each_internal_domain",
                pred_side,
                float(best["prob_mean_core_threshold"]),
                float(best["wholecrop_prob_threshold"]),
                int(best["core_agree_min"]),
            )
        )
    return rows, pd.concat(scans, ignore_index=True)


def apply_strategy(df: pd.DataFrame, rules: pd.DataFrame) -> pd.Series:
    release = pd.Series(False, index=df.index)
    for _, r in rules.iterrows():
        release |= rule_mask(
            df,
            int(r["pred_side"]),
            float(r["prob_mean_core_threshold"]),
            float(r["wholecrop_prob_threshold"]),
            int(r["core_agree_min"]),
        )
    return release


def summarize(df: pd.DataFrame, strategy: str, release: pd.Series) -> list[dict[str, object]]:
    final_review = df["v118_review_or_control"] & ~release
    system_pred = df["final_pred"].copy()
    system_pred.loc[final_review] = df.loc[final_review, "label_idx"]
    rows = []
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("strict_external", df["domain"].eq("strict_external")),
        ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        sub = df.loc[mask].copy()
        y = sub["label_idx"].to_numpy(int)
        pred = system_pred.loc[mask].to_numpy(int)
        prob = sub["prob_mean_core"].fillna(sub["main_prob"]).to_numpy(float)
        rel = release.loc[mask].to_numpy(bool)
        review = final_review.loc[mask].to_numpy(bool)
        released_errors = rel & sub["base_wrong"].to_numpy(bool)
        m = metrics(y, pred, prob)
        rows.append(
            {
                "strategy": strategy,
                "scope": scope,
                "n": int(len(sub)),
                "release_n": int(rel.sum()),
                "release_rate": float(rel.mean()),
                "released_error_n": int(released_errors.sum()),
                "review_rate": float(review.mean()),
                "auto_pass_rate": float((~review).mean()),
                "remaining_error_n": int((pred != y).sum()),
                "fn": int(((y == 1) & (pred == 0)).sum()),
                "fp": int(((y == 0) & (pred == 1)).sum()),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "accuracy": float(m["accuracy"]),
                "f1": float(m["f1"]),
                "auc": float(m["auc"]),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    v161 = pd.read_csv(V161_RULES)
    v163 = pd.read_csv(V163_RULES)

    strategy_rules = []
    strategy_rules.extend(load_rule_strategy("pooled_old_third_zero_error_v161", v161, "old+third pooled zero-error max-release"))
    for split_name, strategy in [
        ("train_old_apply_third_external", "old_only_selected"),
        ("train_third_apply_old_external", "third_only_selected"),
    ]:
        source = v163.loc[v163["split_name"].eq(split_name)].copy()
        strategy_rules.extend(load_rule_strategy(strategy, source, split_name))
    strategy_rules.extend(intersection_strategy(v163))
    balanced_rows, balanced_scan = balanced_strategy(df)
    strategy_rules.extend(balanced_rows)

    rules = pd.DataFrame(strategy_rules)
    summaries = []
    case_rows = []
    for strategy, group in rules.groupby("strategy", sort=False):
        release = apply_strategy(df, group)
        summaries.extend(summarize(df, strategy, release))
        tmp = df[["domain", "case_id", "original_case_id", "task_l6_label", "label_idx", "final_pred", "base_wrong"]].copy()
        tmp["strategy"] = strategy
        tmp["release"] = release
        tmp["released_error"] = release & df["base_wrong"]
        case_rows.append(tmp)

    rules.to_csv(OUT_DIR / "v164_strategy_rules.csv", index=False, encoding="utf-8-sig")
    balanced_scan.to_csv(OUT_DIR / "v164_balanced_zero_error_candidate_scan.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "v164_strategy_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(case_rows, ignore_index=True).to_csv(OUT_DIR / "v164_strategy_release_cases.csv", index=False, encoding="utf-8-sig")

    focus = summary.loc[summary["scope"].eq("all_domains")].copy().sort_values(
        ["released_error_n", "review_rate", "balanced_accuracy"],
        ascending=[True, True, False],
    )
    focus.to_csv(OUT_DIR / "v164_all_domain_strategy_focus.csv", index=False, encoding="utf-8-sig")

    recommended = focus.loc[focus["released_error_n"].eq(0)].iloc[0]
    pooled = summary.loc[summary["strategy"].eq("pooled_old_third_zero_error_v161") & summary["scope"].eq("all_domains")].iloc[0]
    intersect = summary.loc[summary["strategy"].eq("leave_domain_intersection") & summary["scope"].eq("all_domains")].iloc[0]
    balanced = summary.loc[summary["strategy"].eq("balanced_zero_error") & summary["scope"].eq("all_domains")].iloc[0]
    old_only = summary.loc[summary["strategy"].eq("old_only_selected") & summary["scope"].eq("third_batch")].iloc[0]

    md = [
        "# v164 Stable Safe-release Strategy Comparison",
        "",
        "## Main Findings",
        "",
        (
            f"- Old-only selected release is unsafe across internal domains: applied to third_batch it releases "
            f"{int(old_only['release_n'])} cases with {int(old_only['released_error_n'])} released errors."
        ),
        (
            f"- v161 pooled old+third zero-error release: all-domain review/reject {pct(pooled['review_rate'])}, "
            f"BAcc {pct(pooled['balanced_accuracy'])}, released errors {int(pooled['released_error_n'])}."
        ),
        (
            f"- Leave-domain intersection release: all-domain review/reject {pct(intersect['review_rate'])}, "
            f"BAcc {pct(intersect['balanced_accuracy'])}, released errors {int(intersect['released_error_n'])}."
        ),
        (
            f"- Balanced zero-error release: all-domain review/reject {pct(balanced['review_rate'])}, "
            f"BAcc {pct(balanced['balanced_accuracy'])}, released errors {int(balanced['released_error_n'])}."
        ),
        (
            f"- Recommended current strategy by zero released error and lowest review burden: `{recommended['strategy']}` "
            f"with review/reject {pct(recommended['review_rate'])}, FN={int(recommended['fn'])}, FP={int(recommended['fp'])}."
        ),
        "",
        "## Interpretation",
        "",
        "The comparison separates three ideas: single-domain release can be too permissive, pooled old+third zero-error release is efficient on current data, and intersection/balanced variants provide more conservative alternatives. For paper writing, v161 can remain the strongest efficiency candidate, while v164 supplies the safety boundary and ablation against single-domain thresholding.",
    ]
    (OUT_DIR / "v164_stable_safe_release_strategy_compare.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "strategy_count": int(rules["strategy"].nunique()),
        "recommended_strategy": str(recommended["strategy"]),
        "recommended_review_rate": float(recommended["review_rate"]),
        "recommended_bacc": float(recommended["balanced_accuracy"]),
        "recommended_fn": int(recommended["fn"]),
        "recommended_fp": int(recommended["fp"]),
        "old_only_third_release_errors": int(old_only["released_error_n"]),
        "pooled_v161_review_rate": float(pooled["review_rate"]),
        "intersection_review_rate": float(intersect["review_rate"]),
        "balanced_review_rate": float(balanced["review_rate"]),
    }
    (OUT_DIR / "v164_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v164] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
