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


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v163_safe_release_leave_domain_validation_20260527"


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


def candidate_thresholds(sub: pd.DataFrame, pred_side: int) -> tuple[list[float], list[float]]:
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    if pred_side == 1:
        quantiles = [1.0 - q for q in quantiles]
    core = sorted(set(float(x) for x in np.nanquantile(sub["prob_mean_core"].dropna(), quantiles)))
    wc = sorted(set(float(x) for x in np.nanquantile(sub["wholecrop_prob"].dropna(), quantiles)))
    return core, wc


def scan_train(df: pd.DataFrame, train_mask: pd.Series, pred_side: int) -> pd.DataFrame:
    train_review = df["v118_review_or_control"] & train_mask & df["final_pred"].eq(pred_side)
    sub = df.loc[train_review].copy()
    if sub.empty:
        return pd.DataFrame()
    core_grid, wc_grid = candidate_thresholds(sub, pred_side)
    rows = []
    for core, wc, agree_min in product(core_grid, wc_grid, [2, 3]):
        mask = rule_mask(df, pred_side, core, wc, agree_min)
        train_release = mask & train_mask
        train_errors = int((train_release & df["base_wrong"]).sum())
        if train_errors != 0 or int(train_release.sum()) == 0:
            continue
        rows.append(
            {
                "pred_side": pred_side,
                "rule": (
                    f"pred={pred_side}, "
                    f"prob_mean_core {'<=' if pred_side == 0 else '>='} {core:.6f}, "
                    f"wholecrop_prob {'<=' if pred_side == 0 else '>='} {wc:.6f}, "
                    f"core_agree_count>={agree_min}"
                ),
                "prob_mean_core_threshold": float(core),
                "wholecrop_prob_threshold": float(wc),
                "core_agree_min": int(agree_min),
                "train_release_n": int(train_release.sum()),
                "train_released_errors": train_errors,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["train_release_n", "core_agree_min"], ascending=[False, True])


def select_split_rules(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    selected = []
    for pred_side in [0, 1]:
        scan = scan_train(df, train_mask, pred_side)
        if not scan.empty:
            selected.append(scan.iloc[0])
    return pd.DataFrame(selected)


def apply_rules(df: pd.DataFrame, rules: pd.DataFrame) -> pd.Series:
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


def summarize(df: pd.DataFrame, release: pd.Series, split_name: str, train_domains: list[str]) -> list[dict[str, object]]:
    rows = []
    final_review = df["v118_review_or_control"] & ~release
    system_pred = df["final_pred"].copy()
    system_pred.loc[final_review] = df.loc[final_review, "label_idx"]
    for scope in ["old_data", "third_batch", "strict_external", "all_domains"]:
        mask = df["domain"].isin(["old_data", "third_batch", "strict_external"]) if scope == "all_domains" else df["domain"].eq(scope)
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
                "split_name": split_name,
                "train_domains": ",".join(train_domains),
                "scope": scope,
                "is_train_scope": scope in train_domains,
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
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    split_specs = [
        ("train_old_apply_third_external", ["old_data"]),
        ("train_third_apply_old_external", ["third_batch"]),
        ("train_old_third_apply_external", ["old_data", "third_batch"]),
    ]
    all_rules = []
    all_rows = []
    for split_name, train_domains in split_specs:
        train_mask = df["domain"].isin(train_domains)
        rules = select_split_rules(df, train_mask)
        rules.insert(0, "split_name", split_name)
        rules.insert(1, "train_domains", ",".join(train_domains))
        all_rules.append(rules)
        release = apply_rules(df, rules)
        all_rows.extend(summarize(df, release, split_name, train_domains))

    rules_df = pd.concat(all_rules, ignore_index=True)
    summary = pd.DataFrame(all_rows)
    rules_df.to_csv(OUT_DIR / "v163_leave_domain_selected_release_rules.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v163_leave_domain_release_summary.csv", index=False, encoding="utf-8-sig")

    test_rows = summary.loc[~summary["is_train_scope"] & summary["scope"].isin(["old_data", "third_batch", "strict_external"])].copy()
    total_test_release = int(test_rows["release_n"].sum())
    total_test_errors = int(test_rows["released_error_n"].sum())
    old_third = summary.loc[
        summary["split_name"].isin(["train_old_apply_third_external", "train_third_apply_old_external"])
        & ~summary["is_train_scope"]
        & summary["scope"].isin(["old_data", "third_batch"])
    ].copy()
    cross_internal_errors = int(old_third["released_error_n"].sum())
    cross_internal_release = int(old_third["release_n"].sum())
    full = summary.loc[summary["split_name"].eq("train_old_third_apply_external") & summary["scope"].eq("all_domains")].iloc[0]

    md = [
        "# v163 Safe-release Leave-domain Validation",
        "",
        "## Main Findings",
        "",
        (
            f"- Leave-domain test releases {total_test_release} reviewed cases outside each training scope, "
            f"with {total_test_errors} released errors observed."
        ),
        (
            f"- Cross internal validation old<->third releases {cross_internal_release} reviewed cases, "
            f"with {cross_internal_errors} released errors."
        ),
        (
            f"- The old+third selected rule set reproduces the v161 candidate: review/reject {pct(full['review_rate'])}, "
            f"auto-pass {pct(full['auto_pass_rate'])}, BAcc {pct(full['balanced_accuracy'])}, FN={int(full['fn'])}, FP={int(full['fp'])}."
        ),
        "",
        "## Interpretation",
        "",
        "This is stronger than a single internal fit because the release rule is re-selected on one internal domain and tested on the other. If released-error counts remain zero, the safe-release module can be reported as a stability-supported candidate; it still requires prospective validation before being treated as a locked threshold.",
    ]
    (OUT_DIR / "v163_safe_release_leave_domain_validation.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "split_count": int(len(split_specs)),
        "total_test_release_n": total_test_release,
        "total_test_released_error_n": total_test_errors,
        "cross_internal_release_n": cross_internal_release,
        "cross_internal_released_error_n": cross_internal_errors,
        "old_third_selected_all_domain_review_rate": float(full["review_rate"]),
        "old_third_selected_all_domain_bacc": float(full["balanced_accuracy"]),
        "old_third_selected_fn": int(full["fn"]),
        "old_third_selected_fp": int(full["fp"]),
    }
    (OUT_DIR / "v163_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v163] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
