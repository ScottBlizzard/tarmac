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


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v167_nested_safe_release_validation_20260527"


def load_cases() -> pd.DataFrame:
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v118_review_or_control", "v111_review_or_control", "v118_extra_review"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct", "core_agree_count", "fold_id"]:
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


def scan_train_rules(df: pd.DataFrame, train_mask: pd.Series, pred_side: int) -> pd.DataFrame:
    sub = df.loc[df["v118_review_or_control"] & train_mask & df["final_pred"].eq(pred_side)].copy()
    if sub.empty:
        return pd.DataFrame()
    core_grid, wc_grid = candidate_thresholds(sub, pred_side)
    rows = []
    for core, wc, agree in product(core_grid, wc_grid, [2, 3]):
        mask = rule_mask(df, pred_side, core, wc, agree)
        old = mask & train_mask & df["domain"].eq("old_data")
        third = mask & train_mask & df["domain"].eq("third_batch")
        train_release = old | third
        old_err = int((old & df["base_wrong"]).sum())
        third_err = int((third & df["base_wrong"]).sum())
        if old_err or third_err or int(train_release.sum()) == 0:
            continue
        rows.append(
            {
                "pred_side": pred_side,
                "prob_mean_core_threshold": float(core),
                "wholecrop_prob_threshold": float(wc),
                "core_agree_min": int(agree),
                "old_train_release_n": int(old.sum()),
                "third_train_release_n": int(third.sum()),
                "train_release_n": int(train_release.sum()),
                "min_train_domain_release_n": int(min(old.sum(), third.sum())),
                "train_released_errors": 0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["min_train_domain_release_n", "train_release_n", "core_agree_min"],
        ascending=[False, False, True],
    )


def select_rules(df: pd.DataFrame, train_mask: pd.Series, fold_id: int) -> pd.DataFrame:
    rows = []
    for pred_side in [0, 1]:
        scan = scan_train_rules(df, train_mask, pred_side)
        scan.to_csv(OUT_DIR / f"v167_fold{fold_id}_pred{pred_side}_candidate_scan.csv", index=False, encoding="utf-8-sig")
        if scan.empty:
            continue
        best = scan.iloc[0].to_dict()
        best["fold_id"] = int(fold_id)
        best["rule"] = (
            f"pred={pred_side}, "
            f"prob_mean_core {'<=' if pred_side == 0 else '>='} {best['prob_mean_core_threshold']:.6f}, "
            f"wholecrop_prob {'<=' if pred_side == 0 else '>='} {best['wholecrop_prob_threshold']:.6f}, "
            f"core_agree_count>={int(best['core_agree_min'])}"
        )
        rows.append(best)
    return pd.DataFrame(rows)


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


def summarize_scope(df: pd.DataFrame, release: pd.Series, workflow: str, scope: str, mask: pd.Series) -> dict[str, object]:
    final_review = df["v118_review_or_control"] & ~release
    system_pred = df["final_pred"].copy()
    system_pred.loc[final_review] = df.loc[final_review, "label_idx"]
    sub = df.loc[mask].copy()
    y = sub["label_idx"].to_numpy(int)
    pred = system_pred.loc[mask].to_numpy(int)
    prob = sub["prob_mean_core"].fillna(sub["main_prob"]).to_numpy(float)
    rel = release.loc[mask].to_numpy(bool)
    review = final_review.loc[mask].to_numpy(bool)
    released_errors = rel & sub["base_wrong"].to_numpy(bool)
    m = metrics(y, pred, prob)
    return {
        "workflow": workflow,
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    internal = df["domain"].isin(["old_data", "third_batch"])
    folds = sorted(df.loc[internal, "fold_id"].unique().tolist())

    all_rules = []
    nested_release = pd.Series(False, index=df.index)
    fold_rows = []
    external_rows = []
    external_release_votes = pd.Series(0, index=df.index, dtype=int)

    for fold in folds:
        train_mask = internal & df["fold_id"].ne(fold)
        heldout_mask = internal & df["fold_id"].eq(fold)
        rules = select_rules(df, train_mask, int(fold))
        all_rules.append(rules)
        release_all = apply_rules(df, rules)
        nested_release.loc[heldout_mask] = release_all.loc[heldout_mask]
        external_mask = df["domain"].eq("strict_external")
        external_release_votes.loc[external_mask] += release_all.loc[external_mask].astype(int)

        fold_rows.append(
            summarize_scope(
                df,
                release_all & heldout_mask,
                f"nested_fold_{fold}",
                f"heldout_fold_{fold}",
                heldout_mask,
            )
        )
        external_rows.append(
            summarize_scope(
                df,
                release_all & external_mask,
                f"fold_{fold}_rules_on_external",
                "strict_external",
                external_mask,
            )
        )

    rules_df = pd.concat(all_rules, ignore_index=True)
    rules_df.to_csv(OUT_DIR / "v167_nested_selected_rules_by_fold.csv", index=False, encoding="utf-8-sig")

    nested_release_full = nested_release.copy()
    nested_summary_rows = fold_rows.copy()
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("internal_all", internal),
        ("all_domains_internal_nested_only", internal),
    ]:
        nested_summary_rows.append(summarize_scope(df, nested_release_full, "nested_internal_safe_release", scope, mask))

    # External is not part of nested selection. Consensus rows show how often fold-specific rules agree.
    external_mask = df["domain"].eq("strict_external")
    for vote_min in [1, 3, 5]:
        ext_release = pd.Series(False, index=df.index)
        ext_release.loc[external_mask] = external_release_votes.loc[external_mask] >= vote_min
        nested_summary_rows.append(
            summarize_scope(df, ext_release, f"external_consensus_vote_ge_{vote_min}", "strict_external", external_mask)
        )

    fold_summary = pd.DataFrame(fold_rows)
    external_fold_summary = pd.DataFrame(external_rows)
    summary = pd.DataFrame(nested_summary_rows)
    fold_summary.to_csv(OUT_DIR / "v167_nested_heldout_fold_summary.csv", index=False, encoding="utf-8-sig")
    external_fold_summary.to_csv(OUT_DIR / "v167_external_by_fold_rule_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v167_nested_safe_release_summary.csv", index=False, encoding="utf-8-sig")

    case_cols = [
        "domain",
        "fold_id",
        "case_id",
        "original_case_id",
        "task_l6_label",
        "label_idx",
        "final_pred",
        "base_wrong",
        "prob_mean_core",
        "wholecrop_prob",
        "main_prob",
        "core_agree_count",
    ]
    case_out = df[case_cols].copy()
    case_out["nested_internal_release"] = nested_release_full
    case_out["nested_internal_released_error"] = nested_release_full & df["base_wrong"]
    case_out["external_release_vote_n"] = external_release_votes
    case_out.to_csv(OUT_DIR / "v167_nested_safe_release_cases.csv", index=False, encoding="utf-8-sig")

    internal_row = summary.loc[
        summary["workflow"].eq("nested_internal_safe_release") & summary["scope"].eq("internal_all")
    ].iloc[0]
    old_row = summary.loc[summary["workflow"].eq("nested_internal_safe_release") & summary["scope"].eq("old_data")].iloc[0]
    third_row = summary.loc[summary["workflow"].eq("nested_internal_safe_release") & summary["scope"].eq("third_batch")].iloc[0]
    ext_vote5 = summary.loc[
        summary["workflow"].eq("external_consensus_vote_ge_5") & summary["scope"].eq("strict_external")
    ].iloc[0]

    md = [
        "# v167 Nested Safe-release Validation",
        "",
        "## Main Findings",
        "",
        (
            f"- Fold-wise nested internal validation releases {int(internal_row['release_n'])} reviewed cases "
            f"from old+third with {int(internal_row['released_error_n'])} released errors."
        ),
        (
            f"- Internal review/reject rate becomes {pct(internal_row['review_rate'])}; "
            f"old_data {pct(old_row['review_rate'])}, third_batch {pct(third_row['review_rate'])}."
        ),
        (
            f"- Internal BAcc is {pct(internal_row['balanced_accuracy'])}, FN={int(internal_row['fn'])}, FP={int(internal_row['fp'])}."
        ),
        (
            f"- If external cases are released only when all 5 fold-specific rules agree, strict external releases "
            f"{int(ext_vote5['release_n'])} cases with {int(ext_vote5['released_error_n'])} released errors; "
            f"review/reject {pct(ext_vote5['review_rate'])}, BAcc {pct(ext_vote5['balanced_accuracy'])}."
        ),
        "",
        "## Boundary",
        "",
        "This is the first nested test of safe-release. The old+third pooled rule remains the best current operating point, but nested validation determines how strongly we can claim stability. External consensus is reported as an observation only; strict external labels are not used for rule selection.",
    ]
    (OUT_DIR / "v167_nested_safe_release_validation.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "fold_count": int(len(folds)),
        "nested_internal_release_n": int(internal_row["release_n"]),
        "nested_internal_released_error_n": int(internal_row["released_error_n"]),
        "nested_internal_review_rate": float(internal_row["review_rate"]),
        "nested_internal_bacc": float(internal_row["balanced_accuracy"]),
        "nested_internal_fn": int(internal_row["fn"]),
        "nested_internal_fp": int(internal_row["fp"]),
        "external_vote5_release_n": int(ext_vote5["release_n"]),
        "external_vote5_released_error_n": int(ext_vote5["released_error_n"]),
        "external_vote5_review_rate": float(ext_vote5["review_rate"]),
        "external_vote5_bacc": float(ext_vote5["balanced_accuracy"]),
    }
    (OUT_DIR / "v167_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v167] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
