from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v167_nested_safe_release_validation_20260527 import (
    apply_rules,
    load_cases,
    scan_train_rules,
    select_rules,
    summarize_scope,
)


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v168_conservative_nested_safe_release_20260527"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def rule_from_row(strategy: str, fold_id: int, pred_side: int, row: pd.Series) -> dict[str, object]:
    return {
        "strategy": strategy,
        "fold_id": int(fold_id),
        "pred_side": int(pred_side),
        "prob_mean_core_threshold": float(row["prob_mean_core_threshold"]),
        "wholecrop_prob_threshold": float(row["wholecrop_prob_threshold"]),
        "core_agree_min": int(row["core_agree_min"]),
    }


def select_domain_intersection(df: pd.DataFrame, train_mask: pd.Series, fold_id: int) -> pd.DataFrame:
    rows = []
    for pred_side in [0, 1]:
        old_scan = scan_train_rules(df, train_mask & df["domain"].eq("old_data"), pred_side)
        third_scan = scan_train_rules(df, train_mask & df["domain"].eq("third_batch"), pred_side)
        if old_scan.empty or third_scan.empty:
            continue
        old = old_scan.iloc[0]
        third = third_scan.iloc[0]
        if pred_side == 0:
            core = min(float(old["prob_mean_core_threshold"]), float(third["prob_mean_core_threshold"]))
            wc = min(float(old["wholecrop_prob_threshold"]), float(third["wholecrop_prob_threshold"]))
        else:
            core = max(float(old["prob_mean_core_threshold"]), float(third["prob_mean_core_threshold"]))
            wc = max(float(old["wholecrop_prob_threshold"]), float(third["wholecrop_prob_threshold"]))
        rows.append(
            {
                "strategy": "outer_domain_intersection",
                "fold_id": int(fold_id),
                "pred_side": int(pred_side),
                "prob_mean_core_threshold": core,
                "wholecrop_prob_threshold": wc,
                "core_agree_min": max(int(old["core_agree_min"]), int(third["core_agree_min"])),
            }
        )
    return pd.DataFrame(rows)


def select_inner_fold_intersection(df: pd.DataFrame, outer_train_mask: pd.Series, outer_fold: int, all_folds: list[int]) -> pd.DataFrame:
    rows = []
    inner_rule_parts = []
    train_folds = [f for f in all_folds if f != outer_fold]
    for inner_holdout in train_folds:
        inner_train = outer_train_mask & df["fold_id"].ne(inner_holdout)
        r = select_rules(df, inner_train, int(outer_fold * 10 + inner_holdout))
        if not r.empty:
            inner_rule_parts.append(r)
    if not inner_rule_parts:
        return pd.DataFrame()
    all_inner = pd.concat(inner_rule_parts, ignore_index=True)
    for pred_side in [0, 1]:
        sub = all_inner.loc[all_inner["pred_side"].astype(int).eq(pred_side)]
        if sub.empty:
            continue
        if pred_side == 0:
            core = float(sub["prob_mean_core_threshold"].min())
            wc = float(sub["wholecrop_prob_threshold"].min())
        else:
            core = float(sub["prob_mean_core_threshold"].max())
            wc = float(sub["wholecrop_prob_threshold"].max())
        rows.append(
            {
                "strategy": "outer_inner_fold_intersection",
                "fold_id": int(outer_fold),
                "pred_side": int(pred_side),
                "prob_mean_core_threshold": core,
                "wholecrop_prob_threshold": wc,
                "core_agree_min": int(sub["core_agree_min"].max()),
            }
        )
    return pd.DataFrame(rows)


def add_rule_text(rules: pd.DataFrame) -> pd.DataFrame:
    out = rules.copy()
    texts = []
    for _, r in out.iterrows():
        pred = int(r["pred_side"])
        op = "<=" if pred == 0 else ">="
        texts.append(
            f"pred={pred}, prob_mean_core {op} {float(r['prob_mean_core_threshold']):.6f}, "
            f"wholecrop_prob {op} {float(r['wholecrop_prob_threshold']):.6f}, "
            f"core_agree_count>={int(r['core_agree_min'])}"
        )
    out["rule"] = texts
    return out


def evaluate_strategy(df: pd.DataFrame, rules_by_fold: pd.DataFrame, strategy: str, folds: list[int]) -> tuple[list[dict[str, object]], pd.Series]:
    internal = df["domain"].isin(["old_data", "third_batch"])
    nested_release = pd.Series(False, index=df.index)
    fold_rows = []
    for fold in folds:
        heldout = internal & df["fold_id"].eq(fold)
        rules = rules_by_fold.loc[rules_by_fold["fold_id"].astype(int).eq(fold)]
        release_all = apply_rules(df, rules)
        nested_release.loc[heldout] = release_all.loc[heldout]
        fold_rows.append(summarize_scope(df, release_all & heldout, f"{strategy}_fold_{fold}", f"heldout_fold_{fold}", heldout))
    rows = fold_rows
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("internal_all", internal),
    ]:
        rows.append(summarize_scope(df, nested_release, strategy, scope, mask))
    return rows, nested_release


def external_consensus(df: pd.DataFrame, rules_by_fold: pd.DataFrame, strategy: str, folds: list[int]) -> list[dict[str, object]]:
    external = df["domain"].eq("strict_external")
    votes = pd.Series(0, index=df.index, dtype=int)
    for fold in folds:
        rules = rules_by_fold.loc[rules_by_fold["fold_id"].astype(int).eq(fold)]
        release_all = apply_rules(df, rules)
        votes.loc[external] += release_all.loc[external].astype(int)
    rows = []
    for vote_min in [1, 3, 5]:
        release = pd.Series(False, index=df.index)
        release.loc[external] = votes.loc[external] >= vote_min
        rows.append(summarize_scope(df, release, f"{strategy}_external_vote_ge_{vote_min}", "strict_external", external))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    internal = df["domain"].isin(["old_data", "third_batch"])
    folds = sorted(df.loc[internal, "fold_id"].unique().tolist())

    rule_parts = []
    for fold in folds:
        train_mask = internal & df["fold_id"].ne(fold)
        domain_rules = select_domain_intersection(df, train_mask, int(fold))
        inner_rules = select_inner_fold_intersection(df, train_mask, int(fold), folds)
        rule_parts.extend([domain_rules, inner_rules])
    rules = add_rule_text(pd.concat(rule_parts, ignore_index=True))
    rules.to_csv(OUT_DIR / "v168_conservative_nested_rules.csv", index=False, encoding="utf-8-sig")

    all_rows = []
    case_parts = []
    for strategy in ["outer_domain_intersection", "outer_inner_fold_intersection"]:
        srules = rules.loc[rules["strategy"].eq(strategy)].copy()
        rows, release = evaluate_strategy(df, srules, strategy, folds)
        rows += external_consensus(df, srules, strategy, folds)
        all_rows.extend(rows)
        tmp = df[["domain", "fold_id", "case_id", "original_case_id", "task_l6_label", "label_idx", "final_pred", "base_wrong", "prob_mean_core", "wholecrop_prob", "main_prob", "core_agree_count"]].copy()
        tmp["strategy"] = strategy
        tmp["nested_release"] = release
        tmp["nested_released_error"] = release & df["base_wrong"]
        case_parts.append(tmp)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT_DIR / "v168_conservative_nested_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(case_parts, ignore_index=True).to_csv(OUT_DIR / "v168_conservative_nested_cases.csv", index=False, encoding="utf-8-sig")

    focus = summary.loc[summary["scope"].isin(["internal_all", "old_data", "third_batch"])].copy()
    focus.to_csv(OUT_DIR / "v168_internal_focus.csv", index=False, encoding="utf-8-sig")
    domain_internal = summary.loc[summary["workflow"].eq("outer_domain_intersection") & summary["scope"].eq("internal_all")].iloc[0]
    inner_internal = summary.loc[summary["workflow"].eq("outer_inner_fold_intersection") & summary["scope"].eq("internal_all")].iloc[0]
    domain_ext5 = summary.loc[summary["workflow"].eq("outer_domain_intersection_external_vote_ge_5")].iloc[0]
    inner_ext5 = summary.loc[summary["workflow"].eq("outer_inner_fold_intersection_external_vote_ge_5")].iloc[0]

    md = [
        "# v168 Conservative Nested Safe-release",
        "",
        "## Main Findings",
        "",
        (
            f"- Domain-intersection nested release: internal release {int(domain_internal['release_n'])}, "
            f"released errors {int(domain_internal['released_error_n'])}, review/reject {pct(domain_internal['review_rate'])}, "
            f"BAcc {pct(domain_internal['balanced_accuracy'])}."
        ),
        (
            f"- Inner-fold-intersection nested release: internal release {int(inner_internal['release_n'])}, "
            f"released errors {int(inner_internal['released_error_n'])}, review/reject {pct(inner_internal['review_rate'])}, "
            f"BAcc {pct(inner_internal['balanced_accuracy'])}."
        ),
        (
            f"- Strict external all-fold consensus for domain-intersection releases {int(domain_ext5['release_n'])} with "
            f"{int(domain_ext5['released_error_n'])} released errors; inner-fold consensus releases {int(inner_ext5['release_n'])} with "
            f"{int(inner_ext5['released_error_n'])} released errors."
        ),
        "",
        "## Interpretation",
        "",
        "v167 showed that fold-wise max-release can be too aggressive. v168 tests two stricter choices. If either keeps held-out released errors at zero, that strategy can replace v161 as the more defensible nested safe-release operating point; otherwise v161 should remain a current-data candidate rather than a validated stable threshold.",
    ]
    (OUT_DIR / "v168_conservative_nested_safe_release.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "domain_intersection_internal_release_n": int(domain_internal["release_n"]),
        "domain_intersection_internal_released_error_n": int(domain_internal["released_error_n"]),
        "domain_intersection_review_rate": float(domain_internal["review_rate"]),
        "domain_intersection_bacc": float(domain_internal["balanced_accuracy"]),
        "inner_fold_intersection_internal_release_n": int(inner_internal["release_n"]),
        "inner_fold_intersection_internal_released_error_n": int(inner_internal["released_error_n"]),
        "inner_fold_intersection_review_rate": float(inner_internal["review_rate"]),
        "inner_fold_intersection_bacc": float(inner_internal["balanced_accuracy"]),
        "domain_external_vote5_release_n": int(domain_ext5["release_n"]),
        "domain_external_vote5_released_error_n": int(domain_ext5["released_error_n"]),
        "inner_external_vote5_release_n": int(inner_ext5["release_n"]),
        "inner_external_vote5_released_error_n": int(inner_ext5["released_error_n"]),
    }
    (OUT_DIR / "v168_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v168] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
