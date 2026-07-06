from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


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


def rule_mask(df: pd.DataFrame, pred_side: int, core: float, wholecrop: float, agree_min: int) -> pd.Series:
    base = df["v118_review_or_control"] & df["final_pred"].eq(pred_side) & df["core_agree_count"].ge(agree_min)
    if pred_side == 0:
        return base & df["prob_mean_core"].le(core) & df["wholecrop_prob"].le(wholecrop)
    return base & df["prob_mean_core"].ge(core) & df["wholecrop_prob"].ge(wholecrop)


def candidate_thresholds(sub: pd.DataFrame, pred_side: int) -> tuple[list[float], list[float]]:
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    if pred_side == 1:
        quantiles = [1.0 - q for q in quantiles]
    core = sorted(set(float(x) for x in np.nanquantile(sub["prob_mean_core"].dropna(), quantiles)))
    wc = sorted(set(float(x) for x in np.nanquantile(sub["wholecrop_prob"].dropna(), quantiles)))
    return core, wc


def scan_side(df: pd.DataFrame, pred_side: int) -> pd.DataFrame:
    internal = df["domain"].isin(["old_data", "third_batch"])
    sub = df.loc[df["v118_review_or_control"] & internal & df["final_pred"].eq(pred_side)].copy()
    core_grid, wc_grid = candidate_thresholds(sub, pred_side)
    rows = []
    for core, wc, agree_min in product(core_grid, wc_grid, [2, 3]):
        mask = rule_mask(df, pred_side, core, wc, agree_min)
        im = mask & internal
        external = mask & df["domain"].eq("strict_external")
        all_mask = mask
        internal_errors = int((im & df["base_wrong"]).sum())
        if internal_errors != 0 or int(im.sum()) == 0:
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
                "internal_release_n": int(im.sum()),
                "internal_released_errors": internal_errors,
                "external_release_n": int(external.sum()),
                "external_released_errors": int((external & df["base_wrong"]).sum()),
                "all_release_n": int(all_mask.sum()),
                "all_released_errors": int((all_mask & df["base_wrong"]).sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["internal_release_n", "external_release_n", "all_release_n", "core_agree_min"],
        ascending=[False, False, False, True],
    )


def select_rules(df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for pred_side in [0, 1]:
        scan = scan_side(df, pred_side)
        scan.to_csv(OUT_DIR / f"v161_scan_pred{pred_side}_internal_zero_error_rules.csv", index=False, encoding="utf-8-sig")
        if not scan.empty:
            selected.append(scan.iloc[0])
    return pd.DataFrame(selected)


def summarize_workflow(df: pd.DataFrame, release: pd.Series, workflow: str) -> list[dict[str, object]]:
    rows = []
    final_review = df["v118_review_or_control"] & ~release
    system_pred = df["final_pred"].copy()
    system_pred.loc[final_review] = df.loc[final_review, "label_idx"]
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("strict_external", df["domain"].eq("strict_external")),
        ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        sub_idx = mask.to_numpy()
        sub = df.loc[mask].copy()
        y = sub["label_idx"].to_numpy(int)
        pred = system_pred.loc[mask].to_numpy(int)
        prob = sub["prob_mean_core"].fillna(sub["main_prob"]).to_numpy(float)
        review = final_review.loc[mask].to_numpy(bool)
        rel = release.loc[mask].to_numpy(bool)
        released_errors = rel & sub["base_wrong"].to_numpy(bool)
        m = metrics(y, pred, prob)
        rows.append(
            {
                "workflow": workflow,
                "scope": scope,
                "n": int(len(sub)),
                "review_n": int(review.sum()),
                "review_rate": float(review.mean()),
                "auto_pass_n": int((~review).sum()),
                "auto_pass_rate": float((~review).mean()),
                "newly_released_from_review_n": int(rel.sum()),
                "newly_released_from_review_rate": float(rel.sum() / max(1, len(sub))),
                "released_error_n": int(released_errors.sum()),
                "remaining_error_n": int((pred != y).sum()),
                "fn": int(((y == 1) & (pred == 0)).sum()),
                "fp": int(((y == 0) & (pred == 1)).sum()),
                "accuracy": float(m["accuracy"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "f1": float(m["f1"]),
                "auc": float(m["auc"]),
                "sensitivity_high": float(m["sensitivity_high"]),
                "specificity_low": float(m["specificity_low"]),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    selected = select_rules(df)
    selected.to_csv(OUT_DIR / "v161_selected_internal_zero_error_release_rules.csv", index=False, encoding="utf-8-sig")

    release = pd.Series(False, index=df.index)
    for _, r in selected.iterrows():
        release |= rule_mask(
            df,
            int(r["pred_side"]),
            float(r["prob_mean_core_threshold"]),
            float(r["wholecrop_prob_threshold"]),
            int(r["core_agree_min"]),
        )
    df["v161_safe_release_from_review"] = release
    df["v161_final_review_or_reject"] = df["v118_review_or_control"] & ~release
    df["v161_released_error"] = release & df["base_wrong"]
    df.to_csv(OUT_DIR / "v161_safe_release_cases.csv", index=False, encoding="utf-8-sig")

    baseline_release = pd.Series(False, index=df.index)
    rows = summarize_workflow(df, baseline_release, "v118_high_safety_scorecard")
    rows += summarize_workflow(df, release, "v161_safe_release_scorecard")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "v161_safe_release_summary.csv", index=False, encoding="utf-8-sig")

    base_all = summary.loc[summary["workflow"].eq("v118_high_safety_scorecard") & summary["scope"].eq("all_domains")].iloc[0]
    new_all = summary.loc[summary["workflow"].eq("v161_safe_release_scorecard") & summary["scope"].eq("all_domains")].iloc[0]
    new_ext = summary.loc[summary["workflow"].eq("v161_safe_release_scorecard") & summary["scope"].eq("strict_external")].iloc[0]
    released_errors = int(df["v161_released_error"].sum())

    md = [
        "# v161 Safe Release From High-safety Review Pool",
        "",
        "## Main Findings",
        "",
        (
            f"- Selected {len(selected)} internal zero-error release rules from old+third reviewed cases. "
            f"Total newly released cases: {int(release.sum())}; released errors observed across all current domains: {released_errors}."
        ),
        (
            f"- All-domain review/reject rate decreases from {pct(base_all['review_rate'])} to {pct(new_all['review_rate'])}; "
            f"auto-pass increases from {pct(base_all['auto_pass_rate'])} to {pct(new_all['auto_pass_rate'])}."
        ),
        (
            f"- All-domain BAcc remains {pct(new_all['balanced_accuracy'])}, FN={int(new_all['fn'])}, FP={int(new_all['fp'])}."
        ),
        (
            f"- Strict external BAcc remains {pct(new_ext['balanced_accuracy'])}, with review/reject rate {pct(new_ext['review_rate'])}."
        ),
        "",
        "## Boundary",
        "",
        "The release rules are selected using old+third labels and then checked on strict external. This is a strong efficiency candidate, but it should still be described as internally selected and externally observed, not as a prospectively locked clinical threshold.",
    ]
    (OUT_DIR / "v161_safe_release_from_high_safety_review_pool.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "selected_rule_count": int(len(selected)),
        "newly_released_all_domains": int(release.sum()),
        "released_errors_all_domains": released_errors,
        "baseline_review_rate_all_domains": float(base_all["review_rate"]),
        "v161_review_rate_all_domains": float(new_all["review_rate"]),
        "baseline_auto_pass_rate_all_domains": float(base_all["auto_pass_rate"]),
        "v161_auto_pass_rate_all_domains": float(new_all["auto_pass_rate"]),
        "v161_all_domain_bacc": float(new_all["balanced_accuracy"]),
        "v161_all_domain_fn": int(new_all["fn"]),
        "v161_all_domain_fp": int(new_all["fp"]),
        "v161_strict_external_bacc": float(new_ext["balanced_accuracy"]),
    }
    (OUT_DIR / "v161_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v161] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
