from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v199_directional_error_flip_corrector_20260527 import (
    OUT_DIR as V199_OUT_DIR,
    apply_rule,
    fit_direction_scores,
    make_dataset,
    summarize_subset,
    train_select_candidate,
)
from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v200_support_aware_directional_flip_20260527"


def no_action(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flip_trigger"] = False
    out["flip_direction"] = "none"
    out["flip_model"] = "none"
    out["flip_threshold"] = float("nan")
    out["flip_score"] = 0.0
    out["flip_pred"] = out["final_pred"].astype(int)
    out["flip_error"] = False
    out["rescued_by_flip"] = False
    out["hurt_by_flip"] = False
    out["remaining_review"] = out["adaptive_review"].astype(bool)
    out["system_pred"] = out["flip_pred"].astype(int)
    out.loc[out["remaining_review"], "system_pred"] = out.loc[out["remaining_review"], "label_idx"].astype(int)
    out["selection_source"] = "no_supported_same_domain_rule"
    out["selected_candidate_id"] = "none"
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_cols = make_dataset()
    scores_path = V199_OUT_DIR / "v199_directional_error_scores.csv"
    quality_path = V199_OUT_DIR / "v199_directional_error_model_quality.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        quality = pd.read_csv(quality_path)
    else:
        scores, quality = fit_direction_scores(df, feature_cols)

    frames = []
    selection_rows = []
    internal = df["domain"].isin(["old_data", "third_batch"])
    folds = sorted(int(x) for x in df.loc[internal, "fold_id"].dropna().unique() if int(x) >= 0)
    for fold in folds:
        for domain in ["old_data", "third_batch"]:
            train_mask = df["domain"].eq(domain) & df["adaptive_review"] & df["fold_id"].ne(fold)
            held_mask = df["domain"].eq(domain) & df["fold_id"].eq(fold)
            chosen = train_select_candidate(df, scores, train_mask)
            if chosen["selection_status"] == "train_safe_rescue_candidate":
                applied = apply_rule(df, scores, str(chosen["direction"]), str(chosen["model"]), float(chosen["threshold"]))
                held = applied.loc[held_mask].copy()
                source = "same_domain_nested_supported_rule"
            else:
                held = no_action(df.loc[held_mask].copy())
                source = "same_domain_no_safe_rescue_rule"
            held["selection_source"] = source
            held["selected_candidate_id"] = str(chosen["candidate_id"]) if source != "same_domain_no_safe_rescue_rule" else "none"
            frames.append(held)
            held_review = held.loc[held["adaptive_review"].astype(bool)].copy()
            selection_rows.append(
                {
                    **chosen,
                    "heldout_fold": int(fold),
                    "heldout_domain": domain,
                    "applied": source == "same_domain_nested_supported_rule",
                    "heldout_review_n": int(held_review.shape[0]),
                    "heldout_flip_n": int(held_review["flip_trigger"].sum()),
                    "heldout_flip_error_n": int(held_review["flip_error"].sum()),
                    "heldout_rescued_n": int(held_review["rescued_by_flip"].sum()),
                    "heldout_hurt_n": int(held_review["hurt_by_flip"].sum()),
                }
            )

    strict = no_action(df.loc[df["domain"].eq("strict_external")].copy())
    strict["selection_source"] = "strict_external_no_label_support_no_flip"
    frames.append(strict)

    cases = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for scope, mask in [
        ("old_data_nested", cases["domain"].eq("old_data")),
        ("third_batch_nested", cases["domain"].eq("third_batch")),
        ("internal_nested_old_third", cases["domain"].isin(["old_data", "third_batch"])),
        ("strict_external_locked_no_flip", cases["domain"].eq("strict_external")),
        ("all_domains_nested_plus_locked_external", cases["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        summary_rows.append(summarize_subset(cases.loc[mask].copy(), scope, "support_aware_nested_or_no_flip"))
    summary = pd.DataFrame(summary_rows)
    selections = pd.DataFrame(selection_rows)

    cases.to_csv(OUT_DIR / "v200_support_aware_flip_cases.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(OUT_DIR / "v200_support_aware_selected_rules.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v200_support_aware_flip_summary.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT_DIR / "v200_reused_directional_error_model_quality.csv", index=False, encoding="utf-8-sig")

    all_row = summary.loc[summary["scope"].eq("all_domains_nested_plus_locked_external")].iloc[0]
    internal_row = summary.loc[summary["scope"].eq("internal_nested_old_third")].iloc[0]
    strict_row = summary.loc[summary["scope"].eq("strict_external_locked_no_flip")].iloc[0]
    report = {
        "selection_rule": "A directional flip is allowed only when the same domain's training complement has a no-harm rescue rule; otherwise the case remains reviewed.",
        "all_domain_bacc": float(all_row["balanced_accuracy"]),
        "all_domain_remaining_review_rate": float(all_row["remaining_review_rate"]),
        "all_domain_flip_n": int(all_row["flip_n"]),
        "all_domain_flip_error_n": int(all_row["flip_error_n"]),
        "all_domain_rescued_n": int(all_row["rescued_n"]),
        "all_domain_hurt_n": int(all_row["hurt_n"]),
        "internal_bacc": float(internal_row["balanced_accuracy"]),
        "internal_remaining_review_rate": float(internal_row["remaining_review_rate"]),
        "internal_flip_error_n": int(internal_row["flip_error_n"]),
        "strict_external_bacc": float(strict_row["balanced_accuracy"]),
        "strict_external_remaining_review_rate": float(strict_row["remaining_review_rate"]),
        "strict_external_flip_error_n": int(strict_row["flip_error_n"]),
    }
    (OUT_DIR / "v200_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# v200 Support-aware Directional Flip",
        "",
        "## Purpose",
        "",
        "v199 showed that a global FN-risk flip can rescue a third-batch B2 case but hurts an old-data AB case. v200 adds a support constraint: a direction-specific flip is allowed only when the same domain's training complement contains a no-harm rescue rule. Unsupported domains remain in review.",
        "",
        "## Result",
        "",
        f"- All-domain BAcc: {100 * report['all_domain_bacc']:.2f}%; remaining review/reject: {100 * report['all_domain_remaining_review_rate']:.2f}%.",
        f"- Flips: {report['all_domain_flip_n']}; rescued: {report['all_domain_rescued_n']}; hurt/action-errors: {report['all_domain_hurt_n']} / {report['all_domain_flip_error_n']}.",
        f"- Strict external is not auto-flipped because no same-domain label support exists; BAcc: {100 * report['strict_external_bacc']:.2f}%, review/reject: {100 * report['strict_external_remaining_review_rate']:.2f}%.",
        "",
        "## Interpretation",
        "",
        "This is a cautious automatic-correction candidate. It demonstrates that direction-risk can safely correct a small number of supported-domain cases, but its current scale is too small to serve as the main efficiency module. v195 remains the main review-compression module.",
    ]
    (OUT_DIR / "v200_support_aware_directional_flip.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v200] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
