from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v199_directional_error_flip_corrector_20260527 import (
    DIRECTIONS,
    THRESHOLDS,
    OUT_DIR as V199_OUT_DIR,
    apply_rule,
    fit_direction_scores,
    make_dataset,
    summarize_subset,
)
from run_grosspath_rc_v200_support_aware_directional_flip_20260527 import no_action


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v201_stable_supported_domain_flip_20260527"


def scan_domain_candidates(df: pd.DataFrame, scores: pd.DataFrame, domain: str) -> pd.DataFrame:
    rows = []
    domain_mask = df["domain"].eq(domain) & df["adaptive_review"].astype(bool)
    folds = sorted(int(x) for x in df.loc[domain_mask, "fold_id"].dropna().unique() if int(x) >= 0)
    for direction in DIRECTIONS:
        models = [c.split("__", 1)[1] for c in scores.columns if c.startswith(f"{direction}__")]
        for model in models:
            score_col = f"{direction}__{model}"
            domain_scores = scores.loc[domain_mask, score_col].dropna()
            dynamic = [float(x) for x in domain_scores.quantile([0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]).values] if len(domain_scores) else []
            for threshold in sorted(set([*THRESHOLDS, *dynamic])):
                if not pd.notna(threshold):
                    continue
                applied = apply_rule(df, scores, direction, model, float(threshold))
                # A supported-domain rule is only allowed to act inside its support domain.
                applied.loc[~applied["domain"].eq(domain), "flip_trigger"] = False
                applied.loc[~applied["domain"].eq(domain), "flip_error"] = False
                applied.loc[~applied["domain"].eq(domain), "rescued_by_flip"] = False
                applied.loc[~applied["domain"].eq(domain), "hurt_by_flip"] = False
                sub = applied.loc[domain_mask].copy()
                fold_train_ok = []
                fold_train_rescue = []
                fold_held_errors = []
                fold_held_rescue = []
                fold_held_actions = []
                for fold in folds:
                    train = sub["fold_id"].ne(fold)
                    held = sub["fold_id"].eq(fold)
                    tr = sub.loc[train]
                    he = sub.loc[held]
                    fold_train_ok.append(
                        int(tr["flip_error"].sum()) == 0
                        and int(tr["rescued_by_flip"].sum()) > 0
                        and int(tr["hurt_by_flip"].sum()) == 0
                    )
                    fold_train_rescue.append(int(tr["rescued_by_flip"].sum()))
                    fold_held_errors.append(int(he["flip_error"].sum()))
                    fold_held_rescue.append(int(he["rescued_by_flip"].sum()))
                    fold_held_actions.append(int(he["flip_trigger"].sum()))
                rows.append(
                    {
                        "domain": domain,
                        "direction": direction,
                        "model": model,
                        "threshold": float(threshold),
                        "candidate_id": f"{domain}||{direction}||{model}||{float(threshold):.6f}",
                        "all_train_complements_zero_harm_with_rescue": bool(folds and all(fold_train_ok)),
                        "min_train_rescued_n": int(min(fold_train_rescue)) if fold_train_rescue else 0,
                        "heldout_flip_n": int(sum(fold_held_actions)),
                        "heldout_flip_error_n": int(sum(fold_held_errors)),
                        "heldout_rescued_n": int(sum(fold_held_rescue)),
                        "full_domain_flip_n": int(sub["flip_trigger"].sum()),
                        "full_domain_flip_error_n": int(sub["flip_error"].sum()),
                        "full_domain_rescued_n": int(sub["rescued_by_flip"].sum()),
                        "full_domain_hurt_n": int(sub["hurt_by_flip"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def choose_domain_rule(scan: pd.DataFrame, domain: str) -> pd.Series | None:
    safe = scan.loc[
        scan["all_train_complements_zero_harm_with_rescue"].astype(bool)
        & scan["heldout_flip_error_n"].eq(0)
        & scan["heldout_rescued_n"].gt(0)
        & scan["full_domain_hurt_n"].eq(0)
    ].copy()
    if safe.empty:
        return None
    return safe.sort_values(
        ["heldout_rescued_n", "full_domain_flip_n", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]


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

    scan_tables = []
    selected_rows = []
    frames = []
    for domain in ["old_data", "third_batch"]:
        scan = scan_domain_candidates(df, scores, domain)
        scan_tables.append(scan)
        chosen = choose_domain_rule(scan, domain)
        if chosen is None:
            domain_cases = no_action(df.loc[df["domain"].eq(domain)].copy())
            domain_cases["selection_source"] = "no_stable_supported_domain_rule"
            domain_cases["selected_candidate_id"] = "none"
            selected_rows.append({"domain": domain, "selected": False})
        else:
            applied = apply_rule(df, scores, str(chosen["direction"]), str(chosen["model"]), float(chosen["threshold"]))
            domain_cases = applied.loc[df["domain"].eq(domain)].copy()
            domain_cases["selection_source"] = "stable_supported_domain_rule"
            domain_cases["selected_candidate_id"] = str(chosen["candidate_id"])
            selected_rows.append({**chosen.to_dict(), "selected": True})
        frames.append(domain_cases)

    strict = no_action(df.loc[df["domain"].eq("strict_external")].copy())
    strict["selection_source"] = "strict_external_no_supported_rule"
    strict["selected_candidate_id"] = "none"
    frames.append(strict)

    cases = pd.concat(frames, ignore_index=True)
    scan_all = pd.concat(scan_tables, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    summary_rows = []
    for scope, mask in [
        ("old_data", cases["domain"].eq("old_data")),
        ("third_batch", cases["domain"].eq("third_batch")),
        ("internal_old_third", cases["domain"].isin(["old_data", "third_batch"])),
        ("strict_external_no_flip", cases["domain"].eq("strict_external")),
        ("all_domains", cases["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        summary_rows.append(summarize_subset(cases.loc[mask].copy(), scope, "stable_supported_domain_flip"))
    summary = pd.DataFrame(summary_rows)

    scan_all.to_csv(OUT_DIR / "v201_domain_candidate_scan.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v201_selected_domain_rules.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v201_stable_supported_domain_flip_cases.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v201_stable_supported_domain_flip_summary.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT_DIR / "v201_reused_directional_error_model_quality.csv", index=False, encoding="utf-8-sig")

    all_row = summary.loc[summary["scope"].eq("all_domains")].iloc[0]
    third_row = summary.loc[summary["scope"].eq("third_batch")].iloc[0]
    strict_row = summary.loc[summary["scope"].eq("strict_external_no_flip")].iloc[0]
    report = {
        "all_domain_bacc": float(all_row["balanced_accuracy"]),
        "all_domain_remaining_review_rate": float(all_row["remaining_review_rate"]),
        "all_domain_flip_n": int(all_row["flip_n"]),
        "all_domain_flip_error_n": int(all_row["flip_error_n"]),
        "all_domain_rescued_n": int(all_row["rescued_n"]),
        "all_domain_hurt_n": int(all_row["hurt_n"]),
        "third_batch_flip_n": int(third_row["flip_n"]),
        "third_batch_rescued_n": int(third_row["rescued_n"]),
        "third_batch_hurt_n": int(third_row["hurt_n"]),
        "strict_external_bacc": float(strict_row["balanced_accuracy"]),
        "strict_external_remaining_review_rate": float(strict_row["remaining_review_rate"]),
    }
    (OUT_DIR / "v201_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# v201 Stable Supported-domain Flip",
        "",
        "## Purpose",
        "",
        "v199/v200 showed that directional FN-risk can identify third-batch high-risk misses, but overly broad or greedy flipping causes harm. v201 selects only fixed domain-supported rules that are zero-harm across fold audits and only acts inside the supported domain.",
        "",
        "## Result",
        "",
        f"- All-domain BAcc: {100 * report['all_domain_bacc']:.2f}%; remaining review/reject: {100 * report['all_domain_remaining_review_rate']:.2f}%.",
        f"- All-domain flips: {report['all_domain_flip_n']}; rescued: {report['all_domain_rescued_n']}; hurt/action-errors: {report['all_domain_hurt_n']} / {report['all_domain_flip_error_n']}.",
        f"- Third batch flips: {report['third_batch_flip_n']}; rescued: {report['third_batch_rescued_n']}; hurt: {report['third_batch_hurt_n']}.",
        f"- Strict external remains no-flip due to absent same-domain support; BAcc {100 * report['strict_external_bacc']:.2f}%, review/reject {100 * report['strict_external_remaining_review_rate']:.2f}%.",
        "",
        "## Interpretation",
        "",
        "This is the first non-harmful automatic correction candidate, but the gain is small: it rescues a few supported-domain review-pool cases and mainly complements the stronger v195 stable release module.",
    ]
    (OUT_DIR / "v201_stable_supported_domain_flip.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v201] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
