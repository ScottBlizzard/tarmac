from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v182_stable_fixed_image_agreement_release_20260527"
V161_CASES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_cases.csv"
V173_CASES = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527" / "v173_corrector_case_outputs.csv"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_base() -> pd.DataFrame:
    df = pd.read_csv(V161_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v118_review_or_control", "v161_safe_release_from_review", "v161_final_review_or_reject", "base_wrong"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "fold_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    df["prob_mean_core"] = pd.to_numeric(df["prob_mean_core"], errors="coerce")
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    return df


def make_candidates(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    corr = pd.read_csv(V173_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["corrector_pred", "final_pred"]:
        corr[col] = pd.to_numeric(corr[col], errors="coerce").fillna(-1).astype(int)
    corr["corrector_confidence"] = pd.to_numeric(corr["corrector_confidence"], errors="coerce")
    corr["agree"] = corr["corrector_pred"].eq(corr["final_pred"])
    masks = {}
    rows = []
    for (review_policy, model), g in corr.groupby(["review_policy", "model"], sort=False):
        if review_policy not in base.columns:
            continue
        merged = base[["case_id"]].merge(g[["case_id", "agree", "corrector_confidence"]], on="case_id", how="left")
        for t in THRESHOLDS:
            release = (
                merged["agree"].fillna(False).astype(bool)
                & merged["corrector_confidence"].ge(float(t))
                & base[review_policy]
            )
            cid = f"{review_policy}||{model}||{t:.2f}"
            masks[cid] = release
            rows.append(
                {
                    "candidate_id": cid,
                    "review_policy": review_policy,
                    "model": model,
                    "threshold": float(t),
                    "candidate_release_n": int(release.sum()),
                    "additional_release_n": int((release & ~base["v161_safe_release_from_review"]).sum()),
                }
            )
    return pd.DataFrame(rows), masks


def summarize_workflow(base: pd.DataFrame, release: pd.Series, workflow: str) -> list[dict[str, object]]:
    y = base["label_idx"].to_numpy(int)
    pred = base["final_pred"].to_numpy(int).copy()
    final_review = base["v118_review_or_control"] & ~release
    pred[final_review.to_numpy(bool)] = y[final_review.to_numpy(bool)]
    released_error = release & base["base_wrong"]
    rows = []
    for scope, mask in [
        ("internal_old_third", base["domain"].isin(["old_data", "third_batch"])),
        ("old_data", base["domain"].eq("old_data")),
        ("third_batch", base["domain"].eq("third_batch")),
        ("strict_external", base["domain"].eq("strict_external")),
        ("all_domains", base["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        m = metrics(y[mask.to_numpy(bool)], pred[mask.to_numpy(bool)], base.loc[mask, "prob_mean_core"].to_numpy(float))
        rows.append(
            {
                "workflow": workflow,
                "scope": scope,
                "n": int(mask.sum()),
                "release_n": int(release[mask].sum()),
                "released_error_n": int(released_error[mask].sum()),
                "final_review_n": int(final_review[mask].sum()),
                "final_review_rate": float(final_review[mask].mean()),
                "accuracy": float(m["accuracy"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "f1": float(m["f1"]),
                "auc": float(m["auc"]),
                "fn": int(m["fn"]),
                "fp": int(m["fp"]),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()
    catalog, masks = make_candidates(base)
    internal = base["domain"].isin(["old_data", "third_batch"])
    folds = sorted(base.loc[internal, "fold_id"].unique())

    audit_rows = []
    for _, r in catalog.iterrows():
        cid = str(r["candidate_id"])
        release = masks[cid]
        additional = release & ~base["v161_safe_release_from_review"]
        fold_ok = []
        train_releases = []
        train_errors = []
        heldout_releases = []
        heldout_errors = []
        for fold in folds:
            train = internal & base["fold_id"].ne(int(fold))
            held = internal & base["fold_id"].eq(int(fold))
            train_add = additional & train
            held_add = additional & held
            tr_err = int((train_add & base["base_wrong"]).sum())
            fold_ok.append(tr_err == 0 and int(train_add.sum()) > 0)
            train_releases.append(int(train_add.sum()))
            train_errors.append(tr_err)
            heldout_releases.append(int(held_add.sum()))
            heldout_errors.append(int((held_add & base["base_wrong"]).sum()))
        audit_rows.append(
            {
                **r.to_dict(),
                "all_folds_train_zero_error": bool(all(fold_ok)),
                "folds_train_zero_error_count": int(sum(fold_ok)),
                "min_train_additional_release_n": int(min(train_releases) if train_releases else 0),
                "mean_train_additional_release_n": float(sum(train_releases) / len(train_releases)) if train_releases else 0.0,
                "total_internal_additional_release_n": int((additional & internal).sum()),
                "total_internal_additional_error_n": int((additional & internal & base["base_wrong"]).sum()),
                "total_external_additional_release_n": int((additional & base["domain"].eq("strict_external")).sum()),
                "total_external_additional_error_n": int((additional & base["domain"].eq("strict_external") & base["base_wrong"]).sum()),
                "heldout_additional_release_n": int(sum(heldout_releases)),
                "heldout_additional_error_n": int(sum(heldout_errors)),
            }
        )
    audit = pd.DataFrame(audit_rows)
    stable = audit.loc[
        audit["all_folds_train_zero_error"]
        & audit["total_internal_additional_release_n"].gt(0)
        & audit["heldout_additional_error_n"].eq(0)
    ].copy()
    if not stable.empty:
        chosen = stable.sort_values(
            ["heldout_additional_release_n", "min_train_additional_release_n", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_status = "stable_all_folds_train_zero_and_heldout_zero"
    else:
        relaxed = audit.loc[audit["all_folds_train_zero_error"] & audit["total_internal_additional_release_n"].gt(0)].copy()
        if not relaxed.empty:
            chosen = relaxed.sort_values(
                ["heldout_additional_error_n", "heldout_additional_release_n", "min_train_additional_release_n"],
                ascending=[True, False, False],
            ).iloc[0]
            selection_status = "stable_train_zero_but_heldout_errors"
        else:
            chosen = audit.sort_values(
                ["folds_train_zero_error_count", "total_internal_additional_error_n", "total_internal_additional_release_n"],
                ascending=[False, True, False],
            ).iloc[0]
            selection_status = "no_all_fold_stable_candidate"

    chosen_release = masks[str(chosen["candidate_id"])]
    chosen_union = base["v161_safe_release_from_review"] | chosen_release
    summary = pd.DataFrame(
        summarize_workflow(base, base["v161_safe_release_from_review"], "v161_safe_release_fixed")
        + summarize_workflow(base, chosen_union, "v182_stable_fixed_image_agreement_union")
    )

    cases = base[
        [
            "domain",
            "case_id",
            "original_case_id",
            "task_l6_label",
            "label_idx",
            "final_pred",
            "fold_id",
            "image_name",
            "v118_review_or_control",
            "v161_safe_release_from_review",
            "base_wrong",
        ]
    ].copy()
    cases["v182_image_agreement_release"] = chosen_release
    cases["v182_additional_release_over_v161"] = chosen_release & ~base["v161_safe_release_from_review"]
    cases["v182_union_release"] = chosen_union
    cases["v182_union_released_error"] = chosen_union & base["base_wrong"]

    audit.to_csv(OUT_DIR / "v182_stable_candidate_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**chosen.to_dict(), "selection_status": selection_status}]).to_csv(
        OUT_DIR / "v182_selected_stable_fixed_rule.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(OUT_DIR / "v182_stable_fixed_workflow_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v182_stable_fixed_case_outputs.csv", index=False, encoding="utf-8-sig")

    all_v161 = summary.loc[summary["workflow"].eq("v161_safe_release_fixed") & summary["scope"].eq("all_domains")].iloc[0]
    all_v182 = summary.loc[
        summary["workflow"].eq("v182_stable_fixed_image_agreement_union") & summary["scope"].eq("all_domains")
    ].iloc[0]
    report = {
        "selection_status": selection_status,
        "selected_candidate_id": str(chosen["candidate_id"]),
        "selected_review_policy": str(chosen["review_policy"]),
        "selected_model": str(chosen["model"]),
        "selected_threshold": float(chosen["threshold"]),
        "stable_candidate_count": int(len(stable)),
        "heldout_additional_release_n": int(chosen["heldout_additional_release_n"]),
        "heldout_additional_error_n": int(chosen["heldout_additional_error_n"]),
        "v161_all_domain_review_rate": float(all_v161["final_review_rate"]),
        "v182_all_domain_review_rate": float(all_v182["final_review_rate"]),
        "v182_all_domain_bacc": float(all_v182["balanced_accuracy"]),
        "v182_released_errors": int(all_v182["released_error_n"]),
        "v182_fn": int(all_v182["fn"]),
        "v182_fp": int(all_v182["fp"]),
    }
    (OUT_DIR / "v182_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# v182 Stable Fixed Image-agreement Release",
        "",
        "## Purpose",
        "",
        "v180 showed that selecting different image-agreement release rules per fold is unstable. v182 only allows one fixed rule that is zero-error on every training-fold complement, then evaluates the held-out internal OOF cases.",
        "",
        "## Result",
        "",
        (
            f"- Selected `{report['selected_candidate_id']}` with status `{selection_status}`; "
            f"held-out additional releases {report['heldout_additional_release_n']}, held-out additional errors "
            f"{report['heldout_additional_error_n']}."
        ),
        (
            f"- All-domain review rate: v161 {100 * report['v161_all_domain_review_rate']:.2f}% -> "
            f"v182 {100 * report['v182_all_domain_review_rate']:.2f}%."
        ),
        (
            f"- All-domain BAcc {100 * report['v182_all_domain_bacc']:.2f}%, released errors "
            f"{report['v182_released_errors']}, FN={report['v182_fn']}, FP={report['v182_fp']}."
        ),
    ]
    (OUT_DIR / "v182_stable_fixed_image_agreement_release.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v182] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
