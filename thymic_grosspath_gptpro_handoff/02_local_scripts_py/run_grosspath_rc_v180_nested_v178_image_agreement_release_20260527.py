from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v180_nested_v178_image_agreement_release_20260527"
V161_CASES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_cases.csv"
V173_CASES = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527" / "v173_corrector_case_outputs.csv"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_base() -> pd.DataFrame:
    df = pd.read_csv(V161_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v118_review_or_control", "v161_safe_release_from_review", "v161_final_review_or_reject", "base_wrong"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "fold_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in ["prob_mean_core"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    return df


def candidate_matrix(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    corr = pd.read_csv(V173_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["corrector_pred", "final_pred"]:
        corr[col] = pd.to_numeric(corr[col], errors="coerce").fillna(-1).astype(int)
    corr["corrector_confidence"] = pd.to_numeric(corr["corrector_confidence"], errors="coerce")
    corr["agree"] = corr["corrector_pred"].eq(corr["final_pred"])

    rows = []
    masks: dict[str, pd.Series] = {}
    for (review_policy, model), g in corr.groupby(["review_policy", "model"], sort=False):
        merged = base[["case_id"]].merge(
            g[["case_id", "agree", "corrector_confidence"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        for t in THRESHOLDS:
            if review_policy not in base.columns:
                continue
            mask = merged["agree"].fillna(False).astype(bool) & merged["corrector_confidence"].ge(t) & base[review_policy]
            cid = f"{review_policy}||{model}||{t:.2f}"
            masks[cid] = mask
            additional = mask & ~base["v161_safe_release_from_review"]
            rows.append(
                {
                    "candidate_id": cid,
                    "review_policy": review_policy,
                    "model": model,
                    "threshold": float(t),
                    "candidate_release_n": int(mask.sum()),
                    "additional_release_n": int(additional.sum()),
                }
            )
    return pd.DataFrame(rows), masks


def choose_for_fold(base: pd.DataFrame, cand: pd.DataFrame, masks: dict[str, pd.Series], heldout_fold: int) -> dict[str, object]:
    train = base["domain"].isin(["old_data", "third_batch"]) & base["fold_id"].ne(heldout_fold)
    rows = []
    for _, r in cand.iterrows():
        m = masks[str(r["candidate_id"])]
        add = m & ~base["v161_safe_release_from_review"]
        add_train = add & train
        rows.append(
            {
                **r.to_dict(),
                "train_additional_release_n": int(add_train.sum()),
                "train_additional_error_n": int((add_train & base["base_wrong"]).sum()),
            }
        )
    scan = pd.DataFrame(rows)
    viable = scan.loc[scan["train_additional_error_n"].eq(0) & scan["train_additional_release_n"].gt(0)].copy()
    if viable.empty:
        return {
            "fold_id": heldout_fold,
            "candidate_id": "",
            "selection_status": "no_viable_zero_error_candidate",
            "train_additional_release_n": 0,
            "train_additional_error_n": 0,
        }
    chosen = viable.sort_values(
        ["train_additional_release_n", "threshold", "candidate_release_n"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    chosen["fold_id"] = heldout_fold
    chosen["selection_status"] = "fold_train_internal_zero_error"
    return chosen


def workflow_summary(base: pd.DataFrame, release: pd.Series, workflow: str) -> list[dict[str, object]]:
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
    cand, masks = candidate_matrix(base)
    internal = base["domain"].isin(["old_data", "third_batch"])

    selections = []
    nested_additional = pd.Series(False, index=base.index)
    for fold in sorted(base.loc[internal, "fold_id"].unique()):
        chosen = choose_for_fold(base, cand, masks, int(fold))
        selections.append(chosen)
        cid = str(chosen.get("candidate_id", ""))
        heldout = internal & base["fold_id"].eq(int(fold))
        if cid and cid in masks:
            add = masks[cid] & ~base["v161_safe_release_from_review"]
            nested_additional |= add & heldout

    # For strict external, only release if all fold-selected candidates agree; this is conservative and does not use labels.
    selected_ids = [str(x.get("candidate_id", "")) for x in selections if str(x.get("candidate_id", ""))]
    if selected_ids:
        ext_consensus = base["domain"].eq("strict_external")
        for cid in selected_ids:
            ext_consensus &= masks[cid]
        ext_additional = ext_consensus & ~base["v161_safe_release_from_review"]
    else:
        ext_additional = pd.Series(False, index=base.index)

    nested_union = base["v161_safe_release_from_review"] | nested_additional | ext_additional
    selected_df = pd.DataFrame(selections)
    selected_df.to_csv(OUT_DIR / "v180_fold_selected_rules.csv", index=False, encoding="utf-8-sig")
    cand.to_csv(OUT_DIR / "v180_candidate_rule_catalog.csv", index=False, encoding="utf-8-sig")

    case_out = base[
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
    case_out["v180_nested_additional_release"] = nested_additional
    case_out["v180_external_consensus_additional_release"] = ext_additional
    case_out["v180_nested_union_release"] = nested_union
    case_out["v180_union_released_error"] = nested_union & base["base_wrong"]
    case_out.to_csv(OUT_DIR / "v180_nested_case_outputs.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    summary_rows += workflow_summary(base, base["v161_safe_release_from_review"], "v161_safe_release_fixed")
    summary_rows += workflow_summary(base, nested_union, "v180_nested_image_agreement_union")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "v180_nested_workflow_summary.csv", index=False, encoding="utf-8-sig")

    all_v161 = summary.loc[summary["workflow"].eq("v161_safe_release_fixed") & summary["scope"].eq("all_domains")].iloc[0]
    all_v180 = summary.loc[
        summary["workflow"].eq("v180_nested_image_agreement_union") & summary["scope"].eq("all_domains")
    ].iloc[0]
    internal_v180 = summary.loc[
        summary["workflow"].eq("v180_nested_image_agreement_union") & summary["scope"].eq("internal_old_third")
    ].iloc[0]
    report = {
        "fold_count": int(len(selections)),
        "nested_internal_additional_release_n": int((nested_additional & internal).sum()),
        "nested_internal_additional_error_n": int((nested_additional & internal & base["base_wrong"]).sum()),
        "external_consensus_additional_release_n": int(ext_additional.sum()),
        "external_consensus_additional_error_n": int((ext_additional & base["base_wrong"]).sum()),
        "v161_all_domain_review_rate": float(all_v161["final_review_rate"]),
        "v180_all_domain_review_rate": float(all_v180["final_review_rate"]),
        "v180_all_domain_bacc": float(all_v180["balanced_accuracy"]),
        "v180_all_domain_released_errors": int(all_v180["released_error_n"]),
        "v180_internal_review_rate": float(internal_v180["final_review_rate"]),
        "v180_internal_released_errors": int(internal_v180["released_error_n"]),
        "v180_fn": int(all_v180["fn"]),
        "v180_fp": int(all_v180["fp"]),
    }
    (OUT_DIR / "v180_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# v180 Nested Validation for v178 Image-agreement Release",
        "",
        "## Purpose",
        "",
        "v178 selected an image-agreement release rule on all internal folds. v180 repeats the selection fold-wise: each held-out internal fold is released only by a rule selected from the other folds. Strict external uses a conservative all-fold consensus of the selected rules.",
        "",
        "## Result",
        "",
        (
            f"- Nested internal additional releases: {report['nested_internal_additional_release_n']}; "
            f"additional released errors: {report['nested_internal_additional_error_n']}."
        ),
        (
            f"- All-domain review rate: v161 {100 * report['v161_all_domain_review_rate']:.2f}% -> "
            f"v180 {100 * report['v180_all_domain_review_rate']:.2f}%."
        ),
        (
            f"- All-domain BAcc {100 * report['v180_all_domain_bacc']:.2f}%, released errors "
            f"{report['v180_all_domain_released_errors']}, FN={report['v180_fn']}, FP={report['v180_fp']}."
        ),
        "",
        "## Boundary",
        "",
        "This validates only the incremental image-agreement release on top of the fixed v161 safe-release workflow. It does not revalidate the original v161 rule selection.",
    ]
    (OUT_DIR / "v180_nested_v178_image_agreement_release.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v180] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
