from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v178_safe_release_union_with_image_agreement_20260527"
V161_CASES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_cases.csv"
V173_CASES = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527" / "v173_corrector_case_outputs.csv"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_base() -> pd.DataFrame:
    df = pd.read_csv(V161_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v118_review_or_control", "v161_safe_release_from_review", "v161_final_review_or_reject", "base_wrong"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "fold_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in ["prob_mean_core", "wholecrop_prob", "main_prob", "robust_prob"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    return df


def load_corrector() -> pd.DataFrame:
    df = pd.read_csv(V173_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "corrector_pred", "fold_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in ["corrector_prob_high", "corrector_confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["corrector_agrees_base"] = df["corrector_pred"].eq(df["final_pred"])
    return df


def summarize(base: pd.DataFrame, union_release: pd.Series, workflow: str, candidate_release: pd.Series) -> list[dict[str, object]]:
    y_all = base["label_idx"].to_numpy(int)
    pred = base["final_pred"].to_numpy(int).copy()
    final_review = base["v118_review_or_control"] & ~union_release
    pred[final_review.to_numpy(bool)] = y_all[final_review.to_numpy(bool)]
    released_error = union_release & base["base_wrong"]
    additional_error = candidate_release & ~base["v161_safe_release_from_review"] & base["base_wrong"]
    rows = []
    for scope, mask in [
        ("old_data", base["domain"].eq("old_data")),
        ("third_batch", base["domain"].eq("third_batch")),
        ("strict_external", base["domain"].eq("strict_external")),
        ("all_domains", base["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        m = metrics(y_all[mask.to_numpy(bool)], pred[mask.to_numpy(bool)], base.loc[mask, "prob_mean_core"].to_numpy(float))
        rows.append(
            {
                "workflow": workflow,
                "scope": scope,
                "n": int(mask.sum()),
                "auto_release_n": int(union_release[mask].sum()),
                "auto_release_rate": float(union_release[mask].mean()),
                "additional_image_release_n": int((candidate_release & ~base["v161_safe_release_from_review"])[mask].sum()),
                "final_review_n": int(final_review[mask].sum()),
                "final_review_rate": float(final_review[mask].mean()),
                "released_error_n": int(released_error[mask].sum()),
                "additional_released_error_n": int(additional_error[mask].sum()),
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
    corr = load_corrector()
    internal = base["domain"].isin(["old_data", "third_batch"])

    rows = []
    summary_rows = []
    selected_rows = []
    case_frames = []

    baseline_release = base["v161_safe_release_from_review"].copy()
    summary_rows += summarize(base, baseline_release, "v161_safe_release", pd.Series(False, index=base.index))

    for (review_policy, model), g in corr.groupby(["review_policy", "model"], sort=False):
        g = g.copy()
        merged = base[["case_id", "domain", "base_wrong", "v161_safe_release_from_review"]].merge(
            g[["case_id", "corrector_agrees_base", "corrector_confidence"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        for threshold in THRESHOLDS:
            candidate_release = merged["corrector_agrees_base"].fillna(False).astype(bool) & merged[
                "corrector_confidence"
            ].ge(threshold)
            # Candidate releases are only allowed inside the corresponding review policy's review pool.
            if review_policy in base.columns:
                candidate_release &= base[review_policy]
            else:
                continue
            additional = candidate_release & ~base["v161_safe_release_from_review"]
            add_internal = additional & internal
            union_release = base["v161_safe_release_from_review"] | candidate_release
            rows.append(
                {
                    "review_policy": review_policy,
                    "model": model,
                    "threshold": float(threshold),
                    "candidate_release_n": int(candidate_release.sum()),
                    "additional_release_n": int(additional.sum()),
                    "additional_internal_release_n": int(add_internal.sum()),
                    "additional_internal_released_error_n": int((add_internal & base["base_wrong"]).sum()),
                    "additional_external_release_n": int((additional & base["domain"].eq("strict_external")).sum()),
                    "additional_external_released_error_n": int(
                        (additional & base["domain"].eq("strict_external") & base["base_wrong"]).sum()
                    ),
                    "union_release_n": int(union_release.sum()),
                    "union_released_error_n": int((union_release & base["base_wrong"]).sum()),
                    "final_review_n": int((base["v118_review_or_control"] & ~union_release).sum()),
                    "final_review_rate": float((base["v118_review_or_control"] & ~union_release).mean()),
                }
            )
            workflow = f"v178_union_{review_policy}_{model}_t{threshold:.2f}"
            summary_rows += summarize(base, union_release, workflow, candidate_release)
            tmp = base[
                [
                    "domain",
                    "case_id",
                    "original_case_id",
                    "task_l6_label",
                    "label_idx",
                    "final_pred",
                    "image_name",
                    "v161_safe_release_from_review",
                    "v118_review_or_control",
                    "base_wrong",
                ]
            ].copy()
            tmp["review_policy"] = review_policy
            tmp["model"] = model
            tmp["threshold"] = float(threshold)
            tmp["image_agreement_release"] = candidate_release
            tmp["additional_release_over_v161"] = additional
            tmp["union_release"] = union_release
            tmp["union_released_error"] = union_release & base["base_wrong"]
            case_frames.append(tmp)

    scan = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    cases = pd.concat(case_frames, ignore_index=True) if case_frames else pd.DataFrame()

    # Selection uses old+third only: zero additional internal release error, then maximize additional internal release.
    candidates = scan.loc[
        scan["additional_internal_released_error_n"].eq(0) & scan["additional_internal_release_n"].gt(0)
    ].copy()
    if not candidates.empty:
        chosen = candidates.sort_values(
            ["additional_internal_release_n", "final_review_rate", "additional_external_released_error_n"],
            ascending=[False, True, True],
        ).iloc[0]
        selected_rows.append(chosen.to_dict() | {"selection_status": "internal_zero_additional_release_error"})
        selected_workflow = f"v178_union_{chosen['review_policy']}_{chosen['model']}_t{float(chosen['threshold']):.2f}"
    else:
        selected_workflow = "v161_safe_release"

    selected = pd.DataFrame(selected_rows)
    selected_summary = summary.loc[summary["workflow"].isin(["v161_safe_release", selected_workflow])].copy()
    selected_cases = cases.loc[cases["review_policy"].eq(selected.iloc[0]["review_policy"]) & cases["model"].eq(selected.iloc[0]["model"]) & np.isclose(cases["threshold"], float(selected.iloc[0]["threshold"]))].copy() if not selected.empty else pd.DataFrame()

    scan.to_csv(OUT_DIR / "v178_image_agreement_release_scan.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v178_all_workflow_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v178_selected_internal_zero_error_rule.csv", index=False, encoding="utf-8-sig")
    selected_summary.to_csv(OUT_DIR / "v178_selected_workflow_summary.csv", index=False, encoding="utf-8-sig")
    selected_cases.to_csv(OUT_DIR / "v178_selected_case_outputs.csv", index=False, encoding="utf-8-sig")

    all_rows = selected_summary.loc[selected_summary["scope"].eq("all_domains")].copy()
    base_all = all_rows.loc[all_rows["workflow"].eq("v161_safe_release")].iloc[0]
    new_all = all_rows.loc[all_rows["workflow"].ne("v161_safe_release")].iloc[0] if (all_rows["workflow"].ne("v161_safe_release")).any() else base_all

    md = [
        "# v178 Safe-release Union With Image Agreement",
        "",
        "## Purpose",
        "",
        "This experiment asks whether an image-only agreement signal can safely release additional reviewed cases on top of v161 safe-release. Selection uses old+third internal zero additional-release-error only; strict external is observed after selection.",
        "",
        "## Selected Result",
        "",
    ]
    if not selected.empty:
        r = selected.iloc[0]
        md.append(
            f"- Selected `{r['review_policy']}` / `{r['model']}` at threshold {float(r['threshold']):.2f}; "
            f"additional internal releases {int(r['additional_internal_release_n'])}, internal added errors 0."
        )
    md.append(
        f"- All-domain review rate changes from {100 * float(base_all['final_review_rate']):.2f}% to "
        f"{100 * float(new_all['final_review_rate']):.2f}%; released errors remain {int(new_all['released_error_n'])}."
    )
    md.append(
        f"- All-domain BAcc {100 * float(new_all['balanced_accuracy']):.2f}%, FN={int(new_all['fn'])}, FP={int(new_all['fp'])}."
    )
    md += [
        "",
        "## Boundary",
        "",
        "The gain is modest, so this should be written as a conservative efficiency refinement rather than a new main module.",
    ]
    (OUT_DIR / "v178_safe_release_union_with_image_agreement.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "scan_rows": int(len(scan)),
        "selected": selected.iloc[0].to_dict() if not selected.empty else None,
        "v161_all_domain_review_rate": float(base_all["final_review_rate"]),
        "v178_all_domain_review_rate": float(new_all["final_review_rate"]),
        "v178_all_domain_bacc": float(new_all["balanced_accuracy"]),
        "v178_released_errors": int(new_all["released_error_n"]),
        "v178_fn": int(new_all["fn"]),
        "v178_fp": int(new_all["fp"]),
    }
    (OUT_DIR / "v178_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v178] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
