from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527 import (
    ACTION_MODES,
    THRESHOLDS,
    apply_candidate,
    load_base,
    load_correctors,
)


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v196_nested_adaptive_release_validation_20260527"


def merged_inputs() -> pd.DataFrame:
    base = load_base()
    corr = load_correctors()
    return base.merge(
        corr[
            [
                "case_id",
                "review_policy",
                "model",
                "feature_set",
                "corrector_prob_high",
                "corrector_pred",
                "corrector_confidence",
            ]
        ],
        on="case_id",
        how="left",
        validate="one_to_many",
    )


def candidate_frames(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame]] = []
    for (source_policy, model, feature_set), g in df.groupby(["review_policy", "model", "feature_set"], sort=False):
        for mode in ACTION_MODES:
            for threshold in THRESHOLDS:
                cid = f"{source_policy}||{model}||{mode}||{threshold:.3f}"
                applied = apply_candidate(g, mode, threshold)
                applied["candidate_id"] = cid
                applied["source_review_policy"] = source_policy
                applied["corrector_model"] = model
                applied["feature_set"] = feature_set
                applied["action_mode"] = mode
                applied["threshold"] = float(threshold)
                out.append((cid, applied))
    return out


def action_stats(df: pd.DataFrame, mask: pd.Series) -> dict[str, int | float]:
    sub = df.loc[mask].copy()
    return {
        "n": int(len(sub)),
        "auto_action_n": int(sub["auto_correct_action"].sum()),
        "agreement_release_n": int(sub["action_agrees_base"].sum()),
        "correction_attempt_n": int(sub["action_disagrees_base"].sum()),
        "action_error_n": int(sub["action_error"].sum()),
        "rescued_n": int(sub["rescued_by_action"].sum()),
        "hurt_n": int(sub["hurt_by_action"].sum()),
        "remaining_review_n": int(sub["remaining_review"].sum()),
        "remaining_review_rate": float(sub["remaining_review"].mean()) if len(sub) else np.nan,
    }


def select_on_train(candidates: list[tuple[str, pd.DataFrame]], train_mask_by_case: pd.Series) -> tuple[str, pd.DataFrame, dict[str, object]]:
    rows = []
    for cid, applied in candidates:
        mask = applied["case_id"].map(train_mask_by_case).fillna(False).astype(bool)
        s = action_stats(applied, mask)
        rows.append(
            {
                "candidate_id": cid,
                "source_review_policy": str(applied["source_review_policy"].iloc[0]),
                "corrector_model": str(applied["corrector_model"].iloc[0]),
                "feature_set": str(applied["feature_set"].iloc[0]),
                "action_mode": str(applied["action_mode"].iloc[0]),
                "threshold": float(applied["threshold"].iloc[0]),
                **{f"train_{k}": v for k, v in s.items()},
            }
        )
    rank = pd.DataFrame(rows)
    safe = rank.loc[rank["train_action_error_n"].eq(0) & rank["train_auto_action_n"].gt(0)].copy()
    if safe.empty:
        chosen = rank.sort_values(["train_action_error_n", "train_auto_action_n"], ascending=[True, False]).iloc[0]
        status = "no_zero_error_train_candidate"
    else:
        correction_safe = safe.loc[safe["train_rescued_n"].gt(0) & safe["train_hurt_n"].eq(0)].copy()
        if not correction_safe.empty:
            chosen = correction_safe.sort_values(
                ["train_rescued_n", "train_auto_action_n", "train_remaining_review_rate"],
                ascending=[False, False, True],
            ).iloc[0]
            status = "zero_error_train_with_rescue"
        else:
            agreement = safe.loc[safe["action_mode"].eq("agreement_release_only")].copy()
            pool = agreement if not agreement.empty else safe
            chosen = pool.sort_values(
                ["train_auto_action_n", "train_remaining_review_rate", "threshold"],
                ascending=[False, True, False],
            ).iloc[0]
            status = "zero_error_train_release_only"
    chosen_id = str(chosen["candidate_id"])
    applied = next(frame for cid, frame in candidates if cid == chosen_id)
    return chosen_id, applied, {**chosen.to_dict(), "selection_status": status}


def summarize_system(df: pd.DataFrame, scope_name: str, mask: pd.Series) -> dict[str, object]:
    sub = df.loc[mask].copy()
    y = sub["label_idx"].astype(int).to_numpy()
    pred = sub["system_pred"].astype(int).to_numpy()
    base = sub["final_pred"].astype(int).to_numpy()
    prob = sub["prob_mean_core"].to_numpy(float)
    m = metrics(y, pred, prob)
    base_m = metrics(y, base, prob)
    row = {
        "scope": scope_name,
        "n": int(len(sub)),
        "baseline_adaptive_review_n": int(sub["adaptive_review"].sum()),
        "baseline_adaptive_review_rate": float(sub["adaptive_review"].mean()) if len(sub) else np.nan,
        "auto_action_n": int(sub["auto_correct_action"].sum()),
        "agreement_release_n": int(sub["action_agrees_base"].sum()),
        "correction_attempt_n": int(sub["action_disagrees_base"].sum()),
        "action_error_n": int(sub["action_error"].sum()),
        "rescued_n": int(sub["rescued_by_action"].sum()),
        "hurt_n": int(sub["hurt_by_action"].sum()),
        "remaining_review_n": int(sub["remaining_review"].sum()),
        "remaining_review_rate": float(sub["remaining_review"].mean()) if len(sub) else np.nan,
        "base_raw_error_n": int((base != y).sum()),
        "base_raw_balanced_accuracy": float(base_m["balanced_accuracy"]),
    }
    row.update({k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in m.items()})
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = merged_inputs()
    candidates = candidate_frames(merged)
    base_unique = load_base()
    internal_cases = base_unique.loc[base_unique["domain"].isin(["old_data", "third_batch"])].copy()
    folds = sorted(int(x) for x in internal_cases["fold_id"].dropna().unique() if int(x) >= 0)
    nested_frames = []
    selection_rows = []

    for fold in folds:
        train_cases = internal_cases.loc[internal_cases["fold_id"].ne(fold), "case_id"]
        held_cases = internal_cases.loc[internal_cases["fold_id"].eq(fold), "case_id"]
        train_mask_by_case = pd.Series(False, index=base_unique["case_id"].astype(str))
        train_mask_by_case.loc[train_cases.astype(str)] = True
        selected_id, applied, chosen = select_on_train(candidates, train_mask_by_case)
        chosen["heldout_fold"] = int(fold)
        held = applied.loc[applied["case_id"].isin(held_cases)].copy()
        held["selection_source"] = "nested_internal"
        held["selected_candidate_id"] = selected_id
        nested_frames.append(held)
        held_stats = action_stats(applied, applied["case_id"].isin(held_cases))
        selection_rows.append({**chosen, **{f"heldout_{k}": v for k, v in held_stats.items()}})

    full_internal_mask_by_case = pd.Series(False, index=base_unique["case_id"].astype(str))
    full_internal_mask_by_case.loc[internal_cases["case_id"].astype(str)] = True
    external_id, external_applied, external_chosen = select_on_train(candidates, full_internal_mask_by_case)
    strict_cases = base_unique.loc[base_unique["domain"].eq("strict_external"), "case_id"]
    strict = external_applied.loc[external_applied["case_id"].isin(strict_cases)].copy()
    strict["selection_source"] = "locked_external_from_full_internal"
    strict["selected_candidate_id"] = external_id
    nested_frames.append(strict)
    external_chosen["heldout_fold"] = -1
    strict_stats = action_stats(external_applied, external_applied["case_id"].isin(strict_cases))
    selection_rows.append({**external_chosen, **{f"heldout_{k}": v for k, v in strict_stats.items()}})

    cases = pd.concat(nested_frames, ignore_index=True)
    cases["system_pred"] = cases["auto_correct_pred"].astype(int)
    cases.loc[cases["remaining_review"], "system_pred"] = cases.loc[cases["remaining_review"], "label_idx"].astype(int)

    summary_rows = []
    for scope, mask in [
        ("old_data_nested", cases["domain"].eq("old_data")),
        ("third_batch_nested", cases["domain"].eq("third_batch")),
        ("internal_nested_old_third", cases["domain"].isin(["old_data", "third_batch"])),
        ("strict_external_locked", cases["domain"].eq("strict_external")),
        ("all_domains_nested_plus_locked_external", cases["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        summary_rows.append(summarize_system(cases, scope, mask))
    summary = pd.DataFrame(summary_rows)
    selections = pd.DataFrame(selection_rows)

    cases.to_csv(OUT_DIR / "v196_nested_release_cases.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(OUT_DIR / "v196_nested_fold_selected_rules.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v196_nested_release_summary.csv", index=False, encoding="utf-8-sig")

    all_row = summary.loc[summary["scope"].eq("all_domains_nested_plus_locked_external")].iloc[0]
    internal_row = summary.loc[summary["scope"].eq("internal_nested_old_third")].iloc[0]
    strict_row = summary.loc[summary["scope"].eq("strict_external_locked")].iloc[0]
    report = {
        "selection_rule": "For each held-out internal fold, choose a zero-action-error rule on the remaining old+third folds, maximizing auto actions; strict external uses a rule chosen on all internal folds.",
        "all_domain_bacc": float(all_row["balanced_accuracy"]),
        "all_domain_remaining_review_rate": float(all_row["remaining_review_rate"]),
        "all_domain_action_error_n": int(all_row["action_error_n"]),
        "all_domain_auto_action_n": int(all_row["auto_action_n"]),
        "all_domain_correction_attempt_n": int(all_row["correction_attempt_n"]),
        "internal_bacc": float(internal_row["balanced_accuracy"]),
        "internal_remaining_review_rate": float(internal_row["remaining_review_rate"]),
        "internal_action_error_n": int(internal_row["action_error_n"]),
        "strict_external_bacc": float(strict_row["balanced_accuracy"]),
        "strict_external_remaining_review_rate": float(strict_row["remaining_review_rate"]),
        "strict_external_action_error_n": int(strict_row["action_error_n"]),
        "selected_external_candidate_id": str(external_id),
    }
    (OUT_DIR / "v196_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# v196 Nested Adaptive Release Validation",
        "",
        "## Purpose",
        "",
        "v195 found a strong high-confidence agreement release rule, but v196 checks whether this type of rule remains safe when each internal fold selects its rule without seeing the held-out fold labels.",
        "",
        "## Result",
        "",
        f"- All-domain nested+locked BAcc: {100 * report['all_domain_bacc']:.2f}%; remaining review/reject: {100 * report['all_domain_remaining_review_rate']:.2f}%.",
        f"- All-domain auto actions: {report['all_domain_auto_action_n']}; correction attempts: {report['all_domain_correction_attempt_n']}; action errors: {report['all_domain_action_error_n']}.",
        f"- Internal nested BAcc: {100 * report['internal_bacc']:.2f}%; internal remaining review/reject: {100 * report['internal_remaining_review_rate']:.2f}%; action errors: {report['internal_action_error_n']}.",
        f"- Strict external locked BAcc: {100 * report['strict_external_bacc']:.2f}%; remaining review/reject: {100 * report['strict_external_remaining_review_rate']:.2f}%; action errors: {report['strict_external_action_error_n']}.",
        "",
        "## Interpretation",
        "",
        "If v196 matches v195 closely, the paper can claim a nested-validated high-confidence agreement-release module. If correction attempts remain near zero, the automatic correction claim should remain bounded: the current mature module is safe release and rejection compression, not autonomous label flipping.",
    ]
    (OUT_DIR / "v196_nested_adaptive_release_validation.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v196] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
