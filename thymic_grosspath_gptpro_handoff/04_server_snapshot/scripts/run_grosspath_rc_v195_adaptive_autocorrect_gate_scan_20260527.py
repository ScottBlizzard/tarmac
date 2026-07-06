from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527"
V185_CASES = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_cases.csv"
V182_CASES = ROOT / "outputs" / "grosspath_rc_v182_stable_fixed_image_agreement_release_20260527" / "v182_stable_fixed_case_outputs.csv"
V173_CASES = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527" / "v173_corrector_case_outputs.csv"

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.985, 0.99, 0.995]
ACTION_MODES = ["agreement_release_only", "correction_only", "release_or_correct"]
DOMAINS = ["old_data", "third_batch", "strict_external", "all_domains"]


def load_base() -> pd.DataFrame:
    base = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    fold = pd.read_csv(V182_CASES, dtype={"case_id": str, "original_case_id": str})[
        ["case_id", "fold_id", "base_wrong"]
    ].copy()
    base = base.merge(fold, on="case_id", how="left", validate="one_to_one")
    for col in ["adaptive_review", "adaptive_auto_decision", "fixed_v118_review", "fixed_v182_review", "base_wrong"]:
        base[col] = as_bool(base[col])
    for col in ["label_idx", "final_pred", "fold_id"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(-1).astype(int)
    base["prob_mean_core"] = pd.to_numeric(base["prob_mean_core"], errors="coerce")
    base["base_wrong"] = base["final_pred"].astype(int).ne(base["label_idx"].astype(int))
    return base


def load_correctors() -> pd.DataFrame:
    corr = pd.read_csv(V173_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "corrector_pred"]:
        corr[col] = pd.to_numeric(corr[col], errors="coerce").fillna(-1).astype(int)
    corr["corrector_confidence"] = pd.to_numeric(corr["corrector_confidence"], errors="coerce")
    corr["corrector_prob_high"] = pd.to_numeric(corr["corrector_prob_high"], errors="coerce")
    corr["corrector_agrees_base"] = corr["corrector_pred"].eq(corr["final_pred"])
    return corr


def action_mask(df: pd.DataFrame, mode: str, threshold: float) -> pd.Series:
    high_conf = df["corrector_confidence"].ge(float(threshold)) & df["adaptive_review"]
    agree = df["corrector_pred"].eq(df["final_pred"])
    if mode == "agreement_release_only":
        return high_conf & agree
    if mode == "correction_only":
        return high_conf & ~agree
    if mode == "release_or_correct":
        return high_conf
    raise ValueError(mode)


def apply_candidate(df: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    action = action_mask(out, mode, threshold)
    out["auto_correct_action"] = action
    out["auto_correct_pred"] = out["final_pred"].astype(int)
    out.loc[action, "auto_correct_pred"] = out.loc[action, "corrector_pred"].astype(int)
    out["remaining_review"] = out["adaptive_review"] & ~out["auto_correct_action"]
    out["system_pred"] = out["auto_correct_pred"].astype(int)
    out.loc[out["remaining_review"], "system_pred"] = out.loc[out["remaining_review"], "label_idx"].astype(int)
    y = out["label_idx"].astype(int)
    base = out["final_pred"].astype(int)
    pred = out["auto_correct_pred"].astype(int)
    out["action_disagrees_base"] = out["auto_correct_action"] & pred.ne(base)
    out["action_agrees_base"] = out["auto_correct_action"] & pred.eq(base)
    out["action_error"] = out["auto_correct_action"] & pred.ne(y)
    out["rescued_by_action"] = out["auto_correct_action"] & base.ne(y) & pred.eq(y)
    out["hurt_by_action"] = out["auto_correct_action"] & base.eq(y) & pred.ne(y)
    out["rescued_fn"] = out["rescued_by_action"] & y.eq(1) & base.eq(0)
    out["rescued_fp"] = out["rescued_by_action"] & y.eq(0) & base.eq(1)
    out["hurt_to_fn"] = out["hurt_by_action"] & y.eq(1) & pred.eq(0)
    out["hurt_to_fp"] = out["hurt_by_action"] & y.eq(0) & pred.eq(1)
    return out


def summarize_scope(df: pd.DataFrame, scope: str, candidate_id: str, mode: str, threshold: float) -> dict[str, object]:
    if scope == "all_domains":
        sub = df.loc[df["domain"].isin(["old_data", "third_batch", "strict_external"])].copy()
    else:
        sub = df.loc[df["domain"].eq(scope)].copy()
    y = sub["label_idx"].astype(int).to_numpy()
    system_pred = sub["system_pred"].astype(int).to_numpy()
    base_pred = sub["final_pred"].astype(int).to_numpy()
    prob = sub["prob_mean_core"].to_numpy(float)
    mm = metrics(y, system_pred, prob)
    base_m = metrics(y, base_pred, prob)
    row = {
        "candidate_id": candidate_id,
        "source_review_policy": str(sub["review_policy"].iloc[0]) if len(sub) else "",
        "corrector_model": str(sub["model"].iloc[0]) if len(sub) else "",
        "feature_set": str(sub["feature_set"].iloc[0]) if len(sub) else "",
        "action_mode": mode,
        "threshold": float(threshold),
        "scope": scope,
        "n": int(len(sub)),
        "baseline_adaptive_review_n": int(sub["adaptive_review"].sum()),
        "baseline_adaptive_review_rate": float(sub["adaptive_review"].mean()) if len(sub) else np.nan,
        "auto_action_n": int(sub["auto_correct_action"].sum()),
        "auto_action_rate": float(sub["auto_correct_action"].mean()) if len(sub) else np.nan,
        "true_correction_n": int(sub["action_disagrees_base"].sum()),
        "agreement_release_n": int(sub["action_agrees_base"].sum()),
        "remaining_review_n": int(sub["remaining_review"].sum()),
        "remaining_review_rate": float(sub["remaining_review"].mean()) if len(sub) else np.nan,
        "action_error_n": int(sub["action_error"].sum()),
        "rescued_n": int(sub["rescued_by_action"].sum()),
        "hurt_n": int(sub["hurt_by_action"].sum()),
        "rescued_fn": int(sub["rescued_fn"].sum()),
        "rescued_fp": int(sub["rescued_fp"].sum()),
        "hurt_to_fn": int(sub["hurt_to_fn"].sum()),
        "hurt_to_fp": int(sub["hurt_to_fp"].sum()),
        "base_raw_errors": int(base_pred.astype(int).astype(object).shape[0] - (base_pred == y).sum()),
        "system_errors_after_review": int((system_pred != y).sum()),
        "base_balanced_accuracy_before_review": float(base_m["balanced_accuracy"]),
    }
    row.update({k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in mm.items()})
    return row


def audit_candidate(df: pd.DataFrame, candidate_id: str, mode: str, threshold: float) -> dict[str, object]:
    internal = df["domain"].isin(["old_data", "third_batch"])
    folds = sorted(int(x) for x in df.loc[internal, "fold_id"].dropna().unique() if int(x) >= 0)
    fold_rows = []
    for fold in folds:
        train = internal & df["fold_id"].ne(fold)
        held = internal & df["fold_id"].eq(fold)
        for split_name, mask in [("train_complement", train), ("heldout_fold", held)]:
            sub = df.loc[mask].copy()
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold_id": fold,
                    "split_name": split_name,
                    "action_mode": mode,
                    "threshold": float(threshold),
                    "auto_action_n": int(sub["auto_correct_action"].sum()),
                    "true_correction_n": int(sub["action_disagrees_base"].sum()),
                    "action_error_n": int(sub["action_error"].sum()),
                    "rescued_n": int(sub["rescued_by_action"].sum()),
                    "hurt_n": int(sub["hurt_by_action"].sum()),
                }
            )
    audit = pd.DataFrame(fold_rows)
    train = audit.loc[audit["split_name"].eq("train_complement")]
    held = audit.loc[audit["split_name"].eq("heldout_fold")]
    return {
        "candidate_id": candidate_id,
        "action_mode": mode,
        "threshold": float(threshold),
        "all_train_complements_zero_action_error": bool(
            len(train) > 0
            and train["action_error_n"].eq(0).all()
            and train["auto_action_n"].gt(0).all()
        ),
        "all_train_complements_have_true_correction": bool(
            len(train) > 0
            and train["true_correction_n"].gt(0).all()
        ),
        "min_train_auto_action_n": int(train["auto_action_n"].min()) if len(train) else 0,
        "min_train_true_correction_n": int(train["true_correction_n"].min()) if len(train) else 0,
        "heldout_auto_action_n": int(held["auto_action_n"].sum()) if len(held) else 0,
        "heldout_true_correction_n": int(held["true_correction_n"].sum()) if len(held) else 0,
        "heldout_action_error_n": int(held["action_error_n"].sum()) if len(held) else 0,
        "heldout_rescued_n": int(held["rescued_n"].sum()) if len(held) else 0,
        "heldout_hurt_n": int(held["hurt_n"].sum()) if len(held) else 0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()
    corr = load_correctors()
    merged = base.merge(
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
    rows = []
    audit_rows = []
    case_frames = []
    for (source_policy, model, feature_set), g in merged.groupby(["review_policy", "model", "feature_set"], sort=False):
        for mode in ACTION_MODES:
            for threshold in THRESHOLDS:
                cid = f"{source_policy}||{model}||{mode}||{threshold:.3f}"
                applied = apply_candidate(g, mode, threshold)
                audit_rows.append(audit_candidate(applied, cid, mode, threshold))
                for scope in DOMAINS:
                    rows.append(summarize_scope(applied, scope, cid, mode, threshold))
                slim = applied[
                    [
                        "domain",
                        "case_id",
                        "original_case_id",
                        "task_l6_label",
                        "label_idx",
                        "final_pred",
                        "prob_mean_core",
                        "image_name",
                        "adaptive_review",
                        "adaptive_policy_branch",
                        "review_policy",
                        "model",
                        "feature_set",
                        "corrector_prob_high",
                        "corrector_pred",
                        "corrector_confidence",
                        "auto_correct_action",
                        "action_disagrees_base",
                        "remaining_review",
                        "action_error",
                        "rescued_by_action",
                        "hurt_by_action",
                    ]
                ].copy()
                slim["candidate_id"] = cid
                slim["action_mode"] = mode
                slim["threshold"] = float(threshold)
                case_frames.append(slim)

    summary = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    internal = summary.loc[summary["scope"].isin(["old_data", "third_batch"])].copy()
    internal_pivot = internal.pivot_table(
        index="candidate_id",
        values=[
            "action_error_n",
            "auto_action_n",
            "true_correction_n",
            "remaining_review_rate",
            "balanced_accuracy",
            "rescued_n",
            "hurt_n",
        ],
        aggfunc={
            "action_error_n": "sum",
            "auto_action_n": "sum",
            "true_correction_n": "sum",
            "remaining_review_rate": "mean",
            "balanced_accuracy": "mean",
            "rescued_n": "sum",
            "hurt_n": "sum",
        },
    ).reset_index()
    candidate_rank = audit.merge(internal_pivot, on="candidate_id", how="left", validate="one_to_one")
    candidate_rank["lockable_zero_error_action"] = (
        candidate_rank["all_train_complements_zero_action_error"]
        & candidate_rank["heldout_action_error_n"].eq(0)
        & candidate_rank["heldout_auto_action_n"].gt(0)
    )
    candidate_rank["lockable_true_correction"] = (
        candidate_rank["lockable_zero_error_action"]
        & candidate_rank["heldout_true_correction_n"].gt(0)
        & candidate_rank["true_correction_n"].gt(0)
    )

    if candidate_rank["lockable_true_correction"].any():
        selected_kind = "lockable_true_correction"
        selected = candidate_rank.loc[candidate_rank["lockable_true_correction"]].sort_values(
            ["true_correction_n", "auto_action_n", "remaining_review_rate"],
            ascending=[False, False, True],
        ).iloc[0]
    elif candidate_rank["lockable_zero_error_action"].any():
        selected_kind = "lockable_zero_error_release"
        selected = candidate_rank.loc[candidate_rank["lockable_zero_error_action"]].sort_values(
            ["auto_action_n", "true_correction_n", "remaining_review_rate"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        selected_kind = "no_lockable_autocorrection"
        selected = candidate_rank.sort_values(
            ["all_train_complements_zero_action_error", "heldout_action_error_n", "auto_action_n"],
            ascending=[False, True, False],
        ).iloc[0]

    selected_summary = summary.loc[summary["candidate_id"].eq(selected["candidate_id"])].copy()
    selected_cases = pd.concat(case_frames, ignore_index=True)
    selected_cases = selected_cases.loc[selected_cases["candidate_id"].eq(selected["candidate_id"])].copy()
    selected_cases = selected_cases.loc[
        selected_cases["auto_correct_action"] | selected_cases["action_error"] | selected_cases["rescued_by_action"] | selected_cases["hurt_by_action"]
    ].copy()

    summary.to_csv(OUT_DIR / "v195_all_candidate_scope_summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT_DIR / "v195_candidate_fold_audit.csv", index=False, encoding="utf-8-sig")
    candidate_rank.sort_values(
        ["lockable_true_correction", "lockable_zero_error_action", "true_correction_n", "auto_action_n"],
        ascending=[False, False, False, False],
    ).to_csv(OUT_DIR / "v195_candidate_rank.csv", index=False, encoding="utf-8-sig")
    selected_summary.to_csv(OUT_DIR / "v195_selected_candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected_cases.to_csv(OUT_DIR / "v195_selected_candidate_action_cases.csv", index=False, encoding="utf-8-sig")

    all_row = selected_summary.loc[selected_summary["scope"].eq("all_domains")].iloc[0]
    strict_row = selected_summary.loc[selected_summary["scope"].eq("strict_external")].iloc[0]
    report = {
        "selection_kind": selected_kind,
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_action_mode": str(selected["action_mode"]),
        "selected_threshold": float(selected["threshold"]),
        "lockable_true_correction_count": int(candidate_rank["lockable_true_correction"].sum()),
        "lockable_zero_error_action_count": int(candidate_rank["lockable_zero_error_action"].sum()),
        "all_domain_bacc": float(all_row["balanced_accuracy"]),
        "all_domain_remaining_review_rate": float(all_row["remaining_review_rate"]),
        "all_domain_action_error_n": int(all_row["action_error_n"]),
        "all_domain_true_correction_n": int(all_row["true_correction_n"]),
        "all_domain_rescued_n": int(all_row["rescued_n"]),
        "all_domain_hurt_n": int(all_row["hurt_n"]),
        "strict_external_bacc": float(strict_row["balanced_accuracy"]),
        "strict_external_remaining_review_rate": float(strict_row["remaining_review_rate"]),
        "strict_external_action_error_n": int(strict_row["action_error_n"]),
    }
    (OUT_DIR / "v195_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# v195 Adaptive Autocorrect Gate Scan",
        "",
        "## Purpose",
        "",
        "v194 showed that the current framework has strong selective safety but weak evidence for automatic correction. v195 scans high-confidence second-stage actions inside the v185 adaptive review pool and separates three action types: agreement release, correction-only, and release-or-correct.",
        "",
        "## Selected Result",
        "",
        f"- Selection kind: `{selected_kind}`.",
        f"- Selected candidate: `{report['selected_candidate_id']}`.",
        f"- All-domain BAcc: {100 * report['all_domain_bacc']:.2f}%; remaining review/reject: {100 * report['all_domain_remaining_review_rate']:.2f}%.",
        f"- All-domain auto-action errors: {report['all_domain_action_error_n']}; true corrections: {report['all_domain_true_correction_n']}; rescued: {report['all_domain_rescued_n']}; hurt: {report['all_domain_hurt_n']}.",
        f"- Strict external BAcc: {100 * report['strict_external_bacc']:.2f}%; remaining review/reject: {100 * report['strict_external_remaining_review_rate']:.2f}%; action errors: {report['strict_external_action_error_n']}.",
        "",
        "## Interpretation",
        "",
        "If the selected kind is `lockable_true_correction`, we have a candidate automatic correction module with fold-wise zero-action-error evidence. If it falls back to `lockable_zero_error_release`, the evidence supports safer release but not true correction. If it is `no_lockable_autocorrection`, automatic correction remains a research gap and should not be written as a solved module.",
    ]
    (OUT_DIR / "v195_adaptive_autocorrect_gate_scan.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v195] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
