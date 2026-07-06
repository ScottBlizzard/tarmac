from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v190_fn_sentinel_nested_scan_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"
V185_CASES = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_cases.csv"


@dataclass(frozen=True)
class Candidate:
    name: str
    mask: pd.Series
    description: str


def load_cases() -> pd.DataFrame:
    v118 = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    v185 = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    keep_v118 = [
        "case_id",
        "main_prob",
        "robust_prob",
        "wholecrop_prob",
        "v105_crop_prob",
        "core_agree_count",
        "fold_id",
        "view_type_final",
        "shift_category",
    ]
    df = v185.merge(v118[keep_v118], on="case_id", how="inner", validate="one_to_one")
    for col in ["fixed_v118_review", "fixed_v182_review", "adaptive_review", "adaptive_auto_decision"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "fold_id", "core_agree_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in ["prob_mean_core", "main_prob", "robust_prob", "wholecrop_prob", "v105_crop_prob"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    df["error_direction"] = "correct"
    df.loc[df["label_idx"].eq(1) & df["final_pred"].eq(0), "error_direction"] = "FN_high_to_low"
    df.loc[df["label_idx"].eq(0) & df["final_pred"].eq(1), "error_direction"] = "FP_low_to_high"
    df["auto_lowrisk"] = df["adaptive_auto_decision"] & df["final_pred"].eq(0)
    df["pm_minus_whole"] = df["prob_mean_core"] - df["wholecrop_prob"]
    df["pm_minus_main"] = df["prob_mean_core"] - df["main_prob"]
    df["robust_minus_main"] = df["robust_prob"] - df["main_prob"]
    df["crop_minus_whole"] = df["v105_crop_prob"] - df["wholecrop_prob"]
    return df.reset_index(drop=True)


def make_candidates(df: pd.DataFrame) -> list[Candidate]:
    base = df["auto_lowrisk"]
    candidates: list[Candidate] = []

    def add(name: str, mask: pd.Series, desc: str) -> None:
        candidates.append(Candidate(name=name, mask=(base & mask), description=desc))

    for t in [0.25, 0.30, 0.35, 0.40, 0.425, 0.45, 0.50, 0.55]:
        add(f"pm_ge_{t:.3f}", df["prob_mean_core"].ge(t), f"auto-lowrisk and prob_mean_core >= {t:.3f}")
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        add(f"robust_ge_{t:.3f}", df["robust_prob"].ge(t), f"auto-lowrisk and robust_prob >= {t:.3f}")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        add(f"main_ge_{t:.3f}", df["main_prob"].ge(t), f"auto-lowrisk and main_prob >= {t:.3f}")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35]:
        add(f"crop_ge_{t:.3f}", df["v105_crop_prob"].ge(t), f"auto-lowrisk and v105_crop_prob >= {t:.3f}")

    for p in [0.30, 0.35, 0.40, 0.425, 0.45]:
        for agree in [1, 2]:
            add(
                f"pm_ge_{p:.3f}_agree_le_{agree}",
                df["prob_mean_core"].ge(p) & df["core_agree_count"].le(agree),
                f"auto-lowrisk, prob_mean_core >= {p:.3f}, core_agree_count <= {agree}",
            )
    for p in [0.30, 0.35, 0.40, 0.425]:
        for w in [0.15, 0.20, 0.25, 0.30, 0.40]:
            add(
                f"pm_ge_{p:.3f}_whole_le_{w:.2f}",
                df["prob_mean_core"].ge(p) & df["wholecrop_prob"].le(w),
                f"auto-lowrisk, prob_mean_core >= {p:.3f}, wholecrop_prob <= {w:.2f}",
            )
    for r in [0.30, 0.35, 0.40]:
        for w in [0.15, 0.20, 0.25, 0.30, 0.40]:
            add(
                f"robust_ge_{r:.3f}_whole_le_{w:.2f}",
                df["robust_prob"].ge(r) & df["wholecrop_prob"].le(w),
                f"auto-lowrisk, robust_prob >= {r:.3f}, wholecrop_prob <= {w:.2f}",
            )
    for d in [0.10, 0.15, 0.20, 0.25, 0.30]:
        add(
            f"pm_minus_whole_ge_{d:.2f}",
            df["pm_minus_whole"].ge(d),
            f"auto-lowrisk and prob_mean_core - wholecrop_prob >= {d:.2f}",
        )
    for d in [0.05, 0.10, 0.15, 0.20]:
        add(
            f"pm_minus_main_ge_{d:.2f}",
            df["pm_minus_main"].ge(d),
            f"auto-lowrisk and prob_mean_core - main_prob >= {d:.2f}",
        )
        add(
            f"robust_minus_main_ge_{d:.2f}",
            df["robust_minus_main"].ge(d),
            f"auto-lowrisk and robust_prob - main_prob >= {d:.2f}",
        )
        add(
            f"crop_minus_whole_ge_{d:.2f}",
            df["crop_minus_whole"].ge(d),
            f"auto-lowrisk and v105_crop_prob - wholecrop_prob >= {d:.2f}",
        )
    return candidates


def candidate_stats(df: pd.DataFrame, mask: pd.Series, scope_mask: pd.Series) -> dict[str, int]:
    m = mask & scope_mask
    return {
        "review_n": int(m.sum()),
        "clean_review_n": int((m & ~df["base_wrong"]).sum()),
        "rescued_fn_n": int((m & df["error_direction"].eq("FN_high_to_low")).sum()),
        "rescued_fp_n": int((m & df["error_direction"].eq("FP_low_to_high")).sum()),
        "rescued_error_n": int((m & df["base_wrong"]).sum()),
    }


def scan_candidates(df: pd.DataFrame, candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    scopes = {
        "old_data": df["domain"].eq("old_data"),
        "third_batch": df["domain"].eq("third_batch"),
        "strict_external": df["domain"].eq("strict_external"),
        "internal_old_third": df["domain"].isin(["old_data", "third_batch"]),
        "all_domains": df["domain"].isin(["old_data", "third_batch", "strict_external"]),
    }
    for c in candidates:
        row = {"candidate": c.name, "description": c.description}
        for scope, smask in scopes.items():
            stats = candidate_stats(df, c.mask, smask)
            for k, v in stats.items():
                row[f"{scope}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def choose_candidate_for_train(df: pd.DataFrame, candidates: list[Candidate], train_mask: pd.Series) -> Candidate | None:
    rows = []
    for c in candidates:
        stats = candidate_stats(df, c.mask, train_mask)
        if stats["rescued_fn_n"] <= 0:
            continue
        rows.append({"candidate": c, **stats})
    if not rows:
        return None
    ranked = sorted(
        rows,
        key=lambda x: (
            -int(x["rescued_fn_n"]),
            int(x["clean_review_n"]),
            int(x["review_n"]),
            str(x["candidate"].name),
        ),
    )
    return ranked[0]["candidate"]


def summarize_workflow(df: pd.DataFrame, sentinel_review: pd.Series, workflow: str) -> list[dict[str, object]]:
    review = df["adaptive_review"] | sentinel_review
    y = df["label_idx"].to_numpy(int)
    pred = df["final_pred"].to_numpy(int).copy()
    pred[review.to_numpy(bool)] = y[review.to_numpy(bool)]
    rows = []
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("strict_external", df["domain"].eq("strict_external")),
        ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        m = metrics(y[mask.to_numpy(bool)], pred[mask.to_numpy(bool)], df.loc[mask, "prob_mean_core"].to_numpy(float))
        add = sentinel_review & ~df["adaptive_review"]
        rows.append(
            {
                "workflow": workflow,
                "scope": scope,
                "n": int(mask.sum()),
                "additional_sentinel_review_n": int((add & mask).sum()),
                "additional_sentinel_clean_review_n": int((add & mask & ~df["base_wrong"]).sum()),
                "sentinel_rescued_error_n": int((add & mask & df["base_wrong"]).sum()),
                "sentinel_rescued_fn_n": int((add & mask & df["error_direction"].eq("FN_high_to_low")).sum()),
                "review_or_reject_n": int((review & mask).sum()),
                "review_or_reject_rate": float((review & mask).mean()),
                "auto_decision_n": int((~review & mask).sum()),
                "auto_error_n": int((~review & mask & df["base_wrong"]).sum()),
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
    df = load_cases()
    candidates = make_candidates(df)
    scan = scan_candidates(df, candidates)
    scan.to_csv(OUT_DIR / "v190_fn_sentinel_candidate_scan.csv", index=False, encoding="utf-8-sig")

    internal = df["domain"].isin(["old_data", "third_batch"])
    nested_review = pd.Series(False, index=df.index)
    selections = []
    for fold in sorted(df.loc[internal, "fold_id"].unique()):
        train = internal & df["fold_id"].ne(int(fold))
        held = internal & df["fold_id"].eq(int(fold))
        chosen = choose_candidate_for_train(df, candidates, train)
        if chosen is None:
            selections.append({"fold_id": int(fold), "candidate": "", "description": "", "selection_status": "no_train_rescue_candidate"})
            continue
        nested_review |= chosen.mask & held
        stats_train = candidate_stats(df, chosen.mask, train)
        stats_held = candidate_stats(df, chosen.mask, held)
        selections.append(
            {
                "fold_id": int(fold),
                "candidate": chosen.name,
                "description": chosen.description,
                "selection_status": "selected_by_train_fn_rescue_then_min_clean_review",
                **{f"train_{k}": v for k, v in stats_train.items()},
                **{f"heldout_{k}": v for k, v in stats_held.items()},
            }
        )

    selected_names = {s["candidate"] for s in selections if s.get("candidate")}
    external_review = pd.Series(False, index=df.index)
    if selected_names:
        # Conservative external application: review strict-external auto-lowrisk cases if any fold-selected sentinel fires.
        name_to_candidate = {c.name: c for c in candidates}
        ext = df["domain"].eq("strict_external")
        for name in selected_names:
            external_review |= name_to_candidate[name].mask & ext
    sentinel_review = nested_review | external_review

    full_fit = choose_candidate_for_train(df, candidates, internal)
    full_fit_review = full_fit.mask if full_fit is not None else pd.Series(False, index=df.index)

    summary = pd.DataFrame(
        summarize_workflow(df, pd.Series(False, index=df.index), "v185_adaptive_baseline")
        + summarize_workflow(df, sentinel_review, "v190_nested_fn_sentinel")
        + summarize_workflow(df, full_fit_review, "v190_fullfit_oracle_style_sentinel")
    )
    summary.to_csv(OUT_DIR / "v190_fn_sentinel_workflow_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(OUT_DIR / "v190_fold_selected_sentinels.csv", index=False, encoding="utf-8-sig")

    cases = df[
        [
            "domain",
            "case_id",
            "original_case_id",
            "task_l6_label",
            "label_idx",
            "final_pred",
            "error_direction",
            "fold_id",
            "prob_mean_core",
            "main_prob",
            "robust_prob",
            "wholecrop_prob",
            "v105_crop_prob",
            "core_agree_count",
            "image_name",
            "adaptive_review",
            "adaptive_auto_decision",
            "auto_lowrisk",
            "base_wrong",
        ]
    ].copy()
    cases["v190_nested_sentinel_review"] = sentinel_review
    cases["v190_nested_additional_review"] = sentinel_review & ~df["adaptive_review"]
    cases["v190_fullfit_sentinel_review"] = full_fit_review
    cases["v190_fullfit_additional_review"] = full_fit_review & ~df["adaptive_review"]
    cases.to_csv(OUT_DIR / "v190_fn_sentinel_case_outputs.csv", index=False, encoding="utf-8-sig")

    all_base = summary.loc[summary["workflow"].eq("v185_adaptive_baseline") & summary["scope"].eq("all_domains")].iloc[0]
    all_nested = summary.loc[summary["workflow"].eq("v190_nested_fn_sentinel") & summary["scope"].eq("all_domains")].iloc[0]
    residual = cases.loc[cases["original_case_id"].astype(str).eq("2516531")].iloc[0]
    report = {
        "candidate_count": int(len(candidates)),
        "fold_selection_count": int(len(selections)),
        "selected_candidates": sorted(selected_names),
        "baseline_auto_error_n": int(all_base["auto_error_n"]),
        "nested_auto_error_n": int(all_nested["auto_error_n"]),
        "nested_additional_review_n": int(all_nested["additional_sentinel_review_n"]),
        "nested_sentinel_rescued_fn_n": int(all_nested["sentinel_rescued_fn_n"]),
        "nested_review_rate": float(all_nested["review_or_reject_rate"]),
        "nested_bacc": float(all_nested["balanced_accuracy"]),
        "case_2516531_nested_reviewed": bool(residual["v190_nested_sentinel_review"]),
        "case_2516531_fullfit_reviewed": bool(residual["v190_fullfit_sentinel_review"]),
        "fullfit_candidate": full_fit.name if full_fit else None,
    }
    (OUT_DIR / "v190_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# v190 Fold-wise FN Sentinel Scan",
        "",
        "## Purpose",
        "",
        "Scan simple high-risk-miss sentinel rules only inside adaptive auto-decided low-risk cases. Fold-wise selection excludes the held-out fold, so case 2516531 is not directly used to select the rule for its own fold.",
        "",
        "## Result",
        "",
        (
            f"- Baseline adaptive auto errors: {report['baseline_auto_error_n']}; nested sentinel auto errors: "
            f"{report['nested_auto_error_n']}."
        ),
        (
            f"- Nested sentinel adds {report['nested_additional_review_n']} reviews and rescues "
            f"{report['nested_sentinel_rescued_fn_n']} FN(s); review rate becomes "
            f"{100 * report['nested_review_rate']:.2f}%, BAcc {100 * report['nested_bacc']:.2f}%."
        ),
        (
            f"- Case 2516531 reviewed by nested sentinel: {report['case_2516531_nested_reviewed']}; "
            f"reviewed by full-fit oracle-style sentinel: {report['case_2516531_fullfit_reviewed']}."
        ),
        "",
        "## Boundary",
        "",
        "If nested selection fails to rescue 2516531, the result should be treated as evidence that the last FN is not recoverable by simple stable probability sentinels without using post-hoc case-specific tuning.",
    ]
    (OUT_DIR / "v190_fn_sentinel_nested_scan.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v190] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
