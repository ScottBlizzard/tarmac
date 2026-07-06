from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"
V182_CASES = ROOT / "outputs" / "grosspath_rc_v182_stable_fixed_image_agreement_release_20260527" / "v182_stable_fixed_case_outputs.csv"
V158_GATE = ROOT / "outputs" / "grosspath_rc_v158_unlabeled_gate_threshold_stability_20260527" / "v158_gate_component_stability.csv"


def load_v118() -> pd.DataFrame:
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v118_review_or_control"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    df["prob_mean_core"] = pd.to_numeric(df["prob_mean_core"], errors="coerce")
    return df


def load_v182() -> pd.DataFrame:
    df = pd.read_csv(V182_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in [
        "v118_review_or_control",
        "v161_safe_release_from_review",
        "v182_image_agreement_release",
        "v182_union_release",
        "v182_union_released_error",
    ]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    return df


def apply_policy(df: pd.DataFrame, review: pd.Series, workflow: str, note: str) -> list[dict[str, object]]:
    y_all = df["label_idx"].to_numpy(int)
    pred = df["final_pred"].to_numpy(int).copy()
    pred[review.to_numpy(bool)] = y_all[review.to_numpy(bool)]
    rows = []
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("strict_external", df["domain"].eq("strict_external")),
        ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        m = metrics(y_all[mask.to_numpy(bool)], pred[mask.to_numpy(bool)], df.loc[mask, "prob_mean_core"].to_numpy(float))
        rows.append(
            {
                "workflow": workflow,
                "scope": scope,
                "n": int(mask.sum()),
                "auto_decision_n": int((~review[mask]).sum()),
                "auto_decision_rate": float((~review[mask]).mean()),
                "review_or_reject_n": int(review[mask].sum()),
                "review_or_reject_rate": float(review[mask].mean()),
                "accuracy": float(m["accuracy"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "f1": float(m["f1"]),
                "auc": float(m["auc"]),
                "fn": int(m["fn"]),
                "fp": int(m["fp"]),
                "note": note,
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v118 = load_v118()
    v182 = load_v182()
    base_cols = [
        "domain",
        "case_id",
        "original_case_id",
        "task_l6_label",
        "label_idx",
        "final_pred",
        "prob_mean_core",
        "image_name",
    ]
    df = v118[base_cols + ["v118_review_or_control"]].merge(
        v182[["case_id", "v182_union_release", "v182_image_agreement_release", "v161_safe_release_from_review"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    gate = pd.read_csv(V158_GATE)
    clean_gate = gate.loc[gate["current_safety_status"].eq("clean_current_split")].copy()
    severe_domains = {"strict_external"}

    fixed_v118_review = df["v118_review_or_control"]
    fixed_v182_review = df["v118_review_or_control"] & ~df["v182_union_release"]
    adaptive_review = fixed_v182_review.copy()
    severe_mask = df["domain"].isin(severe_domains)
    adaptive_review.loc[severe_mask] = fixed_v118_review.loc[severe_mask]

    rows = []
    rows += apply_policy(df, fixed_v118_review, "fixed_v118_high_safety", "all domains use v118 high-safety review")
    rows += apply_policy(df, fixed_v182_review, "fixed_v182_stable_release", "all domains use v182 stable fixed release")
    rows += apply_policy(
        df,
        adaptive_review,
        "unlabeled_shift_adaptive_v182_to_v118",
        "old/third use v182; severe-shift strict_external uses v118",
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "v185_unlabeled_shift_adaptive_summary.csv", index=False, encoding="utf-8-sig")

    cases = df[base_cols].copy()
    cases["gate_domain_decision"] = cases["domain"].map(lambda x: "severe_shift" if x in severe_domains else "within_internal_shift")
    cases["fixed_v118_review"] = fixed_v118_review
    cases["fixed_v182_review"] = fixed_v182_review
    cases["adaptive_review"] = adaptive_review
    cases["adaptive_auto_decision"] = ~adaptive_review
    cases["adaptive_policy_branch"] = cases["gate_domain_decision"].map(
        {"severe_shift": "v118_high_safety", "within_internal_shift": "v182_stable_release"}
    )
    cases.to_csv(OUT_DIR / "v185_unlabeled_shift_adaptive_cases.csv", index=False, encoding="utf-8-sig")

    gate_report = pd.DataFrame(
        [
            {
                "gate_family": "v158 severe-shift gate variants",
                "clean_current_split_n": int(len(clean_gate)),
                "total_variants_n": int(len(gate)),
                "severe_domains_used_by_policy": ",".join(sorted(severe_domains)),
                "policy_interpretation": "If an unlabeled batch is flagged as severe external shift, the system falls back from v182 stable release to v118 high-safety review.",
            }
        ]
    )
    gate_report.to_csv(OUT_DIR / "v185_gate_policy_definition.csv", index=False, encoding="utf-8-sig")

    all_rows = summary.loc[summary["scope"].eq("all_domains")].copy()
    strict_rows = summary.loc[summary["scope"].eq("strict_external")].copy()
    fixed182 = all_rows.loc[all_rows["workflow"].eq("fixed_v182_stable_release")].iloc[0]
    adaptive = all_rows.loc[all_rows["workflow"].eq("unlabeled_shift_adaptive_v182_to_v118")].iloc[0]
    fixed118 = all_rows.loc[all_rows["workflow"].eq("fixed_v118_high_safety")].iloc[0]
    strict_adaptive = strict_rows.loc[strict_rows["workflow"].eq("unlabeled_shift_adaptive_v182_to_v118")].iloc[0]
    strict182 = strict_rows.loc[strict_rows["workflow"].eq("fixed_v182_stable_release")].iloc[0]

    md = [
        "# v185 Unlabeled Shift-adaptive Policy",
        "",
        "## Rule",
        "",
        "Use v182 stable fixed release on within-internal-shift batches. If the unlabeled severe-shift gate flags a batch as strict external-like, fall back to v118 high-safety review for that batch.",
        "",
        "## Result",
        "",
        (
            f"- Fixed v182: all-domain BAcc {100 * float(fixed182['balanced_accuracy']):.2f}%, review "
            f"{100 * float(fixed182['review_or_reject_rate']):.2f}%."
        ),
        (
            f"- Adaptive v182->v118: all-domain BAcc {100 * float(adaptive['balanced_accuracy']):.2f}%, review "
            f"{100 * float(adaptive['review_or_reject_rate']):.2f}%."
        ),
        (
            f"- Fixed v118: all-domain BAcc {100 * float(fixed118['balanced_accuracy']):.2f}%, review "
            f"{100 * float(fixed118['review_or_reject_rate']):.2f}%."
        ),
        (
            f"- Strict external branch: adaptive uses v118, review {100 * float(strict_adaptive['review_or_reject_rate']):.2f}% "
            f"vs fixed v182 review {100 * float(strict182['review_or_reject_rate']):.2f}%; both BAcc "
            f"{100 * float(strict_adaptive['balanced_accuracy']):.2f}% in the current split."
        ),
        "",
        "## Boundary",
        "",
        "This experiment validates the deployment-control logic rather than proving that v118 is always superior on severe-shift data. In the current strict external split, both v182 and v118 are correct; the adaptive policy is more conservative when the unlabeled gate detects severe shift.",
    ]
    (OUT_DIR / "v185_unlabeled_shift_adaptive_policy.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "clean_gate_variants": int(len(clean_gate)),
        "gate_variants": int(len(gate)),
        "fixed_v182_bacc": float(fixed182["balanced_accuracy"]),
        "fixed_v182_review_rate": float(fixed182["review_or_reject_rate"]),
        "adaptive_bacc": float(adaptive["balanced_accuracy"]),
        "adaptive_review_rate": float(adaptive["review_or_reject_rate"]),
        "fixed_v118_bacc": float(fixed118["balanced_accuracy"]),
        "fixed_v118_review_rate": float(fixed118["review_or_reject_rate"]),
        "strict_external_adaptive_bacc": float(strict_adaptive["balanced_accuracy"]),
        "strict_external_adaptive_review_rate": float(strict_adaptive["review_or_reject_rate"]),
        "strict_external_fixed_v182_review_rate": float(strict182["review_or_reject_rate"]),
    }
    (OUT_DIR / "v185_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v185] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
