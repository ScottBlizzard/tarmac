from __future__ import annotations

import json

import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v189_adaptive_error_review_anatomy_20260527"
V185_CASES = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_cases.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["fixed_v118_review", "fixed_v182_review", "adaptive_review", "adaptive_auto_decision"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    df["error_direction"] = "correct"
    df.loc[df["label_idx"].eq(1) & df["final_pred"].eq(0), "error_direction"] = "FN_high_to_low"
    df.loc[df["label_idx"].eq(0) & df["final_pred"].eq(1), "error_direction"] = "FP_low_to_high"
    df["adaptive_auto_error"] = df["adaptive_auto_decision"] & df["base_wrong"]
    df["adaptive_review_catches_base_error"] = df["adaptive_review"] & df["base_wrong"]

    keep_cols = [
        "domain",
        "case_id",
        "original_case_id",
        "task_l6_label",
        "label_idx",
        "final_pred",
        "error_direction",
        "prob_mean_core",
        "image_name",
        "gate_domain_decision",
        "adaptive_policy_branch",
        "adaptive_auto_decision",
        "adaptive_review",
    ]
    auto_errors = df.loc[df["adaptive_auto_error"], keep_cols].copy()
    reviewed_errors = df.loc[df["adaptive_review_catches_base_error"], keep_cols].copy()
    review_pool = df.loc[df["adaptive_review"], keep_cols + ["base_wrong"]].copy()

    auto_errors.to_csv(OUT_DIR / "v189_adaptive_auto_residual_errors.csv", index=False, encoding="utf-8-sig")
    reviewed_errors.to_csv(OUT_DIR / "v189_adaptive_review_captured_base_errors.csv", index=False, encoding="utf-8-sig")
    review_pool.to_csv(OUT_DIR / "v189_adaptive_review_pool_cases.csv", index=False, encoding="utf-8-sig")

    rows = []
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("strict_external", df["domain"].eq("strict_external")),
        ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        sub = df.loc[mask]
        review = sub["adaptive_review"]
        auto = sub["adaptive_auto_decision"]
        rows.append(
            {
                "scope": scope,
                "n": int(len(sub)),
                "auto_decision_n": int(auto.sum()),
                "auto_error_n": int((auto & sub["base_wrong"]).sum()),
                "review_n": int(review.sum()),
                "review_base_error_n": int((review & sub["base_wrong"]).sum()),
                "review_clean_n": int((review & ~sub["base_wrong"]).sum()),
                "fn_auto_error_n": int((auto & sub["error_direction"].eq("FN_high_to_low")).sum()),
                "fp_auto_error_n": int((auto & sub["error_direction"].eq("FP_low_to_high")).sum()),
                "fn_review_captured_n": int((review & sub["error_direction"].eq("FN_high_to_low")).sum()),
                "fp_review_captured_n": int((review & sub["error_direction"].eq("FP_low_to_high")).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "v189_adaptive_error_review_summary.csv", index=False, encoding="utf-8-sig")

    by_label = (
        df.groupby(["task_l6_label", "adaptive_review", "base_wrong"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["task_l6_label", "adaptive_review", "base_wrong"])
    )
    by_label.to_csv(OUT_DIR / "v189_adaptive_review_by_label.csv", index=False, encoding="utf-8-sig")

    all_row = summary.loc[summary["scope"].eq("all_domains")].iloc[0]
    md = [
        "# v189 Adaptive Error and Review Anatomy",
        "",
        "## Main Finding",
        "",
        (
            f"- Adaptive workflow auto-decides {int(all_row['auto_decision_n'])} cases with "
            f"{int(all_row['auto_error_n'])} residual auto error(s)."
        ),
        (
            f"- The review/reject pool contains {int(all_row['review_n'])} cases and captures "
            f"{int(all_row['review_base_error_n'])} base-model error(s), including "
            f"{int(all_row['fn_review_captured_n'])} high-risk-to-low-risk FN and "
            f"{int(all_row['fp_review_captured_n'])} low-risk-to-high-risk FP."
        ),
        "",
        "## Residual Auto Errors",
    ]
    if auto_errors.empty:
        md.append("- No residual automatic errors.")
    else:
        for _, r in auto_errors.iterrows():
            md.append(
                f"- {r['domain']} / {r['original_case_id']} / {r['task_l6_label']} / {r['error_direction']} / "
                f"image={r.get('image_name', '')}"
            )
    md += [
        "",
        "## Files",
        "",
        "- v189_adaptive_auto_residual_errors.csv",
        "- v189_adaptive_review_captured_base_errors.csv",
        "- v189_adaptive_review_pool_cases.csv",
        "- v189_adaptive_error_review_summary.csv",
        "- v189_adaptive_review_by_label.csv",
    ]
    (OUT_DIR / "v189_adaptive_error_review_anatomy.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "auto_decision_n": int(all_row["auto_decision_n"]),
        "auto_error_n": int(all_row["auto_error_n"]),
        "review_n": int(all_row["review_n"]),
        "review_base_error_n": int(all_row["review_base_error_n"]),
        "fn_auto_error_n": int(all_row["fn_auto_error_n"]),
        "fp_auto_error_n": int(all_row["fp_auto_error_n"]),
        "residual_auto_error_cases": auto_errors["original_case_id"].astype(str).tolist(),
    }
    (OUT_DIR / "v189_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v189] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
