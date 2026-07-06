from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v120_residual_fn_crop_rescue_validation_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def metrics(df: pd.DataFrame, review: pd.Series | np.ndarray) -> dict[str, float | int]:
    review = pd.Series(review, index=df.index).astype(bool)
    wrong = df["final_correct"].eq(0)
    rem = (~review) & wrong
    fn = int((rem & df["label_idx"].eq(1) & df["final_pred"].eq(0)).sum())
    fp = int((rem & df["label_idx"].eq(0) & df["final_pred"].eq(1)).sum())
    pos = int(df["label_idx"].eq(1).sum())
    neg = int(df["label_idx"].eq(0).sum())
    sens = (pos - fn) / pos if pos else np.nan
    spec = (neg - fp) / neg if neg else np.nan
    return {
        "n": int(len(df)),
        "control_rate": float(review.mean()),
        "remaining_error_n": int(rem.sum()),
        "fn": fn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def crop_rescue_flag(df: pd.DataFrame, base_review: pd.Series, crop_min: float, core_max: float) -> pd.Series:
    auto_low = (~base_review.astype(bool)) & df["domain"].isin(["old_data", "third_batch"]) & df["final_pred"].eq(0)
    return (
        auto_low
        & pd.to_numeric(df["v105_crop_prob"], errors="coerce").ge(crop_min)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
    )


def scan_crop_rules(train: pd.DataFrame, base_review: pd.Series) -> pd.DataFrame:
    rows = []
    for crop_min in np.round(np.arange(0.20, 0.451, 0.01), 3):
        for core_max in np.round(np.arange(0.35, 0.551, 0.01), 3):
            flag = crop_rescue_flag(train, base_review, float(crop_min), float(core_max))
            review = base_review.astype(bool) | flag
            m = metrics(train, review)
            rows.append(
                {
                    "crop_min": float(crop_min),
                    "core_max": float(core_max),
                    "extra_review_n": int(flag.sum()),
                    "captured_error_n": int((flag & train["final_correct"].eq(0)).sum()),
                    "clean_review_n": int((flag & train["final_correct"].eq(1)).sum()),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def select_crop_rule(train: pd.DataFrame, base_review: pd.Series) -> pd.Series | None:
    base_m = metrics(train, base_review)
    if int(base_m["fn"]) == 0:
        return None
    scan = scan_crop_rules(train, base_review)
    candidates = scan[(scan["fn"].lt(base_m["fn"])) & (scan["fp"].eq(base_m["fp"]))].copy()
    if candidates.empty:
        candidates = scan.copy()
    return candidates.sort_values(
        ["remaining_error_n", "fn", "fp", "extra_review_n", "clean_review_n", "balanced_accuracy"],
        ascending=[True, True, True, True, True, False],
    ).iloc[0]


def evaluate_scope_rows(df: pd.DataFrame, review_col: str, workflow: str) -> pd.DataFrame:
    rows = []
    for scope, sub in [
        ("all_domains", df),
        ("internal_all", df[df["domain"].isin(["old_data", "third_batch"])]),
        ("old_data", df[df["domain"].eq("old_data")]),
        ("third_batch", df[df["domain"].eq("third_batch")]),
        ("strict_external", df[df["domain"].eq("strict_external")]),
    ]:
        row = {"workflow": workflow, "scope": scope}
        row.update(metrics(sub, sub[review_col].astype(bool)))
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v118_review_or_control"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct", "fold_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    # Post-hoc upper-bound rescue, intentionally marked as such.
    posthoc_flag = crop_rescue_flag(df, df["v118_review_or_control"], 0.25, 0.43)
    df["posthoc_crop_rescue_review"] = df["v118_review_or_control"] | posthoc_flag
    df["posthoc_crop_rescue_extra"] = posthoc_flag

    # Fold-wise validation: choose crop rescue on four internal folds, apply to held-out fold.
    df["nested_crop_rescue_extra"] = False
    internal = df["domain"].isin(["old_data", "third_batch"])
    rules = []
    for fold in sorted(df.loc[internal, "fold_id"].unique()):
        train_idx = df.index[internal & df["fold_id"].ne(fold)]
        val_idx = df.index[internal & df["fold_id"].eq(fold)]
        train = df.loc[train_idx].copy()
        val = df.loc[val_idx].copy()
        selected = select_crop_rule(train, train["v118_review_or_control"].astype(bool))
        if selected is None:
            val_flag = pd.Series(False, index=val.index)
            rules.append(
                {
                    "fold_id": int(fold),
                    "selected": "no_train_fn_noop",
                    "crop_min": np.nan,
                    "core_max": np.nan,
                    "train_extra_review_n": 0,
                    "train_remaining_error_n": int(metrics(train, train["v118_review_or_control"])["remaining_error_n"]),
                    "train_fn": 0,
                    "train_fp": int(metrics(train, train["v118_review_or_control"])["fp"]),
                    "val_extra_review_n": 0,
                    "val_captured_error_n": 0,
                    "val_clean_review_n": 0,
                }
            )
        else:
            val_flag = crop_rescue_flag(
                val,
                val["v118_review_or_control"].astype(bool),
                float(selected["crop_min"]),
                float(selected["core_max"]),
            )
            rules.append(
                {
                    "fold_id": int(fold),
                    "selected": "crop_rule",
                    "crop_min": float(selected["crop_min"]),
                    "core_max": float(selected["core_max"]),
                    "train_extra_review_n": int(selected["extra_review_n"]),
                    "train_remaining_error_n": int(selected["remaining_error_n"]),
                    "train_fn": int(selected["fn"]),
                    "train_fp": int(selected["fp"]),
                    "val_extra_review_n": int(val_flag.sum()),
                    "val_captured_error_n": int((val_flag & val["final_correct"].eq(0)).sum()),
                    "val_clean_review_n": int((val_flag & val["final_correct"].eq(1)).sum()),
                }
            )
        df.loc[val_idx, "nested_crop_rescue_extra"] = val_flag.to_numpy(bool)

    df["nested_crop_rescue_review"] = df["v118_review_or_control"] | df["nested_crop_rescue_extra"].astype(bool)

    summary = pd.concat(
        [
            evaluate_scope_rows(df, "v118_review_or_control", "v118_global_two_signal"),
            evaluate_scope_rows(df, "posthoc_crop_rescue_review", "posthoc_crop_rescue_upper_bound"),
            evaluate_scope_rows(df, "nested_crop_rescue_review", "nested_crop_rescue_validation"),
        ],
        ignore_index=True,
    )
    rule_frame = pd.DataFrame(rules)
    df.to_csv(OUT_DIR / "v120_residual_fn_crop_rescue_cases.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v120_residual_fn_crop_rescue_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v120_residual_fn_crop_rescue_summary_formatted.csv", index=False, encoding="utf-8-sig")
    rule_frame.to_csv(OUT_DIR / "v120_nested_crop_rescue_fold_rules.csv", index=False, encoding="utf-8-sig")

    base_all = summary[(summary["workflow"].eq("v118_global_two_signal")) & (summary["scope"].eq("all_domains"))].iloc[0]
    post_all = summary[
        (summary["workflow"].eq("posthoc_crop_rescue_upper_bound")) & (summary["scope"].eq("all_domains"))
    ].iloc[0]
    nested_all = summary[
        (summary["workflow"].eq("nested_crop_rescue_validation")) & (summary["scope"].eq("all_domains"))
    ].iloc[0]
    lines = [
        "# v120 Residual FN Crop Rescue Validation",
        "",
        "v120 tests whether the remaining FN 2516531 can be rescued by a crop-probability rule without post-hoc leakage.",
        "",
        f"- v118 base: BAcc {pct(base_all['balanced_accuracy'])}, control {pct(base_all['control_rate'])}, FN={int(base_all['fn'])}, FP={int(base_all['fp'])}.",
        f"- Post-hoc crop rescue: BAcc {pct(post_all['balanced_accuracy'])}, control {pct(post_all['control_rate'])}, FN={int(post_all['fn'])}, FP={int(post_all['fp'])}.",
        f"- Nested crop rescue: BAcc {pct(nested_all['balanced_accuracy'])}, control {pct(nested_all['control_rate'])}, FN={int(nested_all['fn'])}, FP={int(nested_all['fp'])}.",
        "",
        "The post-hoc crop rule catches 2516531, but fold-wise selection cannot rescue it because the fold containing 2516531 has no residual FN in its training folds. This remains upper-bound evidence, not a formal deployable improvement.",
    ]
    (OUT_DIR / "v120_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(format_table(summary[summary["scope"].isin(["all_domains", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print(rule_frame.to_string(index=False))


if __name__ == "__main__":
    main()
