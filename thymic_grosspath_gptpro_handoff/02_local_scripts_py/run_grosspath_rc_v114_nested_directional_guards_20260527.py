from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v114_nested_directional_guards_20260527"

V112_CASES = ROOT / "outputs" / "grosspath_rc_v112_symmetric_lowrisk_guard_20260527" / "v112_lowrisk_guard_cases_with_flags.csv"
V108_INTERNAL = ROOT / "outputs" / "grosspath_rc_v108_v105_crop_proxy_external_scorecard_20260527" / "v108_internal_cases_with_flags.csv"
SELECTED_OOF = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "44_old_third_unified_feature_cv_20260523"
    / "selected_unified_feature_oof_predictions.csv"
)


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


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


def fp_guard_flag(df: pd.DataFrame, base_review: pd.Series, wc_max: float, core_max: float, main_max: float, robust_max: float) -> pd.Series:
    auto_high = (~base_review) & df["domain"].isin(["old_data", "third_batch"]) & df["final_pred"].eq(1)
    return (
        auto_high
        & pd.to_numeric(df["wholecrop_prob"], errors="coerce").le(wc_max)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
        & pd.to_numeric(df["main_prob"], errors="coerce").le(main_max)
        & pd.to_numeric(df["robust_prob"], errors="coerce").le(robust_max)
    )


def crop_guard_flag(df: pd.DataFrame, base_review: pd.Series, crop_min: float, core_max: float) -> pd.Series:
    auto_low = (~base_review) & df["domain"].isin(["old_data", "third_batch"]) & df["final_pred"].eq(0)
    return (
        auto_low
        & pd.to_numeric(df["v105_crop_prob"], errors="coerce").ge(crop_min)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
    )


def select_fp_rule(train: pd.DataFrame, base_review: pd.Series) -> dict[str, float | int]:
    rows = []
    for wc_max in np.round(np.arange(0.500, 0.701, 0.025), 3):
        for core_max in np.round(np.arange(0.725, 0.826, 0.025), 3):
            for main_max in np.round(np.arange(0.900, 0.976, 0.025), 3):
                for robust_max in np.round(np.arange(0.750, 0.851, 0.025), 3):
                    flag = fp_guard_flag(train, base_review, float(wc_max), float(core_max), float(main_max), float(robust_max))
                    review = base_review | flag
                    m = metrics(train, review)
                    rows.append(
                        {
                            "wc_max": float(wc_max),
                            "fp_core_max": float(core_max),
                            "main_max": float(main_max),
                            "robust_max": float(robust_max),
                            "extra_review_n": int(flag.sum()),
                            "captured_error_n": int((flag & train["final_correct"].eq(0)).sum()),
                            "clean_review_n": int((flag & train["final_correct"].eq(1)).sum()),
                            **m,
                        }
                    )
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(
        ["remaining_error_n", "fp", "extra_review_n", "clean_review_n", "balanced_accuracy"],
        ascending=[True, True, True, True, False],
    ).iloc[0]
    return selected.to_dict()


def select_crop_rule(train: pd.DataFrame, base_review: pd.Series) -> dict[str, float | int]:
    base_m = metrics(train, base_review)
    if int(base_m["fn"]) == 0:
        return {
            "crop_min": 9.0,
            "crop_core_max": -1.0,
            "extra_review_n": 0,
            "captured_error_n": 0,
            "clean_review_n": 0,
            **base_m,
            "no_train_fn": 1,
        }
    rows = []
    for crop_min in np.round(np.arange(0.20, 0.451, 0.01), 3):
        for core_max in np.round(np.arange(0.35, 0.551, 0.01), 3):
            flag = crop_guard_flag(train, base_review, float(crop_min), float(core_max))
            review = base_review | flag
            m = metrics(train, review)
            rows.append(
                {
                    "crop_min": float(crop_min),
                    "crop_core_max": float(core_max),
                    "extra_review_n": int(flag.sum()),
                    "captured_error_n": int((flag & train["final_correct"].eq(0)).sum()),
                    "clean_review_n": int((flag & train["final_correct"].eq(1)).sum()),
                    **m,
                    "no_train_fn": 0,
                }
            )
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(
        ["remaining_error_n", "fn", "extra_review_n", "clean_review_n", "balanced_accuracy"],
        ascending=[True, True, True, True, False],
    ).iloc[0]
    return selected.to_dict()


def attach_fold_and_crop(df: pd.DataFrame) -> pd.DataFrame:
    internal = df[df["domain"].isin(["old_data", "third_batch"])].copy()
    external = df[df["domain"].eq("strict_external")].copy()

    fold_map = pd.read_csv(SELECTED_OOF, dtype={"case_id": str, "original_case_id": str})
    fold_map["domain_key"] = fold_map["domain"].map({"old": "old_data", "third": "third_batch"})
    fold_map = fold_map[["domain_key", "case_id", "original_case_id", "fold_id"]].drop_duplicates()
    internal = internal.merge(
        fold_map,
        left_on=["domain", "case_id", "original_case_id"],
        right_on=["domain_key", "case_id", "original_case_id"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["domain_key"])
    if internal["fold_id"].isna().any():
        missing = internal.loc[internal["fold_id"].isna(), ["domain", "case_id", "original_case_id"]].head(10)
        raise RuntimeError(f"Missing fold ids:\n{missing}")
    internal["fold_id"] = internal["fold_id"].astype(int)

    crop = pd.read_csv(V108_INTERNAL, dtype={"case_id": str, "original_case_id": str})
    crop = crop[["domain", "case_id", "original_case_id", "v105_crop_prob"]]
    internal = internal.merge(crop, on=["domain", "case_id", "original_case_id"], how="left", validate="many_to_one")
    if internal["v105_crop_prob"].isna().any():
        missing = internal.loc[internal["v105_crop_prob"].isna(), ["domain", "case_id", "original_case_id"]].head(10)
        raise RuntimeError(f"Missing crop probabilities:\n{missing}")

    external["fold_id"] = -1
    external["v105_crop_prob"] = -1.0
    return pd.concat([internal, external], ignore_index=True, sort=False)


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
    raw = pd.read_csv(V112_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "final_correct"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(int)
    raw["v111_review_or_control"] = as_bool(raw["v111_review_or_control"])
    raw = attach_fold_and_crop(raw)

    df = raw.copy()
    df["nested_v112_extra_review"] = False
    df["nested_v113_extra_review"] = False
    rule_rows = []
    internal_idx = df.index[df["domain"].isin(["old_data", "third_batch"])]

    for fold in sorted(df.loc[internal_idx, "fold_id"].unique()):
        train_idx = df.index[df["domain"].isin(["old_data", "third_batch"]) & df["fold_id"].ne(fold)]
        val_idx = df.index[df["domain"].isin(["old_data", "third_batch"]) & df["fold_id"].eq(fold)]
        train = df.loc[train_idx].copy()
        val = df.loc[val_idx].copy()

        train_base = train["v111_review_or_control"].astype(bool)
        fp_rule = select_fp_rule(train, train_base)
        val_base = val["v111_review_or_control"].astype(bool)
        val_fp = fp_guard_flag(
            val,
            val_base,
            float(fp_rule["wc_max"]),
            float(fp_rule["fp_core_max"]),
            float(fp_rule["main_max"]),
            float(fp_rule["robust_max"]),
        )
        df.loc[val_idx, "nested_v112_extra_review"] = val_fp.to_numpy(bool)

        train_fp = fp_guard_flag(
            train,
            train_base,
            float(fp_rule["wc_max"]),
            float(fp_rule["fp_core_max"]),
            float(fp_rule["main_max"]),
            float(fp_rule["robust_max"]),
        )
        train_after_fp = train_base | train_fp
        crop_rule = select_crop_rule(train, train_after_fp)
        val_after_fp = val_base | val_fp
        val_crop = crop_guard_flag(
            val,
            val_after_fp,
            float(crop_rule["crop_min"]),
            float(crop_rule["crop_core_max"]),
        )
        df.loc[val_idx, "nested_v113_extra_review"] = val_crop.to_numpy(bool)

        rule_rows.append(
            {
                "fold_id": int(fold),
                "fp_wc_max": fp_rule["wc_max"],
                "fp_core_max": fp_rule["fp_core_max"],
                "fp_main_max": fp_rule["main_max"],
                "fp_robust_max": fp_rule["robust_max"],
                "fp_train_remaining_error_n": fp_rule["remaining_error_n"],
                "fp_train_fn": fp_rule["fn"],
                "fp_train_fp": fp_rule["fp"],
                "fp_train_extra_review_n": fp_rule["extra_review_n"],
                "crop_min": crop_rule["crop_min"],
                "crop_core_max": crop_rule["crop_core_max"],
                "crop_train_remaining_error_n": crop_rule["remaining_error_n"],
                "crop_train_fn": crop_rule["fn"],
                "crop_train_fp": crop_rule["fp"],
                "crop_train_extra_review_n": crop_rule["extra_review_n"],
                "crop_no_train_fn": crop_rule["no_train_fn"],
                "val_n": int(len(val)),
                "val_fp_extra_n": int(val_fp.sum()),
                "val_crop_extra_n": int(val_crop.sum()),
            }
        )

    # Strict external remains the v79-strict branch; nested internal guards are not applied to it.
    df["nested_v112_review_or_control"] = df["v111_review_or_control"].astype(bool) | df["nested_v112_extra_review"].astype(bool)
    df["nested_v113_review_or_control"] = df["nested_v112_review_or_control"].astype(bool) | df["nested_v113_extra_review"].astype(bool)

    base_summary = evaluate_scope_rows(df, "v111_review_or_control", "v111_base")
    v112_summary = evaluate_scope_rows(df, "nested_v112_review_or_control", "nested_v112_fp_guard")
    v113_summary = evaluate_scope_rows(df, "nested_v113_review_or_control", "nested_v112_plus_nested_v113")
    summary = pd.concat([base_summary, v112_summary, v113_summary], ignore_index=True)

    df["nested_v112_rescued_error"] = df["nested_v112_extra_review"].astype(bool) & df["final_correct"].eq(0)
    df["nested_v113_rescued_error"] = df["nested_v113_extra_review"].astype(bool) & df["final_correct"].eq(0)

    pd.DataFrame(rule_rows).to_csv(OUT_DIR / "v114_fold_selected_rules.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v114_nested_directional_guard_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v114_nested_directional_guard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "v114_nested_directional_guard_cases.csv", index=False, encoding="utf-8-sig")

    nested = summary[(summary["workflow"].eq("nested_v112_plus_nested_v113")) & (summary["scope"].eq("all_domains"))].iloc[0]
    nested_third = summary[(summary["workflow"].eq("nested_v112_plus_nested_v113")) & (summary["scope"].eq("third_batch"))].iloc[0]
    lines = [
        "# v114 Nested Directional Guard Validation",
        "",
        "v114 selects v112/v113 thresholds on four internal folds and applies them to the held-out fold. Strict external remains the v79-strict severe-shift branch.",
        "",
        f"- Nested all-domain: BAcc {pct(nested['balanced_accuracy'])}, control {pct(nested['control_rate'])}, FN={int(nested['fn'])}, FP={int(nested['fp'])}.",
        f"- Nested third batch: BAcc {pct(nested_third['balanced_accuracy'])}, control {pct(nested_third['control_rate'])}, FN={int(nested_third['fn'])}, FP={int(nested_third['fp'])}.",
        f"- Fold-selected v112 rescued errors: {int(df['nested_v112_rescued_error'].sum())}.",
        f"- Fold-selected v113 rescued errors: {int(df['nested_v113_rescued_error'].sum())}.",
        "",
        "This is the first check of whether the post-hoc zero-error candidate survives held-out fold selection.",
    ]
    (OUT_DIR / "v114_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(format_table(summary[summary["scope"].isin(["all_domains", "internal_all", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print("fold rules:")
    print(pd.DataFrame(rule_rows).to_string(index=False))


if __name__ == "__main__":
    main()
