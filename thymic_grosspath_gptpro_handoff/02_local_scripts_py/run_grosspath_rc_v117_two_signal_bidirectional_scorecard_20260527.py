from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v114_nested_directional_guards_20260527 as v114  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v117_two_signal_bidirectional_scorecard_20260527"


def fp_guard_two_signal(df: pd.DataFrame, base_review: pd.Series, wc_max: float, core_max: float) -> pd.Series:
    auto_high = (~base_review.astype(bool)) & df["domain"].isin(["old_data", "third_batch"]) & df["final_pred"].eq(1)
    return (
        auto_high
        & pd.to_numeric(df["wholecrop_prob"], errors="coerce").le(wc_max)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
    )


def scan_two_signal_rules(train: pd.DataFrame, base_review: pd.Series) -> pd.DataFrame:
    rows = []
    for wc_max in np.round(np.arange(0.500, 0.751, 0.025), 3):
        for core_max in np.round(np.arange(0.700, 0.901, 0.025), 3):
            flag = fp_guard_two_signal(train, base_review, float(wc_max), float(core_max))
            review = base_review.astype(bool) | flag
            m = v114.metrics(train, review)
            rows.append(
                {
                    "wc_max": float(wc_max),
                    "core_max": float(core_max),
                    "extra_review_n": int(flag.sum()),
                    "captured_error_n": int((flag & train["final_correct"].eq(0)).sum()),
                    "clean_review_n": int((flag & train["final_correct"].eq(1)).sum()),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def select_min_review(scan: pd.DataFrame) -> pd.Series:
    candidates = scan[(scan["fp"].eq(0)) & (scan["fn"].le(1))].copy()
    if candidates.empty:
        candidates = scan.copy()
    return candidates.sort_values(
        ["remaining_error_n", "fp", "extra_review_n", "clean_review_n", "balanced_accuracy", "wc_max", "core_max"],
        ascending=[True, True, True, True, False, True, True],
    ).iloc[0]


def select_stable_envelope(scan: pd.DataFrame, cushion: int = 1) -> pd.Series:
    candidates = scan[(scan["fp"].eq(0)) & (scan["fn"].le(1))].copy()
    if candidates.empty:
        candidates = scan.copy()
    min_extra = int(candidates["extra_review_n"].min())
    candidates = candidates[candidates["extra_review_n"].le(min_extra + cushion)].copy()
    return candidates.sort_values(
        ["wc_max", "core_max", "balanced_accuracy", "extra_review_n"],
        ascending=[False, False, False, True],
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
        row.update(v114.metrics(sub, sub[review_col].astype(bool)))
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(v114.pct)
    return out


def run_nested(df: pd.DataFrame, selector_name: str) -> tuple[pd.DataFrame, list[dict[str, int | float | str]]]:
    out = df.copy()
    flag_col = f"{selector_name}_extra_review"
    review_col = f"{selector_name}_review_or_control"
    out[flag_col] = False
    rules: list[dict[str, int | float | str]] = []
    internal_mask = out["domain"].isin(["old_data", "third_batch"])

    for fold in sorted(out.loc[internal_mask, "fold_id"].unique()):
        train_idx = out.index[internal_mask & out["fold_id"].ne(fold)]
        val_idx = out.index[internal_mask & out["fold_id"].eq(fold)]
        train = out.loc[train_idx].copy()
        val = out.loc[val_idx].copy()
        train_base = train["v111_review_or_control"].astype(bool)
        scan = scan_two_signal_rules(train, train_base)
        selected = select_min_review(scan) if selector_name == "two_signal_min" else select_stable_envelope(scan)
        val_flag = fp_guard_two_signal(
            val,
            val["v111_review_or_control"].astype(bool),
            float(selected["wc_max"]),
            float(selected["core_max"]),
        )
        out.loc[val_idx, flag_col] = val_flag.to_numpy(bool)
        rules.append(
            {
                "selector": selector_name,
                "fold_id": int(fold),
                "wc_max": float(selected["wc_max"]),
                "core_max": float(selected["core_max"]),
                "train_extra_review_n": int(selected["extra_review_n"]),
                "train_remaining_error_n": int(selected["remaining_error_n"]),
                "train_fn": int(selected["fn"]),
                "train_fp": int(selected["fp"]),
                "val_n": int(len(val)),
                "val_extra_review_n": int(val_flag.sum()),
                "val_captured_error_n": int((val_flag & val["final_correct"].eq(0)).sum()),
                "val_clean_review_n": int((val_flag & val["final_correct"].eq(1)).sum()),
            }
        )

    out[review_col] = out["v111_review_or_control"].astype(bool) | out[flag_col].astype(bool)
    out[f"{selector_name}_rescued_error"] = out[flag_col].astype(bool) & out["final_correct"].eq(0)
    out[f"{selector_name}_extra_clean_review"] = out[flag_col].astype(bool) & out["final_correct"].eq(1)
    return out, rules


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(v114.V112_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "final_correct"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(int)
    raw["v111_review_or_control"] = v114.as_bool(raw["v111_review_or_control"])
    base = v114.attach_fold_and_crop(raw)

    min_cases, min_rules = run_nested(base, "two_signal_min")
    stable_cases, stable_rules = run_nested(base, "two_signal_stable")

    combined = base.copy()
    for col in [
        "two_signal_min_extra_review",
        "two_signal_min_review_or_control",
        "two_signal_min_rescued_error",
        "two_signal_min_extra_clean_review",
    ]:
        combined[col] = min_cases[col].to_numpy()
    for col in [
        "two_signal_stable_extra_review",
        "two_signal_stable_review_or_control",
        "two_signal_stable_rescued_error",
        "two_signal_stable_extra_clean_review",
    ]:
        combined[col] = stable_cases[col].to_numpy()

    summary = pd.concat(
        [
            evaluate_scope_rows(combined, "v111_review_or_control", "v111_base"),
            evaluate_scope_rows(combined, "two_signal_min_review_or_control", "nested_two_signal_min_review"),
            evaluate_scope_rows(combined, "two_signal_stable_review_or_control", "nested_two_signal_stable_envelope"),
        ],
        ignore_index=True,
    )
    rules = pd.DataFrame(min_rules + stable_rules)

    summary.to_csv(OUT_DIR / "v117_two_signal_scorecard_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v117_two_signal_scorecard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    rules.to_csv(OUT_DIR / "v117_fold_selected_two_signal_rules.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUT_DIR / "v117_two_signal_scorecard_cases.csv", index=False, encoding="utf-8-sig")

    stable_all = summary[
        (summary["workflow"].eq("nested_two_signal_stable_envelope")) & (summary["scope"].eq("all_domains"))
    ].iloc[0]
    min_all = summary[
        (summary["workflow"].eq("nested_two_signal_min_review")) & (summary["scope"].eq("all_domains"))
    ].iloc[0]
    rescued = combined[combined["two_signal_stable_rescued_error"]][
        ["domain", "fold_id", "original_case_id", "task_l6_label"]
    ].astype(str)
    rescued_text = "; ".join(
        f"{r.domain}:fold{r.fold_id}:{r.original_case_id}/{r.task_l6_label}" for r in rescued.itertuples(index=False)
    )
    lines = [
        "# v117 Two-signal Bidirectional Scorecard",
        "",
        "v117 tests whether the v115 low-risk FP guard can be reduced to only two signals: wholecrop probability and core-model mean probability.",
        "",
        f"- Min-review all-domain: BAcc {v114.pct(min_all['balanced_accuracy'])}, control {v114.pct(min_all['control_rate'])}, FN={int(min_all['fn'])}, FP={int(min_all['fp'])}.",
        f"- Stable-envelope all-domain: BAcc {v114.pct(stable_all['balanced_accuracy'])}, control {v114.pct(stable_all['control_rate'])}, FN={int(stable_all['fn'])}, FP={int(stable_all['fp'])}.",
        f"- Stable-envelope rescued held-out FP cases: {rescued_text}.",
        "",
        "If stable-envelope matches v115, the low-risk guard no longer needs main_prob/robust_prob and can be written as a simpler directional scorecard.",
    ]
    (OUT_DIR / "v117_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    focus = summary[summary["scope"].isin(["all_domains", "internal_all", "third_batch", "strict_external"])]
    print(format_table(focus).to_string(index=False))
    print()
    print(rules.to_string(index=False))
    print()
    print("stable rescued:")
    print(rescued.to_string(index=False))


if __name__ == "__main__":
    main()
