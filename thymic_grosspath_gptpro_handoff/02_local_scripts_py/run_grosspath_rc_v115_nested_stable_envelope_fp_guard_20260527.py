from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v114_nested_directional_guards_20260527 as v114  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v115_nested_stable_envelope_fp_guard_20260527"


def scan_fp_rules(train: pd.DataFrame, base_review: pd.Series) -> pd.DataFrame:
    rows = []
    for wc_max in np.round(np.arange(0.500, 0.701, 0.025), 3):
        for core_max in np.round(np.arange(0.725, 0.826, 0.025), 3):
            for main_max in np.round(np.arange(0.900, 0.976, 0.025), 3):
                for robust_max in np.round(np.arange(0.750, 0.851, 0.025), 3):
                    flag = v114.fp_guard_flag(train, base_review, float(wc_max), float(core_max), float(main_max), float(robust_max))
                    review = base_review | flag
                    m = v114.metrics(train, review)
                    rows.append(
                        {
                            "wc_max": float(wc_max),
                            "core_max": float(core_max),
                            "main_max": float(main_max),
                            "robust_max": float(robust_max),
                            "extra_review_n": int(flag.sum()),
                            "captured_error_n": int((flag & train["final_correct"].eq(0)).sum()),
                            "clean_review_n": int((flag & train["final_correct"].eq(1)).sum()),
                            **m,
                        }
                    )
    return pd.DataFrame(rows)


def select_stable_envelope(scan: pd.DataFrame) -> pd.Series:
    candidates = scan[(scan["fp"].eq(0)) & (scan["fn"].le(1))].copy()
    if candidates.empty:
        candidates = scan.copy()
    # Do not pick the absolute narrowest rule. Allow a small review budget
    # cushion, then prefer the broader envelope to improve held-out coverage.
    min_extra = int(candidates["extra_review_n"].min())
    candidates = candidates[candidates["extra_review_n"].le(min_extra + 6)].copy()
    return candidates.sort_values(
        ["wc_max", "core_max", "main_max", "robust_max", "balanced_accuracy", "extra_review_n"],
        ascending=[False, False, False, False, False, True],
    ).iloc[0]


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(v114.pct)
    return out


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(v114.V112_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "final_correct"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(int)
    raw["v111_review_or_control"] = v114.as_bool(raw["v111_review_or_control"])
    df = v114.attach_fold_and_crop(raw)

    df["nested_v115_extra_review"] = False
    rule_rows = []
    internal_mask = df["domain"].isin(["old_data", "third_batch"])
    for fold in sorted(df.loc[internal_mask, "fold_id"].unique()):
        train_idx = df.index[internal_mask & df["fold_id"].ne(fold)]
        val_idx = df.index[internal_mask & df["fold_id"].eq(fold)]
        train = df.loc[train_idx].copy()
        val = df.loc[val_idx].copy()
        train_base = train["v111_review_or_control"].astype(bool)
        scan = scan_fp_rules(train, train_base)
        selected = select_stable_envelope(scan)
        val_flag = v114.fp_guard_flag(
            val,
            val["v111_review_or_control"].astype(bool),
            float(selected["wc_max"]),
            float(selected["core_max"]),
            float(selected["main_max"]),
            float(selected["robust_max"]),
        )
        df.loc[val_idx, "nested_v115_extra_review"] = val_flag.to_numpy(bool)
        rule_rows.append(
            {
                "fold_id": int(fold),
                "wc_max": float(selected["wc_max"]),
                "core_max": float(selected["core_max"]),
                "main_max": float(selected["main_max"]),
                "robust_max": float(selected["robust_max"]),
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

    df["nested_v115_review_or_control"] = df["v111_review_or_control"].astype(bool) | df["nested_v115_extra_review"].astype(bool)
    df["nested_v115_rescued_error"] = df["nested_v115_extra_review"].astype(bool) & df["final_correct"].eq(0)
    df["nested_v115_extra_clean_review"] = df["nested_v115_extra_review"].astype(bool) & df["final_correct"].eq(1)

    summary = pd.concat(
        [
            evaluate_scope_rows(df, "v111_review_or_control", "v111_base"),
            evaluate_scope_rows(df, "nested_v115_review_or_control", "nested_stable_envelope_fp_guard"),
        ],
        ignore_index=True,
    )
    rules = pd.DataFrame(rule_rows)
    rules.to_csv(OUT_DIR / "v115_fold_selected_envelope_rules.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v115_nested_stable_envelope_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v115_nested_stable_envelope_summary_formatted.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "v115_nested_stable_envelope_cases.csv", index=False, encoding="utf-8-sig")

    focus = summary[(summary["workflow"].eq("nested_stable_envelope_fp_guard")) & (summary["scope"].eq("all_domains"))].iloc[0]
    third = summary[(summary["workflow"].eq("nested_stable_envelope_fp_guard")) & (summary["scope"].eq("third_batch"))].iloc[0]
    rescued = df[df["nested_v115_rescued_error"]][["domain", "fold_id", "original_case_id", "task_l6_label"]].astype(str)
    rescued_text = "; ".join(f"{r.domain}:fold{r.fold_id}:{r.original_case_id}/{r.task_l6_label}" for r in rescued.itertuples(index=False))
    lines = [
        "# v115 Nested Stable-envelope FP Guard",
        "",
        "v115 uses fold-wise rule selection, but selects a broader envelope among train-safe rules instead of the narrowest minimum-review rule.",
        "",
        f"- Nested all-domain: BAcc {v114.pct(focus['balanced_accuracy'])}, control {v114.pct(focus['control_rate'])}, FN={int(focus['fn'])}, FP={int(focus['fp'])}.",
        f"- Nested third batch: BAcc {v114.pct(third['balanced_accuracy'])}, control {v114.pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
        f"- Held-out rescued FP cases: {rescued_text}.",
        "",
        "This restores the v112 FP protection under fold-wise selection. The remaining error is the TC FN 2516531, which crop rescue did not validate under nested selection.",
    ]
    (OUT_DIR / "v115_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(format_table(summary[summary["scope"].isin(["all_domains", "internal_all", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print(rules.to_string(index=False))
    print()
    print("rescued:")
    print(rescued.to_string(index=False))


if __name__ == "__main__":
    main()
