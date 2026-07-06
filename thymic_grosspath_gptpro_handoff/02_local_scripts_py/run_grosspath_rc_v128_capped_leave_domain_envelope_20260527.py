from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v128_capped_leave_domain_envelope_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"

WC_CAP = 0.775
CORE_CAP = 0.825
CUSHION = 6


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


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy", "holdout_bacc"]:
            out[col] = out[col].map(pct)
    return out


def fp_guard(df: pd.DataFrame, base_review: pd.Series, wc_max: float, core_max: float) -> pd.Series:
    auto_high = df["domain"].isin(["old_data", "third_batch"]) & (~base_review.astype(bool)) & df["final_pred"].eq(1)
    return (
        auto_high
        & pd.to_numeric(df["wholecrop_prob"], errors="coerce").le(wc_max)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
    )


def scan_rules(train: pd.DataFrame, base_review: pd.Series) -> pd.DataFrame:
    rows = []
    for wc_max in np.round(np.arange(0.400, WC_CAP + 0.001, 0.025), 3):
        for core_max in np.round(np.arange(0.650, CORE_CAP + 0.001, 0.025), 3):
            extra = fp_guard(train, base_review, float(wc_max), float(core_max))
            review = base_review.astype(bool) | extra
            m = metrics(train, review)
            rows.append(
                {
                    "wc_max": float(wc_max),
                    "core_max": float(core_max),
                    "extra_review_n": int(extra.sum()),
                    "captured_error_n": int((extra & train["final_correct"].eq(0)).sum()),
                    "clean_review_n": int((extra & train["final_correct"].eq(1)).sum()),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def select_capped_envelope(scan: pd.DataFrame, base_m: dict[str, float | int]) -> pd.Series:
    candidates = scan[(scan["fn"].le(base_m["fn"])) & (scan["fp"].eq(0))].copy()
    if candidates.empty:
        candidates = scan.copy()
    min_extra = int(candidates["extra_review_n"].min())
    candidates = candidates[candidates["extra_review_n"].le(min_extra + CUSHION)].copy()
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
        row.update(metrics(sub, sub[review_col].astype(bool)))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v111_review_or_control", "v118_review_or_control"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    df["capped_leave_domain_extra_review"] = False
    internal_domains = ["old_data", "third_batch"]
    rule_rows = []
    rescued_rows = []
    for holdout in internal_domains:
        train_domain = [d for d in internal_domains if d != holdout][0]
        train = df[df["domain"].eq(train_domain)].copy()
        hold = df[df["domain"].eq(holdout)].copy()
        base_m = metrics(train, train["v111_review_or_control"].astype(bool))
        scan = scan_rules(train, train["v111_review_or_control"].astype(bool))
        selected = select_capped_envelope(scan, base_m)
        train_extra = fp_guard(train, train["v111_review_or_control"].astype(bool), float(selected["wc_max"]), float(selected["core_max"]))
        hold_extra = fp_guard(hold, hold["v111_review_or_control"].astype(bool), float(selected["wc_max"]), float(selected["core_max"]))
        df.loc[hold.index, "capped_leave_domain_extra_review"] = hold_extra.to_numpy(bool)
        train_m = metrics(train, train["v111_review_or_control"].astype(bool) | train_extra)
        hold_m = metrics(hold, hold["v111_review_or_control"].astype(bool) | hold_extra)
        rule_rows.append(
            {
                "train_domain": train_domain,
                "holdout_domain": holdout,
                "wc_max": float(selected["wc_max"]),
                "core_max": float(selected["core_max"]),
                "wc_cap": WC_CAP,
                "core_cap": CORE_CAP,
                "cushion": CUSHION,
                "train_extra_review_n": int(train_extra.sum()),
                "train_captured_error_n": int((train_extra & train["final_correct"].eq(0)).sum()),
                "train_clean_review_n": int((train_extra & train["final_correct"].eq(1)).sum()),
                "train_remaining_error_n": int(train_m["remaining_error_n"]),
                "train_fn": int(train_m["fn"]),
                "train_fp": int(train_m["fp"]),
                "holdout_extra_review_n": int(hold_extra.sum()),
                "holdout_captured_error_n": int((hold_extra & hold["final_correct"].eq(0)).sum()),
                "holdout_clean_review_n": int((hold_extra & hold["final_correct"].eq(1)).sum()),
                "holdout_remaining_error_n": int(hold_m["remaining_error_n"]),
                "holdout_fn": int(hold_m["fn"]),
                "holdout_fp": int(hold_m["fp"]),
                "holdout_bacc": float(hold_m["balanced_accuracy"]),
            }
        )
        rescued = hold[hold_extra & hold["final_correct"].eq(0)][
            ["domain", "original_case_id", "task_l6_label", "wholecrop_prob", "prob_mean_core"]
        ].copy()
        rescued["train_domain"] = train_domain
        rescued_rows.append(rescued)

    df["capped_leave_domain_review_or_control"] = (
        df["v111_review_or_control"].astype(bool) | df["capped_leave_domain_extra_review"].astype(bool)
    )
    df["capped_leave_domain_rescued_error"] = df["capped_leave_domain_extra_review"].astype(bool) & df["final_correct"].eq(0)
    df["capped_leave_domain_extra_clean_review"] = df["capped_leave_domain_extra_review"].astype(bool) & df["final_correct"].eq(1)

    summary = pd.concat(
        [
            evaluate_scope_rows(df, "v111_review_or_control", "v111_base"),
            evaluate_scope_rows(df, "v118_review_or_control", "v118_global_all_internal_rule"),
            evaluate_scope_rows(df, "capped_leave_domain_review_or_control", "capped_leave_domain_envelope"),
        ],
        ignore_index=True,
    )
    rules = pd.DataFrame(rule_rows)
    rescued_cases = pd.concat(rescued_rows, ignore_index=True) if rescued_rows else pd.DataFrame()
    summary.to_csv(OUT_DIR / "v128_capped_leave_domain_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v128_capped_leave_domain_summary_formatted.csv", index=False, encoding="utf-8-sig")
    rules.to_csv(OUT_DIR / "v128_capped_leave_domain_rules.csv", index=False, encoding="utf-8-sig")
    format_table(rules).to_csv(OUT_DIR / "v128_capped_leave_domain_rules_formatted.csv", index=False, encoding="utf-8-sig")
    rescued_cases.to_csv(OUT_DIR / "v128_capped_leave_domain_rescued_cases.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "v128_capped_leave_domain_cases.csv", index=False, encoding="utf-8-sig")

    all_row = summary[
        summary["workflow"].eq("capped_leave_domain_envelope") & summary["scope"].eq("all_domains")
    ].iloc[0]
    rescued_text = "; ".join(rescued_cases["original_case_id"].astype(str).tolist()) if not rescued_cases.empty else "none"
    lines = [
        "# v128 Capped Leave-domain Envelope",
        "",
        f"v128 uses a capped stable-envelope rule with wc_cap={WC_CAP:.3f}, core_cap={CORE_CAP:.3f}, cushion={CUSHION}.",
        "",
        f"- All-domain: BAcc {pct(all_row['balanced_accuracy'])}, control {pct(all_row['control_rate'])}, FN={int(all_row['fn'])}, FP={int(all_row['fp'])}.",
        f"- Held-out rescued cases: {rescued_text}.",
        "",
        "This checks whether a capped envelope can preserve the v118 result under leave-internal-domain threshold selection without the over-review seen in v127.",
    ]
    (OUT_DIR / "v128_key_messages.md").write_text("\n".join(lines), encoding="utf-8-sig")

    print("Wrote", OUT_DIR)
    print(format_table(summary[summary["scope"].isin(["all_domains", "old_data", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print(format_table(rules).to_string(index=False))
    print()
    print(rescued_cases.to_string(index=False))


if __name__ == "__main__":
    main()
