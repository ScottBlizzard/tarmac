from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v130_leave_one_fp_out_validation_20260527"
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
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy", "full_bacc"]:
            out[col] = out[col].map(pct)
    return out


def fp_guard(df: pd.DataFrame, base_review: pd.Series, wc_max: float, core_max: float) -> pd.Series:
    auto_high = df["domain"].isin(["old_data", "third_batch"]) & (~base_review.astype(bool)) & df["final_pred"].eq(1)
    return (
        auto_high
        & pd.to_numeric(df["wholecrop_prob"], errors="coerce").le(wc_max)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
    )


def scan_rules(train: pd.DataFrame, base_review: pd.Series, capped: bool) -> pd.DataFrame:
    wc_stop = WC_CAP if capped else 0.900
    core_stop = CORE_CAP if capped else 0.900
    rows = []
    for wc_max in np.round(np.arange(0.400, wc_stop + 0.001, 0.025), 3):
        for core_max in np.round(np.arange(0.650, core_stop + 0.001, 0.025), 3):
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


def select_rule(scan: pd.DataFrame, base_m: dict[str, float | int], selector: str) -> pd.Series:
    candidates = scan[(scan["fn"].le(base_m["fn"])) & (scan["fp"].eq(0))].copy()
    if candidates.empty:
        candidates = scan.copy()
    if selector == "min_review":
        return candidates.sort_values(
            ["extra_review_n", "clean_review_n", "remaining_error_n", "balanced_accuracy", "wc_max", "core_max"],
            ascending=[True, True, True, False, True, True],
        ).iloc[0]
    if selector == "capped_stable_envelope":
        min_extra = int(candidates["extra_review_n"].min())
        candidates = candidates[candidates["extra_review_n"].le(min_extra + CUSHION)].copy()
        return candidates.sort_values(
            ["wc_max", "core_max", "balanced_accuracy", "extra_review_n"],
            ascending=[False, False, False, True],
        ).iloc[0]
    raise ValueError(selector)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v111_review_or_control", "v118_review_or_control", "v118_extra_review"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    base_review = df["v111_review_or_control"].astype(bool)
    residual_fp = df[
        df["domain"].isin(["old_data", "third_batch"])
        & (~base_review)
        & df["final_correct"].eq(0)
        & df["label_idx"].eq(0)
        & df["final_pred"].eq(1)
    ].copy()
    if residual_fp.empty:
        raise RuntimeError("No residual FP cases found.")

    rows = []
    for held in residual_fp.itertuples():
        train = df.drop(index=held.Index).copy()
        for selector, capped in [("min_review", False), ("capped_stable_envelope", True)]:
            train_base = train["v111_review_or_control"].astype(bool)
            train_m = metrics(train, train_base)
            scan = scan_rules(train, train_base, capped=capped)
            selected = select_rule(scan, train_m, selector)
            full_extra = fp_guard(df, base_review, float(selected["wc_max"]), float(selected["core_max"]))
            full_review = base_review | full_extra
            full_m = metrics(df, full_review)
            held_captured = bool(full_extra.loc[held.Index])
            rows.append(
                {
                    "selector": selector,
                    "heldout_case_id": str(held.original_case_id),
                    "heldout_domain": str(held.domain),
                    "heldout_label": str(held.task_l6_label),
                    "heldout_wholecrop_prob": float(held.wholecrop_prob),
                    "heldout_core_prob": float(held.prob_mean_core),
                    "wc_max": float(selected["wc_max"]),
                    "core_max": float(selected["core_max"]),
                    "train_extra_review_n": int(selected["extra_review_n"]),
                    "train_clean_review_n": int(selected["clean_review_n"]),
                    "train_remaining_error_n": int(selected["remaining_error_n"]),
                    "train_fn": int(selected["fn"]),
                    "train_fp": int(selected["fp"]),
                    "heldout_captured": held_captured,
                    "full_extra_review_n": int(full_extra.sum()),
                    "full_captured_error_n": int((full_extra & df["final_correct"].eq(0)).sum()),
                    "full_clean_review_n": int((full_extra & df["final_correct"].eq(1)).sum()),
                    "full_remaining_error_n": int(full_m["remaining_error_n"]),
                    "full_fn": int(full_m["fn"]),
                    "full_fp": int(full_m["fp"]),
                    "full_bacc": float(full_m["balanced_accuracy"]),
                    "full_control_rate": float(full_m["control_rate"]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("selector", sort=False)
        .agg(
            heldout_cases=("heldout_case_id", "count"),
            heldout_captured_n=("heldout_captured", "sum"),
            mean_full_extra_review_n=("full_extra_review_n", "mean"),
            mean_full_clean_review_n=("full_clean_review_n", "mean"),
            mean_full_control_rate=("full_control_rate", "mean"),
            min_full_bacc=("full_bacc", "min"),
            max_full_fp=("full_fp", "max"),
            max_full_fn=("full_fn", "max"),
        )
        .reset_index()
    )
    summary["heldout_capture_rate"] = summary["heldout_captured_n"] / summary["heldout_cases"]

    detail.to_csv(OUT_DIR / "v130_leave_one_fp_out_detail.csv", index=False, encoding="utf-8-sig")
    format_table(detail).to_csv(OUT_DIR / "v130_leave_one_fp_out_detail_formatted.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v130_leave_one_fp_out_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v130_leave_one_fp_out_summary_formatted.csv", index=False, encoding="utf-8-sig")

    min_row = summary[summary["selector"].eq("min_review")].iloc[0]
    cap_row = summary[summary["selector"].eq("capped_stable_envelope")].iloc[0]
    lines = [
        "# v130 Leave-one-FP-out Validation",
        "",
        "v130 removes each residual FP from threshold selection, then checks whether the selected two-signal rule still captures the held-out FP.",
        "",
        f"- Minimum-review selector captured {int(min_row['heldout_captured_n'])}/{int(min_row['heldout_cases'])} held-out FP cases; worst full FP={int(min_row['max_full_fp'])}.",
        f"- Capped stable-envelope selector captured {int(cap_row['heldout_captured_n'])}/{int(cap_row['heldout_cases'])} held-out FP cases; worst full FP={int(cap_row['max_full_fp'])}, mean control {pct(cap_row['mean_full_control_rate'])}.",
        "",
        "This directly checks whether the FP guard generalizes across the three residual FP examples rather than simply memorizing all of them.",
    ]
    (OUT_DIR / "v130_key_messages.md").write_text("\n".join(lines), encoding="utf-8-sig")

    print("Wrote", OUT_DIR)
    print(format_table(summary).to_string(index=False))
    print()
    print(format_table(detail).to_string(index=False))


if __name__ == "__main__":
    main()
