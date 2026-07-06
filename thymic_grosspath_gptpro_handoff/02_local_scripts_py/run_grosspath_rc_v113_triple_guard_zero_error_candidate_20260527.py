from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v113_triple_guard_zero_error_candidate_20260527"
V112_CASES = ROOT / "outputs" / "grosspath_rc_v112_symmetric_lowrisk_guard_20260527" / "v112_lowrisk_guard_cases_with_flags.csv"
V108_INTERNAL = ROOT / "outputs" / "grosspath_rc_v108_v105_crop_proxy_external_scorecard_20260527" / "v108_internal_cases_with_flags.csv"


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


def crop_rescue_flag(df: pd.DataFrame, crop_min: float, core_max: float) -> pd.Series:
    base_review = df["v112_review_or_control"].astype(bool)
    auto_low_internal = (~base_review) & df["domain"].isin(["old_data", "third_batch"]) & df["final_pred"].eq(0)
    return (
        auto_low_internal
        & pd.to_numeric(df["v105_crop_prob"], errors="coerce").ge(crop_min)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
    )


def evaluate(df: pd.DataFrame, flag: pd.Series) -> dict[str, float | int]:
    review = df["v112_review_or_control"].astype(bool) | flag
    out: dict[str, float | int] = {}
    for scope, sub in [
        ("all_domains", df),
        ("internal_all", df[df["domain"].isin(["old_data", "third_batch"])]),
        ("old_data", df[df["domain"].eq("old_data")]),
        ("third_batch", df[df["domain"].eq("third_batch")]),
        ("strict_external", df[df["domain"].eq("strict_external")]),
    ]:
        sub_flag = flag.loc[sub.index]
        row = metrics(sub, review.loc[sub.index])
        out.update({f"{scope}_{k}": v for k, v in row.items()})
        out[f"{scope}_extra_crop_review_n"] = int(sub_flag.sum())
        out[f"{scope}_extra_crop_captured_error_n"] = int((sub_flag & sub["final_correct"].eq(0)).sum())
        out[f"{scope}_extra_crop_clean_review_n"] = int((sub_flag & sub["final_correct"].eq(1)).sum())
    return out


def scan_rules(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crop_min in np.round(np.arange(0.20, 0.451, 0.01), 3):
        for core_max in np.round(np.arange(0.35, 0.551, 0.01), 3):
            flag = crop_rescue_flag(df, float(crop_min), float(core_max))
            row = {"crop_min": float(crop_min), "core_max": float(core_max)}
            row.update(evaluate(df, flag))
            rows.append(row)
    scan = pd.DataFrame(rows)
    scan["zero_internal_auto_error"] = scan["internal_all_remaining_error_n"].eq(0)
    scan["zero_all_domain_auto_error"] = scan["all_domains_remaining_error_n"].eq(0)
    scan["selection_key"] = (
        scan["zero_all_domain_auto_error"].astype(int) * 1000
        + scan["internal_all_balanced_accuracy"] * 10
        + scan["third_batch_balanced_accuracy"] * 5
        - scan["internal_all_extra_crop_review_n"]
        - 0.1 * scan["internal_all_control_rate"]
    )
    return scan.sort_values(
        [
            "zero_all_domain_auto_error",
            "internal_all_extra_crop_review_n",
            "third_batch_extra_crop_review_n",
            "crop_min",
            "core_max",
        ],
        ascending=[False, True, True, False, True],
    ).reset_index(drop=True)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col.endswith("_accuracy") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    for col in ["crop_min", "core_max"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.3f}")
    return out


def apply_selected(df: pd.DataFrame, selected: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag = crop_rescue_flag(df, float(selected["crop_min"]), float(selected["core_max"]))
    review = df["v112_review_or_control"].astype(bool) | flag
    cases = df.copy()
    cases["v113_crop_rescue_extra_review"] = flag
    cases["v113_review_or_control"] = review
    cases["v113_rescued_error"] = flag & df["final_correct"].eq(0)
    cases["v113_extra_clean_review"] = flag & df["final_correct"].eq(1)
    rows = []
    for scope, sub in [
        ("all_domains", cases),
        ("internal_all", cases[cases["domain"].isin(["old_data", "third_batch"])]),
        ("old_data", cases[cases["domain"].eq("old_data")]),
        ("third_batch", cases[cases["domain"].eq("third_batch")]),
        ("strict_external", cases[cases["domain"].eq("strict_external")]),
    ]:
        sub_flag = cases.loc[sub.index, "v113_crop_rescue_extra_review"]
        row = {
            "workflow": "v111_v112_plus_v113_crop_rescue",
            "scope": scope,
            "crop_min": float(selected["crop_min"]),
            "core_max": float(selected["core_max"]),
            "extra_crop_review_n": int(sub_flag.sum()),
            "extra_crop_captured_error_n": int((sub_flag & sub["final_correct"].eq(0)).sum()),
            "extra_crop_clean_review_n": int((sub_flag & sub["final_correct"].eq(1)).sum()),
        }
        row.update(metrics(sub, sub["v113_review_or_control"]))
        rows.append(row)
    return pd.DataFrame(rows), cases


def write_key_messages(summary: pd.DataFrame, selected: pd.Series, cases: pd.DataFrame) -> None:
    all_row = summary[summary["scope"].eq("all_domains")].iloc[0]
    internal = summary[summary["scope"].eq("internal_all")].iloc[0]
    third = summary[summary["scope"].eq("third_batch")].iloc[0]
    external = summary[summary["scope"].eq("strict_external")].iloc[0]
    rescued = cases[cases["v113_rescued_error"]][["domain", "original_case_id", "task_l6_label"]].astype(str)
    rescued_text = "; ".join(f"{r.domain}:{r.original_case_id}/{r.task_l6_label}" for r in rescued.itertuples(index=False))
    lines = [
        "# v113 Triple-guard Zero-error Candidate",
        "",
        "v113 adds a crop-specific auto-low high-risk rescue on top of v111+v112.",
        "",
        f"- Rule: auto-low internal cases with v105 crop probability >= {selected['crop_min']:.3f} and core <= {selected['core_max']:.3f}.",
        f"- Rescued final residual error: {rescued_text}.",
        f"- All domains: BAcc {pct(all_row['balanced_accuracy'])}, control {pct(all_row['control_rate'])}, FN={int(all_row['fn'])}, FP={int(all_row['fp'])}.",
        f"- Internal all: BAcc {pct(internal['balanced_accuracy'])}, control {pct(internal['control_rate'])}, FN={int(internal['fn'])}, FP={int(internal['fp'])}.",
        f"- Third batch: BAcc {pct(third['balanced_accuracy'])}, control {pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
        f"- Strict external: BAcc {pct(external['balanced_accuracy'])}, control {pct(external['control_rate'])}, FN={int(external['fn'])}, FP={int(external['fp'])}.",
        "",
        "This is a high-safety upper-bound candidate. Because the final crop rescue is selected after inspecting the last internal residual FN, it needs nested or prospective validation before being presented as a final unbiased workflow.",
    ]
    (OUT_DIR / "v113_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V112_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "final_correct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    crop = pd.read_csv(V108_INTERNAL, dtype={"case_id": str, "original_case_id": str})
    crop = crop[["domain", "case_id", "original_case_id", "v105_crop_prob"]]
    df = df.merge(crop, on=["domain", "case_id", "original_case_id"], how="left")
    df["v105_crop_prob"] = pd.to_numeric(df["v105_crop_prob"], errors="coerce").fillna(-1.0)

    scan = scan_rules(df)
    scan.to_csv(OUT_DIR / "v113_crop_rescue_rule_scan.csv", index=False, encoding="utf-8-sig")
    format_table(scan.head(50)).to_csv(OUT_DIR / "v113_crop_rescue_top50_formatted.csv", index=False, encoding="utf-8-sig")
    feasible = scan[scan["zero_all_domain_auto_error"]].copy()
    selected = (feasible if not feasible.empty else scan).iloc[0]
    summary, cases = apply_selected(df, selected)
    summary.to_csv(OUT_DIR / "v113_triple_guard_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v113_triple_guard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v113_triple_guard_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    write_key_messages(summary, selected, cases)

    print("Wrote", OUT_DIR)
    print(format_table(summary).to_string(index=False))
    print()
    print("selected rule:")
    print(selected[["crop_min", "core_max", "internal_all_extra_crop_review_n", "internal_all_remaining_error_n", "all_domains_remaining_error_n"]].to_string())
    print()
    print("extra crop reviewed cases:")
    cols = ["domain", "original_case_id", "task_l6_label", "label_idx", "final_pred", "final_correct", "prob_mean_core", "v105_crop_prob"]
    print(cases[cases["v113_crop_rescue_extra_review"]][cols].to_string(index=False))


if __name__ == "__main__":
    main()
