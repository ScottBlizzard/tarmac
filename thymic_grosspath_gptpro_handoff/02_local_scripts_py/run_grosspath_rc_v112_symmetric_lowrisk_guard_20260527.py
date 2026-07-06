from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v112_symmetric_lowrisk_guard_20260527"
V111_CASES = ROOT / "outputs" / "grosspath_rc_v111_branch_specific_v109_v79strict_20260527" / "v111_branch_specific_cases_with_flags.csv"


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


def candidate_flag(
    df: pd.DataFrame,
    wc_max: float,
    core_max: float,
    main_max: float,
    robust_max: float,
) -> pd.Series:
    base_review = df["v111_review_or_control"].astype(bool)
    auto_high = (~base_review) & df["domain"].isin(["old_data", "third_batch"]) & df["final_pred"].eq(1)
    return (
        auto_high
        & pd.to_numeric(df["wholecrop_prob"], errors="coerce").le(wc_max)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
        & pd.to_numeric(df["main_prob"], errors="coerce").le(main_max)
        & pd.to_numeric(df["robust_prob"], errors="coerce").le(robust_max)
    )


def evaluate_rule(df: pd.DataFrame, flag: pd.Series) -> dict[str, float | int]:
    review = df["v111_review_or_control"].astype(bool) | flag
    out: dict[str, float | int] = {}
    for scope, sub in [
        ("all_domains", df),
        ("internal_all", df[df["domain"].isin(["old_data", "third_batch"])]),
        ("old_data", df[df["domain"].eq("old_data")]),
        ("third_batch", df[df["domain"].eq("third_batch")]),
        ("strict_external", df[df["domain"].eq("strict_external")]),
    ]:
        sub_flag = flag.loc[sub.index]
        sub_review = review.loc[sub.index]
        m = metrics(sub, sub_review)
        out.update({f"{scope}_{k}": v for k, v in m.items()})
        out[f"{scope}_extra_fp_guard_review_n"] = int(sub_flag.sum())
        out[f"{scope}_extra_fp_guard_captured_error_n"] = int((sub_flag & sub["final_correct"].eq(0)).sum())
        out[f"{scope}_extra_fp_guard_clean_review_n"] = int((sub_flag & sub["final_correct"].eq(1)).sum())
    return out


def scan_rules(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # The broad exploratory pass showed useful candidates only around this
    # compact region; keeping the formal scan narrow avoids spending minutes
    # on thresholds that cannot capture all residual internal FP cases.
    for wc_max in np.round(np.arange(0.500, 0.701, 0.025), 3):
        for core_max in np.round(np.arange(0.725, 0.826, 0.025), 3):
            for main_max in np.round(np.arange(0.900, 0.976, 0.025), 3):
                for robust_max in np.round(np.arange(0.750, 0.851, 0.025), 3):
                    flag = candidate_flag(df, float(wc_max), float(core_max), float(main_max), float(robust_max))
                    row = {
                        "wc_max": float(wc_max),
                        "core_max": float(core_max),
                        "main_max": float(main_max),
                        "robust_max": float(robust_max),
                    }
                    row.update(evaluate_rule(df, flag))
                    rows.append(row)
    scan = pd.DataFrame(rows)
    scan["passes_internal_constraint"] = (
        scan["internal_all_fn"].le(1)
        & scan["third_batch_fn"].le(1)
        & scan["internal_all_fp"].le(0)
        & scan["third_batch_fp"].le(0)
        & scan["internal_all_balanced_accuracy"].ge(0.997)
        & scan["third_batch_balanced_accuracy"].ge(0.993)
    )
    scan["selection_key"] = (
        scan["passes_internal_constraint"].astype(int) * 1000
        + scan["internal_all_balanced_accuracy"] * 10
        + scan["third_batch_balanced_accuracy"] * 5
        - scan["internal_all_control_rate"]
        - 0.3 * scan["third_batch_control_rate"]
    )
    return scan.sort_values(
        [
            "passes_internal_constraint",
            "internal_all_extra_fp_guard_review_n",
            "third_batch_extra_fp_guard_review_n",
            "internal_all_balanced_accuracy",
            "third_batch_balanced_accuracy",
        ],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col.endswith("_accuracy") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    for col in ["wc_max", "core_max", "main_max", "robust_max"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.3f}")
    return out


def apply_selected(df: pd.DataFrame, selected: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag = candidate_flag(
        df,
        float(selected["wc_max"]),
        float(selected["core_max"]),
        float(selected["main_max"]),
        float(selected["robust_max"]),
    )
    review = df["v111_review_or_control"].astype(bool) | flag
    flagged = df.copy()
    flagged["v112_lowrisk_guard_extra_review"] = flag
    flagged["v112_review_or_control"] = review
    flagged["v112_rescued_error"] = flag & df["final_correct"].eq(0)
    flagged["v112_extra_clean_review"] = flag & df["final_correct"].eq(1)
    rows = []
    for scope, sub in [
        ("all_domains", flagged),
        ("internal_all", flagged[flagged["domain"].isin(["old_data", "third_batch"])]),
        ("old_data", flagged[flagged["domain"].eq("old_data")]),
        ("third_batch", flagged[flagged["domain"].eq("third_batch")]),
        ("strict_external", flagged[flagged["domain"].eq("strict_external")]),
    ]:
        sub_flag = flagged.loc[sub.index, "v112_lowrisk_guard_extra_review"]
        row = {
            "workflow": "v111_plus_v112_symmetric_lowrisk_guard",
            "scope": scope,
            "wc_max": float(selected["wc_max"]),
            "core_max": float(selected["core_max"]),
            "main_max": float(selected["main_max"]),
            "robust_max": float(selected["robust_max"]),
            "extra_fp_guard_review_n": int(sub_flag.sum()),
            "extra_fp_guard_captured_error_n": int((sub_flag & sub["final_correct"].eq(0)).sum()),
            "extra_fp_guard_clean_review_n": int((sub_flag & sub["final_correct"].eq(1)).sum()),
        }
        row.update(metrics(sub, sub["v112_review_or_control"]))
        rows.append(row)
    return pd.DataFrame(rows), flagged


def write_key_messages(summary: pd.DataFrame, selected: pd.Series, cases: pd.DataFrame) -> None:
    all_row = summary[summary["scope"].eq("all_domains")].iloc[0]
    internal = summary[summary["scope"].eq("internal_all")].iloc[0]
    third = summary[summary["scope"].eq("third_batch")].iloc[0]
    external = summary[summary["scope"].eq("strict_external")].iloc[0]
    rescued = cases[cases["v112_rescued_error"]][["domain", "original_case_id", "task_l6_label"]].astype(str)
    rescued_text = "; ".join(f"{r.domain}:{r.original_case_id}/{r.task_l6_label}" for r in rescued.itertuples(index=False))
    lines = [
        "# v112 Symmetric Low-risk Guard",
        "",
        "The rule is selected only on old+third internal data and audits strict external afterward.",
        "",
        f"- Rule: auto-high internal cases with wholecrop <= {selected['wc_max']:.3f}, core <= {selected['core_max']:.3f}, main <= {selected['main_max']:.3f}, robust <= {selected['robust_max']:.3f}.",
        f"- Rescued internal FP cases: {rescued_text}.",
        f"- All domains: BAcc {pct(all_row['balanced_accuracy'])}, control {pct(all_row['control_rate'])}, FN={int(all_row['fn'])}, FP={int(all_row['fp'])}.",
        f"- Internal all: BAcc {pct(internal['balanced_accuracy'])}, control {pct(internal['control_rate'])}, FN={int(internal['fn'])}, FP={int(internal['fp'])}.",
        f"- Third batch: BAcc {pct(third['balanced_accuracy'])}, control {pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
        f"- Strict external: BAcc {pct(external['balanced_accuracy'])}, control {pct(external['control_rate'])}, FN={int(external['fn'])}, FP={int(external['fp'])}.",
        "",
        "This is a symmetric complement to the high-risk FN scorecard: v109 rescues auto-low high-risk misses, while v112 catches auto-high low-risk upgrades.",
    ]
    (OUT_DIR / "v112_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V111_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "final_correct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    scan = scan_rules(df)
    scan.to_csv(OUT_DIR / "v112_lowrisk_guard_rule_scan.csv", index=False, encoding="utf-8-sig")
    format_table(scan.head(50)).to_csv(OUT_DIR / "v112_lowrisk_guard_top50_formatted.csv", index=False, encoding="utf-8-sig")

    feasible = scan[scan["passes_internal_constraint"]].copy()
    selected = (feasible if not feasible.empty else scan).iloc[0]
    summary, cases = apply_selected(df, selected)
    summary.to_csv(OUT_DIR / "v112_lowrisk_guard_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v112_lowrisk_guard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v112_lowrisk_guard_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    write_key_messages(summary, selected, cases)

    print("Wrote", OUT_DIR)
    print(format_table(summary).to_string(index=False))
    print()
    print("selected rule:")
    print(selected[["wc_max", "core_max", "main_max", "robust_max", "internal_all_extra_fp_guard_review_n", "internal_all_fp", "third_batch_fp"]].to_string())
    print()
    print("rescued cases:")
    cols = ["domain", "original_case_id", "task_l6_label", "label_idx", "final_pred", "prob_mean_core", "main_prob", "robust_prob", "wholecrop_prob"]
    print(cases[cases["v112_rescued_error"]][cols].to_string(index=False))


if __name__ == "__main__":
    main()
