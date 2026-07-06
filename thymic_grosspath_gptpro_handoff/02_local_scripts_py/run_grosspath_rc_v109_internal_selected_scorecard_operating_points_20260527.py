from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v109_internal_selected_scorecard_operating_points_20260527"

V106_DIR = ROOT / "outputs" / "grosspath_rc_v106_external_compatible_wholecrop_scorecard_20260527"
V108_DIR = ROOT / "outputs" / "grosspath_rc_v108_v105_crop_proxy_external_scorecard_20260527"


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


def apply_rule(df: pd.DataFrame, prob_col: str, prob_threshold: float, core_max: float) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    return (
        low_auto
        & pd.to_numeric(df[prob_col], errors="coerce").ge(prob_threshold)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(core_max)
    )


def evaluate_rule(
    family: str,
    prob_col: str,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    prob_threshold: float,
    core_max: float,
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "family": family,
        "prob_col": prob_col,
        "prob_threshold": float(prob_threshold),
        "core_max": float(core_max),
    }
    int_extra = apply_rule(internal, prob_col, prob_threshold, core_max)
    int_review = internal["review_or_control"].astype(bool) | int_extra
    for scope, sub in [("internal_all", internal), ("old_data", internal[internal["domain"].eq("old_data")]), ("third_batch", internal[internal["domain"].eq("third_batch")])]:
        sub_extra = int_extra.loc[sub.index]
        m = metrics(sub, int_review.loc[sub.index])
        out.update({f"{scope}_{k}": v for k, v in m.items()})
        out[f"{scope}_extra_review_n"] = int(sub_extra.sum())
        out[f"{scope}_extra_captured_error_n"] = int((sub_extra & sub["final_correct"].eq(0)).sum())
        out[f"{scope}_extra_clean_review_n"] = int((sub_extra & sub["final_correct"].eq(1)).sum())

    ext_extra = apply_rule(external, prob_col, prob_threshold, core_max)
    ext_review = external["review_or_control"].astype(bool) | ext_extra
    m = metrics(external, ext_review)
    out.update({f"strict_external_{k}": v for k, v in m.items()})
    out["strict_external_extra_review_n"] = int(ext_extra.sum())
    out["strict_external_extra_captured_error_n"] = int((ext_extra & external["final_correct"].eq(0)).sum())
    out["strict_external_extra_clean_review_n"] = int((ext_extra & external["final_correct"].eq(1)).sum())
    return out


def scan_family(family: str, prob_col: str, internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prob_threshold in np.round(np.arange(0.20, 0.601, 0.025), 3):
        for core_max in np.round(np.arange(0.20, 0.601, 0.025), 3):
            rows.append(evaluate_rule(family, prob_col, internal, external, float(prob_threshold), float(core_max)))
    frame = pd.DataFrame(rows)
    frame["passes_internal_constraint"] = (
        frame["internal_all_balanced_accuracy"].ge(0.990)
        & frame["third_batch_balanced_accuracy"].ge(0.985)
        & frame["internal_all_fn"].le(1)
        & frame["third_batch_fn"].le(1)
    )
    frame["internal_selection_key"] = (
        frame["passes_internal_constraint"].astype(int) * 1000
        + frame["third_batch_balanced_accuracy"] * 10
        + frame["internal_all_balanced_accuracy"] * 5
        - frame["internal_all_control_rate"]
        - 0.5 * frame["third_batch_control_rate"]
    )
    return frame.sort_values(
        [
            "passes_internal_constraint",
            "internal_all_control_rate",
            "third_batch_control_rate",
            "third_batch_balanced_accuracy",
            "internal_all_balanced_accuracy",
        ],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)


def format_for_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "family",
        "prob_threshold",
        "core_max",
        "passes_internal_constraint",
        "internal_all_control_rate",
        "internal_all_balanced_accuracy",
        "internal_all_fn",
        "internal_all_fp",
        "third_batch_control_rate",
        "third_batch_balanced_accuracy",
        "third_batch_fn",
        "third_batch_fp",
        "strict_external_control_rate",
        "strict_external_balanced_accuracy",
        "strict_external_fn",
        "strict_external_fp",
        "strict_external_extra_review_n",
    ]
    out = df[cols].copy()
    for col in out.columns:
        if col.endswith("_rate") or col.endswith("_accuracy"):
            out[col] = out[col].map(pct)
    return out


def write_key_messages(selected: pd.DataFrame) -> None:
    lines = [
        "# v109 Internal-selected Scorecard Operating Points",
        "",
        "Rules are selected only by old+third internal constraints, then applied to strict external for audit.",
        "",
    ]
    for row in selected.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row['family']}",
                "",
                f"- Selected rule: probability >= {row['prob_threshold']:.3f}, core <= {row['core_max']:.3f}.",
                f"- Internal all: BAcc {pct(row['internal_all_balanced_accuracy'])}, control {pct(row['internal_all_control_rate'])}, FN={int(row['internal_all_fn'])}, FP={int(row['internal_all_fp'])}.",
                f"- Third batch: BAcc {pct(row['third_batch_balanced_accuracy'])}, control {pct(row['third_batch_control_rate'])}, FN={int(row['third_batch_fn'])}, FP={int(row['third_batch_fp'])}.",
                f"- Strict external audit: BAcc {pct(row['strict_external_balanced_accuracy'])}, control {pct(row['strict_external_control_rate'])}, FN={int(row['strict_external_fn'])}, FP={int(row['strict_external_fp'])}.",
                "",
            ]
        )
    (OUT_DIR / "v109_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    whole_internal = pd.read_csv(V106_DIR / "v106_internal_cases_with_flags.csv", dtype={"case_id": str, "original_case_id": str})
    whole_external = pd.read_csv(V106_DIR / "v106_strict_external_cases_with_flags.csv", dtype={"case_id": str, "original_case_id": str})
    crop_internal = pd.read_csv(V108_DIR / "v108_internal_cases_with_flags.csv", dtype={"case_id": str, "original_case_id": str})
    crop_external = pd.read_csv(V108_DIR / "v108_strict_external_cases_with_flags.csv", dtype={"case_id": str, "original_case_id": str})

    whole_scan = scan_family("wholecrop_refit", "wholecrop_prob", whole_internal, whole_external.rename(columns={"wholecrop_refit_prob": "wholecrop_prob"}))
    crop_scan = scan_family("v105_crop_proxy", "v105_crop_prob", crop_internal, crop_external.rename(columns={"v105_crop_proxy_prob": "v105_crop_prob"}))
    all_scan = pd.concat([whole_scan, crop_scan], ignore_index=True)
    all_scan.to_csv(OUT_DIR / "v109_all_rule_scan.csv", index=False, encoding="utf-8-sig")

    selected_rows = []
    for family, group in all_scan.groupby("family", sort=False):
        candidates = group[group["passes_internal_constraint"]].copy()
        if candidates.empty:
            candidates = group.copy()
        selected_rows.append(candidates.sort_values(["internal_all_control_rate", "third_batch_control_rate"], ascending=[True, True]).iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.to_csv(OUT_DIR / "v109_internal_selected_operating_points.csv", index=False, encoding="utf-8-sig")
    format_for_report(selected).to_csv(OUT_DIR / "v109_internal_selected_operating_points_formatted.csv", index=False, encoding="utf-8-sig")
    format_for_report(all_scan.head(50)).to_csv(OUT_DIR / "v109_top50_rule_scan_formatted.csv", index=False, encoding="utf-8-sig")
    write_key_messages(selected)

    print("Wrote", OUT_DIR)
    print(format_for_report(selected).to_string(index=False))


if __name__ == "__main__":
    main()
