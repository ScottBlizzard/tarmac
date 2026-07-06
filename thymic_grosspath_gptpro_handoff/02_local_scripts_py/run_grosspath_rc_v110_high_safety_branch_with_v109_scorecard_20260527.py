from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v110_high_safety_branch_with_v109_scorecard_20260527"

ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
UNIFIED_OOF = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "44_old_third_unified_feature_cv_20260523"
    / "unified_feature_cv_all_oof_predictions.csv"
)
V106_EXTERNAL_PROB = (
    ROOT
    / "outputs"
    / "grosspath_rc_v106_external_compatible_wholecrop_scorecard_20260527"
    / "v106_strict_external_wholecrop_refit_probabilities.csv"
)

LIGHT_POLICY = "adaptive_v50_to_v79_light"
STRICT_POLICY = "adaptive_v50_to_v79_strict"
VARIANT = "whole_crop"
MODEL_NAME = "logreg_bal_c0.03"
WEIGHT_MODE = "domain_label"
PROB_THRESHOLD = 0.25
CORE_MAX = 0.325


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


def attach_wholecrop_prob(frame: pd.DataFrame) -> pd.DataFrame:
    internal = frame[frame["domain"].isin(["old_data", "third_batch"])].copy()
    external = frame[frame["domain"].eq("strict_external")].copy()

    if not internal.empty:
        internal["domain_key"] = internal["domain"].map({"old_data": "old", "third_batch": "third"})
        oof = pd.read_csv(UNIFIED_OOF, dtype={"original_case_id": str})
        oof = oof.loc[oof["variant"].eq(VARIANT) & oof["model"].eq(MODEL_NAME) & oof["weight_mode"].eq(WEIGHT_MODE)].copy()
        oof["domain_key"] = oof["domain"].astype(str)
        prob = oof[["domain_key", "original_case_id", "oof_prob_high"]].rename(columns={"oof_prob_high": "wholecrop_prob"})
        internal = internal.merge(prob, on=["domain_key", "original_case_id"], how="left", validate="many_to_one")
        if internal["wholecrop_prob"].isna().any():
            missing = internal.loc[internal["wholecrop_prob"].isna(), ["case_id", "original_case_id", "domain"]].head(10)
            raise RuntimeError(f"Missing internal wholecrop probabilities:\n{missing}")
        internal = internal.drop(columns=["domain_key"])

    if not external.empty:
        ext_prob = pd.read_csv(V106_EXTERNAL_PROB, dtype={"case_id": str}).rename(columns={"wholecrop_refit_prob": "wholecrop_prob"})
        external = external.merge(ext_prob[["case_id", "wholecrop_prob"]], on="case_id", how="left", validate="many_to_one")
        if external["wholecrop_prob"].isna().any():
            missing = external.loc[external["wholecrop_prob"].isna(), ["case_id", "original_case_id"]].head(10)
            raise RuntimeError(f"Missing external wholecrop probabilities:\n{missing}")

    return pd.concat([internal, external], ignore_index=True, sort=False)


def apply_scorecard(df: pd.DataFrame) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    return (
        low_auto
        & pd.to_numeric(df["wholecrop_prob"], errors="coerce").ge(PROB_THRESHOLD)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(CORE_MAX)
    )


def evaluate(policy_name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra = apply_scorecard(frame)
    review = frame["review_or_control"].astype(bool) | extra
    flagged = frame.copy()
    flagged["v110_extra_review"] = extra
    flagged["v110_review_or_control"] = review
    flagged["v110_rescued_error"] = extra & frame["final_correct"].eq(0)
    flagged["v110_extra_clean_review"] = extra & frame["final_correct"].eq(1)
    rows = []
    scopes = [("all_domains", frame)] + list(frame.groupby("domain", sort=False))
    internal = frame[frame["domain"].isin(["old_data", "third_batch"])]
    if not internal.empty:
        scopes.insert(1, ("internal_all", internal))
    for scope, sub in scopes:
        sub_extra = extra.loc[sub.index]
        row = {
            "workflow": policy_name,
            "scope": scope,
            "prob_threshold": PROB_THRESHOLD,
            "core_max": CORE_MAX,
            "extra_review_n": int(sub_extra.sum()),
            "extra_captured_error_n": int((sub_extra & sub["final_correct"].eq(0)).sum()),
            "extra_clean_review_n": int((sub_extra & sub["final_correct"].eq(1)).sum()),
        }
        row.update(metrics(sub, review.loc[sub.index]))
        rows.append(row)
    return pd.DataFrame(rows), flagged


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    for col in ["prob_threshold", "core_max"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.3f}")
    return out


def write_key_messages(summary: pd.DataFrame) -> None:
    lines = [
        "# v110 High-safety Branch with v109 Scorecard",
        "",
        f"Scorecard: wholecrop probability >= {PROB_THRESHOLD:.3f} and core <= {CORE_MAX:.3f}.",
        "",
    ]
    for workflow in ["light_branch_plus_v109", "strict_branch_plus_v109"]:
        sub = summary[summary["workflow"].eq(workflow)]
        all_row = sub[sub["scope"].eq("all_domains")].iloc[0]
        third = sub[sub["scope"].eq("third_batch")].iloc[0]
        external = sub[sub["scope"].eq("strict_external")].iloc[0]
        lines.extend(
            [
                f"## {workflow}",
                "",
                f"- All domains: BAcc {pct(all_row['balanced_accuracy'])}, control {pct(all_row['control_rate'])}, FN={int(all_row['fn'])}, FP={int(all_row['fp'])}.",
                f"- Third batch: BAcc {pct(third['balanced_accuracy'])}, control {pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
                f"- Strict external: BAcc {pct(external['balanced_accuracy'])}, control {pct(external['control_rate'])}, FN={int(external['fn'])}, FP={int(external['fp'])}.",
                "",
            ]
        )
    (OUT_DIR / "v110_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES, dtype={"case_id": str, "original_case_id": str})

    light = attach_wholecrop_prob(routes[routes["policy"].eq(LIGHT_POLICY)].copy())
    strict = attach_wholecrop_prob(routes[routes["policy"].eq(STRICT_POLICY)].copy())

    light_summary, light_cases = evaluate("light_branch_plus_v109", light)
    strict_summary, strict_cases = evaluate("strict_branch_plus_v109", strict)
    summary = pd.concat([light_summary, strict_summary], ignore_index=True)

    summary.to_csv(OUT_DIR / "v110_high_safety_branch_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v110_high_safety_branch_summary_formatted.csv", index=False, encoding="utf-8-sig")
    light_cases.to_csv(OUT_DIR / "v110_light_branch_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    strict_cases.to_csv(OUT_DIR / "v110_strict_branch_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    write_key_messages(summary)

    focus = summary[summary["scope"].isin(["all_domains", "internal_all", "third_batch", "strict_external"])]
    print("Wrote", OUT_DIR)
    print(format_table(focus).to_string(index=False))


if __name__ == "__main__":
    main()
