from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v123_paired_stats_confidence_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


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
        "pos_n": pos,
        "neg_n": neg,
    }


def metric_with_ci(df: pd.DataFrame, review: pd.Series, workflow: str, scope: str) -> dict[str, float | int | str]:
    m = metrics(df, review)
    sens_lo, sens_hi = wilson_ci(int(m["pos_n"] - m["fn"]), int(m["pos_n"]))
    spec_lo, spec_hi = wilson_ci(int(m["neg_n"] - m["fp"]), int(m["neg_n"]))
    bacc_lo = (sens_lo + spec_lo) / 2
    bacc_hi = (sens_hi + spec_hi) / 2
    return {
        "workflow": workflow,
        "scope": scope,
        **m,
        "sensitivity_ci_low": sens_lo,
        "sensitivity_ci_high": sens_hi,
        "specificity_ci_low": spec_lo,
        "specificity_ci_high": spec_hi,
        "balanced_accuracy_ci_low": bacc_lo,
        "balanced_accuracy_ci_high": bacc_hi,
    }


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in [
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "sensitivity_ci_low",
            "sensitivity_ci_high",
            "specificity_ci_low",
            "specificity_ci_high",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
            "base_bacc",
            "new_bacc",
            "delta_bacc",
            "delta_bacc_ci_low",
            "delta_bacc_ci_high",
            "delta_control",
        ]:
            out[col] = out[col].map(pct)
    return out


def paired_bootstrap_delta(df: pd.DataFrame, base_review: pd.Series, new_review: pd.Series, n_boot: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(20260527)
    pos_mask = df["label_idx"].eq(1).to_numpy()
    neg_mask = df["label_idx"].eq(0).to_numpy()
    base_correct = (base_review.astype(bool) | df["final_correct"].eq(1)).to_numpy()
    new_correct = (new_review.astype(bool) | df["final_correct"].eq(1)).to_numpy()

    base_pos = base_correct[pos_mask].astype(float)
    base_neg = base_correct[neg_mask].astype(float)
    new_pos = new_correct[pos_mask].astype(float)
    new_neg = new_correct[neg_mask].astype(float)

    base = 0.5 * (base_pos.mean() + base_neg.mean())
    new = 0.5 * (new_pos.mean() + new_neg.mean())

    pos_draw = rng.integers(0, len(base_pos), size=(n_boot, len(base_pos)))
    neg_draw = rng.integers(0, len(base_neg), size=(n_boot, len(base_neg)))
    base_boot = 0.5 * (base_pos[pos_draw].mean(axis=1) + base_neg[neg_draw].mean(axis=1))
    new_boot = 0.5 * (new_pos[pos_draw].mean(axis=1) + new_neg[neg_draw].mean(axis=1))
    arr = new_boot - base_boot
    return {
        "base_bacc": base,
        "new_bacc": new,
        "delta_bacc": new - base,
        "delta_bacc_ci_low": float(np.quantile(arr, 0.025)),
        "delta_bacc_ci_high": float(np.quantile(arr, 0.975)),
    }


def mcnemar_exact(base_correct: pd.Series, new_correct: pd.Series) -> dict[str, float | int]:
    b = int((~base_correct & new_correct).sum())
    c = int((base_correct & ~new_correct).sum())
    n = b + c
    if n == 0:
        p = 1.0
    else:
        # Two-sided exact binomial under p=0.5.
        from math import comb

        tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
        p = min(1.0, 2 * tail)
    return {"base_wrong_new_correct": b, "base_correct_new_wrong": c, "mcnemar_exact_p": p}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["v111_review_or_control", "v118_review_or_control", "v118_extra_review"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "final_correct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    scopes = [
        ("all_domains", df),
        ("internal_all", df[df["domain"].isin(["old_data", "third_batch"])]),
        ("old_data", df[df["domain"].eq("old_data")]),
        ("third_batch", df[df["domain"].eq("third_batch")]),
        ("strict_external", df[df["domain"].eq("strict_external")]),
    ]
    metric_rows = []
    pair_rows = []
    for scope, sub in scopes:
        base_review = sub["v111_review_or_control"].astype(bool)
        new_review = sub["v118_review_or_control"].astype(bool)
        metric_rows.append(metric_with_ci(sub, base_review, "v111_base", scope))
        metric_rows.append(metric_with_ci(sub, new_review, "v118_global_two_signal", scope))
        base_auto_correct = base_review | sub["final_correct"].eq(1)
        new_auto_correct = new_review | sub["final_correct"].eq(1)
        pair = {
            "scope": scope,
            **paired_bootstrap_delta(sub, base_review, new_review),
            **mcnemar_exact(base_auto_correct, new_auto_correct),
            "extra_review_n": int((new_review & ~base_review).sum()),
            "extra_captured_error_n": int((new_review & ~base_review & sub["final_correct"].eq(0)).sum()),
            "extra_clean_review_n": int((new_review & ~base_review & sub["final_correct"].eq(1)).sum()),
        }
        pair_rows.append(pair)

    metrics_df = pd.DataFrame(metric_rows)
    paired_df = pd.DataFrame(pair_rows)
    metrics_df.to_csv(OUT_DIR / "v123_metric_wilson_ci.csv", index=False, encoding="utf-8-sig")
    format_table(metrics_df).to_csv(OUT_DIR / "v123_metric_wilson_ci_formatted.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(OUT_DIR / "v123_paired_delta_stats.csv", index=False, encoding="utf-8-sig")
    format_table(paired_df).to_csv(OUT_DIR / "v123_paired_delta_stats_formatted.csv", index=False, encoding="utf-8-sig")

    all_pair = paired_df[paired_df["scope"].eq("all_domains")].iloc[0]
    third_pair = paired_df[paired_df["scope"].eq("third_batch")].iloc[0]
    all_metric = metrics_df[(metrics_df["workflow"].eq("v118_global_two_signal")) & (metrics_df["scope"].eq("all_domains"))].iloc[0]
    lines = [
        "# v123 Paired Statistics and Confidence Bounds",
        "",
        "v123 adds same-case paired statistics and Wilson confidence bounds for v111 -> v118.",
        "",
        f"- v118 all-domain BAcc {pct(all_metric['balanced_accuracy'])}, Wilson interval {pct(all_metric['balanced_accuracy_ci_low'])}-{pct(all_metric['balanced_accuracy_ci_high'])}.",
        f"- All-domain paired delta vs v111: {pct(all_pair['delta_bacc'])}, bootstrap CI {pct(all_pair['delta_bacc_ci_low'])} to {pct(all_pair['delta_bacc_ci_high'])}; errors rescued={int(all_pair['base_wrong_new_correct'])}, harmed={int(all_pair['base_correct_new_wrong'])}, McNemar p={all_pair['mcnemar_exact_p']:.4f}.",
        f"- Third-batch paired delta vs v111: {pct(third_pair['delta_bacc'])}, bootstrap CI {pct(third_pair['delta_bacc_ci_low'])} to {pct(third_pair['delta_bacc_ci_high'])}; errors rescued={int(third_pair['base_wrong_new_correct'])}, harmed={int(third_pair['base_correct_new_wrong'])}.",
        "",
        "This should be written as a same-case error reduction and efficiency result, not as a large statistically powered superiority claim.",
    ]
    (OUT_DIR / "v123_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(format_table(metrics_df[metrics_df["scope"].isin(["all_domains", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print(format_table(paired_df).to_string(index=False))


if __name__ == "__main__":
    main()
