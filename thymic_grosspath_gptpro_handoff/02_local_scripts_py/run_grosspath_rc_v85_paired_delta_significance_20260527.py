from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_ROUTES = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527" / "v80_tiered_lowrisk_guard_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v85_paired_delta_significance_20260527"
N_BOOT = 3000
SEED = 20260527


PAIRS = [
    ("v75_vs_v50", "v50_main", "v75_quality_lowconf", "High-risk miss protection vs v50"),
    ("v79_light_vs_v75", "v75_quality_lowconf", "v79_light_lowrisk_guard", "Light low-risk guard vs v75"),
    ("v79_strict_vs_light", "v79_light_lowrisk_guard", "v79_strict_lowrisk_guard", "Strict guard vs light guard"),
    ("v79_light_vs_v50", "v50_main", "v79_light_lowrisk_guard", "Light full workflow vs v50"),
    ("v79_strict_vs_v50", "v50_main", "v79_strict_lowrisk_guard", "Strict full workflow vs v50"),
]


def metric_values(y: np.ndarray, pred: np.ndarray, control: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    control = np.asarray(control, dtype=bool)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    acc = (tp + tn) / len(y) if len(y) else np.nan
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "control_rate": float(control.mean()) if len(control) else np.nan,
        "fn": fn,
        "fp": fp,
        "error_n": int((pred != y).sum()),
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    # Two-sided exact binomial McNemar p-value with p=0.5.
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * prob)


def pair_table(df: pd.DataFrame, before: str, after: str) -> pd.DataFrame:
    key_cols = ["domain", "case_id"]
    before_df = df.loc[df["policy"].eq(before)].copy()
    after_df = df.loc[df["policy"].eq(after)].copy()
    keep = key_cols + ["label_idx", "final_pred", "review_or_control", "final_correct"]
    b = before_df[keep].rename(
        columns={
            "final_pred": "pred_before",
            "review_or_control": "control_before",
            "final_correct": "correct_before",
        }
    )
    a = after_df[keep].rename(
        columns={
            "final_pred": "pred_after",
            "review_or_control": "control_after",
            "final_correct": "correct_after",
        }
    )
    merged = b.merge(a, on=key_cols + ["label_idx"], how="inner", validate="one_to_one")
    if len(merged) != len(before_df):
        raise ValueError(f"Pair merge changed rows for {before}->{after}: {len(before_df)} to {len(merged)}")
    return merged


def delta_metrics(tab: pd.DataFrame) -> dict[str, float | int]:
    y = tab["label_idx"].to_numpy(int)
    mb = metric_values(y, tab["pred_before"].to_numpy(int), tab["control_before"].to_numpy(int))
    ma = metric_values(y, tab["pred_after"].to_numpy(int), tab["control_after"].to_numpy(int))
    row: dict[str, float | int] = {}
    for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "control_rate"]:
        row[f"before_{key}"] = mb[key]
        row[f"after_{key}"] = ma[key]
        row[f"delta_{key}"] = float(ma[key] - mb[key])  # type: ignore[operator]
    for key in ["fn", "fp", "error_n"]:
        row[f"before_{key}"] = mb[key]
        row[f"after_{key}"] = ma[key]
        row[f"delta_{key}"] = int(ma[key] - mb[key])  # type: ignore[operator]
    before_correct = tab["correct_before"].to_numpy(int).astype(bool)
    after_correct = tab["correct_after"].to_numpy(int).astype(bool)
    row["after_correct_before_wrong"] = int((~before_correct & after_correct).sum())
    row["after_wrong_before_correct"] = int((before_correct & ~after_correct).sum())
    row["mcnemar_p"] = exact_mcnemar_p(int(row["after_correct_before_wrong"]), int(row["after_wrong_before_correct"]))
    return row


def bootstrap_delta(tab: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    y = tab["label_idx"].to_numpy(int)
    idx_low = np.where(y == 0)[0]
    idx_high = np.where(y == 1)[0]
    rows = []
    for _ in range(N_BOOT):
        sample = np.concatenate(
            [
                rng.choice(idx_low, size=len(idx_low), replace=True),
                rng.choice(idx_high, size=len(idx_high), replace=True),
            ]
        )
        rows.append(delta_metrics(tab.iloc[sample].reset_index(drop=True)))
    return pd.DataFrame(rows)


def summarize_boot(point: dict[str, float | int], boot: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = dict(point)
    for col in ["delta_accuracy", "delta_balanced_accuracy", "delta_sensitivity", "delta_specificity", "delta_control_rate"]:
        vals = boot[col].dropna().to_numpy(float)
        out[f"{col}_ci025"] = float(np.quantile(vals, 0.025))
        out[f"{col}_ci975"] = float(np.quantile(vals, 0.975))
        out[f"{col}_p_gt0"] = float((vals > 0).mean())
    for col in ["delta_fn", "delta_fp", "delta_error_n"]:
        vals = boot[col].dropna().to_numpy(float)
        out[f"{col}_ci025"] = float(np.quantile(vals, 0.025))
        out[f"{col}_ci975"] = float(np.quantile(vals, 0.975))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CASE_ROUTES)
    rng = np.random.default_rng(SEED)
    summary_rows = []
    boot_rows = []
    pair_rows = []
    for domain in ["old_data", "third_batch", "strict_external"]:
        domain_df = df.loc[df["domain"].eq(domain)].copy()
        for pair_id, before, after, label in PAIRS:
            tab = pair_table(domain_df, before, after)
            point = delta_metrics(tab)
            point.update({"domain": domain, "pair_id": pair_id, "comparison": label, "before_policy": before, "after_policy": after, "n": int(len(tab))})
            boot = bootstrap_delta(tab, rng)
            boot.insert(0, "domain", domain)
            boot.insert(1, "pair_id", pair_id)
            boot_rows.append(boot)
            summary_rows.append(summarize_boot(point, boot))
            changed = tab.loc[tab["correct_before"].ne(tab["correct_after"])].copy()
            changed.insert(1, "pair_id", pair_id)
            pair_rows.append(changed)

    summary = pd.DataFrame(summary_rows)
    boots = pd.concat(boot_rows, ignore_index=True)
    changed_cases = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    summary.to_csv(OUT_DIR / "v85_paired_delta_ci_summary.csv", index=False, encoding="utf-8-sig")
    boots.to_csv(OUT_DIR / "v85_paired_delta_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    changed_cases.to_csv(OUT_DIR / "v85_paired_changed_cases.csv", index=False, encoding="utf-8-sig")

    print("Paired delta CI summary:")
    cols = [
        "domain",
        "pair_id",
        "delta_balanced_accuracy",
        "delta_balanced_accuracy_ci025",
        "delta_balanced_accuracy_ci975",
        "delta_sensitivity",
        "delta_specificity",
        "delta_control_rate",
        "delta_fn",
        "delta_fp",
        "after_correct_before_wrong",
        "after_wrong_before_correct",
        "mcnemar_p",
    ]
    print(summary[cols].to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
