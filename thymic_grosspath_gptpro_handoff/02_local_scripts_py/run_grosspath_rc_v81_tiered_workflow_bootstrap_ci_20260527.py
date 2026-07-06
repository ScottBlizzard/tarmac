from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_CASE = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527" / "v80_tiered_lowrisk_guard_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v81_tiered_workflow_bootstrap_ci_20260527"
N_BOOT = 5000
SEED = 20260527
Z95 = 1.959963984540054


def wilson_ci(success: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = success / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return max(0.0, (centre - radius) / denom), min(1.0, (centre + radius) / denom)


def metrics(y: np.ndarray, pred: np.ndarray, control: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    control = np.asarray(control, dtype=bool)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    accuracy = (tp + tn) / len(y) if len(y) else np.nan
    return {
        "n": int(len(y)),
        "n_low": int((y == 0).sum()),
        "n_high": int((y == 1).sum()),
        "control_rate": float(control.mean()) if len(control) else np.nan,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fn": fn,
        "fp": fp,
        "remaining_error_n": int((pred != y).sum()),
    }


def bootstrap_group(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    y = df["label_idx"].to_numpy(int)
    pred = df["final_pred"].to_numpy(int)
    control = df["review_or_control"].to_numpy(int).astype(bool)
    idx_low = np.where(y == 0)[0]
    idx_high = np.where(y == 1)[0]
    rows = []
    for _ in range(N_BOOT):
        # Stratified bootstrap keeps low/high prevalence stable for BAcc CI.
        sample = np.concatenate(
            [
                rng.choice(idx_low, size=len(idx_low), replace=True),
                rng.choice(idx_high, size=len(idx_high), replace=True),
            ]
        )
        rows.append(metrics(y[sample], pred[sample], control[sample]))
    return pd.DataFrame(rows)


def summarize_ci(point: dict[str, float | int], boot: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = dict(point)
    for col in ["control_rate", "accuracy", "balanced_accuracy", "sensitivity", "specificity"]:
        vals = boot[col].dropna().to_numpy(float)
        row[f"{col}_ci025"] = float(np.quantile(vals, 0.025))
        row[f"{col}_ci975"] = float(np.quantile(vals, 0.975))
    for col in ["fn", "fp", "remaining_error_n"]:
        vals = boot[col].dropna().to_numpy(float)
        row[f"{col}_ci025"] = float(np.quantile(vals, 0.025))
        row[f"{col}_ci975"] = float(np.quantile(vals, 0.975))

    n = int(point["n"])
    n_low = int(point["n_low"])
    n_high = int(point["n_high"])
    tp = n_high - int(point["fn"])
    tn = n_low - int(point["fp"])
    correct = n - int(point["remaining_error_n"])
    control_n = int(round(float(point["control_rate"]) * n))
    acc_l, acc_u = wilson_ci(correct, n)
    sens_l, sens_u = wilson_ci(tp, n_high)
    spec_l, spec_u = wilson_ci(tn, n_low)
    ctrl_l, ctrl_u = wilson_ci(control_n, n)
    row["accuracy_wilson_low"] = acc_l
    row["accuracy_wilson_high"] = acc_u
    row["sensitivity_wilson_low"] = sens_l
    row["sensitivity_wilson_high"] = sens_u
    row["specificity_wilson_low"] = spec_l
    row["specificity_wilson_high"] = spec_u
    row["balanced_accuracy_wilson_low"] = float(np.nanmean([sens_l, spec_l]))
    row["balanced_accuracy_wilson_high"] = float(np.nanmean([sens_u, spec_u]))
    row["control_rate_wilson_low"] = ctrl_l
    row["control_rate_wilson_high"] = ctrl_u
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CASE)
    rng = np.random.default_rng(SEED)
    rows = []
    boot_rows = []
    for (domain, policy), sub in df.groupby(["domain", "policy"], sort=False):
        y = sub["label_idx"].to_numpy(int)
        pred = sub["final_pred"].to_numpy(int)
        control = sub["review_or_control"].to_numpy(int).astype(bool)
        point = metrics(y, pred, control)
        point["domain"] = domain
        point["policy"] = policy
        boot = bootstrap_group(sub, rng)
        boot.insert(0, "domain", domain)
        boot.insert(1, "policy", policy)
        boot_rows.append(boot)
        rows.append(summarize_ci(point, boot))

    ci = pd.DataFrame(rows)
    boot_all = pd.concat(boot_rows, ignore_index=True)
    ci.to_csv(OUT_DIR / "v81_tiered_workflow_bootstrap_ci_summary.csv", index=False, encoding="utf-8-sig")
    boot_all.to_csv(OUT_DIR / "v81_tiered_workflow_bootstrap_samples.csv", index=False, encoding="utf-8-sig")

    print("Bootstrap CI summary:")
    show = ci[
        [
            "domain",
            "policy",
            "control_rate",
            "balanced_accuracy",
            "balanced_accuracy_wilson_low",
            "balanced_accuracy_wilson_high",
            "sensitivity",
            "sensitivity_wilson_low",
            "sensitivity_wilson_high",
            "specificity",
            "specificity_wilson_low",
            "specificity_wilson_high",
            "fn",
            "fp",
        ]
    ].copy()
    print(show.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
