from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V37_DIR = ROOT / "outputs" / "grosspath_rc_v37_rank_normalized_risk_20260527"
V27_DIR = ROOT / "outputs" / "grosspath_rc_v27_unified_workflow_20260527"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v39_external_bootstrap_ci_20260527"
N_BOOT = 5000
SEED = 20260527


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "accuracy": float((tp + tn) / len(y)),
        "balanced_accuracy": float((sens + spec) / 2),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tp": tp,
    }


def summarize_bootstrap(name: str, y: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> dict[str, float | int | str]:
    point = metrics(y, pred)
    values = {k: [] for k in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp"]}
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        m = metrics(y[idx], pred[idx])
        for k in values:
            values[k].append(m[k])
    row: dict[str, float | int | str] = {"policy": name, "n": n, **point}
    for k, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        row[f"{k}_ci025"] = float(np.nanpercentile(arr, 2.5))
        row[f"{k}_ci975"] = float(np.nanpercentile(arr, 97.5))
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cases = pd.read_csv(V37_DIR / "v37_rank_controller_case_routes_external.csv")
    rows = []

    # P2 baseline from the target 0.95 slice to avoid duplicate rows.
    base = cases.loc[cases["target_dev_bacc"].eq(0.95)].copy()
    y = base["label_idx"].to_numpy(dtype=int)
    p2 = base["p2_pred"].to_numpy(dtype=int)
    rows.append(summarize_bootstrap("P2_auto_baseline", y, p2, rng))

    for target in [0.95, 0.97]:
        sub = cases.loc[cases["target_dev_bacc"].eq(target)].copy()
        rows.append(
            summarize_bootstrap(
                f"v37_rank_controller_target_{int(target * 100)}",
                sub["label_idx"].to_numpy(dtype=int),
                sub["final_pred_oracle_review"].to_numpy(dtype=int),
                rng,
            )
        )

    # Include P7 from unified workflow as a previous high-safety comparator.
    v27 = pd.read_csv(V27_DIR / "v27_unified_case_routes_external.csv")
    p7 = v27.loc[v27["policy"].eq("P7_logistic_recall90_quality_or_safety_review")].copy()
    rows.append(
        summarize_bootstrap(
            "P7_quality_safety_review",
            p7["label_idx"].to_numpy(dtype=int),
            p7["workflow_final_pred"].to_numpy(dtype=int),
            rng,
        )
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v39_external_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    print(
        out[
            [
                "policy",
                "accuracy",
                "accuracy_ci025",
                "accuracy_ci975",
                "balanced_accuracy",
                "balanced_accuracy_ci025",
                "balanced_accuracy_ci975",
                "fn",
                "fn_ci025",
                "fn_ci975",
                "fp",
                "fp_ci025",
                "fp_ci975",
            ]
        ].to_string(index=False)
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
