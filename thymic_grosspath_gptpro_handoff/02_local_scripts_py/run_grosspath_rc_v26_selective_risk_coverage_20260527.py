from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
V25_DIR = ROOT / "outputs" / "grosspath_rc_v25_auto_quality_gate_20260527"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v26_selective_risk_coverage_20260527"

MAIN_THRESHOLD = 0.595
ROBUST_THRESHOLD = 0.57


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv")
    qpred = pd.read_csv(V25_DIR / "v25_quality_gate_case_predictions.csv")
    risk_cols = [col for col in qpred.columns if col.endswith("_risk_oof")]
    df = df.merge(qpred[["case_id", "original_case_id", *risk_cols]], on=["case_id", "original_case_id"], how="left")
    df["main_pred"] = (df["prob_base162"].astype(float) >= MAIN_THRESHOLD).astype(int)
    df["robust_prob"] = df[["prob_base162", "prob103_vitl", "prob_mean_core"]].mean(axis=1).astype(float)
    df["robust_pred"] = (df["robust_prob"] >= ROBUST_THRESHOLD).astype(int)
    df["safety_trigger"] = ((df["main_pred"] == 0) & (df["robust_pred"] == 1)).astype(int)
    df["safety_risk"] = df["safety_trigger"].astype(float)
    df["main_uncertainty"] = 1.0 - (df["prob_base162"].astype(float) - MAIN_THRESHOLD).abs().clip(0, 1)
    df["model_disagreement"] = (df["robust_prob"] - df["prob_base162"].astype(float)).abs().clip(0, 1)
    df["heuristic_quality_risk"] = (pd.to_numeric(df["quality_score"], errors="coerce").fillna(0).rsub(100) / 100).clip(0, 1)

    risk_specs = {
        "main_uncertainty_only": "main_uncertainty",
        "heuristic_quality_score": "heuristic_quality_risk",
        "quality_rf_oof": "quality_rf_risk_oof",
        "quality_extra_trees_oof": "quality_extra_trees_risk_oof",
        "quality_logistic_oof": "quality_logistic_risk_oof",
    }
    rows: list[dict[str, object]] = []
    case_rows: list[pd.DataFrame] = []
    y = df["label_idx"].to_numpy(dtype=int)
    main = df["main_pred"].to_numpy(dtype=int)
    safety_switched = np.where(df["safety_trigger"].to_numpy(dtype=bool), df["robust_pred"].to_numpy(dtype=int), main)

    for risk_name, risk_col in risk_specs.items():
        if risk_col not in df.columns:
            continue
        risk = pd.to_numeric(df[risk_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        combined_risks = {
            risk_name: risk,
            f"{risk_name}+safety_binary": np.maximum(risk, df["safety_risk"].to_numpy(dtype=float)),
            f"{risk_name}+safety_soft": np.maximum(risk, 0.75 * df["safety_risk"].to_numpy(dtype=float)),
        }
        for combined_name, combined_risk in combined_risks.items():
            for coverage in np.arange(0.20, 1.001, 0.05):
                n_pass = max(1, int(round(len(df) * float(coverage))))
                order = np.argsort(combined_risk)
                pass_idx = order[:n_pass]
                review_idx = order[n_pass:]
                for pred_name, pred in [("main", main), ("safety_switch", safety_switched)]:
                    m = metrics(y[pass_idx], pred[pass_idx])
                    row = {
                        "risk_policy": combined_name,
                        "prediction_policy": pred_name,
                        "target_coverage": float(coverage),
                        "pass_n": int(len(pass_idx)),
                        "pass_rate": len(pass_idx) / len(df),
                        "review_n": int(len(review_idx)),
                        "review_rate": len(review_idx) / len(df),
                    }
                    row.update({f"pass_{k}": v for k, v in m.items()})
                    rows.append(row)
            # Export three practically useful cutoffs.
            for coverage in [0.40, 0.50, 0.60]:
                n_pass = max(1, int(round(len(df) * float(coverage))))
                order = np.argsort(combined_risk)
                pass_mask = np.zeros(len(df), dtype=bool)
                pass_mask[order[:n_pass]] = True
                tmp = df[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "quality_group", "quality_score", "prob_base162", "robust_prob", "main_pred", "robust_pred", "safety_trigger"]].copy()
                tmp["risk_policy"] = combined_name
                tmp["coverage"] = coverage
                tmp["risk_score"] = combined_risk
                tmp["route"] = np.where(pass_mask, "auto_pass", "review")
                tmp["main_correct"] = (tmp["main_pred"] == tmp["label_idx"]).astype(int)
                tmp["safety_switch_pred"] = safety_switched
                tmp["safety_switch_correct"] = (tmp["safety_switch_pred"] == tmp["label_idx"]).astype(int)
                case_rows.append(tmp)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v26_selective_risk_coverage_metrics.csv", index=False, encoding="utf-8-sig")
    if case_rows:
        pd.concat(case_rows, ignore_index=True).to_csv(OUT_DIR / "v26_selective_case_routes.csv", index=False, encoding="utf-8-sig")

    print(f"[done] {OUT_DIR}")
    for min_acc in [0.80, 0.85, 0.90]:
        feasible = out[(out["pass_accuracy"] >= min_acc)].sort_values(["pass_rate", "pass_balanced_accuracy"], ascending=False).head(12)
        print(f"\nFeasible pass accuracy >= {min_acc}:")
        print(feasible[["risk_policy", "prediction_policy", "pass_rate", "review_rate", "pass_accuracy", "pass_balanced_accuracy", "pass_sensitivity", "pass_specificity", "pass_fn", "pass_fp"]].to_string(index=False))


if __name__ == "__main__":
    main()
