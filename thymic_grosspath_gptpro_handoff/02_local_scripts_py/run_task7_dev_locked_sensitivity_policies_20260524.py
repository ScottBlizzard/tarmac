from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
EXT = BASE / "task7_external_runs"
DEV_SUMMARY = EXT / "90_locked87_tta_plus_siglipL77_devblend_external_audit_20260524" / "dev_blend_summary.csv"
EXT89 = EXT / "89_locked87_tta_external_audit_20260524" / "external_locked_extra_predictions.csv"
EXT79 = EXT / "79_locked67_siglipL_devfusion_external_audit_20260524" / "external_predictions_with_siglipL77_and_dev_fusions.csv"
OUT = EXT / "91_locked87_siglip77_sensitivity_policy_audit_20260524"


def metric_dict(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity_high": float(tp / max(tp + fn, 1)),
        "specificity_low": float(tn / max(tn + fp, 1)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def blend_prob(p87: np.ndarray, p77: np.ndarray, method: str, weight: float) -> np.ndarray:
    if method == "prob_avg":
        return (1.0 - weight) * p87 + weight * p77
    if method == "logit_avg":
        return sigmoid((1.0 - weight) * logit(p87) + weight * logit(p77))
    if method == "siglip_rescue":
        return np.clip(p87 + weight * np.maximum(0.0, p77 - p87), 0.0, 1.0)
    if method == "siglip_dampen":
        return np.clip(p87 - weight * np.maximum(0.0, p87 - p77), 0.0, 1.0)
    raise ValueError(method)


def load_external() -> pd.DataFrame:
    p89 = pd.read_csv(EXT89, dtype={"case_id": str})
    p79 = pd.read_csv(EXT79, dtype={"case_id": str})
    frame = p89.merge(p79[["case_id", "siglipL77_prob_high"]], on="case_id", how="left")
    frame["tta87_prob"] = frame["locked_extra_prob_high"].astype(float)
    frame["siglipL77_prob"] = frame["siglipL77_prob_high"].astype(float)
    return frame


def policy_rows(summary: pd.DataFrame) -> dict[str, pd.Series]:
    summary = summary.copy()
    summary["sens_policy_score"] = (
        0.26 * summary["old_balanced_accuracy"].astype(float)
        + 0.22 * summary["third_balanced_accuracy"].astype(float)
        + 0.20 * summary["third_sensitivity_high"].clip(upper=0.80).astype(float)
        + 0.12 * summary["old_sensitivity_high"].clip(upper=0.93).astype(float)
        + 0.10 * summary["all_balanced_accuracy"].astype(float)
        + 0.10 * summary["all_accuracy"].astype(float)
        - 1.5 * (0.90 - summary["old_accuracy"].astype(float)).clip(lower=0.0)
        - 1.5 * (0.90 - summary["old_balanced_accuracy"].astype(float)).clip(lower=0.0)
    )
    policies: dict[str, pd.Series] = {}
    for name, guard_col in [("sens_guard92", "old_guard_092"), ("sens_guard90", "old_guard_090")]:
        pool = summary[summary[guard_col].astype(bool)]
        if pool.empty:
            pool = summary
        policies[name] = pool.sort_values(
            ["sens_policy_score", "third_sensitivity_high", "third_balanced_accuracy", "all_accuracy"],
            ascending=False,
        ).iloc[0]
    pool92 = summary[summary["old_guard_092"].astype(bool)]
    if pool92.empty:
        pool92 = summary
    policies["third_sens_guard92"] = pool92.sort_values(
        ["third_sensitivity_high", "third_balanced_accuracy", "selection_score", "all_accuracy"],
        ascending=False,
    ).iloc[0]
    policies["balanced_guard92_reference"] = pool92.sort_values(
        ["selection_score", "third_balanced_accuracy", "third_accuracy"],
        ascending=False,
    ).iloc[0]
    return policies


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(DEV_SUMMARY)
    policies = policy_rows(summary)
    ext = load_external()
    rows: list[dict[str, Any]] = []
    pred = ext.copy()
    for policy_name, selected in policies.items():
        prob = blend_prob(
            ext["tta87_prob"].to_numpy(float),
            ext["siglipL77_prob"].to_numpy(float),
            str(selected["blend_method"]),
            float(selected["siglip_weight"]),
        )
        threshold = float(selected["all_threshold"])
        pred[f"{policy_name}_prob_high"] = prob
        pred[f"{policy_name}_pred_idx"] = (prob >= threshold).astype(int)
        for subset, mask in [
            ("all", np.ones(len(ext), dtype=bool)),
            ("strict", ext["strict_task7_eval"].astype(int).to_numpy() == 1),
        ]:
            group = ext.loc[mask]
            row = {
                "policy": policy_name,
                "subset": subset,
                "method_name": selected["method_name"],
                "blend_method": selected["blend_method"],
                "siglip_weight": float(selected["siglip_weight"]),
                "objective": selected["objective"],
                "dev_all_accuracy": float(selected["all_accuracy"]),
                "dev_old_accuracy": float(selected["old_accuracy"]),
                "dev_third_accuracy": float(selected["third_accuracy"]),
                "dev_third_sensitivity_high": float(selected["third_sensitivity_high"]),
            }
            row.update(metric_dict(group["label_idx"].astype(int).to_numpy(), prob[mask], threshold))
            rows.append(row)
    policy_summary = pd.DataFrame(rows)
    policy_summary.to_csv(OUT / "external_sensitivity_policy_summary.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT / "external_sensitivity_policy_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "sensitivity_policy_report.json").write_text(
        json.dumps(
            {
                "boundary": "Policies selected from old+third development blend summary only; strict external folder used for locked audit.",
                "policies": {k: v.to_dict() for k, v in policies.items()},
                "external_summary": policy_summary.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(policy_summary.to_string(index=False))
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
