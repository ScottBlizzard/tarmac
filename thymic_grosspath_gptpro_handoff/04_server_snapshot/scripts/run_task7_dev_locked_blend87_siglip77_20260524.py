from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
ADAPT = BASE / "task7_adaptation_runs"
EXT = BASE / "task7_external_runs"

OOF87 = ADAPT / "87_old_third_no64_meta_stack_plus_dinov3vitl_tta86_20260524" / "selected_guard92_oof_predictions.csv"
OOF77 = ADAPT / "77_old_third_siglip_vitl384_wpc_feature_cv_20260524" / "selected_dinov3_feature_oof_predictions.csv"
EXT89 = EXT / "89_locked87_tta_external_audit_20260524" / "external_locked_extra_predictions.csv"
EXT79 = EXT / "79_locked67_siglipL_devfusion_external_audit_20260524" / "external_predictions_with_siglipL77_and_dev_fusions.csv"
OUT = EXT / "90_locked87_tta_plus_siglipL77_devblend_external_audit_20260524"


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


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, dict[str, Any]]:
    best_key: tuple[float, ...] | None = None
    best_t = 0.5
    best_row: dict[str, Any] | None = None
    for threshold in np.linspace(0.05, 0.95, 181):
        row = metric_dict(y, prob, float(threshold))
        if objective == "accuracy":
            key = (row["accuracy"], row["balanced_accuracy"], row["f1"], -abs(float(threshold) - 0.5))
        elif objective == "high_sensitivity":
            key = (min(row["sensitivity_high"], 0.80), row["balanced_accuracy"], row["accuracy"], -abs(float(threshold) - 0.5))
        elif objective == "old_third_balanced":
            key = (row["balanced_accuracy"], row["accuracy"], row["f1"], -abs(float(threshold) - 0.5))
        else:
            raise ValueError(objective)
        if best_key is None or key > best_key:
            best_key = key
            best_t = float(threshold)
            best_row = row
    assert best_row is not None
    return best_t, best_row


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
        # Keep the strong TTA meta model as anchor; only move upward when SigLIP is clearly more high-risk.
        return np.clip(p87 + weight * np.maximum(0.0, p77 - p87), 0.0, 1.0)
    if method == "siglip_dampen":
        # Complementary conservative variant: only move downward when SigLIP is clearly more low-risk.
        return np.clip(p87 - weight * np.maximum(0.0, p87 - p77), 0.0, 1.0)
    raise ValueError(method)


def evaluate_dev(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y = frame["label_idx"].astype(int).to_numpy()
    row = {f"all_{k}": v for k, v in metric_dict(y, prob, threshold).items()}
    for domain in ["old", "third"]:
        mask = frame["domain"].eq(domain).to_numpy()
        metrics = metric_dict(y[mask], prob[mask], threshold)
        row.update({f"{domain}_{k}": v for k, v in metrics.items()})
    return row


def selection_score(row: dict[str, Any]) -> float:
    return (
        0.36 * float(row["old_balanced_accuracy"])
        + 0.34 * float(row["third_balanced_accuracy"])
        + 0.15 * float(row["old_accuracy"])
        + 0.15 * float(row["third_accuracy"])
        - 1.5 * max(0.0, 0.92 - float(row["old_accuracy"]))
        - 1.5 * max(0.0, 0.92 - float(row["old_balanced_accuracy"]))
    )


def load_dev() -> pd.DataFrame:
    p87 = pd.read_csv(OOF87, dtype={"case_id": str})
    p77 = pd.read_csv(OOF77, dtype={"case_id": str})
    frame = p87[
        ["case_id", "original_case_id", "domain", "third_split", "fold_id", "task_l6_label", "task_l7_label", "label_idx"]
    ].merge(
        p77[["case_id", "oof_prob_high"]].rename(columns={"oof_prob_high": "siglipL77_prob"}),
        on="case_id",
        how="left",
    )
    frame["tta87_prob"] = p87["oof_prob_high"].astype(float).to_numpy()
    if frame["siglipL77_prob"].isna().any():
        missing = frame.loc[frame["siglipL77_prob"].isna(), "case_id"].head(20).tolist()
        raise KeyError(f"Missing SigLIP OOF probabilities: {missing}")
    return frame


def load_external() -> pd.DataFrame:
    p89 = pd.read_csv(EXT89, dtype={"case_id": str})
    p79 = pd.read_csv(EXT79, dtype={"case_id": str})
    frame = p89.merge(p79[["case_id", "siglipL77_prob_high"]], on="case_id", how="left")
    frame["tta87_prob"] = frame["locked_extra_prob_high"].astype(float)
    frame["siglipL77_prob"] = frame["siglipL77_prob_high"].astype(float)
    if frame["siglipL77_prob"].isna().any():
        missing = frame.loc[frame["siglipL77_prob"].isna(), "case_id"].head(20).tolist()
        raise KeyError(f"Missing SigLIP external probabilities: {missing}")
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_dev()
    y = dev["label_idx"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    candidates = [("tta87_anchor", "prob_avg", 0.0)]
    for method in ["prob_avg", "logit_avg", "siglip_rescue", "siglip_dampen"]:
        for weight in [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
            candidates.append((f"{method}_w{weight:g}", method, weight))

    for name, method, weight in candidates:
        prob = blend_prob(dev["tta87_prob"].to_numpy(float), dev["siglipL77_prob"].to_numpy(float), method, weight)
        for objective in ["old_third_balanced", "accuracy", "high_sensitivity"]:
            threshold, _ = choose_threshold(y, prob, objective)
            row = {
                "method_name": name,
                "blend_method": method,
                "siglip_weight": float(weight),
                "objective": objective,
            }
            row.update(evaluate_dev(dev, prob, threshold))
            row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
            row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
            row["selection_score"] = selection_score(row)
            rows.append(row)
            pred = dev.copy()
            pred["method_name"] = name
            pred["objective"] = objective
            pred["blend_prob_high"] = prob
            pred["threshold"] = threshold
            pred["blend_pred_idx"] = (prob >= threshold).astype(int)
            pred_frames.append(pred)

    summary = pd.DataFrame(rows).sort_values(
        ["old_guard_092", "selection_score", "third_balanced_accuracy", "third_accuracy"],
        ascending=False,
    )
    summary.to_csv(OUT / "dev_blend_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_frames, ignore_index=True).to_csv(OUT / "dev_blend_all_oof_predictions.csv", index=False, encoding="utf-8-sig")

    selected = summary.iloc[0].to_dict()
    ext = load_external()
    ext_prob = blend_prob(
        ext["tta87_prob"].to_numpy(float),
        ext["siglipL77_prob"].to_numpy(float),
        str(selected["blend_method"]),
        float(selected["siglip_weight"]),
    )
    threshold = float(selected["all_threshold"])
    ext["locked_blend_prob_high"] = ext_prob
    ext["locked_blend_threshold"] = threshold
    ext["locked_blend_pred_idx"] = (ext_prob >= threshold).astype(int)
    ext["locked_blend_correct"] = (ext["locked_blend_pred_idx"].astype(int) == ext["label_idx"].astype(int)).astype(int)

    ext_rows: list[dict[str, Any]] = []
    for subset, mask in [
        ("all", np.ones(len(ext), dtype=bool)),
        ("strict", ext["strict_task7_eval"].astype(int).to_numpy() == 1),
    ]:
        group = ext.loc[mask]
        row = {
            "subset": subset,
            "method_name": selected["method_name"],
            "blend_method": selected["blend_method"],
            "siglip_weight": float(selected["siglip_weight"]),
            "objective": selected["objective"],
        }
        row.update(metric_dict(group["label_idx"].astype(int).to_numpy(), group["locked_blend_prob_high"].astype(float).to_numpy(), threshold))
        ext_rows.append(row)

    ext.to_csv(OUT / "external_locked_blend_predictions.csv", index=False, encoding="utf-8-sig")
    external_summary = pd.DataFrame(ext_rows)
    external_summary.to_csv(OUT / "external_locked_blend_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "locked_blend_report.json").write_text(
        json.dumps(
            {
                "boundary": "Blend method, weight and threshold selected only on old+third development OOF; strict external folder used once for locked audit.",
                "dev_sources": {"tta87_oof": str(OOF87), "siglipL77_oof": str(OOF77)},
                "external_sources": {"tta87_external": str(EXT89), "siglipL77_external": str(EXT79)},
                "selected": selected,
                "external_summary": ext_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[selected_dev]")
    print(pd.DataFrame([selected]).to_string(index=False))
    print("[external_locked]")
    print(external_summary.to_string(index=False))
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
