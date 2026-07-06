from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_task7_no64_guarded_adapt_overlay_20260522 import (  # noqa: E402
    add_route_scores,
    apply_overlay,
    reconstruct_no64_old,
)
from run_task7_no64_guarded_adapt_tuned_overlay_20260522 import split_candidate_train  # noqa: E402


BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
RUN64 = BASE / "task7_gross_feature_runs" / "64_image_only_hardcore_reviewer_20260521"
ADAPT_CACHE = BASE / "task7_adaptation_runs" / "11_unified_two_stage_adapt72_20260521"
THIRD_EXTERNAL = BASE / "task7_external_runs" / "04_third_batch_whole_plus_crop_64style_20260521"
SELECTED_REPORT = BASE / "task7_adaptation_runs" / "25_no64_adapt_tuned_overlay_20260522" / "adapt_tuned_overlay_report.json"
HEADONLY_OOF = (
    BASE
    / "task7_adaptation_runs"
    / "47_old_third_all_balanced_whole_headonly_summary_20260523"
    / "mean"
    / "crop_finetune_full_oof_case_predictions.csv"
)
OUT = BASE / "task7_adaptation_runs" / "48_no64_selected_plus_allthird_headonly_rescue_20260523"


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def load_base_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    report = json.loads(SELECTED_REPORT.read_text(encoding="utf-8"))
    selected = report["selected_by_old_plus_adapt"]
    old_no64 = reconstruct_no64_old(ROOT, RUN64, None)
    train_probs = pd.read_csv(ADAPT_CACHE / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    hold_probs = pd.read_csv(ADAPT_CACHE / "candidate_holdout_probs.csv", dtype={"case_id": str})
    old_prob_rows, adapt_prob_rows = split_candidate_train(train_probs)
    third_base_all = pd.read_csv(THIRD_EXTERNAL / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})

    old = old_no64.merge(old_prob_rows, on="case_id", how="inner")
    old["base_pred_for_overlay"] = old["no64_final_pred_idx"].astype(int)
    adapt = third_base_all.merge(adapt_prob_rows, on="case_id", how="inner")
    adapt["base_pred_for_overlay"] = adapt["final_pred_idx"].astype(int)
    hold = third_base_all.merge(hold_probs, on="case_id", how="inner")
    hold["base_pred_for_overlay"] = hold["final_pred_idx"].astype(int)

    candidate_cols = [c for c in train_probs.columns if c != "case_id"]
    adapt_name = str(selected["adapt_candidate"])
    adapt_t = float(selected["adapt_threshold"])
    route_name = str(selected["route_name"])
    route_threshold = float(selected["route_threshold"])

    def apply_selected(frame: pd.DataFrame, y_col: str, base_prob_col: str, base_pred_col: str) -> pd.DataFrame:
        y = frame[y_col].to_numpy(int)
        base_prob = frame[base_prob_col].to_numpy(float)
        base_pred = frame[base_pred_col].to_numpy(int)
        route = add_route_scores(frame, candidate_cols, base_prob_col, adapt_name, adapt_t)[route_name]
        row, final_pred, final_prob, routed = apply_overlay(
            y,
            base_prob,
            base_pred,
            frame[adapt_name].to_numpy(float),
            adapt_t,
            route,
            route_threshold,
        )
        out = frame.copy()
        out["selected_base_prob"] = final_prob
        out["selected_base_pred"] = final_pred
        out["selected_base_routed"] = routed.astype(int)
        out["selected_base_correct"] = (final_pred == y).astype(int)
        out.attrs["metrics"] = row
        return out

    old = apply_selected(old, "label_idx", "no64_final_prob_high", "no64_final_pred_idx")
    adapt = apply_selected(adapt, "label_idx", "final_prob_high", "final_pred_idx")
    hold = apply_selected(hold, "label_idx", "final_prob_high", "final_pred_idx")
    return old, adapt, hold, selected


def add_headonly(frame: pd.DataFrame, group: str, head: pd.DataFrame) -> pd.DataFrame:
    h = head[head["eval_group"].eq(group)].copy()
    h["case_id"] = h["case_id_examples"].astype(str)
    h = h[["case_id", "prob_high", "pred_idx_050"]].rename(
        columns={"prob_high": "headonly_prob_high", "pred_idx_050": "headonly_pred_idx_050"}
    )
    out = frame.merge(h, on="case_id", how="left")
    if out["headonly_prob_high"].isna().any():
        missing = out.loc[out["headonly_prob_high"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing head-only predictions for {group}: {missing}")
    return out


def apply_low_to_high_rescue(frame: pd.DataFrame, new_threshold: float, base_prob_max: float, gap_min: float) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    y = frame["label_idx"].to_numpy(int)
    base_prob = frame["selected_base_prob"].to_numpy(float)
    base_pred = frame["selected_base_pred"].to_numpy(int)
    new_prob = frame["headonly_prob_high"].to_numpy(float)
    routed = (base_pred == 0) & (new_prob >= new_threshold) & (base_prob <= base_prob_max) & ((new_prob - base_prob) >= gap_min)
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    final_pred[routed] = 1
    final_prob[routed] = new_prob[routed]
    row = metrics(y, final_pred, final_prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "rescue_n": int(((base_pred != y) & (final_pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, final_pred, final_prob, routed


def save_cases(prefix: str, frame: pd.DataFrame, final_pred: np.ndarray, final_prob: np.ndarray, routed: np.ndarray) -> None:
    keep = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "selected_base_prob",
        "selected_base_pred",
        "headonly_prob_high",
    ]
    cols = [c for c in keep if c in frame.columns]
    out = frame[cols].copy()
    out["rescue_routed"] = routed.astype(int)
    out["final_prob_high"] = final_prob
    out["final_pred_idx"] = final_pred
    out["final_correct"] = (final_pred == frame["label_idx"].to_numpy(int)).astype(int)
    out.to_csv(OUT / f"{prefix}_case_predictions.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old, adapt, hold, selected = load_base_frames()
    head = pd.read_csv(HEADONLY_OOF, dtype={"case_id_examples": str, "original_case_id": str})
    old = add_headonly(old, "old", head)
    adapt = add_headonly(adapt, "adapt72", head)
    hold = add_headonly(hold, "third_holdout", head)

    base_rows = {
        "old": metrics(old["label_idx"].to_numpy(int), old["selected_base_pred"].to_numpy(int), old["selected_base_prob"].to_numpy(float)),
        "adapt": metrics(adapt["label_idx"].to_numpy(int), adapt["selected_base_pred"].to_numpy(int), adapt["selected_base_prob"].to_numpy(float)),
        "holdout": metrics(hold["label_idx"].to_numpy(int), hold["selected_base_pred"].to_numpy(int), hold["selected_base_prob"].to_numpy(float)),
    }

    rows: list[dict[str, object]] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for new_threshold in np.round(np.arange(0.50, 0.951, 0.03), 2):
        for base_prob_max in [0.30, 0.40, 0.50, 0.60]:
            for gap_min in [0.00, 0.10, 0.20, 0.30, 0.40]:
                old_row, old_pred, old_prob, old_routed = apply_low_to_high_rescue(old, float(new_threshold), float(base_prob_max), float(gap_min))
                adapt_row, adapt_pred, adapt_prob, adapt_routed = apply_low_to_high_rescue(adapt, float(new_threshold), float(base_prob_max), float(gap_min))
                hold_row, hold_pred, hold_prob, hold_routed = apply_low_to_high_rescue(hold, float(new_threshold), float(base_prob_max), float(gap_min))
                row: dict[str, object] = {
                    "new_threshold": float(new_threshold),
                    "base_prob_max": float(base_prob_max),
                    "gap_min": float(gap_min),
                }
                row.update({f"old_{k}": v for k, v in old_row.items()})
                row.update({f"adapt_{k}": v for k, v in adapt_row.items()})
                row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                row["old_non_drop"] = bool(row["old_accuracy"] >= base_rows["old"]["accuracy"] and row["old_balanced_accuracy"] >= base_rows["old"]["balanced_accuracy"])
                row["adapt_non_drop"] = bool(
                    row["adapt_accuracy"] >= base_rows["adapt"]["accuracy"]
                    and row["adapt_balanced_accuracy"] >= base_rows["adapt"]["balanced_accuracy"]
                    and row["adapt_tp"] >= base_rows["adapt"]["tp"]
                )
                row["selection_score_safe"] = (
                    float(row["adapt_balanced_accuracy"])
                    + 0.5 * float(row["adapt_accuracy"])
                    + 0.03 * float(row["adapt_net_rescue"])
                    - 0.03 * float(row["old_hurt_n"])
                )
                rows.append(row)
                cache[len(rows) - 1] = (old_pred, old_prob, old_routed, adapt_pred, adapt_prob, adapt_routed, hold_pred, hold_prob, hold_routed)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "headonly_low_to_high_rescue_scan.csv", index=False, encoding="utf-8-sig")
    safe = summary[summary["old_guard_092"] & summary["adapt_non_drop"]].copy()
    if safe.empty:
        safe = summary[summary["old_guard_092"]].copy()
    selected_df = safe.sort_values(["selection_score_safe", "adapt_balanced_accuracy", "adapt_accuracy"], ascending=False)
    hold_ref = summary[summary["old_guard_092"]].sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False)
    hold_acc_ref = summary[summary["old_guard_092"]].sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    selected_df.head(100).to_csv(OUT / "top_selected_by_old_adapt_only.csv", index=False, encoding="utf-8-sig")
    hold_ref.head(100).to_csv(OUT / "top_holdout_bacc_audit.csv", index=False, encoding="utf-8-sig")
    hold_acc_ref.head(100).to_csv(OUT / "top_holdout_acc_audit.csv", index=False, encoding="utf-8-sig")

    selected_row = None if selected_df.empty else selected_df.iloc[0].to_dict()
    hold_bacc_row = None if hold_ref.empty else hold_ref.iloc[0].to_dict()
    hold_acc_row = None if hold_acc_ref.empty else hold_acc_ref.iloc[0].to_dict()

    def materialize(row: dict[str, object] | None, prefix: str) -> None:
        if row is None:
            return
        match = summary[
            (summary["new_threshold"] == row["new_threshold"])
            & (summary["base_prob_max"] == row["base_prob_max"])
            & (summary["gap_min"] == row["gap_min"])
        ]
        idx = int(match.index[0])
        old_pred, old_prob, old_routed, adapt_pred, adapt_prob, adapt_routed, hold_pred, hold_prob, hold_routed = cache[idx]
        save_cases(f"{prefix}_old", old, old_pred, old_prob, old_routed)
        save_cases(f"{prefix}_adapt", adapt, adapt_pred, adapt_prob, adapt_routed)
        save_cases(f"{prefix}_holdout", hold, hold_pred, hold_prob, hold_routed)

    materialize(selected_row, "selected")
    materialize(hold_bacc_row, "holdout_bacc_audit")
    materialize(hold_acc_row, "holdout_acc_audit")

    report = {
        "protocol": {
            "base": "No.64 + adapt tuned overlay selected by old OOF + adapt72",
            "candidate": "old+third all balanced DINOv2 whole head-only OOF model",
            "allowed_action": "low-to-high rescue only",
            "selection_uses_holdout": False,
            "selection_data": "old OOF + adapt72; third_holdout is audit only",
        },
        "base_metrics": base_rows,
        "base_policy": selected,
        "selected_by_old_adapt": selected_row,
        "best_holdout_bacc_audit": hold_bacc_row,
        "best_holdout_acc_audit": hold_acc_row,
        "n_policies": int(len(summary)),
        "n_safe_old_adapt": int(len(summary[summary["old_guard_092"] & summary["adapt_non_drop"]])),
    }
    (OUT / "headonly_low_to_high_rescue_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
