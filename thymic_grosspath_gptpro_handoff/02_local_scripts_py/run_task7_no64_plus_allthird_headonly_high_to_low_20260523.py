from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_task7_no64_plus_allthird_headonly_rescue_20260523 import (  # noqa: E402
    HEADONLY_OOF,
    OUT as LOW_TO_HIGH_OUT,
    add_headonly,
    load_base_frames,
    metrics,
)


BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
OUT = BASE / "task7_adaptation_runs" / "49_no64_selected_plus_allthird_headonly_high_to_low_20260523"


def apply_high_to_low(frame: pd.DataFrame, new_low_threshold: float, base_prob_min: float, gap_min: float):
    y = frame["label_idx"].to_numpy(int)
    base_prob = frame["selected_base_prob"].to_numpy(float)
    base_pred = frame["selected_base_pred"].to_numpy(int)
    new_prob = frame["headonly_prob_high"].to_numpy(float)
    routed = (base_pred == 1) & (new_prob <= new_low_threshold) & (base_prob >= base_prob_min) & ((base_prob - new_prob) >= gap_min)
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    final_pred[routed] = 0
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
    out["high_to_low_routed"] = routed.astype(int)
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

    rows = []
    cache = {}
    for new_low_threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        for base_prob_min in [0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
            for gap_min in [0.00, 0.10, 0.20, 0.30, 0.40]:
                old_row, old_pred, old_prob, old_routed = apply_high_to_low(old, new_low_threshold, base_prob_min, gap_min)
                adapt_row, adapt_pred, adapt_prob, adapt_routed = apply_high_to_low(adapt, new_low_threshold, base_prob_min, gap_min)
                hold_row, hold_pred, hold_prob, hold_routed = apply_high_to_low(hold, new_low_threshold, base_prob_min, gap_min)
                row = {
                    "new_low_threshold": float(new_low_threshold),
                    "base_prob_min": float(base_prob_min),
                    "gap_min": float(gap_min),
                }
                row.update({f"old_{k}": v for k, v in old_row.items()})
                row.update({f"adapt_{k}": v for k, v in adapt_row.items()})
                row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                row["adapt_non_drop"] = bool(
                    row["adapt_accuracy"] >= base_rows["adapt"]["accuracy"]
                    and row["adapt_balanced_accuracy"] >= base_rows["adapt"]["balanced_accuracy"]
                    and row["adapt_tp"] >= base_rows["adapt"]["tp"]
                )
                row["selection_score_safe"] = (
                    float(row["adapt_accuracy"])
                    + 0.8 * float(row["adapt_balanced_accuracy"])
                    + 0.03 * float(row["adapt_net_rescue"])
                    - 0.03 * float(row["old_hurt_n"])
                )
                rows.append(row)
                cache[len(rows) - 1] = (old_pred, old_prob, old_routed, adapt_pred, adapt_prob, adapt_routed, hold_pred, hold_prob, hold_routed)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "headonly_high_to_low_scan.csv", index=False, encoding="utf-8-sig")
    safe = summary[summary["old_guard_092"] & summary["adapt_non_drop"]].copy()
    if safe.empty:
        safe = summary[summary["old_guard_092"]].copy()
    selected_df = safe.sort_values(["selection_score_safe", "adapt_accuracy", "adapt_balanced_accuracy"], ascending=False)
    hold_acc = summary[summary["old_guard_092"]].sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    hold_bacc = summary[summary["old_guard_092"]].sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False)

    selected_df.head(100).to_csv(OUT / "top_selected_by_old_adapt_only.csv", index=False, encoding="utf-8-sig")
    hold_acc.head(100).to_csv(OUT / "top_holdout_acc_audit.csv", index=False, encoding="utf-8-sig")
    hold_bacc.head(100).to_csv(OUT / "top_holdout_bacc_audit.csv", index=False, encoding="utf-8-sig")

    def materialize(row: dict[str, object] | None, prefix: str) -> None:
        if row is None:
            return
        match = summary[
            (summary["new_low_threshold"] == row["new_low_threshold"])
            & (summary["base_prob_min"] == row["base_prob_min"])
            & (summary["gap_min"] == row["gap_min"])
        ]
        idx = int(match.index[0])
        old_pred, old_prob, old_routed, adapt_pred, adapt_prob, adapt_routed, hold_pred, hold_prob, hold_routed = cache[idx]
        save_cases(f"{prefix}_old", old, old_pred, old_prob, old_routed)
        save_cases(f"{prefix}_adapt", adapt, adapt_pred, adapt_prob, adapt_routed)
        save_cases(f"{prefix}_holdout", hold, hold_pred, hold_prob, hold_routed)

    selected_row = None if selected_df.empty else selected_df.iloc[0].to_dict()
    hold_acc_row = None if hold_acc.empty else hold_acc.iloc[0].to_dict()
    hold_bacc_row = None if hold_bacc.empty else hold_bacc.iloc[0].to_dict()
    materialize(selected_row, "selected")
    materialize(hold_acc_row, "holdout_acc_audit")
    materialize(hold_bacc_row, "holdout_bacc_audit")

    report = {
        "protocol": {
            "base": "No.64 + adapt tuned overlay selected by old OOF + adapt72",
            "candidate": "old+third all balanced DINOv2 whole head-only OOF model",
            "allowed_action": "high-to-low correction only",
            "selection_uses_holdout": False,
            "selection_data": "old OOF + adapt72; third_holdout is audit only",
        },
        "base_metrics": base_rows,
        "base_policy": selected,
        "selected_by_old_adapt": selected_row,
        "best_holdout_acc_audit": hold_acc_row,
        "best_holdout_bacc_audit": hold_bacc_row,
        "n_policies": int(len(summary)),
        "n_safe_old_adapt": int(len(summary[summary["old_guard_092"] & summary["adapt_non_drop"]])),
        "low_to_high_output_dir": str(LOW_TO_HIGH_OUT),
    }
    (OUT / "headonly_high_to_low_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
