from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thymic_baseline.config import get_task
from thymic_baseline.metrics import summarize_prediction_frame


def metrics_row(name: str, frame: pd.DataFrame, class_names: tuple[str, ...]) -> dict[str, object]:
    row: dict[str, object] = {"group": name, "n": int(len(frame))}
    if frame.empty:
        return row
    row.update(summarize_prediction_frame(frame, class_names))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Task7 external TTA prediction CSV.")
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task", default="task7_lowrisk_vs_highrisk_tc")
    args = parser.parse_args()

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.output_dir) if args.output_dir else pred_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    task = get_task(args.task)
    pred = pd.read_csv(pred_csv)
    required = {"label_idx", "pred_idx", "prob_low_risk_group", "prob_high_risk_group"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Missing required columns in {pred_csv}: {missing}")

    rows = [metrics_row("overall", pred, task.class_names)]
    if "task_l6_label" in pred.columns:
        for subtype, group in pred.groupby("task_l6_label", dropna=False, sort=True):
            rows.append(metrics_row(f"subtype:{subtype}", group, task.class_names))
    if "domain" in pred.columns:
        for domain, group in pred.groupby("domain", dropna=False, sort=True):
            rows.append(metrics_row(f"domain:{domain}", group, task.class_names))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "external_tta_metrics_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "pred_csv": str(pred_csv),
        "n": int(len(pred)),
        "overall": rows[0],
    }
    (out_dir / "external_tta_metrics_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
