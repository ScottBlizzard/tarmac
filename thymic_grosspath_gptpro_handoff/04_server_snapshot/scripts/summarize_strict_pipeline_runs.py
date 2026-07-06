from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_ORDER = (
    "accuracy",
    "balanced_accuracy",
    "auc",
    "f1",
    "macro_f1",
    "macro_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strict SuRImage pipeline metric files.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing oof_metrics.csv.")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def summarize_one(run_dir: Path) -> list[dict[str, object]]:
    metrics_path = run_dir / "oof_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    frame = frame[(frame["split"].isin(["test_oof", "test"])) & (frame["aggregation"] == "case_mean")].copy()
    rows: list[dict[str, object]] = []
    for stage_level, group in frame.groupby("level", sort=False):
        row: dict[str, object] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "stage": stage_level,
        }
        for metric in METRIC_ORDER:
            if metric in group.columns:
                value = group[metric].dropna()
                if not value.empty:
                    row[metric] = float(value.iloc[0])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for value in args.run_dir:
        rows.extend(summarize_one(Path(value)))
    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved summary to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
