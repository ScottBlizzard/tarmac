from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect comparable Task7 OOF and source-LODO metrics.")
    parser.add_argument("--search-root", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def overall_metrics(metrics: pd.DataFrame) -> pd.Series:
    if "group_type" in metrics.columns:
        rows = metrics[(metrics["group_type"] == "overall") & (metrics["group"] == "all")]
    else:
        rows = metrics[
            (metrics["split"] == "test_oof")
            & (metrics["level"] == "case")
            & (metrics["aggregation"] == "mean")
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected one overall metric row, found {len(rows)}")
    return rows.iloc[0]


def canonical_source_bacc(run_dir: Path, metrics: pd.DataFrame) -> dict[str, float]:
    if "group_type" in metrics.columns:
        rows = metrics[metrics["group_type"] == "source_dataset"]
        values: dict[str, list[float]] = {}
        for row in rows.itertuples(index=False):
            group = str(row.group)
            canonical = "third_batch" if group.startswith("third_batch") else group
            values.setdefault(canonical, []).append(float(row.balanced_accuracy))
        return {key: min(items) for key, items in values.items()}

    domain_path = run_dir / "oof_domain_metrics_mean.csv"
    if not domain_path.exists():
        return {}
    domains = pd.read_csv(domain_path, encoding="utf-8-sig")
    values = {}
    for canonical in ("batch1", "batch2", "third"):
        rows = domains[domains["split"] == f"test_oof:{canonical}"]
        if len(rows) == 1:
            key = "third_batch" if canonical == "third" else canonical
            values[key] = float(rows.iloc[0]["balanced_accuracy"])
    return values


def infer_split_mode(run_dir: Path) -> str:
    config = read_json(run_dir / "args.json") or read_json(run_dir / "run_config.json")
    split_mode = str(config.get("split_mode", ""))
    if split_mode:
        return split_mode
    return "source_lodo" if "source_lodo" in run_dir.name else "fivefold"


def collect_run(metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parent
    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")
    overall = overall_metrics(metrics)
    source_values = canonical_source_bacc(run_dir, metrics)
    fold_path = run_dir / "cv_fold_summary.csv"
    fold_metrics = pd.read_csv(fold_path, encoding="utf-8-sig") if fold_path.exists() else pd.DataFrame()
    fold_column = next(
        (
            column
            for column in ("test_case_mean_balanced_accuracy", "test_balanced_accuracy")
            if column in fold_metrics.columns
        ),
        "",
    )
    run_match = re.match(r"(\d+)", run_dir.name)
    return {
        "run_id": int(run_match.group(1)) if run_match else None,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "split_mode": infer_split_mode(run_dir),
        "balanced_accuracy": float(overall["balanced_accuracy"]),
        "auc": float(overall["auc"]),
        "sensitivity": float(overall["sensitivity"]),
        "specificity": float(overall["specificity"]),
        "min_class_recall": min(float(overall["sensitivity"]), float(overall["specificity"])),
        "batch1_bacc": source_values.get("batch1"),
        "batch2_bacc": source_values.get("batch2"),
        "third_batch_bacc": source_values.get("third_batch"),
        "min_source_bacc": min(source_values.values()) if source_values else None,
        "min_fold_bacc": (
            float(fold_metrics[fold_column].min())
            if not fold_metrics.empty and fold_column
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    metric_paths = []
    for root in args.search_root:
        metric_paths.extend(Path(root).rglob("oof_metrics.csv"))
    rows = []
    for path in sorted(set(metric_paths)):
        if "smoke" in path.parts or "superseded" in path.parts:
            continue
        try:
            rows.append(collect_run(path))
        except Exception as exc:
            rows.append({"run_name": path.parent.name, "run_dir": str(path.parent), "error": str(exc)})
    results = pd.DataFrame(rows).sort_values(["split_mode", "balanced_accuracy"], ascending=[True, False])
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
