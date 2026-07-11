from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def value_for(metrics: pd.DataFrame, group_type: str, group: str, column: str) -> float:
    rows = metrics[(metrics["group_type"].astype(str) == group_type) & (metrics["group"].astype(str) == group)]
    return float(rows.iloc[0][column]) if not rows.empty else float("nan")


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root)
    rows = []
    for metric_path in sorted(root.glob("*/oof_metrics.csv")):
        run_dir = metric_path.parent
        metrics = pd.read_csv(metric_path, encoding="utf-8-sig")
        config_path = run_dir / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        source_rows = metrics[metrics["group_type"].astype(str).eq("source_dataset")]
        source_bacc = pd.to_numeric(source_rows["balanced_accuracy"], errors="coerce")
        subtype_path = run_dir / "oof_subtype_summary.csv"
        subtype = pd.read_csv(subtype_path, encoding="utf-8-sig") if subtype_path.exists() else pd.DataFrame()
        subtype_accuracy = (
            subtype.set_index(subtype["subtype"].astype(str))["risk_accuracy"].to_dict() if not subtype.empty else {}
        )
        row = {
            "run_tag": run_dir.name,
            "model_name": config.get("feature_bank_config", {}).get("model_name", ""),
            "views": ",".join(config.get("feature_bank_config", {}).get("views", [])),
            "pooling": config.get("pooling", ""),
            "expert_mode": config.get("expert_mode", ""),
            "risk_objective": config.get("risk_objective", ""),
            "overall_auc": value_for(metrics, "overall", "all", "auc"),
            "overall_accuracy": value_for(metrics, "overall", "all", "accuracy"),
            "overall_bacc": value_for(metrics, "overall", "all", "balanced_accuracy"),
            "overall_sensitivity": value_for(metrics, "overall", "all", "sensitivity"),
            "overall_specificity": value_for(metrics, "overall", "all", "specificity"),
            "old_bacc": value_for(metrics, "domain", "old_data", "balanced_accuracy"),
            "third_bacc": value_for(metrics, "domain", "third_batch", "balanced_accuracy"),
            "min_source_bacc": float(source_bacc.min()) if not source_bacc.empty else float("nan"),
            "mean_source_bacc": float(source_bacc.mean()) if not source_bacc.empty else float("nan"),
            "B1_risk_accuracy": float(subtype_accuracy.get("B1", np.nan)),
            "B2_risk_accuracy": float(subtype_accuracy.get("B2", np.nan)),
            "AB_risk_accuracy": float(subtype_accuracy.get("AB", np.nan)),
            "B3_risk_accuracy": float(subtype_accuracy.get("B3", np.nan)),
            "TC_risk_accuracy": float(subtype_accuracy.get("TC", np.nan)),
        }
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No complete dense CV runs found under {root}")
    summary = pd.DataFrame(rows).sort_values(
        ["min_source_bacc", "overall_bacc", "overall_auc"], ascending=False, na_position="last"
    )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
