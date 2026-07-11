from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock one LoRA candidate by internal OOF before external evaluation.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_candidate(run_dir: Path) -> dict | None:
    metric_path = run_dir / "oof_metrics.csv"
    domain_path = run_dir / "oof_domain_metrics_mean.csv"
    if not metric_path.exists() or not domain_path.exists():
        return None
    metrics = pd.read_csv(metric_path, encoding="utf-8-sig")
    overall = metrics[
        metrics["split"].astype(str).eq("test_oof")
        & metrics["level"].astype(str).eq("case")
        & metrics["aggregation"].astype(str).eq("mean")
    ]
    if overall.empty:
        return None
    domains = pd.read_csv(domain_path, encoding="utf-8-sig")
    source_rows = domains[
        domains["split"].astype(str).str.startswith("test_oof:")
        & ~domains["split"].astype(str).isin(["test_oof:old", "test_oof:third"])
    ]
    row = overall.iloc[0]
    return {
        "run_tag": run_dir.name,
        "internal_run_dir": str(run_dir),
        "internal_auc": float(row["auc"]),
        "internal_accuracy": float(row["accuracy"]),
        "internal_bacc": float(row["balanced_accuracy"]),
        "internal_sensitivity": float(row["sensitivity"]),
        "internal_specificity": float(row["specificity"]),
        "internal_min_source_bacc": float(pd.to_numeric(source_rows["balanced_accuracy"]).min()),
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = [candidate for run_dir in sorted(runs_root.iterdir()) if run_dir.is_dir() if (candidate := read_candidate(run_dir))]
    if not candidates:
        raise RuntimeError(f"No complete LoRA OOF runs found under {runs_root}")
    frame = pd.DataFrame(candidates).sort_values(
        ["internal_min_source_bacc", "internal_bacc", "internal_auc"], ascending=False
    )
    locked = frame.head(1).copy()
    manifest_path = output_root / "LOCKED_LORA_EXTERNAL_CANDIDATE.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, encoding="utf-8-sig")
        if existing.to_dict(orient="records") != locked.to_dict(orient="records"):
            raise RuntimeError("A different locked LoRA candidate already exists; refusing to overwrite it.")
    else:
        locked.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        (output_root / "LOCK_RATIONALE.json").write_text(
            json.dumps(
                {
                    "selection_rule": "maximum internal minimum-source BAcc, then overall BAcc, then AUC",
                    "external_labels_used_for_selection": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print("[locked-lora]\n" + locked.to_string(index=False), flush=True)
    row = locked.iloc[0]
    evaluation_dir = output_root / str(row["run_tag"])
    if not (evaluation_dir / "external_lora_dense_metrics.csv").exists():
        command = [
            sys.executable,
            "/workspace/thymic_project/scripts/evaluate_task7_lora_dense_external_20260711.py",
            "--internal-run-dir",
            str(row["internal_run_dir"]),
            "--output-dir",
            str(evaluation_dir),
            "--batch-size",
            "2",
            "--num-workers",
            "2",
            "--device",
            args.device,
        ]
        print("[command] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
    metrics = pd.read_csv(evaluation_dir / "external_lora_dense_metrics.csv", encoding="utf-8-sig")
    metrics.to_csv(output_root / "locked_lora_external_summary.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
