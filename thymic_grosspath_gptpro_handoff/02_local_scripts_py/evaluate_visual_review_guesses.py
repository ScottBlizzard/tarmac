from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_batches(batch_dir: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(batch_dir.glob("visual_review_batch_*.csv")):
        parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(f"No batch CSVs found in {batch_dir}")
    return pd.concat(parts, ignore_index=True)


def evaluate_task(df: pd.DataFrame, truth_col: str, guess_col: str) -> dict[str, object]:
    sub = df[df[truth_col].fillna("").astype(str) != ""].copy()
    sub[guess_col] = sub[guess_col].fillna("").astype(str)
    attempted = sub[sub[guess_col] != ""].copy()
    if attempted.empty:
        return {
            "n_truth": len(sub),
            "n_attempted": 0,
            "coverage": 0.0,
            "accuracy": None,
        }
    attempted["correct"] = attempted[truth_col].astype(str) == attempted[guess_col]
    return {
        "n_truth": len(sub),
        "n_attempted": len(attempted),
        "coverage": round(len(attempted) / len(sub), 4),
        "accuracy": round(float(attempted["correct"].mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()

    df = load_batches(Path(args.batch_dir))
    task5 = evaluate_task(df, "task5_label", "ai_blind_guess_task5")
    task6 = evaluate_task(df, "task6_label", "ai_blind_guess_task6")

    print("task5", task5)
    print("task6", task6)


if __name__ == "__main__":
    main()
