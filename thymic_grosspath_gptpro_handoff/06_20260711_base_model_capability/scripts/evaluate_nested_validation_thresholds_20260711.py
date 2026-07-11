from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fold-specific thresholds selected only on each fold's validation partition."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def probability_column(frame: pd.DataFrame) -> str:
    for column in ("prob_high", "prob_high_risk_group", "probability"):
        if column in frame.columns:
            return column
    raise ValueError(f"No high-risk probability column in {frame.columns.tolist()}")


def read_fold_predictions(fold_dir: Path, split: str) -> pd.DataFrame:
    candidates = [
        fold_dir / f"{split}_case_predictions_mean.csv",
        fold_dir / f"{split}_predictions_with_metadata.csv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No {split} prediction file under {fold_dir}")
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    column = probability_column(frame)
    result = frame.copy()
    result["prob_high"] = pd.to_numeric(result[column], errors="raise").astype(float)
    result["label_idx"] = pd.to_numeric(result["label_idx"], errors="raise").astype(int)
    if "case_id" not in result.columns:
        if "original_case_id" not in result.columns:
            raise ValueError(f"Predictions lack case identifiers: {path}")
        result["case_id"] = result["original_case_id"].astype(str)
    if result["case_id"].duplicated().any():
        raise ValueError(f"Duplicate case ids in {path}")
    return result


def candidate_thresholds(probability: np.ndarray) -> np.ndarray:
    unique = np.unique(np.clip(probability.astype(float), 0.0, 1.0))
    if len(unique) == 1:
        candidates = unique
    else:
        candidates = np.concatenate(
            ([0.0], (unique[:-1] + unique[1:]) / 2.0, [1.0])
        )
    return np.unique(np.concatenate((candidates, [0.5])))


def select_threshold(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation partition must contain both risk classes")
    rows = []
    for threshold in candidate_thresholds(probability):
        prediction = (probability >= threshold).astype(int)
        score = float(balanced_accuracy_score(labels, prediction))
        rows.append((score, -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    best = max(rows)
    return best[3], best[0]


def metric_row(frame: pd.DataFrame, prediction_column: str, group_type: str, group: str) -> dict:
    labels = frame["label_idx"].to_numpy(dtype=int)
    prediction = frame[prediction_column].to_numpy(dtype=int)
    probability = frame["prob_high"].to_numpy(dtype=float)
    sensitivity = float(((prediction == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1))
    specificity = float(((prediction == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1))
    return {
        "decision_rule": prediction_column,
        "group_type": group_type,
        "group": group,
        "n": len(frame),
        "auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "min_class_recall": min(sensitivity, specificity),
    }


def metadata_map(run_dir: Path) -> pd.DataFrame:
    for name in ("oof_case_predictions_mean.csv", "oof_predictions.csv"):
        path = run_dir / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
        frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
        columns = [
            column
            for column in ("case_id", "original_case_id", "source_dataset", "task_l6_label")
            if column in frame.columns
        ]
        if "case_id" in columns:
            return frame[columns].drop_duplicates("case_id")
    return pd.DataFrame(columns=["case_id"])


def canonical_source(values: pd.Series) -> pd.Series:
    result = values.fillna("unknown").astype(str)
    return result.mask(result.str.startswith("third_batch", na=False), "third_batch")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs = sorted(
        (path for path in run_dir.glob("fold_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not fold_dirs:
        raise ValueError(f"No fold directories under {run_dir}")

    threshold_rows = []
    test_frames = []
    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split("_")[-1])
        validation = read_fold_predictions(fold_dir, "val")
        test = read_fold_predictions(fold_dir, "test")
        threshold, validation_bacc = select_threshold(
            validation["label_idx"].to_numpy(dtype=int), validation["prob_high"].to_numpy(dtype=float)
        )
        test = test.copy()
        test["fold_id"] = fold_id
        test["validation_selected_threshold"] = threshold
        test["pred_fixed_0_5"] = (test["prob_high"] >= 0.5).astype(int)
        test["pred_nested_validation_threshold"] = (test["prob_high"] >= threshold).astype(int)
        test_frames.append(test)
        threshold_rows.append(
            {
                "fold_id": fold_id,
                "validation_n": len(validation),
                "validation_selected_threshold": threshold,
                "validation_balanced_accuracy": validation_bacc,
                "test_n": len(test),
            }
        )

    predictions = pd.concat(test_frames, ignore_index=True)
    if predictions["case_id"].duplicated().any():
        raise ValueError("Fold test predictions contain duplicate case ids")
    metadata = metadata_map(run_dir)
    if not metadata.empty:
        predictions = predictions.merge(metadata, on="case_id", how="left", suffixes=("", "_oof"))
    if "source_dataset" in predictions.columns:
        predictions["source_dataset"] = canonical_source(predictions["source_dataset"])

    rows = []
    for decision in ("pred_fixed_0_5", "pred_nested_validation_threshold"):
        rows.append(metric_row(predictions, decision, "overall", "all"))
        if "source_dataset" in predictions.columns:
            for source, group in predictions.groupby("source_dataset", dropna=False):
                rows.append(metric_row(group, decision, "source_dataset", str(source)))

    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "fold_validation_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(output_dir / "nested_threshold_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "nested_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
