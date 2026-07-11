from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


METRICS = ("balanced_accuracy", "auc", "sensitivity", "specificity", "min_source_bacc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired source-and-label-stratified bootstrap for internal fusions.")
    parser.add_argument("--old-oof", required=True)
    parser.add_argument("--old-lodo", required=True)
    parser.add_argument("--new-oof", required=True)
    parser.add_argument("--new-lodo", required=True)
    parser.add_argument("--old-probability-column", default="prob_high_fused")
    parser.add_argument("--new-probability-column", default="prob_high")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def canonical_source(values: pd.Series) -> pd.Series:
    source = values.fillna("unknown").astype(str)
    return source.mask(source.str.startswith("third_batch", na=False), "third_batch")


def load_frame(path: str, probability_column: str, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {"case_id", "label_idx", "source_dataset", probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    result = frame[["case_id", "label_idx", "source_dataset", probability_column]].copy()
    result = result.rename(columns={probability_column: f"probability_{model_name}"})
    result["label_idx"] = pd.to_numeric(result["label_idx"], errors="raise").astype(int)
    result["source_dataset"] = canonical_source(result["source_dataset"])
    return result.sort_values("case_id").reset_index(drop=True)


def paired_frame(old_path: str, new_path: str, old_column: str, new_column: str) -> pd.DataFrame:
    old = load_frame(old_path, old_column, "old")
    new = load_frame(new_path, new_column, "new")
    paired = old.merge(
        new[["case_id", "label_idx", "source_dataset", "probability_new"]],
        on=["case_id", "label_idx", "source_dataset"],
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(old) or len(paired) != len(new):
        raise ValueError(f"Prediction alignment failed: old={len(old)} new={len(new)} paired={len(paired)}")
    return paired


def binary_metrics(labels: np.ndarray, probability: np.ndarray, source: np.ndarray) -> dict[str, float]:
    predicted = probability >= 0.5
    positive = labels == 1
    negative = ~positive
    sensitivity = float((predicted & positive).sum() / max(positive.sum(), 1))
    specificity = float(((~predicted) & negative).sum() / max(negative.sum(), 1))
    source_bacc = []
    for source_name in np.unique(source):
        selected = source == source_name
        source_labels = labels[selected]
        source_predicted = predicted[selected]
        source_positive = source_labels == 1
        source_negative = ~source_positive
        source_sensitivity = float(
            (source_predicted & source_positive).sum() / max(source_positive.sum(), 1)
        )
        source_specificity = float(
            ((~source_predicted) & source_negative).sum() / max(source_negative.sum(), 1)
        )
        source_bacc.append((source_sensitivity + source_specificity) / 2.0)
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "auc": float(roc_auc_score(labels, probability)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "min_source_bacc": min(source_bacc),
    }


def stratified_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for _, group in frame.groupby(["source_dataset", "label_idx"], sort=True):
        indices = group.index.to_numpy(dtype=int)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled)


def bootstrap_protocol(
    protocol: str,
    frame: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    labels = frame["label_idx"].to_numpy(dtype=int)
    source = frame["source_dataset"].to_numpy(dtype=str)
    old_probability = frame["probability_old"].to_numpy(dtype=float)
    new_probability = frame["probability_new"].to_numpy(dtype=float)
    point = {
        "old": binary_metrics(labels, old_probability, source),
        "new": binary_metrics(labels, new_probability, source),
    }
    samples = {model: {metric: [] for metric in METRICS} for model in ("old", "new")}
    differences = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selected = stratified_indices(frame, rng)
        old_metric = binary_metrics(labels[selected], old_probability[selected], source[selected])
        new_metric = binary_metrics(labels[selected], new_probability[selected], source[selected])
        for metric in METRICS:
            samples["old"][metric].append(old_metric[metric])
            samples["new"][metric].append(new_metric[metric])
            differences[metric].append(new_metric[metric] - old_metric[metric])

    metric_rows = []
    difference_rows = []
    for model in ("old", "new"):
        for metric in METRICS:
            values = np.asarray(samples[model][metric], dtype=float)
            metric_rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "metric": metric,
                    "point_estimate": point[model][metric],
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "iterations": iterations,
                }
            )
    for metric in METRICS:
        values = np.asarray(differences[metric], dtype=float)
        difference_rows.append(
            {
                "protocol": protocol,
                "metric": metric,
                "new_minus_old": point["new"][metric] - point["old"][metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "probability_difference_gt_zero": float((values > 0).mean()),
                "iterations": iterations,
            }
        )
    return metric_rows, difference_rows


def main() -> None:
    args = parse_args()
    protocols = {
        "fivefold_oof": paired_frame(
            args.old_oof,
            args.new_oof,
            args.old_probability_column,
            args.new_probability_column,
        ),
        "source_lodo": paired_frame(
            args.old_lodo,
            args.new_lodo,
            args.old_probability_column,
            args.new_probability_column,
        ),
    }
    metric_rows = []
    difference_rows = []
    for offset, (protocol, frame) in enumerate(protocols.items()):
        protocol_metrics, protocol_differences = bootstrap_protocol(
            protocol,
            frame,
            args.iterations,
            args.seed + offset,
        )
        metric_rows.extend(protocol_metrics)
        difference_rows.extend(protocol_differences)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame(metric_rows)
    difference_frame = pd.DataFrame(difference_rows)
    metric_frame.to_csv(output_dir / "metric_ci.csv", index=False, encoding="utf-8-sig")
    difference_frame.to_csv(output_dir / "paired_differences.csv", index=False, encoding="utf-8-sig")
    print(difference_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
