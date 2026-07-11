from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score


METRICS = ("accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap locked Task7 external forced-classification metrics.")
    parser.add_argument("--dense-output-root", required=True)
    parser.add_argument("--lora-output-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def metric_values(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "auc": float(roc_auc_score(labels, probability)),
        "sensitivity": float(recall_score(labels, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(labels, prediction, pos_label=0, zero_division=0)),
    }


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if not len(indices):
            raise ValueError("Each external domain must contain both risk classes.")
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(parts)


def main() -> None:
    args = parse_args()
    dense_root = Path(args.dense_output_root)
    lora_root = Path(args.lora_output_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense = pd.read_csv(
        dense_root / "locked_equal_fusion_external_predictions.csv",
        dtype={"case_id": str},
        encoding="utf-8-sig",
    )
    manifest = pd.read_csv(
        lora_root / "LOCKED_LORA_EXTERNAL_CANDIDATE.csv", encoding="utf-8-sig"
    )
    lora_tag = str(manifest.iloc[0]["run_tag"])
    lora = pd.read_csv(
        lora_root / lora_tag / "external_lora_dense_predictions.csv",
        dtype={"case_id": str},
        encoding="utf-8-sig",
    )
    dense_probability = next(
        (column for column in ("prob_high_fused", "prob_high") if column in dense.columns),
        None,
    )
    if dense_probability is None:
        raise ValueError("Dense prediction file lacks prob_high_fused or prob_high")
    print(f"[dense-probability-column] {dense_probability}", flush=True)
    keep = ["case_id", "label_idx", "domain", dense_probability]
    dense_for_merge = dense[keep].rename(columns={dense_probability: "prob_dense"})
    lora_for_merge = lora[["case_id", "label_idx", "domain", "prob_high"]].rename(
        columns={"prob_high": "prob_lora"}
    )
    paired = dense_for_merge.merge(
        lora_for_merge,
        on=["case_id", "label_idx", "domain"],
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(dense) or len(paired) != len(lora):
        raise RuntimeError("Dense and LoRA locked external predictions do not cover identical cases.")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    difference_rows: list[dict] = []
    for domain, group in paired.groupby("domain", sort=True):
        labels = group["label_idx"].to_numpy(dtype=int)
        probabilities = {
            "locked_dense_fusion": group["prob_dense"].to_numpy(dtype=float),
            "locked_lora": group["prob_lora"].to_numpy(dtype=float),
        }
        estimates = {name: metric_values(labels, probability) for name, probability in probabilities.items()}
        bootstrap = {
            name: {metric: [] for metric in METRICS} for name in probabilities
        }
        differences = {metric: [] for metric in METRICS}
        for _ in range(int(args.iterations)):
            indices = stratified_bootstrap_indices(labels, rng)
            sampled_labels = labels[indices]
            sampled = {
                name: metric_values(sampled_labels, probability[indices])
                for name, probability in probabilities.items()
            }
            for name in probabilities:
                for metric in METRICS:
                    bootstrap[name][metric].append(sampled[name][metric])
            for metric in METRICS:
                differences[metric].append(
                    sampled["locked_dense_fusion"][metric] - sampled["locked_lora"][metric]
                )
        for name in probabilities:
            for metric in METRICS:
                values = np.asarray(bootstrap[name][metric], dtype=float)
                rows.append(
                    {
                        "domain": domain,
                        "model": name,
                        "n": len(group),
                        "metric": metric,
                        "estimate": estimates[name][metric],
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "bootstrap_iterations": int(args.iterations),
                    }
                )
        for metric in METRICS:
            values = np.asarray(differences[metric], dtype=float)
            difference_rows.append(
                {
                    "domain": domain,
                    "contrast": "locked_dense_fusion_minus_locked_lora",
                    "metric": metric,
                    "estimate_difference": (
                        estimates["locked_dense_fusion"][metric]
                        - estimates["locked_lora"][metric]
                    ),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_iterations": int(args.iterations),
                }
            )

    pd.DataFrame(rows).to_csv(
        output_dir / "locked_external_metric_bootstrap_ci.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(difference_rows).to_csv(
        output_dir / "locked_external_paired_model_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
