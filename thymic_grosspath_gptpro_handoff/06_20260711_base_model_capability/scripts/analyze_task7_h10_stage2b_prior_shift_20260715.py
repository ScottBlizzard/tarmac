from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROLE_FACTORS = {
    "anchor_warmup": {
        "canonical_anchor": 1.50,
        "stable_noncanonical": 1.00,
        "learnable_boundary": 0.75,
        "persistent_canonical_failure": 0.75,
        "persistent_mimic_failure": 0.50,
        "persistent_sparse_or_missing": 0.50,
    },
    "boundary_bridge": {
        "canonical_anchor": 1.00,
        "stable_noncanonical": 0.75,
        "learnable_boundary": 1.50,
        "persistent_canonical_failure": 1.00,
        "persistent_mimic_failure": 0.75,
        "persistent_sparse_or_missing": 0.50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit H10 Stage-2B prior shift.")
    parser.add_argument("--roles-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def normalized_curriculum_weights(labels: np.ndarray, roles: np.ndarray, stage: str) -> np.ndarray:
    counts = pd.Series(labels).value_counts().to_dict()
    base = np.asarray([1.0 / counts[int(label)] for label in labels], dtype=float)
    base /= base.mean()
    factors = ROLE_FACTORS[stage]
    weights = base * np.asarray([factors[str(role)] for role in roles], dtype=float)
    median = float(np.median(weights))
    weights = np.clip(weights, 0.2 * median, 5.0 * median)
    return weights / weights.mean()


def bacc(labels: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    prediction = np.asarray(probability) >= threshold
    labels = np.asarray(labels, dtype=int)
    low = labels == 0
    return float((np.mean(~prediction[low]) + np.mean(prediction[~low])) / 2)


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.r_[0.0, np.asarray(probability, dtype=float), 1.0])
    values = np.asarray([bacc(labels, probability, threshold) for threshold in candidates])
    best = values.max()
    positions = np.flatnonzero(np.isclose(values, best, atol=1e-12, rtol=0.0))
    selected = positions[np.argmin(np.abs(candidates[positions] - 0.5))]
    return float(candidates[selected]), float(best)


def pooled_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "n": int(len(labels)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "accuracy": float(np.mean(predictions == labels)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> None:
    args = parse_args()
    roles_dir = Path(args.roles_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mass_rows = []
    threshold_rows = []
    pooled = {
        name: {"labels": [], "predictions": []}
        for name in (
            "candidate_fixed",
            "candidate_validation_threshold",
            "baseline_fixed",
            "baseline_validation_threshold",
        )
    }
    for fold in range(1, 6):
        role = pd.read_csv(
            roles_dir / f"fold_{fold}/nested_training_roles_server_only.csv",
            dtype={"case_id": str},
            encoding="utf-8-sig",
        )
        labels = role["label_idx"].to_numpy(int)
        roles = role["training_role"].astype(str).to_numpy()
        for stage in ROLE_FACTORS:
            weights = normalized_curriculum_weights(labels, roles, stage)
            low_mass = float(weights[labels == 0].sum() / weights.sum())
            mass_rows.append(
                {
                    "fold_id": fold,
                    "stage": stage,
                    "low_risk_weight_mass": low_mass,
                    "high_risk_weight_mass": 1.0 - low_mass,
                    "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
                }
            )

        for method, directory in (("candidate", candidate_dir), ("baseline", baseline_dir)):
            validation = pd.read_csv(directory / f"fold_{fold}/validation_predictions.csv")
            test = pd.read_csv(directory / f"fold_{fold}/test_predictions.csv")
            threshold, validation_bacc = choose_threshold(
                validation["label_idx"].to_numpy(int),
                validation["prob_high"].to_numpy(float),
            )
            test_labels = test["label_idx"].to_numpy(int)
            test_probability = test["prob_high"].to_numpy(float)
            fixed_predictions = (test_probability >= 0.5).astype(int)
            calibrated_predictions = (test_probability >= threshold).astype(int)
            threshold_rows.append(
                {
                    "fold_id": fold,
                    "method": method,
                    "validation_threshold": threshold,
                    "validation_bacc": validation_bacc,
                    "test_fixed_bacc": bacc(test_labels, test_probability, 0.5),
                    "test_validation_threshold_bacc": bacc(
                        test_labels, test_probability, threshold
                    ),
                }
            )
            pooled[f"{method}_fixed"]["labels"].append(test_labels)
            pooled[f"{method}_fixed"]["predictions"].append(fixed_predictions)
            pooled[f"{method}_validation_threshold"]["labels"].append(test_labels)
            pooled[f"{method}_validation_threshold"]["predictions"].append(
                calibrated_predictions
            )

    mass = pd.DataFrame(mass_rows)
    thresholds = pd.DataFrame(threshold_rows)
    pooled_rows = []
    for method, values in pooled.items():
        labels = np.concatenate(values["labels"])
        predictions = np.concatenate(values["predictions"])
        pooled_rows.append({"method": method, **pooled_metrics(labels, predictions)})
    pooled_frame = pd.DataFrame(pooled_rows)
    summary = {
        "mean_weight_mass_by_stage": mass.groupby("stage")[[
            "low_risk_weight_mass",
            "high_risk_weight_mass",
        ]].mean().to_dict(orient="index"),
        "pooled_threshold_audit": pooled_frame.set_index("method").to_dict(orient="index"),
    }

    mass.to_csv(output_dir / "class_weight_mass_by_fold_stage.csv", index=False)
    thresholds.to_csv(output_dir / "validation_threshold_audit_by_fold.csv", index=False)
    pooled_frame.to_csv(output_dir / "validation_threshold_audit_pooled.csv", index=False)
    (output_dir / "prior_shift_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nWEIGHT MASS\n" + mass.to_string(index=False))
    print("\nTHRESHOLDS\n" + thresholds.to_string(index=False))
    print("\nPOOLED\n" + pooled_frame.to_string(index=False))


if __name__ == "__main__":
    main()
