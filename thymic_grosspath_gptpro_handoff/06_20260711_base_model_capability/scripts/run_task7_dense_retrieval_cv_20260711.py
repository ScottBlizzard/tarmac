from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_task7_dense_feature_cv_20260711 import DEFAULT_SPLIT_CSV, SUBTYPE_NAMES, prepare_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument(
        "--concept-csv",
        default="/workspace/thymic_project/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv",
    )
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--knn-k", type=int, default=11)
    return parser.parse_args()


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def softmax(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), 1e-12)


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def centroid_probability(train_x, train_y, test_x, class_count: int, temperature: float) -> np.ndarray:
    prototypes = []
    for class_idx in range(class_count):
        prototype = normalize(train_x[train_y == class_idx].mean(axis=0, keepdims=True))[0]
        prototypes.append(prototype)
    logits = normalize(test_x) @ np.stack(prototypes).T / max(float(temperature), 1e-6)
    return softmax(logits)


def knn_probability(train_x, train_y, test_x, k: int, temperature: float) -> np.ndarray:
    similarity = normalize(test_x) @ normalize(train_x).T
    k = min(int(k), similarity.shape[1])
    nearest = np.argpartition(similarity, -k, axis=1)[:, -k:]
    nearest_similarity = np.take_along_axis(similarity, nearest, axis=1)
    weights = softmax(nearest_similarity / max(float(temperature), 1e-6), axis=1)
    nearest_labels = train_y[nearest]
    return np.sum(weights * nearest_labels, axis=1)


def main() -> None:
    args = parse_args()
    bank = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, _, _, _ = prepare_metadata(bank, args.split_csv, args.concept_csv, [])
    dense = np.load(bank / "dense_features.float16.npy", mmap_mode="r")
    pooled = np.asarray(dense, dtype=np.float32).mean(axis=2).reshape(len(dense), -1)
    labels = metadata["label_idx"].to_numpy(dtype=int)
    subtypes = metadata["subtype_idx"].to_numpy(dtype=int)
    folds = metadata["master_fold_id"].to_numpy(dtype=int)
    methods = ["risk_centroid", "subtype_centroid", "knn", "logistic"]
    predictions = {method: np.full(len(metadata), np.nan, dtype=float) for method in methods}

    for fold in sorted(np.unique(folds)):
        val_fold = (int(fold) % 5) + 1
        train_mask = ~np.isin(folds, [fold, val_fold])
        test_mask = folds == fold
        train_x = pooled[train_mask]
        test_x = pooled[test_mask]
        train_y = labels[train_mask]
        train_subtype = subtypes[train_mask]

        risk_probability = centroid_probability(train_x, train_y, test_x, 2, args.temperature)[:, 1]
        subtype_probability = centroid_probability(
            train_x, train_subtype, test_x, len(SUBTYPE_NAMES), args.temperature
        )
        predictions["risk_centroid"][test_mask] = risk_probability
        predictions["subtype_centroid"][test_mask] = subtype_probability[:, 3:].sum(axis=1)
        predictions["knn"][test_mask] = knn_probability(
            train_x, train_y, test_x, args.knn_k, args.temperature
        )
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=20260711 + int(fold),
        )
        classifier.fit(train_scaled, train_y)
        predictions["logistic"][test_mask] = classifier.predict_proba(test_scaled)[:, 1]

    rows = []
    prediction_frame = metadata.copy()
    for method in methods:
        probability = predictions[method]
        if np.isnan(probability).any():
            raise RuntimeError(f"Missing OOF predictions for {method}")
        prediction_frame[f"prob_{method}"] = probability
        rows.append({"method": method, "group_type": "overall", "group": "all", **metrics(labels, probability)})
        for domain, group in metadata.groupby("source_dataset"):
            index = group.index.to_numpy(dtype=int)
            rows.append(
                {
                    "method": method,
                    "group_type": "source_dataset",
                    "group": str(domain),
                    **metrics(labels[index], probability[index]),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "retrieval_oof_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(output_dir / "retrieval_oof_predictions.csv", index=False, encoding="utf-8-sig")
    overall = result[result["group_type"].eq("overall")].sort_values("balanced_accuracy", ascending=False)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
