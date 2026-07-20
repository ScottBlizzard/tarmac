from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SEED = 20260721
CONCEPTS = (
    "C1_capsule_clear_complete",
    "C2_capsule_disrupted_unclear",
)
SOURCES = ("batch1", "batch2", "third_batch")
C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class Projection:
    scaler: StandardScaler
    pca: PCA

    def transform(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled)


@dataclass
class FoldArtifact:
    fold: int
    test_indices: np.ndarray
    trainval_known_indices: np.ndarray
    trainval_features: np.ndarray
    test_features: np.ndarray
    selected_c: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked H17-0 historical C1/C2 concept probe."
    )
    parser.add_argument("--region-assets-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--concept-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shuffle-repeats", type=int, default=50)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "prevalence": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def safe_binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return {
            "n": int(len(labels)),
            "prevalence": float(labels.mean()) if len(labels) else float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "auc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    return binary_metrics(labels, probabilities)


def select_key(metrics: dict[str, Any], c_value: float) -> tuple[float, float, float]:
    return (
        float(metrics["balanced_accuracy"]),
        float(metrics["auc"]),
        -abs(np.log10(c_value) + 1.0),
    )


def fit_projection(
    features: np.ndarray,
    fit_indices: np.ndarray,
    components: int,
    seed: int,
) -> Projection:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[fit_indices])
    maximum = min(
        int(components),
        int(scaled.shape[0] - 1),
        int(scaled.shape[1]),
    )
    if maximum < 2:
        raise ValueError("Insufficient samples for H17-0 PCA")
    pca = PCA(
        n_components=maximum,
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(scaled)
    return Projection(scaler=scaler, pca=pca)


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=4000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(features, labels)
    return model


def fold_indices(
    metadata: pd.DataFrame, fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_values = metadata["master_fold_id"].to_numpy(dtype=int)
    validation_fold = fold % 5 + 1
    test = np.flatnonzero(fold_values == fold)
    validation = np.flatnonzero(fold_values == validation_fold)
    train = np.flatnonzero(
        (fold_values != fold) & (fold_values != validation_fold)
    )
    if not len(train) or not len(validation) or not len(test):
        raise ValueError(f"Empty H17-0 partition for fold {fold}")
    return train, validation, test


def load_inputs(
    region_assets_dir: Path,
    metadata_path: Path,
    concept_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    metadata = pd.read_csv(
        metadata_path,
        usecols=[
            "feature_row",
            "case_id",
            "source_dataset",
            "master_fold_id",
            "is_frozen_external",
        ],
        dtype={"case_id": str},
        encoding="utf-8-sig",
    ).sort_values("feature_row").reset_index(drop=True)
    metadata.columns = [str(column).lstrip("\ufeff") for column in metadata.columns]
    metadata["source_dataset"] = (
        metadata["source_dataset"]
        .astype(str)
        .map(lambda value: "third_batch" if value.startswith("third_batch") else value)
    )
    if len(metadata) != 591 or metadata["case_id"].nunique() != 591:
        raise ValueError("H17-0 requires the locked 591-case internal cohort")
    if not np.array_equal(metadata["feature_row"], np.arange(591)):
        raise ValueError("Feature rows are not contiguous")
    external_values = metadata["is_frozen_external"]
    if external_values.dtype == bool:
        external_mask = external_values.to_numpy(dtype=bool)
    else:
        external_mask = (
            external_values.astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes"})
            .to_numpy(dtype=bool)
        )
    if external_mask.any():
        raise ValueError("Strict external rows entered H17-0")
    if metadata["source_dataset"].value_counts().to_dict() != {
        "batch1": 117,
        "batch2": 168,
        "third_batch": 306,
    }:
        raise ValueError("Source totals differ from the locked cohort")
    if set(metadata["master_fold_id"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Master folds are incomplete")

    order = pd.read_csv(
        region_assets_dir / "region_case_order_server_only.csv",
        usecols=["feature_row", "case_id", "master_fold_id"],
        dtype={"case_id": str},
        encoding="utf-8-sig",
    ).sort_values("feature_row").reset_index(drop=True)
    if not order["case_id"].equals(metadata["case_id"]):
        raise ValueError("H13 region bank and H16 metadata case order differ")
    if not np.array_equal(
        order["master_fold_id"].to_numpy(dtype=int),
        metadata["master_fold_id"].to_numpy(dtype=int),
    ):
        raise ValueError("H13 region bank and H16 fold IDs differ")

    concepts = pd.read_csv(
        concept_path,
        usecols=["case_id", "source_dataset", *CONCEPTS],
        dtype={"case_id": str},
        encoding="utf-8-sig",
    )
    concepts.columns = [str(column).lstrip("\ufeff") for column in concepts.columns]
    concepts["source_dataset"] = (
        concepts["source_dataset"]
        .astype(str)
        .map(lambda value: "third_batch" if value.startswith("third_batch") else value)
    )
    concepts = metadata[["case_id", "source_dataset"]].merge(
        concepts,
        on=["case_id", "source_dataset"],
        how="left",
        validate="one_to_one",
    )
    if concepts[list(CONCEPTS)].isna().any().any():
        raise ValueError("H14 concept labels do not cover all 591 internal cases")
    for concept in CONCEPTS:
        states = set(concepts[concept].astype(int).unique())
        if not states.issubset({-1, 0, 1}):
            raise ValueError(f"Unexpected states for {concept}: {states}")

    c1 = concepts[CONCEPTS[0]].to_numpy(dtype=int)
    c2 = concepts[CONCEPTS[1]].to_numpy(dtype=int)
    known = (c1 >= 0) & (c2 >= 0)
    if int(known.sum()) != 445:
        raise ValueError("C1/C2 known count differs from the locked H14 result")
    if not np.array_equal(c2[known], 1 - c1[known]):
        raise ValueError("C1/C2 are not exact complements on known cases")
    if not np.array_equal(c1 < 0, c2 < 0):
        raise ValueError("C1/C2 unknown masks differ")

    region_features = np.load(
        region_assets_dir / "region_features.float16.npy", mmap_mode="r"
    )
    descriptors = np.load(
        region_assets_dir / "region_descriptors.float32.npy", mmap_mode="r"
    )
    pairs = np.load(
        region_assets_dir / "region_pair_features.float32.npy", mmap_mode="r"
    )
    if region_features.shape != (591, 16, 1024):
        raise ValueError(f"Unexpected region feature shape {region_features.shape}")
    if descriptors.shape[:2] != (591, 16) or pairs.shape[:3] != (591, 16, 16):
        raise ValueError("Unexpected H13 descriptor or pair feature shapes")
    return metadata, concepts, region_features, descriptors, pairs


def build_candidate_features(
    region_features: np.ndarray,
    descriptors: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, tuple[np.ndarray, int]]:
    embedding = np.asarray(region_features, dtype=np.float32)
    descriptor = np.asarray(descriptors, dtype=np.float32)
    pair = np.asarray(pairs, dtype=np.float32)

    specimen = embedding[:, 0]
    core = embedding[:, 1]
    outer = embedding[:, 2]
    global_features = np.concatenate(
        [specimen, descriptor[:, 0]],
        axis=1,
    )
    outer_features = np.concatenate(
        [outer, descriptor[:, 2]],
        axis=1,
    )
    spatial_features = np.concatenate(
        [
            specimen,
            core,
            outer,
            outer - core,
            descriptor[:, 0],
            descriptor[:, 1],
            descriptor[:, 2],
            pair[:, 0, 2],
            pair[:, 1, 2],
        ],
        axis=1,
    )
    candidates = {
        "P0_global_specimen": (global_features, 32),
        "P1_outer_boundary": (outer_features, 32),
        "P1_spatial_roles": (spatial_features, 64),
    }
    for name, (features, _) in candidates.items():
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite feature values in {name}")
    return candidates


def run_candidate_concept(
    candidate: str,
    features: np.ndarray,
    components: int,
    concept: str,
    metadata: pd.DataFrame,
    concepts: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, list[FoldArtifact], pd.DataFrame]:
    labels_all = concepts[concept].to_numpy(dtype=int)
    known_all = labels_all >= 0
    rows: list[pd.DataFrame] = []
    artifacts: list[FoldArtifact] = []
    selection_rows: list[dict[str, Any]] = []

    for fold in range(1, 6):
        train, validation, test = fold_indices(metadata, fold)
        projection = fit_projection(
            features,
            train,
            components=components,
            seed=seed + 1000 * fold,
        )
        train_known = train[known_all[train]]
        validation_known = validation[known_all[validation]]
        if np.unique(labels_all[train_known]).size < 2:
            raise ValueError(f"{concept} fold {fold} train has one class")
        if np.unique(labels_all[validation_known]).size < 2:
            raise ValueError(f"{concept} fold {fold} validation has one class")
        x_train = projection.transform(features[train_known])
        x_validation = projection.transform(features[validation_known])
        y_train = labels_all[train_known]
        y_validation = labels_all[validation_known]

        selected_c = None
        selected_key = None
        for c_value in C_GRID:
            model = fit_logistic(
                x_train,
                y_train,
                c_value=c_value,
                seed=seed + 1000 * fold,
            )
            probability = model.predict_proba(x_validation)[:, 1]
            metrics = binary_metrics(y_validation, probability)
            key = select_key(metrics, c_value)
            selection_rows.append(
                {
                    "candidate": candidate,
                    "concept": concept,
                    "fold": fold,
                    "c_value": c_value,
                    "selected": False,
                    **metrics,
                }
            )
            if selected_key is None or key > selected_key:
                selected_key = key
                selected_c = c_value
        assert selected_c is not None
        for record in reversed(selection_rows):
            if (
                record["candidate"] == candidate
                and record["concept"] == concept
                and record["fold"] == fold
                and record["c_value"] == selected_c
            ):
                record["selected"] = True
                break

        trainval = np.concatenate([train, validation])
        trainval_known = trainval[known_all[trainval]]
        final_projection = fit_projection(
            features,
            trainval,
            components=components,
            seed=seed + 1000 * fold + 97,
        )
        x_trainval = final_projection.transform(features[trainval_known])
        x_test = final_projection.transform(features[test])
        model = fit_logistic(
            x_trainval,
            labels_all[trainval_known],
            c_value=float(selected_c),
            seed=seed + 1000 * fold + 97,
        )
        probability = model.predict_proba(x_test)[:, 1]
        frame = metadata.iloc[test][
            ["feature_row", "case_id", "source_dataset", "master_fold_id"]
        ].copy()
        frame.insert(0, "candidate", candidate)
        frame.insert(1, "concept", concept)
        frame["state"] = labels_all[test]
        frame["known"] = frame["state"].ge(0)
        frame["probability"] = probability
        frame["prediction"] = (probability >= 0.5).astype(int)
        frame["correct"] = np.where(
            frame["known"],
            frame["prediction"].eq(frame["state"]),
            False,
        )
        rows.append(frame)
        artifacts.append(
            FoldArtifact(
                fold=fold,
                test_indices=test,
                trainval_known_indices=trainval_known,
                trainval_features=x_trainval,
                test_features=x_test,
                selected_c=float(selected_c),
            )
        )

    return (
        pd.concat(rows, ignore_index=True),
        artifacts,
        pd.DataFrame(selection_rows),
    )


def source_one_hot(metadata: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            metadata["source_dataset"].astype(str).eq(source).to_numpy(dtype=float)
            for source in SOURCES
        ]
    )


def run_source_only_concept(
    concept: str,
    metadata: pd.DataFrame,
    concepts: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    features = source_one_hot(metadata)
    labels_all = concepts[concept].to_numpy(dtype=int)
    known_all = labels_all >= 0
    rows: list[pd.DataFrame] = []
    for fold in range(1, 6):
        train, validation, test = fold_indices(metadata, fold)
        train_known = train[known_all[train]]
        validation_known = validation[known_all[validation]]
        selected_c = None
        selected_key = None
        for c_value in C_GRID:
            model = fit_logistic(
                features[train_known],
                labels_all[train_known],
                c_value=c_value,
                seed=seed + 2000 * fold,
            )
            probability = model.predict_proba(features[validation_known])[:, 1]
            metrics = binary_metrics(labels_all[validation_known], probability)
            key = select_key(metrics, c_value)
            if selected_key is None or key > selected_key:
                selected_key = key
                selected_c = c_value
        assert selected_c is not None
        trainval = np.concatenate([train, validation])
        trainval_known = trainval[known_all[trainval]]
        model = fit_logistic(
            features[trainval_known],
            labels_all[trainval_known],
            c_value=float(selected_c),
            seed=seed + 2000 * fold + 97,
        )
        probability = model.predict_proba(features[test])[:, 1]
        frame = metadata.iloc[test][
            ["feature_row", "case_id", "source_dataset", "master_fold_id"]
        ].copy()
        frame.insert(0, "candidate", "P3_source_only_concept")
        frame.insert(1, "concept", concept)
        frame["state"] = labels_all[test]
        frame["known"] = frame["state"].ge(0)
        frame["probability"] = probability
        frame["prediction"] = (probability >= 0.5).astype(int)
        frame["correct"] = np.where(
            frame["known"],
            frame["prediction"].eq(frame["state"]),
            False,
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def run_source_classifier(
    features: np.ndarray,
    components: int,
    metadata: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_to_index = {source: index for index, source in enumerate(SOURCES)}
    labels = metadata["source_dataset"].map(source_to_index).to_numpy(dtype=int)
    probabilities = np.zeros((len(metadata), len(SOURCES)), dtype=float)
    predictions = np.zeros(len(metadata), dtype=int)
    for fold in range(1, 6):
        train, validation, test = fold_indices(metadata, fold)
        projection = fit_projection(
            features,
            train,
            components=components,
            seed=seed + 3000 * fold,
        )
        x_train = projection.transform(features[train])
        x_validation = projection.transform(features[validation])
        selected_c = None
        selected_key = None
        for c_value in C_GRID:
            model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=4000,
                random_state=seed + 3000 * fold,
                solver="lbfgs",
            )
            model.fit(x_train, labels[train])
            probability = model.predict_proba(x_validation)
            prediction = probability.argmax(axis=1)
            bacc = balanced_accuracy_score(labels[validation], prediction)
            auc = roc_auc_score(
                labels[validation],
                probability,
                labels=np.arange(len(SOURCES)),
                multi_class="ovr",
                average="macro",
            )
            key = (float(bacc), float(auc), -abs(np.log10(c_value) + 1.0))
            if selected_key is None or key > selected_key:
                selected_key = key
                selected_c = c_value
        assert selected_c is not None
        trainval = np.concatenate([train, validation])
        final_projection = fit_projection(
            features,
            trainval,
            components=components,
            seed=seed + 3000 * fold + 97,
        )
        model = LogisticRegression(
            C=float(selected_c),
            class_weight="balanced",
            max_iter=4000,
            random_state=seed + 3000 * fold + 97,
            solver="lbfgs",
        )
        model.fit(final_projection.transform(features[trainval]), labels[trainval])
        fold_probability = model.predict_proba(
            final_projection.transform(features[test])
        )
        probabilities[test] = fold_probability
        predictions[test] = fold_probability.argmax(axis=1)

    frame = metadata[
        ["feature_row", "case_id", "source_dataset", "master_fold_id"]
    ].copy()
    frame["source_idx"] = labels
    frame["pred_source_idx"] = predictions
    for source_index, source in enumerate(SOURCES):
        frame[f"prob_{source}"] = probabilities[:, source_index]
    metrics = {
        "n": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_ovr_auc": float(
            roc_auc_score(
                labels,
                probabilities,
                labels=np.arange(len(SOURCES)),
                multi_class="ovr",
                average="macro",
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
    }
    return frame, metrics


def source_stratified_shuffle(
    labels: np.ndarray,
    indices: np.ndarray,
    metadata: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = labels.copy()
    source_values = metadata["source_dataset"].astype(str).to_numpy()
    for source in SOURCES:
        positions = np.flatnonzero(source_values[indices] == source)
        if len(positions):
            shuffled[positions] = rng.permutation(shuffled[positions])
    return shuffled


def run_shuffle_controls(
    artifacts: list[FoldArtifact],
    concept: str,
    metadata: pd.DataFrame,
    concepts: pd.DataFrame,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_all = concepts[concept].to_numpy(dtype=int)
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 100_000 + repeat)
        fold_rows: list[pd.DataFrame] = []
        for artifact in artifacts:
            trainval_known = artifact.trainval_known_indices
            original = labels_all[trainval_known]
            shuffled = source_stratified_shuffle(
                original,
                trainval_known,
                metadata,
                rng,
            )
            if np.unique(shuffled).size < 2:
                raise ValueError("Shuffled control lost a concept class")
            model = fit_logistic(
                artifact.trainval_features,
                shuffled,
                c_value=artifact.selected_c,
                seed=seed + 100_000 + 1000 * artifact.fold + repeat,
            )
            probability = model.predict_proba(artifact.test_features)[:, 1]
            test = artifact.test_indices
            frame = metadata.iloc[test][
                ["feature_row", "case_id", "source_dataset", "master_fold_id"]
            ].copy()
            frame["repeat"] = repeat
            frame["fold"] = artifact.fold
            frame["state"] = labels_all[test]
            frame["known"] = frame["state"].ge(0)
            frame["probability"] = probability
            fold_rows.append(frame)
        combined = pd.concat(fold_rows, ignore_index=True)
        known = combined["known"].to_numpy(dtype=bool)
        metrics = binary_metrics(
            combined.loc[known, "state"].to_numpy(dtype=int),
            combined.loc[known, "probability"].to_numpy(dtype=float),
        )
        metric_rows.append({"repeat": repeat, **metrics})
        prediction_rows.append(combined)
    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    for (candidate, concept), group in predictions.groupby(
        ["candidate", "concept"], sort=False
    ):
        known = group.loc[group["known"]].copy()
        aggregate_rows.append(
            {
                "candidate": candidate,
                "concept": concept,
                **binary_metrics(
                    known["state"].to_numpy(dtype=int),
                    known["probability"].to_numpy(dtype=float),
                ),
            }
        )
        for fold, fold_group in known.groupby("master_fold_id", sort=True):
            stratified_rows.append(
                {
                    "candidate": candidate,
                    "concept": concept,
                    "stratum_type": "fold",
                    "stratum": str(int(fold)),
                    **safe_binary_metrics(
                        fold_group["state"].to_numpy(dtype=int),
                        fold_group["probability"].to_numpy(dtype=float),
                    ),
                }
            )
        for source, source_group in known.groupby("source_dataset", sort=False):
            stratified_rows.append(
                {
                    "candidate": candidate,
                    "concept": concept,
                    "stratum_type": "source",
                    "stratum": str(source),
                    **safe_binary_metrics(
                        source_group["state"].to_numpy(dtype=int),
                        source_group["probability"].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(stratified_rows)


def candidate_macro(
    aggregate: pd.DataFrame,
    stratified: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "auc",
        "sensitivity",
        "specificity",
    ]
    aggregate_macro = (
        aggregate.groupby("candidate", as_index=False)[metric_columns]
        .mean()
        .rename(columns={column: f"macro_{column}" for column in metric_columns})
    )
    aggregate_macro["known_per_concept"] = (
        aggregate.groupby("candidate")["n"].min().to_numpy(dtype=int)
    )
    stratified_macro = (
        stratified.groupby(
            ["candidate", "stratum_type", "stratum"],
            as_index=False,
        )[metric_columns]
        .mean()
        .rename(columns={column: f"macro_{column}" for column in metric_columns})
    )
    return aggregate_macro, stratified_macro


def evaluate_gate(
    aggregate_macro: pd.DataFrame,
    stratified_macro: pd.DataFrame,
    shuffle_metrics: pd.DataFrame,
) -> dict[str, Any]:
    lookup = aggregate_macro.set_index("candidate")
    selected = lookup.loc["P1_spatial_roles"]
    global_probe = lookup.loc["P0_global_specimen"]
    outer_probe = lookup.loc["P1_outer_boundary"]
    source_only = lookup.loc["P3_source_only_concept"]

    selected_strata = stratified_macro.loc[
        stratified_macro["candidate"].eq("P1_spatial_roles")
    ]
    fold_auc = selected_strata.loc[
        selected_strata["stratum_type"].eq("fold"), "macro_auc"
    ]
    source_auc = selected_strata.loc[
        selected_strata["stratum_type"].eq("source"), "macro_auc"
    ]
    shuffle_auc_mean = float(shuffle_metrics["auc"].mean())
    shuffle_auc_q95 = float(shuffle_metrics["auc"].quantile(0.95))
    true_shuffle_gap = float(selected["macro_auc"] - shuffle_auc_mean)
    source_only_gap = float(selected["macro_auc"] - source_only["macro_auc"])

    checks = {
        "macro_auc_ge_0_75": bool(selected["macro_auc"] >= 0.75),
        "macro_bacc_ge_0_68": bool(selected["macro_balanced_accuracy"] >= 0.68),
        "folds_auc_gt_0_70_ge_4": bool((fold_auc > 0.70).sum() >= 4),
        "all_sources_auc_ge_0_65": bool(
            len(source_auc) == 3 and (source_auc >= 0.65).all()
        ),
        "true_minus_shuffle_auc_ge_0_10": bool(true_shuffle_gap >= 0.10),
        "source_only_gap_ge_0_05": bool(source_only_gap >= 0.05),
        "outer_boundary_auc_ge_0_70": bool(outer_probe["macro_auc"] >= 0.70),
        "spatial_not_below_global_by_gt_0_02": bool(
            selected["macro_auc"] >= global_probe["macro_auc"] - 0.02
        ),
    }
    passed = all(checks.values())
    return {
        "status": (
            "PASS_H17_0_PROCEED_60_CASE_MORPHOLOGY_PILOT"
            if passed
            else "FAIL_HISTORICAL_TEXT_PROBE_RUN_20_CASE_VISIBILITY_MINIPILOT"
        ),
        "passed": passed,
        "selected_candidate": "P1_spatial_roles",
        "selected_macro_auc": float(selected["macro_auc"]),
        "selected_macro_balanced_accuracy": float(
            selected["macro_balanced_accuracy"]
        ),
        "global_macro_auc": float(global_probe["macro_auc"]),
        "outer_boundary_macro_auc": float(outer_probe["macro_auc"]),
        "source_only_macro_auc": float(source_only["macro_auc"]),
        "source_only_auc_gap": source_only_gap,
        "shuffle_auc_mean": shuffle_auc_mean,
        "shuffle_auc_q95": shuffle_auc_q95,
        "true_minus_shuffle_auc": true_shuffle_gap,
        "fold_macro_auc": [float(value) for value in fold_auc],
        "source_macro_auc": [float(value) for value in source_auc],
        "checks": checks,
        "independent_historical_concept_dimensions": 1,
        "c1_c2_exact_complements": True,
        "risk_labels_read": False,
        "subtype_labels_read": False,
        "strict_external_read": False,
    }


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError(f"H17-0 seed is locked to {SEED}")
    if args.shuffle_repeats < 20:
        raise ValueError("H17-0 requires at least 20 shuffle repeats")
    set_seed(args.seed)

    region_assets_dir = Path(args.region_assets_dir)
    metadata_path = Path(args.metadata_csv)
    concept_path = Path(args.concept_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "RUN.status"
    status_path.write_text("running_h17_0\n", encoding="utf-8")

    metadata, concepts, region_features, descriptors, pairs = load_inputs(
        region_assets_dir,
        metadata_path,
        concept_path,
    )
    candidates = build_candidate_features(region_features, descriptors, pairs)

    prediction_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    selected_artifacts: list[FoldArtifact] | None = None
    for candidate, (features, components) in candidates.items():
        for concept in CONCEPTS:
            predictions, artifacts, selections = run_candidate_concept(
                candidate,
                features,
                components,
                concept,
                metadata,
                concepts,
                args.seed,
            )
            prediction_frames.append(predictions)
            selection_frames.append(selections)
            if (
                candidate == "P1_spatial_roles"
                and concept == "C1_capsule_clear_complete"
            ):
                selected_artifacts = artifacts

    for concept in CONCEPTS:
        prediction_frames.append(
            run_source_only_concept(concept, metadata, concepts, args.seed)
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    aggregate, stratified = summarize_predictions(predictions)
    aggregate_macro, stratified_macro = candidate_macro(aggregate, stratified)

    if selected_artifacts is None:
        raise AssertionError("P1 spatial fold artifacts were not retained")
    shuffle_predictions, shuffle_metrics = run_shuffle_controls(
        selected_artifacts,
        "C1_capsule_clear_complete",
        metadata,
        concepts,
        repeats=args.shuffle_repeats,
        seed=args.seed,
    )
    source_predictions, source_metrics = run_source_classifier(
        candidates["P1_spatial_roles"][0],
        candidates["P1_spatial_roles"][1],
        metadata,
        args.seed,
    )
    gate = evaluate_gate(aggregate_macro, stratified_macro, shuffle_metrics)
    gate["source_classifier"] = source_metrics

    write_csv(output_dir / "oof_concept_predictions_server_only.csv", predictions)
    write_csv(output_dir / "validation_selection.csv", selections)
    write_csv(output_dir / "concept_metrics.csv", aggregate)
    write_csv(output_dir / "concept_stratified_metrics.csv", stratified)
    write_csv(output_dir / "candidate_macro_metrics.csv", aggregate_macro)
    write_csv(output_dir / "candidate_stratified_macro_metrics.csv", stratified_macro)
    write_csv(
        output_dir / "shuffle_predictions_server_only.csv",
        shuffle_predictions,
    )
    write_csv(output_dir / "shuffle_metrics.csv", shuffle_metrics)
    write_csv(
        output_dir / "source_classifier_oof_server_only.csv",
        source_predictions,
    )
    write_json(output_dir / "h17_0_gate.json", gate)

    config = {
        "experiment": "H17_0_HISTORICAL_CONCEPT_PROBE_20260721",
        "seed": args.seed,
        "shuffle_repeats": args.shuffle_repeats,
        "concepts": list(CONCEPTS),
        "independent_concept_dimensions": 1,
        "candidate_dimensions": {
            name: {
                "raw_feature_dim": int(features.shape[1]),
                "pca_components": int(components),
            }
            for name, (features, components) in candidates.items()
        },
        "c_grid": list(C_GRID),
        "input_sha256": {
            "metadata_csv": sha256(metadata_path),
            "concept_csv": sha256(concept_path),
            "region_features": sha256(
                region_assets_dir / "region_features.float16.npy"
            ),
            "region_descriptors": sha256(
                region_assets_dir / "region_descriptors.float32.npy"
            ),
            "region_pair_features": sha256(
                region_assets_dir / "region_pair_features.float32.npy"
            ),
        },
        "feature_roles": {
            "P0_global_specimen": ["specimen"],
            "P1_outer_boundary": ["outer_ring"],
            "P1_spatial_roles": [
                "specimen",
                "core",
                "outer_ring",
                "outer_minus_core",
                "specimen_to_outer_pair",
                "core_to_outer_pair",
            ],
        },
        "label_shuffle": "within source on each outer train+validation partition",
        "risk_labels_read": False,
        "subtype_labels_read": False,
        "strict_external_read": False,
    }
    write_json(output_dir / "h17_0_config.json", config)
    status_path.write_text(gate["status"] + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
