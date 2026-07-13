from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


CHANNELS = ("y", "cb", "cr")
VIEWS = ("whole", "specimen_crop")
N_LEVELS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fixed Haar-frequency separability by source versus Task7 risk."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--project-root", default="/workspace/thymic_project")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=100)
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_registry(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {"case_id", "source_dataset", "label_idx", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing registry columns: {sorted(missing)}")
    frame = frame.sort_values(["source_dataset", "case_id"]).drop_duplicates("case_id")
    if len(frame) != 591:
        raise ValueError(f"Expected 591 unique internal cases, found {len(frame)}")
    frame["source"] = frame["source_dataset"].map(
        lambda value: "third_batch" if str(value).startswith("third_batch") else str(value)
    )
    if set(frame["source"]) != {"batch1", "batch2", "third_batch"}:
        raise ValueError("Unexpected acquisition sources")
    frame["risk"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    if set(frame["risk"]) != {0, 1}:
        raise ValueError("Task7 risk labels must be binary")
    missing_images = [value for value in frame["image_path"] if not Path(str(value)).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} selected images")
    return frame.reset_index(drop=True)


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def color_channels(image: Image.Image) -> dict[str, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return {
        "y": 0.299 * r + 0.587 * g + 0.114 * b,
        "cb": -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5,
        "cr": 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5,
    }


def haar_decompose(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top_left = array[0::2, 0::2]
    top_right = array[0::2, 1::2]
    bottom_left = array[1::2, 0::2]
    bottom_right = array[1::2, 1::2]
    ll = (top_left + top_right + bottom_left + bottom_right) * 0.5
    horizontal = (top_left - top_right + bottom_left - bottom_right) * 0.5
    vertical = (top_left + top_right - bottom_left - bottom_right) * 0.5
    diagonal = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return ll, horizontal, vertical, diagonal


def haar_features(image: Image.Image, view: str) -> dict[str, float]:
    output: dict[str, float] = {}
    epsilon = 1e-12
    for channel_name, channel in color_channels(image).items():
        current = channel
        level_energies: list[float] = []
        for level in range(1, N_LEVELS + 1):
            current, horizontal, vertical, diagonal = haar_decompose(current)
            energies = {
                "horizontal": float(np.mean(np.square(horizontal, dtype=np.float64))),
                "vertical": float(np.mean(np.square(vertical, dtype=np.float64))),
                "diagonal": float(np.mean(np.square(diagonal, dtype=np.float64))),
            }
            detail_energy = float(sum(energies.values()))
            approximation_energy = float(np.mean(np.square(current, dtype=np.float64)))
            total_energy = detail_energy + approximation_energy
            prefix = f"{view}__{channel_name}__level{level}"
            for orientation, energy in energies.items():
                output[f"{prefix}__log_energy_{orientation}"] = float(np.log10(energy + epsilon))
            output[f"{prefix}__log_energy_all"] = float(np.log10(detail_energy + epsilon))
            output[f"{prefix}__log_relative_detail"] = float(
                np.log10((detail_energy + epsilon) / (total_energy + epsilon))
            )
            level_energies.append(detail_energy)
        output[f"{view}__{channel_name}__log_fine_to_coarse_ratio"] = float(
            np.log10((level_energies[0] + epsilon) / (level_energies[-1] + epsilon))
        )
    return output


def extract_features(
    registry: pd.DataFrame,
    image_size: int,
    project_root: Path,
) -> tuple[np.ndarray, list[str]]:
    if image_size % (2**N_LEVELS):
        raise ValueError(f"image-size must be divisible by {2**N_LEVELS}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from thymic_baseline.cropping import extract_specimen_crop

    records: list[dict[str, float]] = []
    for row_index, row in registry.iterrows():
        with Image.open(str(row["image_path"])) as source:
            whole = source.convert("RGB")
        views = {
            "whole": resize_square(whole, image_size),
            "specimen_crop": resize_square(extract_specimen_crop(whole), image_size),
        }
        feature_row: dict[str, float] = {}
        for view in VIEWS:
            feature_row.update(haar_features(views[view], view))
        records.append(feature_row)
        if (row_index + 1) % 50 == 0 or row_index + 1 == len(registry):
            print(f"[extract] {row_index + 1}/{len(registry)}", flush=True)
    feature_names = sorted(records[0])
    matrix = np.asarray([[record[name] for name in feature_names] for record in records], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Frequency feature matrix contains non-finite values")
    return matrix, feature_names


def design_matrix(values: np.ndarray, categories: list[str]) -> np.ndarray:
    text = values.astype(str)
    columns = [np.ones(len(text), dtype=np.float64)]
    columns.extend((text == category).astype(np.float64) for category in categories[1:])
    return np.column_stack(columns)


def residualize_train_test(
    train: np.ndarray,
    test: np.ndarray,
    train_confound: np.ndarray,
    test_confound: np.ndarray,
    categories: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_design = design_matrix(train_confound, categories)
    test_design = design_matrix(test_confound, categories)
    coefficients, _, _, _ = np.linalg.lstsq(train_design, train.astype(np.float64), rcond=None)
    return train - train_design @ coefficients, test - test_design @ coefficients


def joint_strata(target: np.ndarray, confound: np.ndarray) -> np.ndarray:
    return np.asarray([f"{left}|{right}" for left, right in zip(target, confound)], dtype=str)


def cross_validated_scores(
    features: np.ndarray,
    target: np.ndarray,
    confound: np.ndarray,
    confound_categories: list[str],
    seed: int,
    repeats: int,
) -> np.ndarray:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    strata = joint_strata(target, confound)
    scores: list[float] = []
    for train_index, test_index in splitter.split(features, strata):
        train_x, test_x = residualize_train_test(
            features[train_index],
            features[test_index],
            confound[train_index],
            confound[test_index],
            confound_categories,
        )
        scaler = StandardScaler().fit(train_x)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(scaler.transform(train_x), target[train_index])
        prediction = model.predict(scaler.transform(test_x))
        scores.append(float(balanced_accuracy_score(target[test_index], prediction)))
    return np.asarray(scores, dtype=np.float64)


def stratified_permutation(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = values.copy()
    for group in np.unique(strata):
        indices = np.flatnonzero(strata == group)
        output[indices] = rng.permutation(output[indices])
    return output


def task_summary(
    features: np.ndarray,
    target: np.ndarray,
    confound: np.ndarray,
    confound_categories: list[str],
    chance: float,
    seed: int,
    repeats: int,
    permutations: int,
) -> dict[str, Any]:
    scores = cross_validated_scores(
        features, target, confound, confound_categories, seed=seed, repeats=repeats
    )
    repeat_means = scores.reshape(repeats, 5).mean(axis=1)
    observed = float(scores.mean())
    rng = np.random.default_rng(seed + 91)
    null_scores: list[float] = []
    for permutation_index in range(permutations):
        permuted = stratified_permutation(target, confound, rng)
        null = cross_validated_scores(
            features,
            permuted,
            confound,
            confound_categories,
            seed=seed + 1000 + permutation_index,
            repeats=1,
        )
        null_scores.append(float(null.mean()))
    null_array = np.asarray(null_scores, dtype=np.float64)
    return {
        "balanced_accuracy": observed,
        "repeat_mean_ci95": [
            float(np.quantile(repeat_means, 0.025)),
            float(np.quantile(repeat_means, 0.975)),
        ],
        "fold_score_sd": float(scores.std(ddof=1)),
        "chance": chance,
        "normalized_above_chance": float((observed - chance) / (1.0 - chance)),
        "permutation_p_one_sided": float((1 + np.sum(null_array >= observed)) / (permutations + 1)),
        "permutation_null_mean": float(null_array.mean()),
        "permutation_null_ci95": [
            float(np.quantile(null_array, 0.025)),
            float(np.quantile(null_array, 0.975)),
        ],
        "cv_repeats": repeats,
        "permutations": permutations,
    }


def residual_sum_squares(design: np.ndarray, features: np.ndarray) -> np.ndarray:
    coefficients, _, _, _ = np.linalg.lstsq(design, features.astype(np.float64), rcond=None)
    residual = features - design @ coefficients
    return np.square(residual, dtype=np.float64).sum(axis=0)


def feature_effects(
    features: np.ndarray,
    feature_names: list[str],
    sources: np.ndarray,
    risks: np.ndarray,
) -> pd.DataFrame:
    source_categories = sorted(np.unique(sources).astype(str).tolist())
    source_design = design_matrix(sources, source_categories)
    risk_column = risks.astype(np.float64).reshape(-1, 1)
    full_design = np.column_stack([source_design, risk_column])
    risk_only_design = np.column_stack([np.ones(len(risks)), risk_column])
    source_only_design = source_design
    full_rss = residual_sum_squares(full_design, features)
    no_source_rss = residual_sum_squares(risk_only_design, features)
    no_risk_rss = residual_sum_squares(source_only_design, features)
    source_increment = np.maximum(0.0, no_source_rss - full_rss)
    risk_increment = np.maximum(0.0, no_risk_rss - full_rss)
    source_eta = source_increment / np.maximum(source_increment + full_rss, 1e-12)
    risk_eta = risk_increment / np.maximum(risk_increment + full_rss, 1e-12)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "partial_eta2_source_controlling_risk": source_eta,
            "partial_eta2_risk_controlling_source": risk_eta,
            "source_minus_risk_partial_eta2": source_eta - risk_eta,
        }
    ).sort_values("source_minus_risk_partial_eta2", ascending=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(Path(args.registry))
    features, feature_names = extract_features(
        registry, image_size=int(args.image_size), project_root=Path(args.project_root)
    )
    np.save(output_dir / "frequency_features.float32.npy", features)
    registry[["case_id", "source", "risk"]].to_csv(
        output_dir / "frequency_feature_metadata.csv", index=False, encoding="utf-8-sig"
    )

    sources = registry["source"].to_numpy(dtype=str)
    risks = registry["risk"].to_numpy(dtype=int)
    effects = feature_effects(features, feature_names, sources, risks)
    effects.to_csv(output_dir / "frequency_feature_effects.csv", index=False)

    source_summary = task_summary(
        features,
        target=sources,
        confound=risks.astype(str),
        confound_categories=["0", "1"],
        chance=1.0 / 3.0,
        seed=int(args.seed),
        repeats=int(args.cv_repeats),
        permutations=int(args.permutations),
    )
    risk_summary = task_summary(
        features,
        target=risks,
        confound=sources,
        confound_categories=sorted(np.unique(sources).tolist()),
        chance=0.5,
        seed=int(args.seed) + 17,
        repeats=int(args.cv_repeats),
        permutations=int(args.permutations),
    )
    source_eta = effects["partial_eta2_source_controlling_risk"].to_numpy()
    risk_eta = effects["partial_eta2_risk_controlling_source"].to_numpy()
    normalized_gap = float(
        source_summary["normalized_above_chance"] - risk_summary["normalized_above_chance"]
    )
    source_dominant = bool(normalized_gap >= 0.10 and np.median(source_eta) > np.median(risk_eta))
    result = {
        "audit_role": "diagnostic_only_not_a_model_nomination",
        "cases": int(len(registry)),
        "features": int(features.shape[1]),
        "views": list(VIEWS),
        "channels": list(CHANNELS),
        "haar_levels": N_LEVELS,
        "image_size": int(args.image_size),
        "source_counts": registry["source"].value_counts().sort_index().astype(int).to_dict(),
        "risk_counts": {
            str(key): int(value) for key, value in registry["risk"].value_counts().sort_index().items()
        },
        "source_prediction_controlling_risk": source_summary,
        "risk_prediction_controlling_source": risk_summary,
        "normalized_source_minus_risk_gap": normalized_gap,
        "partial_eta2": {
            "source_median": float(np.median(source_eta)),
            "source_p90": float(np.quantile(source_eta, 0.90)),
            "source_maximum": float(np.max(source_eta)),
            "risk_median": float(np.median(risk_eta)),
            "risk_p90": float(np.quantile(risk_eta, 0.90)),
            "risk_maximum": float(np.max(risk_eta)),
            "features_source_eta_greater_than_risk": int(np.sum(source_eta > risk_eta)),
        },
        "preregistered_source_dominance_rule": {
            "normalized_gap_at_least": 0.10,
            "source_median_eta_greater_than_risk_median_eta": True,
            "rule_met": source_dominant,
        },
        "privacy": "case-level features and metadata remain server-side",
    }
    write_json(output_dir / "frequency_source_vs_risk_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
