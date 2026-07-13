from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score


EXPECTED_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
EXPECTED_SOURCES = ("batch1", "batch2", "third_batch")
EXPECTED_SUBTYPES = {"A": 44, "AB": 262, "B1": 62, "B2": 89, "B3": 24, "TC": 110}
LOW_RISK = {"A", "AB", "B1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare immutable assets for the locked Task7 spatial-relational experiment."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--c1-oof", required=True)
    parser.add_argument("--c1-lodo", required=True)
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-file", action="append", default=[])
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def canonical_source(value: object) -> str:
    text = str(value)
    if text.startswith("third_batch"):
        return "third_batch"
    return text


def normalize_label(frame: pd.DataFrame) -> pd.Series:
    column = "label_idx" if "label_idx" in frame.columns else "risk_label"
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any():
        mapped = frame[column].astype(str).str.lower().map(
            {
                "low": 0,
                "low_risk_group": 0,
                "high": 1,
                "high_risk_group": 1,
            }
        )
        numeric = numeric.fillna(mapped)
    if numeric.isna().any():
        raise ValueError(f"Could not normalize labels from column {column}")
    return numeric.astype(int)


def load_metadata(feature_bank_dir: Path, split_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        feature_bank_dir / "metadata.csv",
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    metadata.columns = [str(column).lstrip("\ufeff") for column in metadata.columns]
    split = pd.read_csv(split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split.columns = [str(column).lstrip("\ufeff") for column in split.columns]
    split = split[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    authoritative = metadata[["case_id"]].merge(split, on="case_id", how="left")["master_fold_id"]
    fallback = pd.to_numeric(metadata.get("master_fold_id"), errors="coerce")
    metadata["master_fold_id"] = pd.to_numeric(authoritative, errors="coerce").fillna(fallback)
    metadata["source_dataset_raw"] = metadata["source_dataset"].astype(str)
    metadata["source_dataset"] = metadata["source_dataset_raw"].map(canonical_source)
    metadata["task_l6_label"] = metadata["task_l6_label"].astype(str)
    metadata["label_idx"] = (~metadata["task_l6_label"].isin(LOW_RISK)).astype(int)
    metadata["feature_row"] = np.arange(len(metadata), dtype=int)

    if len(metadata) != 591:
        raise ValueError(f"Expected 591 metadata rows, found {len(metadata)}")
    if metadata["case_id"].isna().any() or metadata["case_id"].duplicated().any():
        raise ValueError("case_id must be complete and unique")
    if "original_case_id" in metadata.columns:
        original = metadata["original_case_id"].dropna().astype(str)
        if original.duplicated().any():
            duplicates = original[original.duplicated(keep=False)].unique().tolist()
            raise ValueError(f"Duplicated original_case_id values: {duplicates[:10]}")
    if metadata["master_fold_id"].isna().any():
        raise ValueError("Missing master_fold_id values")
    metadata["master_fold_id"] = metadata["master_fold_id"].astype(int)
    if sorted(metadata["master_fold_id"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("Expected master folds 1-5")
    if tuple(sorted(metadata["source_dataset"].unique().tolist())) != tuple(sorted(EXPECTED_SOURCES)):
        raise ValueError(f"Unexpected canonical sources: {metadata['source_dataset'].unique().tolist()}")
    subtype_counts = metadata["task_l6_label"].value_counts().to_dict()
    if subtype_counts != EXPECTED_SUBTYPES:
        raise ValueError(f"Unexpected subtype totals: {subtype_counts}")
    return metadata


def load_prediction(path: Path, metadata: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    probability_column = "prob_high" if "prob_high" in frame.columns else "probability"
    required = {"case_id", probability_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing {sorted(required - set(frame.columns))}")
    if frame["case_id"].duplicated().any() or len(frame) != len(metadata):
        raise ValueError(f"{name} prediction IDs are not unique and complete")
    result = frame[["case_id", probability_column]].copy()
    result["prob_high"] = pd.to_numeric(result.pop(probability_column), errors="raise")
    if "label_idx" in frame.columns or "risk_label" in frame.columns:
        result["prediction_label"] = normalize_label(frame)
    aligned = metadata[
        ["case_id", "label_idx", "source_dataset", "task_l6_label", "master_fold_id"]
    ].merge(result, on="case_id", how="left", validate="one_to_one")
    if aligned["prob_high"].isna().any():
        raise ValueError(f"{name} predictions do not align to all metadata cases")
    if "prediction_label" in aligned.columns and not np.array_equal(
        aligned["label_idx"].to_numpy(), aligned["prediction_label"].to_numpy()
    ):
        raise ValueError(f"{name} labels disagree with feature metadata")
    aligned["model"] = name
    return aligned


def metric_record(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    prediction = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        "auc": float(roc_auc_score(y, p)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def baseline_audit(predictions: dict[str, dict[str, pd.DataFrame]], output_dir: Path) -> None:
    overall_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []
    for protocol, models in predictions.items():
        for model, frame in models.items():
            overall_rows.append({"protocol": protocol, "model": model, **metric_record(frame["label_idx"], frame["prob_high"])})
            for source, group in frame.groupby("source_dataset", sort=False):
                source_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "source_dataset": source,
                        **metric_record(group["label_idx"], group["prob_high"]),
                    }
                )
            for subtype, group in frame.groupby("task_l6_label", sort=False):
                predicted = (group["prob_high"].to_numpy(dtype=float) >= 0.5).astype(int)
                subtype_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "subtype": subtype,
                        "n": int(len(group)),
                        "risk_accuracy": float(np.mean(predicted == group["label_idx"].to_numpy(dtype=int))),
                        "predicted_high_n": int(predicted.sum()),
                    }
                )
    pd.DataFrame(overall_rows).to_csv(output_dir / "baseline_overall_metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / "baseline_source_metrics.csv", index=False)
    pd.DataFrame(subtype_rows).to_csv(output_dir / "baseline_subtype_metrics.csv", index=False)


def compute_view_bounds(
    image: Image.Image,
    detect_specimen_bbox: Any,
    expand_bbox: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    detected = detect_specimen_bbox(rgb)
    crop = expand_bbox(detected, rgb.shape[:2], margin_ratio=0.12)
    tight = expand_bbox(detected, rgb.shape[:2], margin_ratio=0.02)
    tx1, ty1, tx2, ty2 = tight
    tight_width = tx2 - tx1
    tight_height = ty2 - ty1
    tile_width = max(1, int(round(tight_width * 0.70)))
    tile_height = max(1, int(round(tight_height * 0.70)))
    pixel_bounds = [(0, 0, width, height), crop]
    for quadrant in range(4):
        start_y = ty1 if quadrant < 2 else ty1 + max(0, tight_height - tile_height)
        start_x = tx1 if quadrant % 2 == 0 else tx1 + max(0, tight_width - tile_width)
        pixel_bounds.append((start_x, start_y, start_x + tile_width, start_y + tile_height))
    normalized = np.asarray(
        [
            [x1 / width, y1 / height, x2 / width, y2 / height]
            for x1, y1, x2, y2 in pixel_bounds
        ],
        dtype=np.float32,
    )
    if normalized.shape != (6, 4) or not np.isfinite(normalized).all():
        raise ValueError("Invalid normalized view bounds")
    if (normalized < 0).any() or (normalized > 1).any():
        raise ValueError("View bounds fall outside the original image")
    if (normalized[:, 2] <= normalized[:, 0]).any() or (normalized[:, 3] <= normalized[:, 1]).any():
        raise ValueError("View bounds have non-positive extent")
    details = {
        "image_width": int(width),
        "image_height": int(height),
        "detected_x1": int(detected[0]),
        "detected_y1": int(detected[1]),
        "detected_x2": int(detected[2]),
        "detected_y2": int(detected[3]),
    }
    return normalized, details


def prepare_coordinates(metadata: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, pd.DataFrame]:
    project_root = Path("/workspace/thymic_project")
    for item in (project_root, project_root / "scripts"):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    from thymic_baseline.cropping import detect_specimen_bbox, expand_bbox

    view_bounds = np.zeros((len(metadata), 6, 4), dtype=np.float32)
    coordinate_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    for row_index, row in metadata.iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            bounds, details = compute_view_bounds(image, detect_specimen_bbox, expand_bbox)
        view_bounds[row_index] = bounds
        coordinate_row: dict[str, Any] = {
            "feature_row": int(row_index),
            "case_id": str(row["case_id"]),
            **details,
        }
        for view_index, view_name in enumerate(EXPECTED_VIEWS):
            for coordinate_index, coordinate_name in enumerate(("x1", "y1", "x2", "y2")):
                coordinate_row[f"{view_name}_{coordinate_name}"] = float(
                    bounds[view_index, coordinate_index]
                )
        coordinate_rows.append(coordinate_row)
        resolution_rows.append(
            {
                "source_dataset": row["source_dataset"],
                "task_l6_label": row["task_l6_label"],
                "width": details["image_width"],
                "height": details["image_height"],
                "long_edge": max(details["image_width"], details["image_height"]),
                "short_edge": min(details["image_width"], details["image_height"]),
                "aspect_ratio": details["image_width"] / max(details["image_height"], 1),
                "image_count": pd.to_numeric(row.get("image_count"), errors="coerce"),
                "original_image_count": pd.to_numeric(row.get("original_image_count"), errors="coerce"),
            }
        )
    bounds_path = output_dir / "view_bounds.float32.npy"
    coordinates_path = output_dir / "coordinate_metadata.csv"
    np.save(bounds_path, view_bounds, allow_pickle=False)
    pd.DataFrame(coordinate_rows).to_csv(coordinates_path, index=False)
    return bounds_path, coordinates_path, pd.DataFrame(resolution_rows)


def summarize_resolutions(frame: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for group_columns in (("source_dataset",), ("source_dataset", "task_l6_label")):
        for keys, group in frame.groupby(list(group_columns), sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {column: value for column, value in zip(group_columns, keys)}
            row.update(
                {
                    "n": int(len(group)),
                    "width_min": int(group["width"].min()),
                    "width_median": float(group["width"].median()),
                    "width_max": int(group["width"].max()),
                    "height_min": int(group["height"].min()),
                    "height_median": float(group["height"].median()),
                    "height_max": int(group["height"].max()),
                    "short_edge_median": float(group["short_edge"].median()),
                    "aspect_ratio_median": float(group["aspect_ratio"].median()),
                    "multi_image_n": int((group["original_image_count"].fillna(1) > 1).sum()),
                }
            )
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "image_resolution_summary.csv", index=False)


def environment_lock() -> str:
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    header = [
        f"python=={platform.python_version()}",
        f"platform=={platform.platform()}",
    ]
    return "\n".join(header + packages) + "\n"


def main() -> None:
    args = parse_args()
    feature_bank_dir = Path(args.feature_bank_dir)
    split_csv = Path(args.split_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = feature_bank_dir / "feature_bank_config.json"
    feature_path = feature_bank_dir / "dense_features.float16.npy"
    feature_config = json.loads(config_path.read_text(encoding="utf-8"))
    if feature_config.get("complete") is not True:
        raise ValueError("Feature bank is not complete")
    if tuple(feature_config.get("views", [])) != EXPECTED_VIEWS:
        raise ValueError(f"Unexpected view order: {feature_config.get('views')}")
    if feature_config.get("feature_shape") != [591, 6, 1024, 1024]:
        raise ValueError(f"Unexpected feature shape: {feature_config.get('feature_shape')}")
    features = np.load(feature_path, mmap_mode="r")
    if features.shape != (591, 6, 1024, 1024) or features.dtype != np.float16:
        raise ValueError(f"Feature array mismatch: {features.shape} {features.dtype}")
    del features

    metadata = load_metadata(feature_bank_dir, split_csv)
    source_subtype = pd.crosstab(metadata["source_dataset"], metadata["task_l6_label"]).reindex(
        index=EXPECTED_SOURCES, columns=EXPECTED_SUBTYPES, fill_value=0
    )
    source_risk = pd.crosstab(metadata["source_dataset"], metadata["label_idx"]).reindex(
        index=EXPECTED_SOURCES, columns=[0, 1], fill_value=0
    )
    source_subtype.to_csv(output_dir / "source_subtype_counts.csv")
    source_risk.rename(columns={0: "low", 1: "high"}).to_csv(output_dir / "source_risk_counts.csv")

    prediction_paths = {
        "oof": {"C1": Path(args.c1_oof), "C2": Path(args.c2_oof)},
        "lodo": {"C1": Path(args.c1_lodo), "C2": Path(args.c2_lodo)},
    }
    predictions = {
        protocol: {
            model: load_prediction(path, metadata, f"{protocol}-{model}")
            for model, path in models.items()
        }
        for protocol, models in prediction_paths.items()
    }
    for protocol, models in predictions.items():
        if not np.array_equal(models["C1"]["case_id"].to_numpy(), models["C2"]["case_id"].to_numpy()):
            raise ValueError(f"C1/C2 case order mismatch under {protocol}")
        if not np.array_equal(models["C1"]["label_idx"].to_numpy(), models["C2"]["label_idx"].to_numpy()):
            raise ValueError(f"C1/C2 label mismatch under {protocol}")
    baseline_audit(predictions, output_dir)

    bounds_path, coordinates_path, resolution_frame = prepare_coordinates(metadata, output_dir)
    summarize_resolutions(resolution_frame, output_dir)

    environment_path = output_dir / "environment_lock.txt"
    environment_path.write_text(environment_lock(), encoding="utf-8")
    asset_paths = {
        "dense_features": feature_path,
        "feature_config": config_path,
        "metadata": feature_bank_dir / "metadata.csv",
        "split_csv": split_csv,
        "c1_oof": Path(args.c1_oof),
        "c1_lodo": Path(args.c1_lodo),
        "c2_oof": Path(args.c2_oof),
        "c2_lodo": Path(args.c2_lodo),
        "view_bounds": bounds_path,
        "coordinate_metadata": coordinates_path,
        "environment_lock": environment_path,
    }
    for index, code_file in enumerate(args.code_file):
        asset_paths[f"code_{index}"] = Path(code_file)
    manifest = {
        "protocol": "H2_CANONICAL_SPATIAL_RELATIONAL_20260713",
        "complete": True,
        "case_count": int(len(metadata)),
        "case_id_order_sha256": hashlib.sha256(
            "\n".join(metadata["case_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "source_counts": metadata["source_dataset"].value_counts().sort_index().to_dict(),
        "subtype_counts": metadata["task_l6_label"].value_counts().to_dict(),
        "risk_counts": metadata["label_idx"].value_counts().sort_index().to_dict(),
        "feature_shape": [591, 6, 1024, 1024],
        "views": list(EXPECTED_VIEWS),
        "assets": {name: file_record(path) for name, path in asset_paths.items()},
    }
    write_json(output_dir / "integrity_manifest.json", manifest)
    write_json(
        output_dir / "coordinate_manifest_config.json",
        {
            "complete": True,
            "shape": [591, 6, 4],
            "dtype": "float32",
            "view_order": list(EXPECTED_VIEWS),
            "whole_bounds": "complete original image",
            "crop_margin_ratio": 0.12,
            "quadrant_bbox_margin_ratio": 0.02,
            "quadrant_fraction": 0.70,
            "bounds_sha256": manifest["assets"]["view_bounds"]["sha256"],
            "coordinate_metadata_sha256": manifest["assets"]["coordinate_metadata"]["sha256"],
        },
    )
    (output_dir / "PREPARATION.status").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        output = None
        try:
            parsed = parse_args()
            output = Path(parsed.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "PREPARATION.status").write_text(
                f"failed: {type(error).__name__}: {error}\n", encoding="utf-8"
            )
        finally:
            raise
