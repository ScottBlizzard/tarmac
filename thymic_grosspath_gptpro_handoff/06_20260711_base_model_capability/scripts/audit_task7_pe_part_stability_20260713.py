from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


VARIANTS = ("photometric", "blur_jpeg", "horizontal_flip")
N_CLUSTERS = 3
PROJECTION_DIM = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit label-free PE-Spatial part-map stability without training a classifier."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--project-root", default="/workspace/thymic_project")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--model-code-dir", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-num-patches", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
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
        raise ValueError(f"Expected 591 internal cases, found {len(frame)}")
    frame["source"] = frame["source_dataset"].map(
        lambda value: "third_batch" if str(value).startswith("third_batch") else str(value)
    )
    frame["risk"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    if set(frame["source"]) != {"batch1", "batch2", "third_batch"}:
        raise ValueError("Unexpected source labels")
    if set(frame["risk"]) != {0, 1}:
        raise ValueError("Unexpected Task7 risk labels")
    if any(not Path(str(value)).is_file() for value in frame["image_path"]):
        raise FileNotFoundError("At least one selected image is missing")
    return frame.reset_index(drop=True)


def make_variant(image: Image.Image, name: str) -> Image.Image:
    if name == "photometric":
        output = ImageEnhance.Brightness(image).enhance(1.08)
        output = ImageEnhance.Contrast(output).enhance(1.10)
        return ImageEnhance.Color(output).enhance(0.90)
    if name == "blur_jpeg":
        output = image.filter(ImageFilter.GaussianBlur(radius=0.8))
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=85, subsampling=2, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    if name == "horizontal_flip":
        return ImageOps.mirror(image)
    raise ValueError(f"Unknown variant: {name}")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def project_tokens(tokens: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return normalize_rows(normalize_rows(tokens.astype(np.float32)) @ projection)


def assign_centroids(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = (
        np.square(values, dtype=np.float64).sum(axis=1, keepdims=True)
        - 2.0 * values @ centroids.T
        + np.square(centroids, dtype=np.float64).sum(axis=1)[None, :]
    )
    return np.argmin(distances, axis=1).astype(np.int16)


def cluster_entropy(labels: np.ndarray) -> tuple[float, float]:
    proportions = np.bincount(labels, minlength=N_CLUSTERS).astype(np.float64)
    proportions /= proportions.sum()
    nonzero = proportions[proportions > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(N_CLUSTERS))
    return entropy, float(proportions.max())


def specimen_grid_mask(image: Image.Image, rows: int, columns: int, project_root: Path) -> np.ndarray:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from thymic_baseline.cropping import detect_specimen_mask

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = (detect_specimen_mask(rgb) > 0).astype(np.uint8) * 255
    resized = Image.fromarray(mask, mode="L").resize(
        (columns, rows), resample=Image.Resampling.NEAREST
    )
    return (np.asarray(resized, dtype=np.uint8).reshape(-1) > 0).astype(np.int8)


def categorical_design(values: np.ndarray, categories: list[str]) -> np.ndarray:
    text = values.astype(str)
    columns = [np.ones(len(text), dtype=np.float64)]
    columns.extend((text == category).astype(np.float64) for category in categories[1:])
    return np.column_stack(columns)


def residual_sum_squares(design: np.ndarray, values: np.ndarray) -> float:
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    return float(np.square(residual, dtype=np.float64).sum())


def partial_eta(values: np.ndarray, sources: np.ndarray, risks: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(values)
    values = values[valid].astype(np.float64)
    sources = sources[valid]
    risks = risks[valid]
    source_categories = sorted(np.unique(sources).astype(str).tolist())
    source_design = categorical_design(sources, source_categories)
    risk_column = risks.astype(np.float64).reshape(-1, 1)
    full = np.column_stack([source_design, risk_column])
    risk_only = np.column_stack([np.ones(len(values)), risk_column])
    full_rss = residual_sum_squares(full, values)
    no_source_rss = residual_sum_squares(risk_only, values)
    no_risk_rss = residual_sum_squares(source_design, values)
    source_increment = max(0.0, no_source_rss - full_rss)
    risk_increment = max(0.0, no_risk_rss - full_rss)
    source_eta = source_increment / max(source_increment + full_rss, 1e-12)
    risk_eta = risk_increment / max(risk_increment + full_rss, 1e-12)
    return float(source_eta), float(risk_eta)


def metric_summary(values: pd.Series) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    scripts_dir = project_root / "scripts"
    for path in (project_root, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from extract_task7_h3_dense_bank_20260713 import PeAdapter

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    adapter_args = SimpleNamespace(
        model_id=args.model_checkpoint,
        model_code_dir=args.model_code_dir,
        code_revision=args.code_revision,
        max_num_patches=int(args.max_num_patches),
    )
    adapter = PeAdapter(adapter_args, device)
    rng = np.random.default_rng(int(args.seed))
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(PROJECTION_DIM),
        size=(int(adapter.feature_dim), PROJECTION_DIM),
    ).astype(np.float32)

    registry = load_registry(Path(args.registry))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row_index, row in registry.iterrows():
        with Image.open(str(row["image_path"])) as source_image:
            image = source_image.convert("RGB")
        clean = adapter.extract(image)
        clean_tokens = clean.tokens.numpy().astype(np.float32)
        rows_count, columns_count = clean.spatial_shape
        if len(clean_tokens) != rows_count * columns_count:
            raise ValueError("PE token count does not match its spatial grid")
        clean_projected = project_tokens(clean_tokens, projection)
        clustering = KMeans(
            n_clusters=N_CLUSTERS,
            n_init=10,
            max_iter=100,
            random_state=int(args.seed) + row_index,
        ).fit(clean_projected)
        clean_labels = clustering.labels_.astype(np.int16)
        entropy, largest_fraction = cluster_entropy(clean_labels)
        specimen_mask = specimen_grid_mask(
            image, rows_count, columns_count, project_root=project_root
        )
        specimen_fraction = float(specimen_mask.mean())
        specimen_nmi = (
            float(normalized_mutual_info_score(specimen_mask, clean_labels))
            if 0.0 < specimen_fraction < 1.0
            else float("nan")
        )
        result_row: dict[str, Any] = {
            "case_id": str(row["case_id"]),
            "source": str(row["source"]),
            "risk": int(row["risk"]),
            "token_rows": int(rows_count),
            "token_columns": int(columns_count),
            "cluster_entropy_normalized": entropy,
            "largest_cluster_fraction": largest_fraction,
            "specimen_mask_fraction": specimen_fraction,
            "cluster_specimen_mask_nmi": specimen_nmi,
        }
        clean_normalized = normalize_rows(clean_tokens)
        for variant_name in VARIANTS:
            variant = adapter.extract(make_variant(image, variant_name))
            if variant.spatial_shape != clean.spatial_shape:
                raise ValueError(f"Variant grid changed for {variant_name}")
            variant_tokens = variant.tokens.numpy().astype(np.float32)
            if variant_name == "horizontal_flip":
                variant_tokens = variant_tokens.reshape(
                    rows_count, columns_count, -1
                )[:, ::-1, :].reshape(len(clean_tokens), -1)
            variant_normalized = normalize_rows(variant_tokens)
            variant_projected = project_tokens(variant_tokens, projection)
            variant_labels = assign_centroids(variant_projected, clustering.cluster_centers_)
            clean_occupancy = np.bincount(clean_labels, minlength=N_CLUSTERS) / len(clean_labels)
            variant_occupancy = np.bincount(variant_labels, minlength=N_CLUSTERS) / len(
                variant_labels
            )
            prefix = f"{variant_name}__"
            result_row[f"{prefix}label_agreement"] = float(
                np.mean(clean_labels == variant_labels)
            )
            result_row[f"{prefix}adjusted_rand"] = float(
                adjusted_rand_score(clean_labels, variant_labels)
            )
            result_row[f"{prefix}mean_token_cosine"] = float(
                np.mean(np.sum(clean_normalized * variant_normalized, axis=1))
            )
            result_row[f"{prefix}occupancy_total_variation"] = float(
                0.5 * np.abs(clean_occupancy - variant_occupancy).sum()
            )
        rows.append(result_row)
        if (row_index + 1) % 25 == 0 or row_index + 1 == len(registry):
            print(f"[part-audit] {row_index + 1}/{len(registry)}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "pe_part_stability_case_metrics.csv", index=False, encoding="utf-8-sig")
    metric_columns = [
        column
        for column in metrics.columns
        if column not in {"case_id", "source", "risk", "token_rows", "token_columns"}
    ]
    effects: list[dict[str, Any]] = []
    sources = metrics["source"].to_numpy(dtype=str)
    risks = metrics["risk"].to_numpy(dtype=int)
    for metric in metric_columns:
        source_eta, risk_eta = partial_eta(
            pd.to_numeric(metrics[metric], errors="coerce").to_numpy(dtype=np.float64),
            sources,
            risks,
        )
        effects.append(
            {
                "metric": metric,
                "partial_eta2_source_controlling_risk": source_eta,
                "partial_eta2_risk_controlling_source": risk_eta,
                "source_minus_risk_partial_eta2": source_eta - risk_eta,
            }
        )
    effects_frame = pd.DataFrame(effects).sort_values(
        "source_minus_risk_partial_eta2", ascending=False
    )
    effects_frame.to_csv(output_dir / "pe_part_stability_effects.csv", index=False)

    overall = {metric: metric_summary(metrics[metric]) for metric in metric_columns}
    by_source: dict[str, Any] = {}
    for source, group in metrics.groupby("source", sort=True):
        by_source[source] = {
            metric: float(pd.to_numeric(group[metric], errors="coerce").mean())
            for metric in metric_columns
        }

    median_aris = [overall[f"{variant}__adjusted_rand"]["median"] for variant in VARIANTS]
    median_cosines = [
        overall[f"{variant}__mean_token_cosine"]["median"] for variant in VARIANTS
    ]
    stable = bool(min(median_aris) >= 0.70 and min(median_cosines) >= 0.85)
    nondegenerate = bool(
        overall["cluster_entropy_normalized"]["median"] >= 0.60
        and overall["largest_cluster_fraction"]["median"] <= 0.80
    )
    background_dominated = bool(overall["cluster_specimen_mask_nmi"]["median"] >= 0.50)
    stability_metric_names = [
        f"{variant}__{suffix}"
        for variant in VARIANTS
        for suffix in ("adjusted_rand", "mean_token_cosine", "occupancy_total_variation")
    ]
    stability_effects = effects_frame[effects_frame["metric"].isin(stability_metric_names)]
    source_sensitive = bool(
        stability_effects["partial_eta2_source_controlling_risk"].median()
        > stability_effects["partial_eta2_risk_controlling_source"].median()
        and stability_effects["partial_eta2_source_controlling_risk"].max() >= 0.05
    )
    result = {
        "audit_role": "label_free_diagnostic_only_not_a_model_nomination",
        "cases": int(len(metrics)),
        "encoder": "facebook/PE-Spatial-L14-448 aligned final dense layer",
        "clusters_per_image": N_CLUSTERS,
        "random_projection_dim": PROJECTION_DIM,
        "variants": list(VARIANTS),
        "overall": overall,
        "by_source_mean": by_source,
        "decision_rules": {
            "stable_if_worst_median_ari_at_least": 0.70,
            "stable_if_worst_median_cosine_at_least": 0.85,
            "nondegenerate_if_median_entropy_at_least": 0.60,
            "nondegenerate_if_median_largest_cluster_at_most": 0.80,
            "background_dominated_if_median_nmi_at_least": 0.50,
            "source_sensitive_if_source_eta_dominates_and_max_at_least": 0.05,
        },
        "decisions": {
            "stable": stable,
            "nondegenerate": nondegenerate,
            "background_dominated": background_dominated,
            "source_sensitive": source_sensitive,
            "supports_blinded_part_annotation_followup": bool(
                stable and nondegenerate and not background_dominated and not source_sensitive
            ),
            "authorizes_classifier": False,
        },
        "privacy": "case-level metrics remain server-side; no dense token bank retained",
    }
    write_json(output_dir / "pe_part_stability_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
