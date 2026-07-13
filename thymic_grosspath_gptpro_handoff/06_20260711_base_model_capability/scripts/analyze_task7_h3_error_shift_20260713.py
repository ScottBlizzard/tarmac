from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import fisher_exact, mannwhitneyu

from analyze_dense_errors_with_doctor_concepts_20260711 import (
    CONCEPTS,
    benjamini_hochberg,
    odds_ratio,
)


QUALITY_COLUMNS = [
    "image_width",
    "image_height",
    "image_megapixels",
    "image_abs_log_aspect",
    "gray_mean",
    "gray_std",
    "saturation_mean",
    "dark_fraction",
    "bright_fraction",
    "near_white_fraction",
    "near_black_fraction",
    "gray_entropy",
    "gradient_mean",
    "laplacian_variance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit H3 candidate error shifts using server-side concepts and image quality."
    )
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--candidate-lodo", required=True)
    parser.add_argument("--candidate-name", default="pe_spatial_l14_448")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument(
        "--concept-csv",
        default="/workspace/thymic_project/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_prediction(path: Path, require_original_case_id: bool) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {
        "case_id",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        "prob_high",
    }
    if require_original_case_id:
        required.add("original_case_id")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if len(frame) != 591 or frame["case_id"].nunique() != 591:
        raise ValueError(f"Expected 591 unique cases in {path}")
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["prob_high"] = pd.to_numeric(frame["prob_high"], errors="raise")
    frame["pred_idx"] = (frame["prob_high"] >= 0.5).astype(int)
    return frame


def image_quality(path: str) -> dict[str, float]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        image.thumbnail((256, 256), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = (
        rgb[..., 0] * 0.299
        + rgb[..., 1] * 0.587
        + rgb[..., 2] * 0.114
    )
    histogram = np.histogram(gray, bins=64, range=(0.0, 1.0))[0].astype(float)
    histogram /= max(histogram.sum(), 1.0)
    nonzero = histogram[histogram > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    gradient = (
        np.abs(np.diff(gray, axis=0)).mean()
        + np.abs(np.diff(gray, axis=1)).mean()
    ) / 2.0
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * gray[1:-1, 1:-1]
        )
        laplacian_variance = float(laplacian.var())
    else:
        laplacian_variance = float("nan")
    near_white = np.all(rgb >= 0.94, axis=-1)
    near_black = np.all(rgb <= 0.04, axis=-1)
    return {
        "image_width": float(width),
        "image_height": float(height),
        "image_megapixels": float(width * height / 1_000_000.0),
        "image_abs_log_aspect": float(abs(math.log(max(width, 1) / max(height, 1)))),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "dark_fraction": float((gray <= 0.05).mean()),
        "bright_fraction": float((gray >= 0.95).mean()),
        "near_white_fraction": float(near_white.mean()),
        "near_black_fraction": float(near_black.mean()),
        "gray_entropy": entropy,
        "gradient_mean": float(gradient),
        "laplacian_variance": laplacian_variance,
    }


def error_type(label: pd.Series, prediction: pd.Series) -> np.ndarray:
    return np.select(
        [
            label.eq(0) & prediction.eq(0),
            label.eq(0) & prediction.eq(1),
            label.eq(1) & prediction.eq(0),
            label.eq(1) & prediction.eq(1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="unknown",
    )


def comparison_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series, pd.Series]]:
    return [
        (
            "candidate_low_fp_vs_tn",
            frame["candidate_error_type"].eq("FP"),
            frame["candidate_error_type"].eq("TN"),
        ),
        (
            "candidate_high_fn_vs_tp",
            frame["candidate_error_type"].eq("FN"),
            frame["candidate_error_type"].eq("TP"),
        ),
        (
            "candidate_batch1_high_fn_vs_tp",
            frame["source_dataset"].eq("batch1")
            & frame["candidate_error_type"].eq("FN"),
            frame["source_dataset"].eq("batch1")
            & frame["candidate_error_type"].eq("TP"),
        ),
        (
            "candidate_b2_fn_vs_tp",
            frame["task_l6_label"].eq("B2")
            & frame["candidate_error_type"].eq("FN"),
            frame["task_l6_label"].eq("B2")
            & frame["candidate_error_type"].eq("TP"),
        ),
        (
            "candidate_harm_vs_stable_correct",
            frame["c2_correct"] & ~frame["candidate_correct"],
            frame["c2_correct"] & frame["candidate_correct"],
        ),
        (
            "candidate_high_harm_vs_stable_correct",
            frame["label_idx"].eq(1)
            & frame["c2_correct"]
            & ~frame["candidate_correct"],
            frame["label_idx"].eq(1)
            & frame["c2_correct"]
            & frame["candidate_correct"],
        ),
        (
            "candidate_rescue_vs_persistent_error",
            ~frame["c2_correct"] & frame["candidate_correct"],
            ~frame["c2_correct"] & ~frame["candidate_correct"],
        ),
    ]


def concept_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, group_a_mask, group_b_mask in comparison_masks(frame):
        eligible = frame["concept_has_gross_text"].eq(1)
        group_a = frame[group_a_mask & eligible]
        group_b = frame[group_b_mask & eligible]
        for concept in CONCEPTS:
            a_positive = int(pd.to_numeric(group_a[concept], errors="coerce").fillna(0).gt(0).sum())
            b_positive = int(pd.to_numeric(group_b[concept], errors="coerce").fillna(0).gt(0).sum())
            a_negative = len(group_a) - a_positive
            b_negative = len(group_b) - b_positive
            p_value = (
                float(
                    fisher_exact(
                        [[a_positive, a_negative], [b_positive, b_negative]],
                        alternative="two-sided",
                    ).pvalue
                )
                if len(group_a) and len(group_b)
                else float("nan")
            )
            rows.append(
                {
                    "comparison": comparison,
                    "feature": concept,
                    "group_a_n": int(len(group_a)),
                    "group_b_n": int(len(group_b)),
                    "group_a_prevalence": a_positive / len(group_a) if len(group_a) else np.nan,
                    "group_b_prevalence": b_positive / len(group_b) if len(group_b) else np.nan,
                    "prevalence_difference": (
                        a_positive / len(group_a) - b_positive / len(group_b)
                        if len(group_a) and len(group_b)
                        else np.nan
                    ),
                    "odds_ratio": odds_ratio(a_positive, a_negative, b_positive, b_negative),
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = result.groupby("comparison", group_keys=False)["p_value"].apply(
        benjamini_hochberg
    )
    return result.sort_values(["comparison", "q_value_bh", "p_value"])


def quality_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, group_a_mask, group_b_mask in comparison_masks(frame):
        for feature in QUALITY_COLUMNS:
            group_a = pd.to_numeric(frame.loc[group_a_mask, feature], errors="coerce").dropna()
            group_b = pd.to_numeric(frame.loc[group_b_mask, feature], errors="coerce").dropna()
            if len(group_a) and len(group_b):
                test = mannwhitneyu(group_a, group_b, alternative="two-sided")
                rank_biserial = 2.0 * float(test.statistic) / (len(group_a) * len(group_b)) - 1.0
                p_value = float(test.pvalue)
            else:
                rank_biserial = float("nan")
                p_value = float("nan")
            rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "group_a_n": int(len(group_a)),
                    "group_b_n": int(len(group_b)),
                    "group_a_median": float(group_a.median()) if len(group_a) else np.nan,
                    "group_b_median": float(group_b.median()) if len(group_b) else np.nan,
                    "median_difference": (
                        float(group_a.median() - group_b.median())
                        if len(group_a) and len(group_b)
                        else np.nan
                    ),
                    "rank_biserial": rank_biserial,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = result.groupby("comparison", group_keys=False)["p_value"].apply(
        benjamini_hochberg
    )
    return result.sort_values(["comparison", "q_value_bh", "p_value"])


def transition_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby([*group_columns, "transition"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values([*group_columns, "transition"])
    )


def top_rows(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    significant = frame[frame["q_value_bh"].le(0.10)].copy()
    if significant.empty:
        significant = frame.sort_values(["q_value_bh", "p_value"]).head(limit).copy()
    return significant.head(limit).replace({np.nan: None}).to_dict(orient="records")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c2 = load_prediction(Path(args.c2_lodo), require_original_case_id=True)
    candidate = load_prediction(Path(args.candidate_lodo), require_original_case_id=False)
    c2_columns = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        "prob_high",
        "pred_idx",
    ]
    candidate_columns = ["case_id", "prob_high", "pred_idx"]
    frame = c2[c2_columns].rename(
        columns={"prob_high": "c2_prob_high", "pred_idx": "c2_pred_idx"}
    ).merge(
        candidate[candidate_columns].rename(
            columns={
                "prob_high": "candidate_prob_high",
                "pred_idx": "candidate_pred_idx",
            }
        ),
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != 591:
        raise ValueError("C2 and candidate predictions did not align to 591 cases")

    registry = pd.read_csv(
        args.registry_csv,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    registry.columns = [str(column).lstrip("\ufeff") for column in registry.columns]
    registry = registry[
        [
            "case_id",
            "original_case_id",
            "image_path",
            "image_count",
            "original_image_count",
        ]
    ].drop_duplicates("case_id")
    frame = frame.merge(
        registry,
        on=["case_id", "original_case_id"],
        how="left",
        validate="one_to_one",
    )
    if frame["image_path"].isna().any():
        raise ValueError("Missing registry image paths")

    concept = pd.read_csv(
        args.concept_csv,
        usecols=["original_case_id", "concept_has_gross_text", *CONCEPTS],
        dtype={"original_case_id": str},
        encoding="utf-8-sig",
    )
    concept.columns = [str(column).lstrip("\ufeff") for column in concept.columns]
    concept = concept.drop_duplicates("original_case_id")
    frame = frame.merge(concept, on="original_case_id", how="left", validate="one_to_one")
    frame["concept_has_gross_text"] = pd.to_numeric(
        frame["concept_has_gross_text"], errors="coerce"
    ).fillna(0).astype(int)

    quality_rows = [image_quality(str(path)) for path in frame["image_path"]]
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(quality_rows)], axis=1)
    frame["c2_correct"] = frame["c2_pred_idx"].eq(frame["label_idx"])
    frame["candidate_correct"] = frame["candidate_pred_idx"].eq(frame["label_idx"])
    frame["c2_error_type"] = error_type(frame["label_idx"], frame["c2_pred_idx"])
    frame["candidate_error_type"] = error_type(
        frame["label_idx"], frame["candidate_pred_idx"]
    )
    frame["transition"] = np.select(
        [
            frame["c2_correct"] & frame["candidate_correct"],
            ~frame["c2_correct"] & frame["candidate_correct"],
            frame["c2_correct"] & ~frame["candidate_correct"],
            ~frame["c2_correct"] & ~frame["candidate_correct"],
        ],
        ["stable_correct", "candidate_rescue", "candidate_harm", "persistent_error"],
        default="unknown",
    )

    concept_result = concept_associations(frame)
    quality_result = quality_associations(frame)
    source_transitions = transition_table(frame, ["source_dataset"])
    subtype_transitions = transition_table(frame, ["task_l6_label"])
    source_subtype_errors = (
        frame.groupby(["source_dataset", "task_l6_label", "candidate_error_type"])
        .size()
        .rename("n")
        .reset_index()
    )

    frame.to_csv(
        output_dir / "case_level_error_shift_server_only.csv",
        index=False,
        encoding="utf-8-sig",
    )
    concept_result.to_csv(output_dir / "concept_associations.csv", index=False)
    quality_result.to_csv(output_dir / "quality_associations.csv", index=False)
    source_transitions.to_csv(output_dir / "source_transitions.csv", index=False)
    subtype_transitions.to_csv(output_dir / "subtype_transitions.csv", index=False)
    source_subtype_errors.to_csv(output_dir / "source_subtype_error_counts.csv", index=False)

    transition_counts = frame["transition"].value_counts().to_dict()
    summary = {
        "candidate": args.candidate_name,
        "n": int(len(frame)),
        "doctor_concept_coverage": int(frame["concept_has_gross_text"].sum()),
        "transition_counts": {str(key): int(value) for key, value in transition_counts.items()},
        "top_concept_rows": {
            comparison: top_rows(group)
            for comparison, group in concept_result.groupby("comparison", sort=True)
        },
        "top_quality_rows": {
            comparison: top_rows(group)
            for comparison, group in quality_result.groupby("comparison", sort=True)
        },
    }
    (output_dir / "error_shift_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
