from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

CONCEPTS = [
    "boundary_clear",
    "boundary_unclear",
    "capsule_any",
    "capsule_complete",
    "capsule_absent",
    "invasion",
    "fat_involved_or_attached",
    "lung_attached",
    "pericardium_attached",
    "pleura_attached",
    "hemorrhage",
    "necrosis",
    "cystic_change",
    "calcification",
    "nodular_lobulated",
    "septum",
    "homogeneous",
    "gray_white",
    "gray_yellow",
    "gray_red",
    "gray_brown",
    "gray_black",
    "texture_soft",
    "texture_medium",
    "texture_tough",
    "texture_fragile",
    "cut_surface_mentioned",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, help="tag=path[:probability_column]")
    parser.add_argument(
        "--concept-csv",
        default="/workspace/thymic_project/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--high-confidence", type=float, default=0.80)
    parser.add_argument("--max-correct-controls-per-subtype", type=int, default=8)
    return parser.parse_args()


def parse_prediction_spec(specification: str) -> tuple[str, Path, str]:
    tag, remainder = specification.split("=", 1)
    probability_column = "prob_high"
    path_text = remainder
    if ":" in remainder:
        possible_path, possible_column = remainder.rsplit(":", 1)
        if possible_column.startswith("prob_"):
            path_text = possible_path
            probability_column = possible_column
    return tag, Path(path_text), probability_column


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    count = len(ordered)
    adjusted = ordered.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def association_rows(frame: pd.DataFrame, tag: str) -> list[dict]:
    rows = []
    comparisons = [
        ("low_fp_vs_tn", frame["label_idx"].eq(0), "FP", "TN"),
        ("high_fn_vs_tp", frame["label_idx"].eq(1), "FN", "TP"),
    ]
    for comparison, class_mask, error_name, correct_name in comparisons:
        subset = frame[class_mask & frame["concept_has_gross_text"].eq(1)].copy()
        error = subset[subset["error_type"].eq(error_name)]
        correct = subset[subset["error_type"].eq(correct_name)]
        for concept in CONCEPTS:
            error_positive = int(pd.to_numeric(error[concept], errors="coerce").fillna(0).gt(0).sum())
            correct_positive = int(pd.to_numeric(correct[concept], errors="coerce").fillna(0).gt(0).sum())
            error_negative = len(error) - error_positive
            correct_negative = len(correct) - correct_positive
            rows.append(
                {
                    "model_tag": tag,
                    "comparison": comparison,
                    "concept": concept,
                    "error_n": len(error),
                    "correct_n": len(correct),
                    "error_positive": error_positive,
                    "error_negative": error_negative,
                    "correct_positive": correct_positive,
                    "correct_negative": correct_negative,
                    "error_prevalence": error_positive / len(error) if len(error) else np.nan,
                    "correct_prevalence": correct_positive / len(correct) if len(correct) else np.nan,
                    "prevalence_difference": (
                        error_positive / len(error) - correct_positive / len(correct)
                        if len(error) and len(correct)
                        else np.nan
                    ),
                    "odds_ratio": odds_ratio(
                        error_positive, error_negative, correct_positive, correct_negative
                    ),
                    "fisher_p": (
                        float(
                            fisher_exact(
                                [[error_positive, error_negative], [correct_positive, correct_negative]],
                                alternative="two-sided",
                            ).pvalue
                        )
                        if len(error) and len(correct)
                        else np.nan
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concept = pd.read_csv(
        args.concept_csv,
        usecols=["original_case_id", "concept_has_gross_text", *CONCEPTS],
        dtype={"original_case_id": str},
        encoding="utf-8-sig",
    )
    concept.columns = [str(column).lstrip("\ufeff") for column in concept.columns]
    concept = concept.drop_duplicates("original_case_id")
    association = []
    review_frames = []
    for specification in args.prediction:
        tag, prediction_path, probability_column = parse_prediction_spec(specification)
        prediction = pd.read_csv(
            prediction_path, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig"
        )
        prediction.columns = [str(column).lstrip("\ufeff") for column in prediction.columns]
        if "original_case_id" not in prediction.columns:
            raise ValueError(f"Prediction file lacks original_case_id: {prediction_path}")
        prediction["prob_high"] = pd.to_numeric(prediction[probability_column], errors="raise")
        prediction["pred_idx"] = (prediction["prob_high"] >= 0.5).astype(int)
        prediction["confidence"] = np.maximum(prediction["prob_high"], 1.0 - prediction["prob_high"])
        prediction["error_type"] = np.select(
            [
                prediction["label_idx"].eq(0) & prediction["pred_idx"].eq(0),
                prediction["label_idx"].eq(0) & prediction["pred_idx"].eq(1),
                prediction["label_idx"].eq(1) & prediction["pred_idx"].eq(0),
                prediction["label_idx"].eq(1) & prediction["pred_idx"].eq(1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="unknown",
        )
        merged = prediction.merge(concept, on="original_case_id", how="left")
        merged["concept_has_gross_text"] = pd.to_numeric(
            merged["concept_has_gross_text"], errors="coerce"
        ).fillna(0).astype(int)
        merged.to_csv(output_dir / f"{tag}_predictions_with_doctor_concepts.csv", index=False, encoding="utf-8-sig")
        association.extend(association_rows(merged, tag))

        errors = merged[merged["error_type"].isin(["FP", "FN"])].copy()
        errors["review_stratum"] = np.where(
            errors["confidence"] >= args.high_confidence, "high_confidence_error", "lower_confidence_error"
        )
        controls = []
        correct = merged[merged["error_type"].isin(["TN", "TP"])].copy()
        for subtype, group in correct.groupby("task_l6_label"):
            controls.append(
                group.sort_values("confidence", ascending=False).head(args.max_correct_controls_per_subtype)
            )
        control_frame = pd.concat(controls, ignore_index=True) if controls else correct.head(0)
        control_frame["review_stratum"] = "high_confidence_correct_control"
        review = pd.concat([errors, control_frame], ignore_index=True)
        review.insert(0, "model_tag", tag)
        review["doctor_image_quality"] = ""
        review["doctor_view_type"] = ""
        review["doctor_visible_morphology"] = ""
        review["doctor_primary_boundary"] = ""
        review["doctor_visually_judgeable"] = ""
        review["doctor_error_hypothesis"] = ""
        review["doctor_comment"] = ""
        review_frames.append(review)

    association_frame = pd.DataFrame(association).sort_values(
        ["model_tag", "comparison", "prevalence_difference"], ascending=[True, True, False]
    )
    association_frame["fisher_q_bh"] = association_frame.groupby(
        ["model_tag", "comparison"], group_keys=False
    )["fisher_p"].apply(benjamini_hochberg)
    association_frame.to_csv(
        output_dir / "doctor_concept_error_associations.csv", index=False, encoding="utf-8-sig"
    )
    review_frame = pd.concat(review_frames, ignore_index=True)
    review_columns = [
        "model_tag",
        "review_stratum",
        "case_id",
        "original_case_id",
        "domain",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        "pred_idx",
        "prob_high",
        "confidence",
        "error_type",
        "image_name",
        "image_path",
        *CONCEPTS,
        "doctor_image_quality",
        "doctor_view_type",
        "doctor_visible_morphology",
        "doctor_primary_boundary",
        "doctor_visually_judgeable",
        "doctor_error_hypothesis",
        "doctor_comment",
    ]
    available_columns = [column for column in review_columns if column in review_frame.columns]
    review_frame[available_columns].to_csv(
        output_dir / "physician_error_review_candidates.csv", index=False, encoding="utf-8-sig"
    )
    for (tag, comparison), group in association_frame.groupby(["model_tag", "comparison"]):
        print(f"\n[{tag} {comparison}]", flush=True)
        print(
            group.reindex(group["prevalence_difference"].abs().sort_values(ascending=False).index)
            .head(10)
            [[
                "concept",
                "error_prevalence",
                "correct_prevalence",
                "prevalence_difference",
                "odds_ratio",
                "fisher_p",
                "fisher_q_bh",
            ]]
            .to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
