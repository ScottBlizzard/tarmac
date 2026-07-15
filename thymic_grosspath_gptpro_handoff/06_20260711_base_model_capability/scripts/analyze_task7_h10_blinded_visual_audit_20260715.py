from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


PROBABILITY_COLUMNS = [
    "prob_INTERNAL_NATURAL",
    "prob_INTERNAL_RISK_BALANCED",
    "prob_INTERNAL_SUBTYPE_TEMPERED",
]

# These post-unblinding assignments separate visually recoverable model misses from
# phenotype-label conflicts and images that do not expose enough diagnostic evidence.
HARD_ATTRIBUTION = {
    "V011": "evidence_limited_or_ambiguous",
    "V012": "evidence_limited_or_ambiguous",
    "V013": "phenotype_label_mimic",
    "V014": "phenotype_label_mimic",
    "V023": "evidence_limited_or_ambiguous",
    "V029": "recoverable_visual_model_miss",
    "V038": "phenotype_label_mimic",
    "V040": "recoverable_visual_model_miss",
    "V043": "evidence_limited_or_ambiguous",
    "V046": "recoverable_visual_model_miss",
    "V047": "phenotype_label_mimic",
    "V050": "phenotype_label_mimic",
    "V051": "phenotype_label_mimic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the locked H10 GPT-blinded gross-image audit."
    )
    parser.add_argument("--blind-key", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--matched-h10", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def require_unique(frame: pd.DataFrame, column: str, expected_n: int | None = None) -> None:
    if column not in frame.columns:
        raise ValueError(f"Missing required column: {column}")
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"{column} must be complete and unique")
    if expected_n is not None and len(frame) != expected_n:
        raise ValueError(f"Expected {expected_n} rows, found {len(frame)}")


def largest_component(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_size = 0
    best_box: tuple[int, int, int, int] | None = None
    for row, col in np.argwhere(mask):
        row = int(row)
        col = int(col)
        if visited[row, col]:
            continue
        queue: deque[tuple[int, int]] = deque([(row, col)])
        visited[row, col] = True
        size = 0
        min_row = max_row = row
        min_col = max_col = col
        while queue:
            current_row, current_col = queue.popleft()
            size += 1
            min_row = min(min_row, current_row)
            max_row = max(max_row, current_row)
            min_col = min(min_col, current_col)
            max_col = max(max_col, current_col)
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and mask[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
        if size > best_size:
            best_size = size
            best_box = (min_row, min_col, max_row + 1, max_col + 1)
    return best_size, best_box


def approximate_foreground(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float32)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    blue_background = (
        (blue >= 70)
        & (blue >= red * 1.12)
        & (blue >= green * 1.04)
        & ((blue - red) >= 20)
    )
    raw_foreground = ~blue_background

    mask_image = Image.fromarray(raw_foreground.astype(np.uint8) * 255)
    mask_image.thumbnail((128, 128), Image.Resampling.NEAREST)
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5)).filter(
        ImageFilter.MinFilter(5)
    )
    component_mask = np.asarray(mask_image) >= 128
    component_size, box = largest_component(component_mask)
    component_fraction = component_size / component_mask.size
    if box is None:
        box_fraction = 0.0
        center_distance = 1.0
    else:
        min_row, min_col, max_row, max_col = box
        box_fraction = ((max_row - min_row) * (max_col - min_col)) / component_mask.size
        center_row = (min_row + max_row) / 2 / component_mask.shape[0]
        center_col = (min_col + max_col) / 2 / component_mask.shape[1]
        center_distance = math.sqrt((center_row - 0.5) ** 2 + (center_col - 0.5) ** 2)
    return {
        "approx_nonblue_fraction": float(raw_foreground.mean()),
        "approx_largest_component_fraction": float(component_fraction),
        "approx_component_box_fraction": float(box_fraction),
        "approx_component_center_distance": float(center_distance),
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - spread, center + spread


def summarize_tier(group: pd.DataFrame) -> pd.Series:
    decisive = group["visual_risk_impression"].isin(["low", "high"])
    correct = group.loc[decisive, "visual_correct"].astype(bool)
    ci_low, ci_high = wilson_interval(int(correct.sum()), int(decisive.sum()))
    foreground = group["approx_largest_component_fraction"]
    return pd.Series(
        {
            "n": int(len(group)),
            "adequate_evidence_n": int(group["image_evidence_sufficiency"].eq("adequate").sum()),
            "adequate_evidence_rate": float(group["image_evidence_sufficiency"].eq("adequate").mean()),
            "cut_surface_yes_n": int(group["cut_surface_exposed"].eq("yes").sum()),
            "cut_surface_yes_rate": float(group["cut_surface_exposed"].eq("yes").mean()),
            "visual_decisive_n": int(decisive.sum()),
            "visual_decisive_rate": float(decisive.mean()),
            "visual_correct_n": int(correct.sum()),
            "visual_accuracy_decisive": float(correct.mean()) if len(correct) else float("nan"),
            "visual_accuracy_ci_low": ci_low,
            "visual_accuracy_ci_high": ci_high,
            "visual_high_confidence_n": int(group["visual_confidence"].eq("high").sum()),
            "high_heterogeneity_n": int(group["surface_or_cut_heterogeneity"].eq("high").sum()),
            "foreground_component_median": float(foreground.median()),
            "foreground_component_q25": float(foreground.quantile(0.25)),
            "foreground_component_q75": float(foreground.quantile(0.75)),
            "minimum_true_probability_median": float(group["minimum_true_probability"].median()),
            "mean_true_probability_median": float(group["mean_true_probability"].median()),
        }
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(args.blind_key, dtype={"case_id": str}, encoding="utf-8-sig")
    observations = pd.read_csv(args.observations, encoding="utf-8-sig")
    matched = pd.read_csv(args.matched_h10, dtype={"case_id": str}, encoding="utf-8-sig")
    require_unique(key, "audit_code", expected_n=60)
    require_unique(observations, "audit_code", expected_n=60)
    require_unique(matched, "case_id", expected_n=117)
    if set(key["audit_code"]) != set(observations["audit_code"]):
        raise ValueError("Blind key and observations contain different audit codes")

    joined = key.merge(observations, on="audit_code", how="inner", validate="one_to_one")
    joined["label_idx"] = pd.to_numeric(joined["label_idx"], errors="raise").astype(int)
    for column in PROBABILITY_COLUMNS:
        joined[column] = pd.to_numeric(joined[column], errors="raise").astype(float)
    joined["truth_risk"] = joined["label_idx"].map({0: "low", 1: "high"})
    joined["visual_result"] = np.where(
        joined["visual_risk_impression"].eq("indeterminate"),
        "indeterminate",
        np.where(
            joined["visual_risk_impression"].eq(joined["truth_risk"]),
            "correct",
            "wrong",
        ),
    )
    joined["visual_correct"] = joined["visual_result"].eq("correct")

    true_probabilities = np.column_stack(
        [
            np.where(joined["label_idx"].to_numpy() == 1, joined[column], 1 - joined[column])
            for column in PROBABILITY_COLUMNS
        ]
    )
    joined["minimum_true_probability"] = true_probabilities.min(axis=1)
    joined["mean_true_probability"] = true_probabilities.mean(axis=1)

    foreground_rows = []
    for row in joined.itertuples(index=False):
        foreground_rows.append(
            {"audit_code": row.audit_code, **approximate_foreground(Path(row.image_path))}
        )
    foreground = pd.DataFrame(foreground_rows)
    joined = joined.merge(foreground, on="audit_code", how="left", validate="one_to_one")

    tier_order = ["easy", "boundary", "hard"]
    tier_summary = (
        joined.groupby("difficulty_tier", sort=False)
        .apply(summarize_tier, include_groups=False)
        .reindex(tier_order)
        .reset_index()
    )
    visual_by_tier = (
        pd.crosstab(joined["difficulty_tier"], joined["visual_result"])
        .reindex(index=tier_order, fill_value=0)
        .reset_index()
    )
    evidence_by_tier = (
        pd.crosstab(
            joined["difficulty_tier"], joined["image_evidence_sufficiency"]
        )
        .reindex(index=tier_order, fill_value=0)
        .reset_index()
    )
    pattern_by_tier = (
        pd.crosstab(joined["difficulty_tier"], joined["physician_pattern"])
        .reindex(index=tier_order, fill_value=0)
        .reset_index()
    )

    matched["difficulty_tier"] = np.select(
        [matched["model_correct_count"].eq(3), matched["model_correct_count"].eq(0)],
        ["easy", "hard"],
        default="boundary",
    )
    subtype_tiers = (
        pd.crosstab(matched["task_l6_label"], matched["difficulty_tier"])
        .reindex(columns=tier_order, fill_value=0)
        .reset_index()
    )
    subtype_tiers["total"] = subtype_tiers[tier_order].sum(axis=1)
    subtype_tiers["hard_rate"] = subtype_tiers["hard"] / subtype_tiers["total"]

    hard = joined[joined["difficulty_tier"].eq("hard")].copy()
    if set(hard["audit_code"]) != set(HARD_ATTRIBUTION):
        raise ValueError("Hard attribution mapping does not match the blinded hard set")
    hard["post_unblind_attribution"] = hard["audit_code"].map(HARD_ATTRIBUTION)
    hard_attribution = (
        hard.groupby("post_unblind_attribution", sort=True)
        .agg(
            n=("audit_code", "size"),
            low_risk_n=("label_idx", lambda values: int((values == 0).sum())),
            high_risk_n=("label_idx", lambda values: int((values == 1).sum())),
            adequate_evidence_n=(
                "image_evidence_sufficiency",
                lambda values: int((values == "adequate").sum()),
            ),
            visual_correct_n=("visual_result", lambda values: int((values == "correct").sum())),
            visual_wrong_n=("visual_result", lambda values: int((values == "wrong").sum())),
            visual_indeterminate_n=(
                "visual_result",
                lambda values: int((values == "indeterminate").sum()),
            ),
        )
        .reset_index()
    )

    output_frames = {
        "blinded_visual_joined_local_only.csv": joined,
        "tier_summary.csv": tier_summary,
        "visual_result_by_tier.csv": visual_by_tier,
        "evidence_by_tier.csv": evidence_by_tier,
        "physician_pattern_by_tier.csv": pattern_by_tier,
        "matched117_subtype_difficulty.csv": subtype_tiers,
        "hard_case_attribution_local_only.csv": hard,
        "hard_attribution_summary.csv": hard_attribution,
    }
    for name, frame in output_frames.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")

    summary = {
        "audit_n": int(len(joined)),
        "matched_local_h10_n": int(len(matched)),
        "tier_counts": joined["difficulty_tier"].value_counts().to_dict(),
        "hard_attribution_counts": hard["post_unblind_attribution"].value_counts().to_dict(),
        "notes": [
            "The 60-case visual observations were recorded before the blind key was opened.",
            "Foreground occupancy is an approximate blue-background segmentation metric, not a pathology annotation.",
            "The 117 matched local high-resolution cases are a selected subtype-balanced review subset, not the full 591-case cohort.",
        ],
    }
    write_json(output_dir / "audit_summary.json", summary)

    print("TIER SUMMARY")
    print(tier_summary.to_string(index=False))
    print("\nVISUAL RESULT BY TIER")
    print(visual_by_tier.to_string(index=False))
    print("\nMATCHED 117 SUBTYPE DIFFICULTY")
    print(subtype_tiers.to_string(index=False))
    print("\nHARD ATTRIBUTION")
    print(hard_attribution.to_string(index=False))


if __name__ == "__main__":
    main()
