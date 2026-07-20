from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from prepare_task7_h17_morphology_visibility_minipilot_20260721 import (
    CONCEPT_COLUMNS,
    REGIONS,
    RELATION_COLUMNS,
    write_csv,
    write_json,
)


CONCEPT_STATES = {"present", "absent", "not_visible_or_uncertain"}
REGION_STATES = {"visible", "not_visible", "uncertain"}
EXCLUDED_FROM_DIAGNOSTIC_COUNT = {
    "M9_interfering_artifact",
    "M10_valid_cut_surface_visible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and score the H17 morphology visibility mini-pilot."
    )
    parser.add_argument("--blinded-packet-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_form(path: Path, expected_reader: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"pilot_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    forbidden = [
        column
        for column in frame.columns
        if any(
            token in column.lower()
            for token in (
                "risk_judgment",
                "risk_label",
                "task_l6",
                "subtype",
                "model_prob",
                "model_correct",
                "confidence",
            )
        )
    ]
    if forbidden:
        raise ValueError(f"Forbidden risk/model columns in reader form: {forbidden}")
    required = {
        "pilot_id",
        "reader_id",
        "annotation_status",
        *CONCEPT_COLUMNS,
        *RELATION_COLUMNS,
    }
    for region in REGIONS:
        required.update(
            {
                f"{region}_visibility",
                f"{region}_x1_norm",
                f"{region}_y1_norm",
                f"{region}_x2_norm",
                f"{region}_y2_norm",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reader form is missing columns: {missing}")
    if len(frame) != 20 or frame["pilot_id"].nunique() != 20:
        raise ValueError(f"{path.name} is not a 20-case unique form")
    if set(frame["reader_id"].dropna().astype(str)) != {expected_reader}:
        raise ValueError(f"{path.name} reader_id differs from {expected_reader}")
    return frame.sort_values("pilot_id").reset_index(drop=True)


def pending_record(reader1: pd.DataFrame, reader2: pd.DataFrame) -> dict[str, Any]:
    status1 = reader1["annotation_status"].fillna("pending").astype(str).str.strip()
    status2 = reader2["annotation_status"].fillna("pending").astype(str).str.strip()
    complete1 = int(status1.eq("complete").sum())
    complete2 = int(status2.eq("complete").sum())
    return {
        "status": "PENDING_INDEPENDENT_MORPHOLOGY_ANNOTATION",
        "passed": False,
        "reader_1_complete": complete1,
        "reader_2_complete": complete2,
        "reader_1_pending": int(len(reader1) - complete1),
        "reader_2_pending": int(len(reader2) - complete2),
        "risk_labels_read": False,
        "subtype_labels_read": False,
        "strict_external_read": False,
    }


def normalize_state(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def validate_states(reader: pd.DataFrame) -> None:
    for column in CONCEPT_COLUMNS:
        values = {normalize_state(value) for value in reader[column]}
        if not values.issubset(CONCEPT_STATES):
            raise ValueError(f"Unexpected concept states in {column}: {sorted(values)}")
    for column in RELATION_COLUMNS:
        values = {normalize_state(value) for value in reader[column]}
        if not values.issubset(CONCEPT_STATES):
            raise ValueError(f"Unexpected relation states in {column}: {sorted(values)}")
    for region in REGIONS:
        column = f"{region}_visibility"
        values = {normalize_state(value) for value in reader[column]}
        if not values.issubset(REGION_STATES):
            raise ValueError(f"Unexpected region states in {column}: {sorted(values)}")


def gwet_ac1_binary(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=int)
    second = np.asarray(second, dtype=int)
    if len(first) == 0:
        return float("nan")
    observed = float((first == second).mean())
    positive_rate = float((first.mean() + second.mean()) / 2.0)
    chance = float(2.0 * positive_rate * (1.0 - positive_rate))
    if np.isclose(1.0 - chance, 0.0):
        return float("nan")
    return float((observed - chance) / (1.0 - chance))


def concept_metrics(
    concept: str,
    reader1: pd.DataFrame,
    reader2: pd.DataFrame,
) -> dict[str, Any]:
    state1 = reader1[concept].map(normalize_state)
    state2 = reader2[concept].map(normalize_state)
    judgeable1 = state1.isin({"present", "absent"})
    judgeable2 = state2.isin({"present", "absent"})
    both = judgeable1 & judgeable2
    labels1 = state1.loc[both].map({"absent": 0, "present": 1}).to_numpy(dtype=int)
    labels2 = state2.loc[both].map({"absent": 0, "present": 1}).to_numpy(dtype=int)
    raw_agreement = float((state1 == state2).mean())
    if len(labels1):
        binary_agreement = float((labels1 == labels2).mean())
        kappa = float(cohen_kappa_score(labels1, labels2))
        ac1 = gwet_ac1_binary(labels1, labels2)
    else:
        binary_agreement = float("nan")
        kappa = float("nan")
        ac1 = float("nan")
    agreement_pass = bool(
        len(labels1) >= 10
        and (
            (np.isfinite(ac1) and ac1 >= 0.60)
            or (np.isfinite(kappa) and kappa >= 0.60)
        )
    )
    visibility_pass = bool(judgeable1.mean() >= 0.70 and judgeable2.mean() >= 0.70)
    return {
        "concept": concept,
        "reader_1_judgeable_n": int(judgeable1.sum()),
        "reader_2_judgeable_n": int(judgeable2.sum()),
        "reader_1_judgeable_fraction": float(judgeable1.mean()),
        "reader_2_judgeable_fraction": float(judgeable2.mean()),
        "both_judgeable_n": int(both.sum()),
        "raw_three_state_agreement": raw_agreement,
        "binary_agreement": binary_agreement,
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
        "visibility_pass": visibility_pass,
        "agreement_pass": agreement_pass,
        "diagnostic_concept_pass": bool(
            concept not in EXCLUDED_FROM_DIAGNOSTIC_COUNT
            and visibility_pass
            and agreement_pass
        ),
    }


def parse_box(row: pd.Series, region: str) -> np.ndarray | None:
    columns = [
        f"{region}_x1_norm",
        f"{region}_y1_norm",
        f"{region}_x2_norm",
        f"{region}_y2_norm",
    ]
    values = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"Invalid normalized ROI for {row['pilot_id']} {region}: {values}")
    return values


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (first[2] - first[0]) * (first[3] - first[1])
    area2 = (second[2] - second[0]) * (second[3] - second[1])
    return float(intersection / max(area1 + area2 - intersection, 1e-12))


def region_metrics(
    region: str,
    reader1: pd.DataFrame,
    reader2: pd.DataFrame,
) -> dict[str, Any]:
    visibility_column = f"{region}_visibility"
    state1 = reader1[visibility_column].map(normalize_state)
    state2 = reader2[visibility_column].map(normalize_state)
    jointly_visible = state1.eq("visible") & state2.eq("visible")
    ious: list[float] = []
    for index in np.flatnonzero(jointly_visible.to_numpy(dtype=bool)):
        box1 = parse_box(reader1.iloc[index], region)
        box2 = parse_box(reader2.iloc[index], region)
        if box1 is not None and box2 is not None:
            ious.append(box_iou(box1, box2))
    return {
        "region": region,
        "visibility_agreement": float((state1 == state2).mean()),
        "jointly_visible_n": int(jointly_visible.sum()),
        "boxes_compared_n": int(len(ious)),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "median_iou": float(np.median(ious)) if ious else float("nan"),
        "iou_ge_0_50_fraction": (
            float((np.asarray(ious) >= 0.50).mean()) if ious else float("nan")
        ),
    }


def main() -> None:
    args = parse_args()
    packet_dir = Path(args.blinded_packet_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader1 = read_form(packet_dir / "READER_1_MORPHOLOGY.csv", "reader_1")
    reader2 = read_form(packet_dir / "READER_2_MORPHOLOGY.csv", "reader_2")
    if not reader1["pilot_id"].equals(reader2["pilot_id"]):
        raise ValueError("Reader forms contain different pilot IDs")

    complete1 = reader1["annotation_status"].fillna("").astype(str).str.strip().eq("complete")
    complete2 = reader2["annotation_status"].fillna("").astype(str).str.strip().eq("complete")
    if not complete1.all() or not complete2.all():
        decision = pending_record(reader1, reader2)
        write_json(output_dir / "h17_minipilot_gate.json", decision)
        (output_dir / "RUN.status").write_text(decision["status"] + "\n", encoding="utf-8")
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
        return

    validate_states(reader1)
    validate_states(reader2)
    concept_frame = pd.DataFrame(
        [
            concept_metrics(concept, reader1, reader2)
            for concept in CONCEPT_COLUMNS
        ]
    )
    region_frame = pd.DataFrame(
        [region_metrics(region, reader1, reader2) for region in REGIONS]
    )
    diagnostic_pass_count = int(concept_frame["diagnostic_concept_pass"].sum())
    required_regions = {
        "Z1_main_cut_surface",
        "Z2_capsule_outer_interface",
        "Z3_internal_abnormality",
    }
    required_region_rows = region_frame.loc[
        region_frame["region"].isin(required_regions)
    ]
    region_agreement_pass = bool(
        len(required_region_rows) == 3
        and (
            required_region_rows["mean_iou"].fillna(0.0) >= 0.60
        ).all()
    )
    checks = {
        "at_least_four_diagnostic_concepts_pass": diagnostic_pass_count >= 4,
        "z1_z2_z3_mean_iou_ge_0_60": region_agreement_pass,
        "reader_forms_complete": True,
        "risk_fields_absent": True,
    }
    passed = all(checks.values())
    decision = {
        "status": (
            "PASS_H17_MINIPILOT_PROCEED_60_CASE_PILOT"
            if passed
            else "FAIL_H17_MINIPILOT_STOP_CURRENT_CONCEPT_SCHEMA"
        ),
        "passed": passed,
        "diagnostic_concepts_passed": diagnostic_pass_count,
        "diagnostic_concept_names": concept_frame.loc[
            concept_frame["diagnostic_concept_pass"], "concept"
        ].tolist(),
        "checks": checks,
        "risk_labels_read": False,
        "subtype_labels_read": False,
        "strict_external_read": False,
    }
    write_csv(output_dir / "concept_agreement_metrics.csv", concept_frame)
    write_csv(output_dir / "region_agreement_metrics.csv", region_frame)
    write_json(output_dir / "h17_minipilot_gate.json", decision)
    (output_dir / "RUN.status").write_text(decision["status"] + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
