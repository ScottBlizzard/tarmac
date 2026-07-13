from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REFERENCE = "c1_siglipl512_summary"
CANDIDATES = (
    "siglip2_so400m_naflex",
    "radiov25_l",
    "pe_spatial_l14_448",
    "medsiglip_448",
    "siglip2_large_p16_512",
    "siglip2_base_naflex",
)
VISION_PARAMETER_COUNTS = {
    "siglip2_base_naflex": 92_930_304,
    "pe_spatial_l14_448": 303_964_160,
    "siglip2_large_p16_512": 316_742_656,
    "radiov25_l": 319_881_225,
    "siglip2_so400m_naflex": 427_888_064,
    "medsiglip_448": 428_565_440,
}
EXPECTED_FOLDS = {"fivefold": 5, "source_lodo": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize H3A using validation-only candidate ranking."
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_mode(candidate_dir: Path, mode: str) -> dict[str, Any] | None:
    mode_dir = candidate_dir / mode
    if not (mode_dir / "RUN.status").is_file():
        return None
    if (mode_dir / "RUN.status").read_text(encoding="utf-8").strip() != "complete":
        return None
    summaries = json.loads((mode_dir / "fold_summaries.json").read_text(encoding="utf-8"))
    if len(summaries) != EXPECTED_FOLDS[mode]:
        raise ValueError(f"{candidate_dir.name}/{mode} has {len(summaries)} folds")
    validation = [float(item["best_val_bacc"]) for item in summaries]
    overall = json.loads((mode_dir / "overall_metrics.json").read_text(encoding="utf-8"))
    subtype = pd.read_csv(mode_dir / "subtype_metrics.csv").set_index("subtype")
    source = pd.read_csv(mode_dir / "source_metrics.csv")
    return {
        "validation_bacc_by_fold": validation,
        "validation_bacc_mean": sum(validation) / len(validation),
        "test_balanced_accuracy": float(overall["balanced_accuracy"]),
        "test_auc": float(overall["auc"]),
        "test_sensitivity": float(overall["sensitivity"]),
        "test_specificity": float(overall["specificity"]),
        "test_b1_accuracy": float(subtype.loc["B1", "risk_accuracy"]),
        "test_b2_accuracy": float(subtype.loc["B2", "risk_accuracy"]),
        "test_worst_source_bacc": float(source["balanced_accuracy"].min()),
    }


def candidate_record(runs_root: Path, candidate: str) -> dict[str, Any]:
    candidate_dir = runs_root / candidate
    modes = {mode: load_mode(candidate_dir, mode) for mode in EXPECTED_FOLDS}
    complete = all(value is not None for value in modes.values())
    return {
        "candidate": candidate,
        "is_reference": candidate == REFERENCE,
        "complete": complete,
        "vision_parameter_count": VISION_PARAMETER_COUNTS.get(candidate),
        "fivefold": modes["fivefold"],
        "source_lodo": modes["source_lodo"],
    }


def flat_record(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": record["candidate"],
        "is_reference": record["is_reference"],
        "complete": record["complete"],
        "vision_parameter_count": record["vision_parameter_count"],
    }
    for mode in EXPECTED_FOLDS:
        values = record[mode]
        for key in (
            "validation_bacc_mean",
            "test_balanced_accuracy",
            "test_auc",
            "test_sensitivity",
            "test_specificity",
            "test_b1_accuracy",
            "test_b2_accuracy",
            "test_worst_source_bacc",
        ):
            row[f"{mode}_{key}"] = None if values is None else values[key]
    return row


def write_markdown(
    path: Path,
    table: pd.DataFrame,
    screen_complete: bool,
    ranking: list[str],
    shortlist: list[str],
) -> None:
    columns = [
        "candidate",
        "complete",
        "vision_parameter_count",
        "fivefold_validation_bacc_mean",
        "source_lodo_validation_bacc_mean",
        "fivefold_test_balanced_accuracy",
        "source_lodo_test_balanced_accuracy",
        "source_lodo_test_sensitivity",
        "source_lodo_test_specificity",
        "source_lodo_test_b1_accuracy",
        "source_lodo_test_b2_accuracy",
    ]
    selected = table[columns]
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in selected.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        table_lines.append("| " + " | ".join(cells) + " |")
    lines = [
        "# H3A Representation Screen Aggregate",
        "",
        f"- Screen complete: `{str(screen_complete).lower()}`",
        "- Ranking keys: five-fold validation BAcc, then source-LODO validation "
        "BAcc, then smaller vision encoder.",
        "- Outer test metrics below are descriptive and never enter candidate ranking.",
        f"- Current validation-only ranking: `{', '.join(ranking) or 'none'}`",
        f"- Locked H3B shortlist: `{', '.join(shortlist) or 'not locked'}`",
        "",
        *table_lines,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [candidate_record(runs_root, name) for name in (REFERENCE, *CANDIDATES)]
    complete_candidates = [
        item for item in records if not item["is_reference"] and item["complete"]
    ]
    ranked = sorted(
        complete_candidates,
        key=lambda item: (
            -item["fivefold"]["validation_bacc_mean"],
            -item["source_lodo"]["validation_bacc_mean"],
            item["vision_parameter_count"],
            item["candidate"],
        ),
    )
    ranking = [item["candidate"] for item in ranked]
    screen_complete = len(complete_candidates) == len(CANDIDATES)
    shortlist = ranking[:2] if screen_complete else []
    payload = {
        "selection_uses_outer_test_metrics": False,
        "ranking_keys": [
            "fivefold_validation_bacc_mean_desc",
            "source_lodo_validation_bacc_mean_desc",
            "vision_parameter_count_asc",
            "candidate_asc",
        ],
        "expected_candidates": list(CANDIDATES),
        "screen_complete": screen_complete,
        "validation_only_ranking": ranking,
        "locked_h3b_shortlist": shortlist,
        "records": records,
    }
    (output_dir / "h3a_screen_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table = pd.DataFrame(flat_record(item) for item in records)
    table.to_csv(output_dir / "h3a_screen_summary.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        output_dir / "h3a_screen_summary.md",
        table,
        screen_complete,
        ranking,
        shortlist,
    )
    print(json.dumps({"screen_complete": screen_complete, "ranking": ranking, "shortlist": shortlist}, indent=2))


if __name__ == "__main__":
    main()
