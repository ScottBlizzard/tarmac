#!/usr/bin/env python3
"""Summarize candidate task distributions and optional fold balance."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


TASK_COLUMNS = {
    "task_A_low_high": "task_A_distribution.csv",
    "task_B_thymoma_5class": "task_B_distribution.csv",
    "task_C_8class": "task_C_distribution.csv",
    "task_D_l1": "task_D_l1_distribution.csv",
    "task_D_l2": "task_D_l2_distribution.csv",
    "task_D_l3": "task_D_l3_distribution.csv",
    "task_D_l4": "task_D_l4_distribution.csv",
}

EXCLUDE_VALUES = {"", "na", "exclude", "review"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_mapping(mapping_csv: Path) -> dict[str, dict[str, str]]:
    return {row["who_type"]: row for row in read_csv(mapping_csv)}


def distribution(labels: list[dict[str, str]], mapping: dict[str, dict[str, str]], task_column: str) -> list[dict[str, object]]:
    counter: Counter[str] = Counter()
    excluded = 0
    review = 0
    for row in labels:
        who = row["who_type"]
        mapped = mapping.get(who, {}).get(task_column, "exclude")
        if mapped == "review":
            review += 1
            continue
        if mapped in EXCLUDE_VALUES:
            excluded += 1
            continue
        counter[mapped] += 1
    total_included = sum(counter.values())
    rows = [
        {
            "label": label,
            "n_cases": count,
            "fraction_of_included": round(count / total_included, 6) if total_included else 0,
        }
        for label, count in sorted(counter.items())
    ]
    rows.append({"label": "__included_total__", "n_cases": total_included, "fraction_of_included": 1 if total_included else 0})
    rows.append({"label": "__excluded__", "n_cases": excluded, "fraction_of_included": ""})
    rows.append({"label": "__review__", "n_cases": review, "fraction_of_included": ""})
    return rows


def split_qc(labels: list[dict[str, str]], mapping: dict[str, dict[str, str]], split_csv: Path, output_dir: Path) -> None:
    splits = {row["case_id"]: row for row in read_csv(split_csv)}
    labels_by_case = {row["case_id"]: row for row in labels}
    rows = []
    fold_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    problems = []

    for case_id, label_row in labels_by_case.items():
        split_row = splits.get(case_id)
        if not split_row:
            problems.append({"case_id": case_id, "qc_flag": "missing_split"})
            continue
        fold = split_row["master_fold_id"]
        who = label_row["who_type"]
        fold_counter[("who_type", who)][fold] += 1
        for task_column in TASK_COLUMNS:
            mapped = mapping.get(who, {}).get(task_column, "exclude")
            if mapped in EXCLUDE_VALUES:
                mapped = f"__{mapped or 'excluded'}__"
            fold_counter[(task_column, mapped)][fold] += 1

    split_case_counts = Counter(row["case_id"] for row in read_csv(split_csv))
    for case_id, count in split_case_counts.items():
        if count > 1:
            problems.append({"case_id": case_id, "qc_flag": "duplicate_case_in_split"})

    for (level, label), counter in sorted(fold_counter.items()):
        for fold in ["1", "2", "3", "4", "5"]:
            rows.append({"level": level, "label": label, "fold_id": fold, "n_cases": counter.get(fold, 0)})

    write_csv(output_dir / "current_160_fold_distribution.csv", rows, ["level", "label", "fold_id", "n_cases"])
    write_csv(output_dir / "current_160_split_qc.csv", problems, ["case_id", "qc_flag"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize candidate task feasibility.")
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--mapping-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_csv(args.labels_csv)
    mapping = build_mapping(args.mapping_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = ["# Candidate Task Feasibility", ""]
    for task_column, filename in TASK_COLUMNS.items():
        rows = distribution(labels, mapping, task_column)
        write_csv(args.output_dir / filename, rows, ["label", "n_cases", "fraction_of_included"])
        included = next(row["n_cases"] for row in rows if row["label"] == "__included_total__")
        labels_only = [row for row in rows if not str(row["label"]).startswith("__")]
        min_class = min((int(row["n_cases"]) for row in labels_only), default=0)
        report_lines.append(f"## {task_column}")
        report_lines.append("")
        report_lines.append(f"- included cases: {included}")
        report_lines.append(f"- included classes: {len(labels_only)}")
        report_lines.append(f"- minimum class size: {min_class}")
        report_lines.append("")

    (args.output_dir / "task_feasibility_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    if args.split_csv:
        split_qc(labels, mapping, args.split_csv, args.output_dir)

    print(f"Saved task feasibility outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
