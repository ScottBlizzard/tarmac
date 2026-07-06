#!/usr/bin/env python3
"""Generate patient-level stratified splits for full thymic candidate tasks."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full-data candidate-task 5-fold splits.")
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--mapping-csv", required=True, type=Path)
    parser.add_argument("--task", required=True, help="Task column in candidate_task_mapping.csv.")
    parser.add_argument("--stratify-by", default="who_type", choices=("who_type", "task_label"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = read_csv(args.labels_csv)
    mapping = build_mapping(args.mapping_csv)

    rows = []
    for row in labels:
        who = row.get("who_type", "").strip()
        task_label = mapping.get(who, {}).get(args.task, "exclude")
        if task_label in EXCLUDE_VALUES:
            continue
        rows.append({**row, "task_label": task_label})

    if not rows:
        raise SystemExit(f"No cases remained for task {args.task}.")

    seen_case_ids: set[str] = set()
    case_rows = []
    for row in rows:
        case_id = row["case_id"]
        if case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)
        case_rows.append(row)
    stratify_col = "who_type" if args.stratify_by == "who_type" else "task_label"
    counts = Counter(row[stratify_col] for row in case_rows)
    too_small = {label: count for label, count in counts.items() if count < args.n_splits}
    if too_small:
        raise SystemExit(
            f"Cannot create {args.n_splits}-fold split; classes below n_splits in {stratify_col}: "
            f"{too_small}"
        )

    rng = random.Random(args.seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        grouped[row[stratify_col]].append(row)

    assigned_rows: list[dict[str, object]] = []
    for label, group in sorted(grouped.items()):
        group = list(group)
        rng.shuffle(group)
        for idx, row in enumerate(group):
            assigned_rows.append({**row, "master_fold_id": (idx % args.n_splits) + 1})

    split_rows = []
    for row in assigned_rows:
        split_rows.append(
            {
                "case_id": row["case_id"],
                "patient_id": row.get("patient_id", row["case_id"]),
                "who_type": row.get("who_type", ""),
                "task": args.task,
                "task_label": row["task_label"],
                "stratification_label": row[stratify_col],
                "include_main_study": row.get("include_main_study", 1),
                "master_fold_id": int(row["master_fold_id"]),
            }
        )

    fold_counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for row in split_rows:
        fold = int(row["master_fold_id"])
        fold_counts[("task_label", row["task_label"])][fold] += 1
        fold_counts[("who_type", row["who_type"])][fold] += 1

    fold_dist_rows = []
    for (level, label), counter in sorted(fold_counts.items()):
        for fold in range(1, args.n_splits + 1):
            fold_dist_rows.append({"level": level, "label": label, "fold_id": fold, "n_cases": counter.get(fold, 0)})

    split_case_counts = Counter(row["case_id"] for row in split_rows)
    split_qc_rows = [
        {"case_id": case_id, "qc_flag": "duplicate_case_in_split"}
        for case_id, count in split_case_counts.items()
        if count > 1
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "split_5fold.csv", split_rows, list(split_rows[0].keys()))
    write_csv(args.output_dir / "fold_distribution.csv", fold_dist_rows, ["level", "label", "fold_id", "n_cases"])
    write_csv(args.output_dir / "split_qc.csv", split_qc_rows, ["case_id", "qc_flag"])

    summary = [
        "# Full Data Split Summary",
        "",
        f"- task: `{args.task}`",
        f"- stratify_by: `{args.stratify_by}`",
        f"- n_splits: {args.n_splits}",
        f"- seed: {args.seed}",
        f"- included_cases: {len(split_rows)}",
        f"- split_qc_problems: {len(split_qc_rows)}",
        "",
        "## Task Label Counts",
        "",
    ]
    for label, count in sorted(Counter(row["task_label"] for row in split_rows).items()):
        summary.append(f"- {label}: {count}")
    (args.output_dir / "split_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Saved split outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
