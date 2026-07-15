from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


EXPECTED_SUBTYPES = {"A": 44, "AB": 262, "B1": 62, "B2": 89, "B3": 24, "TC": 110}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build H10 internal folds using subtype only, never source/batch."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_path = Path(args.output_csv)
    manifest_path = Path(args.manifest_json)
    metadata = pd.read_csv(
        metadata_path,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    required = {"case_id", "original_case_id", "task_l6_label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(metadata.columns))}")
    if len(metadata) != 591 or metadata["case_id"].duplicated().any():
        raise ValueError("Expected 591 unique internal case IDs")
    if metadata["original_case_id"].duplicated().any():
        raise ValueError("H10 requires unique original pathology IDs")
    counts = metadata["task_l6_label"].value_counts().to_dict()
    if counts != EXPECTED_SUBTYPES:
        raise ValueError(f"Unexpected subtype totals: {counts}")

    # Sorting makes the split invariant to feature-bank row order. Source is not read.
    work = metadata[["case_id", "task_l6_label"]].sort_values("case_id").reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    work["master_fold_id"] = 0
    for fold_id, (_, test_rows) in enumerate(
        splitter.split(work["case_id"], work["task_l6_label"]), start=1
    ):
        work.loc[test_rows, "master_fold_id"] = fold_id
    if not work["master_fold_id"].between(1, 5).all():
        raise ValueError("Incomplete H10 fold assignment")

    fold_sizes = work["master_fold_id"].value_counts().sort_index()
    subtype_by_fold = pd.crosstab(work["master_fold_id"], work["task_l6_label"])
    if int(fold_sizes.max() - fold_sizes.min()) > 1:
        raise ValueError(f"Fold sizes are not balanced: {fold_sizes.to_dict()}")
    if any(int(subtype_by_fold[column].max() - subtype_by_fold[column].min()) > 1 for column in subtype_by_fold):
        raise ValueError(f"Subtype folds are not balanced:\n{subtype_by_fold}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work[["case_id", "master_fold_id"]].to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    manifest = {
        "experiment": "H10_INTERNAL_PHENOTYPE_DIFFICULTY_REDESIGN_20260715",
        "case_count": 591,
        "unique_case_count": int(work["case_id"].nunique()),
        "stratification_columns": ["task_l6_label"],
        "forbidden_split_columns": ["source_dataset", "domain", "dataset_role"],
        "n_splits": 5,
        "shuffle": True,
        "seed": args.seed,
        "fold_sizes": {str(key): int(value) for key, value in fold_sizes.items()},
        "subtype_by_fold": {
            str(index): {str(key): int(value) for key, value in row.items()}
            for index, row in subtype_by_fold.to_dict(orient="index").items()
        },
        "split_sha256": sha256_file(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
