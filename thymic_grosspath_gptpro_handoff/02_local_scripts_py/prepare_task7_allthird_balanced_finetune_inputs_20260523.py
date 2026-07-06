from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare old+third no-repetition balanced 5-fold inputs for Task7 fine-tuning.")
    parser.add_argument(
        "--input-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/06_adapt72_highfocus_finetune_inputs_20260521",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523",
    )
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def stable_group_seed(seed: int, value: object) -> int:
    text = str(value)
    total = 0
    for ch in text:
        total = (total * 131 + ord(ch)) % 1000003
    return seed + total


def assign_balanced_folds(registry: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    fold_ids = [1, 2, 3, 4, 5]
    strat_cols = ["source_dataset", "task_l6_label", "task_l7_label"]
    for key, group in registry.groupby(strat_cols, sort=True):
        shuffled = group.sample(frac=1.0, random_state=stable_group_seed(seed, key))
        for idx, (_, row) in enumerate(shuffled.iterrows()):
            rows.append(
                {
                    "case_id": row["case_id"],
                    "patient_id": row.get("patient_id", row["case_id"]),
                    "master_fold_id": fold_ids[idx % len(fold_ids)],
                    "notes": "old_third_all_balanced_no_repetition",
                }
            )
    return pd.DataFrame(rows).sort_values(["master_fold_id", "case_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(input_dir / "registry.csv", dtype={"case_id": str, "original_case_id": str})
    adapt_rep = registry["source_dataset"].eq("third_batch_adapt72_highfocus") & registry["case_id"].str.contains("_rep", regex=False)
    kept = registry[~adapt_rep].copy().reset_index(drop=True)

    if kept["case_id"].duplicated().any():
        dup = kept.loc[kept["case_id"].duplicated(), "case_id"].head(10).tolist()
        raise ValueError(f"Duplicated case_id after removing adapt repetitions: {dup}")
    if kept["original_case_id"].duplicated().any():
        dup = kept.loc[kept["original_case_id"].duplicated(), "original_case_id"].head(10).tolist()
        raise ValueError(f"Duplicated original_case_id after removing adapt repetitions: {dup}")

    split = assign_balanced_folds(kept, args.seed)
    missing = set(kept["case_id"]) - set(split["case_id"])
    extra = set(split["case_id"]) - set(kept["case_id"])
    if missing or extra:
        raise RuntimeError(f"registry/split mismatch missing={len(missing)} extra={len(extra)}")

    kept.to_csv(output_dir / "registry.csv", index=False, encoding="utf-8-sig")
    split.to_csv(output_dir / "split.csv", index=False, encoding="utf-8-sig")
    kept.groupby(["source_dataset", "task_l6_label", "task_l7_label"]).size().reset_index(name="n").to_csv(
        output_dir / "label_counts.csv", index=False, encoding="utf-8-sig"
    )
    kept[["case_id", "source_dataset", "task_l6_label", "task_l7_label"]].merge(split[["case_id", "master_fold_id"]], on="case_id").groupby(
        ["master_fold_id", "source_dataset", "task_l7_label"]
    ).size().reset_index(name="n").to_csv(output_dir / "split_counts.csv", index=False, encoding="utf-8-sig")

    print(f"registry rows={len(kept)} cases={kept['case_id'].nunique()}")
    print(f"removed adapt repetitions={int(adapt_rep.sum())}")
    print(pd.read_csv(output_dir / "split_counts.csv").to_string(index=False))


if __name__ == "__main__":
    main()
