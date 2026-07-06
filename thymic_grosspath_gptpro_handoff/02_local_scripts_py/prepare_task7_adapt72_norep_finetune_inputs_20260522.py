from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare no-repetition adapt72 fine-tune registry/split for Task7.")
    parser.add_argument(
        "--input-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/06_adapt72_highfocus_finetune_inputs_20260521",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/34_adapt72_highfocus_norep_finetune_inputs_20260522",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def balanced_assign_adapt_folds(adapt: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    fold_ids = [2, 3, 4, 5]
    for _, group in adapt.groupby("task_l6_label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed + abs(hash(str(group["task_l6_label"].iloc[0]))) % 10000)
        for idx, (_, row) in enumerate(shuffled.iterrows()):
            item = row.copy()
            item["master_fold_id"] = fold_ids[idx % len(fold_ids)]
            rows.append(item)
    assigned = pd.DataFrame(rows).sort_values(["master_fold_id", "task_l6_label", "case_id"]).reset_index(drop=True)
    return assigned


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(in_dir / "registry.csv", dtype={"case_id": str, "original_case_id": str})
    split = pd.read_csv(in_dir / "split.csv", dtype={"case_id": str})

    adapt_mask = registry["source_dataset"].eq("third_batch_adapt72_highfocus")
    adapt = registry[adapt_mask].copy()
    adapt_base = adapt[~adapt["case_id"].str.contains("_rep", regex=False)].copy()
    if adapt_base["original_case_id"].duplicated().any():
        dup = adapt_base.loc[adapt_base["original_case_id"].duplicated(), "original_case_id"].head(10).tolist()
        raise ValueError(f"Duplicated adapt original_case_id after no-rep filter: {dup}")

    kept_registry = pd.concat([registry[~adapt_mask], adapt_base], ignore_index=True)

    kept_nonadapt_split = split[split["case_id"].isin(registry.loc[~adapt_mask, "case_id"])].copy()
    adapt_split = adapt_base[["case_id", "patient_id", "task_l6_label"]].copy()
    adapt_split = balanced_assign_adapt_folds(adapt_split, args.seed)
    adapt_split["notes"] = "third_adapt72_highfocus_norep_balanced"
    adapt_split = adapt_split[["case_id", "patient_id", "master_fold_id", "notes"]]
    kept_split = pd.concat([kept_nonadapt_split, adapt_split], ignore_index=True)

    missing = set(kept_registry["case_id"]) - set(kept_split["case_id"])
    extra = set(kept_split["case_id"]) - set(kept_registry["case_id"])
    if missing or extra:
        raise RuntimeError(f"registry/split mismatch missing={len(missing)} extra={len(extra)}")

    kept_registry.to_csv(out_dir / "registry.csv", index=False, encoding="utf-8-sig")
    kept_split.to_csv(out_dir / "split.csv", index=False, encoding="utf-8-sig")

    split_counts = (
        kept_registry[["case_id", "source_dataset", "task_l6_label", "task_l7_label"]]
        .merge(kept_split[["case_id", "master_fold_id"]], on="case_id", how="left")
        .groupby(["master_fold_id", "source_dataset", "task_l7_label"])
        .size()
        .reset_index(name="n")
    )
    split_counts.to_csv(out_dir / "split_counts.csv", index=False, encoding="utf-8-sig")
    (
        kept_registry.groupby(["source_dataset", "task_l6_label", "task_l7_label"])
        .size()
        .reset_index(name="n")
        .to_csv(out_dir / "label_counts.csv", index=False, encoding="utf-8-sig")
    )
    print(f"registry rows={len(kept_registry)} cases={kept_registry['case_id'].nunique()}")
    print(f"adapt no-rep cases={len(adapt_base)}")
    print(split_counts.to_string(index=False))


if __name__ == "__main__":
    main()
