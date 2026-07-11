from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MASTER = (
    "/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/"
    "task7_four_domain_master_registry.csv"
)
DEFAULT_CACHE = "/root/thymic_task7_internal_registry_cached_max2048_20260711.csv"
DEFAULT_SPLIT = (
    "/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/"
    "45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay internal cached paths on the locked Task7 master registry.")
    parser.add_argument("--master-registry", default=DEFAULT_MASTER)
    parser.add_argument("--cache-registry", default=DEFAULT_CACHE)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = pd.read_csv(
        args.master_registry,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    master = master[master["domain"].isin(["old_data", "third_batch"])].copy()
    cache = pd.read_csv(
        args.cache_registry,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    cache_paths = cache[["case_id", "training_image_path"]].drop_duplicates("case_id")
    if len(cache_paths) != 591:
        raise ValueError(f"Expected 591 cached internal paths, found {len(cache_paths)}")
    merged = master.merge(cache_paths, on="case_id", how="left", validate="one_to_one")
    if len(merged) != 591 or merged["training_image_path"].isna().any():
        raise RuntimeError("Cached path merge did not cover all 591 internal cases.")

    split = pd.read_csv(args.split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split.columns = [str(column).lstrip("\ufeff") for column in split.columns]
    fold_column = "master_fold_id" if "master_fold_id" in split.columns else "fold_id"
    fold_map = split.drop_duplicates("case_id").set_index("case_id")[fold_column]
    missing_fold = sorted(set(merged["case_id"]) - set(fold_map.index))
    if missing_fold:
        raise ValueError(f"Locked split misses internal cases: {missing_fold[:10]}")
    merged["master_fold_id"] = merged["case_id"].map(fold_map).astype(int)
    merged["image_path"] = merged.pop("training_image_path")
    merged["image_exists"] = merged["image_path"].map(lambda value: Path(str(value)).exists())
    if not merged["image_exists"].all():
        missing = merged.loc[~merged["image_exists"], "image_path"].tolist()
        raise FileNotFoundError(f"Missing cached images: {missing[:10]}")

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print(
        f"cached_master_registry={output} rows={len(merged)} domains={merged['domain'].value_counts().to_dict()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
