from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_REGISTRY = (
    "/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/"
    "task7_four_domain_master_registry.csv"
)
DEFAULT_SPLIT = (
    "/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/"
    "45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
)
DEFAULT_VARIANT_ROOT = (
    "/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/"
    "variant_inputs_all/selected_images"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 17-case two-image Task7 sensitivity registry.")
    parser.add_argument("--registry-csv", default=DEFAULT_REGISTRY)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT)
    parser.add_argument("--variant-root", default=DEFAULT_VARIANT_ROOT)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(
        args.registry_csv,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    split = pd.read_csv(args.split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split.columns = [str(column).lstrip("\ufeff") for column in split.columns]
    fold_column = "master_fold_id" if "master_fold_id" in split.columns else "fold_id"
    fold_map = split.drop_duplicates("case_id").set_index("case_id")[fold_column]

    original_count = pd.to_numeric(registry["original_image_count"], errors="coerce")
    dual = registry[(registry["domain"].eq("old_data")) & (original_count >= 2)].copy()
    if len(dual) != 17:
        raise ValueError(f"Expected 17 dual-image internal cases, found {len(dual)}")

    variant_root = Path(args.variant_root)
    rows: list[dict] = []
    for row in dual.to_dict(orient="records"):
        base_case_id = str(row["case_id"])
        original_case_id = str(row["original_case_id"]).removesuffix(".0")
        source = str(row["source_dataset"])
        first_matches = sorted(variant_root.glob(f"{source}_{original_case_id}_{original_case_id}-1-*"))
        if len(first_matches) != 1:
            raise FileNotFoundError(
                f"Expected one first image for {base_case_id}, found {len(first_matches)}: {first_matches}"
            )
        second_path = Path(str(row["image_path"]))
        if not second_path.exists():
            raise FileNotFoundError(second_path)
        authoritative_fold = int(fold_map.loc[base_case_id])
        for image_index, image_path in [(1, first_matches[0]), (2, second_path)]:
            item = dict(row)
            item["case_id"] = f"{base_case_id}__image{image_index}"
            item["original_case_id"] = base_case_id
            item["image_name"] = image_path.name
            item["image_path"] = str(image_path)
            item["master_fold_id"] = authoritative_fold
            item["selection_rule"] = f"dual_sensitivity_image{image_index}"
            item["image_count"] = 1
            item["original_image_count"] = 2
            item["image_exists"] = True
            rows.append(item)

    output = pd.DataFrame(rows).sort_values(["original_case_id", "case_id"]).reset_index(drop=True)
    if len(output) != 34 or output["case_id"].duplicated().any():
        raise RuntimeError("Dual-image registry integrity check failed.")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"dual_image_registry={output_path} cases={output['original_case_id'].nunique()} rows={len(output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
