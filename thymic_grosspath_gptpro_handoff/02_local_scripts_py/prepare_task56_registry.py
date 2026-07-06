from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASK5_MAP = {
    "thymoma_A": "A_AB",
    "thymoma_AB": "A_AB",
    "thymoma_B1": "B123",
    "thymoma_B2": "B123",
    "thymoma_B3": "B123",
    "thymic_carcinoma": "TC",
}

TASK6_MAP = {
    "thymoma_A": "A",
    "thymoma_AB": "AB",
    "thymoma_B1": "B1",
    "thymoma_B2": "B2",
    "thymoma_B3": "B3",
    "thymic_carcinoma": "TC",
}

TASK7_MAP = {
    "thymoma_A": "low_risk_group",
    "thymoma_AB": "low_risk_group",
    "thymoma_B1": "low_risk_group",
    "thymoma_B2": "high_risk_group",
    "thymoma_B3": "high_risk_group",
    "thymic_carcinoma": "high_risk_group",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare derived registry for Task5/Task6/Task7.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    return parser.parse_args()


def build_summary(registry: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = registry.merge(split_df[["case_id", "master_fold_id"]], on="case_id", how="left")
    rows: list[dict[str, object]] = []
    for task_name, label_col in [
        ("task5_threeclass", "task_l5_label"),
        ("task6_sixclass", "task_l6_label"),
        ("task7_lowhigh_tc", "task_l7_label"),
    ]:
        filtered = merged[merged[label_col].astype(str).str.strip() != ""].copy()
        for label_name, group in filtered.groupby(label_col, dropna=False):
            row = {
                "task": task_name,
                "label_name": str(label_name),
                "num_cases": int(group["case_id"].nunique()),
                "num_images": int(group["image_count"].astype(int).sum()),
            }
            for fold_id, fold_group in group.groupby("master_fold_id", dropna=False):
                row[f"fold_{int(fold_id)}_cases"] = int(fold_group["case_id"].nunique())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "label_name"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry_csv, dtype={"case_id": str})
    split_df = pd.read_csv(args.split_csv, dtype={"case_id": str})

    derived = registry.copy()
    derived["task_l5_label"] = derived["who_type_raw"].map(TASK5_MAP).fillna("")
    derived["task_l6_label"] = derived["who_type_raw"].map(TASK6_MAP).fillna("")
    derived["task_l7_label"] = derived["who_type_raw"].map(TASK7_MAP).fillna("")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(output_csv, index=False)

    summary_df = build_summary(derived, split_df)
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote derived registry: {output_csv}")
    print(f"Wrote task summary: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
