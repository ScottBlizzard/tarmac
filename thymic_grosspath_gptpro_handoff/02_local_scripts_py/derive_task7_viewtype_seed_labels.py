from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive Task7 view-type seed labels from P1/P2 manual review files.")
    parser.add_argument("--p1-csv", required=True)
    parser.add_argument("--p2-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def infer_p1_label(row: pd.Series) -> str:
    cut = str(row.get("cut_surface_degree", "")).strip().lower()
    outer = str(row.get("outer_surface_degree", "")).strip().lower()
    mixed = str(row.get("mixed_context", "")).strip().lower()
    if mixed == "yes":
        if cut == "weak" and outer == "strong":
            return "outer_surface"
        return "mixed"
    if cut == "strong" and outer in {"weak", "none", ""}:
        return "cut_surface"
    if outer == "strong" and cut in {"weak", "none", ""}:
        return "outer_surface"
    if cut == "moderate" and outer == "weak":
        return "cut_surface"
    return "mixed"


def main() -> None:
    args = parse_args()
    p1 = pd.read_csv(args.p1_csv, encoding="utf-8-sig")
    p2 = pd.read_csv(args.p2_csv, encoding="utf-8-sig")

    p1 = p1.copy()
    p1["training_case_id"] = p1["training_case_id"].astype(str)
    p1["view_type_seed"] = p1.apply(infer_p1_label, axis=1)
    p1["view_label_source"] = "p1_heuristic"

    p2 = p2.copy()
    p2["training_case_id"] = p2["training_case_id"].astype(str)
    p2 = p2[p2["view_type_round1"].fillna("").astype(str).str.strip() != ""].copy()
    p2["view_type_seed"] = p2["view_type_round1"].astype(str).str.strip()
    p2["view_label_source"] = "p2_manual"

    cols = ["training_case_id", "view_type_seed", "view_label_source"]
    merged = pd.concat([p1[cols], p2[cols]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["training_case_id"], keep="last").sort_values("training_case_id").reset_index(drop=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("Wrote:", out_path)
    print(merged["view_type_seed"].value_counts().to_string())


if __name__ == "__main__":
    main()
