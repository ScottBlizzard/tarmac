from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter server-only H10 roles by case ID.")
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    case_ids = set(Path(args.case_ids).read_text(encoding="utf-8").split())
    roles = pd.read_csv(
        args.roles,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    subset = roles[roles["original_case_id"].isin(case_ids)].copy()
    subset.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        {
            "requested_ids": len(case_ids),
            "matched_h10_cases": len(subset),
            "subtypes": subset["task_l6_label"].value_counts().sort_index().to_dict(),
            "roles": subset["diagnostic_role"].value_counts().sort_index().to_dict(),
        }
    )


if __name__ == "__main__":
    main()
