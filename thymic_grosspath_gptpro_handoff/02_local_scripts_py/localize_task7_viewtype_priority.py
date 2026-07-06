from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority-csv", required=True)
    parser.add_argument("--batch1-root", required=True)
    parser.add_argument("--batch2-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.priority_csv)
    batch1_root = Path(args.batch1_root)
    batch2_root = Path(args.batch2_root)

    def local_path(row: pd.Series) -> str:
        if row["source_dataset"] == "batch1":
            rel = row.get("selected_image_relpath", "")
            return str(batch1_root / rel) if rel else ""
        return str(batch2_root / row["training_image_name"])

    df["localized_review_path"] = df.apply(local_path, axis=1)
    df["localized_review_exists"] = df["localized_review_path"].apply(
        lambda p: Path(p).exists() if p else False
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
