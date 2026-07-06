from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_AI_COLS = [
    "ai_visual_summary",
    "ai_color_tone",
    "ai_shape_margin",
    "ai_texture_pattern",
    "ai_salient_clues",
    "ai_background_note",
    "ai_blind_guess_task5",
    "ai_blind_guess_task6",
    "ai_guess_confidence",
    "ai_priority_note",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    parts = []
    for path in sorted(batch_dir.glob("visual_review_batch_*.csv")):
        df = pd.read_csv(path)
        df["source_batch_file"] = path.name
        parts.append(df)
        missing = {col: int(df[col].fillna("").astype(str).eq("").sum()) for col in REQUIRED_AI_COLS if col in df.columns}
        print(path.name, "rows", len(df), "missing", missing)

    if not parts:
        raise FileNotFoundError(f"No batch CSVs found in {batch_dir}")

    merged = pd.concat(parts, ignore_index=True).sort_values("review_index")
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print("written", out)
    print("rows", len(merged))


if __name__ == "__main__":
    main()
