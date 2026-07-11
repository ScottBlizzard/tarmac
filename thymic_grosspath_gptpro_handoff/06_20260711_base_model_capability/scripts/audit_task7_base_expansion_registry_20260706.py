from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()

    registry_csv = Path(args.registry_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(registry_csv, dtype=str)
    df["label_idx"] = df["label_idx"].astype(int)
    df["image_exists"] = df["image_path"].map(lambda p: Path(str(p)).exists())
    df["file_size"] = df["image_path"].map(lambda p: Path(str(p)).stat().st_size if Path(str(p)).exists() else None)
    if args.hash_images:
        df["sha256"] = df["image_path"].map(lambda p: sha256_file(Path(str(p))) if Path(str(p)).exists() else "")

    domain_counts = (
        df.groupby("domain")
        .agg(
            n=("case_id", "count"),
            low=("label_idx", lambda s: int((s == 0).sum())),
            high=("label_idx", lambda s: int((s == 1).sum())),
            missing_images=("image_exists", lambda s: int((~s).sum())),
            unique_case_ids=("case_id", "nunique"),
            unique_original_case_ids=("original_case_id", "nunique"),
        )
        .reset_index()
    )

    duplicate_case_ids = df[df.duplicated("case_id", keep=False)].sort_values(["case_id", "domain"])
    duplicate_image_paths = df[df.duplicated("image_path", keep=False)].sort_values(["image_path", "domain"])
    cross_domain_original = (
        df.groupby("original_case_id")["domain"].nunique().reset_index(name="domain_n")
    )
    cross_domain_original = cross_domain_original[cross_domain_original["domain_n"] > 1]
    cross_domain_original_cases = df[df["original_case_id"].isin(cross_domain_original["original_case_id"])].sort_values(
        ["original_case_id", "domain"]
    )

    domain_counts.to_csv(out_dir / "domain_counts.csv", index=False, encoding="utf-8-sig")
    duplicate_case_ids.to_csv(out_dir / "duplicate_case_ids.csv", index=False, encoding="utf-8-sig")
    duplicate_image_paths.to_csv(out_dir / "duplicate_image_paths.csv", index=False, encoding="utf-8-sig")
    cross_domain_original_cases.to_csv(out_dir / "cross_domain_original_case_id_collisions.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out_dir / "registry_with_audit_fields.csv", index=False, encoding="utf-8-sig")

    report = {
        "registry_csv": str(registry_csv),
        "n_rows": int(len(df)),
        "domain_counts": domain_counts.to_dict(orient="records"),
        "duplicate_case_id_rows": int(len(duplicate_case_ids)),
        "duplicate_image_path_rows": int(len(duplicate_image_paths)),
        "cross_domain_original_case_collision_rows": int(len(cross_domain_original_cases)),
        "missing_image_rows": int((~df["image_exists"]).sum()),
        "hashed_images": bool(args.hash_images),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Task7 Base Expansion Registry Audit",
        "",
        f"- Registry: `{registry_csv}`",
        f"- Rows: {len(df)}",
        f"- Missing images: {report['missing_image_rows']}",
        f"- Duplicate case-id rows: {report['duplicate_case_id_rows']}",
        f"- Duplicate image-path rows: {report['duplicate_image_path_rows']}",
        f"- Cross-domain original-case-id collision rows: {report['cross_domain_original_case_collision_rows']}",
        "",
        "## Domain Counts",
        "",
        dataframe_to_markdown(domain_counts),
        "",
        "## Output Files",
        "",
        "- `domain_counts.csv`",
        "- `duplicate_case_ids.csv`",
        "- `duplicate_image_paths.csv`",
        "- `cross_domain_original_case_id_collisions.csv`",
        "- `registry_with_audit_fields.csv`",
        "- `audit_report.json`",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
