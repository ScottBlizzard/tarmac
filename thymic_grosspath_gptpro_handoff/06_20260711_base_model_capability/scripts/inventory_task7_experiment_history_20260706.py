from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def classify_direction(text: str) -> str:
    t = text.lower()
    tags = []
    for key, tag in [
        ("externalmimic", "external_mimic"),
        ("domainrobust", "domain_robust"),
        ("domain_label", "domain_label"),
        ("classsampler", "class_sampler"),
        ("meta_stack", "meta_stack"),
        ("stack", "stacking"),
        ("blend", "blend"),
        ("tta", "tta"),
        ("crop", "crop"),
        ("wpc", "whole_plus_crop"),
        ("dinov3", "dinov3"),
        ("dino", "dino"),
        ("siglip", "siglip"),
        ("convnext", "convnext"),
        ("eva", "eva"),
        ("aimv2", "aimv2"),
        ("vitamin", "vitamin"),
        ("clip", "clip"),
        ("qkvb", "qkvb"),
        ("whole", "whole"),
        ("last2", "last2blocks"),
        ("full_vlowlr", "full_finetune"),
        ("headonly", "headonly"),
        ("style", "style_aug"),
        ("sampler", "sampler"),
        ("guard", "guarded_selection"),
    ]:
        if key in t:
            tags.append(tag)
    return ";".join(dict.fromkeys(tags)) if tags else "unclassified"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def numeric(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    try:
        return float(row[col])
    except Exception:
        return np.nan


def pick_row(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=object)
    for col in ["selection_score", "all_balanced_accuracy", "balanced_accuracy", "test_case_mean_balanced_accuracy"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                return df.loc[vals.idxmax()]
    return df.iloc[0]


def summarize_csv(path: Path, project_root: Path) -> dict | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    run_dir = path.parent.name
    source = rel(path, project_root)
    text = f"{run_dir} {path.name}"
    row: dict[str, object] = {
        "run_dir": run_dir,
        "source_file": source,
        "direction_tags": classify_direction(text),
        "n_rows": int(len(df)),
        "columns": "|".join(map(str, df.columns)),
    }

    if "cv_fold_summary.csv" in path.name and "test_case_mean_balanced_accuracy" in df.columns:
        row.update(
            {
                "table_kind": "cv_fold_summary_mean",
                "all_balanced_accuracy": float(pd.to_numeric(df["test_case_mean_balanced_accuracy"], errors="coerce").mean()),
                "all_accuracy": float(pd.to_numeric(df.get("test_case_mean_accuracy"), errors="coerce").mean())
                if "test_case_mean_accuracy" in df.columns
                else np.nan,
                "primary_metric": float(pd.to_numeric(df.get("test_case_mean_primary_metric"), errors="coerce").mean())
                if "test_case_mean_primary_metric" in df.columns
                else np.nan,
                "folds": int(len(df)),
            }
        )
        return row

    best = pick_row(df)
    if best.empty:
        return None

    row["table_kind"] = "best_row_by_selection_or_balanced_accuracy"
    descriptor_cols = [
        "feature_set",
        "feature_model",
        "global_pool",
        "input_variant",
        "image_size",
        "estimator",
        "model",
        "weight_mode",
        "objective",
        "method",
        "weight",
        "subset",
        "source",
        "extra_tag",
        "threshold",
        "all_threshold",
        "old_threshold",
        "third_threshold",
    ]
    for col in descriptor_cols:
        if col in best.index:
            row[col] = best[col]

    for prefix in ["all", "old", "third", "adapt", "holdout"]:
        for metric in ["accuracy", "balanced_accuracy", "auc", "sensitivity_high", "specificity_low", "f1"]:
            row[f"{prefix}_{metric}"] = numeric(best, f"{prefix}_{metric}")
    for metric in ["accuracy", "balanced_accuracy", "auc", "sensitivity_high", "specificity_low", "f1"]:
        row[f"external_{metric}"] = numeric(best, metric)
    row["selection_score"] = numeric(best, "selection_score")
    row["old_guard_090"] = best.get("old_guard_090", np.nan)
    row["old_guard_092"] = best.get("old_guard_092", np.nan)
    row["folds"] = np.nan
    return row


def summarize_json(path: Path, project_root: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    flat = {}
    for key in ["boundary", "n_internal", "n_external_strict", "n_candidates"]:
        if key in data:
            flat[key] = data[key]
    if not flat:
        return None
    return {
        "run_dir": path.parent.name,
        "source_file": rel(path, project_root),
        "direction_tags": classify_direction(f"{path.parent.name} {path.name}"),
        "table_kind": "json_report_context",
        "n_rows": np.nan,
        **flat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/thymic_project")
    parser.add_argument("--out-dir", default="experiments/base_model_expansion_20260706/outputs/history_inventory")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [
        project_root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs",
        project_root / "outputs/batch1_batch2_task567_20260514/task7_external_runs",
        project_root / "outputs/grosspath_rc_v135_stage1_base_candidate_scan_20260527",
    ]
    patterns = [
        "*summary.csv",
        "cv_fold_summary.csv",
        "*report.json",
    ]
    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if path.suffix.lower() == ".csv":
                    item = summarize_csv(path, project_root)
                else:
                    item = summarize_json(path, project_root)
                if item:
                    rows.append(item)

    inv = pd.DataFrame(rows)
    inv.to_csv(out_dir / "historical_task7_experiment_inventory.csv", index=False, encoding="utf-8-sig")
    inv.to_json(out_dir / "historical_task7_experiment_inventory.json", orient="records", force_ascii=False, indent=2)

    score_cols = [c for c in inv.columns if c.endswith("balanced_accuracy") or c in ["selection_score", "primary_metric"]]
    sort_col = "third_balanced_accuracy" if "third_balanced_accuracy" in inv.columns else "all_balanced_accuracy"
    top = inv.copy()
    if sort_col in top.columns:
        top["_sort"] = pd.to_numeric(top[sort_col], errors="coerce")
        top = top.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    keep = [
        "run_dir",
        "source_file",
        "direction_tags",
        "table_kind",
        "selection_score",
        "all_balanced_accuracy",
        "old_balanced_accuracy",
        "third_balanced_accuracy",
        "adapt_balanced_accuracy",
        "holdout_balanced_accuracy",
        "external_balanced_accuracy",
        "primary_metric",
        "old_guard_092",
        "objective",
        "feature_model",
        "input_variant",
        "estimator",
        "model",
        "weight_mode",
        "method",
        "subset",
        "source",
        "extra_tag",
    ]
    keep = [c for c in keep if c in top.columns]
    top[keep].head(120).to_csv(out_dir / "historical_task7_top_rows.csv", index=False, encoding="utf-8-sig")

    by_tag_rows = []
    exploded = inv.copy()
    exploded["tag"] = exploded["direction_tags"].fillna("unclassified").astype(str).str.split(";")
    exploded = exploded.explode("tag")
    for tag, g in exploded.groupby("tag", sort=True):
        by_tag_rows.append(
            {
                "tag": tag,
                "n_tables": int(len(g)),
                "best_all_bacc": float(pd.to_numeric(g.get("all_balanced_accuracy"), errors="coerce").max())
                if "all_balanced_accuracy" in g.columns
                else np.nan,
                "best_third_bacc": float(pd.to_numeric(g.get("third_balanced_accuracy"), errors="coerce").max())
                if "third_balanced_accuracy" in g.columns
                else np.nan,
                "best_external_bacc": float(pd.to_numeric(g.get("external_balanced_accuracy"), errors="coerce").max())
                if "external_balanced_accuracy" in g.columns
                else np.nan,
            }
        )
    by_tag = pd.DataFrame(by_tag_rows)
    by_tag.to_csv(out_dir / "historical_task7_direction_tag_summary.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Historical Task7 Experiment Inventory",
        "",
        f"- Parsed rows: {len(inv)}",
        f"- Output CSV: `{out_dir / 'historical_task7_experiment_inventory.csv'}`",
        f"- Top rows CSV: `{out_dir / 'historical_task7_top_rows.csv'}`",
        "",
        "This is an automatic inventory of historical result tables. It is for triage, not a final manuscript table.",
        "",
        "## Best Direction Tags",
        "",
        dataframe_to_markdown(
            by_tag.sort_values(["best_external_bacc", "best_third_bacc", "best_all_bacc"], ascending=False).head(40)
        ),
    ]
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(inv[keep].head(40).to_string(index=False))
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
