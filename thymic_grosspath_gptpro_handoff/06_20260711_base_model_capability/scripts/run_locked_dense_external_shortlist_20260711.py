from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_SCRIPTS = Path("/workspace/thymic_project/scripts")
if str(PROJECT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPTS))

from evaluate_task7_dense_external_20260711 import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock an internal-only dense shortlist before external evaluation.")
    parser.add_argument("--internal-summary", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--external-bank-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--min-overall-bacc", type=float, default=0.68)
    parser.add_argument("--min-source-bacc", type=float, default=0.60)
    parser.add_argument(
        "--fusion-recipe",
        default="",
        help="Optional internal-only locked member manifest; rows are evaluated and fused in order.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def choose_candidates(summary: pd.DataFrame, max_candidates: int) -> list[tuple[str, pd.Series]]:
    selected: list[tuple[str, pd.Series]] = []
    used_tags: set[str] = set()

    def add(reason: str, row: pd.Series) -> None:
        tag = str(row["run_tag"])
        if tag not in used_tags and len(selected) < max_candidates:
            selected.append((reason, row))
            used_tags.add(tag)

    robustness = summary.sort_values(["min_source_bacc", "overall_bacc", "overall_auc"], ascending=False)
    if not robustness.empty:
        add("best_internal_worst_source", robustness.iloc[0])
    performance = summary.sort_values(["overall_bacc", "overall_auc", "min_source_bacc"], ascending=False)
    for _, row in performance.iterrows():
        if str(row["run_tag"]) not in used_tags:
            add("best_internal_overall_bacc", row)
            break
    first_model = str(selected[0][1]["model_name"]) if selected else ""
    diverse = summary[summary["model_name"].astype(str).ne(first_model)].sort_values(
        ["min_source_bacc", "overall_bacc", "overall_auc"], ascending=False
    )
    if not diverse.empty:
        add("best_distinct_backbone", diverse.iloc[0])
    for _, row in robustness.iterrows():
        if len(selected) >= max_candidates:
            break
        add("next_internal_robust_candidate", row)
    return selected


def run_command(arguments: list[str]) -> None:
    print("[command] " + " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.internal_summary)
    runs_root = Path(args.runs_root)
    bank_root = Path(args.external_bank_root)
    output_root = Path(args.output_root)
    bank_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary = summary[
        (pd.to_numeric(summary["overall_bacc"], errors="coerce") >= args.min_overall_bacc)
        & (pd.to_numeric(summary["min_source_bacc"], errors="coerce") >= args.min_source_bacc)
        & ~summary["views"].astype(str).str.contains("background_only", na=False)
        & ~summary["run_tag"].astype(str).str.startswith("210_")
    ].copy()
    if summary.empty:
        raise RuntimeError("No internal candidate passed the predeclared shortlist thresholds.")
    if args.fusion_recipe:
        recipe = pd.read_csv(args.fusion_recipe, encoding="utf-8-sig").sort_values("member_rank")
        if len(recipe) > args.max_candidates:
            raise ValueError(
                f"Locked recipe has {len(recipe)} members, exceeding --max-candidates={args.max_candidates}."
            )
        selected = []
        for tag in recipe["run_tag"].astype(str):
            matches = summary[summary["run_tag"].astype(str).eq(tag)]
            if matches.empty:
                raise RuntimeError(f"Locked fusion member is absent from the eligible summary: {tag}")
            selected.append(("locked_internal_equal_fusion", matches.iloc[0]))
    else:
        selected = choose_candidates(summary, args.max_candidates)
    manifest_rows = []
    for reason, row in selected:
        tag = str(row["run_tag"])
        run_dir = runs_root / tag
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        bank_config = config["feature_bank_config"]
        manifest_rows.append(
            {
                "selection_reason": reason,
                "run_tag": tag,
                "internal_run_dir": str(run_dir),
                "model_name": bank_config["model_name"],
                "views": ",".join(bank_config["views"]),
                "image_size": int(bank_config["image_size"]),
                "internal_overall_bacc": float(row["overall_bacc"]),
                "internal_overall_auc": float(row["overall_auc"]),
                "internal_min_source_bacc": float(row["min_source_bacc"]),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "LOCKED_DENSE_EXTERNAL_SHORTLIST.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, encoding="utf-8-sig")
        if existing.to_dict(orient="records") != manifest.to_dict(orient="records"):
            raise RuntimeError("A different locked external shortlist already exists; refusing to overwrite it.")
    else:
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        (output_root / "LOCKED_FUSION_RECIPE.json").write_text(
            json.dumps(
                {
                    "recipe": "equal_probability_average_of_all_locked_candidates",
                    "members": manifest["run_tag"].tolist(),
                    "selection_uses_external_labels": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print("[locked-shortlist]\n" + manifest.to_string(index=False), flush=True)

    evaluation_dirs = []
    for row in manifest.to_dict(orient="records"):
        bank_slug = safe_slug(f"{row['model_name']}__{row['views']}__{row['image_size']}")
        external_bank = bank_root / bank_slug
        config_path = external_bank / "feature_bank_config.json"
        complete = False
        if config_path.exists():
            complete = bool(json.loads(config_path.read_text(encoding="utf-8")).get("complete"))
        if not complete:
            run_command(
                [
                    sys.executable,
                    "/root/thymic_queue_scripts_20260711/extract_task7_dense_token_bank_20260711.py",
                    "--model-name",
                    row["model_name"],
                    "--image-size",
                    str(row["image_size"]),
                    "--views",
                    row["views"],
                    "--domains",
                    "strict_external,new_external_160",
                    "--output-dir",
                    str(external_bank),
                    "--batch-size",
                    "4",
                    "--num-workers",
                    "4",
                ]
            )
        evaluation_dir = output_root / row["run_tag"]
        evaluation_dirs.append((row["run_tag"], evaluation_dir))
        if not (evaluation_dir / "external_dense_metrics.csv").exists():
            run_command(
                [
                    sys.executable,
                    "/root/thymic_queue_scripts_20260711/evaluate_task7_dense_external_20260711.py",
                    "--internal-run-dir",
                    row["internal_run_dir"],
                    "--external-bank-dir",
                    str(external_bank),
                    "--output-dir",
                    str(evaluation_dir),
                    "--batch-size",
                    "24",
                    "--device",
                    args.device,
                ]
            )

    external_rows = []
    prediction_frames = []
    for candidate_index, (tag, evaluation_dir) in enumerate(evaluation_dirs):
        metrics = pd.read_csv(evaluation_dir / "external_dense_metrics.csv", encoding="utf-8-sig")
        for _, metric in metrics[metrics["group_type"].eq("domain")].iterrows():
            external_rows.append({"run_tag": tag, "domain": metric["group"], **metric.to_dict()})
        prediction = pd.read_csv(
            evaluation_dir / "external_dense_predictions.csv", dtype={"case_id": str}, encoding="utf-8-sig"
        )
        base_columns = ["case_id", "domain", "label_idx", "task_l6_label"]
        prediction_frames.append(
            prediction[base_columns + ["prob_high"]].rename(
                columns={"prob_high": f"prob_candidate_{candidate_index}"}
            )
        )
    pd.DataFrame(external_rows).to_csv(
        output_root / "locked_dense_external_single_model_summary.csv", index=False, encoding="utf-8-sig"
    )

    fused = prediction_frames[0]
    for frame in prediction_frames[1:]:
        fused = fused.merge(
            frame,
            on=["case_id", "domain", "label_idx", "task_l6_label"],
            how="inner",
            validate="one_to_one",
        )
    probability_columns = [column for column in fused.columns if column.startswith("prob_candidate_")]
    fused["prob_high"] = fused[probability_columns].mean(axis=1)
    fused["pred_idx"] = (fused["prob_high"] >= 0.5).astype(int)
    fused.to_csv(output_root / "locked_equal_fusion_external_predictions.csv", index=False, encoding="utf-8-sig")
    fusion_rows = []
    for domain, group in fused.groupby("domain"):
        fusion_rows.append({"domain": domain, **summarize(group, "prob_high")})
    fusion_summary = pd.DataFrame(fusion_rows)
    fusion_summary.to_csv(
        output_root / "locked_equal_fusion_external_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("[locked-fusion]\n" + fusion_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
