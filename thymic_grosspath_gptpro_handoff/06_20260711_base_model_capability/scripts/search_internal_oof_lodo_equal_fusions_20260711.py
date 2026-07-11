from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equal-weight fusion search using internal OOF and source-LODO only.")
    parser.add_argument("--manifest-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--require-distinct-backbones", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-output-dir", default="")
    return parser.parse_args()


def canonical_source(values: pd.Series) -> pd.Series:
    source = values.fillna("unknown").astype(str)
    return source.mask(source.str.startswith("third_batch", na=False), "third_batch")


def load_predictions(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    probability_column = "prob_high" if "prob_high" in frame.columns else "prob_high_risk_group"
    required = {"case_id", "label_idx", probability_column, "source_dataset"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file {path} lacks columns {sorted(missing)}")
    columns = ["case_id", "label_idx", probability_column, "source_dataset"]
    columns.extend(
        column for column in ("original_case_id", "task_l6_label") if column in frame.columns
    )
    result = frame[columns].copy()
    result = result.rename(columns={probability_column: "probability"})
    result["label_idx"] = pd.to_numeric(result["label_idx"], errors="raise").astype(int)
    result["probability"] = pd.to_numeric(result["probability"], errors="raise").astype(float)
    result["source_dataset"] = canonical_source(result["source_dataset"])
    if result["case_id"].duplicated().any():
        raise ValueError(f"Prediction file has duplicate case ids: {path}")
    return result.sort_values("case_id").reset_index(drop=True)


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    sensitivity = float(((prediction == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1))
    specificity = float(((prediction == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1))
    return {
        "bacc": float(balanced_accuracy_score(labels, prediction)),
        "auc": float(roc_auc_score(labels, probability)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "min_class_recall": min(sensitivity, specificity),
    }


def protocol_metrics(reference: pd.DataFrame, probability: np.ndarray, prefix: str) -> dict[str, float]:
    labels = reference["label_idx"].to_numpy(dtype=int)
    result = {f"{prefix}_{key}": value for key, value in metrics(labels, probability).items()}
    source_bacc = []
    for source, indices in reference.groupby("source_dataset").groups.items():
        selected = np.asarray(list(indices), dtype=int)
        value = metrics(labels[selected], probability[selected])["bacc"]
        result[f"{prefix}_{source}_bacc"] = value
        source_bacc.append(value)
    result[f"{prefix}_min_source_bacc"] = min(source_bacc)
    return result


def validate_alignment(reference: pd.DataFrame, candidate: pd.DataFrame, tag: str, protocol: str) -> None:
    if not reference["case_id"].equals(candidate["case_id"]):
        raise ValueError(f"Case alignment mismatch for {tag} {protocol}")
    if not reference["label_idx"].equals(candidate["label_idx"]):
        raise ValueError(f"Label mismatch for {tag} {protocol}")
    if not reference["source_dataset"].equals(candidate["source_dataset"]):
        raise ValueError(f"Source mismatch for {tag} {protocol}")


def main() -> None:
    args = parse_args()
    manifest = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in args.manifest_csv],
        ignore_index=True,
    )
    required = {"tag", "backbone_group", "oof_path", "lodo_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest lacks columns {sorted(missing)}")

    candidates = {}
    oof_reference = None
    lodo_reference = None
    for row in manifest.itertuples(index=False):
        oof = load_predictions(str(row.oof_path))
        lodo = load_predictions(str(row.lodo_path))
        if oof_reference is None:
            oof_reference = oof
            lodo_reference = lodo
        else:
            validate_alignment(oof_reference, oof, str(row.tag), "oof")
            validate_alignment(lodo_reference, lodo, str(row.tag), "lodo")
        candidates[str(row.tag)] = {
            "backbone_group": str(row.backbone_group),
            "oof": oof["probability"].to_numpy(dtype=float),
            "lodo": lodo["probability"].to_numpy(dtype=float),
        }

    assert oof_reference is not None and lodo_reference is not None
    rows = []
    tags = sorted(candidates)
    for size in range(1, min(args.max_members, len(tags)) + 1):
        for members in itertools.combinations(tags, size):
            groups = [candidates[tag]["backbone_group"] for tag in members]
            if args.require_distinct_backbones and len(set(groups)) != len(groups):
                continue
            oof_probability = np.mean([candidates[tag]["oof"] for tag in members], axis=0)
            lodo_probability = np.mean([candidates[tag]["lodo"] for tag in members], axis=0)
            row = {
                "member_count": size,
                "members": "+".join(members),
                "backbone_groups": "+".join(groups),
            }
            row.update(protocol_metrics(oof_reference, oof_probability, "oof"))
            row.update(protocol_metrics(lodo_reference, lodo_probability, "lodo"))
            row["selection_score"] = (
                0.30 * row["oof_bacc"]
                + 0.30 * row["lodo_bacc"]
                + 0.25 * row["lodo_min_source_bacc"]
                + 0.15 * row["lodo_min_class_recall"]
            )
            row["protocol_floor"] = min(
                row["oof_bacc"],
                row["lodo_bacc"],
                row["lodo_min_source_bacc"],
                row["lodo_min_class_recall"],
            )
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "protocol_floor", "lodo_bacc", "oof_bacc"], ascending=False
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(results.head(30).to_string(index=False), flush=True)
    if args.top_output_dir:
        top_output_dir = Path(args.top_output_dir)
        top_output_dir.mkdir(parents=True, exist_ok=True)
        top = results.iloc[0]
        top_members = str(top["members"]).split("+")
        recipe = {
            "members": top_members,
            "backbone_groups": str(top["backbone_groups"]).split("+"),
            "member_count": int(top["member_count"]),
            "selection_score": float(top["selection_score"]),
            "protocol_floor": float(top["protocol_floor"]),
            "selection_data": "internal fivefold OOF and canonical three-source LODO only",
        }
        (top_output_dir / "recipe.json").write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame([top]).to_csv(top_output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        for protocol, reference in (("oof", oof_reference), ("lodo", lodo_reference)):
            probability = np.mean([candidates[tag][protocol] for tag in top_members], axis=0)
            prediction = reference.copy()
            prediction["prob_high"] = probability
            prediction["pred_idx"] = (probability >= 0.5).astype(int)
            prediction["correct"] = prediction["pred_idx"].eq(prediction["label_idx"])
            prediction.to_csv(
                top_output_dir / f"{protocol}_predictions.csv", index=False, encoding="utf-8-sig"
            )
            if "task_l6_label" in prediction.columns:
                subtype = (
                    prediction.groupby(["task_l6_label", "label_idx"], dropna=False)
                    .agg(
                        n=("case_id", "size"),
                        risk_accuracy=("correct", "mean"),
                        mean_prob_high=("prob_high", "mean"),
                    )
                    .reset_index()
                )
                subtype.to_csv(
                    top_output_dir / f"{protocol}_subtype_metrics.csv", index=False, encoding="utf-8-sig"
                )


if __name__ == "__main__":
    main()
