from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_internal_oof_lodo_equal_fusions_20260711 import (
    load_predictions,
    protocol_metrics,
    validate_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize one named equal-weight internal fusion.")
    parser.add_argument("--manifest-csv", action="append", required=True)
    parser.add_argument("--members", required=True, help="Comma-separated manifest tags")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = [item.strip() for item in args.members.split(",") if item.strip()]
    if len(members) != len(set(members)) or not members:
        raise ValueError("Members must be a non-empty list of unique tags")
    manifest = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in args.manifest_csv],
        ignore_index=True,
    ).drop_duplicates("tag", keep="last")
    manifest = manifest.set_index("tag", drop=False)
    missing = sorted(set(members) - set(manifest.index.astype(str)))
    if missing:
        raise ValueError(f"Members missing from manifests: {missing}")

    candidates = {}
    references: dict[str, pd.DataFrame] = {}
    backbone_groups = []
    for tag in members:
        row = manifest.loc[tag]
        backbone_groups.append(str(row["backbone_group"]))
        candidate = {}
        for protocol, column in (("oof", "oof_path"), ("lodo", "lodo_path")):
            prediction = load_predictions(str(row[column]))
            if protocol not in references:
                references[protocol] = prediction
            else:
                validate_alignment(references[protocol], prediction, tag, protocol)
            candidate[protocol] = prediction["probability"].to_numpy(dtype=float)
        candidates[tag] = candidate

    if len(set(backbone_groups)) != len(backbone_groups):
        raise ValueError(f"Fusion does not use distinct backbone groups: {backbone_groups}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_row = {
        "member_count": len(members),
        "members": "+".join(members),
        "backbone_groups": "+".join(backbone_groups),
    }
    for protocol in ("oof", "lodo"):
        reference = references[protocol]
        probability = np.mean([candidates[tag][protocol] for tag in members], axis=0)
        metric_row.update(protocol_metrics(reference, probability, protocol))
        prediction = reference.drop(columns=["probability"]).copy()
        prediction["prob_high"] = probability
        prediction["pred_idx"] = (probability >= 0.5).astype(int)
        prediction["correct"] = prediction["pred_idx"].eq(prediction["label_idx"])
        prediction.to_csv(output_dir / f"{protocol}_predictions.csv", index=False, encoding="utf-8-sig")
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
                output_dir / f"{protocol}_subtype_metrics.csv", index=False, encoding="utf-8-sig"
            )

    metric_row["selection_score"] = (
        0.30 * metric_row["oof_bacc"]
        + 0.30 * metric_row["lodo_bacc"]
        + 0.25 * metric_row["lodo_min_source_bacc"]
        + 0.15 * metric_row["lodo_min_class_recall"]
    )
    metric_row["protocol_floor"] = min(
        metric_row["oof_bacc"],
        metric_row["lodo_bacc"],
        metric_row["lodo_min_source_bacc"],
        metric_row["lodo_min_class_recall"],
    )
    pd.DataFrame([metric_row]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    recipe = {
        "members": members,
        "backbone_groups": backbone_groups,
        "member_count": len(members),
        "weighting": "equal probability average",
        "selection_data": "internal fivefold OOF and canonical three-source LODO only",
        "search_space_size": 46871,
        "selection_score": metric_row["selection_score"],
        "protocol_floor": metric_row["protocol_floor"],
    }
    (output_dir / "recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame([metric_row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
