from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from search_internal_oof_lodo_equal_fusions_20260711 import load_predictions, validate_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fivefold meta-crossfit sensitivity analysis for equal-weight internal fusion selection."
    )
    parser.add_argument("--manifest-csv", action="append", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-members", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser.parse_args()


def vector_metrics(labels: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = probability >= 0.5
    positive = labels == 1
    negative = ~positive
    sensitivity = prediction[:, positive].mean(axis=1)
    specificity = (~prediction[:, negative]).mean(axis=1)
    return (sensitivity + specificity) / 2.0, sensitivity, specificity


def source_min_bacc(
    labels: np.ndarray, probability: np.ndarray, source: np.ndarray
) -> np.ndarray:
    values = []
    for source_name in sorted(np.unique(source)):
        selected = source == source_name
        bacc, _, _ = vector_metrics(labels[selected], probability[:, selected])
        values.append(bacc)
    return np.min(np.stack(values, axis=1), axis=1)


def scalar_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    labels = frame["label_idx"].to_numpy(dtype=int)
    source = frame["source_dataset"].to_numpy(dtype=str)
    bacc, sensitivity, specificity = vector_metrics(labels, probability[None, :])
    source_min = source_min_bacc(labels, probability[None, :], source)
    return {
        "balanced_accuracy": float(bacc[0]),
        "auc": float(roc_auc_score(labels, probability)),
        "sensitivity": float(sensitivity[0]),
        "specificity": float(specificity[0]),
        "min_class_recall": float(min(sensitivity[0], specificity[0])),
        "min_source_bacc": float(source_min[0]),
    }


def metric_rows(protocol: str, frame: pd.DataFrame, probability: np.ndarray) -> list[dict]:
    rows = [{"protocol": protocol, "group_type": "overall", "group": "all", **scalar_metrics(frame, probability)}]
    for source_name, indices in frame.groupby("source_dataset").groups.items():
        selected = np.asarray(list(indices), dtype=int)
        rows.append(
            {
                "protocol": protocol,
                "group_type": "source_dataset",
                "group": source_name,
                **scalar_metrics(frame.iloc[selected].reset_index(drop=True), probability[selected]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    manifest = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in args.manifest_csv],
        ignore_index=True,
    ).drop_duplicates("tag", keep="last")
    tags = manifest["tag"].astype(str).tolist()
    groups = manifest["backbone_group"].astype(str).tolist()
    candidates: dict[str, dict[str, np.ndarray]] = {}
    references: dict[str, pd.DataFrame] = {}
    for row in manifest.itertuples(index=False):
        tag = str(row.tag)
        candidates[tag] = {}
        for protocol, path in (("oof", row.oof_path), ("lodo", row.lodo_path)):
            prediction = load_predictions(str(path))
            if protocol not in references:
                references[protocol] = prediction
            else:
                validate_alignment(references[protocol], prediction, tag, protocol)
            candidates[tag][protocol] = prediction["probability"].to_numpy(dtype=np.float32)

    split = pd.read_csv(args.split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split.columns = [str(column).lstrip("\ufeff") for column in split.columns]
    fold_map = split.set_index("case_id")["master_fold_id"]
    reference = references["oof"]
    fold_id = reference["case_id"].map(fold_map)
    if fold_id.isna().any():
        missing = reference.loc[fold_id.isna(), "case_id"].tolist()[:10]
        raise ValueError(f"Cases missing from split registry: {missing}")
    folds = fold_id.astype(int).to_numpy()

    combinations = []
    for size in range(1, min(args.max_members, len(tags)) + 1):
        for member_indices in itertools.combinations(range(len(tags)), size):
            member_groups = [groups[index] for index in member_indices]
            if len(set(member_groups)) != len(member_groups):
                continue
            combinations.append(member_indices)
    oof_base = np.stack([candidates[tag]["oof"] for tag in tags], axis=0)
    lodo_base = np.stack([candidates[tag]["lodo"] for tag in tags], axis=0)
    fused_oof = np.empty((len(combinations), len(reference)), dtype=np.float32)
    fused_lodo = np.empty_like(fused_oof)
    for index, member_indices in enumerate(combinations):
        fused_oof[index] = oof_base[list(member_indices)].mean(axis=0)
        fused_lodo[index] = lodo_base[list(member_indices)].mean(axis=0)

    selected_rows = []
    crossfit_oof = np.full(len(reference), np.nan, dtype=np.float32)
    crossfit_lodo = np.full(len(reference), np.nan, dtype=np.float32)
    for held_fold in sorted(np.unique(folds)):
        train_mask = folds != held_fold
        held_mask = ~train_mask
        labels = reference.loc[train_mask, "label_idx"].to_numpy(dtype=int)
        source = reference.loc[train_mask, "source_dataset"].to_numpy(dtype=str)
        best_key = None
        best_index = None
        best_values = None
        for start in range(0, len(combinations), args.chunk_size):
            stop = min(start + args.chunk_size, len(combinations))
            oof_probability = fused_oof[start:stop, train_mask]
            lodo_probability = fused_lodo[start:stop, train_mask]
            oof_bacc, _, _ = vector_metrics(labels, oof_probability)
            lodo_bacc, lodo_sensitivity, lodo_specificity = vector_metrics(labels, lodo_probability)
            lodo_min_source = source_min_bacc(labels, lodo_probability, source)
            lodo_min_class = np.minimum(lodo_sensitivity, lodo_specificity)
            score = 0.30 * oof_bacc + 0.30 * lodo_bacc + 0.25 * lodo_min_source + 0.15 * lodo_min_class
            floor = np.minimum.reduce((oof_bacc, lodo_bacc, lodo_min_source, lodo_min_class))
            order = np.lexsort((oof_bacc, lodo_bacc, floor, score))
            local = int(order[-1])
            key = (float(score[local]), float(floor[local]), float(lodo_bacc[local]), float(oof_bacc[local]))
            if best_key is None or key > best_key:
                best_key = key
                best_index = start + local
                best_values = {
                    "selection_score": key[0],
                    "protocol_floor": key[1],
                    "training_lodo_bacc": key[2],
                    "training_oof_bacc": key[3],
                    "training_lodo_min_source_bacc": float(lodo_min_source[local]),
                    "training_lodo_min_class_recall": float(lodo_min_class[local]),
                }
        assert best_index is not None and best_values is not None
        member_indices = combinations[best_index]
        member_tags = [tags[index] for index in member_indices]
        crossfit_oof[held_mask] = fused_oof[best_index, held_mask]
        crossfit_lodo[held_mask] = fused_lodo[best_index, held_mask]
        selected_rows.append(
            {
                "held_meta_fold": int(held_fold),
                "held_n": int(held_mask.sum()),
                "member_count": len(member_tags),
                "members": "+".join(member_tags),
                "backbone_groups": "+".join(groups[index] for index in member_indices),
                **best_values,
            }
        )

    if not np.isfinite(crossfit_oof).all() or not np.isfinite(crossfit_lodo).all():
        raise RuntimeError("Crossfit fusion left missing probabilities")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(
        output_dir / "meta_fold_selected_recipes.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    for protocol, probability in (("oof", crossfit_oof), ("lodo", crossfit_lodo)):
        prediction = references[protocol].drop(columns=["probability"]).copy()
        prediction["master_fold_id"] = folds
        prediction["prob_high"] = probability
        prediction["pred_idx"] = (probability >= 0.5).astype(int)
        prediction["correct"] = prediction["pred_idx"].eq(prediction["label_idx"])
        prediction.to_csv(
            output_dir / f"meta_crossfit_{protocol}_predictions.csv", index=False, encoding="utf-8-sig"
        )
        rows.extend(metric_rows(protocol, prediction, probability))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "meta_crossfit_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "search_space.txt").write_text(
        f"candidate_models={len(tags)}\nvalid_equal_weight_combinations={len(combinations)}\nmax_members={args.max_members}\n",
        encoding="ascii",
    )
    print(pd.DataFrame(selected_rows).to_string(index=False), flush=True)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
