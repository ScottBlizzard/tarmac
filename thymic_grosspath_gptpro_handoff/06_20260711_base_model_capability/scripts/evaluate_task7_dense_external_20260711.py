from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from run_task7_dense_feature_cv_20260711 import DenseTask7Model, SUBTYPE_NAMES, SUBTYPE_TO_INDEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked external evaluation for a frozen dense-feature CV run.")
    parser.add_argument("--internal-run-dir", required=True)
    parser.add_argument("--external-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    y_true = frame["label_idx"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, probability))
    except ValueError:
        auc = float("nan")
    return {
        "n": len(frame),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_model(config: dict, feature_shape: tuple[int, ...]) -> DenseTask7Model:
    _, num_views, _, feature_dim = feature_shape
    concept_count = len(config.get("concept_columns", [])) if float(config.get("concept_loss_weight", 0.0)) > 0 else 0
    return DenseTask7Model(
        feature_dim=feature_dim,
        num_views=num_views,
        hidden_dim=int(config.get("hidden_dim", 256)),
        attention_dim=int(config.get("attention_dim", 128)),
        dropout=float(config.get("dropout", 0.25)),
        pooling=str(config.get("pooling", "gated")),
        expert_mode=str(config.get("expert_mode", "none")),
        num_concepts=concept_count,
        num_groups=max(1, len(config.get("group_names", []))),
        prototype_temperature=float(config.get("prototype_temperature", 0.12)),
        boundary_fusion_alpha=float(config.get("boundary_fusion_alpha", 0.65)),
        domain_adversarial_lambda=float(config.get("domain_adversarial_lambda", 0.5)),
        risk_from_subtype_alpha=float(config.get("risk_from_subtype_alpha", 0.0)),
        sentinel_fusion_alpha=float(config.get("sentinel_fusion_alpha", 0.0)),
        mixstyle_probability=float(config.get("mixstyle_probability", 0.0)),
        mixstyle_alpha=float(config.get("mixstyle_alpha", 0.1)),
    )


def infer_fold(model: DenseTask7Model, features: np.ndarray, batch_size: int, device: torch.device):
    model.eval()
    risk_parts = []
    subtype_parts = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.array(features[start : start + batch_size], dtype=np.float32, copy=True)).to(device)
            output = model(batch)
            risk_parts.append(output["risk_log_prob"].exp()[:, 1].cpu().numpy())
            subtype_parts.append(torch.softmax(output["subtype_logits"], dim=1).cpu().numpy())
    return np.concatenate(risk_parts), np.concatenate(subtype_parts)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.internal_run_dir)
    bank_dir = Path(args.external_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    internal_bank_config = run_config["feature_bank_config"]
    external_bank_config = json.loads((bank_dir / "feature_bank_config.json").read_text(encoding="utf-8"))
    for key in ["model_name", "views", "image_size"]:
        if internal_bank_config.get(key) != external_bank_config.get(key):
            raise ValueError(
                f"Internal/external feature bank mismatch for {key}: "
                f"{internal_bank_config.get(key)} != {external_bank_config.get(key)}"
            )
    metadata = pd.read_csv(
        bank_dir / "metadata.csv", dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig"
    )
    metadata.columns = [str(column).lstrip("\ufeff") for column in metadata.columns]
    metadata["label_idx"] = pd.to_numeric(metadata["label_idx"], errors="raise").astype(int)
    metadata["subtype_idx"] = metadata["task_l6_label"].map(SUBTYPE_TO_INDEX).fillna(-1).astype(int)
    features = np.load(bank_dir / "dense_features.float16.npy", mmap_mode="r")
    device = torch.device(args.device)
    fold_probabilities = []
    fold_subtype_probabilities = []
    checkpoint_paths = sorted(run_dir.glob("fold_*/best_model.pt"))
    if len(checkpoint_paths) != 5:
        raise ValueError(f"Expected five fold checkpoints, found {len(checkpoint_paths)}")
    load_reports = []
    for checkpoint in checkpoint_paths:
        model = build_model(run_config, tuple(features.shape)).to(device)
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys for {checkpoint}: {unexpected[:10]}")
        risk_probability, subtype_probability = infer_fold(model, features, args.batch_size, device)
        fold_probabilities.append(risk_probability)
        fold_subtype_probabilities.append(subtype_probability)
        load_reports.append({"checkpoint": str(checkpoint), "missing_keys": missing})
        del model
        torch.cuda.empty_cache()

    prediction = metadata.copy()
    for fold_index, probability in enumerate(fold_probabilities, start=1):
        prediction[f"prob_high_fold_{fold_index}"] = probability
    prediction["prob_high"] = np.mean(np.stack(fold_probabilities), axis=0)
    subtype_probability = np.mean(np.stack(fold_subtype_probabilities), axis=0)
    for subtype_index, subtype_name in enumerate(SUBTYPE_NAMES):
        prediction[f"prob_subtype_{subtype_name}"] = subtype_probability[:, subtype_index]
    prediction["pred_idx"] = (prediction["prob_high"] >= 0.5).astype(int)
    prediction["pred_subtype_idx"] = np.argmax(subtype_probability, axis=1)
    prediction.to_csv(output_dir / "external_dense_predictions.csv", index=False, encoding="utf-8-sig")

    rows = [{"group_type": "overall", "group": "all", **summarize(prediction, "prob_high")}]
    for group_column in ["domain", "source_dataset"]:
        for group_name, group in prediction.groupby(group_column, dropna=False):
            rows.append(
                {"group_type": group_column, "group": str(group_name), **summarize(group, "prob_high")}
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "external_dense_metrics.csv", index=False, encoding="utf-8-sig")

    subtype_rows = []
    for subtype, group in prediction.groupby("task_l6_label"):
        evaluable_subtype = group["subtype_idx"] >= 0
        subtype_rows.append(
            {
                "subtype": subtype,
                "n": len(group),
                "risk_accuracy": float((group["pred_idx"] == group["label_idx"]).mean()),
                "mean_prob_high": float(group["prob_high"].mean()),
                "subtype_head_accuracy": (
                    float(
                        (
                            group.loc[evaluable_subtype, "pred_subtype_idx"]
                            == group.loc[evaluable_subtype, "subtype_idx"]
                        ).mean()
                    )
                    if evaluable_subtype.any()
                    else float("nan")
                ),
            }
        )
    pd.DataFrame(subtype_rows).to_csv(
        output_dir / "external_dense_subtype_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "internal_run_dir": str(run_dir),
                "external_bank_dir": str(bank_dir),
                "run_config": run_config,
                "external_bank_config": external_bank_config,
                "checkpoint_load_reports": load_reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
