from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

from run_task7_dense_feature_cv_20260711 import DenseTask7Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate first/selected/two-image OOF sensitivity.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(config: dict, feature_shape: tuple[int, ...]) -> DenseTask7Model:
    _, num_views, _, feature_dim = feature_shape
    concept_count = (
        len(config.get("concept_columns", []))
        if float(config.get("concept_loss_weight", 0.0)) > 0
        else 0
    )
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


def infer(model: DenseTask7Model, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.array(features[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            parts.append(model(batch)["risk_log_prob"].exp()[:, 1].cpu().numpy())
    return np.concatenate(parts)


def summarize(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    y_true = frame["label_idx"].to_numpy(dtype=int)
    predicted = (frame[probability_column].to_numpy(dtype=float) >= 0.5).astype(int)
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "errors": int((predicted != y_true).sum()),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    bank_dir = Path(args.feature_bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        bank_dir / "metadata.csv",
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    metadata.columns = [str(column).lstrip("\ufeff") for column in metadata.columns]
    features = np.load(bank_dir / "dense_features.float16.npy", mmap_mode="r")
    if len(metadata) != 34 or metadata["original_case_id"].nunique() != 17:
        raise ValueError("Expected 34 images from 17 dual-image cases.")
    device = torch.device(args.device)
    metadata["prob_high_oof"] = np.nan
    for fold in range(1, 6):
        row_indices = np.flatnonzero(
            pd.to_numeric(metadata["master_fold_id"], errors="raise").to_numpy(dtype=int) == fold
        )
        if not len(row_indices):
            continue
        model = build_model(config, tuple(features.shape)).to(device)
        checkpoint = run_dir / f"fold_{fold}" / "best_model.pt"
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {unexpected[:10]}")
        metadata.loc[row_indices, "prob_high_oof"] = infer(
            model, features[row_indices], args.batch_size, device
        )
        del model
        torch.cuda.empty_cache()
    if metadata["prob_high_oof"].isna().any():
        raise RuntimeError("Some dual-image rows did not receive an OOF prediction.")

    metadata["image_index"] = metadata["case_id"].str.extract(
        r"__image([12])$", expand=False
    ).astype(int)
    wide = metadata.pivot(
        index="original_case_id", columns="image_index", values="prob_high_oof"
    ).rename(columns={1: "prob_image1", 2: "prob_selected_image2"})
    labels = metadata.groupby("original_case_id", as_index=True)["label_idx"].first().astype(int)
    folds = metadata.groupby("original_case_id", as_index=True)["master_fold_id"].first().astype(int)
    wide = wide.join(labels).join(folds).reset_index()
    wide["prob_two_image_mean"] = wide[["prob_image1", "prob_selected_image2"]].mean(axis=1)

    historical = pd.read_csv(
        run_dir / "oof_predictions.csv", dtype={"case_id": str}, encoding="utf-8-sig"
    )[["case_id", "prob_high"]].rename(
        columns={"case_id": "original_case_id", "prob_high": "historical_selected_oof_prob"}
    )
    wide = wide.merge(historical, on="original_case_id", how="left", validate="one_to_one")
    wide["selected_reextract_abs_diff"] = (
        wide["prob_selected_image2"] - wide["historical_selected_oof_prob"]
    ).abs()
    if float(wide["selected_reextract_abs_diff"].max()) > 1e-3:
        raise RuntimeError("Re-extracted selected-image probabilities do not reproduce the locked OOF run.")

    for name in ["image1", "selected_image2", "two_image_mean"]:
        wide[f"pred_{name}"] = (wide[f"prob_{name}"] >= 0.5).astype(int)
        wide[f"correct_{name}"] = (wide[f"pred_{name}"] == wide["label_idx"]).astype(int)
    wide.to_csv(output_dir / "dual_image_oof_predictions.csv", index=False, encoding="utf-8-sig")

    rows = []
    for name in ["image1", "selected_image2", "two_image_mean"]:
        rows.append({"strategy": name, **summarize(wide, f"prob_{name}")})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "dual_image_oof_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
