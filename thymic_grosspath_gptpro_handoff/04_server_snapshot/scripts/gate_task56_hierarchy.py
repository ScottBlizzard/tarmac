from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


COARSE_CLASS_NAMES = ["A_AB", "B123", "TC"]
FINE_CLASS_NAMES = ["A", "AB", "B1", "B2", "B3", "TC"]
FINE_TO_COARSE = {
    "A": "A_AB",
    "AB": "A_AB",
    "B1": "B123",
    "B2": "B123",
    "B3": "B123",
    "TC": "TC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Task5-to-Task6 hierarchical gating.")
    parser.add_argument("--coarse-run", required=True)
    parser.add_argument("--fine-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta-grid", default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    parser.add_argument("--mix-grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def prob_cols(class_names: list[str]) -> list[str]:
    return [f"prob_{name}" for name in class_names]


def read_case_predictions(run_dir: Path, fold_id: int, split: str) -> pd.DataFrame:
    path = run_dir / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compute_metrics(frame: pd.DataFrame, prob_columns: list[str]) -> dict[str, float]:
    y_true = frame["label_idx"].to_numpy(dtype=int)
    y_pred = frame["pred_idx"].to_numpy(dtype=int)
    probs = frame[prob_columns].to_numpy(dtype=float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["macro_auc"] = float("nan")
    return metrics


def build_group_gate(coarse_frame: pd.DataFrame) -> np.ndarray:
    coarse_probs = coarse_frame[prob_cols(COARSE_CLASS_NAMES)].to_numpy(dtype=float)
    col_map = {name: idx for idx, name in enumerate(COARSE_CLASS_NAMES)}
    gate = np.zeros((len(coarse_frame), len(FINE_CLASS_NAMES)), dtype=float)
    for fine_idx, fine_name in enumerate(FINE_CLASS_NAMES):
        coarse_name = FINE_TO_COARSE[fine_name]
        gate[:, fine_idx] = coarse_probs[:, col_map[coarse_name]]
    return gate


def hierarchical_gate(
    fine_probs: np.ndarray,
    group_gate: np.ndarray,
    beta: float,
    mix: float,
    eps: float,
) -> np.ndarray:
    if beta == 0.0 or mix == 0.0:
        return fine_probs.copy()
    gated = fine_probs * np.power(group_gate + eps, beta)
    gated = gated / gated.sum(axis=1, keepdims=True)
    mixed = (1.0 - mix) * fine_probs + mix * gated
    mixed = mixed / mixed.sum(axis=1, keepdims=True)
    return mixed


def apply_gate_to_frames(
    coarse_frame: pd.DataFrame,
    fine_frame: pd.DataFrame,
    beta: float,
    mix: float,
    eps: float,
) -> pd.DataFrame:
    base_cols = [col for col in fine_frame.columns if not col.startswith("prob_") and col != "pred_idx"]
    coarse_prob_columns = prob_cols(COARSE_CLASS_NAMES)
    coarse_renamed = coarse_frame[["case_id"] + coarse_prob_columns].rename(
        columns={col: f"coarse_{col}" for col in coarse_prob_columns}
    )
    merged = fine_frame[base_cols + prob_cols(FINE_CLASS_NAMES)].merge(coarse_renamed, on="case_id", how="inner")
    if len(merged) != len(fine_frame):
        raise ValueError("Case alignment mismatch between coarse and fine predictions.")
    fine_probs = merged[prob_cols(FINE_CLASS_NAMES)].to_numpy(dtype=float)
    coarse_probs = merged[[f"coarse_{col}" for col in coarse_prob_columns]].rename(
        columns={f"coarse_{col}": col for col in coarse_prob_columns}
    )
    gate = build_group_gate(coarse_probs)
    out_probs = hierarchical_gate(fine_probs, gate, beta=beta, mix=mix, eps=eps)

    out = merged[base_cols].copy()
    for idx, col in enumerate(prob_cols(FINE_CLASS_NAMES)):
        out[col] = out_probs[:, idx]
    out["pred_idx"] = out_probs.argmax(axis=1)
    return out


def main() -> None:
    args = parse_args()
    coarse_run = Path(args.coarse_run)
    fine_run = Path(args.fine_run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_grid = [float(item.strip()) for item in args.beta_grid.split(",") if item.strip()]
    mix_grid = [float(item.strip()) for item in args.mix_grid.split(",") if item.strip()]
    fold_ids = sorted(int(path.name.split("_")[1]) for path in fine_run.glob("fold_*") if path.is_dir())

    prob_columns = prob_cols(FINE_CLASS_NAMES)
    fold_rows: list[dict[str, float | int]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        val_coarse = read_case_predictions(coarse_run, fold_id, "val")
        val_fine = read_case_predictions(fine_run, fold_id, "val")
        test_coarse = read_case_predictions(coarse_run, fold_id, "test")
        test_fine = read_case_predictions(fine_run, fold_id, "test")

        best = None
        for beta in beta_grid:
            for mix in mix_grid:
                gated_val = apply_gate_to_frames(val_coarse, val_fine, beta=beta, mix=mix, eps=args.eps)
                val_metrics = compute_metrics(gated_val, prob_columns)
                candidate = (
                    val_metrics["macro_f1"],
                    val_metrics["macro_auc"],
                    -abs(mix - 0.5),
                    -beta,
                    beta,
                    mix,
                    val_metrics,
                )
                if best is None or candidate > best:
                    best = candidate

        assert best is not None
        beta = best[4]
        mix = best[5]
        val_metrics = best[6]
        gated_test = apply_gate_to_frames(test_coarse, test_fine, beta=beta, mix=mix, eps=args.eps)
        test_metrics = compute_metrics(gated_test, prob_columns)
        gated_test.insert(0, "fold_id", fold_id)
        oof_frames.append(gated_test)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "beta": beta,
                "mix": mix,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_auc": val_metrics["macro_auc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_macro_auc": test_metrics["macro_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            }
        )

    oof = pd.concat(oof_frames, ignore_index=True)
    oof_metrics = compute_metrics(oof, prob_columns)
    oof.to_csv(output_dir / "oof_case_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "test_oof",
                "level": "case",
                "aggregation": "hierarchical_gate",
                **oof_metrics,
            }
        ]
    ).to_csv(output_dir / "oof_metrics.csv", index=False)

    print(f"Gated folds: {fold_ids}")
    print(
        "OOF hierarchical gate | "
        f"macro_f1={oof_metrics['macro_f1']:.4f} | "
        f"macro_auc={oof_metrics['macro_auc']:.4f} | "
        f"acc={oof_metrics['accuracy']:.4f} | "
        f"bacc={oof_metrics['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
