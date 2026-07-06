from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict fold-wise validation-selected blending for Task7 binary runs."
    )
    parser.add_argument(
        "--run-dirs",
        required=True,
        help="Comma-separated run directories; each must contain fold_<k>/val_case_predictions_mean.csv and test_case_predictions_mean.csv",
    )
    parser.add_argument(
        "--selection-metric",
        default="accuracy",
        choices=("accuracy", "balanced_accuracy", "f1"),
    )
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def load_fold_df(run_dir: Path, fold_id: int, split: str) -> pd.DataFrame:
    file_path = run_dir / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv"
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path)
    need = {
        "case_id": "case_id",
        "label_idx": "label",
        "prob_high_risk_group": "prob",
    }
    missing = [col for col in need if col not in df.columns]
    if missing:
        raise KeyError(f"{file_path} missing columns: {missing}")
    return df[list(need)].rename(columns=need)


def generate_weights(n_models: int, step: float) -> Iterable[tuple[float, ...]]:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    if n_models == 1:
        yield (1.0,)
        return
    if n_models == 2:
        for a in vals:
            yield (round(float(a), 10), round(float(1.0 - a), 10))
        return
    if n_models == 3:
        for a in vals:
            for b in vals:
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                if abs(round(c / step) - (c / step)) < 1e-9 and c <= 1.0 + 1e-9:
                    c = max(0.0, float(round(c, 10)))
                    yield (float(round(a, 10)), float(round(b, 10)), c)
        return
    raise ValueError("Only 1-3 models supported in this helper.")


def score_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float, metric: str) -> float:
    pred = (probs >= threshold).astype(int)
    if metric == "accuracy":
        return accuracy_score(y_true, pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, pred)
    if metric == "f1":
        return f1_score(y_true, pred)
    raise ValueError(metric)


def find_best_on_validation(val_frames: list[pd.DataFrame], metric: str, step: float) -> tuple[tuple[float, ...], float, float]:
    merged = val_frames[0][["case_id", "label"]].copy()
    for idx, df in enumerate(val_frames):
        merged = merged.merge(df[["case_id", "prob"]].rename(columns={"prob": f"prob_{idx}"}), on="case_id")
    y_true = merged["label"].to_numpy()
    prob_matrix = np.stack([merged[f"prob_{idx}"].to_numpy() for idx in range(len(val_frames))], axis=1)

    best: tuple[tuple[float, float, float], tuple[float, ...], float, float] | None = None
    for weights in generate_weights(prob_matrix.shape[1], step):
        probs = prob_matrix @ np.asarray(weights, dtype=np.float64)
        vals = sorted(set([0.0, 1.0] + [float(x) for x in probs]))
        thresholds = [0.5]
        thresholds.extend((a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:]))
        thresholds = sorted(set(thresholds + vals))
        for thr in thresholds:
            score = score_predictions(y_true, probs, thr, metric)
            key = (score, -abs(thr - 0.5), max(weights))
            if best is None or key > best[0]:
                best = (key, weights, float(thr), float(score))
    assert best is not None
    return best[1], best[2], best[3]


def main() -> None:
    args = parse_args()
    run_dirs = [Path(item.strip()) for item in args.run_dirs.split(",") if item.strip()]
    if not run_dirs:
        raise ValueError("No run dirs given.")

    tests: list[pd.DataFrame] = []
    choices: list[dict[str, object]] = []
    for fold_id in range(1, 6):
        val_frames = [load_fold_df(run_dir, fold_id, "val") for run_dir in run_dirs]
        test_frames = [load_fold_df(run_dir, fold_id, "test") for run_dir in run_dirs]
        weights, threshold, val_score = find_best_on_validation(
            val_frames,
            metric=args.selection_metric,
            step=args.weight_step,
        )

        merged = test_frames[0][["case_id", "label"]].copy()
        for idx, df in enumerate(test_frames):
            merged = merged.merge(df[["case_id", "prob"]].rename(columns={"prob": f"prob_{idx}"}), on="case_id")
        prob_matrix = np.stack([merged[f"prob_{idx}"].to_numpy() for idx in range(len(test_frames))], axis=1)
        probs = prob_matrix @ np.asarray(weights, dtype=np.float64)
        pred = (probs >= threshold).astype(int)
        out = merged[["case_id", "label"]].copy()
        out["prob"] = probs
        out["pred"] = pred
        tests.append(out)
        choices.append(
            {
                "fold": fold_id,
                "weights": {run_dirs[idx].name: float(weights[idx]) for idx in range(len(run_dirs))},
                "threshold": float(threshold),
                "val_score": float(val_score),
            }
        )

    all_df = pd.concat(tests, ignore_index=True).drop_duplicates("case_id")
    y_true = all_df["label"].to_numpy()
    probs = all_df["prob"].to_numpy()
    pred = all_df["pred"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result = {
        "selection_metric": args.selection_metric,
        "run_dirs": [str(item) for item in run_dirs],
        "auc": float(roc_auc_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "choices": choices,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
