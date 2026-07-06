from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v2_robust_ensemble_20260526"


def auc_score(y: np.ndarray, prob: np.ndarray) -> float:
    pos = prob[y == 1]
    neg = prob[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(x > neg).mean() + 0.5 * (x == neg).mean() for x in pos]))


def summarize(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "auc": auc_score(y, prob),
        "accuracy": (tn + tp) / len(y),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def summarize_without_auc(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "accuracy": (tn + tp) / len(y),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> float:
    best_key: tuple[float, float] | None = None
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.951, 0.005):
        metrics = summarize_without_auc(y, prob, float(threshold))
        key = (float(metrics[objective]), -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def source_prob(frame: pd.DataFrame, sources: tuple[str, ...]) -> np.ndarray:
    return frame[list(sources)].mean(axis=1).to_numpy(dtype=float)


def add_eval_rows(rows: list[dict[str, object]], name: str, sources: tuple[str, ...], threshold: float, dev: pd.DataFrame, external: pd.DataFrame) -> None:
    for split_name, frame in [("development", dev), ("external_strict", external)]:
        y = frame["label_idx"].to_numpy(dtype=int)
        prob = source_prob(frame, sources)
        row = {
            "policy": name,
            "sources": "+".join(sources),
            "threshold_from_development": threshold,
            "split": split_name,
        }
        row.update(summarize(y, prob, threshold))
        rows.append(row)
    for domain, group in dev.groupby("domain"):
        y = group["label_idx"].to_numpy(dtype=int)
        prob = source_prob(group, sources)
        row = {
            "policy": name,
            "sources": "+".join(sources),
            "threshold_from_development": threshold,
            "split": f"development:{domain}",
        }
        row.update(summarize(y, prob, threshold))
        rows.append(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(IN_DIR / "v2_development_diagnostic_table.csv")
    external = pd.read_csv(IN_DIR / "v2_external_diagnostic_table.csv")
    candidate_sources = [
        "prob_base162",
        "prob103_vitl",
        "prob107_qkvb",
        "prob_mean_core",
        "prob_stack_plain",
        "prob_stack_balanced",
    ]
    candidate_sources = [col for col in candidate_sources if col in dev.columns and col in external.columns]

    candidates: list[dict[str, object]] = []
    for size in range(1, min(5, len(candidate_sources)) + 1):
        for sources in combinations(candidate_sources, size):
            prob = source_prob(dev, sources)
            threshold = choose_threshold(dev["label_idx"].to_numpy(dtype=int), prob, "balanced_accuracy")
            dev_metrics = summarize(dev["label_idx"].to_numpy(dtype=int), prob, threshold)
            ext_metrics = summarize(external["label_idx"].to_numpy(dtype=int), source_prob(external, sources), threshold)
            candidates.append(
                {
                    "sources": "+".join(sources),
                    "n_sources": len(sources),
                    "threshold": threshold,
                    "dev_accuracy": dev_metrics["accuracy"],
                    "dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
                    "dev_auc": dev_metrics["auc"],
                    "external_accuracy": ext_metrics["accuracy"],
                    "external_balanced_accuracy": ext_metrics["balanced_accuracy"],
                    "external_auc": ext_metrics["auc"],
                    "external_fn": ext_metrics["fn"],
                    "external_fp": ext_metrics["fp"],
                }
            )
    candidates_df = pd.DataFrame(candidates).sort_values(
        ["external_balanced_accuracy", "external_accuracy", "dev_balanced_accuracy"],
        ascending=False,
    )
    candidates_df.to_csv(OUT_DIR / "all_candidate_ensembles_diagnostic.csv", index=False, encoding="utf-8-sig")

    y_dev = dev["label_idx"].to_numpy(dtype=int)
    policies: list[tuple[str, tuple[str, ...], float]] = []
    anchor_sources = ("prob_base162",)
    policies.append(("domain_anchor_base162", anchor_sources, choose_threshold(y_dev, source_prob(dev, anchor_sources), "balanced_accuracy")))
    robust_sources = ("prob_base162", "prob103_vitl", "prob_mean_core")
    robust_sources = tuple(src for src in robust_sources if src in candidate_sources)
    policies.append(("diversity_robust_branch", robust_sources, choose_threshold(y_dev, source_prob(dev, robust_sources), "balanced_accuracy")))
    if {"prob_base162", "prob103_vitl", "prob107_qkvb"}.issubset(candidate_sources):
        tri_sources = ("prob_base162", "prob103_vitl", "prob107_qkvb")
        policies.append(("three_source_mean_branch", tri_sources, choose_threshold(y_dev, source_prob(dev, tri_sources), "balanced_accuracy")))

    rows: list[dict[str, object]] = []
    for name, sources, threshold in policies:
        add_eval_rows(rows, name, sources, threshold, dev, external)
    pd.DataFrame(rows).to_csv(OUT_DIR / "selected_policy_metrics.csv", index=False, encoding="utf-8-sig")

    print("[done]", OUT_DIR)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nTop diagnostic candidates by external balanced accuracy:")
    print(candidates_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
