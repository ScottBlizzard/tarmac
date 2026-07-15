from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler


PROJECT_SCRIPTS = Path(
    os.environ.get("THYMIC_PROJECT_SCRIPTS", "/workspace/thymic_project/scripts")
)
if not PROJECT_SCRIPTS.is_dir():
    PROJECT_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_SCRIPTS))

import run_task7_h3b_masked_gated_20260713 as h3


SAMPLERS = ("natural", "risk_balanced", "subtype_tempered")


def normalized_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0 or not np.isfinite(weights).all():
        raise ValueError("Invalid H10 sampling weights")
    if np.any(weights <= 0):
        raise ValueError("H10 sampling weights must be positive")
    weights /= weights.mean()
    if float(weights.max() / np.median(weights)) > 5.0 + 1e-12:
        raise ValueError("H10 sampler exceeds the preregistered max/median weight ratio")
    return weights


def risk_balanced_weights(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    labels = metadata.iloc[indices]["label_idx"].to_numpy(dtype=int)
    counts = pd.Series(labels).value_counts().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError(f"Both risks are required in every H10 train fold: {counts}")
    return normalized_weights(np.asarray([1.0 / counts[int(label)] for label in labels]))


def subtype_tempered_weights(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    subset = metadata.iloc[indices][["label_idx", "task_l6_label"]].reset_index(drop=True)
    weights = np.zeros(len(subset), dtype=np.float64)
    for risk, risk_rows in subset.groupby("label_idx", sort=True):
        del risk
        subtype_counts = risk_rows["task_l6_label"].value_counts(sort=False)
        target_mass = np.sqrt(subtype_counts.astype(float))
        target_mass /= target_mass.sum()
        for subtype, count in subtype_counts.items():
            positions = risk_rows.index[risk_rows["task_l6_label"] == subtype].to_numpy(int)
            weights[positions] = 0.5 * float(target_mass[subtype]) / int(count)
    return normalized_weights(weights)


def build_sampler(name: str):
    def sampler(metadata: pd.DataFrame, indices: np.ndarray, seed: int):
        generator = torch.Generator()
        generator.manual_seed(seed)
        if name == "natural":
            return RandomSampler(
                range(len(indices)), replacement=False, generator=generator
            )
        if name == "risk_balanced":
            weights = risk_balanced_weights(metadata, indices)
        elif name == "subtype_tempered":
            weights = subtype_tempered_weights(metadata, indices)
        else:
            raise ValueError(name)
        return WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(indices),
            replacement=True,
            generator=generator,
        )

    return sampler


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--h10-sampler", choices=SAMPLERS, required=True)
    wrapper, remaining = parser.parse_known_args()
    if "--split-mode" not in remaining:
        raise ValueError("H10 requires an explicit fivefold split mode")
    split_mode = remaining[remaining.index("--split-mode") + 1]
    if split_mode != "fivefold":
        raise ValueError("H10 internal sampler wrapper forbids source-LODO")
    h3.source_risk_sampler = build_sampler(wrapper.h10_sampler)
    sys.argv = [sys.argv[0], *remaining]
    h3.main()


if __name__ == "__main__":
    main()
