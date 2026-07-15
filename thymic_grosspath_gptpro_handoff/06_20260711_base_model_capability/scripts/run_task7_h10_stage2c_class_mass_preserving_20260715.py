from __future__ import annotations

import numpy as np

import run_task7_h10_nested_phenotype_curriculum_20260715 as h10


ORIGINAL_CURRICULUM_WEIGHTS = h10.curriculum_weights


def class_mass_preserving_weights(
    base: np.ndarray, roles: np.ndarray, epoch: int
) -> tuple[np.ndarray, str]:
    weights, stage = ORIGINAL_CURRICULUM_WEIGHTS(base, roles, epoch)
    base_groups = np.unique(np.round(np.asarray(base, dtype=float), decimals=12))
    if len(base_groups) != 2:
        raise ValueError(f"Expected two risk-balanced base-weight groups: {base_groups}")
    balanced = np.zeros_like(weights, dtype=float)
    for value in base_groups:
        mask = np.isclose(base, value, atol=1e-10, rtol=0.0)
        if not np.any(mask):
            raise ValueError("Empty risk class during class-mass normalization")
        balanced[mask] = weights[mask] / weights[mask].sum()
    balanced *= len(weights) / 2.0
    masses = [float(balanced[np.isclose(base, value, atol=1e-10, rtol=0.0)].sum()) for value in base_groups]
    if not np.allclose(masses, len(weights) / 2.0, atol=1e-9, rtol=0.0):
        raise ValueError(f"Risk-class weight mass changed: {masses}")
    if not np.isclose(balanced.mean(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Class-mass-preserving weights must have mean one")
    return balanced, stage


h10.EXPERIMENT = "H10_STAGE2C_CLASS_MASS_PRESERVING_CURRICULUM_20260715"
h10.CANDIDATE = "H10_STAGE2C_CLASS_MASS_PRESERVING_CURRICULUM"
h10.curriculum_weights = class_mass_preserving_weights


if __name__ == "__main__":
    h10.main()
