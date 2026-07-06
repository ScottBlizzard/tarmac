from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v51_workflow_validation_20260527 as v51  # noqa: E402
import run_grosspath_rc_v54_constrained_policy_search_20260527 as v54  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v54_constrained_policy_search_20260527"
N_BOOT = 5000
SEED = 20260527 + 54


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def bootstrap_ci(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = stratified_indices(y, rng)
        m = v30.metrics_binary(y[idx], pred[idx])
        vals.append(m)
    boot = pd.DataFrame(vals)
    out: dict[str, float | int] = {"n_boot": N_BOOT}
    for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp"]:
        out[f"{key}_median"] = float(boot[key].median())
        out[f"{key}_ci025"] = float(boot[key].quantile(0.025))
        out[f"{key}_ci975"] = float(boot[key].quantile(0.975))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    grid, ext_reviews = v54.build_policy_grid(dev, ext, dev_scores, ext_scores)
    y = ext["label_idx"].to_numpy(dtype=int)
    rows = []

    key_policies = {
        "v50_sens98_spec90": None,
        "v54_low_control_highsens": "fusion_rank::dir_plus_pred_low_fn::budget=0.650",
        "v54_high_spec_safety": "buffer::all_direction::0.575+0.175",
    }

    for name, policy in key_policies.items():
        if name == "v50_sens98_spec90":
            v50_policy = [p for p in v51.POLICIES if p["policy"] == "v50_sens98_spec90"][0]
            review = v51.make_review(ext, ext_scores, v50_policy)
        else:
            review = ext_reviews[str(policy)]
        pred = v51.final_prediction(ext, review)
        m = v30.metrics_binary(y, pred)
        row = {
            "policy_name": name,
            "policy_rule": str(policy) if policy else "v50 base=0.525 + all_direction addon=0.200",
            "review_rate": float(review.mean()),
            **m,
            **bootstrap_ci(y, pred),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v54_key_policy_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    print(out[["policy_name", "review_rate", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp", "balanced_accuracy_ci025", "balanced_accuracy_ci975"]].to_string(index=False))


if __name__ == "__main__":
    main()
