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
import run_grosspath_rc_v59_lowrisk_boundary_specialist_20260527 as v59  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v59_lowrisk_boundary_specialist_20260527"
N_BOOT = 5000
SEED = 20260527 + 59


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def ci(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = stratified_indices(y, rng)
        vals.append(v30.metrics_binary(y[idx], pred[idx]))
    b = pd.DataFrame(vals)
    out: dict[str, float | int] = {"n_boot": N_BOOT}
    for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp"]:
        out[f"{key}_median"] = float(b[key].median())
        out[f"{key}_ci025"] = float(b[key].quantile(0.025))
        out[f"{key}_ci975"] = float(b[key].quantile(0.975))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    numeric, categorical, features = v59.feature_columns(dev, ext)
    dev_score, ext_score, _meta = v59.specialist_scores(dev, ext, features, numeric, categorical, "specialist_logistic")
    _grid, ext_reviews = v54.build_policy_grid(dev, ext, dev_scores, ext_scores)
    base_ext = ext_reviews[v59.BASE_POLICY]
    review = v59.add_pred_high_buffer(ext, base_ext, ext_score, 0.04)
    y = ext["label_idx"].to_numpy(dtype=int)
    pred = v51.final_prediction(ext, review)
    m = v30.metrics_binary(y, pred)
    row = {
        "policy_name": "v59_specialist_logistic_addon04",
        "review_rate": float(review.mean()),
        **m,
        **ci(y, pred),
    }
    out = pd.DataFrame([row])
    out.to_csv(OUT_DIR / "v59_selected_policy_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    print(out[["policy_name", "review_rate", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp", "balanced_accuracy_ci025", "balanced_accuracy_ci975"]].to_string(index=False))


if __name__ == "__main__":
    main()
