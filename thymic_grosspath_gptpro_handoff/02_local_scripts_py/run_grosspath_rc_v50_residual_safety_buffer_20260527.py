from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v50_residual_safety_buffer_20260527"
FIG_DIR = OUT_DIR / "figures"
BASE_BUDGETS = np.round(np.arange(0.45, 0.625, 0.025), 3)
ADDON_RATES = np.round(np.arange(0.0, 0.225, 0.025), 3)
TARGETS = [0.93, 0.95, 0.97, 0.98]
SENSITIVITY_TARGETS = [0.93, 0.95, 0.97, 0.98]
SPECIFICITY_MINIMUMS = [0.85, 0.88, 0.90, 0.92]


def get_scores() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    dev = v30.load_development()
    ext = v30.load_external()
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    masks = v48.error_masks(dev)
    any_dev, any_ext, *_ = v48.generic_oof_external_scores(dev, ext, features, masks["any_wrong"])
    fn_dev, fn_ext, *_ = v48.generic_oof_external_scores(dev, ext, features, masks["fn_high_to_low"])
    fp_dev, fp_ext, *_ = v48.generic_oof_external_scores(dev, ext, features, masks["fp_low_to_high"])
    cond_dev = np.where(dev["p2_pred"].to_numpy(dtype=int) == 0, fn_dev, fp_dev)
    cond_ext = np.where(ext["p2_pred"].to_numpy(dtype=int) == 0, fn_ext, fp_ext)
    return dev, ext, {"any": any_dev, "fn": fn_dev, "fp": fp_dev, "direction": cond_dev}, {"any": any_ext, "fn": fn_ext, "fp": fp_ext, "direction": cond_ext}


def add_top_candidates(
    df: pd.DataFrame,
    base_review: np.ndarray,
    score: np.ndarray,
    addon_rate: float,
    candidate: str,
) -> np.ndarray:
    review = base_review.copy()
    n_add = int(round(len(df) * addon_rate))
    if n_add <= 0:
        return review
    p = df["p2_pred"].to_numpy(dtype=int)
    available = ~review
    if candidate == "pred_low_fn":
        available &= p == 0
    elif candidate == "pred_high_fp":
        available &= p == 1
    elif candidate == "all_direction":
        available &= True
    else:
        raise ValueError(candidate)
    idx = np.flatnonzero(available)
    if len(idx) == 0:
        return review
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    review[order[: min(n_add, len(order))]] = True
    return review


def evaluate(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review] = y[review]
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(df)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "review_precision_vs_p2_error": float((review & masks["any_wrong"]).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def run_grid(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    candidates = [
        ("pred_low_fn", "fn"),
        ("pred_high_fp", "fp"),
        ("all_direction", "direction"),
    ]
    for base_budget in BASE_BUDGETS:
        dev_base = v30.top_budget(dev_scores["any"], float(base_budget))
        ext_base = v30.top_budget(ext_scores["any"], float(base_budget))
        for addon_candidate, score_name in candidates:
            for addon_rate in ADDON_RATES:
                dev_review = add_top_candidates(dev, dev_base, dev_scores[score_name], float(addon_rate), addon_candidate)
                ext_review = add_top_candidates(ext, ext_base, ext_scores[score_name], float(addon_rate), addon_candidate)
                rows.append({"split": "development", "base_budget": float(base_budget), "addon_candidate": addon_candidate, "addon_rate": float(addon_rate), **evaluate(dev, dev_review)})
                rows.append({"split": "external", "base_budget": float(base_budget), "addon_candidate": addon_candidate, "addon_rate": float(addon_rate), **evaluate(ext, ext_review)})
    return pd.DataFrame(rows)


def select_targets(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dev = grid.loc[grid["split"].eq("development")].copy()
    ext = grid.loc[grid["split"].eq("external")].copy()
    for target in TARGETS:
        ok = dev.loc[dev["balanced_accuracy"].ge(target)].copy()
        if ok.empty:
            chosen = dev.sort_values("balanced_accuracy", ascending=False).iloc[0]
        else:
            chosen = ok.sort_values(["review_rate", "base_budget", "addon_rate", "review_precision_vs_p2_error"], ascending=[True, True, True, False]).iloc[0]
        matched = ext.loc[
            ext["base_budget"].eq(chosen["base_budget"])
            & ext["addon_candidate"].eq(chosen["addon_candidate"])
            & ext["addon_rate"].eq(chosen["addon_rate"])
        ].iloc[0]
        rows.append(
            {
                "target_dev_bacc": float(target),
                **{f"dev_{k}": v for k, v in chosen.items() if k != "split"},
                **{f"external_{k}": v for k, v in matched.items() if k != "split"},
            }
        )
    return pd.DataFrame(rows)


def select_sensitivity_first(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dev = grid.loc[grid["split"].eq("development")].copy()
    ext = grid.loc[grid["split"].eq("external")].copy()
    for sens_target in SENSITIVITY_TARGETS:
        for spec_min in SPECIFICITY_MINIMUMS:
            ok = dev.loc[dev["sensitivity"].ge(sens_target) & dev["specificity"].ge(spec_min)].copy()
            if ok.empty:
                continue
            chosen = ok.sort_values(
                ["review_rate", "base_budget", "addon_rate", "balanced_accuracy"],
                ascending=[True, True, True, False],
            ).iloc[0]
            matched = ext.loc[
                ext["base_budget"].eq(chosen["base_budget"])
                & ext["addon_candidate"].eq(chosen["addon_candidate"])
                & ext["addon_rate"].eq(chosen["addon_rate"])
            ].iloc[0]
            rows.append(
                {
                    "dev_sensitivity_target": float(sens_target),
                    "dev_specificity_min": float(spec_min),
                    **{f"dev_{k}": v for k, v in chosen.items() if k != "split"},
                    **{f"external_{k}": v for k, v in matched.items() if k != "split"},
                }
            )
    return pd.DataFrame(rows)


def make_plot(grid: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = grid.loc[grid["split"].eq("external")].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    for candidate in ["pred_low_fn", "pred_high_fp", "all_direction"]:
        sub = ext.loc[(ext["addon_candidate"].eq(candidate)) & (ext["base_budget"].eq(0.60))].sort_values("review_rate")
        if sub.empty:
            continue
        ax.plot(sub["review_rate"] * 100, sub["balanced_accuracy"] * 100, marker="o", linewidth=1.7, label=f"base60 + {candidate}")
    ax.axhline(93, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(95, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review rate (%)")
    ax.set_ylabel("External workflow balanced accuracy (%)")
    ax.set_title("Residual safety buffer after v37 hard-gate review")
    ax.set_xlim(42, 85)
    ax.set_ylim(88, 98.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v50_residual_safety_buffer_external_bacc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v50_residual_safety_buffer_external_bacc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = get_scores()
    grid = run_grid(dev, ext, dev_scores, ext_scores)
    selected = select_targets(grid)
    sensitivity_selected = select_sensitivity_first(grid)
    grid.to_csv(OUT_DIR / "v50_residual_safety_buffer_grid.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v50_dev_selected_residual_safety_buffer.csv", index=False, encoding="utf-8-sig")
    sensitivity_selected.to_csv(OUT_DIR / "v50_sensitivity_first_dev_selected.csv", index=False, encoding="utf-8-sig")
    make_plot(grid)

    show_cols = [
        "target_dev_bacc",
        "dev_base_budget",
        "dev_addon_candidate",
        "dev_addon_rate",
        "external_review_rate",
        "external_balanced_accuracy",
        "external_accuracy",
        "external_fn",
        "external_fp",
        "external_captured_fn_n",
        "external_captured_fp_n",
    ]
    print(selected[show_cols].to_string(index=False))
    if not sensitivity_selected.empty:
        sens_show_cols = [
            "dev_sensitivity_target",
            "dev_specificity_min",
            "dev_base_budget",
            "dev_addon_candidate",
            "dev_addon_rate",
            "external_review_rate",
            "external_sensitivity",
            "external_specificity",
            "external_balanced_accuracy",
            "external_accuracy",
            "external_fn",
            "external_fp",
        ]
        print("\nSensitivity-first selection:")
        print(sensitivity_selected[sens_show_cols].to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
