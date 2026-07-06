from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v64_conformal_autopass_20260527 as v64  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v65_rank_normalized_conformal_autopass_20260527"
FIG_DIR = OUT_DIR / "figures"
SEED = 20260528
N_SPLITS = 300
CAL_FRAC = 0.50
TARGET_ERRORS = [0.05, 0.10, 0.15]
MIN_AUTO_N = 20


def select_coverage(
    score: np.ndarray,
    wrong: np.ndarray,
    indices: np.ndarray,
    target_error: float,
) -> dict[str, float | int]:
    score_cal = score[indices]
    wrong_cal = wrong[indices].astype(int)
    finite = np.isfinite(score_cal)
    score_cal = score_cal[finite]
    wrong_cal = wrong_cal[finite]
    if len(score_cal) == 0:
        return empty_selection()
    order = np.argsort(-score_cal, kind="mergesort")
    sorted_wrong = wrong_cal[order]
    sorted_score = score_cal[order]
    cum_errors = np.cumsum(sorted_wrong)
    n_all = np.arange(1, len(sorted_score) + 1)
    tie_end = np.r_[sorted_score[:-1] != sorted_score[1:], True]
    n = n_all[tie_end]
    errors = cum_errors[tie_end]
    eligible = n >= MIN_AUTO_N
    if eligible.any():
        upper = np.ones_like(n, dtype=float)
        ok_not_all_wrong = errors < n
        upper[ok_not_all_wrong] = v64.beta.ppf(
            1.0 - v64.DELTA,
            errors[ok_not_all_wrong] + 1,
            n[ok_not_all_wrong] - errors[ok_not_all_wrong],
        )
        ok = eligible & (upper <= target_error)
        if ok.any():
            idx = np.flatnonzero(ok)[-1]
            return {
                "cal_auto_n": int(n[idx]),
                "cal_auto_rate": float(n[idx] / len(indices)),
                "cal_error_n": int(errors[idx]),
                "cal_error_rate": float(errors[idx] / n[idx]),
                "cal_error_upper95": float(upper[idx]),
            }
    return empty_selection()


def empty_selection() -> dict[str, float | int]:
    return {
        "cal_auto_n": 0,
        "cal_auto_rate": 0.0,
        "cal_error_n": 0,
        "cal_error_rate": float("nan"),
        "cal_error_upper95": 1.0,
    }


def auto_by_coverage(score: np.ndarray, coverage: float) -> np.ndarray:
    n = len(score)
    k = int(round(n * float(coverage)))
    auto = np.zeros(n, dtype=bool)
    if k <= 0:
        return auto
    order = np.argsort(-score, kind="mergesort")
    auto[order[: min(k, n)]] = True
    return auto


def evaluate_by_coverage(df: pd.DataFrame, score: np.ndarray, coverage: float, prefix: str) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p = df["p2_pred"].to_numpy(dtype=int)
    auto = auto_by_coverage(score, coverage)
    review = ~auto
    final = p.copy()
    final[review] = y[review]
    wrong = p != y
    if auto.any():
        auto_m = v64.v30.metrics_binary(y[auto], p[auto])
        auto_error = float(wrong[auto].mean())
    else:
        auto_m = {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
        auto_error = float("nan")
    workflow = v64.v30.metrics_binary(y, final)
    out: dict[str, float | int] = {
        f"{prefix}_auto_n": int(auto.sum()),
        f"{prefix}_auto_rate": float(auto.mean()),
        f"{prefix}_review_rate": float(review.mean()),
        f"{prefix}_auto_wrong_n": int((auto & wrong).sum()),
        f"{prefix}_auto_error_rate": auto_error,
    }
    out.update({f"{prefix}_auto_{k}": v for k, v in auto_m.items()})
    out.update({f"{prefix}_workflow_{k}": v for k, v in workflow.items()})
    return out


def evaluate_indices_by_coverage(df: pd.DataFrame, score: np.ndarray, indices: np.ndarray, coverage: float, prefix: str) -> dict[str, float | int]:
    tmp = df.iloc[indices].reset_index(drop=True)
    return evaluate_by_coverage(tmp, score[indices], coverage, prefix)


def split_experiment(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    y_dev = dev["label_idx"].to_numpy(dtype=int)
    wrong_dev = dev["p2_pred"].to_numpy(dtype=int) != y_dev
    rng = np.random.default_rng(SEED)
    rows = []
    for split_id in range(N_SPLITS):
        cal_idx, test_idx = v64.stratified_split(y_dev, CAL_FRAC, rng)
        for selector, score_dev in dev_scores.items():
            score_ext = ext_scores[selector]
            for target_error in TARGET_ERRORS:
                selected = select_coverage(score_dev, wrong_dev, cal_idx, target_error)
                coverage = float(selected["cal_auto_rate"])
                row = {
                    "split_id": split_id,
                    "selector": selector,
                    "target_auto_error": target_error,
                    **selected,
                }
                row.update(evaluate_indices_by_coverage(dev, score_dev, test_idx, coverage, "dev_test"))
                row.update(evaluate_by_coverage(ext, score_ext, coverage, "external"))
                rows.append(row)
    return pd.DataFrame(rows)


def full_dev_selected(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    y_dev = dev["label_idx"].to_numpy(dtype=int)
    wrong_dev = dev["p2_pred"].to_numpy(dtype=int) != y_dev
    idx = np.arange(len(dev))
    rows = []
    for selector, score_dev in dev_scores.items():
        score_ext = ext_scores[selector]
        for target_error in TARGET_ERRORS:
            selected = select_coverage(score_dev, wrong_dev, idx, target_error)
            coverage = float(selected["cal_auto_rate"])
            row = {
                "selector": selector,
                "target_auto_error": target_error,
                **selected,
            }
            row.update(evaluate_by_coverage(dev, score_dev, coverage, "development"))
            row.update(evaluate_by_coverage(ext, score_ext, coverage, "external"))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_splits(detail: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "cal_auto_rate",
        "cal_error_rate",
        "cal_error_upper95",
        "dev_test_auto_rate",
        "dev_test_auto_error_rate",
        "dev_test_workflow_balanced_accuracy",
        "external_auto_rate",
        "external_auto_error_rate",
        "external_workflow_balanced_accuracy",
        "external_workflow_sensitivity",
        "external_workflow_specificity",
        "external_workflow_fn",
        "external_workflow_fp",
    ]
    rows = []
    for (selector, target), sub in detail.groupby(["selector", "target_auto_error"]):
        row = {"selector": selector, "target_auto_error": target, "n_splits": len(sub)}
        row["no_auto_rate"] = float((sub["cal_auto_n"] == 0).mean())
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_median"] = float(vals.median())
            row[f"{col}_ci025"] = float(vals.quantile(0.025))
            row[f"{col}_ci975"] = float(vals.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def dev_selected_policy(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, sub in full.groupby("target_auto_error"):
        feasible = sub.loc[sub["cal_auto_n"] > 0].copy()
        if feasible.empty:
            continue
        # This is a development-only choice: maximize calibrated auto coverage.
        chosen = feasible.sort_values(
            ["development_auto_rate", "cal_error_rate", "development_workflow_balanced_accuracy"],
            ascending=[False, True, False],
        ).iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows)


def make_plots(summary: pd.DataFrame, full: pd.DataFrame) -> None:
    v64.configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = {
        "safe_by_any_risk": "any-risk",
        "safe_by_direction_risk": "direction-risk",
        "safe_by_fn_risk": "FN-risk",
        "safe_by_p2_core_confidence": "core-conf",
        "safe_by_p2_main_confidence": "main-conf",
        "safe_by_core_agreement": "core-agree",
        "safe_by_score_margin_agree": "margin-agree",
        "safe_by_quality_score": "quality",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for target, marker in [(0.05, "o"), (0.10, "s"), (0.15, "^")]:
        sub = summary.loc[summary["target_auto_error"].eq(target)].copy()
        ax.scatter(
            sub["external_auto_rate_median"] * 100,
            (1.0 - sub["external_auto_error_rate_median"]) * 100,
            s=66,
            marker=marker,
            alpha=0.82,
            label=f"rank target error ≤ {int(target * 100)}%",
        )
        for _, row in sub.iterrows():
            ax.text(
                row["external_auto_rate_median"] * 100 + 0.4,
                (1.0 - row["external_auto_error_rate_median"]) * 100,
                labels.get(row["selector"], row["selector"]),
                fontsize=7,
            )
    ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(90, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External auto-pass coverage median (%)")
    ax.set_ylabel("External auto-pass accuracy median (%)")
    ax.set_title("Rank-normalized calibrated selective auto-pass")
    ax.set_xlim(-2, 102)
    ax.set_ylim(45, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v65_rank_calibrated_autopass_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v65_rank_calibrated_autopass_transfer.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for target, marker in [(0.05, "o"), (0.10, "s"), (0.15, "^")]:
        sub = full.loc[(full["target_auto_error"].eq(target)) & (full["cal_auto_n"] > 0)].copy()
        ax.scatter(
            sub["external_review_rate"] * 100,
            sub["external_workflow_balanced_accuracy"] * 100,
            s=68,
            marker=marker,
            alpha=0.85,
            label=f"rank target error ≤ {int(target * 100)}%",
        )
        for _, row in sub.iterrows():
            ax.text(row["external_review_rate"] * 100 + 0.4, row["external_workflow_balanced_accuracy"] * 100, labels.get(row["selector"], row["selector"]), fontsize=7)
    ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review rate after rank calibration (%)")
    ax.set_ylabel("External workflow BAcc (%)")
    ax.set_title("Full-development rank-calibrated workflow")
    ax.set_xlim(-2, 102)
    ax.set_ylim(68, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v65_full_dev_rank_calibrated_workflow.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v65_full_dev_rank_calibrated_workflow.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v64.build_scores()
    detail = split_experiment(dev, ext, dev_scores, ext_scores)
    summary = summarize_splits(detail)
    full = full_dev_selected(dev, ext, dev_scores, ext_scores)
    selected = dev_selected_policy(full)

    detail.to_csv(OUT_DIR / "v65_rank_rcps_split_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v65_rank_rcps_split_summary.csv", index=False, encoding="utf-8-sig")
    full.to_csv(OUT_DIR / "v65_full_dev_rank_calibrated_external_eval.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v65_dev_selected_rank_calibrated_policy.csv", index=False, encoding="utf-8-sig")
    make_plots(summary, full)

    show_cols = [
        "target_auto_error",
        "selector",
        "cal_auto_rate",
        "cal_error_rate",
        "cal_error_upper95",
        "external_auto_rate",
        "external_auto_error_rate",
        "external_workflow_balanced_accuracy",
        "external_workflow_sensitivity",
        "external_workflow_specificity",
        "external_workflow_fn",
        "external_workflow_fp",
    ]
    print("\nDevelopment-selected rank-normalized policies:")
    print(selected[show_cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
