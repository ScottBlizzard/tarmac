from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v64_conformal_autopass_20260527"
FIG_DIR = OUT_DIR / "figures"
SEED = 20260527
N_SPLITS = 300
CAL_FRAC = 0.50
DELTA = 0.05
TARGET_ERRORS = [0.05, 0.10, 0.15]
MIN_AUTO_N = 20


def configure_matplotlib_font() -> None:
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def stratified_split(y: np.ndarray, cal_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cal_parts = []
    test_parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        n_cal = int(round(len(idx) * cal_frac))
        cal_parts.append(idx[:n_cal])
        test_parts.append(idx[n_cal:])
    cal = np.concatenate(cal_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(cal)
    rng.shuffle(test)
    return cal, test


def binomial_upper(errors: int, n: int, delta: float = DELTA) -> float:
    if n <= 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def p2_confidence(df: pd.DataFrame, prob_col: str) -> np.ndarray:
    prob = df[prob_col].to_numpy(dtype=float)
    pred = df["p2_pred"].to_numpy(dtype=int)
    return np.where(pred == 1, prob, 1.0 - prob)


def build_scores() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    dev, ext, dev_risk, ext_risk = v50.get_scores()
    dev_scores = {
        "safe_by_any_risk": -dev_risk["any"],
        "safe_by_direction_risk": -dev_risk["direction"],
        "safe_by_fn_risk": -dev_risk["fn"],
        "safe_by_p2_core_confidence": p2_confidence(dev, "prob_mean_core"),
        "safe_by_p2_main_confidence": p2_confidence(dev, "main_prob"),
        "safe_by_core_agreement": dev["core_agree_count"].to_numpy(dtype=float),
        "safe_by_score_margin_agree": dev["score_margin_agree"].to_numpy(dtype=float),
    }
    ext_scores = {
        "safe_by_any_risk": -ext_risk["any"],
        "safe_by_direction_risk": -ext_risk["direction"],
        "safe_by_fn_risk": -ext_risk["fn"],
        "safe_by_p2_core_confidence": p2_confidence(ext, "prob_mean_core"),
        "safe_by_p2_main_confidence": p2_confidence(ext, "main_prob"),
        "safe_by_core_agreement": ext["core_agree_count"].to_numpy(dtype=float),
        "safe_by_score_margin_agree": ext["score_margin_agree"].to_numpy(dtype=float),
    }
    if "quality_score" in dev.columns and "quality_score" in ext.columns:
        dev_scores["safe_by_quality_score"] = pd.to_numeric(dev["quality_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
        ext_scores["safe_by_quality_score"] = pd.to_numeric(ext["quality_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return dev, ext, dev_scores, ext_scores


def candidate_thresholds(score: np.ndarray) -> np.ndarray:
    vals = np.unique(score[np.isfinite(score)])
    vals.sort()
    return vals[::-1]


def select_threshold(
    score: np.ndarray,
    wrong: np.ndarray,
    indices: np.ndarray,
    target_error: float,
    min_auto_n: int = MIN_AUTO_N,
) -> dict[str, float | int]:
    score_cal = score[indices]
    wrong_cal = wrong[indices]
    finite = np.isfinite(score_cal)
    score_cal = score_cal[finite]
    wrong_cal = wrong_cal[finite].astype(int)
    if len(score_cal) == 0:
        return {
            "threshold": float("inf"),
            "cal_auto_n": 0,
            "cal_auto_rate": 0.0,
            "cal_error_n": 0,
            "cal_error_rate": float("nan"),
            "cal_error_upper95": 1.0,
        }
    order = np.argsort(-score_cal, kind="mergesort")
    sorted_score = score_cal[order]
    sorted_wrong = wrong_cal[order]
    cum_errors = np.cumsum(sorted_wrong)
    n_all = np.arange(1, len(sorted_score) + 1)
    # Only evaluate the last item for tied scores, because thresholding includes ties.
    tie_end = np.r_[sorted_score[:-1] != sorted_score[1:], True]
    n = n_all[tie_end]
    errors = cum_errors[tie_end]
    thresholds = sorted_score[tie_end]
    eligible = n >= min_auto_n
    if eligible.any():
        upper = np.ones_like(n, dtype=float)
        not_all_wrong = errors < n
        upper[not_all_wrong] = beta.ppf(1.0 - DELTA, errors[not_all_wrong] + 1, n[not_all_wrong] - errors[not_all_wrong])
        ok = eligible & (upper <= target_error)
        if ok.any():
            # n is ascending, so the last valid point maximizes auto coverage.
            idx = np.flatnonzero(ok)[-1]
            return {
                "threshold": float(thresholds[idx]),
                "cal_auto_n": int(n[idx]),
                "cal_auto_rate": float(n[idx] / len(indices)),
                "cal_error_n": int(errors[idx]),
                "cal_error_rate": float(errors[idx] / n[idx]),
                "cal_error_upper95": float(upper[idx]),
            }
    return {
        "threshold": float("inf"),
        "cal_auto_n": 0,
        "cal_auto_rate": 0.0,
        "cal_error_n": 0,
        "cal_error_rate": float("nan"),
        "cal_error_upper95": 1.0,
    }


def evaluate_subset(df: pd.DataFrame, score: np.ndarray, threshold: float, prefix: str) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p = df["p2_pred"].to_numpy(dtype=int)
    auto = score >= threshold
    review = ~auto
    final = p.copy()
    final[review] = y[review]
    auto_n = int(auto.sum())
    wrong = p != y
    if auto_n > 0:
        auto_m = v30.metrics_binary(y[auto], p[auto])
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
    workflow = v30.metrics_binary(y, final)
    out: dict[str, float | int] = {
        f"{prefix}_auto_n": auto_n,
        f"{prefix}_auto_rate": float(auto.mean()),
        f"{prefix}_review_rate": float(review.mean()),
        f"{prefix}_auto_wrong_n": int((auto & wrong).sum()),
        f"{prefix}_auto_error_rate": auto_error,
    }
    out.update({f"{prefix}_auto_{k}": v for k, v in auto_m.items()})
    out.update({f"{prefix}_workflow_{k}": v for k, v in workflow.items()})
    return out


def evaluate_subset_indices(
    df: pd.DataFrame,
    score: np.ndarray,
    threshold: float,
    indices: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    tmp = df.iloc[indices].reset_index(drop=True)
    return evaluate_subset(tmp, score[indices], threshold, prefix)


def split_calibration_experiment(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    y_dev = dev["label_idx"].to_numpy(dtype=int)
    wrong_dev = dev["p2_pred"].to_numpy(dtype=int) != y_dev
    rng = np.random.default_rng(SEED)
    rows = []
    for split_id in range(N_SPLITS):
        cal_idx, test_idx = stratified_split(y_dev, CAL_FRAC, rng)
        for selector, score_dev in dev_scores.items():
            score_ext = ext_scores[selector]
            for target_error in TARGET_ERRORS:
                selected = select_threshold(score_dev, wrong_dev, cal_idx, target_error)
                threshold = float(selected["threshold"])
                row = {
                    "split_id": split_id,
                    "selector": selector,
                    "target_auto_error": target_error,
                    **selected,
                }
                row.update(evaluate_subset_indices(dev, score_dev, threshold, test_idx, "dev_test"))
                row.update(evaluate_subset(ext, score_ext, threshold, "external"))
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
            selected = select_threshold(score_dev, wrong_dev, idx, target_error)
            threshold = float(selected["threshold"])
            row = {
                "selector": selector,
                "target_auto_error": target_error,
                **selected,
            }
            row.update(evaluate_subset(dev, score_dev, threshold, "development"))
            row.update(evaluate_subset(ext, score_ext, threshold, "external"))
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


def choose_deployable_rows(full: pd.DataFrame) -> pd.DataFrame:
    # Choose the largest external auto coverage among rows with full-dev calibration feasible.
    rows = []
    for target, sub in full.groupby("target_auto_error"):
        feasible = sub.loc[sub["cal_auto_n"] > 0].copy()
        if feasible.empty:
            continue
        chosen = feasible.sort_values(
            ["external_auto_rate", "external_auto_error_rate", "external_workflow_balanced_accuracy"],
            ascending=[False, True, False],
        ).iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows)


def make_plots(summary: pd.DataFrame, full: pd.DataFrame) -> None:
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
    focus = summary.loc[summary["target_auto_error"].isin([0.05, 0.10, 0.15])].copy()
    for target, marker in [(0.05, "o"), (0.10, "s"), (0.15, "^")]:
        sub = focus.loc[focus["target_auto_error"].eq(target)].copy()
        ax.scatter(
            sub["external_auto_rate_median"] * 100,
            (1.0 - sub["external_auto_error_rate_median"]) * 100,
            s=66,
            marker=marker,
            label=f"target auto error ≤ {int(target * 100)}%",
            alpha=0.82,
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
    ax.set_title("Calibrated selective auto-pass transfer")
    ax.set_xlim(-2, 102)
    ax.set_ylim(45, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v64_calibrated_autopass_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v64_calibrated_autopass_transfer.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    full_focus = full.loc[full["cal_auto_n"] > 0].copy()
    for target, marker in [(0.05, "o"), (0.10, "s"), (0.15, "^")]:
        sub = full_focus.loc[full_focus["target_auto_error"].eq(target)].copy()
        ax.scatter(
            sub["external_review_rate"] * 100,
            sub["external_workflow_balanced_accuracy"] * 100,
            marker=marker,
            s=68,
            alpha=0.85,
            label=f"target error ≤ {int(target * 100)}%",
        )
        for _, row in sub.iterrows():
            ax.text(row["external_review_rate"] * 100 + 0.4, row["external_workflow_balanced_accuracy"] * 100, labels.get(row["selector"], row["selector"]), fontsize=7)
    ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review rate after full-dev calibration (%)")
    ax.set_ylabel("External workflow BAcc (%)")
    ax.set_title("Full-development calibrated workflow result")
    ax.set_xlim(-2, 102)
    ax.set_ylim(68, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v64_full_dev_calibrated_workflow.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v64_full_dev_calibrated_workflow.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = build_scores()
    detail = split_calibration_experiment(dev, ext, dev_scores, ext_scores)
    summary = summarize_splits(detail)
    full = full_dev_selected(dev, ext, dev_scores, ext_scores)
    chosen = choose_deployable_rows(full)

    detail.to_csv(OUT_DIR / "v64_rcps_split_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v64_rcps_split_summary.csv", index=False, encoding="utf-8-sig")
    full.to_csv(OUT_DIR / "v64_full_dev_calibrated_external_eval.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUT_DIR / "v64_full_dev_calibrated_selected_rows.csv", index=False, encoding="utf-8-sig")
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
    print("\nFull-development calibrated selected rows:")
    print(chosen[show_cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
