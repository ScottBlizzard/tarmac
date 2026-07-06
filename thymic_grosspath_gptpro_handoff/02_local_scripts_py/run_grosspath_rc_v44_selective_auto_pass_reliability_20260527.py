from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v44_selective_auto_pass_20260527"
FIG_DIR = OUT_DIR / "figures"
BUDGET_GRID = np.round(np.arange(0.0, 0.905, 0.005), 3)


SCENARIOS = [
    {"scenario": "auto_acc90", "auto_accuracy_min": 0.90, "auto_sensitivity_min": None, "workflow_bacc_min": None},
    {"scenario": "auto_acc95", "auto_accuracy_min": 0.95, "auto_sensitivity_min": None, "workflow_bacc_min": None},
    {"scenario": "auto_acc90_sens90", "auto_accuracy_min": 0.90, "auto_sensitivity_min": 0.90, "workflow_bacc_min": None},
    {"scenario": "auto_acc95_sens90", "auto_accuracy_min": 0.95, "auto_sensitivity_min": 0.90, "workflow_bacc_min": None},
    {"scenario": "workflow_bacc95", "auto_accuracy_min": None, "auto_sensitivity_min": None, "workflow_bacc_min": 0.95},
    {"scenario": "workflow_bacc97", "auto_accuracy_min": None, "auto_sensitivity_min": None, "workflow_bacc_min": 0.97},
    {"scenario": "auto_acc90_and_workflow95", "auto_accuracy_min": 0.90, "auto_sensitivity_min": None, "workflow_bacc_min": 0.95},
]


def binary_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    m = v30.metrics_binary(y, pred)
    return {"n": int(len(y)), **m}


def evaluate_budget(df: pd.DataFrame, score: np.ndarray, budget: float) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    review = v30.top_budget(score, float(budget))
    auto = ~review
    final = p2.copy()
    final[review] = y[review]

    workflow = binary_counts(y, final)
    auto_m = binary_counts(y[auto], p2[auto])
    wrong = p2 != y
    high = y == 1
    low = y == 0

    row: dict[str, float | int] = {
        "budget": float(budget),
        "review_n": int(review.sum()),
        "review_rate": float(review.mean()),
        "auto_n": int(auto.sum()),
        "auto_rate": float(auto.mean()),
        "auto_high_n": int((auto & high).sum()),
        "auto_low_n": int((auto & low).sum()),
        "auto_wrong_n": int((auto & wrong).sum()),
        "auto_fn_n": int((auto & high & (p2 == 0)).sum()),
        "auto_fp_n": int((auto & low & (p2 == 1)).sum()),
        "review_captured_wrong_n": int((review & wrong).sum()),
        "review_captured_wrong_rate": float((review & wrong).sum() / wrong.sum()) if wrong.sum() else 0.0,
    }
    row.update({f"workflow_{k}": v for k, v in workflow.items()})
    row.update({f"auto_{k}": v for k, v in auto_m.items()})
    return row


def make_curve(df: pd.DataFrame, score: np.ndarray, split: str) -> pd.DataFrame:
    rows = []
    for budget in BUDGET_GRID:
        row = evaluate_budget(df, score, float(budget))
        row["split"] = split
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_ok(row: pd.Series, scenario: dict[str, object]) -> bool:
    if scenario["auto_accuracy_min"] is not None and not (row["auto_accuracy"] >= float(scenario["auto_accuracy_min"])):
        return False
    if scenario["auto_sensitivity_min"] is not None and not (row["auto_sensitivity"] >= float(scenario["auto_sensitivity_min"])):
        return False
    if scenario["workflow_bacc_min"] is not None and not (row["workflow_balanced_accuracy"] >= float(scenario["workflow_bacc_min"])):
        return False
    return True


def select_scenarios(dev_curve: pd.DataFrame, ext_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        ok = dev_curve.loc[dev_curve.apply(lambda r: scenario_ok(r, sc), axis=1)].copy()
        if ok.empty:
            continue
        selected = ok.sort_values("review_rate").iloc[0]
        ext_match = ext_curve.loc[ext_curve["budget"].eq(selected["budget"])].iloc[0]
        rows.append(
            {
                "scenario": sc["scenario"],
                "selected_budget": float(selected["budget"]),
                "dev_review_rate": float(selected["review_rate"]),
                "dev_auto_rate": float(selected["auto_rate"]),
                "dev_auto_accuracy": float(selected["auto_accuracy"]),
                "dev_auto_sensitivity": float(selected["auto_sensitivity"]),
                "dev_auto_fn_n": int(selected["auto_fn_n"]),
                "dev_workflow_bacc": float(selected["workflow_balanced_accuracy"]),
                "external_review_rate": float(ext_match["review_rate"]),
                "external_auto_rate": float(ext_match["auto_rate"]),
                "external_auto_accuracy": float(ext_match["auto_accuracy"]),
                "external_auto_sensitivity": float(ext_match["auto_sensitivity"]),
                "external_auto_specificity": float(ext_match["auto_specificity"]),
                "external_auto_fn_n": int(ext_match["auto_fn_n"]),
                "external_auto_fp_n": int(ext_match["auto_fp_n"]),
                "external_auto_wrong_n": int(ext_match["auto_wrong_n"]),
                "external_workflow_bacc": float(ext_match["workflow_balanced_accuracy"]),
                "external_workflow_accuracy": float(ext_match["workflow_accuracy"]),
                "external_workflow_fn": int(ext_match["workflow_fn"]),
                "external_workflow_fp": int(ext_match["workflow_fp"]),
            }
        )
    return pd.DataFrame(rows)


def make_plots(curves: pd.DataFrame, scenarios: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = curves.loc[curves["split"].eq("external")].copy()
    dev = curves.loc[curves["split"].eq("development")].copy()

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(dev["review_rate"] * 100, dev["auto_accuracy"] * 100, color="#2471a3", linewidth=2, label="Development auto-pass Acc")
    ax.plot(ext["review_rate"] * 100, ext["auto_accuracy"] * 100, color="#c0392b", linewidth=2, label="External auto-pass Acc")
    ax.plot(ext["review_rate"] * 100, ext["workflow_balanced_accuracy"] * 100, color="#117a65", linewidth=2, linestyle="--", label="External workflow BAcc")
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.65)
    ax.axhline(95, color="#7d6608", linestyle=":", linewidth=1, alpha=0.65)
    ax.set_xlabel("Review / risk-control rate (%)")
    ax.set_ylabel("Metric (%)")
    ax.set_title("Auto-pass reliability vs review burden")
    ax.set_xlim(-2, 92)
    ax.set_ylim(60, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v44_auto_pass_reliability_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v44_auto_pass_reliability_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(ext["review_rate"] * 100, ext["auto_fn_n"], color="#c0392b", marker="o", linewidth=1.8, label="External auto-pass FN")
    ax.plot(ext["review_rate"] * 100, ext["auto_fp_n"], color="#2471a3", marker="s", linewidth=1.8, label="External auto-pass FP")
    for _, row in scenarios.iterrows():
        if row["scenario"] in ["auto_acc90", "auto_acc95", "workflow_bacc97"]:
            ax.axvline(row["external_review_rate"] * 100, color="#566573", linestyle=":", linewidth=1)
            ax.text(row["external_review_rate"] * 100 + 0.6, ax.get_ylim()[1] * 0.75, row["scenario"], rotation=90, fontsize=8)
    ax.set_xlabel("Review / risk-control rate (%)")
    ax.set_ylabel("Residual errors among auto-pass cases")
    ax.set_title("Residual auto-pass errors")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v44_auto_pass_residual_errors.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v44_auto_pass_residual_errors.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    model = v30.make_models(numeric, categorical)["hard_logistic"]
    dev_score, ext_score = v30.oof_and_external_scores(dev, ext, features, model)

    dev_curve = make_curve(dev, dev_score, "development")
    ext_curve = make_curve(ext, ext_score, "external")
    curves = pd.concat([dev_curve, ext_curve], ignore_index=True)
    selected = select_scenarios(dev_curve, ext_curve)

    curves.to_csv(OUT_DIR / "v44_selective_auto_pass_curves.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v44_dev_selected_auto_pass_scenarios_external_eval.csv", index=False, encoding="utf-8-sig")
    make_plots(curves, selected)

    print(selected.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
