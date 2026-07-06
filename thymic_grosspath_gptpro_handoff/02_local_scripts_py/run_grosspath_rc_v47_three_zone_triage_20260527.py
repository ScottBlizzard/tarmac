from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v47_three_zone_triage_20260527"
FIG_DIR = OUT_DIR / "figures"
SAFE_GRID = np.round(np.arange(0.00, 0.405, 0.025), 3)
REVIEW_GRID = np.round(np.arange(0.00, 0.805, 0.025), 3)


SCENARIOS = [
    {"scenario": "clear90_workflow90", "clear_acc_min": 0.90, "workflow_bacc_min": 0.90},
    {"scenario": "clear90_workflow93", "clear_acc_min": 0.90, "workflow_bacc_min": 0.93},
    {"scenario": "clear95_workflow90", "clear_acc_min": 0.95, "workflow_bacc_min": 0.90},
    {"scenario": "clear95_workflow93", "clear_acc_min": 0.95, "workflow_bacc_min": 0.93},
    {"scenario": "clear100_workflow90", "clear_acc_min": 1.00, "workflow_bacc_min": 0.90},
]


def assign_zones(score: np.ndarray, safe_rate: float, review_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(score)
    if safe_rate + review_rate > 1.0:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    order_high = np.argsort(-score, kind="mergesort")
    review_n = int(round(n * review_rate))
    safe_n = int(round(n * safe_rate))
    review = np.zeros(n, dtype=bool)
    safe = np.zeros(n, dtype=bool)
    if review_n:
        review[order_high[:review_n]] = True
    if safe_n:
        order_low = np.argsort(score, kind="mergesort")
        # Exclude review cases if rounding creates an overlap in extreme settings.
        low_candidates = [i for i in order_low if not review[i]]
        safe[np.asarray(low_candidates[:safe_n], dtype=int)] = True
    middle = ~(review | safe)
    return safe, middle, review


def subset_metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if not mask.any():
        return {
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "fn": 0,
            "fp": 0,
        }
    m = v30.metrics_binary(y[mask], pred[mask])
    return {"n": int(mask.sum()), **m}


def evaluate(df: pd.DataFrame, score: np.ndarray, safe_rate: float, review_rate: float) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    safe, middle, review = assign_zones(score, safe_rate, review_rate)
    final = p2.copy()
    final[review] = y[review]
    workflow = v30.metrics_binary(y, final)
    clear = subset_metrics(y, p2, safe)
    middle_m = subset_metrics(y, p2, middle)
    auto = subset_metrics(y, p2, safe | middle)
    wrong = p2 != y
    return {
        "safe_rate_target": float(safe_rate),
        "review_rate_target": float(review_rate),
        "clear_auto_n": int(safe.sum()),
        "clear_auto_rate": float(safe.mean()),
        "middle_auto_n": int(middle.sum()),
        "middle_auto_rate": float(middle.mean()),
        "review_n": int(review.sum()),
        "review_rate": float(review.mean()),
        "clear_wrong_n": int((safe & wrong).sum()),
        "middle_wrong_n": int((middle & wrong).sum()),
        "review_captured_wrong_n": int((review & wrong).sum()),
        **{f"workflow_{k}": v for k, v in workflow.items()},
        **{f"clear_{k}": v for k, v in clear.items()},
        **{f"middle_{k}": v for k, v in middle_m.items()},
        **{f"auto_all_{k}": v for k, v in auto.items()},
    }


def build_grid(df: pd.DataFrame, score: np.ndarray, split: str) -> pd.DataFrame:
    rows = []
    for safe_rate in SAFE_GRID:
        for review_rate in REVIEW_GRID:
            if safe_rate + review_rate > 1.0:
                continue
            row = evaluate(df, score, float(safe_rate), float(review_rate))
            row["split"] = split
            rows.append(row)
    return pd.DataFrame(rows)


def choose_scenarios(dev_grid: pd.DataFrame, ext_grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        ok = dev_grid.loc[
            dev_grid["clear_accuracy"].ge(sc["clear_acc_min"])
            & dev_grid["workflow_balanced_accuracy"].ge(sc["workflow_bacc_min"])
            & dev_grid["clear_auto_n"].gt(0)
        ].copy()
        if ok.empty:
            continue
        # Prefer more clear auto-pass, then less review, then higher workflow BAcc.
        chosen = ok.sort_values(
            ["clear_auto_rate", "review_rate", "workflow_balanced_accuracy"],
            ascending=[False, True, False],
        ).iloc[0]
        ext = ext_grid.loc[
            ext_grid["safe_rate_target"].eq(chosen["safe_rate_target"])
            & ext_grid["review_rate_target"].eq(chosen["review_rate_target"])
        ].iloc[0]
        rows.append(
            {
                "scenario": sc["scenario"],
                "dev_clear_acc_min": sc["clear_acc_min"],
                "dev_workflow_bacc_min": sc["workflow_bacc_min"],
                **{f"dev_{k}": v for k, v in chosen.items() if k != "split"},
                **{f"external_{k}": v for k, v in ext.items() if k != "split"},
            }
        )
    return pd.DataFrame(rows)


def make_cases(ext: pd.DataFrame, ext_score: np.ndarray, selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    y = ext["label_idx"].to_numpy(dtype=int)
    p2 = ext["p2_pred"].to_numpy(dtype=int)
    wrong = p2 != y
    for _, row in selected.iterrows():
        safe, middle, review = assign_zones(ext_score, float(row["dev_safe_rate_target"]), float(row["dev_review_rate_target"]))
        tmp = ext[
            [
                "case_id",
                "original_case_id",
                "source_folder",
                "task_l6_label",
                "task_l7_label",
                "label_idx",
                "image_name",
                "quality_status",
                "quality_score",
                "p2_pred",
                "p2_wrong",
                "main_prob",
                "robust_prob",
                "prob_mean_core",
            ]
        ].copy()
        tmp.insert(0, "scenario", row["scenario"])
        tmp["hard_risk_score"] = ext_score
        tmp["zone"] = np.select([safe, middle, review], ["clear_auto", "middle_auto", "review"], default="unknown")
        tmp["p2_error_direction"] = np.select(
            [wrong & (y == 1) & (p2 == 0), wrong & (y == 0) & (p2 == 1)],
            ["FN_high_to_low", "FP_low_to_high"],
            default="correct",
        )
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_plots(grid: pd.DataFrame, selected: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = grid.loc[grid["split"].eq("external")].copy()

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    # For each review rate, draw best clear accuracy among choices.
    for review_rate in [0.30, 0.50, 0.60, 0.75]:
        sub = ext.loc[np.isclose(ext["review_rate_target"], review_rate)].sort_values("clear_auto_rate")
        if sub.empty:
            continue
        ax.plot(sub["clear_auto_rate"] * 100, sub["clear_accuracy"] * 100, marker="o", linewidth=1.6, label=f"review {int(review_rate*100)}%")
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(95, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Clear auto-pass coverage (%)")
    ax.set_ylabel("External clear auto-pass accuracy (%)")
    ax.set_title("Three-zone triage: clear auto-pass reliability")
    ax.set_xlim(-1, 42)
    ax.set_ylim(45, 102)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v47_three_zone_clear_auto_reliability.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v47_three_zone_clear_auto_reliability.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not selected.empty:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        x = np.arange(len(selected))
        ax.bar(x - 0.25, selected["external_clear_auto_rate"] * 100, width=0.25, label="clear auto", color="#117a65")
        ax.bar(x, selected["external_middle_auto_rate"] * 100, width=0.25, label="middle auto", color="#f5b041")
        ax.bar(x + 0.25, selected["external_review_rate"] * 100, width=0.25, label="review", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(selected["scenario"], rotation=25, ha="right")
        ax.set_ylabel("External case proportion (%)")
        ax.set_title("Selected three-zone workflow composition")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "v47_selected_three_zone_composition.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "v47_selected_three_zone_composition.pdf", dpi=300, bbox_inches="tight")
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

    dev_grid = build_grid(dev, dev_score, "development")
    ext_grid = build_grid(ext, ext_score, "external")
    grid = pd.concat([dev_grid, ext_grid], ignore_index=True)
    selected = choose_scenarios(dev_grid, ext_grid)
    cases = make_cases(ext, ext_score, selected)

    grid.to_csv(OUT_DIR / "v47_three_zone_grid_metrics.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v47_three_zone_dev_selected_external_eval.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v47_three_zone_case_routes_external.csv", index=False, encoding="utf-8-sig")
    make_plots(grid, selected)

    show_cols = [
        "scenario",
        "dev_safe_rate_target",
        "dev_review_rate_target",
        "external_clear_auto_rate",
        "external_clear_accuracy",
        "external_clear_fn",
        "external_clear_fp",
        "external_middle_auto_rate",
        "external_middle_accuracy",
        "external_review_rate",
        "external_workflow_balanced_accuracy",
        "external_workflow_accuracy",
        "external_workflow_fn",
        "external_workflow_fp",
    ]
    print(selected[show_cols].to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
