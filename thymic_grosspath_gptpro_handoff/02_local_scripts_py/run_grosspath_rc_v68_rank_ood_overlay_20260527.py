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
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v67_devfit_ood_quality_controller_20260527 as v67  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v68_rank_ood_overlay_20260527"
FIG_DIR = OUT_DIR / "figures"
EXTRA_RATES = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]


def top_rate(score: np.ndarray, rate: float) -> np.ndarray:
    n = len(score)
    k = int(round(n * rate))
    out = np.zeros(n, dtype=bool)
    if k <= 0:
        return out
    order = np.argsort(-score, kind="mergesort")
    out[order[: min(k, n)]] = True
    return out


def v50_review(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> np.ndarray:
    base = v30.top_budget(scores["any"], 0.525)
    return v50.add_top_candidates(df, base, scores["direction"], 0.200, "all_direction")


def evaluate(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review] = y[review]
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(df)
    m.update(
        {
            "control_n": int(review.sum()),
            "control_rate": float(review.mean()),
            "extra_captured_wrong_n": 0,
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "remaining_error_n": int((final != y).sum()),
        }
    )
    return m


def run_grid(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray], score_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dev = v50_review(dev, dev_scores)
    base_ext = v50_review(ext, ext_scores)
    masks_dev = v48.error_masks(dev)
    masks_ext = v48.error_masks(ext)
    rows = []
    case_rows = []

    rows.append({"policy": "v50_base", "score_name": "none", "extra_rate": 0.0, "split": "development", **evaluate(dev, base_dev)})
    rows.append({"policy": "v50_base", "score_name": "none", "extra_rate": 0.0, "split": "external", **evaluate(ext, base_ext)})

    for score_name in sorted(score_table["score_name"].unique()):
        dscore = score_table.loc[score_table["split"].eq("development") & score_table["score_name"].eq(score_name)].sort_values("case_index")["ood_score"].to_numpy(float)
        escore = score_table.loc[score_table["split"].eq("external") & score_table["score_name"].eq(score_name)].sort_values("case_index")["ood_score"].to_numpy(float)
        for extra_rate in EXTRA_RATES:
            d_extra = top_rate(dscore, extra_rate)
            e_extra = top_rate(escore, extra_rate)
            d_review = base_dev | d_extra
            e_review = base_ext | e_extra
            dm = evaluate(dev, d_review)
            em = evaluate(ext, e_review)
            dm["extra_captured_wrong_n"] = int((d_extra & (~base_dev) & masks_dev["any_wrong"]).sum())
            em["extra_captured_wrong_n"] = int((e_extra & (~base_ext) & masks_ext["any_wrong"]).sum())
            rows.append({"policy": "v50_plus_rank_ood", "score_name": score_name, "extra_rate": extra_rate, "split": "development", **dm})
            rows.append({"policy": "v50_plus_rank_ood", "score_name": score_name, "extra_rate": extra_rate, "split": "external", **em})

            tmp = ext[
                [
                    c
                    for c in [
                        "case_id",
                        "original_case_id",
                        "task_l6_label",
                        "task_l7_label",
                        "image_name",
                        "quality_score",
                        "quality_status",
                        "p2_pred",
                        "main_prob",
                        "robust_prob",
                        "prob_mean_core",
                    ]
                    if c in ext.columns
                ]
            ].copy()
            y = ext["label_idx"].to_numpy(int)
            p2 = ext["p2_pred"].to_numpy(int)
            tmp["score_name"] = score_name
            tmp["extra_rate"] = extra_rate
            tmp["ood_score"] = escore
            tmp["base_v50_review"] = base_ext.astype(int)
            tmp["rank_ood_extra"] = e_extra.astype(int)
            tmp["final_review"] = e_review.astype(int)
            tmp["label_idx"] = y
            tmp["p2_wrong"] = (p2 != y).astype(int)
            tmp["error_direction"] = np.select(
                [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
                ["FN_high_to_low", "FP_low_to_high"],
                default="correct",
            )
            case_rows.append(tmp)

    return pd.DataFrame(rows), pd.concat(case_rows, ignore_index=True)


def select_dev(summary: pd.DataFrame) -> pd.DataFrame:
    dev = summary.loc[summary["split"].eq("development") & summary["policy"].eq("v50_plus_rank_ood")].copy()
    ext = summary.loc[summary["split"].eq("external")].copy()
    scenarios = [
        {"scenario": "dev_min_control_bacc99", "bacc": 0.99, "control_max": 0.86},
        {"scenario": "dev_min_control_sens99_spec98", "sens": 0.99, "spec": 0.98, "control_max": 0.90},
        {"scenario": "dev_extra_catches_wrong", "min_extra_wrong": 1, "control_max": 0.86},
    ]
    rows = []
    for sc in scenarios:
        ok = dev.copy()
        if "bacc" in sc:
            ok = ok.loc[ok["balanced_accuracy"].ge(sc["bacc"])]
        if "sens" in sc:
            ok = ok.loc[ok["sensitivity"].ge(sc["sens"])]
        if "spec" in sc:
            ok = ok.loc[ok["specificity"].ge(sc["spec"])]
        if "min_extra_wrong" in sc:
            ok = ok.loc[ok["extra_captured_wrong_n"].ge(sc["min_extra_wrong"])]
        if "control_max" in sc:
            ok = ok.loc[ok["control_rate"].le(sc["control_max"])]
        if ok.empty:
            continue
        chosen = ok.sort_values(["control_rate", "extra_captured_wrong_n", "balanced_accuracy"], ascending=[True, False, False]).iloc[0]
        match = ext.loc[ext["score_name"].eq(chosen["score_name"]) & ext["extra_rate"].eq(chosen["extra_rate"])].iloc[0]
        row = {"scenario": sc["scenario"]}
        row.update({f"dev_{k}": v for k, v in chosen.items() if k != "split"})
        row.update({f"external_{k}": v for k, v in match.items() if k != "split"})
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    v67.configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = summary.loc[summary["split"].eq("external")].copy()
    base = ext.loc[ext["policy"].eq("v50_base")].iloc[0]
    cand = ext.loc[ext["policy"].eq("v50_plus_rank_ood")].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.scatter(cand["control_rate"] * 100, cand["balanced_accuracy"] * 100, s=28, color="#b0b7c3", alpha=0.5, label="rank-OOD overlays")
    ax.scatter([base["control_rate"] * 100], [base["balanced_accuracy"] * 100], s=95, color="#1f618d", edgecolor="white", label="v50 base")
    if not selected.empty:
        for _, row in selected.iterrows():
            ax.scatter(row["external_control_rate"] * 100, row["external_balanced_accuracy"] * 100, s=90, color="#c0392b", edgecolor="white")
            ax.text(row["external_control_rate"] * 100 + 0.25, row["external_balanced_accuracy"] * 100, row["scenario"], fontsize=7)
    ax.axhline(97, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(99, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External control rate (%)")
    ax.set_ylabel("External workflow BAcc (%)")
    ax.set_title("Rank-normalized OOD overlay over v50")
    ax.set_xlim(72, 96)
    ax.set_ylim(96.8, 100.1)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v68_rank_ood_overlay_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v68_rank_ood_overlay_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    score_table = v67.build_score_table(dev, ext)
    summary, cases = run_grid(dev, ext, dev_scores, ext_scores, score_table)
    selected = select_dev(summary)

    summary.to_csv(OUT_DIR / "v68_rank_ood_overlay_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v68_rank_ood_overlay_case_routes.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v68_dev_selected_rank_ood_external_eval.csv", index=False, encoding="utf-8-sig")
    make_plot(summary, selected)

    print("\nTop external descriptive rank-OOD overlays:")
    top = summary.loc[summary["split"].eq("external")].sort_values(["balanced_accuracy", "control_rate"], ascending=[False, True]).head(20)
    print(top[["policy", "score_name", "extra_rate", "control_rate", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp", "extra_captured_wrong_n"]].to_string(index=False))
    if not selected.empty:
        print("\nDev-selected rank-OOD overlays:")
        cols = [
            "scenario",
            "dev_score_name",
            "dev_extra_rate",
            "dev_control_rate",
            "dev_balanced_accuracy",
            "external_control_rate",
            "external_balanced_accuracy",
            "external_sensitivity",
            "external_specificity",
            "external_fn",
            "external_fp",
            "external_extra_captured_wrong_n",
        ]
        print(selected[cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
