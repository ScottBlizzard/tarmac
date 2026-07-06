from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v67_devfit_ood_quality_controller_20260527"
FIG_DIR = OUT_DIR / "figures"
QUANTILES = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975]
SEED = 20260527


def configure_matplotlib_font() -> None:
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def common_features(dev: pd.DataFrame, ext: pd.DataFrame) -> dict[str, list[str]]:
    image = [c for c in v30.IMAGE_FEATURES if c in dev.columns and c in ext.columns]
    prob = [c for c in v30.PROB_FEATURES if c in dev.columns and c in ext.columns]
    bins = [c for c in v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    compact_prob = [
        c
        for c in [
            "prob_base162",
            "prob103_vitl",
            "prob107_qkvb",
            "prob_mean_core",
            "core_prob_std",
            "core_prob_range",
            "main_prob",
            "robust_prob",
            "main_margin_abs",
            "robust_margin_abs",
            "main_robust_abs_diff",
            "score_margin_agree",
            "core_agree_count",
        ]
        if c in dev.columns and c in ext.columns
    ]
    return {
        "image_common5": image,
        "model_prob": prob,
        "model_prob_compact": compact_prob,
        "image_plus_model": image + compact_prob + bins,
    }


def preprocess_fit(dev: pd.DataFrame, cols: list[str]) -> tuple[Pipeline, np.ndarray]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10, 90))),
        ]
    )
    x_dev = pipe.fit_transform(dev[cols])
    return pipe, x_dev


def mahalanobis_scores(dev: pd.DataFrame, ext: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pipe, x_dev = preprocess_fit(dev, cols)
    x_ext = pipe.transform(ext[cols])
    cov = LedoitWolf().fit(x_dev)
    return cov.mahalanobis(x_dev), cov.mahalanobis(x_ext)


def isolation_scores(dev: pd.DataFrame, ext: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pipe, x_dev = preprocess_fit(dev, cols)
    x_ext = pipe.transform(ext[cols])
    clf = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(x_dev)
    # sklearn's score_samples is high for inliers. Convert to high = abnormal.
    return -clf.score_samples(x_dev), -clf.score_samples(x_ext)


def heuristic_scores(dev: pd.DataFrame, ext: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    specs = {}
    if "main_margin_abs" in dev.columns and "robust_margin_abs" in dev.columns:
        specs["low_confidence"] = (
            1.0 - np.minimum(dev["main_margin_abs"].to_numpy(float), dev["robust_margin_abs"].to_numpy(float)),
            1.0 - np.minimum(ext["main_margin_abs"].to_numpy(float), ext["robust_margin_abs"].to_numpy(float)),
        )
    if "main_robust_abs_diff" in dev.columns:
        specs["main_robust_disagreement"] = (
            dev["main_robust_abs_diff"].to_numpy(float),
            ext["main_robust_abs_diff"].to_numpy(float),
        )
    if "core_prob_range" in dev.columns:
        specs["core_model_range"] = (
            dev["core_prob_range"].to_numpy(float),
            ext["core_prob_range"].to_numpy(float),
        )
    return specs


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
            "auto_n": int((~review).sum()),
            "auto_rate": float((~review).mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "remaining_error_n": int((final != y).sum()),
        }
    )
    return m


def build_score_table(dev: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feat_groups = common_features(dev, ext)
    for group, cols in feat_groups.items():
        if len(cols) < 2:
            continue
        for method, func in [("mahalanobis", mahalanobis_scores), ("isolation_forest", isolation_scores)]:
            try:
                dev_score, ext_score = func(dev, ext, cols)
            except Exception as exc:  # Keep the experiment robust across missing/degenerate columns.
                print(f"[skip] {group}/{method}: {exc}")
                continue
            rows.append((f"{group}_{method}", group, method, cols, dev_score, ext_score))
    for name, (dev_score, ext_score) in heuristic_scores(dev, ext).items():
        rows.append((name, "heuristic_model", "heuristic", [], dev_score, ext_score))

    out_rows = []
    for name, group, method, cols, dev_score, ext_score in rows:
        for split, score in [("development", dev_score), ("external", ext_score)]:
            tmp = pd.DataFrame(
                {
                    "score_name": name,
                    "feature_group": group,
                    "method": method,
                    "n_features": len(cols),
                    "split": split,
                    "case_index": np.arange(len(score)),
                    "ood_score": score,
                }
            )
            out_rows.append(tmp)
    return pd.concat(out_rows, ignore_index=True)


def run_grid(
    dev: pd.DataFrame,
    ext: pd.DataFrame,
    dev_scores: dict[str, np.ndarray],
    ext_scores: dict[str, np.ndarray],
    score_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dev = v50_review(dev, dev_scores)
    base_ext = v50_review(ext, ext_scores)
    score_names = sorted(score_table["score_name"].unique())
    rows = []
    case_rows = []

    rows.append({"policy": "v50_base", "score_name": "none", "quantile": np.nan, "split": "development", **evaluate(dev, base_dev)})
    rows.append({"policy": "v50_base", "score_name": "none", "quantile": np.nan, "split": "external", **evaluate(ext, base_ext)})

    for score_name in score_names:
        dscore = score_table.loc[score_table["split"].eq("development") & score_table["score_name"].eq(score_name)].sort_values("case_index")["ood_score"].to_numpy(float)
        escore = score_table.loc[score_table["split"].eq("external") & score_table["score_name"].eq(score_name)].sort_values("case_index")["ood_score"].to_numpy(float)
        for q in QUANTILES:
            thr = float(np.nanquantile(dscore, q))
            d_extra = dscore >= thr
            e_extra = escore >= thr
            d_review = base_dev | d_extra
            e_review = base_ext | e_extra
            rows.append(
                {
                    "policy": f"v50_plus_devfit_ood_q{int(q * 1000):03d}",
                    "score_name": score_name,
                    "quantile": q,
                    "threshold_from_dev": thr,
                    "extra_control_rate_development": float(d_extra.mean()),
                    "extra_control_rate_external": float(e_extra.mean()),
                    "split": "development",
                    **evaluate(dev, d_review),
                }
            )
            rows.append(
                {
                    "policy": f"v50_plus_devfit_ood_q{int(q * 1000):03d}",
                    "score_name": score_name,
                    "quantile": q,
                    "threshold_from_dev": thr,
                    "extra_control_rate_development": float(d_extra.mean()),
                    "extra_control_rate_external": float(e_extra.mean()),
                    "split": "external",
                    **evaluate(ext, e_review),
                }
            )

            if q in [0.90, 0.95]:
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
                tmp["quantile"] = q
                tmp["ood_score"] = escore
                tmp["base_v50_review"] = base_ext.astype(int)
                tmp["extra_ood_control"] = e_extra.astype(int)
                tmp["final_review"] = e_review.astype(int)
                tmp["label_idx"] = y
                tmp["p2_wrong"] = (p2 != y).astype(int)
                tmp["error_direction"] = np.select(
                    [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
                    ["FN_high_to_low", "FP_low_to_high"],
                    default="correct",
                )
                case_rows.append(tmp)
    return pd.DataFrame(rows), pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()


def select_dev_feasible(summary: pd.DataFrame) -> pd.DataFrame:
    dev = summary.loc[summary["split"].eq("development") & summary["policy"].ne("v50_base")].copy()
    ext = summary.loc[summary["split"].eq("external")].copy()
    rows = []
    scenarios = [
        {"scenario": "dev_bacc99_control85", "dev_bacc_min": 0.99, "dev_control_max": 0.85},
        {"scenario": "dev_bacc985_control82", "dev_bacc_min": 0.985, "dev_control_max": 0.82},
        {"scenario": "dev_sens99_spec98", "dev_sens_min": 0.99, "dev_spec_min": 0.98},
    ]
    for sc in scenarios:
        ok = dev.copy()
        if "dev_bacc_min" in sc:
            ok = ok.loc[ok["balanced_accuracy"].ge(sc["dev_bacc_min"])]
        if "dev_control_max" in sc:
            ok = ok.loc[ok["control_rate"].le(sc["dev_control_max"])]
        if "dev_sens_min" in sc:
            ok = ok.loc[ok["sensitivity"].ge(sc["dev_sens_min"])]
        if "dev_spec_min" in sc:
            ok = ok.loc[ok["specificity"].ge(sc["dev_spec_min"])]
        if ok.empty:
            continue
        chosen = ok.sort_values(["control_rate", "balanced_accuracy", "remaining_error_n"], ascending=[True, False, True]).iloc[0]
        match = ext.loc[ext["score_name"].eq(chosen["score_name"]) & ext["quantile"].eq(chosen["quantile"])].iloc[0]
        row = {"scenario": sc["scenario"]}
        row.update({f"dev_{k}": v for k, v in chosen.items() if k != "split"})
        row.update({f"external_{k}": v for k, v in match.items() if k != "split"})
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = summary.loc[summary["split"].eq("external")].copy()
    base = ext.loc[ext["policy"].eq("v50_base")].iloc[0]
    cand = ext.loc[ext["policy"].ne("v50_base")].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.scatter(cand["control_rate"] * 100, cand["balanced_accuracy"] * 100, s=28, color="#b0b7c3", alpha=0.45, label="dev-fit OOD overlays")
    ax.scatter([base["control_rate"] * 100], [base["balanced_accuracy"] * 100], s=95, color="#1f618d", edgecolor="white", label="v50 base")
    if not selected.empty:
        for _, row in selected.iterrows():
            ax.scatter(row["external_control_rate"] * 100, row["external_balanced_accuracy"] * 100, s=90, color="#c0392b", edgecolor="white")
            ax.text(row["external_control_rate"] * 100 + 0.3, row["external_balanced_accuracy"] * 100, row["scenario"], fontsize=7)
    ax.axhline(97, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(99, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External control rate (%)")
    ax.set_ylabel("External workflow BAcc (%)")
    ax.set_title("Dev-fitted OOD/quality controller over v50")
    ax.set_xlim(70, 102)
    ax.set_ylim(96.5, 100.2)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v67_devfit_ood_overlay_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v67_devfit_ood_overlay_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    score_table = build_score_table(dev, ext)
    summary, cases = run_grid(dev, ext, dev_scores, ext_scores, score_table)
    selected = select_dev_feasible(summary)

    score_table.to_csv(OUT_DIR / "v67_ood_scores_case_level.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v67_devfit_ood_overlay_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v67_devfit_ood_overlay_case_routes_q90_q95.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v67_dev_selected_ood_overlay_external_eval.csv", index=False, encoding="utf-8-sig")
    make_plot(summary, selected)

    ext_show = summary.loc[summary["split"].eq("external")].sort_values(["balanced_accuracy", "control_rate"], ascending=[False, True]).head(15)
    print("\nTop external descriptive overlays:")
    print(ext_show[["policy", "score_name", "quantile", "control_rate", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp", "remaining_error_n"]].to_string(index=False))
    if not selected.empty:
        print("\nDev-selected overlays:")
        cols = [
            "scenario",
            "dev_score_name",
            "dev_quantile",
            "dev_control_rate",
            "dev_balanced_accuracy",
            "external_control_rate",
            "external_balanced_accuracy",
            "external_sensitivity",
            "external_specificity",
            "external_fn",
            "external_fp",
        ]
        print(selected[cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
