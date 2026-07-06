from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
import run_grosspath_rc_v67_devfit_ood_quality_controller_20260527 as v67  # noqa: E402
import run_grosspath_rc_v68_rank_ood_overlay_20260527 as v68  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v70_pseudo_domain_ood_validation_20260527"
FIG_DIR = OUT_DIR / "figures"
EXTRA_RATES = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
SEED = 20260527


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # In the merged development table, third-batch cases have source_folder such as "AB 212";
    # old data keep source_folder as NaN and use source_case_folder/analysis_source_folder.
    out["pseudo_domain"] = np.where(out["source_folder"].notna(), "third_batch", "old_data")
    return out


def v50_review_subset(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> np.ndarray:
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
            "n": int(len(df)),
            "control_n": int(review.sum()),
            "control_rate": float(review.mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "remaining_error_n": int((final != y).sum()),
        }
    )
    return m


def make_subset(df: pd.DataFrame, scores: dict[str, np.ndarray], mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    sub = df.loc[mask].reset_index(drop=True)
    sub_scores = {k: np.asarray(v)[mask] for k, v in scores.items()}
    return sub, sub_scores


def feature_groups(train: pd.DataFrame, target: pd.DataFrame) -> dict[str, list[str]]:
    groups = v67.common_features(train, target)
    return {
        "image_common5": groups["image_common5"],
        "image_plus_model": groups["image_plus_model"],
        "model_prob_compact": groups["model_prob_compact"],
    }


def fit_transform(train: pd.DataFrame, target: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10, 90))),
        ]
    )
    x_train = pipe.fit_transform(train[cols])
    x_target = pipe.transform(target[cols])
    return x_train, x_target


def ood_scores(train: pd.DataFrame, target: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group_name, cols in feature_groups(train, target).items():
        if len(cols) < 2:
            continue
        try:
            x_train, x_target = fit_transform(train, target, cols)
            cov = LedoitWolf().fit(x_train)
            out[f"{group_name}_mahalanobis"] = (cov.mahalanobis(x_train), cov.mahalanobis(x_target))
        except Exception as exc:
            print(f"[skip] {group_name}/mahalanobis: {exc}")
        try:
            x_train, x_target = fit_transform(train, target, cols)
            iso = IsolationForest(n_estimators=400, contamination="auto", random_state=SEED, n_jobs=-1)
            iso.fit(x_train)
            out[f"{group_name}_isolation_forest"] = (-iso.score_samples(x_train), -iso.score_samples(x_target))
        except Exception as exc:
            print(f"[skip] {group_name}/isolation_forest: {exc}")

    heur = v67.heuristic_scores(train, target)
    for name, pair in heur.items():
        out[name] = pair
    return out


def run_direction(
    dev: pd.DataFrame,
    dev_scores: dict[str, np.ndarray],
    train_domain: str,
    target_domain: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = dev["pseudo_domain"].eq(train_domain).to_numpy()
    target_mask = dev["pseudo_domain"].eq(target_domain).to_numpy()
    train, train_scores = make_subset(dev, dev_scores, train_mask)
    target, target_scores = make_subset(dev, dev_scores, target_mask)
    train_base = v50_review_subset(train, train_scores)
    target_base = v50_review_subset(target, target_scores)
    train_masks = v48.error_masks(train)
    target_masks = v48.error_masks(target)
    scores = ood_scores(train, target)

    rows = []
    case_rows = []
    direction = f"{train_domain}_to_{target_domain}"
    rows.append({"direction": direction, "policy": "v50_base", "score_name": "none", "extra_rate": 0.0, "split": "train", **evaluate(train, train_base)})
    rows.append({"direction": direction, "policy": "v50_base", "score_name": "none", "extra_rate": 0.0, "split": "target", **evaluate(target, target_base)})

    for score_name, (train_score, target_score) in scores.items():
        for extra_rate in EXTRA_RATES:
            train_extra = v68.top_rate(train_score, extra_rate)
            target_extra = v68.top_rate(target_score, extra_rate)
            train_review = train_base | train_extra
            target_review = target_base | target_extra
            tr = evaluate(train, train_review)
            te = evaluate(target, target_review)
            tr["extra_captured_wrong_n"] = int((train_extra & (~train_base) & train_masks["any_wrong"]).sum())
            tr["extra_captured_fn_n"] = int((train_extra & (~train_base) & train_masks["fn_high_to_low"]).sum())
            tr["extra_captured_fp_n"] = int((train_extra & (~train_base) & train_masks["fp_low_to_high"]).sum())
            te["extra_captured_wrong_n"] = int((target_extra & (~target_base) & target_masks["any_wrong"]).sum())
            te["extra_captured_fn_n"] = int((target_extra & (~target_base) & target_masks["fn_high_to_low"]).sum())
            te["extra_captured_fp_n"] = int((target_extra & (~target_base) & target_masks["fp_low_to_high"]).sum())
            rows.append({"direction": direction, "policy": "v50_plus_rank_ood", "score_name": score_name, "extra_rate": extra_rate, "split": "train", **tr})
            rows.append({"direction": direction, "policy": "v50_plus_rank_ood", "score_name": score_name, "extra_rate": extra_rate, "split": "target", **te})

            if extra_rate in [0.075, 0.20]:
                tmp = target[
                    [
                        c
                        for c in [
                            "case_id",
                            "original_case_id",
                            "task_l6_label",
                            "task_l7_label",
                            "pseudo_domain",
                            "view_type_final",
                            "p2_pred",
                            "main_prob",
                            "robust_prob",
                            "prob_mean_core",
                        ]
                        if c in target.columns
                    ]
                ].copy()
                y = target["label_idx"].to_numpy(int)
                p2 = target["p2_pred"].to_numpy(int)
                tmp["direction"] = direction
                tmp["score_name"] = score_name
                tmp["extra_rate"] = extra_rate
                tmp["ood_score"] = target_score
                tmp["base_v50_review"] = target_base.astype(int)
                tmp["rank_ood_extra"] = target_extra.astype(int)
                tmp["final_review"] = target_review.astype(int)
                tmp["label_idx"] = y
                tmp["p2_wrong"] = (p2 != y).astype(int)
                tmp["error_direction"] = np.select(
                    [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
                    ["FN_high_to_low", "FP_low_to_high"],
                    default="correct",
                )
                case_rows.append(tmp)

    return pd.DataFrame(rows), pd.concat(case_rows, ignore_index=True)


def select_by_train(summary: pd.DataFrame) -> pd.DataFrame:
    train = summary.loc[summary["split"].eq("train") & summary["policy"].eq("v50_plus_rank_ood")].copy()
    target = summary.loc[summary["split"].eq("target")].copy()
    scenarios = [
        {"scenario": "train_min_control_bacc99", "bacc": 0.99, "control_max": 0.90},
        {"scenario": "train_extra_wrong_low_control", "min_extra_wrong": 1, "control_max": 0.85},
        {"scenario": "train_sens99_spec98", "sens": 0.99, "spec": 0.98, "control_max": 0.90},
    ]
    rows = []
    for direction, sub_train in train.groupby("direction"):
        sub_target = target.loc[target["direction"].eq(direction)]
        for sc in scenarios:
            ok = sub_train.copy()
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
            match = sub_target.loc[sub_target["score_name"].eq(chosen["score_name"]) & sub_target["extra_rate"].eq(chosen["extra_rate"])].iloc[0]
            row = {"direction": direction, "scenario": sc["scenario"]}
            row.update({f"train_{k}": v for k, v in chosen.items() if k not in ["split", "direction"]})
            row.update({f"target_{k}": v for k, v in match.items() if k not in ["split", "direction"]})
            rows.append(row)
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    v67.configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    target = summary.loc[summary["split"].eq("target")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (direction, sub) in zip(axes, target.groupby("direction")):
        base = sub.loc[sub["policy"].eq("v50_base")].iloc[0]
        cand = sub.loc[sub["policy"].eq("v50_plus_rank_ood")]
        ax.scatter(cand["control_rate"] * 100, cand["balanced_accuracy"] * 100, s=28, color="#b0b7c3", alpha=0.5)
        ax.scatter([base["control_rate"] * 100], [base["balanced_accuracy"] * 100], s=90, color="#1f618d", edgecolor="white", label="v50 base")
        sel = selected.loc[selected["direction"].eq(direction)]
        for _, row in sel.iterrows():
            ax.scatter(row["target_control_rate"] * 100, row["target_balanced_accuracy"] * 100, s=80, color="#c0392b", edgecolor="white")
            ax.text(row["target_control_rate"] * 100 + 0.3, row["target_balanced_accuracy"] * 100, row["scenario"], fontsize=7)
        ax.axhline(97, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(99, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(direction)
        ax.set_xlabel("Target-domain control rate (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Target-domain workflow BAcc (%)")
    axes[0].set_ylim(94, 101)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v70_pseudo_domain_ood_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v70_pseudo_domain_ood_transfer.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, _ext, dev_scores, _ext_scores = v50.get_scores()
    dev = add_domain(dev)
    rows = []
    case_rows = []
    for train_domain, target_domain in [("old_data", "third_batch"), ("third_batch", "old_data")]:
        summary, cases = run_direction(dev, dev_scores, train_domain, target_domain)
        rows.append(summary)
        case_rows.append(cases)
    summary_all = pd.concat(rows, ignore_index=True)
    cases_all = pd.concat(case_rows, ignore_index=True)
    selected = select_by_train(summary_all)

    summary_all.to_csv(OUT_DIR / "v70_pseudo_domain_ood_summary.csv", index=False, encoding="utf-8-sig")
    cases_all.to_csv(OUT_DIR / "v70_pseudo_domain_ood_case_routes_q075_q20.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v70_train_selected_target_eval.csv", index=False, encoding="utf-8-sig")
    make_plot(summary_all, selected)

    print("\nTarget-domain top descriptive overlays:")
    top = summary_all.loc[summary_all["split"].eq("target")].sort_values(["direction", "balanced_accuracy", "control_rate"], ascending=[True, False, True]).groupby("direction").head(12)
    print(top[["direction", "policy", "score_name", "extra_rate", "control_rate", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp", "extra_captured_wrong_n"]].to_string(index=False))
    if not selected.empty:
        print("\nTrain-selected target eval:")
        cols = [
            "direction",
            "scenario",
            "train_score_name",
            "train_extra_rate",
            "train_control_rate",
            "train_balanced_accuracy",
            "target_control_rate",
            "target_balanced_accuracy",
            "target_sensitivity",
            "target_specificity",
            "target_fn",
            "target_fp",
            "target_extra_captured_wrong_n",
        ]
        print(selected[cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
