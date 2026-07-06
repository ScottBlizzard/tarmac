from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v67_devfit_ood_quality_controller_20260527 as v67  # noqa: E402
import run_grosspath_rc_v73_pseudodomain_policy_search_20260527 as v73  # noqa: E402
import run_grosspath_rc_v75_joint_quality_risk_ood_search_20260527 as v75  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527"
FIG_DIR = OUT_DIR / "figures"
CORE_QUALITY_FEATURES = ["megapixels", "brightness_mean", "contrast_std", "saturation_mean"]


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = np.where(out["source_folder"].notna(), "third_batch", "old_data")
    return out


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "contrast_std" not in out.columns and "contrast" in out.columns:
        out["contrast_std"] = out["contrast"]
    if "saturation_mean" in out.columns:
        sat = pd.to_numeric(out["saturation_mean"], errors="coerce")
        if sat.median(skipna=True) > 2:
            out["saturation_mean"] = sat / 255.0
    return out


def available_features(reference: pd.DataFrame, target: pd.DataFrame) -> list[str]:
    reference = standardize(reference)
    target = standardize(target)
    return [c for c in CORE_QUALITY_FEATURES if c in reference.columns and c in target.columns]


def feature_shift(reference: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    reference = standardize(reference)
    target = standardize(target)
    rows = []
    for col in features:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        tgt = pd.to_numeric(target[col], errors="coerce").dropna()
        if ref.empty or tgt.empty:
            continue
        q05, q25, q50, q75, q95 = np.nanquantile(ref, [0.05, 0.25, 0.50, 0.75, 0.95])
        iqr = max(q75 - q25, ref.std() if len(ref) > 1 else 0.0, 1e-6)
        t50 = float(np.nanmedian(tgt))
        rows.append(
            {
                "feature": col,
                "reference_median": float(q50),
                "target_median": t50,
                "abs_median_shift_iqr": float(abs(t50 - q50) / iqr),
                "target_outside_ref_05_95_rate": float(((tgt < q05) | (tgt > q95)).mean()),
                "target_below_ref_05_rate": float((tgt < q05).mean()),
                "target_above_ref_95_rate": float((tgt > q95).mean()),
                "target_median_reference_quantile": float((ref <= t50).mean()),
            }
        )
    return pd.DataFrame(rows)


def quality_proxy(reference: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    # Reuse the v74 design but keep v77 self-contained on standardized columns.
    import run_grosspath_rc_v74_dev_prespecified_quality_gate_20260527 as v74  # noqa: WPS433

    return v74.quality_proxy_risk(standardize(reference), standardize(target))


def domain_auc(reference: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> float:
    reference = standardize(reference)
    target = standardize(target)
    x = pd.concat([reference[features], target[features]], ignore_index=True)
    y = np.array([0] * len(reference) + [1] * len(target), dtype=int)
    if min(np.bincount(y)) < 3:
        return np.nan
    n_splits = int(min(5, np.bincount(y).min()))
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10, 90))),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260527)
    prob = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    return max(auc, 1.0 - auc)


def audit_pair(name: str, reference_name: str, target_name: str, reference: pd.DataFrame, target: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    features = available_features(reference, target)
    fshift = feature_shift(reference, target, features)
    qp = quality_proxy(reference, target)
    auc = domain_auc(reference, target, features)
    mean_abs_shift = float(fshift["abs_median_shift_iqr"].mean()) if not fshift.empty else np.nan
    mean_outside = float(fshift["target_outside_ref_05_95_rate"].mean()) if not fshift.empty else np.nan
    qp_mean = float(np.nanmean(qp))
    qp_p75 = float(np.nanquantile(qp, 0.75))
    qp_p90 = float(np.nanquantile(qp, 0.90))
    shift_index = float(mean_abs_shift + mean_outside + qp_mean + max(0.0, auc - 0.5))
    row = {
        "audit_name": name,
        "reference": reference_name,
        "target": target_name,
        "n_reference": int(len(reference)),
        "n_target": int(len(target)),
        "features": ",".join(features),
        "domain_auc_cv": auc,
        "mean_abs_median_shift_iqr": mean_abs_shift,
        "mean_outside_ref_05_95_rate": mean_outside,
        "quality_proxy_mean": qp_mean,
        "quality_proxy_p75": qp_p75,
        "quality_proxy_p90": qp_p90,
        "batch_shift_index": shift_index,
    }
    fshift.insert(0, "audit_name", name)
    fshift.insert(1, "reference", reference_name)
    fshift.insert(2, "target", target_name)
    return row, fshift


def v75_review(reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray]) -> np.ndarray:
    base = v73.v50_review(target, scores)
    joint = v75.joint_scores(reference, target, scores)
    extra = v73.top_by_rate(joint["quality_plus_lowconf_mean"], 0.30) & (~base)
    return base | extra


def evaluate_policy(domain: str, reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray], policy: str) -> dict[str, object]:
    if policy == "v50_main":
        review = v73.v50_review(target, scores)
    elif policy == "v75_quality_lowconf":
        review = v75_review(reference, target, scores)
    else:
        raise ValueError(policy)
    out = v73.evaluate(domain, policy, target, review)
    return out


def classify_shift(row: pd.Series, internal_max: pd.Series) -> str:
    severe = (
        row["batch_shift_index"] > internal_max["batch_shift_index"] * 1.35
        or row["domain_auc_cv"] > max(0.90, internal_max["domain_auc_cv"] + 0.05)
        or row["quality_proxy_mean"] > internal_max["quality_proxy_mean"] * 1.50
    )
    moderate = (
        row["batch_shift_index"] > internal_max["batch_shift_index"]
        or row["quality_proxy_mean"] > internal_max["quality_proxy_mean"]
    )
    if severe:
        return "severe_shift"
    if moderate:
        return "moderate_shift"
    return "within_internal_shift"


def make_plot(audit: pd.DataFrame) -> None:
    v67.configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot = audit.copy()
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    colors = {
        "pseudo_old_to_third": "#d68910",
        "pseudo_third_to_old": "#1f618d",
        "strict_external_vs_dev": "#c0392b",
    }
    for _, row in plot.iterrows():
        ax.scatter(
            row["domain_auc_cv"],
            row["quality_proxy_mean"],
            s=110,
            color=colors.get(row["audit_name"], "#566573"),
            edgecolor="white",
            linewidth=1.0,
        )
        ax.text(row["domain_auc_cv"] + 0.005, row["quality_proxy_mean"] + 0.003, row["target"], fontsize=9)
    ax.set_xlabel("Batch separability AUC from reference (unlabeled)")
    ax.set_ylabel("Mean quality proxy risk")
    ax.set_title("Unlabeled batch shift audit")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v77_batch_shift_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v77_batch_shift_audit.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    old_df, old_scores = v73.subset(dev, dev_scores, dev["domain"].eq("old_data").to_numpy())
    third_df, third_scores = v73.subset(dev, dev_scores, dev["domain"].eq("third_batch").to_numpy())

    audits = []
    feature_rows = []
    specs = [
        ("pseudo_old_to_third", "old_data", "third_batch", old_df, third_df),
        ("pseudo_third_to_old", "third_batch", "old_data", third_df, old_df),
        ("strict_external_vs_dev", "development_all", "strict_external", dev, ext),
    ]
    for name, ref_name, target_name, ref, target in specs:
        row, fshift = audit_pair(name, ref_name, target_name, ref, target)
        audits.append(row)
        feature_rows.append(fshift)
    audit_df = pd.DataFrame(audits)
    feature_df = pd.concat(feature_rows, ignore_index=True)
    internal = audit_df.loc[audit_df["audit_name"].str.startswith("pseudo_")]
    internal_max = internal[["domain_auc_cv", "quality_proxy_mean", "batch_shift_index"]].max()
    audit_df["shift_category"] = audit_df.apply(lambda r: classify_shift(r, internal_max), axis=1)
    audit_df["recommended_policy"] = np.where(audit_df["shift_category"].eq("within_internal_shift"), "v50_main", "v75_quality_lowconf")
    audit_df["recommendation_note"] = np.select(
        [
            audit_df["shift_category"].eq("severe_shift"),
            audit_df["shift_category"].eq("moderate_shift"),
        ],
        [
            "batch exceeds internal development shift; use safety-enhanced workflow and consider retake/review",
            "batch exceeds at least one internal shift metric; use safety-enhanced workflow",
        ],
        default="within internal shift envelope; standard v50 workflow",
    )

    policy_rows = []
    policy_specs = [
        ("old_data", third_df, old_df, old_scores),
        ("third_batch", old_df, third_df, third_scores),
        ("strict_external", dev, ext, ext_scores),
    ]
    for domain, ref, target, scores in policy_specs:
        for policy in ["v50_main", "v75_quality_lowconf"]:
            policy_rows.append(evaluate_policy(domain, ref, target, scores, policy))
    policy_df = pd.DataFrame(policy_rows)

    audit_df.to_csv(OUT_DIR / "v77_unlabeled_batch_shift_audit.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(OUT_DIR / "v77_feature_shift_detail.csv", index=False, encoding="utf-8-sig")
    policy_df.to_csv(OUT_DIR / "v77_policy_effect_by_domain.csv", index=False, encoding="utf-8-sig")
    make_plot(audit_df)

    print("Unlabeled batch shift audit:")
    print(
        audit_df[
            [
                "audit_name",
                "reference",
                "target",
                "domain_auc_cv",
                "mean_abs_median_shift_iqr",
                "mean_outside_ref_05_95_rate",
                "quality_proxy_mean",
                "batch_shift_index",
                "shift_category",
                "recommended_policy",
            ]
        ].to_string(index=False)
    )
    print("\nPolicy effect by domain:")
    print(
        policy_df[
            [
                "split",
                "policy",
                "control_rate",
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "remaining_error_n",
            ]
        ].to_string(index=False)
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
