from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_task7_old_third_unified_feature_cv_20260523 as unified  # noqa: E402


ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
UNIFIED_OOF = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_adaptation_runs" / "44_old_third_unified_feature_cv_20260523" / "unified_feature_cv_all_oof_predictions.csv"
EXTERNAL_WHOLE_CROP_DIR = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "20_external_thymoma_carcinoma_64style_wpc_20260522"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v106_external_compatible_wholecrop_scorecard_20260527"
FIG_DIR = OUT_DIR / "figures"

BASE_POLICY = "adaptive_v50_to_v79_light"
VARIANT = "whole_crop"
MODEL_NAME = "logreg_bal_c0.03"
WEIGHT_MODE = "domain_label"
PROB_THRESHOLD = 0.25
CORE_MAX = 0.35


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def metrics(df: pd.DataFrame, review: pd.Series | np.ndarray) -> dict[str, float | int]:
    review = pd.Series(review, index=df.index).astype(bool)
    wrong = df["final_correct"].eq(0)
    rem = (~review) & wrong
    fn = int((rem & df["label_idx"].eq(1) & df["final_pred"].eq(0)).sum())
    fp = int((rem & df["label_idx"].eq(0) & df["final_pred"].eq(1)).sum())
    pos = int(df["label_idx"].eq(1).sum())
    neg = int(df["label_idx"].eq(0).sum())
    sens = (pos - fn) / pos if pos else np.nan
    spec = (neg - fp) / neg if neg else np.nan
    return {
        "n": len(df),
        "control_rate": float(review.mean()),
        "remaining_error_n": int(rem.sum()),
        "fn": fn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def attach_internal_wholecrop_oof(routes: pd.DataFrame) -> pd.DataFrame:
    base = routes.loc[routes["policy"].eq(BASE_POLICY) & routes["domain"].isin(["old_data", "third_batch"])].copy()
    base["domain_key"] = base["domain"].map({"old_data": "old", "third_batch": "third"})
    oof = pd.read_csv(UNIFIED_OOF)
    oof = oof.loc[
        oof["variant"].eq(VARIANT)
        & oof["model"].eq(MODEL_NAME)
        & oof["weight_mode"].eq(WEIGHT_MODE)
    ].copy()
    oof["domain_key"] = oof["domain"].astype(str)
    prob = oof[["domain_key", "original_case_id", "oof_prob_high"]].rename(columns={"oof_prob_high": "wholecrop_prob"})
    out = base.merge(prob, on=["domain_key", "original_case_id"], how="left", validate="many_to_one")
    if out["wholecrop_prob"].isna().any():
        missing = out.loc[out["wholecrop_prob"].isna(), ["case_id", "original_case_id", "domain"]].head(10)
        raise RuntimeError(f"Missing wholecrop OOF probabilities:\n{missing}")
    return out


def fit_wholecrop_external_model() -> tuple[pd.DataFrame, np.ndarray]:
    old, third = unified.make_frames()
    old_dir, third_dir = unified.VARIANTS[VARIANT]
    old_table, old_features = unified.load_feature_dir(old_dir, third=False)
    third_table, third_features = unified.load_feature_dir(third_dir, third=True)
    ext_table, ext_features = unified.load_feature_dir(EXTERNAL_WHOLE_CROP_DIR, third=True)

    old_x = unified.align_features(old["case_id"], old_table, old_features)
    third_x = unified.align_features(third["case_id"], third_table, third_features)
    frame = pd.concat([old, third], ignore_index=True, sort=False)
    x = np.vstack([old_x, third_x]).astype(np.float32)
    y = frame["label_idx"].astype(int).to_numpy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    n_components = min(128, x_scaled.shape[0] - 1, x_scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=20260523)
    x_model = pca.fit_transform(x_scaled).astype(np.float32)

    model = unified.model_grid()[MODEL_NAME]
    weights = unified.sample_weights(frame, WEIGHT_MODE)
    fitted = unified.fit_model(model, x_model, y, weights)

    ext_x = unified.align_features(ext_table["case_id"], ext_table, ext_features)
    ext_model = pca.transform(scaler.transform(ext_x)).astype(np.float32)
    ext_prob = unified.predict_prob(fitted, ext_model)
    ext_pred = pd.DataFrame(
        {
            "case_id": ext_table["case_id"].astype(str),
            "wholecrop_refit_prob": ext_prob,
            "wholecrop_refit_feature_variant": VARIANT,
            "wholecrop_refit_model": MODEL_NAME,
            "wholecrop_refit_weight_mode": WEIGHT_MODE,
            "wholecrop_refit_pca_components": n_components,
            "wholecrop_refit_pca_explained": float(pca.explained_variance_ratio_.sum()),
        }
    )
    return ext_pred, ext_model


def apply_scorecard(df: pd.DataFrame, prob_col: str) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    return low_auto & pd.to_numeric(df[prob_col], errors="coerce").ge(PROB_THRESHOLD) & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(CORE_MAX)


def evaluate_scopes(df: pd.DataFrame, prob_col: str, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra = apply_scorecard(df, prob_col)
    review = df["review_or_control"].astype(bool) | extra
    flagged = df.copy()
    flagged["v106_extra_review"] = extra
    flagged["v106_review_or_control"] = review
    flagged["v106_rescued_error"] = extra & df["final_correct"].eq(0)
    flagged["v106_extra_clean_review"] = extra & df["final_correct"].eq(1)
    rows = []
    for scope, sub in [("all", df)] + list(df.groupby("domain", sort=False)):
        sub_extra = extra.loc[sub.index]
        row = {
            "setting": setting,
            "scope": scope,
            "variant": VARIANT,
            "model": MODEL_NAME,
            "weight_mode": WEIGHT_MODE,
            "prob_threshold": PROB_THRESHOLD,
            "core_max": CORE_MAX,
            "extra_review_n": int(sub_extra.sum()),
            "extra_captured_error_n": int((sub_extra & sub["final_correct"].eq(0)).sum()),
            "extra_clean_review_n": int((sub_extra & sub["final_correct"].eq(1)).sum()),
        }
        row.update(metrics(sub, review.loc[sub.index]))
        rows.append(row)
    return pd.DataFrame(rows), flagged


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    for col in ["prob_threshold", "core_max"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.2f}")
    return out


def make_plot(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    focus = summary.loc[summary["scope"].isin(["all", "third_batch", "strict_external"])].copy()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = focus["setting"] + "\n" + focus["scope"]
    x = np.arange(len(focus))
    ax.bar(x - 0.18, focus["control_rate"] * 100, width=0.36, label="Control", color="#9E9E9E")
    ax.bar(x + 0.18, focus["balanced_accuracy"] * 100, width=0.36, label="BAcc", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("%")
    ax.set_title("External-compatible whole-crop scorecard")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v106_external_compatible_wholecrop_scorecard.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v106_external_compatible_wholecrop_scorecard.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: pd.DataFrame, internal_cases: pd.DataFrame, external_cases: pd.DataFrame) -> None:
    internal_all = summary.loc[(summary["setting"] == "internal_oof") & (summary["scope"] == "all")].iloc[0]
    third = summary.loc[(summary["setting"] == "internal_oof") & (summary["scope"] == "third_batch")].iloc[0]
    external = summary.loc[(summary["setting"] == "strict_external_refit") & (summary["scope"] == "strict_external")].iloc[0]
    third_rescued = internal_cases.loc[internal_cases["domain"].eq("third_batch") & internal_cases["v106_rescued_error"], "original_case_id"].astype(str).tolist()
    lines = [
        "# v106 External-compatible Whole-crop Scorecard",
        "",
        "## Key Findings",
        "",
        f"- Scorecard: `{VARIANT} / {MODEL_NAME} / {WEIGHT_MODE}` probability >= {PROB_THRESHOLD:.2f} and core <= {CORE_MAX:.2f}.",
        f"- Internal OOF all-domain: BAcc {pct(internal_all['balanced_accuracy'])}, control {pct(internal_all['control_rate'])}, FN={int(internal_all['fn'])}, FP={int(internal_all['fp'])}.",
        f"- Internal OOF third batch: BAcc {pct(third['balanced_accuracy'])}, control {pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
        f"- Third-batch rescued residual errors: {', '.join(third_rescued)}.",
        f"- Strict external refit on old+third, applied to external whole-crop features: BAcc {pct(external['balanced_accuracy'])}, control {pct(external['control_rate'])}, FN={int(external['fn'])}, FP={int(external['fp'])}.",
        "",
        "## Boundary",
        "",
        "This is stronger than v102 because the strict external feature dimensionality matches the internal whole-crop feature family. It is still not the exact v105 crop-based selected_unified scorecard; it is an external-compatible whole-crop surrogate chosen because strict external whole-crop features already exist.",
        "",
    ]
    (OUT_DIR / "v106_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES)
    internal = attach_internal_wholecrop_oof(routes)
    internal_summary, internal_cases = evaluate_scopes(internal, "wholecrop_prob", "internal_oof")

    ext_pred, _ = fit_wholecrop_external_model()
    strict = routes.loc[routes["policy"].eq(BASE_POLICY) & routes["domain"].eq("strict_external")].copy()
    strict = strict.merge(ext_pred, on="case_id", how="left", validate="many_to_one")
    if strict["wholecrop_refit_prob"].isna().any():
        raise RuntimeError("Missing strict external wholecrop refit probabilities")
    external_summary, external_cases = evaluate_scopes(strict, "wholecrop_refit_prob", "strict_external_refit")
    summary = pd.concat([internal_summary, external_summary], ignore_index=True)

    summary.to_csv(OUT_DIR / "v106_scorecard_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v106_scorecard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    internal_cases.to_csv(OUT_DIR / "v106_internal_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    external_cases.to_csv(OUT_DIR / "v106_strict_external_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    ext_pred.to_csv(OUT_DIR / "v106_strict_external_wholecrop_refit_probabilities.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)
    write_summary(summary, internal_cases, external_cases)

    print("Wrote", OUT_DIR)
    print(format_table(summary.loc[summary["scope"].isin(["all", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print("third rescued:")
    print(internal_cases.loc[internal_cases["domain"].eq("third_batch") & internal_cases["v106_rescued_error"], ["original_case_id", "task_l6_label", "prob_mean_core", "wholecrop_prob"]].to_string(index=False))
    print()
    print("strict external extra:")
    print(external_cases.loc[external_cases["v106_extra_review"], ["case_id", "original_case_id", "task_l6_label", "label_idx", "final_pred", "prob_mean_core", "wholecrop_refit_prob", "final_correct"]].to_string(index=False))


if __name__ == "__main__":
    main()
