from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v108_v105_crop_proxy_external_scorecard_20260527"

ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
SELECTED_OOF = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "44_old_third_unified_feature_cv_20260523"
    / "selected_unified_feature_oof_predictions.csv"
)
SELECTED_MODEL = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "44_old_third_unified_feature_cv_20260523"
    / "selected_final_model_old_plus_third.joblib"
)
EXTERNAL_WPC_DIR = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_external_runs"
    / "20_external_thymoma_carcinoma_64style_wpc_20260522"
)

BASE_POLICY = "adaptive_v50_to_v79_light"
PROB_THRESHOLD = 0.25
CORE_MAX = 0.45


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def predict_prob(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    score = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))


def transform_with_saved_preprocessor(bundle: dict, x: np.ndarray) -> np.ndarray:
    pre = bundle.get("preprocessor")
    if pre is None:
        return x.astype(np.float32)
    scaler = pre["scaler"]
    pca = pre["pca"]
    return pca.transform(scaler.transform(x)).astype(np.float32)


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
        "n": int(len(df)),
        "control_rate": float(review.mean()),
        "remaining_error_n": int(rem.sum()),
        "fn": fn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def attach_internal_selected_prob(routes: pd.DataFrame) -> pd.DataFrame:
    base = routes.loc[routes["policy"].eq(BASE_POLICY) & routes["domain"].isin(["old_data", "third_batch"])].copy()
    base["domain_key"] = base["domain"].map({"old_data": "old", "third_batch": "third"})
    oof = pd.read_csv(SELECTED_OOF, dtype={"original_case_id": str})
    oof["domain_key"] = oof["domain"].astype(str)
    prob = oof[["domain_key", "original_case_id", "oof_prob_high"]].rename(columns={"oof_prob_high": "v105_crop_prob"})
    out = base.merge(prob, on=["domain_key", "original_case_id"], how="left", validate="many_to_one")
    if out["v105_crop_prob"].isna().any():
        missing = out.loc[out["v105_crop_prob"].isna(), ["case_id", "original_case_id", "domain"]].head(10)
        raise RuntimeError(f"Missing selected OOF probabilities:\n{missing}")
    return out


def external_crop_proxy_prob() -> pd.DataFrame:
    table = pd.read_csv(EXTERNAL_WPC_DIR / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    features = np.load(EXTERNAL_WPC_DIR / "third_batch_dino_concat_features.npy").astype(np.float32)
    if features.shape[1] % 2 != 0:
        raise ValueError(f"Expected whole+crop feature dim to be even, got {features.shape}")
    half = features.shape[1] // 2
    crop_proxy = features[:, half:].astype(np.float32)

    # Use a relative path here because some Windows Python builds mangle non-ASCII absolute paths.
    rel_model = os.path.join(
        "outputs",
        "batch1_batch2_task567_20260514",
        "task7_adaptation_runs",
        "44_old_third_unified_feature_cv_20260523",
        "selected_final_model_old_plus_third.joblib",
    )
    bundle = joblib.load(rel_model)
    x = transform_with_saved_preprocessor(bundle, crop_proxy)
    prob = predict_prob(bundle["model"], x)
    return pd.DataFrame(
        {
            "case_id": table["case_id"].astype(str),
            "v105_crop_proxy_prob": prob,
            "proxy_source": "second_half_of_existing_whole_plus_crop_feature",
            "wpc_feature_dim": int(features.shape[1]),
            "crop_proxy_dim": int(crop_proxy.shape[1]),
            "selected_model_variant": str(bundle.get("variant")),
            "selected_model_threshold": float(bundle.get("threshold")),
        }
    )


def apply_scorecard(df: pd.DataFrame, prob_col: str) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    return (
        low_auto
        & pd.to_numeric(df[prob_col], errors="coerce").ge(PROB_THRESHOLD)
        & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(CORE_MAX)
    )


def evaluate_scopes(df: pd.DataFrame, prob_col: str, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra = apply_scorecard(df, prob_col)
    review = df["review_or_control"].astype(bool) | extra
    flagged = df.copy()
    flagged["v108_extra_review"] = extra
    flagged["v108_review_or_control"] = review
    flagged["v108_rescued_error"] = extra & df["final_correct"].eq(0)
    flagged["v108_extra_clean_review"] = extra & df["final_correct"].eq(1)
    rows = []
    for scope, sub in [("all", df)] + list(df.groupby("domain", sort=False)):
        sub_extra = extra.loc[sub.index]
        row = {
            "setting": setting,
            "scope": scope,
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


def write_key_messages(summary: pd.DataFrame, internal_cases: pd.DataFrame, external_cases: pd.DataFrame) -> None:
    internal_all = summary.loc[(summary["setting"] == "internal_oof") & (summary["scope"] == "all")].iloc[0]
    third = summary.loc[(summary["setting"] == "internal_oof") & (summary["scope"] == "third_batch")].iloc[0]
    external = summary.loc[(summary["setting"] == "strict_external_crop_proxy") & (summary["scope"] == "strict_external")].iloc[0]
    third_rescued = internal_cases.loc[internal_cases["domain"].eq("third_batch") & internal_cases["v108_rescued_error"], "original_case_id"].astype(str).tolist()
    lines = [
        "# v108 v105 Crop Proxy External Scorecard",
        "",
        "## Key Findings",
        "",
        f"- Scorecard: v105 selected crop probability >= {PROB_THRESHOLD:.2f} and core <= {CORE_MAX:.2f}.",
        f"- Internal OOF all-domain: BAcc {pct(internal_all['balanced_accuracy'])}, control {pct(internal_all['control_rate'])}, FN={int(internal_all['fn'])}, FP={int(internal_all['fp'])}.",
        f"- Internal OOF third batch: BAcc {pct(third['balanced_accuracy'])}, control {pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
        f"- Third-batch rescued residual errors: {', '.join(third_rescued)}.",
        f"- Strict external crop-proxy: BAcc {pct(external['balanced_accuracy'])}, control {pct(external['control_rate'])}, FN={int(external['fn'])}, FP={int(external['fp'])}.",
        "",
        "## Boundary",
        "",
        "This is not exact strict external crop extraction. It applies the saved v105 crop model to the second half of the existing whole+crop external feature. It is a low-cost proxy to decide whether server-side exact crop feature extraction is worth running.",
        "",
    ]
    (OUT_DIR / "v108_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES, dtype={"case_id": str, "original_case_id": str})

    internal = attach_internal_selected_prob(routes)
    internal_summary, internal_cases = evaluate_scopes(internal, "v105_crop_prob", "internal_oof")

    ext_prob = external_crop_proxy_prob()
    strict = routes.loc[routes["policy"].eq(BASE_POLICY) & routes["domain"].eq("strict_external")].copy()
    strict = strict.merge(ext_prob, on="case_id", how="left", validate="many_to_one")
    if strict["v105_crop_proxy_prob"].isna().any():
        missing = strict.loc[strict["v105_crop_proxy_prob"].isna(), ["case_id", "original_case_id"]].head(10)
        raise RuntimeError(f"Missing strict external crop proxy probabilities:\n{missing}")
    external_summary, external_cases = evaluate_scopes(strict, "v105_crop_proxy_prob", "strict_external_crop_proxy")
    summary = pd.concat([internal_summary, external_summary], ignore_index=True)

    summary.to_csv(OUT_DIR / "v108_scorecard_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v108_scorecard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    internal_cases.to_csv(OUT_DIR / "v108_internal_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    external_cases.to_csv(OUT_DIR / "v108_strict_external_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    ext_prob.to_csv(OUT_DIR / "v108_strict_external_crop_proxy_probabilities.csv", index=False, encoding="utf-8-sig")
    write_key_messages(summary, internal_cases, external_cases)

    print("Wrote", OUT_DIR)
    print(format_table(summary.loc[summary["scope"].isin(["all", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print("strict external extra:")
    cols = ["case_id", "original_case_id", "task_l6_label", "label_idx", "final_pred", "prob_mean_core", "v105_crop_proxy_prob", "final_correct"]
    print(external_cases.loc[external_cases["v108_extra_review"], cols].to_string(index=False))


if __name__ == "__main__":
    main()
