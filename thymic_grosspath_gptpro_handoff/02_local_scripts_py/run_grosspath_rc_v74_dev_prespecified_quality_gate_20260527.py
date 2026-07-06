from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v74_dev_prespecified_quality_gate_20260527"


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = np.where(out["source_folder"].notna(), "third_batch", "old_data")
    return out


def subset(df: pd.DataFrame, scores: dict[str, np.ndarray], mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    return df.loc[mask].reset_index(drop=True), {k: np.asarray(v)[mask] for k, v in scores.items()}


def v50_review(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> np.ndarray:
    base = v30.top_budget(scores["any"], 0.525)
    return v50.add_top_candidates(df, base, scores["direction"], 0.200, "all_direction")


def standardize_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "contrast_std" not in out.columns and "contrast" in out.columns:
        out["contrast_std"] = out["contrast"]
    if "saturation_mean" in out.columns:
        sat = pd.to_numeric(out["saturation_mean"], errors="coerce")
        if sat.median(skipna=True) > 2:
            out["saturation_mean"] = sat / 255.0
    return out


def quality_proxy_risk(reference: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Label-free image-readability proxy calibrated on development images only."""
    reference = standardize_quality_columns(reference)
    target = standardize_quality_columns(target)
    parts: list[np.ndarray] = []

    if "megapixels" in reference.columns and "megapixels" in target.columns:
        ref = pd.to_numeric(reference["megapixels"], errors="coerce").dropna()
        val = pd.to_numeric(target["megapixels"], errors="coerce")
        p10 = float(ref.quantile(0.10))
        if p10 > 0:
            parts.append(((p10 - val) / p10).clip(lower=0, upper=1).fillna(0).to_numpy(float))

    if "contrast_std" in reference.columns and "contrast_std" in target.columns:
        ref = pd.to_numeric(reference["contrast_std"], errors="coerce").dropna()
        val = pd.to_numeric(target["contrast_std"], errors="coerce")
        p10 = float(ref.quantile(0.10))
        if p10 > 0:
            parts.append(((p10 - val) / p10).clip(lower=0, upper=1).fillna(0).to_numpy(float))

    for col in ["brightness_mean", "saturation_mean"]:
        if col not in reference.columns or col not in target.columns:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        val = pd.to_numeric(target[col], errors="coerce")
        lo = float(ref.quantile(0.10))
        hi = float(ref.quantile(0.90))
        denom = max(hi - lo, float(ref.std()) if len(ref) > 1 else 1.0, 1e-6)
        risk = pd.Series(np.zeros(len(target)), index=target.index, dtype=float)
        low = val < lo
        high = val > hi
        risk.loc[low] = (lo - val.loc[low]) / denom
        risk.loc[high] = (val.loc[high] - hi) / denom
        parts.append(risk.clip(lower=0, upper=1).fillna(0).to_numpy(float))

    if not parts:
        return np.zeros(len(target), dtype=float)
    return np.nanmean(np.vstack(parts), axis=0)


def evaluate(domain: str, gate_name: str, threshold: float | None, df: pd.DataFrame, review: np.ndarray, extra: np.ndarray | None = None) -> dict[str, object]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review] = y[review]
    masks = v48.error_masks(df)
    m = v30.metrics_binary(y, final)
    row: dict[str, object] = {
        "domain": domain,
        "gate_name": gate_name,
        "quality_threshold": threshold,
        "n": int(len(df)),
        "control_n": int(review.sum()),
        "control_rate": float(review.mean()),
        "auto_n": int((~review).sum()),
        "auto_rate": float((~review).mean()),
        "remaining_error_n": int((final != y).sum()),
        "auto_wrong_n": int(((~review) & masks["any_wrong"]).sum()),
        "auto_fn_n": int(((~review) & masks["fn_high_to_low"]).sum()),
        "auto_fp_n": int(((~review) & masks["fp_low_to_high"]).sum()),
        "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
        "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
        "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
    }
    row.update(m)
    if extra is not None:
        row["quality_extra_n"] = int(extra.sum())
        row["quality_extra_rate"] = float(extra.mean())
        row["quality_extra_captured_wrong_n"] = int((extra & masks["any_wrong"]).sum())
        row["quality_extra_captured_fn_n"] = int((extra & masks["fn_high_to_low"]).sum())
        row["quality_extra_captured_fp_n"] = int((extra & masks["fp_low_to_high"]).sum())
    return row


def case_routes(domain: str, gate_name: str, threshold: float | None, df: pd.DataFrame, review: np.ndarray, extra: np.ndarray) -> pd.DataFrame:
    y = df["label_idx"].to_numpy(int)
    p2 = df["p2_pred"].to_numpy(int)
    final = p2.copy()
    final[review] = y[review]
    cols = [
        c
        for c in [
            "case_id",
            "original_case_id",
            "task_l6_label",
            "task_l7_label",
            "source_folder",
            "view_type_final",
            "image_name",
            "quality_score",
            "quality_proxy_risk",
            "quality_status",
            "p2_pred",
            "main_prob",
            "robust_prob",
            "prob_mean_core",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()
    out.insert(0, "domain", domain)
    out.insert(1, "gate_name", gate_name)
    out["quality_threshold"] = threshold
    out["review_or_control"] = review.astype(int)
    out["quality_extra_control"] = extra.astype(int)
    out["label_idx"] = y
    out["final_pred"] = final
    out["p2_wrong"] = (p2 != y).astype(int)
    out["final_correct"] = (final == y).astype(int)
    out["error_direction"] = np.select(
        [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    return out


def build_domains() -> tuple[pd.DataFrame, dict[str, np.ndarray], list[tuple[str, pd.DataFrame, dict[str, np.ndarray]]]]:
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    domains: list[tuple[str, pd.DataFrame, dict[str, np.ndarray]]] = [("development_all", dev, dev_scores)]
    for name in ["old_data", "third_batch"]:
        mask = dev["domain"].eq(name).to_numpy()
        df, scores = subset(dev, dev_scores, mask)
        domains.append((name, df, scores))
    domains.append(("strict_external", ext, ext_scores))
    return dev, dev_scores, domains


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, _dev_scores, domains = build_domains()
    dev = dev.copy()
    dev["quality_proxy_risk"] = quality_proxy_risk(dev, dev)
    thresholds: list[tuple[str, float | None]] = [("v50_base", None)]
    dev_risk = pd.Series(dev["quality_proxy_risk"])
    for q in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        thresholds.append((f"dev_proxy_risk_q{int(q * 100):02d}", float(dev_risk.quantile(q))))
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40]:
        thresholds.append((f"absolute_proxy_risk_ge_{int(thr * 100):02d}", thr))

    rows = []
    cases = []
    for domain, df, scores in domains:
        df = df.copy()
        df["quality_proxy_risk"] = quality_proxy_risk(dev, df)
        base = v50_review(df, scores)
        q = pd.to_numeric(df["quality_proxy_risk"], errors="coerce")
        for gate_name, threshold in thresholds:
            if threshold is None:
                extra = np.zeros(len(df), dtype=bool)
            else:
                extra = q.ge(threshold).fillna(False).to_numpy() & (~base)
            review = base | extra
            rows.append(evaluate(domain, gate_name, threshold, df, review, extra))
            if domain == "strict_external" or gate_name in ["v50_base", "dev_proxy_risk_q85", "absolute_proxy_risk_ge_30"]:
                cases.append(case_routes(domain, gate_name, threshold, df, review, extra))

    summary = pd.DataFrame(rows)
    case_df = pd.concat(cases, ignore_index=True)
    summary.to_csv(OUT_DIR / "v74_dev_prespecified_quality_gate_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(OUT_DIR / "v74_dev_prespecified_quality_gate_case_routes.csv", index=False, encoding="utf-8-sig")

    focus = summary.loc[summary["domain"].isin(["development_all", "old_data", "third_batch", "strict_external"])].copy()
    focus = focus.loc[
        focus["gate_name"].isin(
            [
                "v50_base",
                "dev_proxy_risk_q75",
                "dev_proxy_risk_q85",
                "dev_proxy_risk_q90",
                "absolute_proxy_risk_ge_25",
                "absolute_proxy_risk_ge_30",
            ]
        )
    ]
    print(
        focus[
            [
                "domain",
                "gate_name",
                "quality_threshold",
                "control_rate",
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "remaining_error_n",
                "quality_extra_captured_wrong_n",
            ]
        ]
        .sort_values(["domain", "balanced_accuracy", "control_rate"], ascending=[True, False, True])
        .to_string(index=False)
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
