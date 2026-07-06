from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from run_grosspath_rc_v134c_cascade_auto_corrector_20260527 import build_internal_external


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v138_fast_directional_router_20260527"
V135_PREDS = ROOT / "outputs" / "grosspath_rc_v135_stage1_base_candidate_scan_20260527" / "v135_stage1_candidate_predictions.csv"

BASE_CANDIDATES = ["robust_prob", "prob_mean_core", "stack_gb_d2_multiview"]
FN_SIGNALS = ["prob_mean_core", "wholecrop_prob", "avg_all6", "stack_gb_d2_multiview"]
FP_LOW_SIGNALS = ["prob_mean_core", "robust_prob", "min_all6", "stack_gb_d2_multiview"]


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float | int]:
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "sensitivity_high": np.nan,
            "specificity_low": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "auc": np.nan,
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": np.nan,
    }
    if prob is not None and len(np.unique(y)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y, prob))
        except ValueError:
            pass
    return out


def choose_base_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def load_probs(scope: str) -> pd.DataFrame:
    preds = pd.read_csv(V135_PREDS, dtype={"case_id": str, "original_case_id": str})
    preds = preds.loc[preds["scope"].eq(scope) & preds["objective"].eq("balanced_accuracy")].copy()
    preds = preds.loc[~preds["candidate"].str.contains("subtype", case=False, na=False)].copy()
    return preds.pivot_table(index="case_id", columns="candidate", values="prob_high", aggfunc="first").reset_index()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    internal, external = build_internal_external()
    external = external.loc[~external["task_l6_label"].astype(str).eq("MNT_assumed_low")].copy()
    internal = internal.merge(load_probs("internal_oof_old_third"), on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))
    external = external.merge(load_probs("strict_external_locked"), on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))
    needed = sorted(set(BASE_CANDIDATES + FN_SIGNALS + FP_LOW_SIGNALS))
    for df in (internal, external):
        for col in needed:
            v135_col = f"{col}_v135"
            if col not in df and v135_col in df:
                df[col] = df[v135_col]
            if col not in df:
                df[col] = df["prob_mean_core"]
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(pd.to_numeric(df["prob_mean_core"], errors="coerce").fillna(0.5))
    return internal.reset_index(drop=True), external.reset_index(drop=True), needed


def threshold_grid(values: np.ndarray, high: bool) -> np.ndarray:
    qs = np.linspace(0.55, 0.95, 5) if high else np.linspace(0.05, 0.45, 5)
    return np.unique(np.quantile(values.astype(float), qs))


def score_policy(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_t: float,
    fn_signal: np.ndarray,
    fn_t: float,
    fp_signal: np.ndarray,
    fp_t: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    base_pred = (base_prob >= base_t).astype(int)
    fn_review = (base_pred == 0) & (fn_signal >= fn_t)
    fp_flip = (base_pred == 1) & (fp_signal <= fp_t)
    final_pred = base_pred.copy()
    final_pred[fp_flip] = 0
    automated = ~fn_review
    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = automated & base_wrong & (~final_wrong)
    hurt = automated & (~base_wrong) & final_wrong
    system_pred = final_pred.copy()
    system_pred[fn_review] = y[fn_review]
    final_prob = base_prob.copy()
    final_prob[fp_flip] = fp_signal[fp_flip]
    row = {
        "n": int(len(y)),
        "base_accuracy": metrics(y, base_pred, base_prob)["accuracy"],
        "base_balanced_accuracy": metrics(y, base_pred, base_prob)["balanced_accuracy"],
        "base_auc": metrics(y, base_pred, base_prob)["auc"],
        "auto_n": int(automated.sum()),
        "auto_rate": float(automated.mean()),
        "review_n": int(fn_review.sum()),
        "review_rate": float(fn_review.mean()),
        "fp_flip_n": int(fp_flip.sum()),
        "fn_review_n": int(fn_review.sum()),
        "rescued_n": int(rescued.sum()),
        "hurt_n": int(hurt.sum()),
        "rescued_fn_n": int((rescued & (y == 1) & (base_pred == 0)).sum()),
        "rescued_fp_n": int((rescued & (y == 0) & (base_pred == 1)).sum()),
        "hurt_fn_n": int((hurt & (y == 1) & (final_pred == 0)).sum()),
        "hurt_fp_n": int((hurt & (y == 0) & (final_pred == 1)).sum()),
    }
    row.update({f"auto_{k}": v for k, v in metrics(y[automated], final_pred[automated], final_prob[automated]).items()})
    row.update({f"system_if_review_corrected_{k}": v for k, v in metrics(y, system_pred, final_prob).items()})
    arrays = {
        "base_pred": base_pred,
        "fn_review": fn_review,
        "fp_flip": fp_flip,
        "automated": automated,
        "final_pred": final_pred,
        "rescued": rescued,
        "hurt": hurt,
    }
    return row, arrays


def make_cases(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    base = row["base_candidate"]
    y = df["label_idx"].astype(int).to_numpy()
    out_row, arr = score_policy(
        y,
        df[base].to_numpy(float),
        float(row["base_threshold"]),
        df[row["fn_signal"]].to_numpy(float),
        float(row["fn_threshold"]),
        df[row["fp_low_signal"]].to_numpy(float),
        float(row["fp_threshold"]),
    )
    cases = df[["domain", "case_id", "original_case_id", "task_l6_label", "label_idx"]].copy()
    for k, v in arr.items():
        cases[k] = v
    cases["final_correct"] = cases["final_pred"].astype(int).to_numpy() == y
    cases["rule_tag"] = (
        f"{base}__fn={row['fn_signal']}@{float(row['fn_threshold']):.3f}"
        f"__fp={row['fp_low_signal']}@{float(row['fp_threshold']):.3f}"
    )
    return cases


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:.2f}%"


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("rate") or col in {
            "base_accuracy",
            "base_balanced_accuracy",
            "base_auc",
            "auto_accuracy",
            "auto_balanced_accuracy",
            "system_if_review_corrected_accuracy",
            "system_if_review_corrected_balanced_accuracy",
        }:
            out[col] = out[col].map(lambda v: pct(float(v)) if pd.notna(v) else "")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal, external, _ = prepare()
    y = internal["label_idx"].astype(int).to_numpy()
    y_ext = external["label_idx"].astype(int).to_numpy()
    rows = []
    for base in BASE_CANDIDATES:
        base_t, base_score = choose_base_threshold(y, internal[base].to_numpy(float))
        for fn_name in FN_SIGNALS:
            fn_grid = threshold_grid(internal[fn_name].to_numpy(float), high=True)
            for fp_name in FP_LOW_SIGNALS:
                fp_grid = threshold_grid(internal[fp_name].to_numpy(float), high=False)
                for fn_t in fn_grid:
                    for fp_t in fp_grid:
                        int_row, _ = score_policy(
                            y,
                            internal[base].to_numpy(float),
                            base_t,
                            internal[fn_name].to_numpy(float),
                            float(fn_t),
                            internal[fp_name].to_numpy(float),
                            float(fp_t),
                        )
                        if int_row["review_rate"] > 0.35 or int_row["auto_rate"] < 0.60 or int_row["auto_balanced_accuracy"] < 0.84:
                            continue
                        ext_row, _ = score_policy(
                            y_ext,
                            external[base].to_numpy(float),
                            base_t,
                            external[fn_name].to_numpy(float),
                            float(fn_t),
                            external[fp_name].to_numpy(float),
                            float(fp_t),
                        )
                        meta = {
                            "base_candidate": base,
                            "base_threshold": base_t,
                            "base_selection_score": base_score,
                            "fn_signal": fn_name,
                            "fn_threshold": float(fn_t),
                            "fp_low_signal": fp_name,
                            "fp_threshold": float(fp_t),
                        }
                        int_row.update(meta)
                        int_row["scope"] = "internal_old_third_rule_selected"
                        ext_row.update(meta)
                        ext_row["scope"] = "strict_external_locked"
                        rows.extend([int_row, ext_row])
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No rules survived constraints.")
    summary = summary.sort_values(
        ["scope", "system_if_review_corrected_balanced_accuracy", "auto_balanced_accuracy", "auto_rate"],
        ascending=[True, False, False, False],
    )
    summary.to_csv(OUT_DIR / "v138_fast_directional_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v138_fast_directional_summary_formatted.csv", index=False, encoding="utf-8-sig")
    top = summary.groupby("scope").head(60)
    top.to_csv(OUT_DIR / "v138_top_by_scope.csv", index=False, encoding="utf-8-sig")
    format_table(top).to_csv(OUT_DIR / "v138_top_by_scope_formatted.csv", index=False, encoding="utf-8-sig")
    case_frames = []
    for _, r in summary.loc[summary["scope"].eq("internal_old_third_rule_selected")].head(10).iterrows():
        ic = make_cases(internal, r)
        ec = make_cases(external, r)
        ic["scope"] = "internal_old_third_rule_selected"
        ec["scope"] = "strict_external_locked"
        case_frames.extend([ic, ec])
    pd.concat(case_frames, ignore_index=True).to_csv(OUT_DIR / "v138_top_rule_cases.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": "Fast directional router. No subtype/task-label one-hot. Rules selected on old+third only; strict external excludes MNT_assumed_low.",
        "n_internal": int(len(internal)),
        "n_external_strict": int(len(external)),
        "kept_rule_pairs": int(len(summary) // 2),
    }
    (OUT_DIR / "v138_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_DIR)
    print(format_table(top).to_string(index=False))


if __name__ == "__main__":
    main()
