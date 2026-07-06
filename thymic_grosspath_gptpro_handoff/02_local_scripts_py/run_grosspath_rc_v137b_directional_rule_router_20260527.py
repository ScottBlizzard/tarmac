from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from run_grosspath_rc_v134c_cascade_auto_corrector_20260527 import build_internal_external


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v137_directional_rule_router_20260527"
V135_PREDS = ROOT / "outputs" / "grosspath_rc_v135_stage1_base_candidate_scan_20260527" / "v135_stage1_candidate_predictions.csv"

BASE_CANDIDATES = ["robust_prob", "prob_mean_core", "stack_gb_d2_multiview"]
FN_SIGNALS = [
    "prob_mean_core",
    "wholecrop_prob",
    "avg_all6",
    "stack_gb_d2_multiview",
]
FP_LOW_SIGNALS = [
    "prob_mean_core",
    "robust_prob",
    "min_all6",
    "stack_gb_d2_multiview",
]


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


def choose_base_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        if objective == "high_sensitivity":
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            score = sens - 0.2 * max(0.0, 0.70 - spec)
        elif objective == "accuracy":
            score = accuracy_score(y, pred)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def load_probs(scope: str) -> pd.DataFrame:
    preds = pd.read_csv(V135_PREDS, dtype={"case_id": str, "original_case_id": str})
    preds = preds.loc[preds["scope"].eq(scope) & preds["objective"].eq("balanced_accuracy")].copy()
    preds = preds.loc[~preds["candidate"].str.contains("subtype", case=False, na=False)].copy()
    return preds.pivot_table(index="case_id", columns="candidate", values="prob_high", aggfunc="first").reset_index()


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    internal, external = build_internal_external()
    external = external.loc[~external["task_l6_label"].astype(str).eq("MNT_assumed_low")].copy()
    internal_probs = load_probs("internal_oof_old_third")
    external_probs = load_probs("strict_external_locked")
    internal = internal.merge(internal_probs, on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))
    external = external.merge(external_probs, on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))
    all_needed = sorted(set(BASE_CANDIDATES + FN_SIGNALS + FP_LOW_SIGNALS))
    for df in [internal, external]:
        for col in all_needed:
            v135_col = f"{col}_v135"
            if col not in df and v135_col in df:
                df[col] = df[v135_col]
            if col not in df:
                df[col] = df["prob_mean_core"]
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(pd.to_numeric(df["prob_mean_core"], errors="coerce").fillna(0.5))
    usable = [c for c in all_needed if c in internal.columns and c in external.columns]
    return internal.reset_index(drop=True), external.reset_index(drop=True), usable


def apply_policy(
    df: pd.DataFrame,
    base_prob: np.ndarray,
    base_threshold: float,
    fn_signal: np.ndarray,
    fn_threshold: float,
    fp_signal: np.ndarray,
    fp_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    y = df["label_idx"].astype(int).to_numpy()
    base_pred = (base_prob >= base_threshold).astype(int)
    fn_review = (base_pred == 0) & (fn_signal >= fn_threshold)
    fp_flip = (base_pred == 1) & (fp_signal <= fp_threshold)
    reject = fn_review
    final_pred = base_pred.copy()
    final_pred[fp_flip] = 0
    automated = ~reject
    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = automated & base_wrong & (~final_wrong)
    hurt = automated & (~base_wrong) & final_wrong
    system_pred = final_pred.copy()
    system_pred[reject] = y[reject]
    final_prob = base_prob.copy()
    final_prob[fp_flip] = fp_signal[fp_flip]

    row: dict[str, object] = {
        "n": int(len(y)),
        "base_accuracy": metrics(y, base_pred, base_prob)["accuracy"],
        "base_balanced_accuracy": metrics(y, base_pred, base_prob)["balanced_accuracy"],
        "base_auc": metrics(y, base_pred, base_prob)["auc"],
        "auto_n": int(automated.sum()),
        "auto_rate": float(automated.mean()),
        "review_n": int(reject.sum()),
        "review_rate": float(reject.mean()),
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
    case = df[["domain", "case_id", "original_case_id", "task_l6_label", "label_idx"]].copy()
    case["base_prob"] = base_prob
    case["base_pred"] = base_pred
    case["fn_signal"] = fn_signal
    case["fp_signal"] = fp_signal
    case["fn_review"] = fn_review
    case["fp_flip"] = fp_flip
    case["automated"] = automated
    case["final_pred"] = final_pred
    case["final_correct"] = final_pred == y
    case["rescued"] = rescued
    case["hurt"] = hurt
    return case, row


def threshold_grid(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "high":
        qs = np.linspace(0.55, 0.95, 5)
    else:
        qs = np.linspace(0.05, 0.45, 5)
    grid = np.unique(np.quantile(values.astype(float), qs))
    return grid[np.isfinite(grid)]


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
    internal, external, usable = prepare_data()
    y = internal["label_idx"].astype(int).to_numpy()
    rows = []
    selected_case_frames = []
    n = 0
    for base_name in [c for c in BASE_CANDIDATES if c in usable]:
        base_prob = internal[base_name].to_numpy(float)
        base_prob_e = external[base_name].to_numpy(float)
        for base_obj in ["balanced_accuracy"]:
            base_t, base_score = choose_base_threshold(y, base_prob, base_obj)
            for fn_name in [c for c in FN_SIGNALS if c in usable]:
                fn_grid = threshold_grid(internal[fn_name].to_numpy(float), "high")
                for fp_name in [c for c in FP_LOW_SIGNALS if c in usable]:
                    fp_grid = threshold_grid(internal[fp_name].to_numpy(float), "low")
                    for fn_t in fn_grid:
                        for fp_t in fp_grid:
                            int_cases, int_row = apply_policy(
                                internal,
                                base_prob,
                                base_t,
                                internal[fn_name].to_numpy(float),
                                float(fn_t),
                                internal[fp_name].to_numpy(float),
                                float(fp_t),
                            )
                            # Keep the scan realistic: rules with almost all cases reviewed or poor internal auto accuracy are not useful.
                            if int_row["review_rate"] > 0.35 or int_row["auto_rate"] < 0.60 or int_row["auto_balanced_accuracy"] < 0.84:
                                continue
                            ext_cases, ext_row = apply_policy(
                                external,
                                base_prob_e,
                                base_t,
                                external[fn_name].to_numpy(float),
                                float(fn_t),
                                external[fp_name].to_numpy(float),
                                float(fp_t),
                            )
                            meta = {
                                "base_candidate": base_name,
                                "base_objective": base_obj,
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
                            n += 1
                            if n % 1000 == 0:
                                print(f"kept {n} rules", flush=True)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No directional rules survived the internal constraints.")
    summary = summary.sort_values(
        ["scope", "system_if_review_corrected_balanced_accuracy", "auto_balanced_accuracy", "auto_rate"],
        ascending=[True, False, False, False],
    )
    summary.to_csv(OUT_DIR / "v137_directional_rule_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v137_directional_rule_summary_formatted.csv", index=False, encoding="utf-8-sig")
    top = summary.groupby("scope").head(80)
    top.to_csv(OUT_DIR / "v137_top_by_scope.csv", index=False, encoding="utf-8-sig")
    format_table(top).to_csv(OUT_DIR / "v137_top_by_scope_formatted.csv", index=False, encoding="utf-8-sig")

    # Save case routes for the top internal-selected rules only to keep files compact.
    top_internal = summary.loc[summary["scope"].eq("internal_old_third_rule_selected")].head(10)
    for _, r in top_internal.iterrows():
        base_name = r["base_candidate"]
        int_cases, _ = apply_policy(
            internal,
            internal[base_name].to_numpy(float),
            float(r["base_threshold"]),
            internal[r["fn_signal"]].to_numpy(float),
            float(r["fn_threshold"]),
            internal[r["fp_low_signal"]].to_numpy(float),
            float(r["fp_threshold"]),
        )
        ext_cases, _ = apply_policy(
            external,
            external[base_name].to_numpy(float),
            float(r["base_threshold"]),
            external[r["fn_signal"]].to_numpy(float),
            float(r["fn_threshold"]),
            external[r["fp_low_signal"]].to_numpy(float),
            float(r["fp_threshold"]),
        )
        tag = f"{base_name}__{r['fn_signal']}__{r['fp_low_signal']}__bt{float(r['base_threshold']):.3f}__ft{float(r['fn_threshold']):.3f}__pt{float(r['fp_threshold']):.3f}"
        int_cases["rule_tag"] = tag
        ext_cases["rule_tag"] = tag
        int_cases["scope"] = "internal_old_third_rule_selected"
        ext_cases["scope"] = "strict_external_locked"
        selected_case_frames.extend([int_cases, ext_cases])
    pd.concat(selected_case_frames, ignore_index=True).to_csv(OUT_DIR / "v137_top_rule_cases.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": "Directional rule scan uses no subtype/task-label one-hot. Thresholds are selected on old+third only; strict external excludes MNT_assumed_low and is locked.",
        "n_internal": int(len(internal)),
        "n_external_strict": int(len(external)),
        "kept_rules": int(len(summary) // 2),
    }
    (OUT_DIR / "v137_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_DIR)
    print(format_table(top).to_string(index=False))


if __name__ == "__main__":
    main()
