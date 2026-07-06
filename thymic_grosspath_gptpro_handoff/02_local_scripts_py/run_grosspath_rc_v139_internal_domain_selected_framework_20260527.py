from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from run_grosspath_rc_v134c_cascade_auto_corrector_20260527 import build_internal_external


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v139_internal_domain_selected_framework_20260527"
V135_PREDS = ROOT / "outputs" / "grosspath_rc_v135_stage1_base_candidate_scan_20260527" / "v135_stage1_candidate_predictions.csv"

BASE_CANDIDATES = [
    "robust_prob",
    "prob_mean_core",
    "main_prob",
    "stack_gb_d2_multiview",
]
FN_SIGNALS = ["prob_mean_core", "wholecrop_prob", "stack_gb_d2_multiview"]
FP_LOW_SIGNALS = ["prob_mean_core", "robust_prob", "stack_gb_d2_multiview"]
CONF_QUANTILES = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
FN_QUANTILES = np.array([0.60, 0.75, 0.85, 0.93])
FP_QUANTILES = np.array([0.07, 0.15, 0.30, 0.42])
MIN_INTERNAL_AUTO_RATE = 0.45


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


def choose_base_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        score = balanced_accuracy_score(y, (prob >= t).astype(int))
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t


def load_probs(scope: str) -> pd.DataFrame:
    preds = pd.read_csv(V135_PREDS, dtype={"case_id": str, "original_case_id": str})
    preds = preds.loc[preds["scope"].eq(scope) & preds["objective"].eq("balanced_accuracy")].copy()
    preds = preds.loc[~preds["candidate"].str.contains("subtype", case=False, na=False)].copy()
    return preds.pivot_table(index="case_id", columns="candidate", values="prob_high", aggfunc="first").reset_index()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return internal.reset_index(drop=True), external.reset_index(drop=True)


def score_policy(
    df: pd.DataFrame,
    base_name: str,
    base_t: float,
    conf_t: float,
    fn_name: str,
    fn_t: float,
    fp_name: str,
    fp_t: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    y = df["label_idx"].astype(int).to_numpy()
    base_prob = df[base_name].to_numpy(float)
    base_pred = (base_prob >= base_t).astype(int)
    base_conf = np.where(base_pred == 1, base_prob, 1.0 - base_prob)
    fn_signal = df[fn_name].to_numpy(float)
    fp_signal = df[fp_name].to_numpy(float)

    high_conf_accept = base_conf >= conf_t
    fn_review = (base_pred == 0) & (fn_signal >= fn_t)
    fp_flip = (base_pred == 1) & (fp_signal <= fp_t)

    final_pred = base_pred.copy()
    final_pred[fp_flip] = 0
    final_prob = base_prob.copy()
    final_prob[fp_flip] = np.minimum(base_prob[fp_flip], fp_signal[fp_flip])

    automated = (high_conf_accept | fp_flip) & (~fn_review)
    review = ~automated
    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = automated & base_wrong & (~final_wrong)
    hurt = automated & (~base_wrong) & final_wrong
    system_pred = final_pred.copy()
    system_pred[review] = y[review]

    row: dict[str, object] = {
        "n": int(len(y)),
        "base_accuracy": metrics(y, base_pred, base_prob)["accuracy"],
        "base_balanced_accuracy": metrics(y, base_pred, base_prob)["balanced_accuracy"],
        "base_auc": metrics(y, base_pred, base_prob)["auc"],
        "auto_n": int(automated.sum()),
        "auto_rate": float(automated.mean()),
        "review_n": int(review.sum()),
        "review_rate": float(review.mean()),
        "high_conf_accept_n": int(high_conf_accept.sum()),
        "fn_review_n": int(fn_review.sum()),
        "fp_flip_n": int(fp_flip.sum()),
        "rescued_n": int(rescued.sum()),
        "hurt_n": int(hurt.sum()),
        "rescued_fn_n": int((rescued & (y == 1) & (base_pred == 0)).sum()),
        "rescued_fp_n": int((rescued & (y == 0) & (base_pred == 1)).sum()),
        "hurt_fn_n": int((hurt & (y == 1) & (final_pred == 0)).sum()),
        "hurt_fp_n": int((hurt & (y == 0) & (final_pred == 1)).sum()),
    }
    row.update({f"auto_{k}": v for k, v in metrics(y[automated], final_pred[automated], final_prob[automated]).items()})
    row.update({f"review_{k}": v for k, v in metrics(y[review], base_pred[review], base_prob[review]).items()})
    row.update({f"system_if_review_corrected_{k}": v for k, v in metrics(y, system_pred, final_prob).items()})
    arrays = {
        "base_prob": base_prob,
        "base_pred": base_pred,
        "base_conf": base_conf,
        "high_conf_accept": high_conf_accept,
        "fn_review": fn_review,
        "fp_flip": fp_flip,
        "automated": automated,
        "review": review,
        "final_pred": final_pred,
        "final_prob": final_prob,
        "rescued": rescued,
        "hurt": hurt,
        "base_wrong": base_wrong,
        "final_wrong": final_wrong,
    }
    return row, arrays


def quantile_grid(values: np.ndarray, qs: np.ndarray) -> np.ndarray:
    out = np.unique(np.quantile(values.astype(float), qs))
    return out[np.isfinite(out)]


def prefixed(scope: str, row: dict[str, object]) -> dict[str, object]:
    return {f"{scope}_{k}": v for k, v in row.items()}


def make_cases(df: pd.DataFrame, rule: pd.Series, scope: str) -> pd.DataFrame:
    _, arr = score_policy(
        df,
        str(rule["base_candidate"]),
        float(rule["base_threshold"]),
        float(rule["conf_threshold"]),
        str(rule["fn_signal"]),
        float(rule["fn_threshold"]),
        str(rule["fp_low_signal"]),
        float(rule["fp_threshold"]),
    )
    cols = ["domain", "case_id", "original_case_id", "task_l6_label", "label_idx"]
    out = df[cols].copy()
    out["scope"] = scope
    for name, values in arr.items():
        out[name] = values
    out["base_correct"] = ~out["base_wrong"].astype(bool)
    out["final_auto_correct"] = ~out["final_wrong"].astype(bool)
    return out


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * float(x):.2f}%"


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_cols = [c for c in out.columns if c.endswith(("accuracy", "balanced_accuracy", "auc", "f1", "rate", "sensitivity_high", "specificity_low"))]
    for col in pct_cols:
        out[col] = out[col].map(lambda v: pct(v) if pd.notna(v) else "")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal, external = prepare()
    y = internal["label_idx"].astype(int).to_numpy()
    rows: list[dict[str, object]] = []

    for base in BASE_CANDIDATES:
        print(f"[v139] scanning base={base}", flush=True)
        base_t = choose_base_threshold(y, internal[base].to_numpy(float))
        base_pred = (internal[base].to_numpy(float) >= base_t).astype(int)
        base_conf = np.where(base_pred == 1, internal[base].to_numpy(float), 1.0 - internal[base].to_numpy(float))
        conf_grid = quantile_grid(base_conf, CONF_QUANTILES)
        for conf_t in conf_grid:
            for fn_name in FN_SIGNALS:
                for fn_t in quantile_grid(internal[fn_name].to_numpy(float), FN_QUANTILES):
                    for fp_name in FP_LOW_SIGNALS:
                        for fp_t in quantile_grid(internal[fp_name].to_numpy(float), FP_QUANTILES):
                            meta = {
                                "base_candidate": base,
                                "base_threshold": float(base_t),
                                "conf_threshold": float(conf_t),
                                "fn_signal": fn_name,
                                "fn_threshold": float(fn_t),
                                "fp_low_signal": fp_name,
                                "fp_threshold": float(fp_t),
                            }
                            all_row, _ = score_policy(internal, base, base_t, conf_t, fn_name, fn_t, fp_name, fp_t)
                            if all_row["auto_rate"] < MIN_INTERNAL_AUTO_RATE:
                                continue
                            old_row, _ = score_policy(internal.loc[internal["domain"].eq("old_data")], base, base_t, conf_t, fn_name, fn_t, fp_name, fp_t)
                            third_row, _ = score_policy(internal.loc[internal["domain"].eq("third_batch")], base, base_t, conf_t, fn_name, fn_t, fp_name, fp_t)
                            ext_row, _ = score_policy(external, base, base_t, conf_t, fn_name, fn_t, fp_name, fp_t)
                            row = dict(meta)
                            row.update(prefixed("internal", all_row))
                            row.update(prefixed("old", old_row))
                            row.update(prefixed("third", third_row))
                            row.update(prefixed("strict_external", ext_row))
                            row["development_min_auto_bacc"] = float(np.nanmin([old_row["auto_balanced_accuracy"], third_row["auto_balanced_accuracy"]]))
                            row["development_min_system_bacc"] = float(np.nanmin([old_row["system_if_review_corrected_balanced_accuracy"], third_row["system_if_review_corrected_balanced_accuracy"]]))
                            row["development_min_auto_rate"] = float(np.nanmin([old_row["auto_rate"], third_row["auto_rate"]]))
                            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No v139 candidate rule survived constraints.")
    summary = summary.sort_values(
        [
            "development_min_system_bacc",
            "development_min_auto_bacc",
            "development_min_auto_rate",
            "internal_system_if_review_corrected_balanced_accuracy",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "v139_internal_domain_selected_framework_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary.head(500)).to_csv(OUT_DIR / "v139_internal_domain_selected_framework_top500_formatted.csv", index=False, encoding="utf-8-sig")

    selected = summary.head(1).iloc[0]
    selected.to_frame().T.to_csv(OUT_DIR / "v139_selected_rule.csv", index=False, encoding="utf-8-sig")
    format_table(selected.to_frame().T).to_csv(OUT_DIR / "v139_selected_rule_formatted.csv", index=False, encoding="utf-8-sig")

    case_frames = [
        make_cases(internal, selected, "internal_old_third"),
        make_cases(internal.loc[internal["domain"].eq("old_data")], selected, "old"),
        make_cases(internal.loc[internal["domain"].eq("third_batch")], selected, "third"),
        make_cases(external, selected, "strict_external_locked"),
    ]
    pd.concat(case_frames, ignore_index=True).to_csv(OUT_DIR / "v139_selected_rule_cases.csv", index=False, encoding="utf-8-sig")

    report = {
        "output_dir": str(OUT_DIR),
        "internal_n": int(len(internal)),
        "external_strict_n": int(len(external)),
        "candidate_rules": int(len(summary)),
        "selection": "Selected only by old+third development domains; strict external is reported after selection.",
        "leakage_guard": "No task_l6_label/subtype features are used as model inputs; task_l6_label is retained only for case analysis.",
        "selected_key_metrics": {
            "internal_auto_rate": float(selected["internal_auto_rate"]),
            "internal_auto_balanced_accuracy": float(selected["internal_auto_balanced_accuracy"]),
            "internal_system_if_review_corrected_balanced_accuracy": float(selected["internal_system_if_review_corrected_balanced_accuracy"]),
            "third_auto_balanced_accuracy": float(selected["third_auto_balanced_accuracy"]),
            "strict_external_auto_rate": float(selected["strict_external_auto_rate"]),
            "strict_external_auto_balanced_accuracy": float(selected["strict_external_auto_balanced_accuracy"]),
            "strict_external_system_if_review_corrected_balanced_accuracy": float(selected["strict_external_system_if_review_corrected_balanced_accuracy"]),
        },
    }
    (OUT_DIR / "v139_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
