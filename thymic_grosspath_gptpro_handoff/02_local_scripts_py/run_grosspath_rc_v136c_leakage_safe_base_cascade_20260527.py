from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_grosspath_rc_v134c_cascade_auto_corrector_20260527 import build_internal_external, base_feature_frame


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v136_leakage_safe_base_cascade_20260527"
V135_PREDS = ROOT / "outputs" / "grosspath_rc_v135_stage1_base_candidate_scan_20260527" / "v135_stage1_candidate_predictions.csv"

BASE_CANDIDATES = [
    "robust_prob",
    "prob_mean_core",
    "stack_gb_d2_core",
    "stack_gb_d2_multiview",
    "stack_extra_d3_core",
    "stack_logreg_c03_multiview",
]

OBJECTIVES = ["balanced_accuracy"]
TARGET_ACCEPT_ACCS = [0.90]
CORR_CONF_THRESHOLDS = [0.00, 0.65]
MIN_ACCEPT_TRAIN = 30


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:.2f}%"


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


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
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        elif objective == "high_sensitivity":
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            score = sens - 0.25 * max(0.0, 0.70 - spec)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def select_accept_threshold(score: np.ndarray, correct: np.ndarray, target_acc: float) -> dict[str, float | int]:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_correct = correct[order].astype(float)
    ks = np.arange(1, len(sorted_score) + 1)
    acc = np.cumsum(sorted_correct) / ks
    ok = (acc >= target_acc) & (ks >= MIN_ACCEPT_TRAIN)
    if not ok.any():
        return {"threshold": np.inf, "train_accept_n": 0, "train_accept_acc": np.nan}
    idx = np.where(ok)[0][-1]
    return {"threshold": float(sorted_score[idx]), "train_accept_n": int(idx + 1), "train_accept_acc": float(acc[idx])}


def fit_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    clf = clone(model)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def make_models(seed: int) -> dict[str, object]:
    return {
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "extra_d3_l8": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb_d2": GradientBoostingClassifier(max_depth=2, learning_rate=0.035, n_estimators=120, random_state=seed),
    }


def load_candidate_probs(scope: str) -> pd.DataFrame:
    preds = pd.read_csv(V135_PREDS, dtype={"case_id": str, "original_case_id": str})
    preds = preds.loc[preds["scope"].eq(scope) & preds["objective"].eq("balanced_accuracy")].copy()
    preds = preds.loc[~preds["candidate"].str.contains("subtype", case=False, na=False)].copy()
    wide = preds.pivot_table(index="case_id", columns="candidate", values="prob_high", aggfunc="first").reset_index()
    return wide


def make_feature_frame(df: pd.DataFrame, cand_prob: np.ndarray, cand_pred: np.ndarray, include_quality: bool) -> pd.DataFrame:
    x = base_feature_frame(df)
    x = x[[c for c in x.columns if not c.startswith("subtype_") and c != "domain_is_external"]].copy()
    if not include_quality and "quality_score" in x:
        x = x.drop(columns=["quality_score"])
    x["cand_prob"] = cand_prob.astype(float)
    x["cand_pred"] = cand_pred.astype(float)
    x["cand_conf"] = np.where(cand_pred.astype(int) == 1, cand_prob, 1.0 - cand_prob)
    x["cand_entropy"] = entropy(cand_prob)
    x["cand_core_gap"] = np.abs(cand_prob - pd.to_numeric(df["prob_mean_core"], errors="coerce").fillna(0.5).to_numpy(float))
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def finalize_cases(
    df: pd.DataFrame,
    cand_prob: np.ndarray,
    cand_pred: np.ndarray,
    accept_score: np.ndarray,
    accept_threshold: np.ndarray | float,
    corr_prob: np.ndarray,
    corr_conf_min: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    y = df["label_idx"].astype(int).to_numpy()
    if np.isscalar(accept_threshold):
        stage1 = accept_score >= float(accept_threshold)
    else:
        stage1 = accept_score >= accept_threshold.astype(float)
    corr_pred = (corr_prob >= 0.5).astype(int)
    corr_conf = np.maximum(corr_prob, 1.0 - corr_prob)
    stage2 = (~stage1) & (corr_conf >= corr_conf_min)
    reject = (~stage1) & (~stage2)
    final_pred = cand_pred.copy()
    final_prob = cand_prob.copy()
    final_pred[stage2] = corr_pred[stage2]
    final_prob[stage2] = corr_prob[stage2]
    automated = stage1 | stage2
    base_wrong = cand_pred != y
    final_wrong = final_pred != y
    rescued = (~stage1) & base_wrong & (~final_wrong)
    hurt = (~stage1) & (~base_wrong) & final_wrong
    system_pred = final_pred.copy()
    system_pred[reject] = y[reject]

    case = df[["domain", "case_id", "original_case_id", "task_l6_label", "label_idx"]].copy()
    case["base_prob"] = cand_prob
    case["base_pred"] = cand_pred
    case["base_correct"] = cand_pred == y
    case["accept_score"] = accept_score
    case["stage1_accept"] = stage1
    case["stage2_prob_high"] = corr_prob
    case["stage2_conf"] = corr_conf
    case["stage2_accept"] = stage2
    case["reject"] = reject
    case["automated"] = automated
    case["final_auto_pred"] = final_pred
    case["final_auto_correct"] = final_pred == y
    case["rescued"] = rescued
    case["hurt"] = hurt

    row: dict[str, object] = {
        "n": int(len(y)),
        "base_accuracy": metrics(y, cand_pred, cand_prob)["accuracy"],
        "base_balanced_accuracy": metrics(y, cand_pred, cand_prob)["balanced_accuracy"],
        "base_auc": metrics(y, cand_pred, cand_prob)["auc"],
        "stage1_accept_n": int(stage1.sum()),
        "stage1_accept_rate": float(stage1.mean()),
        "stage2_accept_n": int(stage2.sum()),
        "stage2_accept_rate": float(stage2.mean()),
        "reject_n": int(reject.sum()),
        "reject_rate": float(reject.mean()),
        "automated_n": int(automated.sum()),
        "automated_rate": float(automated.mean()),
        "rescued_n": int(rescued.sum()),
        "hurt_n": int(hurt.sum()),
        "rescued_fn_n": int((rescued & (y == 1) & (cand_pred == 0)).sum()),
        "rescued_fp_n": int((rescued & (y == 0) & (cand_pred == 1)).sum()),
        "hurt_fn_n": int((hurt & (y == 1) & (final_pred == 0)).sum()),
        "hurt_fp_n": int((hurt & (y == 0) & (final_pred == 1)).sum()),
    }
    row.update({f"stage1_{k}": v for k, v in metrics(y[stage1], cand_pred[stage1], cand_prob[stage1]).items()})
    row.update({f"stage2_{k}": v for k, v in metrics(y[stage2], final_pred[stage2], final_prob[stage2]).items()})
    row.update({f"automated_{k}": v for k, v in metrics(y[automated], final_pred[automated], final_prob[automated]).items()})
    row.update({f"system_if_reject_corrected_{k}": v for k, v in metrics(y, system_pred, final_prob).items()})
    return case, row


def evaluate_config(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    base_name: str,
    base_obj: str,
    target_accept_acc: float,
    gate_name: str,
    gate_model: object | None,
    corr_name: str,
    corr_model: object,
    corr_conf_min: float,
    include_quality: bool,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object], pd.DataFrame]:
    y = internal["label_idx"].astype(int).to_numpy()
    folds = internal["fold_id"].astype(int).to_numpy()
    base_prob = internal[base_name].to_numpy(float)
    base_prob_ext = external[base_name].to_numpy(float)

    base_pred = np.zeros(len(internal), dtype=int)
    accept_score = np.zeros(len(internal), dtype=float)
    corr_prob = np.zeros(len(internal), dtype=float)
    accept_threshold_by_case = np.zeros(len(internal), dtype=float)
    fold_rows: list[dict[str, object]] = []

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        base_t, base_sel = choose_base_threshold(y[train], base_prob[train], base_obj)
        train_base_pred = (base_prob[train] >= base_t).astype(int)
        test_base_pred = (base_prob[test] >= base_t).astype(int)
        base_pred[test] = test_base_pred

        x_train_df = make_feature_frame(internal.loc[train].reset_index(drop=True), base_prob[train], train_base_pred, include_quality)
        x_test_df = make_feature_frame(internal.loc[test].reset_index(drop=True), base_prob[test], test_base_pred, include_quality)
        x_train = x_train_df.to_numpy(float)
        x_test = x_test_df.reindex(columns=x_train_df.columns, fill_value=0.0).to_numpy(float)
        train_correct = train_base_pred == y[train]

        if gate_name == "base_conf":
            train_score = x_train_df["cand_conf"].to_numpy(float)
            test_score = x_test_df["cand_conf"].to_numpy(float)
        else:
            train_score = fit_prob(gate_model, x_train, train_correct.astype(int), x_train)
            test_score = fit_prob(gate_model, x_train, train_correct.astype(int), x_test)
        accept_choice = select_accept_threshold(train_score, train_correct, target_accept_acc)
        accept_score[test] = test_score
        accept_threshold_by_case[test] = float(accept_choice["threshold"])

        train_accept = train_score >= float(accept_choice["threshold"])
        train_indices = np.where(train)[0]
        corr_train_local = ~train_accept
        if corr_train_local.sum() < 40 or len(np.unique(y[train][corr_train_local])) < 2:
            corr_train_local = np.ones(train.sum(), dtype=bool)
        corr_train_idx = train_indices[corr_train_local]
        x_corr_train = make_feature_frame(
            internal.iloc[corr_train_idx].reset_index(drop=True),
            base_prob[corr_train_idx],
            (base_prob[corr_train_idx] >= base_t).astype(int),
            include_quality,
        ).reindex(columns=x_train_df.columns, fill_value=0.0).to_numpy(float)
        corr_prob[test] = fit_prob(corr_model, x_corr_train, y[corr_train_idx], x_test)
        fold_rows.append({"fold_id": int(fold), "base_threshold": base_t, "base_selection_score": base_sel, **accept_choice})

    int_cases, int_row = finalize_cases(internal, base_prob, base_pred, accept_score, accept_threshold_by_case, corr_prob, corr_conf_min)

    y_ext = external["label_idx"].astype(int).to_numpy()
    full_base_t, full_base_sel = choose_base_threshold(y, base_prob, base_obj)
    full_base_pred = (base_prob >= full_base_t).astype(int)
    ext_base_pred = (base_prob_ext >= full_base_t).astype(int)
    x_full_df = make_feature_frame(internal, base_prob, full_base_pred, include_quality)
    x_ext_df = make_feature_frame(external, base_prob_ext, ext_base_pred, include_quality).reindex(columns=x_full_df.columns, fill_value=0.0)
    full_correct = full_base_pred == y
    if gate_name == "base_conf":
        full_score = x_full_df["cand_conf"].to_numpy(float)
        ext_score = x_ext_df["cand_conf"].to_numpy(float)
    else:
        full_score = fit_prob(gate_model, x_full_df.to_numpy(float), full_correct.astype(int), x_full_df.to_numpy(float))
        ext_score = fit_prob(gate_model, x_full_df.to_numpy(float), full_correct.astype(int), x_ext_df.to_numpy(float))
    ext_accept_choice = select_accept_threshold(full_score, full_correct, target_accept_acc)
    full_accept = full_score >= float(ext_accept_choice["threshold"])
    corr_train_local = ~full_accept
    if corr_train_local.sum() < 40 or len(np.unique(y[corr_train_local])) < 2:
        corr_train_local = np.ones(len(y), dtype=bool)
    ext_corr_prob = fit_prob(corr_model, x_full_df.loc[corr_train_local].to_numpy(float), y[corr_train_local], x_ext_df.to_numpy(float))
    ext_cases, ext_row = finalize_cases(
        external,
        base_prob_ext,
        ext_base_pred,
        ext_score,
        float(ext_accept_choice["threshold"]),
        ext_corr_prob,
        corr_conf_min,
    )

    meta = {
        "base_candidate": base_name,
        "base_objective": base_obj,
        "target_accept_acc": target_accept_acc,
        "gate": gate_name,
        "corrector": corr_name,
        "corrector_conf_min": corr_conf_min,
        "include_quality": include_quality,
    }
    int_row.update(meta)
    int_row["scope"] = "internal_nested_old_third"
    int_row["fold_thresholds_json"] = json.dumps(fold_rows, ensure_ascii=False)
    ext_row.update(meta)
    ext_row["scope"] = "strict_external_locked"
    ext_row["external_base_threshold"] = float(full_base_t)
    ext_row["external_base_selection_score"] = float(full_base_sel)
    int_cases = int_cases.assign(**meta, scope="internal_nested_old_third")
    ext_cases = ext_cases.assign(**meta, scope="strict_external_locked")
    return int_row, int_cases, ext_row, ext_cases


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("rate") or col in {
            "base_accuracy",
            "base_balanced_accuracy",
            "base_auc",
            "stage1_accuracy",
            "stage1_balanced_accuracy",
            "stage2_accuracy",
            "stage2_balanced_accuracy",
            "automated_accuracy",
            "automated_balanced_accuracy",
            "system_if_reject_corrected_accuracy",
            "system_if_reject_corrected_balanced_accuracy",
        }:
            out[col] = out[col].map(lambda v: pct(float(v)) if pd.notna(v) else "")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal, external = build_internal_external()
    external = external.loc[~external["task_l6_label"].astype(str).eq("MNT_assumed_low")].copy()

    internal_probs = load_candidate_probs("internal_oof_old_third")
    external_probs = load_candidate_probs("strict_external_locked")
    internal = internal.merge(internal_probs, on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))
    external = external.merge(external_probs, on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))
    for df in [internal, external]:
        for col in BASE_CANDIDATES:
            v135_col = f"{col}_v135"
            if col not in df.columns and v135_col in df.columns:
                df[col] = df[v135_col]
    keep_candidates = [c for c in BASE_CANDIDATES if c in internal.columns and c in external.columns]
    for c in keep_candidates:
        internal[c] = pd.to_numeric(internal[c], errors="coerce").fillna(internal["prob_mean_core"])
        external[c] = pd.to_numeric(external[c], errors="coerce").fillna(external["prob_mean_core"])

    model_bank = make_models(20260527)
    gate_bank: dict[str, object | None] = {"base_conf": None, "gate_logreg": model_bank["logreg_c03"]}
    corr_bank = {"corr_logreg": model_bank["logreg_c03"], "corr_extra": model_bank["extra_d3_l8"]}

    rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    n_configs = 0
    for base_name in keep_candidates:
        for base_obj in OBJECTIVES:
            for target in TARGET_ACCEPT_ACCS:
                for gate_name, gate_model in gate_bank.items():
                    for corr_name, corr_model in corr_bank.items():
                        for corr_conf_min in CORR_CONF_THRESHOLDS:
                            for include_quality in [False, True]:
                                int_row, int_cases, ext_row, ext_cases = evaluate_config(
                                    internal,
                                    external,
                                    base_name,
                                    base_obj,
                                    target,
                                    gate_name,
                                    gate_model,
                                    corr_name,
                                    corr_model,
                                    corr_conf_min,
                                    include_quality,
                                )
                                rows.extend([int_row, ext_row])
                                case_frames.extend([int_cases, ext_cases])
                                n_configs += 1
                                print(
                                    f"[{n_configs}] base={base_name} gate={gate_name} corr={corr_name} "
                                    f"conf={corr_conf_min} quality={include_quality} "
                                    f"int_auto_bacc={int_row['automated_balanced_accuracy']:.4f} "
                                    f"ext_auto_bacc={ext_row['automated_balanced_accuracy']:.4f}",
                                    flush=True,
                                )

    summary = pd.DataFrame(rows)
    sort_cols = ["scope", "system_if_reject_corrected_balanced_accuracy", "automated_balanced_accuracy", "automated_rate"]
    summary = summary.sort_values(sort_cols, ascending=[True, False, False, False])
    summary.to_csv(OUT_DIR / "v136_leakage_safe_cascade_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v136_leakage_safe_cascade_summary_formatted.csv", index=False, encoding="utf-8-sig")
    top = summary.groupby("scope").head(40)
    top.to_csv(OUT_DIR / "v136_top_by_scope.csv", index=False, encoding="utf-8-sig")
    format_table(top).to_csv(OUT_DIR / "v136_top_by_scope_formatted.csv", index=False, encoding="utf-8-sig")
    pd.concat(case_frames, ignore_index=True).to_csv(OUT_DIR / "v136_leakage_safe_cascade_cases.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": "No subtype/task label one-hot features. Base thresholds and gates selected on old+third only; strict external excludes MNT_assumed_low and is locked.",
        "n_internal": int(len(internal)),
        "n_external_strict": int(len(external)),
        "base_candidates": keep_candidates,
        "n_configs": int(n_configs),
    }
    (OUT_DIR / "v136_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_DIR)
    print(format_table(top).to_string(index=False))


if __name__ == "__main__":
    main()
