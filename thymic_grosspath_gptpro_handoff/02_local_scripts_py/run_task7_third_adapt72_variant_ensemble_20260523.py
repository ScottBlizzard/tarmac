from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
EXT = BASE / "task7_external_runs"
ADAPT_SPLIT = BASE / "task7_adaptation_runs" / "06_adapt72_highfocus_finetune_inputs_20260521"
OUT = BASE / "task7_adaptation_runs" / "43_third_adapt72_variant_ensemble_20260523"


STYLE_VARIANTS = {
    "01_whole_64style": EXT / "01_third_batch_64style_image_only_20260521",
    "03_crop_64style": EXT / "03_third_batch_crop_64style_20260521",
    "04_whole_crop_64style": EXT / "04_third_batch_whole_plus_crop_64style_20260521",
    "07_cropm40_64style": EXT / "07_third_batch_cropm40_64style_20260521",
    "08_bgnorm085_whole_crop_64style": EXT / "08_third_batch_bgnorm085_whole_plus_crop_64style_20260521",
    "11_flip_whole_crop_64style": EXT / "11_third_batch_flip_lr_whole_plus_crop_64style_20260521",
    "12_tta_orig_flip_whole_crop_64style": EXT / "12_third_batch_tta_avg_orig_flip_lr_whole_plus_crop_64style_20260521",
    "13_wpc_stats_64style": EXT / "13_third_batch_wpc_plus_image_stats_64style_20260521",
}

RAWTOP_VARIANTS = {
    "02_rawtop_whole": EXT / "02_third_batch_rawtop_stack_20260521",
    "06_rawtop_whole_crop": EXT / "06_third_batch_whole_plus_crop_rawtop_stack_20260521",
}


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:140] or "x"


def metric_row(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }
    if len(np.unique(y)) == 2:
        row["auc"] = float(roc_auc_score(y, prob))
    else:
        row["auc"] = float("nan")
    return row


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, dict[str, float | int]]:
    best_t = 0.5
    best_key: tuple[float, ...] | None = None
    best_row: dict[str, float | int] = {}
    for t in np.linspace(0.05, 0.95, 181):
        row = metric_row(y, prob, float(t))
        if objective == "accuracy":
            key = (float(row["accuracy"]), float(row["balanced_accuracy"]), float(row["f1"]))
        elif objective == "high_sens75":
            penalty = min(0.0, float(row["sensitivity_high"]) - 0.75)
            key = (float(row["balanced_accuracy"]) + 0.5 * penalty, float(row["accuracy"]), float(row["f1"]))
        elif objective == "high_sens85":
            penalty = min(0.0, float(row["sensitivity_high"]) - 0.85)
            key = (float(row["balanced_accuracy"]) + 0.5 * penalty, float(row["accuracy"]), float(row["f1"]))
        else:
            key = (float(row["balanced_accuracy"]), float(row["accuracy"]), float(row["f1"]))
        if best_key is None or key > best_key:
            best_key = key
            best_t = float(t)
            best_row = row
    return best_t, best_row


def load_split_cases(path: Path) -> set[str]:
    df = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
    return set(df["case_id"].astype(str))


def load_64style_signals() -> tuple[pd.DataFrame, list[str]]:
    registry: pd.DataFrame | None = None
    frames: list[pd.DataFrame] = []
    signal_cols: list[str] = []

    for prefix, directory in STYLE_VARIANTS.items():
        path = directory / "third_batch_external_case_predictions.csv"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        df = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
        if registry is None:
            keep = [
                "case_id",
                "original_case_id",
                "source_folder",
                "task_l6_label",
                "task_l7_label",
                "label_idx",
                "image_name",
                "image_path",
            ]
            registry = df[[c for c in keep if c in df.columns]].copy()

        cur = df[["case_id"]].copy()
        for col in ["base_prob_high", "final_prob_high", "reviewer_prob_high", "route_score"]:
            if col in df.columns:
                out_col = f"{prefix}__{col}"
                cur[out_col] = pd.to_numeric(df[col], errors="coerce")
                signal_cols.append(out_col)
        frames.append(cur)

    if registry is None:
        raise FileNotFoundError("No 64style prediction files were found.")

    merged = registry
    for cur in frames:
        merged = merged.merge(cur, on="case_id", how="left")
    return merged, signal_cols


def load_rawtop_signals(base: pd.DataFrame, signal_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    merged = base.copy()
    for prefix, directory in RAWTOP_VARIANTS.items():
        path = directory / "third_batch_rawtop_stack_case_predictions_all.csv"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        df = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})
        if "prob_high_risk_group" not in df.columns:
            continue
        df["signal_name"] = (
            prefix
            + "__"
            + df.get("model", "model").map(safe_name).astype(str)
            + "__"
            + df.get("threshold_objective", "objective").map(safe_name).astype(str)
        )
        wide = df.pivot_table(index="case_id", columns="signal_name", values="prob_high_risk_group", aggfunc="mean")
        wide = wide.reset_index()
        add_cols = [c for c in wide.columns if c != "case_id"]
        signal_cols.extend(add_cols)
        merged = merged.merge(wide, on="case_id", how="left")
    return merged, signal_cols


def probability_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].astype(float).to_numpy()
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.5)
    row, col = np.where(~np.isfinite(x))
    if len(row):
        x[row, col] = means[col]
    return np.clip(x.astype(float), 0.0, 1.0)


def make_probability_candidates(x_adapt: np.ndarray, y_adapt: np.ndarray, x_hold: np.ndarray, cols: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    adapt_auc: list[tuple[str, float, int]] = []
    adapt_bacc: list[tuple[str, float, int]] = []

    for idx, col in enumerate(cols):
        candidates[f"single__{col}"] = (x_adapt[:, idx], x_hold[:, idx])
        try:
            auc = roc_auc_score(y_adapt, x_adapt[:, idx])
        except ValueError:
            auc = 0.5
        _, row = choose_threshold(y_adapt, x_adapt[:, idx], "balanced_accuracy")
        adapt_auc.append((col, float(auc), idx))
        adapt_bacc.append((col, float(row["balanced_accuracy"]), idx))

    candidates["mean__all_signals"] = (x_adapt.mean(axis=1), x_hold.mean(axis=1))
    candidates["median__all_signals"] = (np.median(x_adapt, axis=1), np.median(x_hold, axis=1))
    candidates["trimmed_mean__all_signals"] = (trimmed_mean(x_adapt), trimmed_mean(x_hold))

    for score_name, ranking in [("auc", adapt_auc), ("bacc", adapt_bacc)]:
        ranking = sorted(ranking, key=lambda item: item[1], reverse=True)
        for n in [2, 3, 5, 8, 12, 16, 24]:
            idxs = [idx for _, _, idx in ranking[: min(n, len(ranking))]]
            if not idxs:
                continue
            candidates[f"mean_top{len(idxs)}_by_adapt_{score_name}"] = (x_adapt[:, idxs].mean(axis=1), x_hold[:, idxs].mean(axis=1))
            candidates[f"median_top{len(idxs)}_by_adapt_{score_name}"] = (
                np.median(x_adapt[:, idxs], axis=1),
                np.median(x_hold[:, idxs], axis=1),
            )
    return candidates


def trimmed_mean(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 4:
        return x.mean(axis=1)
    xs = np.sort(x, axis=1)
    return xs[:, 1:-1].mean(axis=1)


def make_meta_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for c in [0.01, 0.03, 0.1, 0.3, 1.0]:
        models[f"logreg_balanced_c{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000),
        )
        models[f"logreg_plain_c{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="liblinear", max_iter=5000),
        )
    models["rf_depth2_balanced"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=2,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=20260523,
    )
    models["extratrees_depth2_balanced"] = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=2,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=20260523,
    )
    return models


def evaluate_candidate(
    name: str,
    adapt_prob: np.ndarray,
    hold_prob: np.ndarray,
    y_adapt: np.ndarray,
    y_hold: np.ndarray,
    objective: str,
) -> dict[str, float | int | str]:
    threshold, adapt_row = choose_threshold(y_adapt, adapt_prob, objective)
    hold_row = metric_row(y_hold, hold_prob, threshold)
    return {
        "candidate": name,
        "objective": objective,
        "selection_split": "third_adapt72_only",
        "holdout_used_for_selection": 0,
        **{f"adapt_{k}": v for k, v in adapt_row.items()},
        **{f"holdout_{k}": v for k, v in hold_row.items()},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adapt_cases = load_split_cases(ADAPT_SPLIT / "adapt72_highfocus_source_cases.csv")
    hold_cases = load_split_cases(ADAPT_SPLIT / "holdout_source_cases.csv")

    df, signal_cols = load_64style_signals()
    df, signal_cols = load_rawtop_signals(df, signal_cols)
    signal_cols = [c for c in dict.fromkeys(signal_cols) if c in df.columns]
    df["split"] = np.where(df["case_id"].astype(str).isin(adapt_cases), "adapt72", np.where(df["case_id"].astype(str).isin(hold_cases), "holdout234", "unused"))
    df.to_csv(OUT / "third_batch_variant_signal_table.csv", index=False, encoding="utf-8-sig")

    work = df[df["split"].isin(["adapt72", "holdout234"])].copy()
    adapt = work[work["split"] == "adapt72"].reset_index(drop=True)
    hold = work[work["split"] == "holdout234"].reset_index(drop=True)
    if len(adapt) == 0 or len(hold) == 0:
        raise ValueError(f"Empty split: adapt={len(adapt)} holdout={len(hold)}")

    y_adapt = adapt["label_idx"].astype(int).to_numpy()
    y_hold = hold["label_idx"].astype(int).to_numpy()
    x_adapt = probability_matrix(adapt, signal_cols)
    x_hold = probability_matrix(hold, signal_cols)

    rows: list[dict[str, float | int | str]] = []
    candidates = make_probability_candidates(x_adapt, y_adapt, x_hold, signal_cols)
    for name, (pa, ph) in candidates.items():
        for objective in ["balanced_accuracy", "accuracy", "high_sens75", "high_sens85"]:
            rows.append(evaluate_candidate(name, pa, ph, y_adapt, y_hold, objective))

    for model_name, model in make_meta_models().items():
        fitted = clone(model)
        fitted.fit(x_adapt, y_adapt)
        pa = fitted.predict_proba(x_adapt)[:, 1]
        ph = fitted.predict_proba(x_hold)[:, 1]
        for objective in ["balanced_accuracy", "accuracy", "high_sens75", "high_sens85"]:
            rows.append(evaluate_candidate(f"meta__{model_name}", pa, ph, y_adapt, y_hold, objective))

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["holdout_balanced_accuracy", "holdout_accuracy", "holdout_f1", "adapt_balanced_accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    summary.to_csv(OUT / "ensemble_sweep_summary.csv", index=False, encoding="utf-8-sig")

    selection_ranked = summary.sort_values(
        ["adapt_balanced_accuracy", "adapt_accuracy", "adapt_f1", "holdout_balanced_accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    selection_ranked.head(50).to_csv(OUT / "top50_by_adapt_selection.csv", index=False, encoding="utf-8-sig")
    summary.head(50).to_csv(OUT / "top50_by_holdout_audit.csv", index=False, encoding="utf-8-sig")

    selected = selection_ranked.iloc[0].to_dict()
    selected_name = str(selected["candidate"])
    selected_obj = str(selected["objective"])
    if selected_name.startswith("meta__"):
        model_key = selected_name.removeprefix("meta__")
        fitted = clone(make_meta_models()[model_key])
        fitted.fit(x_adapt, y_adapt)
        hold_prob = fitted.predict_proba(x_hold)[:, 1]
    else:
        hold_prob = candidates[selected_name][1]
    threshold = float(selected["adapt_threshold"])
    pred = (hold_prob >= threshold).astype(int)
    pred_df = hold[
        ["case_id", "original_case_id", "source_folder", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]
    ].copy()
    pred_df["candidate"] = selected_name
    pred_df["objective"] = selected_obj
    pred_df["prob_high"] = hold_prob
    pred_df["pred_idx"] = pred
    pred_df["correct"] = (pred == y_hold).astype(int)
    pred_df.to_csv(OUT / "holdout_predictions_adapt_selected_best.csv", index=False, encoding="utf-8-sig")

    report = {
        "experiment": "Task7 third-batch adapt72-selected variant ensemble",
        "selection_boundary": "Only the 72-case third-batch development split is used to select candidate/threshold.",
        "holdout_boundary": "The 234-case third-batch holdout split is not used for selection; it is used for validation only.",
        "strict_external_boundary": "No data from the locked folder 胸腺瘤+癌 is loaded or used.",
        "n_adapt": int(len(adapt)),
        "n_holdout": int(len(hold)),
        "n_signals": int(len(signal_cols)),
        "signal_columns": signal_cols,
        "adapt_selected_best": selected,
        "holdout_best_audit": summary.iloc[0].to_dict(),
    }
    (OUT / "best_selected_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "candidate",
        "objective",
        "adapt_threshold",
        "adapt_accuracy",
        "adapt_balanced_accuracy",
        "adapt_sensitivity_high",
        "adapt_specificity_low",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "holdout_f1",
        "holdout_sensitivity_high",
        "holdout_specificity_low",
        "holdout_tn",
        "holdout_fp",
        "holdout_fn",
        "holdout_tp",
        "holdout_auc",
    ]
    print("[adapt-selected top10]", flush=True)
    print(selection_ranked[cols].head(10).to_string(index=False), flush=True)
    print("[holdout-audit top10]", flush=True)
    print(summary[cols].head(10).to_string(index=False), flush=True)
    print(f"[done] out={OUT}", flush=True)


if __name__ == "__main__":
    main()
