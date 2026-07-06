from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_no64_guarded_adapt_overlay_20260522 import (  # noqa: E402
    add_route_scores,
    apply_overlay,
    reconstruct_no64_old,
)


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, object] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def align_features(frame: pd.DataFrame, table_path: Path, npy_path: Path) -> np.ndarray:
    table = pd.read_csv(table_path, dtype={"case_id": str})
    arr = np.load(npy_path).astype(np.float32)
    order = frame[["case_id"]].merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        missing = order.loc[order["feature_idx"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing crop features from {table_path}: {missing}")
    return arr[order["feature_idx"].astype(int).to_numpy()]


@dataclass(frozen=True)
class RescueSpec:
    name: str
    kind: str
    c: float = 1.0
    adapt_weight: float = 1.0
    pca_components: int | None = None
    seed: int = 20260522


def make_rescue_model(spec: RescueSpec):
    if spec.kind == "logreg":
        steps = [StandardScaler()]
        if spec.pca_components is not None:
            steps.append(PCA(n_components=spec.pca_components, random_state=spec.seed))
        steps.append(
            LogisticRegression(
                C=spec.c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=spec.seed,
            )
        )
        return make_pipeline(*steps)
    if spec.kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=spec.pca_components,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=spec.seed,
            n_jobs=-1,
        )
    raise ValueError(spec.kind)


def fit_rescue_oof(
    x_low_train: np.ndarray,
    y_low_train: np.ndarray,
    train_case_ids: np.ndarray,
    x_low_hold: np.ndarray,
    specs: list[RescueSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    is_adapt = pd.Series(train_case_ids).astype(str).str.startswith("third_").to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260522)
    train_probs = pd.DataFrame({"case_id": train_case_ids})
    hold_probs = pd.DataFrame()
    rows: list[dict[str, object]] = []
    for spec in specs:
        model_template = make_rescue_model(spec)
        oof = np.zeros(len(y_low_train), dtype=float)
        hold_folds = []
        for tr, va in skf.split(x_low_train, y_low_train):
            model = clone(model_template)
            weights = np.ones(len(tr), dtype=float)
            weights[is_adapt[tr]] *= spec.adapt_weight
            fit_kwargs = {}
            if spec.kind == "logreg":
                fit_kwargs["logisticregression__sample_weight"] = weights
            else:
                fit_kwargs["sample_weight"] = weights
            model.fit(x_low_train[tr], y_low_train[tr], **fit_kwargs)
            oof[va] = model.predict_proba(x_low_train[va])[:, 1]
            hold_folds.append(model.predict_proba(x_low_hold)[:, 1])
        hold_prob = np.mean(np.vstack(hold_folds), axis=0)
        train_probs[spec.name] = oof
        hold_probs[spec.name] = hold_prob
        pred = (oof >= 0.5).astype(int)
        row = {
            "candidate": spec.name,
            "kind": spec.kind,
            "C": spec.c,
            "adapt_weight": spec.adapt_weight,
            "pca_components_or_depth": spec.pca_components,
            "train_low_n": int(len(y_low_train)),
            "train_low_pos": int(y_low_train.sum()),
        }
        row.update(metric_dict(y_low_train, pred, oof))
        row["old_low_accuracy"] = float(accuracy_score(y_low_train[~is_adapt], pred[~is_adapt]))
        row["adapt_low_accuracy"] = float(accuracy_score(y_low_train[is_adapt], pred[is_adapt]))
        row["old_low_pos"] = int(y_low_train[~is_adapt].sum())
        row["adapt_low_pos"] = int(y_low_train[is_adapt].sum())
        rows.append(row)
    return train_probs, hold_probs, pd.DataFrame(rows)


def threshold_from_budget(score: np.ndarray, eligible_mask: np.ndarray, budget_pct: int) -> float:
    eligible = score[eligible_mask & np.isfinite(score)]
    eligible = eligible[eligible > 0]
    if budget_pct <= 0 or len(eligible) == 0:
        return float("inf")
    k = max(1, int(round(len(score) * budget_pct / 100.0)))
    k = min(k, len(eligible))
    return float(np.sort(eligible)[-k])


def apply_rescue(
    y: np.ndarray,
    base_pred: np.ndarray,
    base_prob: np.ndarray,
    low_case_mask: np.ndarray,
    low_rescue_prob: np.ndarray,
    threshold: float,
    route_score: np.ndarray,
    route_threshold: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    full_rescue_prob = np.zeros(len(y), dtype=float)
    full_rescue_prob[low_case_mask] = low_rescue_prob
    score_full = np.zeros(len(y), dtype=float)
    score_full[low_case_mask] = route_score
    if np.isfinite(route_threshold):
        routed = low_case_mask & (full_rescue_prob >= threshold) & (score_full >= route_threshold)
    else:
        routed = np.zeros(len(y), dtype=bool)
    pred = base_pred.copy()
    prob = base_prob.copy()
    pred[routed] = 1
    prob[routed] = full_rescue_prob[routed]
    row = metric_dict(y, pred, prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "rescue_n": int(((base_pred != y) & (pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (pred != y) & routed).sum()),
            "pass_acc": float((pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "routed_acc": float((pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, pred, prob, routed


def main() -> None:
    root = Path(".").resolve()
    base = root / "outputs/batch1_batch2_task567_20260514"
    out = base / "task7_adaptation_runs/29_no64_crop_rescue_corrector_20260522"
    out.mkdir(parents=True, exist_ok=True)

    run64 = base / "task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521"
    adapt_cache = base / "task7_adaptation_runs/11_unified_two_stage_adapt72_20260521"
    third_external = base / "task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521"
    old_crop_table = base / "task7_gross_feature_runs/67_roi_crop_embedding_probe_20260521/case_dino_concat_feature_table.csv"
    old_crop_npy = base / "task7_gross_feature_runs/67_roi_crop_embedding_probe_20260521/case_dino_concat_features.npy"
    third_crop_table = base / "task7_external_runs/03_third_batch_crop_64style_20260521/third_batch_dino_concat_feature_table.csv"
    third_crop_npy = base / "task7_external_runs/03_third_batch_crop_64style_20260521/third_batch_dino_concat_features.npy"

    old_no64 = reconstruct_no64_old(root, run64, None)
    train_probs = pd.read_csv(adapt_cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    hold_probs = pd.read_csv(adapt_cache / "candidate_holdout_probs.csv", dtype={"case_id": str})
    third_all = pd.read_csv(third_external / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})
    old_prob = train_probs[~train_probs["case_id"].str.startswith("third_")].copy()
    adapt_prob = train_probs[train_probs["case_id"].str.startswith("third_")].copy()
    old = old_no64.merge(old_prob, on="case_id", how="inner")
    adapt = third_all.merge(adapt_prob, on="case_id", how="inner")
    hold = third_all.merge(hold_probs, on="case_id", how="inner")
    old["base_pred_for_overlay"] = old["no64_final_pred_idx"].astype(int)
    adapt["base_pred_for_overlay"] = adapt["final_pred_idx"].astype(int)
    hold["base_pred_for_overlay"] = hold["final_pred_idx"].astype(int)

    candidate_cols = [c for c in train_probs.columns if c != "case_id"]
    selected_adapt_name = "adapt_r2_c0.0003"
    selected_adapt_t = 0.54
    selected_route_name = "disagree_base_low_margin"
    selected_route_t = 1.885296885045483

    def selected_base(frame: pd.DataFrame, y: np.ndarray, base_prob_col: str, base_pred_col: str):
        route_score = add_route_scores(frame, candidate_cols, base_prob_col, selected_adapt_name, selected_adapt_t)[selected_route_name]
        return apply_overlay(
            y,
            frame[base_prob_col].to_numpy(float),
            frame[base_pred_col].to_numpy(int),
            frame[selected_adapt_name].to_numpy(float),
            selected_adapt_t,
            route_score,
            selected_route_t,
        )

    y_old = old["label_idx"].to_numpy(int)
    y_adapt = adapt["label_idx"].to_numpy(int)
    y_hold = hold["label_idx"].to_numpy(int)
    old_base_row, old_base_pred, old_base_prob, old_base_routed = selected_base(
        old, y_old, "no64_final_prob_high", "no64_final_pred_idx"
    )
    adapt_base_row, adapt_base_pred, adapt_base_prob, adapt_base_routed = selected_base(
        adapt, y_adapt, "final_prob_high", "final_pred_idx"
    )
    hold_base_row, hold_base_pred, hold_base_prob, hold_base_routed = selected_base(
        hold, y_hold, "final_prob_high", "final_pred_idx"
    )

    old_crop_x = align_features(old, old_crop_table, old_crop_npy)
    adapt_crop_x = align_features(adapt, third_crop_table, third_crop_npy)
    hold_crop_x = align_features(hold, third_crop_table, third_crop_npy)
    train_frame = pd.concat(
        [
            old[["case_id", "label_idx"]].assign(split_name="old"),
            adapt[["case_id", "label_idx"]].assign(split_name="adapt"),
        ],
        ignore_index=True,
    )
    train_x = np.vstack([old_crop_x, adapt_crop_x])
    train_y = train_frame["label_idx"].to_numpy(int)
    train_base_pred = np.concatenate([old_base_pred, adapt_base_pred])
    train_low_mask = train_base_pred == 0
    hold_low_mask = hold_base_pred == 0
    train_low_x = train_x[train_low_mask]
    train_low_y = train_y[train_low_mask]
    train_low_case_ids = train_frame.loc[train_low_mask, "case_id"].astype(str).to_numpy()
    hold_low_x = hold_crop_x[hold_low_mask]

    specs: list[RescueSpec] = []
    for aw in [1.0, 2.0, 4.0, 8.0]:
        for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
            specs.append(RescueSpec(name=f"logreg_c{c:g}_aw{aw:g}", kind="logreg", c=c, adapt_weight=aw))
    for aw in [1.0, 2.0, 4.0]:
        for c in [0.003, 0.01, 0.03, 0.1]:
            for n in [16, 32, 64]:
                specs.append(RescueSpec(name=f"pca{n}_logreg_c{c:g}_aw{aw:g}", kind="logreg", c=c, adapt_weight=aw, pca_components=n))
    for aw in [1.0, 3.0, 6.0]:
        for depth in [2, 3, 4, None]:
            name_depth = "none" if depth is None else str(depth)
            specs.append(RescueSpec(name=f"et_d{name_depth}_aw{aw:g}", kind="extratrees", adapt_weight=aw, pca_components=depth))

    rescue_train_low_probs, rescue_hold_low_probs, rescue_summary = fit_rescue_oof(
        train_low_x,
        train_low_y,
        train_low_case_ids,
        hold_low_x,
        specs,
    )
    rescue_summary.to_csv(out / "crop_rescue_lowcase_oof_summary.csv", index=False, encoding="utf-8-sig")
    rescue_train_low_probs.to_csv(out / "crop_rescue_train_lowcase_oof_probs.csv", index=False, encoding="utf-8-sig")
    rescue_hold_low_probs.insert(0, "case_id", hold.loc[hold_low_mask, "case_id"].astype(str).to_numpy())
    rescue_hold_low_probs.to_csv(out / "crop_rescue_holdout_lowcase_probs.csv", index=False, encoding="utf-8-sig")

    old_low_n = int((old_base_pred == 0).sum())
    adapt_low_n = int((adapt_base_pred == 0).sum())
    old_low_probs = rescue_train_low_probs.iloc[:old_low_n].reset_index(drop=True)
    adapt_low_probs = rescue_train_low_probs.iloc[old_low_n : old_low_n + adapt_low_n].reset_index(drop=True)
    hold_low_probs = rescue_hold_low_probs.reset_index(drop=True)

    rows: list[dict[str, object]] = []
    case_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    budgets = [0, 1, 2, 3, 5, 8, 10, 15, 20]
    sources = ["old", "adapt", "old_adapt"]
    rescue_cols = [c for c in rescue_train_low_probs.columns if c != "case_id"]
    for rescue_name in rescue_cols:
        old_low_p = old_low_probs[rescue_name].to_numpy(float)
        adapt_low_p = adapt_low_probs[rescue_name].to_numpy(float)
        hold_low_p = hold_low_probs[rescue_name].to_numpy(float)
        for t in thresholds:
            old_score = np.maximum(0.0, old_low_p - t)
            adapt_score = np.maximum(0.0, adapt_low_p - t)
            hold_score = np.maximum(0.0, hold_low_p - t)
            for source in sources:
                source_score = {
                    "old": old_score,
                    "adapt": adapt_score,
                    "old_adapt": np.concatenate([old_score, adapt_score]),
                }[source]
                source_mask = np.ones(len(source_score), dtype=bool)
                for budget in budgets:
                    route_t = threshold_from_budget(source_score, source_mask, budget)
                    old_row, old_pred, old_prob, old_rescue = apply_rescue(
                        y_old,
                        old_base_pred,
                        old_base_prob,
                        old_base_pred == 0,
                        old_low_p,
                        t,
                        old_score,
                        route_t,
                    )
                    adapt_row, _, _, _ = apply_rescue(
                        y_adapt,
                        adapt_base_pred,
                        adapt_base_prob,
                        adapt_base_pred == 0,
                        adapt_low_p,
                        t,
                        adapt_score,
                        route_t,
                    )
                    hold_row, hold_pred, hold_prob, hold_rescue = apply_rescue(
                        y_hold,
                        hold_base_pred,
                        hold_base_prob,
                        hold_base_pred == 0,
                        hold_low_p,
                        t,
                        hold_score,
                        route_t,
                    )
                    row = {
                        "rescue_candidate": rescue_name,
                        "rescue_threshold": t,
                        "threshold_source": source,
                        "budget_pct": budget,
                        "route_threshold": route_t,
                    }
                    row.update({f"old_{k}": v for k, v in old_row.items()})
                    row.update({f"adapt_{k}": v for k, v in adapt_row.items()})
                    row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                    row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                    row["adapt_acc_gain"] = float(row["adapt_accuracy"] - adapt_base_row["accuracy"])
                    row["adapt_bacc_gain"] = float(row["adapt_balanced_accuracy"] - adapt_base_row["balanced_accuracy"])
                    row["adapt_tp_gain"] = int(row["adapt_tp"] - adapt_base_row["tp"])
                    row["adapt_fp_increase"] = int(row["adapt_fp"] - adapt_base_row["fp"])
                    row["holdout_acc_gain"] = float(row["holdout_accuracy"] - hold_base_row["accuracy"])
                    row["holdout_bacc_gain"] = float(row["holdout_balanced_accuracy"] - hold_base_row["balanced_accuracy"])
                    row["holdout_tp_preserved_vs_original"] = bool(row["holdout_tp"] >= 25)
                    # Selection uses old + adapt only. Penalize old harm and adapt FP inflation.
                    row["selection_score_safe"] = (
                        float(row["adapt_accuracy"])
                        + 0.8 * float(row["adapt_balanced_accuracy"])
                        + 0.05 * float(row["adapt_tp_gain"])
                        - 0.04 * float(row["adapt_fp_increase"])
                        - 0.04 * float(row["old_hurt_n"])
                        + 0.02 * float(row["old_rescue_n"])
                    )
                    rows.append(row)
                    case_cache[len(rows) - 1] = (hold_pred, hold_prob, hold_rescue, hold_low_p)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "crop_rescue_overlay_all_policies.csv", index=False, encoding="utf-8-sig")
    guard = summary[summary["old_guard_092"]].copy()
    safe = guard[
        (guard["adapt_acc_gain"] >= 0)
        & (guard["adapt_bacc_gain"] >= 0)
        & (guard["adapt_tp_gain"] >= 0)
    ].copy()
    selected_df = safe.sort_values(
        ["selection_score_safe", "adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"],
        ascending=False,
    )
    if selected_df.empty:
        selected_df = guard.sort_values(["adapt_accuracy", "adapt_balanced_accuracy"], ascending=False)
    hold_ref = guard.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    hold_tp_ref = guard[guard["holdout_tp_preserved_vs_original"]].sort_values(
        ["holdout_accuracy", "holdout_balanced_accuracy"],
        ascending=False,
    )
    selected_df.head(100).to_csv(out / "top_selected_by_old_plus_adapt_only.csv", index=False, encoding="utf-8-sig")
    hold_ref.head(100).to_csv(out / "top_holdout_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")
    hold_tp_ref.head(100).to_csv(out / "top_holdout_tp_preserved_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")

    def one(df: pd.DataFrame) -> dict[str, object] | None:
        return None if df.empty else df.iloc[0].to_dict()

    selected = one(selected_df)
    best_hold = one(hold_ref)
    best_hold_tp = one(hold_tp_ref)

    def save_holdout(prefix: str, row: dict[str, object] | None) -> None:
        if row is None:
            return
        match = summary[
            (summary["rescue_candidate"] == row["rescue_candidate"])
            & (summary["rescue_threshold"] == row["rescue_threshold"])
            & (summary["threshold_source"] == row["threshold_source"])
            & (summary["budget_pct"] == row["budget_pct"])
        ]
        idx = int(match.index[0])
        pred, prob, rescue_mask, hold_low_p = case_cache[idx]
        full_rescue_prob = np.full(len(hold), np.nan, dtype=float)
        full_rescue_prob[hold_base_pred == 0] = hold_low_p
        case = hold[
            [
                "case_id",
                "original_case_id",
                "source_folder",
                "task_l6_label",
                "task_l7_label",
                "label_idx",
                "image_name",
            ]
        ].copy()
        case["selected_base_pred_idx"] = hold_base_pred
        case["selected_base_prob_high"] = hold_base_prob
        case["crop_rescue_prob_high"] = full_rescue_prob
        case["crop_rescue_routed"] = rescue_mask.astype(int)
        case["final_pred_idx"] = pred
        case["final_prob_high"] = prob
        case["final_correct"] = (pred == y_hold).astype(int)
        case.to_csv(out / f"{prefix}_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_holdout("selected_by_old_plus_adapt", selected)
    save_holdout("best_holdout_reference", best_hold)
    save_holdout("best_holdout_tp_preserved_reference", best_hold_tp)

    comp_rows = [
        {
            "name": "selected_base_no64_adapt_overlay",
            "old_accuracy": old_base_row["accuracy"],
            "old_balanced_accuracy": old_base_row["balanced_accuracy"],
            "adapt_accuracy": adapt_base_row["accuracy"],
            "adapt_balanced_accuracy": adapt_base_row["balanced_accuracy"],
            "adapt_tn": adapt_base_row["tn"],
            "adapt_fp": adapt_base_row["fp"],
            "adapt_fn": adapt_base_row["fn"],
            "adapt_tp": adapt_base_row["tp"],
            "holdout_accuracy": hold_base_row["accuracy"],
            "holdout_balanced_accuracy": hold_base_row["balanced_accuracy"],
            "holdout_tn": hold_base_row["tn"],
            "holdout_fp": hold_base_row["fp"],
            "holdout_fn": hold_base_row["fn"],
            "holdout_tp": hold_base_row["tp"],
            "policy": "run25 selected No.64 guarded adapt overlay",
        }
    ]
    for name, row in [
        ("selected_crop_rescue_by_old_plus_adapt", selected),
        ("best_holdout_reference", best_hold),
        ("best_holdout_tp_preserved_reference", best_hold_tp),
    ]:
        if row is None:
            continue
        comp_rows.append(
            {
                "name": name,
                "old_accuracy": row["old_accuracy"],
                "old_balanced_accuracy": row["old_balanced_accuracy"],
                "adapt_accuracy": row["adapt_accuracy"],
                "adapt_balanced_accuracy": row["adapt_balanced_accuracy"],
                "adapt_tn": row["adapt_tn"],
                "adapt_fp": row["adapt_fp"],
                "adapt_fn": row["adapt_fn"],
                "adapt_tp": row["adapt_tp"],
                "holdout_accuracy": row["holdout_accuracy"],
                "holdout_balanced_accuracy": row["holdout_balanced_accuracy"],
                "holdout_tn": row["holdout_tn"],
                "holdout_fp": row["holdout_fp"],
                "holdout_fn": row["holdout_fn"],
                "holdout_tp": row["holdout_tp"],
                "policy": f"{row['rescue_candidate']} t={row['rescue_threshold']} source={row['threshold_source']} budget={row['budget_pct']}",
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "crop_rescue_overlay_key_comparison.csv", index=False, encoding="utf-8-sig")

    report = {
        "protocol": {
            "selection_uses_holdout": False,
            "selection_data": "old OOF + third adapt72 only",
            "holdout_data": "third adapt72 holdout 234 cases",
            "base_policy": "run25 selected No.64 guarded adapt overlay",
            "added_step": "crop-only corrector trained only for selected-base low-risk cases",
        },
        "lowcase_counts": {
            "old_low_n": int((old_base_pred == 0).sum()),
            "old_low_pos": int(y_old[old_base_pred == 0].sum()),
            "adapt_low_n": int((adapt_base_pred == 0).sum()),
            "adapt_low_pos": int(y_adapt[adapt_base_pred == 0].sum()),
            "holdout_low_n": int((hold_base_pred == 0).sum()),
            "holdout_low_pos": int(y_hold[hold_base_pred == 0].sum()),
        },
        "selected_base": {
            "old": old_base_row,
            "adapt": adapt_base_row,
            "holdout": hold_base_row,
        },
        "selected_crop_rescue_by_old_plus_adapt": selected,
        "best_holdout_reference_under_old_guard92": best_hold,
        "best_holdout_tp_preserved_reference_under_old_guard92": best_hold_tp,
        "n_rescue_candidates": int(len(rescue_cols)),
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard)),
        "n_safe_adapt": int(len(safe)),
        "output_dir": str(out),
    }
    (out / "crop_rescue_overlay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
