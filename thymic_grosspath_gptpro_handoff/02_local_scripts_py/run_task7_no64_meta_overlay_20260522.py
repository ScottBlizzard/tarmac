from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_no64_guarded_adapt_overlay_20260522 import reconstruct_no64_old  # noqa: E402


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


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def build_features(frame: pd.DataFrame, prob_cols: list[str], base_prob_col: str, prefix: str) -> pd.DataFrame:
    probs = frame[prob_cols].astype(float).to_numpy()
    base = frame[base_prob_col].astype(float).to_numpy()
    feat = pd.DataFrame(index=frame.index)
    for col in prob_cols:
        feat[f"p_{col}"] = frame[col].astype(float).to_numpy()
        vals = frame[col].astype(float).to_numpy()
        feat[f"m_{col}"] = np.abs(vals - 0.5)
    feat["base_prob"] = base
    feat["base_margin"] = np.abs(base - 0.5)
    feat["base_entropy"] = entropy(base)
    feat["prob_mean"] = probs.mean(axis=1)
    feat["prob_std"] = probs.std(axis=1)
    feat["prob_min"] = probs.min(axis=1)
    feat["prob_max"] = probs.max(axis=1)
    feat["prob_range"] = feat["prob_max"] - feat["prob_min"]
    feat["vote_sum"] = (probs >= 0.5).sum(axis=1)
    feat["vote_frac"] = feat["vote_sum"] / len(prob_cols)
    feat["vote_disagree"] = ((feat["vote_sum"] > 0) & (feat["vote_sum"] < len(prob_cols))).astype(float)
    feat["base_vs_mean"] = base - feat["prob_mean"]
    feat["base_vs_min"] = base - feat["prob_min"]
    feat["base_vs_max"] = base - feat["prob_max"]
    feat["base_pred"] = (base >= 0.5).astype(float)
    feat["majority_pred"] = (feat["vote_frac"] >= 0.5).astype(float)
    feat["base_majority_disagree"] = (feat["base_pred"] != feat["majority_pred"]).astype(float)
    feat["is_third_adapt"] = frame["case_id"].astype(str).str.startswith("third_").astype(float)
    feat["feature_source"] = prefix
    return feat


@dataclass(frozen=True)
class MetaSpec:
    name: str
    kind: str
    c: float = 1.0
    adapt_weight: float = 1.0
    max_depth: int | None = None
    seed: int = 13


def make_model(spec: MetaSpec):
    if spec.kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=spec.c,
                class_weight="balanced",
                max_iter=3000,
                solver="lbfgs",
                random_state=spec.seed,
            ),
        )
    if spec.kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=spec.max_depth,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=spec.seed,
            n_jobs=-1,
        )
    if spec.kind == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_leaf_nodes=7,
            l2_regularization=spec.c,
            max_iter=160,
            random_state=spec.seed,
        )
    raise ValueError(spec.kind)


def fit_predict_meta(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_hold: pd.DataFrame,
    case_ids: np.ndarray,
    specs: list[MetaSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in x_train.columns if c != "feature_source"]
    x = x_train[feature_cols].astype(float).to_numpy()
    xh = x_hold[feature_cols].astype(float).to_numpy()
    is_adapt = pd.Series(case_ids).astype(str).str.startswith("third_").to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260522)
    train_probs = pd.DataFrame({"case_id": case_ids})
    hold_probs = pd.DataFrame({"case_id": x_hold.index.astype(str).to_numpy()})
    train_rows: list[dict[str, object]] = []

    for spec in specs:
        oof = np.zeros(len(y_train), dtype=float)
        hold_fold = []
        for fold, (tr, va) in enumerate(skf.split(x, y_train), start=1):
            model = make_model(spec)
            weights = np.ones(len(tr), dtype=float)
            weights[is_adapt[tr]] *= spec.adapt_weight
            fit_kwargs = {}
            if spec.kind == "logreg":
                fit_kwargs["logisticregression__sample_weight"] = weights
            else:
                fit_kwargs["sample_weight"] = weights
            model.fit(x[tr], y_train[tr], **fit_kwargs)
            oof[va] = model.predict_proba(x[va])[:, 1]
            hold_fold.append(model.predict_proba(xh)[:, 1])
        hold_prob = np.mean(np.vstack(hold_fold), axis=0)
        train_probs[spec.name] = oof
        hold_probs[spec.name] = hold_prob
        pred = (oof >= 0.5).astype(int)
        row = {"candidate": spec.name, "adapt_weight": spec.adapt_weight, "kind": spec.kind}
        row.update(metric_dict(y_train, pred, oof))
        row["old_accuracy"] = float(accuracy_score(y_train[~is_adapt], pred[~is_adapt]))
        row["adapt_accuracy"] = float(accuracy_score(y_train[is_adapt], pred[is_adapt]))
        train_rows.append(row)
    return train_probs, hold_probs, pd.DataFrame(train_rows)


def threshold_from_budget(score: np.ndarray, mask: np.ndarray, budget_pct: int) -> float:
    eligible = score[mask & np.isfinite(score)]
    eligible = eligible[eligible > 0]
    if budget_pct <= 0 or len(eligible) == 0:
        return float("inf")
    k = max(1, int(round(len(score) * budget_pct / 100.0)))
    k = min(k, len(eligible))
    return float(np.sort(eligible)[-k])


def apply_dual_overlay(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    meta_prob: np.ndarray,
    low_to_high_threshold: float,
    high_to_low_threshold: float,
    low_to_high_score: np.ndarray,
    high_to_low_score: np.ndarray,
    low_to_high_route_t: float,
    high_to_low_route_t: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta_pred_low_to_high = meta_prob >= low_to_high_threshold
    meta_pred_high_to_low = meta_prob < high_to_low_threshold
    low_to_high = (base_pred == 0) & meta_pred_low_to_high & (low_to_high_score >= low_to_high_route_t)
    high_to_low = (base_pred == 1) & meta_pred_high_to_low & (high_to_low_score >= high_to_low_route_t)
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    final_prob[low_to_high] = meta_prob[low_to_high]
    final_prob[high_to_low] = meta_prob[high_to_low]
    final_pred[low_to_high] = 1
    final_pred[high_to_low] = 0
    routed = low_to_high | high_to_low
    row = metric_dict(y, final_pred, final_prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "low_to_high_n": int(low_to_high.sum()),
            "high_to_low_n": int(high_to_low.sum()),
            "pass_n": int((~routed).sum()),
            "pass_acc": float((final_pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "routed_acc": float((final_pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "rescue_n": int(((base_pred != y) & (final_pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, final_pred, final_prob, low_to_high, high_to_low


def main() -> None:
    root = Path(".").resolve()
    run64 = root / "outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521"
    adapt_cache = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521"
    third_external = root / "outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521"
    out = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/26_no64_meta_overlay_20260522"
    out.mkdir(parents=True, exist_ok=True)

    old_no64 = reconstruct_no64_old(root, run64, None)
    train_probs_all = pd.read_csv(adapt_cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    hold_probs_base = pd.read_csv(adapt_cache / "candidate_holdout_probs.csv", dtype={"case_id": str})
    third_all = pd.read_csv(third_external / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})
    prob_cols = [c for c in train_probs_all.columns if c != "case_id"]

    old_prob_rows = train_probs_all[~train_probs_all["case_id"].str.startswith("third_")].copy()
    adapt_prob_rows = train_probs_all[train_probs_all["case_id"].str.startswith("third_")].copy()
    old = old_no64.merge(old_prob_rows, on="case_id", how="inner")
    old["base_prob"] = old["no64_final_prob_high"].astype(float)
    old["base_pred"] = old["no64_final_pred_idx"].astype(int)
    adapt = third_all.merge(adapt_prob_rows, on="case_id", how="inner")
    adapt["base_prob"] = adapt["final_prob_high"].astype(float)
    adapt["base_pred"] = adapt["final_pred_idx"].astype(int)
    hold = third_all.merge(hold_probs_base, on="case_id", how="inner")
    hold["base_prob"] = hold["final_prob_high"].astype(float)
    hold["base_pred"] = hold["final_pred_idx"].astype(int)

    train = pd.concat(
        [
            old[["case_id", "label_idx", "base_prob", "base_pred", *prob_cols]],
            adapt[["case_id", "label_idx", "base_prob", "base_pred", *prob_cols]],
        ],
        ignore_index=True,
    )
    hold_eval = hold[["case_id", "label_idx", "base_prob", "base_pred", *prob_cols]].copy()
    x_train = build_features(train, prob_cols, "base_prob", "train")
    x_hold = build_features(hold_eval, prob_cols, "base_prob", "hold")
    x_hold.index = hold_eval["case_id"].astype(str)
    y_train = train["label_idx"].to_numpy(int)

    specs: list[MetaSpec] = []
    for aw in [1.0, 2.0, 4.0, 8.0]:
        for c in [0.03, 0.1, 0.3, 1.0]:
            specs.append(MetaSpec(name=f"logreg_c{c:g}_aw{aw:g}", kind="logreg", c=c, adapt_weight=aw))
    for aw in [1.0, 3.0, 6.0]:
        for depth in [2, 3, 4, None]:
            depth_name = "none" if depth is None else str(depth)
            specs.append(MetaSpec(name=f"et_d{depth_name}_aw{aw:g}", kind="extratrees", adapt_weight=aw, max_depth=depth))
    for aw in [1.0, 3.0, 6.0]:
        for c in [0.01, 0.1, 1.0]:
            specs.append(MetaSpec(name=f"hgb_l2{c:g}_aw{aw:g}", kind="histgb", c=c, adapt_weight=aw))

    meta_train_probs, meta_hold_probs, candidate_summary = fit_predict_meta(
        x_train,
        y_train,
        x_hold,
        train["case_id"].astype(str).to_numpy(),
        specs,
    )
    candidate_summary.to_csv(out / "meta_candidate_oof_summary.csv", index=False, encoding="utf-8-sig")
    meta_train_probs.to_csv(out / "meta_train_oof_probs.csv", index=False, encoding="utf-8-sig")
    meta_hold_probs.to_csv(out / "meta_holdout_probs.csv", index=False, encoding="utf-8-sig")

    old_train = train[~train["case_id"].str.startswith("third_")].reset_index(drop=True)
    adapt_train = train[train["case_id"].str.startswith("third_")].reset_index(drop=True)
    old_meta = meta_train_probs[~meta_train_probs["case_id"].str.startswith("third_")].reset_index(drop=True)
    adapt_meta = meta_train_probs[meta_train_probs["case_id"].str.startswith("third_")].reset_index(drop=True)

    y_old = old_train["label_idx"].to_numpy(int)
    y_adapt = adapt_train["label_idx"].to_numpy(int)
    y_hold = hold_eval["label_idx"].to_numpy(int)
    old_base_prob = old_train["base_prob"].to_numpy(float)
    adapt_base_prob = adapt_train["base_prob"].to_numpy(float)
    hold_base_prob = hold_eval["base_prob"].to_numpy(float)
    old_base_pred = old_train["base_pred"].to_numpy(int)
    adapt_base_pred = adapt_train["base_pred"].to_numpy(int)
    hold_base_pred = hold_eval["base_pred"].to_numpy(int)
    old_base = metric_dict(y_old, old_base_pred, old_base_prob)
    adapt_base = metric_dict(y_adapt, adapt_base_pred, adapt_base_prob)
    hold_base = metric_dict(y_hold, hold_base_pred, hold_base_prob)

    rows: list[dict[str, object]] = []
    case_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    meta_names = [c for c in meta_train_probs.columns if c != "case_id"]
    low_to_high_thresholds = [0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
    high_to_low_thresholds = [0.40, 0.43, 0.45, 0.47, 0.50]
    budgets = [0, 1, 2, 3, 5, 8, 10, 15, 20]
    threshold_sources = ["old", "adapt", "old_adapt"]
    route_modes = ["confidence", "margin_weighted", "candidate_only"]

    for meta_name in meta_names:
        old_p = old_meta[meta_name].to_numpy(float)
        adapt_p = adapt_meta[meta_name].to_numpy(float)
        hold_p = meta_hold_probs[meta_name].to_numpy(float)
        old_base_margin = 1.0 - np.minimum(np.abs(old_base_prob - 0.5) / 0.5, 1.0)
        adapt_base_margin = 1.0 - np.minimum(np.abs(adapt_base_prob - 0.5) / 0.5, 1.0)
        hold_base_margin = 1.0 - np.minimum(np.abs(hold_base_prob - 0.5) / 0.5, 1.0)
        for th_hi in low_to_high_thresholds:
            for th_lo in high_to_low_thresholds:
                if th_lo > th_hi:
                    continue
                for route_mode in route_modes:
                    if route_mode == "confidence":
                        old_lh_score = np.maximum(0.0, old_p - th_hi)
                        adapt_lh_score = np.maximum(0.0, adapt_p - th_hi)
                        hold_lh_score = np.maximum(0.0, hold_p - th_hi)
                        old_hl_score = np.maximum(0.0, th_lo - old_p)
                        adapt_hl_score = np.maximum(0.0, th_lo - adapt_p)
                        hold_hl_score = np.maximum(0.0, th_lo - hold_p)
                    elif route_mode == "margin_weighted":
                        old_lh_score = np.maximum(0.0, old_p - th_hi) * (1.0 + old_base_margin)
                        adapt_lh_score = np.maximum(0.0, adapt_p - th_hi) * (1.0 + adapt_base_margin)
                        hold_lh_score = np.maximum(0.0, hold_p - th_hi) * (1.0 + hold_base_margin)
                        old_hl_score = np.maximum(0.0, th_lo - old_p) * (1.0 + old_base_margin)
                        adapt_hl_score = np.maximum(0.0, th_lo - adapt_p) * (1.0 + adapt_base_margin)
                        hold_hl_score = np.maximum(0.0, th_lo - hold_p) * (1.0 + hold_base_margin)
                    else:
                        old_lh_score = old_p.copy()
                        adapt_lh_score = adapt_p.copy()
                        hold_lh_score = hold_p.copy()
                        old_hl_score = 1.0 - old_p
                        adapt_hl_score = 1.0 - adapt_p
                        hold_hl_score = 1.0 - hold_p
                    for source in threshold_sources:
                        lh_source = {
                            "old": old_lh_score,
                            "adapt": adapt_lh_score,
                            "old_adapt": np.concatenate([old_lh_score, adapt_lh_score]),
                        }[source]
                        hl_source = {
                            "old": old_hl_score,
                            "adapt": adapt_hl_score,
                            "old_adapt": np.concatenate([old_hl_score, adapt_hl_score]),
                        }[source]
                        old_lh_mask = old_base_pred == 0
                        adapt_lh_mask = adapt_base_pred == 0
                        old_hl_mask = old_base_pred == 1
                        adapt_hl_mask = adapt_base_pred == 1
                        source_lh_mask = {
                            "old": old_lh_mask,
                            "adapt": adapt_lh_mask,
                            "old_adapt": np.concatenate([old_lh_mask, adapt_lh_mask]),
                        }[source]
                        source_hl_mask = {
                            "old": old_hl_mask,
                            "adapt": adapt_hl_mask,
                            "old_adapt": np.concatenate([old_hl_mask, adapt_hl_mask]),
                        }[source]
                        for lh_budget in budgets:
                            lh_t = threshold_from_budget(lh_source, source_lh_mask, lh_budget)
                            for hl_budget in budgets:
                                hl_t = threshold_from_budget(hl_source, source_hl_mask, hl_budget)
                                old_row, old_pred, old_prob, old_lh, old_hl = apply_dual_overlay(
                                    y_old,
                                    old_base_prob,
                                    old_base_pred,
                                    old_p,
                                    th_hi,
                                    th_lo,
                                    old_lh_score,
                                    old_hl_score,
                                    lh_t,
                                    hl_t,
                                )
                                adapt_row, adapt_pred, adapt_prob, adapt_lh, adapt_hl = apply_dual_overlay(
                                    y_adapt,
                                    adapt_base_prob,
                                    adapt_base_pred,
                                    adapt_p,
                                    th_hi,
                                    th_lo,
                                    adapt_lh_score,
                                    adapt_hl_score,
                                    lh_t,
                                    hl_t,
                                )
                                hold_row, hold_pred, hold_prob, hold_lh, hold_hl = apply_dual_overlay(
                                    y_hold,
                                    hold_base_prob,
                                    hold_base_pred,
                                    hold_p,
                                    th_hi,
                                    th_lo,
                                    hold_lh_score,
                                    hold_hl_score,
                                    lh_t,
                                    hl_t,
                                )
                                row = {
                                    "meta_candidate": meta_name,
                                    "low_to_high_threshold": th_hi,
                                    "high_to_low_threshold": th_lo,
                                    "route_mode": route_mode,
                                    "threshold_source": source,
                                    "low_to_high_budget_pct": lh_budget,
                                    "high_to_low_budget_pct": hl_budget,
                                    "low_to_high_route_threshold": lh_t,
                                    "high_to_low_route_threshold": hl_t,
                                }
                                row.update({f"old_{k}": v for k, v in old_row.items()})
                                row.update({f"adapt_{k}": v for k, v in adapt_row.items()})
                                row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                                row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                                row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                                row["adapt_tp_preserved"] = bool(row["adapt_tp"] >= adapt_base["tp"])
                                row["holdout_tp_preserved"] = bool(row["holdout_tp"] >= hold_base["tp"])
                                row["adapt_acc_gain"] = float(row["adapt_accuracy"] - adapt_base["accuracy"])
                                row["adapt_bacc_gain"] = float(row["adapt_balanced_accuracy"] - adapt_base["balanced_accuracy"])
                                row["holdout_acc_gain"] = float(row["holdout_accuracy"] - hold_base["accuracy"])
                                row["holdout_bacc_gain"] = float(row["holdout_balanced_accuracy"] - hold_base["balanced_accuracy"])
                                # Selection score deliberately avoids holdout metrics.
                                row["selection_score_safe"] = (
                                    float(row["adapt_accuracy"])
                                    + 0.8 * float(row["adapt_balanced_accuracy"])
                                    + 0.03 * float(row["adapt_net_rescue"])
                                    + 0.01 * float(row["adapt_tp"] - adapt_base["tp"])
                                    - 0.03 * float(row["old_hurt_n"])
                                    - 0.01 * float(row["old_routed_pct"])
                                )
                                rows.append(row)
                                case_cache[len(rows) - 1] = (
                                    old_pred,
                                    old_prob,
                                    old_lh,
                                    old_hl,
                                    hold_pred,
                                    hold_prob,
                                    hold_lh,
                                    hold_hl,
                                    hold_p,
                                )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "meta_overlay_all_policies.csv", index=False, encoding="utf-8-sig")
    guard92 = summary[summary["old_guard_092"]].copy()
    safe = guard92[
        guard92["adapt_tp_preserved"]
        & (guard92["adapt_acc_gain"] >= 0)
        & (guard92["adapt_bacc_gain"] >= 0)
    ].copy()
    selected_df = safe.sort_values(
        ["selection_score_safe", "adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"], ascending=False
    )
    if selected_df.empty:
        selected_df = guard92.sort_values(["adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"], ascending=False)
    holdout_ref = guard92.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    holdout_tp_ref = guard92[guard92["holdout_tp_preserved"]].sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    selected_df.head(100).to_csv(out / "top_selected_by_old_plus_adapt_only.csv", index=False, encoding="utf-8-sig")
    holdout_ref.head(100).to_csv(out / "top_holdout_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")
    holdout_tp_ref.head(100).to_csv(out / "top_holdout_tp_preserved_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")

    def one(df: pd.DataFrame) -> dict[str, object] | None:
        return None if df.empty else df.iloc[0].to_dict()

    selected = one(selected_df)
    best_holdout_ref = one(holdout_ref)
    best_holdout_tp_ref = one(holdout_tp_ref)

    def save_cases(prefix: str, row: dict[str, object] | None) -> None:
        if row is None:
            return
        matches = summary[
            (summary["meta_candidate"] == row["meta_candidate"])
            & (summary["low_to_high_threshold"] == row["low_to_high_threshold"])
            & (summary["high_to_low_threshold"] == row["high_to_low_threshold"])
            & (summary["route_mode"] == row["route_mode"])
            & (summary["threshold_source"] == row["threshold_source"])
            & (summary["low_to_high_budget_pct"] == row["low_to_high_budget_pct"])
            & (summary["high_to_low_budget_pct"] == row["high_to_low_budget_pct"])
        ]
        idx = int(matches.index[0])
        old_pred, old_prob, old_lh, old_hl, hold_pred, hold_prob, hold_lh, hold_hl, hold_meta_prob = case_cache[idx]
        old_case = old[[
            "case_id",
            "original_case_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
            "base_prob",
            "base_pred",
        ]].copy()
        old_case["meta_prob_high"] = old_meta[row["meta_candidate"]].to_numpy(float)
        old_case["overlay_low_to_high"] = old_lh.astype(int)
        old_case["overlay_high_to_low"] = old_hl.astype(int)
        old_case["overlay_final_prob_high"] = old_prob
        old_case["overlay_final_pred_idx"] = old_pred
        old_case["overlay_correct"] = (old_pred == y_old).astype(int)
        old_case.to_csv(out / f"{prefix}_old_case_predictions.csv", index=False, encoding="utf-8-sig")

        hold_case = hold[[
            "case_id",
            "original_case_id",
            "source_folder",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "image_name",
            "base_prob",
            "base_pred",
        ]].copy()
        hold_case["meta_prob_high"] = hold_meta_prob
        hold_case["overlay_low_to_high"] = hold_lh.astype(int)
        hold_case["overlay_high_to_low"] = hold_hl.astype(int)
        hold_case["overlay_final_prob_high"] = hold_prob
        hold_case["overlay_final_pred_idx"] = hold_pred
        hold_case["overlay_correct"] = (hold_pred == y_hold).astype(int)
        hold_case.to_csv(out / f"{prefix}_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_cases("selected_by_old_plus_adapt", selected)
    save_cases("best_holdout_reference", best_holdout_ref)
    save_cases("best_holdout_tp_preserved_reference", best_holdout_tp_ref)

    comp_rows = [
        {
            "name": "base",
            "old_accuracy": old_base["accuracy"],
            "old_balanced_accuracy": old_base["balanced_accuracy"],
            "adapt_accuracy": adapt_base["accuracy"],
            "adapt_balanced_accuracy": adapt_base["balanced_accuracy"],
            "adapt_tn": adapt_base["tn"],
            "adapt_fp": adapt_base["fp"],
            "adapt_fn": adapt_base["fn"],
            "adapt_tp": adapt_base["tp"],
            "holdout_accuracy": hold_base["accuracy"],
            "holdout_balanced_accuracy": hold_base["balanced_accuracy"],
            "holdout_tn": hold_base["tn"],
            "holdout_fp": hold_base["fp"],
            "holdout_fn": hold_base["fn"],
            "holdout_tp": hold_base["tp"],
            "policy": "No.64 protected old + old-only proxy base on third",
        }
    ]
    for name, row in [
        ("selected_by_old_plus_adapt", selected),
        ("best_holdout_reference", best_holdout_ref),
        ("best_holdout_tp_preserved_reference", best_holdout_tp_ref),
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
                "policy": (
                    f"{row['meta_candidate']} hi={row['low_to_high_threshold']} lo={row['high_to_low_threshold']} "
                    f"route={row['route_mode']} source={row['threshold_source']} "
                    f"lh={row['low_to_high_budget_pct']} hl={row['high_to_low_budget_pct']}"
                ),
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "meta_overlay_key_comparison.csv", index=False, encoding="utf-8-sig")

    report = {
        "protocol": {
            "selection_uses_holdout": False,
            "selection_data": "old OOF + third adapt72 only",
            "holdout_data": "third adapt72 holdout 234 cases",
            "method": "OOF meta probability + separate low-to-high and high-to-low overlay budgets",
        },
        "old_base": old_base,
        "adapt_base": adapt_base,
        "holdout_base": hold_base,
        "selected_by_old_plus_adapt": selected,
        "best_holdout_reference_under_old_guard92": best_holdout_ref,
        "best_holdout_tp_preserved_reference_under_old_guard92": best_holdout_tp_ref,
        "n_meta_candidates": int(len(meta_names)),
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard92)),
        "n_safe_adapt": int(len(safe)),
        "output_dir": str(out),
    }
    (out / "meta_overlay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
