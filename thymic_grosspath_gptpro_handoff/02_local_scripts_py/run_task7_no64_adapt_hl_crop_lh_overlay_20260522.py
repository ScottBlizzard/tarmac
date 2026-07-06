from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
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


def align_features(frame: pd.DataFrame, table_path: Path, npy_path: Path) -> np.ndarray:
    table = pd.read_csv(table_path, dtype={"case_id": str})
    arr = np.load(npy_path).astype(np.float32)
    order = frame[["case_id"]].merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        missing = order.loc[order["feature_idx"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing crop features: {missing}")
    return arr[order["feature_idx"].astype(int).to_numpy()]


@dataclass(frozen=True)
class CropSpec:
    name: str
    c: float
    adapt_weight: float


def make_model(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000),
    )


def fit_crop_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_hold: np.ndarray,
    is_adapt: np.ndarray,
    specs: list[CropSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260522)
    train_probs = pd.DataFrame()
    hold_probs = pd.DataFrame()
    rows: list[dict[str, object]] = []
    for spec in specs:
        base_model = make_model(spec.c)
        oof = np.zeros(len(y_train), dtype=float)
        hold_fold = []
        for tr, va in skf.split(x_train, y_train):
            model = clone(base_model)
            weights = np.ones(len(tr), dtype=float)
            weights[is_adapt[tr]] *= spec.adapt_weight
            model.fit(x_train[tr], y_train[tr], logisticregression__sample_weight=weights)
            oof[va] = model.predict_proba(x_train[va])[:, 1]
            hold_fold.append(model.predict_proba(x_hold)[:, 1])
        hold = np.mean(np.vstack(hold_fold), axis=0)
        train_probs[spec.name] = oof
        hold_probs[spec.name] = hold
        pred = (oof >= 0.5).astype(int)
        row = {"candidate": spec.name, "C": spec.c, "adapt_weight": spec.adapt_weight}
        row.update(metric_dict(y_train, pred, oof))
        rows.append(row)
    return train_probs, hold_probs, pd.DataFrame(rows)


def apply_combo(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    hl_prob: np.ndarray,
    hl_t: float,
    lh_prob: np.ndarray,
    lh_t: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high_to_low = (base_pred == 1) & (hl_prob < hl_t)
    pred = base_pred.copy()
    prob = base_prob.copy()
    pred[high_to_low] = 0
    prob[high_to_low] = hl_prob[high_to_low]
    low_to_high = (pred == 0) & (lh_prob >= lh_t)
    pred[low_to_high] = 1
    prob[low_to_high] = lh_prob[low_to_high]
    routed = high_to_low | low_to_high
    row = metric_dict(y, pred, prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "high_to_low_n": int(high_to_low.sum()),
            "low_to_high_n": int(low_to_high.sum()),
            "pass_acc": float((pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "routed_acc": float((pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "rescue_n": int(((base_pred != y) & (pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, pred, prob, high_to_low, low_to_high


def main() -> None:
    root = Path(".").resolve()
    base = root / "outputs/batch1_batch2_task567_20260514"
    out = base / "task7_adaptation_runs/27_no64_adapt_hl_crop_lh_overlay_20260522"
    out.mkdir(parents=True, exist_ok=True)
    run64 = base / "task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521"
    adapt_cache = base / "task7_adaptation_runs/11_unified_two_stage_adapt72_20260521"
    split_dir = base / "task7_adaptation_runs/02_third_logreg_fast_20260521"
    third_external = base / "task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521"
    old_crop_table = base / "task7_gross_feature_runs/67_roi_crop_embedding_probe_20260521/case_dino_concat_feature_table.csv"
    old_crop_npy = base / "task7_gross_feature_runs/67_roi_crop_embedding_probe_20260521/case_dino_concat_features.npy"
    third_crop_table = base / "task7_external_runs/03_third_batch_crop_64style_20260521/third_batch_dino_concat_feature_table.csv"
    third_crop_npy = base / "task7_external_runs/03_third_batch_crop_64style_20260521/third_batch_dino_concat_features.npy"

    old_no64 = reconstruct_no64_old(root, run64, None)
    all_adapt_probs = pd.read_csv(adapt_cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    hold_adapt_probs = pd.read_csv(adapt_cache / "candidate_holdout_probs.csv", dtype={"case_id": str})
    third_all = pd.read_csv(third_external / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})
    adapt_cases = pd.read_csv(split_dir / "adapt72_high_focus_adapt_cases.csv", dtype={"case_id": str})
    hold_cases = pd.read_csv(split_dir / "adapt72_high_focus_holdout_cases.csv", dtype={"case_id": str})

    old_adapt_probs = all_adapt_probs[~all_adapt_probs["case_id"].str.startswith("third_")].copy()
    adapt_adapt_probs = all_adapt_probs[all_adapt_probs["case_id"].str.startswith("third_")].copy()
    old = old_no64.merge(old_adapt_probs, on="case_id", how="inner")
    old["base_prob"] = old["no64_final_prob_high"].astype(float)
    old["base_pred"] = old["no64_final_pred_idx"].astype(int)
    adapt = third_all.merge(adapt_cases[["case_id"]], on="case_id", how="inner").merge(adapt_adapt_probs, on="case_id", how="inner")
    adapt["base_prob"] = adapt["final_prob_high"].astype(float)
    adapt["base_pred"] = adapt["final_pred_idx"].astype(int)
    hold = third_all.merge(hold_cases[["case_id"]], on="case_id", how="inner").merge(hold_adapt_probs, on="case_id", how="inner")
    hold["base_prob"] = hold["final_prob_high"].astype(float)
    hold["base_pred"] = hold["final_pred_idx"].astype(int)

    train_for_crop = pd.concat([old[["case_id", "label_idx"]], adapt[["case_id", "label_idx"]]], ignore_index=True)
    old_x = align_features(old, old_crop_table, old_crop_npy)
    adapt_x = align_features(adapt, third_crop_table, third_crop_npy)
    hold_x = align_features(hold, third_crop_table, third_crop_npy)
    x_train = np.vstack([old_x, adapt_x])
    y_train = train_for_crop["label_idx"].to_numpy(int)
    is_adapt = train_for_crop["case_id"].str.startswith("third_").to_numpy()
    specs = [
        CropSpec(f"crop_logreg_c{c:g}_aw{aw:g}", c, aw)
        for aw in [1.0, 2.0, 4.0]
        for c in [0.0003, 0.001, 0.003, 0.01]
    ]
    crop_train_probs, crop_hold_probs, crop_summary = fit_crop_candidates(x_train, y_train, hold_x, is_adapt, specs)
    crop_train_probs.insert(0, "case_id", train_for_crop["case_id"].astype(str).to_numpy())
    crop_hold_probs.insert(0, "case_id", hold["case_id"].astype(str).to_numpy())
    crop_summary.to_csv(out / "crop_candidate_oof_summary.csv", index=False, encoding="utf-8-sig")
    crop_train_probs.to_csv(out / "crop_candidate_train_oof_probs.csv", index=False, encoding="utf-8-sig")
    crop_hold_probs.to_csv(out / "crop_candidate_holdout_probs.csv", index=False, encoding="utf-8-sig")

    old_crop = crop_train_probs[~crop_train_probs["case_id"].str.startswith("third_")].reset_index(drop=True)
    adapt_crop = crop_train_probs[crop_train_probs["case_id"].str.startswith("third_")].reset_index(drop=True)
    hold_crop = crop_hold_probs.reset_index(drop=True)

    y_old = old["label_idx"].to_numpy(int)
    y_adapt = adapt["label_idx"].to_numpy(int)
    y_hold = hold["label_idx"].to_numpy(int)
    old_base_prob = old["base_prob"].to_numpy(float)
    adapt_base_prob = adapt["base_prob"].to_numpy(float)
    hold_base_prob = hold["base_prob"].to_numpy(float)
    old_base_pred = old["base_pred"].to_numpy(int)
    adapt_base_pred = adapt["base_pred"].to_numpy(int)
    hold_base_pred = hold["base_pred"].to_numpy(int)
    old_base = metric_dict(y_old, old_base_pred, old_base_prob)
    adapt_base = metric_dict(y_adapt, adapt_base_pred, adapt_base_prob)
    hold_base = metric_dict(y_hold, hold_base_pred, hold_base_prob)

    rows: list[dict[str, object]] = []
    case_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    hl_candidates = ["adapt_r4_c0.0003", "adapt_r2_c0.0003", "adapt_r8_c0.0003", "oldonly_c0.0003"]
    hl_candidates = [c for c in hl_candidates if c in old.columns]
    crop_candidates = [c for c in crop_train_probs.columns if c != "case_id"]
    hl_thresholds = [0.44, 0.50, 0.54, 0.57, 0.58, 0.60, 0.62]
    lh_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for hl_name in hl_candidates:
        for crop_name in crop_candidates:
            old_hl = old[hl_name].to_numpy(float)
            adapt_hl = adapt[hl_name].to_numpy(float)
            hold_hl = hold[hl_name].to_numpy(float)
            old_lh = old_crop[crop_name].to_numpy(float)
            adapt_lh = adapt_crop[crop_name].to_numpy(float)
            hold_lh = hold_crop[crop_name].to_numpy(float)
            for hl_t in hl_thresholds:
                for lh_t in lh_thresholds:
                    old_row, old_pred, old_prob, old_h2l, old_l2h = apply_combo(
                        y_old, old_base_prob, old_base_pred, old_hl, hl_t, old_lh, lh_t
                    )
                    adapt_row, _, _, _, _ = apply_combo(
                        y_adapt, adapt_base_prob, adapt_base_pred, adapt_hl, hl_t, adapt_lh, lh_t
                    )
                    hold_row, hold_pred, hold_prob, hold_h2l, hold_l2h = apply_combo(
                        y_hold, hold_base_prob, hold_base_pred, hold_hl, hl_t, hold_lh, lh_t
                    )
                    row = {
                        "hl_candidate": hl_name,
                        "hl_threshold": hl_t,
                        "lh_crop_candidate": crop_name,
                        "lh_threshold": lh_t,
                    }
                    row.update({f"old_{k}": v for k, v in old_row.items()})
                    row.update({f"adapt_{k}": v for k, v in adapt_row.items()})
                    row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                    row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                    row["adapt_tn_preserved"] = bool(row["adapt_tn"] >= adapt_base["tn"])
                    row["adapt_tp_preserved"] = bool(row["adapt_tp"] >= adapt_base["tp"])
                    row["holdout_tp_preserved"] = bool(row["holdout_tp"] >= hold_base["tp"])
                    row["adapt_acc_gain"] = float(row["adapt_accuracy"] - adapt_base["accuracy"])
                    row["adapt_bacc_gain"] = float(row["adapt_balanced_accuracy"] - adapt_base["balanced_accuracy"])
                    row["holdout_acc_gain"] = float(row["holdout_accuracy"] - hold_base["accuracy"])
                    row["holdout_bacc_gain"] = float(row["holdout_balanced_accuracy"] - hold_base["balanced_accuracy"])
                    row["selection_score_safe"] = (
                        float(row["adapt_accuracy"])
                        + 0.8 * float(row["adapt_balanced_accuracy"])
                        + 0.02 * float(row["adapt_net_rescue"])
                        + 0.02 * float(row["adapt_tp"] - adapt_base["tp"])
                        + 0.01 * float(row["adapt_tn"] - adapt_base["tn"])
                        - 0.035 * float(row["old_hurt_n"])
                    )
                    rows.append(row)
                    case_cache[len(rows) - 1] = (hold_pred, hold_prob, hold_h2l, hold_l2h, hold_lh)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "adapt_hl_crop_lh_overlay_all_policies.csv", index=False, encoding="utf-8-sig")
    guard = summary[summary["old_guard_092"]].copy()
    safe = guard[
        guard["adapt_tn_preserved"]
        & guard["adapt_tp_preserved"]
        & (guard["adapt_acc_gain"] >= 0)
        & (guard["adapt_bacc_gain"] >= 0)
    ].copy()
    selected_df = safe.sort_values(
        ["selection_score_safe", "adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"], ascending=False
    )
    if selected_df.empty:
        selected_df = guard.sort_values(["adapt_accuracy", "adapt_balanced_accuracy"], ascending=False)
    hold_ref = guard.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    hold_tp_ref = guard[guard["holdout_tp_preserved"]].sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
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
        m = summary[
            (summary["hl_candidate"] == row["hl_candidate"])
            & (summary["hl_threshold"] == row["hl_threshold"])
            & (summary["lh_crop_candidate"] == row["lh_crop_candidate"])
            & (summary["lh_threshold"] == row["lh_threshold"])
        ]
        idx = int(m.index[0])
        hold_pred, hold_prob, hold_h2l, hold_l2h, hold_lh = case_cache[idx]
        case = hold[[
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
        case["crop_rescue_prob_high"] = hold_lh
        case["overlay_high_to_low"] = hold_h2l.astype(int)
        case["overlay_low_to_high"] = hold_l2h.astype(int)
        case["overlay_final_prob_high"] = hold_prob
        case["overlay_final_pred_idx"] = hold_pred
        case["overlay_correct"] = (hold_pred == y_hold).astype(int)
        case.to_csv(out / f"{prefix}_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_holdout("selected_by_old_plus_adapt", selected)
    save_holdout("best_holdout_reference", best_hold)
    save_holdout("best_holdout_tp_preserved_reference", best_hold_tp)

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
            "policy": "No.64 protected old + third old-only proxy base",
        }
    ]
    for name, row in [
        ("selected_by_old_plus_adapt", selected),
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
                "policy": f"HL {row['hl_candidate']}<{row['hl_threshold']} + crop LH {row['lh_crop_candidate']}>={row['lh_threshold']}",
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "adapt_hl_crop_lh_overlay_key_comparison.csv", index=False, encoding="utf-8-sig")
    report = {
        "protocol": {
            "selection_uses_holdout": False,
            "selection_data": "old OOF + third adapt72 only",
            "holdout_data": "third adapt72 holdout 234 cases",
            "method": "whole+crop adapt head repairs FP; crop-only frozen head rescues FN",
        },
        "old_base": old_base,
        "adapt_base": adapt_base,
        "holdout_base": hold_base,
        "selected_by_old_plus_adapt": selected,
        "best_holdout_reference_under_old_guard92": best_hold,
        "best_holdout_tp_preserved_reference_under_old_guard92": best_hold_tp,
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard)),
        "n_safe_adapt": int(len(safe)),
        "output_dir": str(out),
    }
    (out / "adapt_hl_crop_lh_overlay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
