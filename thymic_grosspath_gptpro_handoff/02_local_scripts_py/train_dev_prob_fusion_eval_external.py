from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUTDIR = Path("outputs/external_quality_gate_20260525")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for col in ["p108", "p104", "p103_vitl448", "p107_qkvb"]:
        x[col] = pd.to_numeric(df[col], errors="coerce")
    x["p162_blend"] = 0.8 * x["p108"] + 0.2 * x["p104"]
    x["mean_all"] = x[["p108", "p104", "p103_vitl448", "p107_qkvb"]].mean(axis=1)
    x["max_all"] = x[["p108", "p104", "p103_vitl448", "p107_qkvb"]].max(axis=1)
    x["min_all"] = x[["p108", "p104", "p103_vitl448", "p107_qkvb"]].min(axis=1)
    x["range_all"] = x["max_all"] - x["min_all"]
    x["diff_162_103"] = x["p162_blend"] - x["p103_vitl448"]
    x["diff_108_103"] = x["p108"] - x["p103_vitl448"]
    x["diff_107_103"] = x["p107_qkvb"] - x["p103_vitl448"]
    x["vote_high_count"] = (x[["p108", "p104", "p103_vitl448", "p107_qkvb"]] >= 0.5).sum(axis=1)
    x["high_sensitive_flag"] = ((x["p103_vitl448"] >= 0.5) & (x["p162_blend"] < 0.595)).astype(int)
    x["low_sensitive_flag"] = ((x["p103_vitl448"] < 0.5) & (x["p162_blend"] >= 0.595)).astype(int)
    return x.fillna(x.median(numeric_only=True))


def load_dev() -> pd.DataFrame:
    p108 = pd.read_csv(OUTDIR / "dev_108_selected_guard92_oof_predictions.csv")
    p104 = pd.read_csv(OUTDIR / "dev_104_selected_guard92_oof_predictions.csv")
    p103 = pd.read_csv(OUTDIR / "dev_103_vitl448_tta_oof_case_predictions.csv")
    p107 = pd.read_csv(OUTDIR / "dev_107_qkvb_tta_oof_case_predictions.csv")

    dev = p108[["case_id", "domain", "third_split", "task_l6_label", "label_idx", "oof_prob_high"]].rename(columns={"oof_prob_high": "p108"})
    dev = dev.merge(p104[["case_id", "oof_prob_high"]].rename(columns={"oof_prob_high": "p104"}), on="case_id", how="inner")
    dev = dev.merge(p103[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": "p103_vitl448"}), on="case_id", how="inner")
    dev = dev.merge(p107[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": "p107_qkvb"}), on="case_id", how="inner")
    return dev


def load_external() -> pd.DataFrame:
    ext = pd.read_csv(OUTDIR / "external_gpt_quality_labels_v1.csv")
    ext = ext.rename(
        columns={
            "prob108": "p108",
            "prob104": "p104",
            "dinov3vitl_qkvb_tta107_prob": "p107_qkvb",
        }
    )
    run163 = pd.read_csv(OUTDIR / "run163_vitl448_tta_predictions.csv")[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": "p103_vitl448"})
    ext = ext.merge(run163, on="case_id", how="left")
    return ext


def metric_row(name: str, df: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict[str, object]:
    y = df["label_idx"].astype(int).to_numpy()
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "subset": name,
        "n": len(df),
        "threshold": threshold,
        "acc": accuracy_score(y, pred) if len(df) else np.nan,
        "bacc": balanced_accuracy_score(y, pred) if len(np.unique(y)) > 1 else np.nan,
        "f1": f1_score(y, pred) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else np.nan,
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sens": tp / (tp + fn) if tp + fn else np.nan,
        "spec": tn / (tn + fp) if tn + fp else np.nan,
    }


def choose_threshold(dev: pd.DataFrame, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    old_mask = dev["domain"].eq("old").to_numpy()
    third_mask = ~old_mask
    y = dev["label_idx"].astype(int).to_numpy()
    for th in np.linspace(0.25, 0.75, 101):
        pred = (prob >= th).astype(int)
        old_acc = accuracy_score(y[old_mask], pred[old_mask])
        old_bacc = balanced_accuracy_score(y[old_mask], pred[old_mask])
        third_acc = accuracy_score(y[third_mask], pred[third_mask])
        third_bacc = balanced_accuracy_score(y[third_mask], pred[third_mask])
        all_bacc = balanced_accuracy_score(y, pred)
        guard = old_acc >= 0.915 and old_bacc >= 0.915
        score = (0.45 * all_bacc + 0.25 * third_bacc + 0.2 * old_bacc + 0.1 * third_acc) if guard else -1
        rows.append(
            {
                "threshold": th,
                "old_acc": old_acc,
                "old_bacc": old_bacc,
                "third_acc": third_acc,
                "third_bacc": third_bacc,
                "all_bacc": all_bacc,
                "guard": guard,
                "score": score,
            }
        )
    scan = pd.DataFrame(rows)
    valid = scan[scan["guard"]]
    if valid.empty:
        best = scan.sort_values("all_bacc", ascending=False).iloc[0]
    else:
        best = valid.sort_values("score", ascending=False).iloc[0]
    return float(best["threshold"]), scan


def main() -> None:
    dev = load_dev()
    ext = load_external()
    x_dev = build_features(dev)
    y_dev = dev["label_idx"].astype(int).to_numpy()
    x_ext = build_features(ext)

    models = {
        "logreg_balanced": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.4, max_iter=5000, random_state=42)),
        "logreg_plain": make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=5000, random_state=42)),
        "hgb_l2": HistGradientBoostingClassifier(max_iter=80, learning_rate=0.04, max_leaf_nodes=7, l2_regularization=0.3, random_state=42),
        "rf_shallow_balanced": RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=12, class_weight="balanced", random_state=42),
    }

    all_rows = []
    threshold_rows = []
    pred_table = ext[["case_id", "original_case_id", "task_l6_label", "label_idx", "strict_task7_eval", "gpt_quality_label_v1"]].copy()
    for name, model in models.items():
        model.fit(x_dev, y_dev)
        prob_dev = model.predict_proba(x_dev)[:, 1]
        th, scan = choose_threshold(dev, prob_dev)
        scan["model"] = name
        threshold_rows.append(scan)
        prob_ext = model.predict_proba(x_ext)[:, 1]
        pred_table[f"{name}_prob"] = prob_ext
        pred_table[f"{name}_pred"] = (prob_ext >= th).astype(int)

        for subset, sub in [
            ("dev_all", dev),
            ("dev_old", dev[dev["domain"].eq("old")]),
            ("dev_third", dev[~dev["domain"].eq("old")]),
        ]:
            idx = sub.index
            row = metric_row(subset, sub, prob_dev[idx], th)
            row["model"] = name
            all_rows.append(row)

        for subset, sub in [
            ("external_all", ext),
            ("external_strict", ext[ext["strict_task7_eval"] == 1]),
            ("external_readable_auto", ext[(ext["strict_task7_eval"] == 1) & (ext["gpt_quality_label_v1"] == "readable_auto")]),
            ("external_auto_plus_review", ext[(ext["strict_task7_eval"] == 1) & (ext["gpt_quality_label_v1"] != "retake_required")]),
        ]:
            row = metric_row(subset, sub, prob_ext[sub.index], th)
            row["model"] = name
            all_rows.append(row)

    res = pd.DataFrame(all_rows)
    res.to_csv(OUTDIR / "dev_prob_fusion_external_eval.csv", index=False, encoding="utf-8-sig")
    pd.concat(threshold_rows, ignore_index=True).to_csv(OUTDIR / "dev_prob_fusion_threshold_scan.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(OUTDIR / "external_dev_prob_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    print(res.sort_values(["subset", "acc"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
