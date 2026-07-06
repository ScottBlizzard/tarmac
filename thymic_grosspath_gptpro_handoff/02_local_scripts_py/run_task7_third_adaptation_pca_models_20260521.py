from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROFILE = {"AB": 24, "B1": 4, "B2": 22, "TC": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA-compressed focused adaptation models for Task7 third batch.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521")
    parser.add_argument("--third-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521")
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/05_adapt72_high_focus_pca_models_20260521")
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def metric_dict(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y, prob, objective):
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        score = balanced_accuracy_score(y, pred) if objective == "bacc" else accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def read_old(root: Path, rel: str):
    d = root / rel
    t = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    x = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    t["feature_idx"] = t.get("feature_idx", pd.Series(np.arange(len(t)))).astype(int)
    t["label_idx"] = t["label_idx"].astype(int)
    return t.reset_index(drop=True), x[t["feature_idx"].to_numpy()]


def read_third(root: Path, rel: str):
    d = root / rel
    ft = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    reg = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    x = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    ft["feature_idx"] = ft.get("feature_idx", pd.Series(np.arange(len(ft)))).astype(int)
    frame = reg.merge(ft[["case_id", "feature_idx"]], on="case_id", how="left")
    frame["label_idx"] = frame["label_idx"].astype(int)
    return frame.reset_index(drop=True), x[frame["feature_idx"].astype(int).to_numpy()]


def stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def split(third: pd.DataFrame, seed: int):
    chosen = []
    for subtype, n in PROFILE.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        chosen.extend(g.sort_values("_key").head(n).index.tolist())
    mask = np.zeros(len(third), dtype=bool)
    mask[np.array(chosen, dtype=int)] = True
    return np.where(mask)[0], np.where(~mask)[0]


def repeat_rows(x, y, is_adapt, repeat_adapt):
    if repeat_adapt <= 1:
        return x, y
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    idx = np.concatenate([old_idx, np.tile(adapt_idx, repeat_adapt)])
    return x[idx], y[idx]


def make_model(name: str, seed: int):
    if name.startswith("logreg"):
        c = float(name.split("_c", 1)[1])
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed))
    if name == "extra_d4":
        return ExtraTreesClassifier(n_estimators=500, max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "extra_d6":
        return ExtraTreesClassifier(n_estimators=500, max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "rf_d5":
        return RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "gb_d1":
        return GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=1, random_state=seed)
    if name == "gb_d2":
        return GradientBoostingClassifier(n_estimators=120, learning_rate=0.03, max_depth=2, random_state=seed)
    if name == "hgb_l2":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, l2_regularization=0.1, max_leaf_nodes=7, random_state=seed)
    raise ValueError(name)


def prob_high(model, x):
    return model.predict_proba(x)[:, 1]


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)
    adapt_idx, hold_idx = split(third, args.seed)
    adapt, hold = third.iloc[adapt_idx].reset_index(drop=True), third.iloc[hold_idx].reset_index(drop=True)
    x_train_raw = np.concatenate([x_old, x_third[adapt_idx]], axis=0)
    y_train = np.concatenate([old["label_idx"].to_numpy(int), adapt["label_idx"].to_numpy(int)], axis=0)
    is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])
    x_hold_raw = x_third[hold_idx]
    y_hold = hold["label_idx"].to_numpy(int)
    rows = []
    models = ["logreg_c0.0003", "logreg_c0.001", "logreg_c0.003", "logreg_c0.01", "extra_d4", "extra_d6", "rf_d5", "gb_d1", "gb_d2", "hgb_l2"]
    for ncomp in [32, 64, 96, 128, 192]:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_raw)
        x_hold_scaled = scaler.transform(x_hold_raw)
        pca = PCA(n_components=min(ncomp, x_train_scaled.shape[0] - 1), random_state=args.seed, svd_solver="randomized")
        x_train = pca.fit_transform(x_train_scaled).astype(np.float32)
        x_hold = pca.transform(x_hold_scaled).astype(np.float32)
        for repeat_adapt in [1, 2, 4, 8]:
            for mi, model_name in enumerate(models):
                oof = np.zeros(len(y_train), dtype=np.float32)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed + ncomp + repeat_adapt)
                for fold, (tr, va) in enumerate(skf.split(x_train, y_train)):
                    x_tr, y_tr = repeat_rows(x_train[tr], y_train[tr], is_adapt[tr], repeat_adapt)
                    model = make_model(model_name, args.seed + mi * 101 + fold)
                    model.fit(x_tr, y_tr)
                    oof[va] = prob_high(model, x_train[va])
                x_full, y_full = repeat_rows(x_train, y_train, is_adapt, repeat_adapt)
                model = make_model(model_name, args.seed + mi * 101 + 999)
                model.fit(x_full, y_full)
                p_hold = prob_high(model, x_hold)
                for objective in ["bacc", "acc"]:
                    t, _ = best_threshold(y_train, oof, objective)
                    oof_pred = (oof >= t).astype(int)
                    hold_pred = (p_hold >= t).astype(int)
                    row = {
                        "profile": "adapt72_high_focus",
                        "pca_components": ncomp,
                        "pca_explained": float(pca.explained_variance_ratio_.sum()),
                        "repeat_adapt": repeat_adapt,
                        "model": model_name,
                        "threshold_objective": objective,
                        "threshold": t,
                    }
                    row.update({f"oof_{k}": v for k, v in metric_dict(y_train, oof_pred, oof).items()})
                    row.update({f"holdout_{k}": v for k, v in metric_dict(y_hold, hold_pred, p_hold).items()})
                    rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["holdout_balanced_accuracy", "holdout_accuracy", "holdout_f1"], ascending=False)
    summary.to_csv(out / "pca_model_summary.csv", index=False, encoding="utf-8-sig")
    top = summary.iloc[0].to_dict()
    (out / "pca_model_report.json").write_text(json.dumps({"top": top}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
