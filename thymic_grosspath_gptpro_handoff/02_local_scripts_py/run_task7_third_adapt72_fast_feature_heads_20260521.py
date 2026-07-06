from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


PROFILE = {"AB": 24, "B1": 4, "B2": 22, "TC": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast frozen-feature head sweep for Task7 old+third-adapt72.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--old-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521",
    )
    parser.add_argument(
        "--third-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/14_adapt72_fast_feature_heads_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def split_profile(third: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    chosen: list[int] = []
    for subtype, n in PROFILE.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        chosen.extend(g.sort_values("_key").head(n).index.tolist())
    mask = np.zeros(len(third), dtype=bool)
    mask[np.array(chosen, dtype=int)] = True
    return np.where(mask)[0], np.where(~mask)[0]


def read_old(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table["feature_idx"].astype(int)
    return table.reset_index(drop=True), feat[table["feature_idx"].to_numpy()]


def read_third(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    registry = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table["feature_idx"].astype(int)
    frame = registry.merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    frame["feature_idx"] = frame["feature_idx"].astype(int)
    return frame.reset_index(drop=True), feat[frame["feature_idx"].to_numpy()]


def repeat_adapt_indices(is_adapt: np.ndarray, repeat_adapt: int) -> np.ndarray:
    if repeat_adapt <= 1:
        return np.arange(len(is_adapt))
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    return np.concatenate([old_idx, np.tile(adapt_idx, repeat_adapt)])


def metric_dict(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, object]:
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
    if score is not None and len(np.unique(y)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y, score))
        except ValueError:
            out["auc"] = np.nan
    return out


def predict_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    return model.predict(x).astype(float)


def best_threshold(y: np.ndarray, score: np.ndarray, objective: str) -> float:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (score >= t).astype(int)
        s = accuracy_score(y, pred) if objective == "accuracy" else balanced_accuracy_score(y, pred)
        if (s, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_s, best_t = float(s), float(t)
    return best_t


def make_model(kind: str, seed: int):
    if kind.startswith("logreg_"):
        c = float(kind.split("_")[1])
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed),
        )
    if kind.startswith("ridge_"):
        alpha = float(kind.split("_")[1])
        return make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, class_weight="balanced", random_state=seed))
    if kind.startswith("linsvc_"):
        c = float(kind.split("_")[1])
        return make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight="balanced", max_iter=8000, random_state=seed))
    if kind.startswith("sgd_"):
        alpha = float(kind.split("_")[1])
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="modified_huber",
                alpha=alpha,
                class_weight="balanced",
                max_iter=4000,
                tol=1e-4,
                random_state=seed,
            ),
        )
    if kind.startswith("rbfsvc_"):
        _, c, gamma = kind.split("_")
        return make_pipeline(
            StandardScaler(),
            SVC(C=float(c), gamma=float(gamma), class_weight="balanced", probability=True, random_state=seed),
        )
    if kind.startswith("knn_"):
        k = int(kind.split("_")[1])
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine"))
    if kind.startswith("extratrees_"):
        depth = None if kind.split("_")[1] == "none" else int(kind.split("_")[1])
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if kind.startswith("rf_"):
        depth = None if kind.split("_")[1] == "none" else int(kind.split("_")[1])
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def transform_features(x_train: np.ndarray, x_hold: np.ndarray, variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if variant == "raw":
        return x_train, x_hold
    if variant.startswith("pca"):
        n = int(variant.replace("pca", ""))
        pipe = make_pipeline(StandardScaler(), PCA(n_components=n, random_state=seed, whiten=True))
        return pipe.fit_transform(x_train), pipe.transform(x_hold)
    raise ValueError(variant)


def run_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    is_adapt: np.ndarray,
    x_hold: np.ndarray,
    variant: str,
    model_kind: str,
    repeat_adapt: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_var, x_hold_var = transform_features(x_train, x_hold, variant, seed)
    oof = np.zeros(len(y_train), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x_var, y_train)):
        idx = repeat_adapt_indices(is_adapt[tr], repeat_adapt)
        model = make_model(model_kind, seed + fold)
        model.fit(x_var[tr][idx], y_train[tr][idx])
        oof[va] = predict_score(model, x_var[va])
    full_idx = repeat_adapt_indices(is_adapt, repeat_adapt)
    model = make_model(model_kind, seed + 999)
    model.fit(x_var[full_idx], y_train[full_idx])
    return oof, predict_score(model, x_hold_var)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)
    adapt_idx, hold_idx = split_profile(third, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    x_train = np.concatenate([x_old, x_third[adapt_idx]], axis=0)
    y_train = np.concatenate([old["label_idx"].to_numpy(int), adapt["label_idx"].to_numpy(int)])
    x_hold = x_third[hold_idx]
    y_hold = hold["label_idx"].to_numpy(int)
    is_old = np.concatenate([np.ones(len(old), dtype=bool), np.zeros(len(adapt), dtype=bool)])
    is_adapt = ~is_old

    variants = ["raw", "pca32", "pca64"]
    model_kinds = [
        "logreg_0.0003",
        "logreg_0.001",
        "logreg_0.003",
        "logreg_0.01",
        "ridge_0.3",
        "ridge_1.0",
        "ridge_3.0",
        "linsvc_0.001",
        "linsvc_0.003",
        "linsvc_0.01",
        "sgd_0.0003",
        "sgd_0.001",
        "sgd_0.003",
        "knn_5",
        "knn_9",
        "extratrees_3",
        "extratrees_5",
        "rf_5",
    ]
    pca_only = ["rbfsvc_0.3_scale", "rbfsvc_1.0_scale", "rbfsvc_3.0_scale"]
    repeats = [1, 2, 4]

    rows: list[dict[str, object]] = []
    pred_tables: list[pd.DataFrame] = []
    for variant in variants:
        kinds = model_kinds + (pca_only if variant != "raw" else [])
        for model_kind in kinds:
            for repeat in repeats:
                name = f"{variant}__{model_kind}__r{repeat}"
                try:
                    oof, hold_score = run_candidate(
                        x_train, y_train, is_adapt, x_hold, variant, model_kind, repeat, args.seed + len(rows) * 13
                    )
                except Exception as exc:  # keep the sweep running; failed heads are recorded.
                    rows.append({"candidate": name, "error": repr(exc)})
                    continue
                for objective in ["balanced_accuracy", "accuracy"]:
                    t = best_threshold(y_train, oof, objective)
                    train_pred = (oof >= t).astype(int)
                    hold_pred = (hold_score >= t).astype(int)
                    row = {
                        "candidate": name,
                        "variant": variant,
                        "model_kind": model_kind,
                        "repeat_adapt": repeat,
                        "threshold_objective": objective,
                        "threshold": t,
                    }
                    row.update({f"train_{k}": v for k, v in metric_dict(y_train, train_pred, oof).items()})
                    row.update({f"old_oof_{k}": v for k, v in metric_dict(y_train[is_old], train_pred[is_old], oof[is_old]).items()})
                    row.update(
                        {f"adapt_oof_{k}": v for k, v in metric_dict(y_train[is_adapt], train_pred[is_adapt], oof[is_adapt]).items()}
                    )
                    row.update({f"holdout_{k}": v for k, v in metric_dict(y_hold, hold_pred, hold_score).items()})
                    row["selection_score"] = (
                        min(float(row["old_oof_balanced_accuracy"]), float(row["adapt_oof_balanced_accuracy"]))
                        + 0.12 * float(row["train_balanced_accuracy"])
                        + 0.05 * min(float(row["old_oof_accuracy"]), float(row["adapt_oof_accuracy"]))
                    )
                    rows.append(row)
                    if len(pred_tables) < 12:
                        pred_tables.append(
                            hold[["case_id", "original_case_id", "task_l6_label", "label_idx", "image_name"]].assign(
                                candidate=name,
                                threshold_objective=objective,
                                threshold=t,
                                score_high=hold_score,
                                pred_idx=hold_pred,
                                correct=(hold_pred == y_hold).astype(int),
                            )
                        )

    summary = pd.DataFrame(rows)
    complete = summary[summary.get("error").isna()] if "error" in summary.columns else summary
    complete = complete.sort_values(
        ["selection_score", "old_oof_balanced_accuracy", "adapt_oof_balanced_accuracy"], ascending=False
    )
    summary.to_csv(out / "fast_feature_head_all_results.csv", index=False, encoding="utf-8-sig")
    complete.to_csv(out / "fast_feature_head_summary.csv", index=False, encoding="utf-8-sig")
    if pred_tables:
        pd.concat(pred_tables, ignore_index=True).to_csv(out / "sample_holdout_predictions.csv", index=False, encoding="utf-8-sig")

    selected = complete.iloc[0].to_dict()
    best_holdout_acc = complete.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False).iloc[0].to_dict()
    best_holdout_bacc = complete.sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False).iloc[0].to_dict()
    report = {
        "selection_uses_third_holdout": False,
        "old_n": int(len(old)),
        "third_adapt_n": int(len(adapt)),
        "third_holdout_n": int(len(hold)),
        "selected_by_oof": selected,
        "best_holdout_accuracy_for_reference": best_holdout_acc,
        "best_holdout_balanced_accuracy_for_reference": best_holdout_bacc,
        "output_dir": str(out),
    }
    (out / "fast_feature_head_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    cols = [
        "candidate",
        "threshold_objective",
        "threshold",
        "old_oof_accuracy",
        "old_oof_balanced_accuracy",
        "adapt_oof_accuracy",
        "adapt_oof_balanced_accuracy",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "holdout_f1",
        "holdout_tn",
        "holdout_fp",
        "holdout_fn",
        "holdout_tp",
    ]
    print("\nTop by OOF")
    print(complete[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
