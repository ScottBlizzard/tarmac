from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outer CV on third batch with old data always included in training.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/19_third_outer_cv_with_old_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def read_old(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    return table.reset_index(drop=True), feat[table["feature_idx"].astype(int).to_numpy()]


def read_third(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    registry = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table["feature_idx"].astype(int)
    frame = registry.merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    frame["feature_idx"] = frame["feature_idx"].astype(int)
    return frame.reset_index(drop=True), feat[frame["feature_idx"].to_numpy()]


def repeat_indices(is_adapt: np.ndarray, repeat_adapt: int) -> np.ndarray:
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    if repeat_adapt <= 1:
        return np.arange(len(is_adapt))
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
    if score is not None:
        out["auc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan
    return out


def make_model(kind: str, seed: int):
    if kind.startswith("logreg"):
        c = float(kind.split("_")[1])
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed),
        )
    if kind.startswith("extratrees"):
        depth = int(kind.split("_")[1])
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def score_model(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def best_threshold(y: np.ndarray, score: np.ndarray, objective: str) -> float:
    best = (float("-inf"), 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        pred = (score >= t).astype(int)
        value = accuracy_score(y, pred) if objective == "accuracy" else balanced_accuracy_score(y, pred)
        cand = (float(value), -abs(float(t) - 0.5))
        if cand > (best[0], -abs(best[1] - 0.5)):
            best = (float(value), float(t))
    return best[1]


def inner_oof(
    x_train: np.ndarray,
    y_train: np.ndarray,
    is_adapt: np.ndarray,
    kind: str,
    repeat_adapt: int,
    seed: int,
) -> np.ndarray:
    oof = np.zeros(len(y_train), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_fold, (tr, va) in enumerate(skf.split(x_train, y_train)):
        idx = repeat_indices(is_adapt[tr], repeat_adapt)
        model = make_model(kind, seed + inner_fold)
        model.fit(x_train[tr][idx], y_train[tr][idx])
        oof[va] = score_model(model, x_train[va])
    return oof


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)
    y_old = old["label_idx"].to_numpy(int)
    y_third = third["label_idx"].to_numpy(int)

    candidates = [
        ("logreg_0.0003", 4),
        ("logreg_0.001", 2),
        ("extratrees_3", 1),
        ("extratrees_5", 1),
    ]
    objective_list = ["accuracy", "balanced_accuracy"]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for kind, repeat_adapt in candidates:
        for objective in objective_list:
            all_pred = np.zeros(len(third), dtype=int)
            all_prob = np.zeros(len(third), dtype=np.float32)
            fold_rows: list[dict[str, object]] = []
            for fold, (tr_idx, te_idx) in enumerate(outer.split(x_third, third["task_l6_label"])):
                x_train = np.concatenate([x_old, x_third[tr_idx]], axis=0)
                y_train = np.concatenate([y_old, y_third[tr_idx]])
                is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(tr_idx), dtype=bool)])
                oof = inner_oof(x_train, y_train, is_adapt, kind, repeat_adapt, args.seed + fold * 100)
                threshold = best_threshold(y_train, oof, objective)
                full_idx = repeat_indices(is_adapt, repeat_adapt)
                model = make_model(kind, args.seed + fold * 1000)
                model.fit(x_train[full_idx], y_train[full_idx])
                prob = score_model(model, x_third[te_idx])
                pred = (prob >= threshold).astype(int)
                all_pred[te_idx] = pred
                all_prob[te_idx] = prob
                fm = {"fold": fold, "kind": kind, "repeat_adapt": repeat_adapt, "threshold_objective": objective, "threshold": threshold}
                fm.update({f"fold_{k}": v for k, v in metric_dict(y_third[te_idx], pred, prob).items()})
                fold_rows.append(fm)
                pred_rows.append(
                    third.iloc[te_idx][
                        ["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]
                    ].assign(
                        kind=kind,
                        repeat_adapt=repeat_adapt,
                        threshold_objective=objective,
                        fold=fold,
                        threshold=threshold,
                        prob_high=prob,
                        pred_idx=pred,
                        correct=(pred == y_third[te_idx]).astype(int),
                    )
                )
            row = {"kind": kind, "repeat_adapt": repeat_adapt, "threshold_objective": objective}
            row.update({f"third_oof_{k}": v for k, v in metric_dict(y_third, all_pred, all_prob).items()})
            rows.append(row)
            pd.DataFrame(fold_rows).to_csv(
                out / f"fold_metrics__{kind}__r{repeat_adapt}__{objective}.csv", index=False, encoding="utf-8-sig"
            )

    summary = pd.DataFrame(rows).sort_values(["third_oof_accuracy", "third_oof_balanced_accuracy"], ascending=False)
    summary.to_csv(out / "third_outer_cv_summary.csv", index=False, encoding="utf-8-sig")
    preds = pd.concat(pred_rows, ignore_index=True)
    preds.to_csv(out / "third_outer_cv_case_predictions.csv", index=False, encoding="utf-8-sig")
    subtype = (
        preds.groupby(["kind", "repeat_adapt", "threshold_objective", "task_l6_label"])
        .agg(n=("case_id", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
        .reset_index()
    )
    subtype.to_csv(out / "third_outer_cv_subtype_metrics.csv", index=False, encoding="utf-8-sig")
    report = {
        "protocol": "5-fold outer CV on third batch; old data is always included in training; threshold chosen inside each fold from training OOF only.",
        "third_n": int(len(third)),
        "old_n": int(len(old)),
        "summary": summary.to_dict("records"),
        "output_dir": str(out),
    }
    (out / "third_outer_cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
