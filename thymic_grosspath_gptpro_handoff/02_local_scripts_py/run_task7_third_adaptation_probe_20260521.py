from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LOW_HIGH = {"AB": 0, "B1": 0, "B2": 1, "TC": 1}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    params: tuple[tuple[str, object], ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 third-batch adaptation probe using frozen whole+crop features.")
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
        "--existing-third-pred",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521/third_batch_external_case_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/01_third_balanced_adapt_probe_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


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


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        elif objective == "f1":
            score = f1_score(y, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def read_old_features(root: Path, rel_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel_dir
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    features = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    table["label_idx"] = table["label_idx"].astype(int)
    x = features[table["feature_idx"].astype(int).to_numpy()]
    return table.reset_index(drop=True), x


def read_third_features(root: Path, rel_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel_dir
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    registry = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    features = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    frame = registry.merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if frame["feature_idx"].isna().any():
        missing = frame.loc[frame["feature_idx"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing third feature rows: {missing}")
    frame["label_idx"] = frame["label_idx"].astype(int)
    x = features[frame["feature_idx"].astype(int).to_numpy()]
    return frame.reset_index(drop=True), x


def stable_order_key(value: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def adaptation_profiles() -> dict[str, dict[str, int]]:
    return {
        "adapt48_balanced": {"AB": 18, "B1": 6, "B2": 12, "TC": 12},
        "adapt60_balanced": {"AB": 24, "B1": 6, "B2": 15, "TC": 15},
        "adapt72_balanced": {"AB": 30, "B1": 6, "B2": 18, "TC": 18},
        "adapt84_balanced": {"AB": 36, "B1": 6, "B2": 21, "TC": 21},
        "adapt72_high_focus": {"AB": 24, "B1": 4, "B2": 22, "TC": 22},
        "adapt96_high_focus": {"AB": 30, "B1": 4, "B2": 26, "TC": 36},
    }


def split_adapt_holdout(third: pd.DataFrame, profile: dict[str, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    adapt_idx: list[int] = []
    for subtype, n in profile.items():
        g = third.loc[third["task_l6_label"].eq(subtype)].copy()
        if len(g) < n:
            raise ValueError(f"Profile asks {n} {subtype}, only {len(g)} available.")
        g["_key"] = g["case_id"].map(lambda x: stable_order_key(str(x), seed))
        chosen = g.sort_values("_key").head(n).index.tolist()
        adapt_idx.extend(chosen)
    adapt_mask = np.zeros(len(third), dtype=bool)
    adapt_mask[np.array(adapt_idx, dtype=int)] = True
    return np.where(adapt_mask)[0], np.where(~adapt_mask)[0]


def candidate_specs() -> list[ModelSpec]:
    specs = []
    for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        specs.append(ModelSpec(f"logreg_c{str(c).replace('.', 'p')}", "logreg", (("C", c),)))
    specs.extend(
        [
            ModelSpec("svc_c03", "svc", (("C", 0.3),)),
            ModelSpec("svc_c1", "svc", (("C", 1.0),)),
            ModelSpec("rf_d4", "rf", (("max_depth", 4),)),
            ModelSpec("rf_d6", "rf", (("max_depth", 6),)),
            ModelSpec("extra_d4", "extra", (("max_depth", 4),)),
            ModelSpec("extra_d6", "extra", (("max_depth", 6),)),
            ModelSpec("gb_d1", "gb", (("max_depth", 1), ("learning_rate", 0.05))),
            ModelSpec("gb_d2", "gb", (("max_depth", 2), ("learning_rate", 0.03))),
            ModelSpec("mlp64", "mlp", (("hidden_layer_sizes", (64,)), ("alpha", 0.03))),
        ]
    )
    return specs


def make_model(spec: ModelSpec, seed: int):
    params = dict(spec.params)
    if spec.kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
        )
    if spec.kind == "svc":
        return make_pipeline(
            StandardScaler(),
            SVC(C=float(params["C"]), kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=seed),
        )
    if spec.kind == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=int(params["max_depth"]),
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if spec.kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=800,
            max_depth=int(params["max_depth"]),
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if spec.kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=160,
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            random_state=seed,
        )
    if spec.kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=float(params["alpha"]),
                max_iter=1200,
                early_stopping=True,
                random_state=seed,
            ),
        )
    raise ValueError(spec.kind)


def prob_high(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-s))
    raise TypeError(type(model))


def repeat_train_rows(x: np.ndarray, y: np.ndarray, is_adapt: np.ndarray, repeat_adapt: int) -> tuple[np.ndarray, np.ndarray]:
    if repeat_adapt <= 1:
        return x, y
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    idx = np.concatenate([old_idx, np.tile(adapt_idx, repeat_adapt)])
    return x[idx], y[idx]


def fit_oof_and_external(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    is_adapt: np.ndarray,
    x_test: np.ndarray,
    repeat_adapt: int,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y_train), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = repeat_train_rows(x_train[tr], y_train[tr], is_adapt[tr], repeat_adapt)
        model = make_model(spec, seed + fold)
        model.fit(x_tr, y_tr)
        oof[va] = prob_high(model, x_train[va])
    x_full, y_full = repeat_train_rows(x_train, y_train, is_adapt, repeat_adapt)
    model = make_model(spec, seed + 1000)
    model.fit(x_full, y_full)
    external = prob_high(model, x_test)
    return oof, external


def subtype_metrics(third_holdout: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    y = third_holdout["label_idx"].to_numpy(dtype=int)
    for subtype, g in third_holdout.groupby("task_l6_label"):
        idx = g.index.to_numpy()
        rows.append(
            {
                "task_l6_label": subtype,
                "n": int(len(g)),
                "correct": int((pred[idx] == y[idx]).sum()),
                "accuracy": float((pred[idx] == y[idx]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, x_old = read_old_features(root, args.old_feature_dir)
    third, x_third = read_third_features(root, args.third_feature_dir)
    y_old = old["label_idx"].to_numpy(dtype=int)

    existing = pd.read_csv(root / args.existing_third_pred, dtype={"case_id": str})
    existing = existing.set_index("case_id")

    all_rows: list[dict[str, object]] = []
    all_case_rows: list[pd.DataFrame] = []
    all_split_rows: list[dict[str, object]] = []

    for profile_name, profile in adaptation_profiles().items():
        adapt_idx, hold_idx = split_adapt_holdout(third, profile, args.seed)
        adapt = third.iloc[adapt_idx].reset_index(drop=True)
        hold = third.iloc[hold_idx].reset_index(drop=True)
        x_adapt = x_third[adapt_idx]
        x_hold = x_third[hold_idx]
        y_adapt = adapt["label_idx"].to_numpy(dtype=int)
        y_hold = hold["label_idx"].to_numpy(dtype=int)

        for split_name, frame in [("adapt", adapt), ("holdout", hold)]:
            vc = frame["task_l6_label"].value_counts().to_dict()
            all_split_rows.append({"profile": profile_name, "split": split_name, "n": int(len(frame)), **{f"n_{k}": int(vc.get(k, 0)) for k in ["AB", "B1", "B2", "TC"]}})

        # Current best 64-style old-only baseline on the same holdout.
        existing_hold = hold[["case_id", "original_case_id", "task_l6_label", "label_idx"]].merge(
            existing[["final_pred_idx", "final_prob_high"]], left_on="case_id", right_index=True, how="left"
        )
        base_pred = existing_hold["final_pred_idx"].astype(int).to_numpy()
        base_prob = existing_hold["final_prob_high"].astype(float).to_numpy()
        base_metric = metric_dict(y_hold, base_pred, base_prob)
        base_row = {
            "profile": profile_name,
            "train_mode": "existing_wpc64_old_only_on_same_holdout",
            "repeat_adapt": 0,
            "model": "existing_wpc64",
            "threshold": np.nan,
            "train_n_old": int(len(old)),
            "train_n_adapt": 0,
            "holdout_n": int(len(hold)),
        }
        base_row.update({f"holdout_{k}": v for k, v in base_metric.items()})
        all_rows.append(base_row)

        for repeat_adapt in [1, 2, 4, 8]:
            x_train = np.concatenate([x_old, x_adapt], axis=0)
            y_train = np.concatenate([y_old, y_adapt], axis=0)
            is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])
            for i, spec in enumerate(candidate_specs()):
                oof, p_hold = fit_oof_and_external(
                    spec,
                    x_train,
                    y_train,
                    is_adapt,
                    x_hold,
                    repeat_adapt,
                    args.seed + 100 * i + repeat_adapt,
                    args.n_splits,
                )
                t, _ = best_threshold(y_train, oof, "balanced_accuracy")
                oof_pred = (oof >= t).astype(int)
                pred_hold = (p_hold >= t).astype(int)
                oof_metric = metric_dict(y_train, oof_pred, oof)
                hold_metric = metric_dict(y_hold, pred_hold, p_hold)
                row = {
                    "profile": profile_name,
                    "train_mode": "old_plus_third_adapt",
                    "repeat_adapt": int(repeat_adapt),
                    "model": spec.name,
                    "threshold": float(t),
                    "train_n_old": int(len(old)),
                    "train_n_adapt": int(len(adapt)),
                    "holdout_n": int(len(hold)),
                }
                row.update({f"oof_{k}": v for k, v in oof_metric.items()})
                row.update({f"holdout_{k}": v for k, v in hold_metric.items()})
                all_rows.append(row)

        # Store predictions for the best row within this profile later.
        pd.DataFrame({"profile": profile_name, "case_id": adapt["case_id"], "split": "adapt"}).to_csv(
            out / f"{profile_name}_adapt_case_ids.csv", index=False, encoding="utf-8-sig"
        )
        hold[["case_id", "original_case_id", "task_l6_label", "label_idx"]].to_csv(
            out / f"{profile_name}_holdout_case_ids.csv", index=False, encoding="utf-8-sig"
        )

    summary = pd.DataFrame(all_rows)
    summary = summary.sort_values(["holdout_balanced_accuracy", "holdout_accuracy", "holdout_f1"], ascending=False)
    summary.to_csv(out / "third_adaptation_probe_summary.csv", index=False, encoding="utf-8-sig")
    split_df = pd.DataFrame(all_split_rows)
    split_df.to_csv(out / "third_adaptation_split_summary.csv", index=False, encoding="utf-8-sig")

    # Refit and export case predictions for the top adapted setting.
    adapted_summary = summary[summary["train_mode"].eq("old_plus_third_adapt")].copy()
    top = adapted_summary.iloc[0].to_dict()
    profile_name = str(top["profile"])
    profile = adaptation_profiles()[profile_name]
    adapt_idx, hold_idx = split_adapt_holdout(third, profile, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    x_adapt = x_third[adapt_idx]
    x_hold = x_third[hold_idx]
    y_adapt = adapt["label_idx"].to_numpy(dtype=int)
    y_hold = hold["label_idx"].to_numpy(dtype=int)
    spec = next(s for s in candidate_specs() if s.name == top["model"])
    x_train = np.concatenate([x_old, x_adapt], axis=0)
    y_train = np.concatenate([y_old, y_adapt], axis=0)
    is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])
    x_full, y_full = repeat_train_rows(x_train, y_train, is_adapt, int(top["repeat_adapt"]))
    model = make_model(spec, args.seed + 9999)
    model.fit(x_full, y_full)
    p_hold = prob_high(model, x_hold)
    pred_hold = (p_hold >= float(top["threshold"])).astype(int)
    case_out = hold[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]].copy()
    case_out["prob_high"] = p_hold
    case_out["pred_idx"] = pred_hold
    case_out["correct"] = (pred_hold == y_hold).astype(int)
    case_out.to_csv(out / "best_adapted_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")
    subtype = subtype_metrics(case_out.reset_index(drop=True), pred_hold)
    subtype.to_csv(out / "best_adapted_holdout_metrics_by_subtype.csv", index=False, encoding="utf-8-sig")

    report = {
        "boundary": {
            "third_batch_used_for_training": True,
            "adaptation_selection": "deterministic stratified case-id hash within each subtype",
            "holdout_rule": "third-batch cases not selected into the adaptation subset are not used for fitting or threshold selection",
            "note": "This is an adaptation exploration. Once a setting is chosen, a fresh locked split should be fixed for final reporting.",
        },
        "top_adapted_setting": top,
        "top_adapted_subtype_metrics": subtype.to_dict("records"),
        "best_existing_wpc64_rows": summary[summary["train_mode"].eq("existing_wpc64_old_only_on_same_holdout")]
        .sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False)
        .head(3)
        .to_dict("records"),
    }
    (out / "third_adaptation_probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = (
        "# Task7 第三批适配训练快速探针\n\n"
        "第三批按亚型确定性抽取一部分进入训练，剩余病例作为同批次 holdout。"
        "训练只使用现有 whole+crop DINO 冻结特征，不重训 backbone。\n\n"
        f"当前最佳适配设置：`{top['profile']}`，模型 `{top['model']}`，repeat_adapt={int(top['repeat_adapt'])}。\n"
        f"holdout ACC={top['holdout_accuracy']:.4f}，BACC={top['holdout_balanced_accuracy']:.4f}，F1={top['holdout_f1']:.4f}，"
        f"TN/FP/FN/TP={int(top['holdout_tn'])}/{int(top['holdout_fp'])}/{int(top['holdout_fn'])}/{int(top['holdout_tp'])}。\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
