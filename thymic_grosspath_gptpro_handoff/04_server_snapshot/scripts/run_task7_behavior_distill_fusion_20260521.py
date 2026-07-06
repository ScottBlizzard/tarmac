from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill/fuse Task7 behavior reviewer into image/model reviewer.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--run64-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521",
    )
    parser.add_argument(
        "--route-case-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/63_gross_hardcore_signal_fixed_20260521/case_level_gross_signal_table.csv",
    )
    parser.add_argument(
        "--review-score-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--dino-feature-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_feature_table.csv",
    )
    parser.add_argument(
        "--dino-feature-npy",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_features.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/65_behavior_distill_fusion_20260521",
    )
    return parser.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
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
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        elif objective == "f1":
            score = f1_score(y, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_s = float(score)
            best_t = float(t)
    return best_t, best_s


def sanitize_route_name(route_col: str) -> str:
    return route_col.replace("route_score__", "").replace("__", "_")


def unsanitize_route_name(route_name: str, route_cols: list[str]) -> str:
    for col in route_cols:
        if sanitize_route_name(col) == route_name:
            return col
    raise KeyError(route_name)


def load_base(run64: Path, route_case_table: Path, review_score_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_oof = pd.read_csv(run64 / "model_non_easy_extra_oof.csv", dtype={"case_id": str})
    base = best_oof[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
            "hard_core",
            "p_best41",
            "pred_best41",
        ]
    ].copy()
    base["fold_id"] = base["fold_id"].astype(int)
    base["label_idx"] = base["label_idx"].astype(int)
    base["pred_best41"] = base["pred_best41"].astype(int)

    route = pd.read_csv(route_case_table, dtype={"case_id": str})
    route_cols = [c for c in route.columns if c.startswith("route_score__") and "model_visible" in c]
    route = route[["case_id"] + route_cols].copy()
    base = base.merge(route, on="case_id", how="left")

    review = pd.read_csv(review_score_csv, dtype={"case_id": str})
    model_cols = [c for c in review.columns if c.startswith("p_") or c.startswith("pred_") or c.startswith("review_score_")]
    review = review[["case_id"] + model_cols].copy()
    feature_base = base[["case_id", "p_best41", "pred_best41"] + route_cols].merge(review, on="case_id", how="left")
    return base, feature_base


def reconstruct_teacher(run64: Path, base: pd.DataFrame) -> pd.DataFrame:
    nested = pd.read_csv(run64 / "nested_route_summary.csv")
    best = nested.iloc[0]
    corrector = str(best["corrector"])
    route_name = str(best["route_name"])
    corr = pd.read_csv(run64 / f"{corrector}_oof.csv", dtype={"case_id": str})
    corr = base[["case_id"]].merge(corr[["case_id", "corrector_prob", "corrector_pred"]], on="case_id", how="left")

    route_cols = [c for c in base.columns if c.startswith("route_score__")]
    route_col = unsanitize_route_name(route_name, route_cols)
    choices = pd.read_csv(run64 / f"{corrector}__route_{route_name}_fold_choices.csv")
    thresholds = {int(row["fold_id"]): float(row["threshold"]) for _, row in choices.iterrows()}

    y = base["label_idx"].astype(int).to_numpy()
    base_prob = base["p_best41"].astype(float).to_numpy()
    base_pred = base["pred_best41"].astype(int).to_numpy()
    corr_prob = corr["corrector_prob"].astype(float).to_numpy()
    corr_pred = corr["corrector_pred"].astype(int).to_numpy()
    score = base[route_col].astype(float).to_numpy()
    folds = base["fold_id"].astype(int).to_numpy()
    routed = np.zeros(len(base), dtype=bool)
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    for fold, threshold in thresholds.items():
        mask = (folds == fold) & (score >= threshold)
        routed[mask] = True
        final_prob[mask] = corr_prob[mask]
        final_pred[mask] = corr_pred[mask]

    out = base[["case_id", "fold_id", "label_idx"]].copy()
    out["teacher_route_col"] = route_col
    out["teacher_routed"] = routed
    out["teacher_prob"] = final_prob
    out["teacher_pred"] = final_pred
    out["teacher_correct"] = (final_pred == y).astype(int)
    out["teacher_corrector_prob"] = corr_prob
    out["teacher_corrector_pred"] = corr_pred
    return out


def align_dino(project_root: Path, args: argparse.Namespace, base: pd.DataFrame) -> np.ndarray:
    table = pd.read_csv(project_root / args.dino_feature_table, dtype={"case_id": str})
    arr = np.load(project_root / args.dino_feature_npy).astype(np.float32)
    order = base[["case_id"]].merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        missing = order.loc[order["feature_idx"].isna(), "case_id"].head().tolist()
        raise KeyError(f"Missing DINO features: {missing}")
    return arr[order["feature_idx"].astype(int).to_numpy()]


def make_student_model(kind: str, seed: int, is_regressor: bool = False):
    if is_regressor:
        if kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=3.0, random_state=seed))
        if kind == "extra_reg":
            return ExtraTreesRegressor(n_estimators=500, max_depth=5, min_samples_leaf=4, random_state=seed, n_jobs=-1)
        raise ValueError(kind)
    if kind == "logreg_c003":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed),
        )
    if kind == "logreg_c01":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed),
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def oof_student_prob(
    x: np.ndarray,
    y_true: np.ndarray,
    y_target: np.ndarray,
    folds: np.ndarray,
    kind: str,
    is_regressor: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prob = np.full(len(y_true), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for fold in sorted(set(folds)):
        tr = folds != fold
        te = folds == fold
        model = make_student_model(kind, 20260521 + int(fold), is_regressor=is_regressor)
        model.fit(x[tr], y_target[tr])
        if is_regressor:
            fold_prob = np.clip(model.predict(x[te]), 1e-5, 1 - 1e-5)
        else:
            fold_prob = model.predict_proba(x[te])[:, 1]
        prob[te] = fold_prob
        t, s = best_threshold(y_true[tr], np.nan_to_num(prob[tr], nan=0.5))
        rows.append({"fold_id": int(fold), "threshold_train_sofar": t, "score_train_sofar": s})
    return np.nan_to_num(prob, nan=0.5), rows


def eval_prob(y: np.ndarray, prob: np.ndarray, folds: np.ndarray, objective: str = "balanced_accuracy") -> tuple[dict[str, object], pd.DataFrame]:
    pred = np.zeros(len(y), dtype=int)
    rows = []
    for fold in sorted(set(folds)):
        tr = folds != fold
        te = folds == fold
        threshold, score = best_threshold(y[tr], prob[tr], objective)
        pred[te] = (prob[te] >= threshold).astype(int)
        rows.append({"fold_id": int(fold), "threshold": float(threshold), "inner_score": float(score)})
    return metrics(y, pred, prob), pd.DataFrame(rows)


def choose_route_threshold(
    y: np.ndarray,
    base_pred: np.ndarray,
    corr_pred: np.ndarray,
    score: np.ndarray,
    train: np.ndarray,
    budgets: list[int],
) -> tuple[float, int, float]:
    best_key = (float("-inf"), float("-inf"), 0)
    best = (float("inf"), 0, 0.0)
    n_train = int(train.sum())
    for budget in budgets:
        if budget <= 0:
            threshold = float("inf")
        else:
            k = max(1, int(round(n_train * budget / 100.0)))
            threshold = float(np.sort(score[train])[-k])
        routed = train & (score >= threshold)
        pred = base_pred.copy()
        pred[routed] = corr_pred[routed]
        acc = float(accuracy_score(y[train], pred[train]))
        bacc = float(balanced_accuracy_score(y[train], pred[train]))
        key = (acc, bacc, -int(routed.sum()))
        if key > best_key:
            best_key = key
            best = (threshold, int(routed.sum()), acc)
    return best


def nested_route_eval(
    base: pd.DataFrame,
    corr_prob: np.ndarray,
    route_col: str,
    corrector_name: str,
    budgets: list[int] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if budgets is None:
        budgets = [0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 45, 50]
    y = base["label_idx"].astype(int).to_numpy()
    folds = base["fold_id"].astype(int).to_numpy()
    base_prob = base["p_best41"].astype(float).to_numpy()
    base_pred = base["pred_best41"].astype(int).to_numpy()
    route_score = base[route_col].astype(float).to_numpy()
    corr_pred = (corr_prob >= 0.5).astype(int)
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    routed = np.zeros(len(base), dtype=bool)
    rows = []
    for fold in sorted(set(folds)):
        train = folds != fold
        test = folds == fold
        threshold, train_routed_n, train_acc = choose_route_threshold(y, base_pred, corr_pred, route_score, train, budgets)
        mask = test & (route_score >= threshold)
        routed[mask] = True
        final_prob[mask] = corr_prob[mask]
        final_pred[mask] = corr_pred[mask]
        rows.append(
            {
                "fold_id": int(fold),
                "threshold": float(threshold),
                "train_routed_n": int(train_routed_n),
                "train_acc": float(train_acc),
                "test_routed_n": int(mask.sum()),
            }
        )
    row = metrics(y, final_pred, final_prob)
    hard = base["hard_core"].astype(int).to_numpy()
    row.update(
        {
            "mode": "nested_route",
            "corrector": corrector_name,
            "route_col": route_col,
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "pass_n": int((~routed).sum()),
            "pass_acc": float((final_pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "routed_acc": float((final_pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "hard_core_routed": int(hard[routed].sum()),
            "hard_core_recall": float(hard[routed].sum() / max(hard.sum(), 1)),
            "rescue_n": int(((base_pred != y) & (final_pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    case = base[["case_id", "original_case_id", "fold_id", "label_idx", "difficulty_fine", "p_best41", "pred_best41"]].copy()
    case["corrector_prob"] = corr_prob
    case["corrector_pred"] = corr_pred
    case["routed"] = routed
    case["final_prob"] = final_prob
    case["final_pred"] = final_pred
    case["final_correct"] = final_pred == y
    return row, pd.DataFrame(rows), case


@dataclass(frozen=True)
class MetaConfig:
    name: str
    feature_set: str
    classifier: str
    c: float = 0.1


def make_meta_model(cfg: MetaConfig, seed: int):
    if cfg.classifier == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=cfg.c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed),
        )
    if cfg.classifier == "extra":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(cfg.classifier)


def oof_meta(x: pd.DataFrame, y: np.ndarray, folds: np.ndarray, cfg: MetaConfig) -> tuple[np.ndarray, pd.DataFrame]:
    prob = np.full(len(y), np.nan, dtype=float)
    rows = []
    x_np = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    for fold in sorted(set(folds)):
        tr = folds != fold
        te = folds == fold
        model = make_meta_model(cfg, 20260701 + int(fold))
        model.fit(x_np[tr], y[tr])
        prob[te] = model.predict_proba(x_np[te])[:, 1]
        t, s = best_threshold(y[tr], np.nan_to_num(prob[tr], nan=0.5))
        rows.append({"fold_id": int(fold), "threshold_train_sofar": float(t), "score_train_sofar": float(s)})
    return np.nan_to_num(prob, nan=0.5), pd.DataFrame(rows)


def load_top_corrector_probs(run64: Path, base: pd.DataFrame, max_files: int = 20) -> pd.DataFrame:
    nested = pd.read_csv(run64 / "nested_route_summary.csv")
    names = []
    for name in nested["corrector"].tolist():
        if name not in names:
            names.append(str(name))
        if len(names) >= max_files:
            break
    out = base[["case_id"]].copy()
    for name in names:
        path = run64 / f"{name}_oof.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"case_id": str})[["case_id", "corrector_prob", "corrector_pred"]]
        df = df.rename(columns={"corrector_prob": f"corrprob_{name}", "corrector_pred": f"corrpred_{name}"})
        out = out.merge(df, on="case_id", how="left")
    return out.drop(columns=["case_id"])


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    run64 = project_root / args.run64_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base, feature_base = load_base(run64, project_root / args.route_case_table, project_root / args.review_score_csv)
    teacher = reconstruct_teacher(run64, base)
    dino = align_dino(project_root, args, base)
    y = base["label_idx"].astype(int).to_numpy()
    folds = base["fold_id"].astype(int).to_numpy()
    route_cols = [c for c in base.columns if c.startswith("route_score__")]

    base_metrics = metrics(y, base["pred_best41"].astype(int).to_numpy(), base["p_best41"].astype(float).to_numpy())
    teacher_metrics = metrics(y, teacher["teacher_pred"].astype(int).to_numpy(), teacher["teacher_prob"].astype(float).to_numpy())
    teacher.to_csv(output_dir / "teacher_best64_case_outputs.csv", index=False, encoding="utf-8-sig")

    student_rows = []
    student_probs: dict[str, np.ndarray] = {}
    student_targets = {
        "true": y.astype(float),
        "teacher_pred": teacher["teacher_pred"].astype(int).to_numpy().astype(float),
    }
    for target_name, target in student_targets.items():
        for kind in ["logreg_c003", "logreg_c01", "extra", "rf"]:
            name = f"dino_student_{target_name}_{kind}"
            prob, fold_rows = oof_student_prob(dino, y, target.astype(int), folds, kind, is_regressor=False)
            student_probs[name] = prob
            row, threshold_rows = eval_prob(y, prob, folds)
            row.update({"name": name, "target": target_name, "student_type": kind, "mode": "student_standalone"})
            student_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(output_dir / f"{name}_train_folds.csv", index=False, encoding="utf-8-sig")
            threshold_rows.to_csv(output_dir / f"{name}_threshold_folds.csv", index=False, encoding="utf-8-sig")
    for kind, target in [("ridge", teacher["teacher_prob"].astype(float).to_numpy()), ("extra_reg", teacher["teacher_prob"].astype(float).to_numpy())]:
        name = f"dino_student_teacher_prob_{kind}"
        prob, fold_rows = oof_student_prob(dino, y, target, folds, kind, is_regressor=True)
        student_probs[name] = prob
        row, threshold_rows = eval_prob(y, prob, folds)
        row.update({"name": name, "target": "teacher_prob", "student_type": kind, "mode": "student_standalone"})
        student_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(output_dir / f"{name}_train_folds.csv", index=False, encoding="utf-8-sig")
        threshold_rows.to_csv(output_dir / f"{name}_threshold_folds.csv", index=False, encoding="utf-8-sig")

    student_summary = pd.DataFrame(student_rows).sort_values(["accuracy", "balanced_accuracy", "f1"], ascending=False)
    student_summary.to_csv(output_dir / "dino_student_summary.csv", index=False, encoding="utf-8-sig")

    route_rows = []
    case_outputs = []
    for name, prob in student_probs.items():
        for route_col in route_cols:
            row, fold_choices, case = nested_route_eval(base, prob, route_col, name)
            route_rows.append(row)
            safe_route = re.sub(r"[^A-Za-z0-9_]+", "_", sanitize_route_name(route_col))
            fold_choices.to_csv(output_dir / f"{name}__route_{safe_route}_fold_choices.csv", index=False, encoding="utf-8-sig")
            case["run_name"] = f"{name}__{safe_route}"
            if len(case_outputs) < 12:
                case_outputs.append(case)
    route_summary = pd.DataFrame(route_rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False)
    route_summary.to_csv(output_dir / "dino_student_nested_route_summary.csv", index=False, encoding="utf-8-sig")

    student_feature_df = pd.DataFrame({f"studentprob_{k}": v for k, v in student_probs.items()})
    top_corr = load_top_corrector_probs(run64, base, max_files=24)
    numeric_base = feature_base.drop(columns=["case_id"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta_feature_sets = {
        "base_routes_students": pd.concat(
            [
                numeric_base[[c for c in numeric_base.columns if c in ["p_best41", "pred_best41"] or c.startswith("route_score__")]],
                student_feature_df,
            ],
            axis=1,
        ),
        "base_routes_topcorr": pd.concat(
            [
                numeric_base[[c for c in numeric_base.columns if c in ["p_best41", "pred_best41"] or c.startswith("route_score__")]],
                top_corr,
            ],
            axis=1,
        ),
        "base_routes_students_topcorr": pd.concat(
            [
                numeric_base[[c for c in numeric_base.columns if c in ["p_best41", "pred_best41"] or c.startswith("route_score__")]],
                student_feature_df,
                top_corr,
            ],
            axis=1,
        ),
        "model_visible_plus_students_topcorr": pd.concat([numeric_base, student_feature_df, top_corr], axis=1),
    }

    meta_rows = []
    meta_probs: dict[str, np.ndarray] = {}
    for feature_name, xdf in meta_feature_sets.items():
        for classifier in ["logreg", "extra"]:
            c_values = [0.01, 0.03, 0.1, 0.3] if classifier == "logreg" else [0.1]
            for c in c_values:
                cfg = MetaConfig(
                    name=f"meta_{feature_name}_{classifier}_c{str(c).replace('.', '')}",
                    feature_set=feature_name,
                    classifier=classifier,
                    c=c,
                )
                prob, fold_rows = oof_meta(xdf, y, folds, cfg)
                meta_probs[cfg.name] = prob
                row, threshold_rows = eval_prob(y, prob, folds)
                row.update({"name": cfg.name, "feature_set": feature_name, "classifier": classifier, "c": c, "mode": "meta_standalone"})
                meta_rows.append(row)
                fold_rows.to_csv(output_dir / f"{cfg.name}_train_folds.csv", index=False, encoding="utf-8-sig")
                threshold_rows.to_csv(output_dir / f"{cfg.name}_threshold_folds.csv", index=False, encoding="utf-8-sig")
                for route_col in route_cols:
                    rrow, fchoices, _ = nested_route_eval(base, prob, route_col, cfg.name)
                    rrow.update(
                        {
                            "name": cfg.name,
                            "feature_set": feature_name,
                            "classifier": classifier,
                            "c": c,
                            "mode": "meta_nested_route",
                        }
                    )
                    meta_rows.append(rrow)
                    safe_route = re.sub(r"[^A-Za-z0-9_]+", "_", sanitize_route_name(route_col))
                    fchoices.to_csv(output_dir / f"{cfg.name}__route_{safe_route}_fold_choices.csv", index=False, encoding="utf-8-sig")

    meta_summary = pd.DataFrame(meta_rows).sort_values(["accuracy", "balanced_accuracy", "f1"], ascending=False)
    meta_summary.to_csv(output_dir / "meta_fusion_summary.csv", index=False, encoding="utf-8-sig")

    all_rows = []
    for _, row in student_summary.iterrows():
        all_rows.append(row.to_dict())
    for _, row in route_summary.iterrows():
        all_rows.append(row.to_dict())
    for _, row in meta_summary.iterrows():
        all_rows.append(row.to_dict())
    combined = pd.DataFrame(all_rows).sort_values(["accuracy", "balanced_accuracy", "f1"], ascending=False)
    combined.to_csv(output_dir / "combined_behavior_distill_fusion_summary.csv", index=False, encoding="utf-8-sig")

    best_name = str(combined.iloc[0].get("name", combined.iloc[0].get("corrector", "")))
    best_case = None
    if best_name in meta_probs:
        row, frows = eval_prob(y, meta_probs[best_name], folds)
        pred = np.zeros(len(y), dtype=int)
        for _, fr in frows.iterrows():
            fold = int(fr["fold_id"])
            mask = folds == fold
            pred[mask] = (meta_probs[best_name][mask] >= float(fr["threshold"])).astype(int)
        best_case = base[["case_id", "original_case_id", "fold_id", "label_idx", "task_l6_label", "difficulty_fine", "p_best41", "pred_best41"]].copy()
        best_case["prob"] = meta_probs[best_name]
        best_case["pred"] = pred
        best_case["correct"] = pred == y
        best_case.to_csv(output_dir / "best_case_outputs.csv", index=False, encoding="utf-8-sig")

    report = {
        "base41": base_metrics,
        "teacher_best64": teacher_metrics,
        "best_student": student_summary.head(10).to_dict(orient="records"),
        "best_student_route": route_summary.head(10).to_dict(orient="records"),
        "best_meta_fusion": meta_summary.head(20).to_dict(orient="records"),
        "best_combined": combined.head(30).to_dict(orient="records"),
        "note": "No doctor gross text, pathology text, or case-id lookup is used. Teacher is the image/model-only behavior reviewer from run64.",
    }
    (output_dir / "behavior_distill_fusion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Base41:", json.dumps(base_metrics, ensure_ascii=False))
    print("Teacher64:", json.dumps(teacher_metrics, ensure_ascii=False))
    print("\nBest DINO students:")
    print(student_summary.head(20).to_string(index=False))
    print("\nBest DINO-student routed:")
    print(route_summary.head(20).to_string(index=False))
    print("\nBest meta fusion:")
    print(meta_summary.head(30).to_string(index=False))
    print("\nBest combined:")
    print(combined.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
