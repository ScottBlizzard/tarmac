from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 image-only hard-core reviewer/corrector.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--best41-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv",
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
        "--route-case-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/63_gross_hardcore_signal_fixed_20260521/case_level_gross_signal_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521",
    )
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
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


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


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


def load_data(project_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    curriculum = pd.read_csv(project_root / args.curriculum_csv, dtype={"case_id": str})
    base = curriculum[["case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].copy()
    base["fold_id"] = base["fold_id"].astype(int)
    base["label_idx"] = base["label_idx"].astype(int)
    base["hard_core"] = (base["difficulty_fine"] == "hard_core").astype(int)

    registry = pd.read_csv(project_root / args.registry_csv, dtype={"case_id": str, "original_case_id": str})
    keep = [c for c in ["case_id", "original_case_id", "image_count", "selection_rule", "task_l6_label", "task_l7_label"] if c in registry.columns]
    base = base.merge(registry[keep], on="case_id", how="left")

    best41 = pd.read_csv(project_root / args.best41_csv, dtype={"case_id": str})
    best41 = best41[["case_id", "final_prob_high", "final_pred", "p_upper", "pred_upper"]].rename(
        columns={
            "final_prob_high": "p_best41",
            "final_pred": "pred_best41",
            "p_upper": "p_upper41",
            "pred_upper": "pred_upper41",
        }
    )
    base = base.merge(best41, on="case_id", how="left")
    base["best41_wrong"] = (base["pred_best41"].astype(int) != base["label_idx"].astype(int)).astype(int)
    base["best41_fn"] = ((base["label_idx"].astype(int) == 1) & (base["pred_best41"].astype(int) == 0)).astype(int)
    base["best41_fp"] = ((base["label_idx"].astype(int) == 0) & (base["pred_best41"].astype(int) == 1)).astype(int)

    review = pd.read_csv(project_root / args.review_score_csv, dtype={"case_id": str})
    allowed_cols = ["case_id"] + [
        c for c in review.columns if c.startswith("p_") or c.startswith("pred_") or c.startswith("review_score_")
    ]
    review = review[allowed_cols].copy()
    base = base.merge(review, on="case_id", how="left")

    numeric_cols = [
        c
        for c in base.columns
        if (
            c.startswith("p_")
            or c.startswith("pred_")
            or c.startswith("review_score_")
            or c in ["p_best41", "pred_best41", "p_upper41", "pred_upper41", "image_count"]
        )
        and pd.api.types.is_numeric_dtype(base[c])
    ]
    model_feat = base[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model_feat["logit_best41"] = logit(base["p_best41"].astype(float).to_numpy())
    model_feat["best41_margin"] = np.abs(base["p_best41"].astype(float).to_numpy() - 0.5)
    model_feat = pd.concat([model_feat, pd.get_dummies(base[["selection_rule"]].fillna(""), dtype=float)], axis=1)

    feature_table = pd.read_csv(project_root / args.dino_feature_table, dtype={"case_id": str})
    features = np.load(project_root / args.dino_feature_npy).astype(np.float32)
    order = base[["case_id"]].merge(feature_table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        raise KeyError(f"Missing DINO features for {order.loc[order['feature_idx'].isna(), 'case_id'].head().tolist()}")
    dino = features[order["feature_idx"].astype(int).to_numpy()]

    route_table = pd.read_csv(project_root / args.route_case_table, dtype={"case_id": str})
    route_table = base[["case_id"]].merge(route_table, on="case_id", how="left")
    route_scores = {
        "hard_core_model_visible_logreg": route_table["route_score__hard_core__model_visible__logreg"].astype(float).to_numpy(),
        "hard_core_model_visible_extra": route_table["route_score__hard_core__model_visible__extra"].astype(float).to_numpy(),
        "best41_wrong_model_visible_extra": route_table["route_score__best41_wrong__model_visible__extra"].astype(float).to_numpy(),
        "best41_fn_model_visible_extra": route_table["route_score__best41_fn__model_visible__extra"].astype(float).to_numpy(),
    }
    return base.reset_index(drop=True), model_feat.reset_index(drop=True), dino, route_scores


@dataclass(frozen=True)
class CorrectorConfig:
    name: str
    feature_set: str
    train_scope: str
    classifier: str
    c: float = 0.1
    objective: str = "balanced_accuracy"


def make_classifier(cfg: CorrectorConfig, seed: int):
    if cfg.classifier == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=cfg.c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed),
        )
    if cfg.classifier == "extra":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if cfg.classifier == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(cfg.classifier)


def feature_matrix(cfg: CorrectorConfig, model_feat: pd.DataFrame, dino: np.ndarray) -> np.ndarray:
    if cfg.feature_set == "model":
        return model_feat.to_numpy(dtype=np.float32)
    if cfg.feature_set == "dino":
        return dino.astype(np.float32)
    if cfg.feature_set == "dino_model":
        return np.concatenate([dino.astype(np.float32), model_feat.to_numpy(dtype=np.float32)], axis=1)
    raise ValueError(cfg.feature_set)


def scope_mask(df: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(df), dtype=bool)
    if scope == "hard_core":
        return df["difficulty_fine"].eq("hard_core").to_numpy()
    if scope == "hard_all":
        return df["difficulty_fine"].isin(["hard_core", "hard_salvage_teacher"]).to_numpy()
    if scope == "non_easy":
        return ~df["difficulty_fine"].eq("easy").to_numpy()
    raise ValueError(scope)


def inner_threshold(x: np.ndarray, y: np.ndarray, folds: np.ndarray, train_mask: np.ndarray, cfg: CorrectorConfig) -> tuple[float, float]:
    prob = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(set(folds[train_mask])):
        tr = train_mask & (folds != fold)
        va = train_mask & (folds == fold)
        if tr.sum() < 12 or va.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_classifier(cfg, 20260521 + int(fold))
        clf.fit(x[tr], y[tr])
        prob[va] = clf.predict_proba(x[va])[:, 1]
    valid = train_mask & ~np.isnan(prob)
    if valid.sum() < 12 or len(np.unique(y[valid])) < 2:
        return 0.5, float("nan")
    return best_threshold(y[valid], prob[valid], cfg.objective)


def train_oof_corrector(df: pd.DataFrame, x: np.ndarray, cfg: CorrectorConfig) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    scope = scope_mask(df, cfg.train_scope)
    prob = np.full(len(df), np.nan, dtype=float)
    pred = np.full(len(df), -1, dtype=int)
    fold_rows: list[dict[str, object]] = []
    for fold in sorted(set(folds)):
        train = (folds != fold) & scope
        test = folds == fold
        if train.sum() < 12 or len(np.unique(y[train])) < 2:
            train = folds != fold
        threshold, inner_score = inner_threshold(x, y, folds, train, cfg)
        clf = make_classifier(cfg, 20260601 + int(fold))
        clf.fit(x[train], y[train])
        fold_prob = clf.predict_proba(x[test])[:, 1]
        prob[test] = fold_prob
        pred[test] = (fold_prob >= threshold).astype(int)
        fold_rows.append(
            {
                "fold_id": int(fold),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "threshold": float(threshold),
                "inner_score": float(inner_score) if not np.isnan(inner_score) else np.nan,
            }
        )
    out = df[
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
    out["corrector_prob"] = prob
    out["corrector_pred"] = pred
    out["base_correct"] = out["pred_best41"].astype(int) == out["label_idx"].astype(int)
    out["corrector_correct"] = out["corrector_pred"].astype(int) == out["label_idx"].astype(int)
    return out, fold_rows


def apply_oracle(df: pd.DataFrame, corr: pd.DataFrame, apply_scope: str) -> dict[str, object]:
    y = df["label_idx"].astype(int).to_numpy()
    pred = df["pred_best41"].astype(int).to_numpy().copy()
    prob = df["p_best41"].astype(float).to_numpy().copy()
    routed = scope_mask(df, apply_scope)
    pred[routed] = corr["corrector_pred"].astype(int).to_numpy()[routed]
    prob[routed] = corr["corrector_prob"].astype(float).to_numpy()[routed]
    row = metric_dict(y, pred, prob)
    row.update(
        {
            "apply_scope": apply_scope,
            "routed_n": int(routed.sum()),
            "routed_acc": float((pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "pass_acc": float((pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "rescue_n": int(((df["pred_best41"].astype(int).to_numpy() != y) & (pred == y) & routed).sum()),
            "hurt_n": int(((df["pred_best41"].astype(int).to_numpy() == y) & (pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row


def choose_route_threshold(
    y: np.ndarray,
    base_pred: np.ndarray,
    corr_pred: np.ndarray,
    score: np.ndarray,
    train: np.ndarray,
    budgets: list[int],
) -> tuple[float, int, float]:
    best = (float("-inf"), -1.0, 0.0, 0)
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
        routed_n = int(routed.sum())
        key = (acc, bacc, -routed_n)
        if key > (best[0], best[1], -best[3]):
            best = (acc, bacc, threshold, routed_n)
    return float(best[2]), int(best[3]), float(best[0])


def nested_route(df: pd.DataFrame, corr: pd.DataFrame, route_score: np.ndarray, route_name: str, budgets: list[int]) -> dict[str, object]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    base_pred = df["pred_best41"].astype(int).to_numpy()
    base_prob = df["p_best41"].astype(float).to_numpy()
    corr_pred = corr["corrector_pred"].astype(int).to_numpy()
    corr_prob = corr["corrector_prob"].astype(float).to_numpy()
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    final_routed = np.zeros(len(df), dtype=bool)
    chosen = []
    for fold in sorted(set(folds)):
        train = folds != fold
        test = folds == fold
        threshold, routed_train_n, train_acc = choose_route_threshold(y, base_pred, corr_pred, route_score, train, budgets)
        routed_test = test & (route_score >= threshold)
        final_pred[routed_test] = corr_pred[routed_test]
        final_prob[routed_test] = corr_prob[routed_test]
        final_routed[routed_test] = True
        chosen.append(
            {
                "fold_id": int(fold),
                "threshold": float(threshold),
                "train_routed_n": int(routed_train_n),
                "train_acc": float(train_acc),
                "test_routed_n": int(routed_test.sum()),
            }
        )
    row = metric_dict(y, final_pred, final_prob)
    hard = df["hard_core"].astype(int).to_numpy()
    row.update(
        {
            "route_name": route_name,
            "routed_n": int(final_routed.sum()),
            "routed_pct": float(final_routed.mean()),
            "pass_n": int((~final_routed).sum()),
            "pass_acc": float((final_pred[~final_routed] == y[~final_routed]).mean()) if (~final_routed).any() else np.nan,
            "routed_acc": float((final_pred[final_routed] == y[final_routed]).mean()) if final_routed.any() else np.nan,
            "hard_core_routed": int(hard[final_routed].sum()),
            "hard_core_recall": float(hard[final_routed].sum() / max(hard.sum(), 1)),
            "rescue_n": int(((base_pred != y) & (final_pred == y) & final_routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & final_routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    row["fold_choices"] = chosen
    return row


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, model_feat, dino, route_scores = load_data(project_root, args)

    y = df["label_idx"].astype(int).to_numpy()
    base_pred = df["pred_best41"].astype(int).to_numpy()
    base_prob = df["p_best41"].astype(float).to_numpy()
    base_metrics = metric_dict(y, base_pred, base_prob)
    (output_dir / "base41_metrics.json").write_text(json.dumps(base_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    configs: list[CorrectorConfig] = []
    for feature_set in ["model", "dino", "dino_model"]:
        for train_scope in ["hard_core", "hard_all", "non_easy", "all"]:
            for classifier in ["logreg", "extra"]:
                if classifier == "logreg":
                    for c in [0.01, 0.03, 0.1, 0.3]:
                        configs.append(
                            CorrectorConfig(
                                name=f"{feature_set}_{train_scope}_{classifier}_c{str(c).replace('.', '')}",
                                feature_set=feature_set,
                                train_scope=train_scope,
                                classifier=classifier,
                                c=c,
                            )
                        )
                else:
                    configs.append(
                        CorrectorConfig(
                            name=f"{feature_set}_{train_scope}_{classifier}",
                            feature_set=feature_set,
                            train_scope=train_scope,
                            classifier=classifier,
                        )
                    )

    rows = []
    budget_rows = []
    for cfg in configs:
        x = feature_matrix(cfg, model_feat, dino)
        corr, fold_rows = train_oof_corrector(df, x, cfg)
        corr.to_csv(output_dir / f"{cfg.name}_oof.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(output_dir / f"{cfg.name}_folds.csv", index=False, encoding="utf-8-sig")
        corrector_all = metric_dict(y, corr["corrector_pred"].astype(int).to_numpy(), corr["corrector_prob"].astype(float).to_numpy())
        for apply_scope in ["hard_core", "hard_all", "non_easy", "all"]:
            row = apply_oracle(df, corr, apply_scope)
            row.update(
                {
                    "mode": "oracle_scope",
                    "corrector": cfg.name,
                    "feature_set": cfg.feature_set,
                    "train_scope": cfg.train_scope,
                    "classifier": cfg.classifier,
                    "c": cfg.c,
                    "corrector_all_acc": corrector_all["accuracy"],
                    "corrector_all_bacc": corrector_all["balanced_accuracy"],
                }
            )
            rows.append(row)
        for route_name, score in route_scores.items():
            row = nested_route(df, corr, score, route_name, budgets=[0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 45, 50])
            fold_choices = row.pop("fold_choices")
            row.update(
                {
                    "mode": "nested_route",
                    "corrector": cfg.name,
                    "feature_set": cfg.feature_set,
                    "train_scope": cfg.train_scope,
                    "classifier": cfg.classifier,
                    "c": cfg.c,
                    "corrector_all_acc": corrector_all["accuracy"],
                    "corrector_all_bacc": corrector_all["balanced_accuracy"],
                }
            )
            budget_rows.append(row)
            pd.DataFrame(fold_choices).to_csv(
                output_dir / f"{cfg.name}__route_{route_name}_fold_choices.csv", index=False, encoding="utf-8-sig"
            )
        pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False).to_csv(
            output_dir / "oracle_scope_summary.partial.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(budget_rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False).to_csv(
            output_dir / "nested_route_summary.partial.csv", index=False, encoding="utf-8-sig"
        )

    oracle = pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False)
    nested = pd.DataFrame(budget_rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False)
    oracle.to_csv(output_dir / "oracle_scope_summary.csv", index=False, encoding="utf-8-sig")
    nested.to_csv(output_dir / "nested_route_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "base41": base_metrics,
        "best_oracle_scope": oracle.head(20).to_dict(orient="records"),
        "best_nested_route": nested.head(20).to_dict(orient="records"),
        "note": "No doctor gross-finding text or case-id matched gross fields are used by the reviewer.",
    }
    (output_dir / "image_only_hardcore_reviewer_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Base41:", json.dumps(base_metrics, ensure_ascii=False))
    print("\nBest oracle-scope image-only reviewer:")
    print(oracle.head(30).to_string(index=False))
    print("\nBest nested-route image-only reviewer:")
    print(nested.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
