from __future__ import annotations

import argparse
import hashlib
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


PROFILE = {"AB": 24, "B1": 4, "B2": 22, "TC": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified two-stage domain-adapted Task7 router on frozen whole+crop features.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521")
    parser.add_argument("--third-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521")
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521")
    parser.add_argument("--seed", type=int, default=20260521)
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
        score = accuracy_score(y, pred) if objective == "accuracy" else balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def read_old(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table.get("feature_idx", pd.Series(np.arange(len(table)))).astype(int)
    table["label_idx"] = table["label_idx"].astype(int)
    return table.reset_index(drop=True), feat[table["feature_idx"].to_numpy()]


def read_third(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    registry = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table.get("feature_idx", pd.Series(np.arange(len(table)))).astype(int)
    frame = registry.merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if frame["feature_idx"].isna().any():
        raise KeyError("Missing third feature rows.")
    frame["label_idx"] = frame["label_idx"].astype(int)
    return frame.reset_index(drop=True), feat[frame["feature_idx"].astype(int).to_numpy()]


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


def repeat_rows(x: np.ndarray, y: np.ndarray, is_adapt: np.ndarray, repeat_adapt: int) -> tuple[np.ndarray, np.ndarray]:
    if repeat_adapt <= 1:
        return x, y
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    idx = np.concatenate([old_idx, np.tile(adapt_idx, repeat_adapt)])
    return x[idx], y[idx]


def make_logreg(c: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed),
    )


def oof_and_holdout(
    x: np.ndarray,
    y: np.ndarray,
    is_adapt: np.ndarray,
    x_hold: np.ndarray,
    c: float,
    repeat_adapt: int,
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x, y)):
        x_tr, y_tr = repeat_rows(x[tr], y[tr], is_adapt[tr], repeat_adapt)
        model = make_logreg(c, seed + fold)
        model.fit(x_tr, y_tr)
        oof[va] = model.predict_proba(x[va])[:, 1]
    x_full, y_full = repeat_rows(x, y, is_adapt, repeat_adapt)
    model = make_logreg(c, seed + 1000)
    model.fit(x_full, y_full)
    hold = model.predict_proba(x_hold)[:, 1]
    return oof, hold


def old_only_oof_and_external(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_adapt: np.ndarray,
    x_hold: np.ndarray,
    c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof = np.zeros(len(y_old), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x_old, y_old)):
        model = make_logreg(c, seed + fold)
        model.fit(x_old[tr], y_old[tr])
        oof[va] = model.predict_proba(x_old[va])[:, 1]
    model = make_logreg(c, seed + 1000)
    model.fit(x_old, y_old)
    return oof, model.predict_proba(x_adapt)[:, 1], model.predict_proba(x_hold)[:, 1]


def entropy_binary(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def build_meta_features(probs: np.ndarray, names: list[str]) -> pd.DataFrame:
    feat = pd.DataFrame()
    for i, name in enumerate(names):
        p = probs[:, i].astype(float)
        feat[f"p_{name}"] = p
        feat[f"margin_{name}"] = np.abs(p - 0.5)
        feat[f"entropy_{name}"] = entropy_binary(p)
    feat["prob_mean"] = probs.mean(axis=1)
    feat["prob_std"] = probs.std(axis=1)
    feat["prob_min"] = probs.min(axis=1)
    feat["prob_max"] = probs.max(axis=1)
    feat["prob_range"] = probs.max(axis=1) - probs.min(axis=1)
    votes = (probs >= 0.5).astype(int)
    feat["vote_sum"] = votes.sum(axis=1)
    feat["vote_disagree"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_router(kind: str, seed: int):
    if kind == "route_logreg_c01":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=seed))
    if kind == "route_logreg_c1":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed))
    if kind == "route_extra_d3":
        return ExtraTreesClassifier(n_estimators=400, max_depth=3, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1)
    if kind == "route_extra_d5":
        return ExtraTreesClassifier(n_estimators=400, max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(kind)


def oof_router_scores(
    x_meta: np.ndarray,
    target: np.ndarray,
    x_hold_meta: np.ndarray,
    kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(target)) < 2:
        return np.zeros(len(target), dtype=np.float32), np.zeros(len(x_hold_meta), dtype=np.float32)
    oof = np.zeros(len(target), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x_meta, target)):
        model = make_router(kind, seed + fold)
        model.fit(x_meta[tr], target[tr])
        oof[va] = model.predict_proba(x_meta[va])[:, 1]
    model = make_router(kind, seed + 1000)
    model.fit(x_meta, target)
    return oof, model.predict_proba(x_hold_meta)[:, 1]


def route_by_budget(score: np.ndarray, budget_pct: int) -> tuple[np.ndarray, float]:
    if budget_pct <= 0:
        return np.zeros(len(score), dtype=bool), float("inf")
    threshold = float(np.quantile(score, 1.0 - budget_pct / 100.0))
    return score >= threshold, threshold


def subset_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, object]:
    if not mask.any():
        return {
            f"{prefix}_n": 0,
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_balanced_accuracy": np.nan,
            f"{prefix}_f1": np.nan,
        }
    m = metric_dict(y[mask], pred[mask], prob[mask])
    return {f"{prefix}_{k}": v for k, v in m.items()}


def evaluate_policy(
    y: np.ndarray,
    base_pred: np.ndarray,
    base_prob: np.ndarray,
    corr_prob: np.ndarray,
    corr_t: float,
    routed: np.ndarray,
    is_old: np.ndarray | None = None,
    is_adapt: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    corr_pred = (corr_prob >= corr_t).astype(int)
    final_prob[routed] = corr_prob[routed]
    final_pred[routed] = corr_pred[routed]
    row = metric_dict(y, final_pred, final_prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "pass_n": int((~routed).sum()),
            "pass_acc": float((final_pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "routed_acc": float((final_pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "rescue_n": int(((base_pred != y) & (final_pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    if is_old is not None and is_adapt is not None:
        row.update(subset_metrics(y, final_pred, final_prob, is_old, "old_oof"))
        row.update(subset_metrics(y, final_pred, final_prob, is_adapt, "adapt_oof"))
    return row, final_pred, final_prob


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)
    adapt_idx, hold_idx = split_profile(third, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    x_adapt, x_hold = x_third[adapt_idx], x_third[hold_idx]
    y_old = old["label_idx"].to_numpy(dtype=int)
    y_adapt = adapt["label_idx"].to_numpy(dtype=int)
    y_hold = hold["label_idx"].to_numpy(dtype=int)

    x_train = np.concatenate([x_old, x_adapt], axis=0)
    y_train = np.concatenate([y_old, y_adapt], axis=0)
    is_old = np.concatenate([np.ones(len(old), dtype=bool), np.zeros(len(adapt), dtype=bool)])
    is_adapt = ~is_old

    cand_train: list[np.ndarray] = []
    cand_hold: list[np.ndarray] = []
    cand_names: list[str] = []
    c_values = [0.0003, 0.001, 0.003, 0.01]

    for c in c_values:
        old_oof, adapt_p, hold_p = old_only_oof_and_external(x_old, y_old, x_adapt, x_hold, c, args.seed + int(c * 1e6))
        cand_train.append(np.concatenate([old_oof, adapt_p]))
        cand_hold.append(hold_p)
        cand_names.append(f"oldonly_c{c:g}")

    for repeat_adapt in [1, 2, 4, 8]:
        train_is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])
        for c in c_values:
            p_train, p_hold = oof_and_holdout(
                x_train,
                y_train,
                train_is_adapt,
                x_hold,
                c,
                repeat_adapt,
                args.seed + repeat_adapt * 1000 + int(c * 1e6),
            )
            cand_train.append(p_train)
            cand_hold.append(p_hold)
            cand_names.append(f"adapt_r{repeat_adapt}_c{c:g}")

    train_probs = np.stack(cand_train, axis=1)
    hold_probs = np.stack(cand_hold, axis=1)
    pd.DataFrame(train_probs, columns=cand_names).assign(case_id=np.concatenate([old["case_id"], adapt["case_id"]])).to_csv(
        out / "candidate_train_oof_probs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(hold_probs, columns=cand_names).assign(case_id=hold["case_id"].to_numpy()).to_csv(
        out / "candidate_holdout_probs.csv", index=False, encoding="utf-8-sig"
    )

    base_rows = []
    base_options = []
    for idx, name in enumerate(cand_names):
        p = train_probs[:, idx]
        p_hold = hold_probs[:, idx]
        for objective in ["balanced_accuracy", "accuracy"]:
            t, _ = best_threshold(y_train, p, objective)
            pred = (p >= t).astype(int)
            hold_pred = (p_hold >= t).astype(int)
            row = {"base_name": name, "base_idx": idx, "threshold_objective": objective, "threshold": t}
            row.update({f"train_{k}": v for k, v in metric_dict(y_train, pred, p).items()})
            row.update({f"old_oof_{k}": v for k, v in metric_dict(y_train[is_old], pred[is_old], p[is_old]).items()})
            row.update({f"adapt_oof_{k}": v for k, v in metric_dict(y_train[is_adapt], pred[is_adapt], p[is_adapt]).items()})
            row.update({f"holdout_{k}": v for k, v in metric_dict(y_hold, hold_pred, p_hold).items()})
            base_rows.append(row)
            base_options.append((row, p, pred, p_hold, hold_pred))
    base_summary = pd.DataFrame(base_rows).sort_values(["train_balanced_accuracy", "old_oof_balanced_accuracy", "adapt_oof_balanced_accuracy"], ascending=False)
    base_summary.to_csv(out / "base_candidate_summary.csv", index=False, encoding="utf-8-sig")

    meta_train = build_meta_features(train_probs, cand_names)
    meta_hold = build_meta_features(hold_probs, cand_names)
    meta_train_np = meta_train.to_numpy(dtype=np.float32)
    meta_hold_np = meta_hold.to_numpy(dtype=np.float32)
    meta_train.to_csv(out / "router_meta_train_features.csv", index=False, encoding="utf-8-sig")

    # Keep the base candidates that are defensible on old OOF; otherwise the route policy
    # can select a third-specific boundary that defeats the goal of a unified model.
    eligible = base_summary[
        (base_summary["old_oof_balanced_accuracy"] >= 0.66)
        & (base_summary["adapt_oof_balanced_accuracy"] >= 0.58)
    ].head(10)
    if eligible.empty:
        eligible = base_summary.head(10)
    eligible_keys = set(zip(eligible["base_idx"].astype(int), eligible["threshold_objective"].astype(str)))

    corrector_indices = sorted(set(base_summary.head(8)["base_idx"].astype(int).tolist() + list(eligible["base_idx"].astype(int).tolist())))
    rows = []
    case_outputs: dict[str, pd.DataFrame] = {}
    router_kinds = ["route_logreg_c01", "route_logreg_c1", "route_extra_d3", "route_extra_d5"]
    budgets = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    corr_thresholds = [0.35, 0.4, 0.45, 0.49, 0.5, 0.55, 0.6, 0.65]

    for base_row, base_prob, base_pred, base_hold_prob, base_hold_pred in base_options:
        key = (int(base_row["base_idx"]), str(base_row["threshold_objective"]))
        if key not in eligible_keys:
            continue
        route_targets = {
            "base_wrong": (base_pred != y_train).astype(int),
            "base_fn_or_low_margin": (((y_train == 1) & (base_pred == 0)) | (np.abs(base_prob - 0.5) < 0.08)).astype(int),
            "low_margin": (np.abs(base_prob - 0.5) < 0.08).astype(int),
        }
        route_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "confidence_margin": (1.0 - np.abs(base_prob - 0.5) * 2.0, 1.0 - np.abs(base_hold_prob - 0.5) * 2.0)
        }
        for target_name, target in route_targets.items():
            if len(np.unique(target)) < 2:
                continue
            route_x_train = np.column_stack([meta_train_np, base_prob, np.abs(base_prob - 0.5), base_pred])
            route_x_hold = np.column_stack([meta_hold_np, base_hold_prob, np.abs(base_hold_prob - 0.5), base_hold_pred])
            for kind in router_kinds:
                rs_train, rs_hold = oof_router_scores(route_x_train, target, route_x_hold, kind, args.seed + len(route_scores) * 17)
                route_scores[f"{target_name}_{kind}"] = (rs_train, rs_hold)

        for route_name, (route_train, route_hold) in route_scores.items():
            for corr_idx in corrector_indices:
                corr_prob = train_probs[:, corr_idx]
                corr_hold_prob = hold_probs[:, corr_idx]
                corr_name = cand_names[corr_idx]
                for corr_t in corr_thresholds:
                    for budget in budgets:
                        routed_train, route_t = route_by_budget(route_train, budget)
                        train_row, _, _ = evaluate_policy(
                            y_train,
                            base_pred,
                            base_prob,
                            corr_prob,
                            corr_t,
                            routed_train,
                            is_old=is_old,
                            is_adapt=is_adapt,
                        )
                        # Selection does not use holdout. Require old-side behavior to stay usable.
                        old_bacc = float(train_row.get("old_oof_balanced_accuracy", 0.0))
                        adapt_bacc = float(train_row.get("adapt_oof_balanced_accuracy", 0.0))
                        pass_acc = float(train_row.get("pass_acc", 0.0))
                        train_row["selection_score"] = min(old_bacc, adapt_bacc) + 0.12 * float(train_row["accuracy"]) + 0.05 * pass_acc
                        train_row.update(
                            {
                                "base_name": str(base_row["base_name"]),
                                "base_idx": int(base_row["base_idx"]),
                                "base_threshold": float(base_row["threshold"]),
                                "base_threshold_objective": str(base_row["threshold_objective"]),
                                "route_name": route_name,
                                "route_threshold": route_t,
                                "budget_pct": int(budget),
                                "corrector_name": corr_name,
                                "corrector_idx": int(corr_idx),
                                "corrector_threshold": float(corr_t),
                            }
                        )
                        routed_hold = route_hold >= route_t if np.isfinite(route_t) else np.zeros(len(route_hold), dtype=bool)
                        hold_row, hold_final_pred, hold_final_prob = evaluate_policy(
                            y_hold,
                            base_hold_pred,
                            base_hold_prob,
                            corr_hold_prob,
                            corr_t,
                            routed_hold,
                        )
                        train_row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                        rows.append(train_row)

    policy_summary = pd.DataFrame(rows).sort_values(
        ["selection_score", "old_oof_balanced_accuracy", "adapt_oof_balanced_accuracy", "accuracy", "net_rescue"],
        ascending=False,
    )
    policy_summary.to_csv(out / "two_stage_policy_summary.csv", index=False, encoding="utf-8-sig")

    selected = policy_summary.iloc[0].to_dict()
    base_idx = int(selected["base_idx"])
    corr_idx = int(selected["corrector_idx"])
    base_prob = train_probs[:, base_idx]
    base_hold_prob = hold_probs[:, base_idx]
    base_pred = (base_prob >= float(selected["base_threshold"])).astype(int)
    base_hold_pred = (base_hold_prob >= float(selected["base_threshold"])).astype(int)
    corr_prob = train_probs[:, corr_idx]
    corr_hold_prob = hold_probs[:, corr_idx]
    route_name = str(selected["route_name"])

    # Recompute selected route score.
    if route_name == "confidence_margin":
        route_train = 1.0 - np.abs(base_prob - 0.5) * 2.0
        route_hold = 1.0 - np.abs(base_hold_prob - 0.5) * 2.0
    else:
        target_kind, router_kind = route_name.rsplit("_route_", 1)
        target_kind = target_kind
        router_kind = "route_" + router_kind
        target_map = {
            "base_wrong": (base_pred != y_train).astype(int),
            "base_fn_or_low_margin": (((y_train == 1) & (base_pred == 0)) | (np.abs(base_prob - 0.5) < 0.08)).astype(int),
            "low_margin": (np.abs(base_prob - 0.5) < 0.08).astype(int),
        }
        route_x_train = np.column_stack([meta_train_np, base_prob, np.abs(base_prob - 0.5), base_pred])
        route_x_hold = np.column_stack([meta_hold_np, base_hold_prob, np.abs(base_hold_prob - 0.5), base_hold_pred])
        route_train, route_hold = oof_router_scores(route_x_train, target_map[target_kind], route_x_hold, router_kind, args.seed + 999)
    routed_train, route_t = route_by_budget(route_train, int(selected["budget_pct"]))
    routed_hold = route_hold >= route_t if np.isfinite(route_t) else np.zeros(len(route_hold), dtype=bool)
    train_eval, train_final_pred, train_final_prob = evaluate_policy(
        y_train,
        base_pred,
        base_prob,
        corr_prob,
        float(selected["corrector_threshold"]),
        routed_train,
        is_old=is_old,
        is_adapt=is_adapt,
    )
    hold_eval, hold_final_pred, hold_final_prob = evaluate_policy(
        y_hold,
        base_hold_pred,
        base_hold_prob,
        corr_hold_prob,
        float(selected["corrector_threshold"]),
        routed_hold,
    )

    train_case = pd.concat(
        [
            old[["case_id", "label_idx"]].assign(source_split="old_oof"),
            adapt[["case_id", "label_idx"]].assign(source_split="third_adapt_oof"),
        ],
        ignore_index=True,
    )
    train_case["base_prob_high"] = base_prob
    train_case["base_pred_idx"] = base_pred
    train_case["route_score"] = route_train
    train_case["routed_to_reviewer"] = routed_train.astype(int)
    train_case["reviewer_prob_high"] = corr_prob
    train_case["final_prob_high"] = train_final_prob
    train_case["final_pred_idx"] = train_final_pred
    train_case["final_correct"] = (train_final_pred == y_train).astype(int)
    train_case.to_csv(out / "selected_train_oof_case_predictions.csv", index=False, encoding="utf-8-sig")

    hold_case = hold[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]].copy()
    hold_case["base_prob_high"] = base_hold_prob
    hold_case["base_pred_idx"] = base_hold_pred
    hold_case["route_score"] = route_hold
    hold_case["routed_to_reviewer"] = routed_hold.astype(int)
    hold_case["reviewer_prob_high"] = corr_hold_prob
    hold_case["final_prob_high"] = hold_final_prob
    hold_case["final_pred_idx"] = hold_final_pred
    hold_case["final_correct"] = (hold_final_pred == y_hold).astype(int)
    hold_case.to_csv(out / "selected_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")
    subtype = hold_case.groupby("task_l6_label").agg(n=("case_id", "size"), correct=("final_correct", "sum"), accuracy=("final_correct", "mean"), routed=("routed_to_reviewer", "sum")).reset_index()
    subtype.to_csv(out / "selected_holdout_metrics_by_subtype.csv", index=False, encoding="utf-8-sig")
    split_summary = pd.DataFrame(
        [
            {"split": "old_train_oof", "n": int(is_old.sum()), **{k: v for k, v in subset_metrics(y_train, train_final_pred, train_final_prob, is_old, "metric").items()}},
            {"split": "third_adapt_oof", "n": int(is_adapt.sum()), **{k: v for k, v in subset_metrics(y_train, train_final_pred, train_final_prob, is_adapt, "metric").items()}},
            {"split": "third_holdout", "n": int(len(y_hold)), **{f"metric_{k}": v for k, v in hold_eval.items()}},
        ]
    )
    split_summary.to_csv(out / "selected_policy_split_metrics.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": {
            "old_data_used": int(len(old)),
            "third_adapt_used": int(len(adapt)),
            "third_holdout_locked": int(len(hold)),
            "selection_uses_holdout": False,
            "model_type": "two-stage frozen-feature domain-adapted router/reviewer",
        },
        "selected_policy_train_oof": train_eval,
        "selected_policy_holdout": hold_eval,
        "selected_policy": selected,
        "holdout_subtype": subtype.to_dict("records"),
    }
    (out / "two_stage_adapted_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected policy")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTop policies")
    print(policy_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
