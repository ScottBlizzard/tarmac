from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "outputs" / "grosspath_rc_v0_20260526"
OUT = ROOT / "outputs" / "grosspath_rc_v1_20260526"
REPORT_DIR = ROOT / "汇报"


PROB_FEATURES = ["prob162_blend", "prob103_vitl", "prob107_qkvb"]
ROUTER_FEATURES = [
    "prob162_blend",
    "prob103_vitl",
    "prob107_qkvb",
    "prob_mean_core",
    "margin162",
    "margin_mean_core",
    "core_prob_std",
    "core_prob_range",
    "core_agree_count",
    "core_agree_all",
    "abs_162_103",
    "abs_162_107",
    "abs_103_107",
    "pred_for_prob162_blend",
    "pred_for_prob103_vitl",
    "pred_for_prob107_qkvb",
]


@dataclass
class ThresholdSpec:
    score_name: str
    target_acc: float
    max_auto_low_high_miss: float | None
    min_coverage: float = 0.05

    @property
    def name(self) -> str:
        miss = "nomisscap" if self.max_auto_low_high_miss is None else f"miss{int(self.max_auto_low_high_miss * 100)}"
        return f"{self.score_name}_acc{int(self.target_acc * 100)}_{miss}"


def ensure_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(V0 / "dev_model_behavior_table.csv")
    ext = pd.read_csv(V0 / "external_model_behavior_table.csv")
    for df in (dev, ext):
        add_common_features(df)
    return dev, ext


def add_common_features(df: pd.DataFrame) -> None:
    for c in PROB_FEATURES:
        if c not in df.columns:
            raise KeyError(f"missing {c}")
    df["prob_mean_core"] = df[PROB_FEATURES].mean(axis=1)
    df["abs_162_103"] = (df["prob162_blend"] - df["prob103_vitl"]).abs()
    df["abs_162_107"] = (df["prob162_blend"] - df["prob107_qkvb"]).abs()
    df["abs_103_107"] = (df["prob103_vitl"] - df["prob107_qkvb"]).abs()
    if "pred_for_prob162_blend" not in df.columns:
        df["pred_for_prob162_blend"] = (df["prob162_blend"] >= 0.595).astype(int)
    if "pred_for_prob103_vitl" not in df.columns:
        df["pred_for_prob103_vitl"] = (df["prob103_vitl"] >= 0.5).astype(int)
    if "pred_for_prob107_qkvb" not in df.columns:
        df["pred_for_prob107_qkvb"] = (df["prob107_qkvb"] >= 0.5).astype(int)
    preds = df[["pred_for_prob162_blend", "pred_for_prob103_vitl", "pred_for_prob107_qkvb"]].astype(int)
    df["core_agree_count"] = preds.apply(lambda r: int(r.value_counts().max()), axis=1)
    df["core_agree_all"] = (df["core_agree_count"] == 3).astype(int)
    df["core_agree_162_103"] = preds["pred_for_prob162_blend"].eq(preds["pred_for_prob103_vitl"]).astype(int)
    df["core_agree_162_107"] = preds["pred_for_prob162_blend"].eq(preds["pred_for_prob107_qkvb"]).astype(int)
    df["margin162"] = (df["prob162_blend"] - 0.595).abs()
    df["margin_mean_core"] = (df["prob_mean_core"] - 0.5).abs()
    df["core_prob_std"] = df[PROB_FEATURES].std(axis=1)
    df["core_prob_range"] = df[PROB_FEATURES].max(axis=1) - df[PROB_FEATURES].min(axis=1)
    df["score_margin_agree"] = (
        0.45 * z01(df["margin162"])
        + 0.25 * z01(df["margin_mean_core"])
        + 0.20 * (df["core_agree_count"] / 3.0)
        - 0.10 * z01(df["core_prob_range"])
    )
    if "reliability_simple" in df.columns:
        df["score_v0_simple"] = df["reliability_simple"].astype(float)
    else:
        df["score_v0_simple"] = df["score_margin_agree"]


def z01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo = float(s.min())
    hi = float(s.max())
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def safe_bacc(y: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, pred))


def select_class_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.r_[0.01, 0.05, np.linspace(0.1, 0.9, 161), 0.95, 0.99, prob])
    best: tuple[float, float, float] | None = None
    best_metrics: dict[str, float] = {}
    for thr in candidates:
        pred = (prob >= thr).astype(int)
        bacc = safe_bacc(y, pred)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, zero_division=0))
        score = bacc + 0.001 * acc
        if best is None or score > best[0]:
            best = (score, float(thr), bacc)
            best_metrics = {"acc": acc, "bacc": bacc, "f1": f1}
    assert best is not None
    return best[1], best_metrics


def train_stackers(dev: pd.DataFrame, ext: pd.DataFrame) -> dict[str, dict[str, object]]:
    y = dev["label_idx"].astype(int).to_numpy()
    folds = sorted(pd.Series(dev["fold_id"]).dropna().astype(int).unique().tolist())
    out: dict[str, dict[str, object]] = {}
    configs = {
        "stack_plain": LogisticRegression(max_iter=3000, solver="lbfgs"),
        "stack_balanced": LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced"),
    }
    x_dev = dev[PROB_FEATURES].astype(float).to_numpy()
    x_ext = ext[PROB_FEATURES].astype(float).to_numpy()

    for name, lr in configs.items():
        oof = np.zeros(len(dev), dtype=float)
        for fold in folds:
            train_mask = dev["fold_id"].astype(int).ne(fold).to_numpy()
            valid_mask = ~train_mask
            model = make_pipeline(StandardScaler(), lr.__class__(**lr.get_params()))
            model.fit(x_dev[train_mask], y[train_mask])
            oof[valid_mask] = model.predict_proba(x_dev[valid_mask])[:, 1]
        full_model = make_pipeline(StandardScaler(), lr.__class__(**lr.get_params()))
        full_model.fit(x_dev, y)
        ext_prob = full_model.predict_proba(x_ext)[:, 1]
        thr, thr_metrics = select_class_threshold(y, oof)
        dev[f"prob_{name}"] = oof
        ext[f"prob_{name}"] = ext_prob
        dev[f"pred_{name}"] = (dev[f"prob_{name}"] >= thr).astype(int)
        ext[f"pred_{name}"] = (ext[f"prob_{name}"] >= thr).astype(int)
        out[name] = {"threshold": thr, "dev_threshold_metrics": thr_metrics}

    # Frozen base thresholds for direct comparison.
    base_thr, base_metrics = select_class_threshold(y, dev["prob162_blend"].astype(float).to_numpy())
    mean_thr, mean_metrics = select_class_threshold(y, dev["prob_mean_core"].astype(float).to_numpy())
    for df in (dev, ext):
        df["prob_base162"] = df["prob162_blend"].astype(float)
        df["pred_base162"] = (df["prob_base162"] >= base_thr).astype(int)
        df["prob_mean_core"] = df["prob_mean_core"].astype(float)
        df["pred_mean_core"] = (df["prob_mean_core"] >= mean_thr).astype(int)
    out["base162"] = {"threshold": base_thr, "dev_threshold_metrics": base_metrics}
    out["mean_core"] = {"threshold": mean_thr, "dev_threshold_metrics": mean_metrics}
    return out


def metric_from_pred(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    if len(y) == 0:
        return {
            "n": 0,
            "acc": float("nan"),
            "bacc": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "sens_high": float("nan"),
            "spec_low": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "bacc": safe_bacc(y, pred),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": safe_auc(y, prob) if prob is not None else float("nan"),
        "sens_high": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "spec_low": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def group_masks_dev(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "dev_all_old_plus_third": pd.Series(True, index=df.index),
        "old": df["domain"].eq("old"),
        "third_all": df["domain"].eq("third"),
        "third_adapt72": df["third_split"].eq("adapt72"),
        "third_holdout234": df["third_split"].eq("holdout234"),
    }


def group_masks_external(df: pd.DataFrame) -> dict[str, pd.Series]:
    quality = df.get("manual_quality_status_v1", pd.Series("", index=df.index)).fillna("")
    return {
        "external_all": pd.Series(True, index=df.index),
        "external_strict": df["strict_task7_eval"].astype(int).eq(1),
        "external_readable_auto": quality.eq("pass_readable"),
    }


def eval_forced_models(dev: pd.DataFrame, ext: pd.DataFrame, stack_meta: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = ["base162", "mean_core", "stack_plain", "stack_balanced"]
    for dataset_name, df, masks in [
        ("dev", dev, group_masks_dev(dev)),
        ("external", ext, group_masks_external(ext)),
    ]:
        for group, mask in masks.items():
            sub = df[mask].copy()
            y = sub["label_idx"].astype(int).to_numpy()
            for model_name in models:
                pred = sub[f"pred_{model_name}"].astype(int).to_numpy()
                prob = sub[f"prob_{model_name}"].astype(float).to_numpy()
                m = metric_from_pred(y, pred, prob)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "group": group,
                        "model": model_name,
                        "threshold_from_dev": stack_meta[model_name]["threshold"],
                        **m,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "v1_forced_model_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def choose_main_model(forced: pd.DataFrame) -> str:
    sub = forced[forced["group"].eq("dev_all_old_plus_third")].copy()
    # Prefer balanced accuracy. If tied, prefer simpler base model.
    order = {"base162": 0, "stack_plain": 1, "stack_balanced": 2, "mean_core": 3}
    sub["simplicity"] = sub["model"].map(order).fillna(9)
    sub = sub.sort_values(["bacc", "acc", "simplicity"], ascending=[False, False, True])
    return str(sub.iloc[0]["model"])


def train_guard(dev: pd.DataFrame, ext: pd.DataFrame, main_model: str) -> dict[str, object]:
    pred_col = f"pred_{main_model}"
    prob_col = f"prob_{main_model}"
    dev["main_pred"] = dev[pred_col].astype(int)
    dev["main_prob"] = dev[prob_col].astype(float)
    ext["main_pred"] = ext[pred_col].astype(int)
    ext["main_prob"] = ext[prob_col].astype(float)
    dev["main_margin"] = (dev["main_prob"] - 0.5).abs()
    ext["main_margin"] = (ext["main_prob"] - 0.5).abs()
    feature_cols = ROUTER_FEATURES + ["main_prob", "main_pred", "main_margin"]
    y_correct = dev["main_pred"].eq(dev["label_idx"].astype(int)).astype(int).to_numpy()
    folds = sorted(pd.Series(dev["fold_id"]).dropna().astype(int).unique().tolist())
    x_dev = dev[feature_cols].astype(float).fillna(0.0).to_numpy()
    x_ext = ext[feature_cols].astype(float).fillna(0.0).to_numpy()

    configs = {
        "guard_plain": LogisticRegression(max_iter=3000, solver="lbfgs"),
        "guard_balanced": LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced"),
    }
    meta: dict[str, object] = {"feature_cols": feature_cols}
    for name, lr in configs.items():
        oof = np.zeros(len(dev), dtype=float)
        for fold in folds:
            train_mask = dev["fold_id"].astype(int).ne(fold).to_numpy()
            valid_mask = ~train_mask
            model = make_pipeline(StandardScaler(), lr.__class__(**lr.get_params()))
            model.fit(x_dev[train_mask], y_correct[train_mask])
            oof[valid_mask] = model.predict_proba(x_dev[valid_mask])[:, 1]
        full_model = make_pipeline(StandardScaler(), lr.__class__(**lr.get_params()))
        full_model.fit(x_dev, y_correct)
        ext_score = full_model.predict_proba(x_ext)[:, 1]
        dev[name] = oof
        ext[name] = ext_score
        meta[name] = {"dev_auc_correct": safe_auc(y_correct, oof)}

    dev["guard_margin_agree"] = dev["score_margin_agree"].astype(float)
    ext["guard_margin_agree"] = ext["score_margin_agree"].astype(float)
    dev["guard_all3"] = dev["core_agree_all"].astype(float)
    ext["guard_all3"] = ext["core_agree_all"].astype(float)
    return meta


def eval_policy(
    df: pd.DataFrame,
    group: str,
    policy: str,
    auto_mask: pd.Series,
    review_mask: pd.Series,
    retake_mask: pd.Series,
) -> dict[str, object]:
    y = df["label_idx"].astype(int)
    pred = df["main_pred"].astype(int)
    auto_y = y[auto_mask].to_numpy(int)
    auto_pred = pred[auto_mask].to_numpy(int)
    m = metric_from_pred(auto_y, auto_pred)
    total = len(df)
    auto_low = auto_mask & pred.eq(0)
    auto_high = auto_mask & pred.eq(1)
    review_or_retake = review_mask | retake_mask
    base_error_review = (
        float(pred[review_mask].ne(y[review_mask]).mean()) if int(review_mask.sum()) else float("nan")
    )
    base_error_retake = (
        float(pred[retake_mask].ne(y[retake_mask]).mean()) if int(retake_mask.sum()) else float("nan")
    )
    high_total = int(y.eq(1).sum())
    high_auto_low = int((y.eq(1) & auto_low).sum())
    high_auto_high = int((y.eq(1) & auto_high).sum())
    high_review_or_retake = int((y.eq(1) & review_or_retake).sum())
    auto_metrics = {
        "auto_n_metric": m["n"],
        "auto_accuracy": m["acc"],
        "auto_balanced_accuracy": m["bacc"],
        "auto_f1": m["f1"],
        "auto_auc": m["auc"],
        "auto_sensitivity_high": m["sens_high"],
        "auto_specificity_low": m["spec_low"],
        "auto_tn": m["tn"],
        "auto_fp": m["fp"],
        "auto_fn": m["fn"],
        "auto_tp": m["tp"],
    }
    return {
        "group": group,
        "policy": policy,
        "total_n": int(total),
        "auto_n": int(auto_mask.sum()),
        "review_n": int(review_mask.sum()),
        "retake_n": int(retake_mask.sum()),
        "auto_coverage": float(auto_mask.sum() / total) if total else float("nan"),
        "review_rate": float(review_mask.sum() / total) if total else float("nan"),
        "retake_rate": float(retake_mask.sum() / total) if total else float("nan"),
        "auto_low_n": int(auto_low.sum()),
        "auto_high_n": int(auto_high.sum()),
        "auto_low_high_miss_rate": float(y[auto_low].eq(1).mean()) if int(auto_low.sum()) else float("nan"),
        "auto_high_ppv": float(y[auto_high].eq(1).mean()) if int(auto_high.sum()) else float("nan"),
        "review_error_rate_main": base_error_review,
        "retake_error_rate_main": base_error_retake,
        "high_total": high_total,
        "high_auto_low_missed": high_auto_low,
        "high_auto_high": high_auto_high,
        "high_review_or_retake": high_review_or_retake,
        "high_risk_not_auto_low_rate": float((high_total - high_auto_low) / high_total) if high_total else float("nan"),
        **auto_metrics,
    }


def policy_metrics_for_threshold(df: pd.DataFrame, score: str, thr: float) -> dict[str, object]:
    auto = df[score].astype(float).ge(thr)
    review = ~auto
    retake = pd.Series(False, index=df.index)
    return eval_policy(df, "dev_select", f"{score}@{thr:.6f}", auto, review, retake)


def select_guard_threshold(dev: pd.DataFrame, spec: ThresholdSpec) -> dict[str, object]:
    scores = dev[spec.score_name].astype(float).to_numpy()
    candidates = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 201)), scores])
    rows: list[dict[str, object]] = []
    for thr in candidates:
        row = policy_metrics_for_threshold(dev, spec.score_name, float(thr))
        row["threshold"] = float(thr)
        rows.append(row)
    grid = pd.DataFrame(rows)
    min_n = max(10, int(np.ceil(len(dev) * spec.min_coverage)))
    ok = grid[grid["auto_n"].ge(min_n)].copy()
    ok = ok[ok["auto_accuracy"].ge(spec.target_acc)].copy()
    if spec.max_auto_low_high_miss is not None:
        ok = ok[
            ok["auto_low_high_miss_rate"].fillna(0.0).le(spec.max_auto_low_high_miss)
        ].copy()
    if not ok.empty:
        chosen = ok.sort_values(["auto_coverage", "auto_accuracy"], ascending=[False, False]).iloc[0]
        reason = "target_met"
    else:
        eligible = grid[grid["auto_n"].ge(min_n)].copy()
        eligible["objective"] = (
            eligible["auto_accuracy"].fillna(0.0)
            - 0.5 * eligible["auto_low_high_miss_rate"].fillna(0.0)
            + 0.05 * eligible["auto_coverage"].fillna(0.0)
        )
        chosen = eligible.sort_values(["objective", "auto_accuracy"], ascending=[False, False]).iloc[0]
        reason = "fallback_best_available"
    out = chosen.to_dict()
    out.update(
        {
            "spec_name": spec.name,
            "score_name": spec.score_name,
            "target_acc": spec.target_acc,
            "max_auto_low_high_miss": spec.max_auto_low_high_miss,
            "selection_reason": reason,
        }
    )
    return out


def evaluate_v1_policies(dev: pd.DataFrame, ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [
        ThresholdSpec("guard_plain", 0.90, 0.10),
        ThresholdSpec("guard_plain", 0.92, 0.10),
        ThresholdSpec("guard_plain", 0.95, 0.10),
        ThresholdSpec("guard_balanced", 0.90, 0.10),
        ThresholdSpec("guard_margin_agree", 0.90, 0.10),
    ]
    threshold_rows = [select_guard_threshold(dev, spec) for spec in specs]
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT / "v1_guard_thresholds_from_dev.csv", index=False, encoding="utf-8-sig")

    def policies(df: pd.DataFrame, quality_gate: bool) -> list[tuple[str, pd.Series, pd.Series, pd.Series]]:
        false = pd.Series(False, index=df.index)
        true = pd.Series(True, index=df.index)
        out: list[tuple[str, pd.Series, pd.Series, pd.Series]] = []
        out.append(("forced_main_all", true, false, false))
        all3 = df["core_agree_all"].astype(int).eq(1)
        agree_162_103 = df["core_agree_162_103"].astype(int).eq(1)
        agree_162_107 = df["core_agree_162_107"].astype(int).eq(1)
        out.append(("consensus_all3", all3, ~all3, false))
        out.append(("consensus_162_103", agree_162_103, ~agree_162_103, false))
        out.append(("consensus_162_107", agree_162_107, ~agree_162_107, false))
        for _, r in thresholds.iterrows():
            score = str(r["score_name"])
            name = str(r["spec_name"])
            thr = float(r["threshold"])
            auto = df[score].astype(float).ge(thr)
            out.append((f"v1_{name}", auto, ~auto, false))
        if quality_gate and "manual_quality_status_v1" in df.columns:
            q = df["manual_quality_status_v1"].fillna("").astype(str)
            retake = q.eq("reject_retake")
            readable = q.eq("pass_readable")
            borderline = q.eq("borderline_review")
            for _, r in thresholds.iterrows():
                score = str(r["score_name"])
                name = str(r["spec_name"])
                thr = float(r["threshold"])
                auto = readable & df[score].astype(float).ge(thr)
                review = (readable & ~df[score].astype(float).ge(thr)) | borderline | (~readable & ~retake)
                out.append((f"quality_v1_{name}", auto, review, retake))
        return out

    dev_rows: list[dict[str, object]] = []
    for group, mask in group_masks_dev(dev).items():
        sub = dev[mask].copy()
        for name, auto, review, retake in policies(sub, quality_gate=False):
            dev_rows.append(eval_policy(sub, group, name, auto, review, retake))

    ext_rows: list[dict[str, object]] = []
    for group, mask in group_masks_external(ext).items():
        sub = ext[mask].copy()
        for name, auto, review, retake in policies(sub, quality_gate=True):
            ext_rows.append(eval_policy(sub, group, name, auto, review, retake))

    dev_policy = pd.DataFrame(dev_rows)
    ext_policy = pd.DataFrame(ext_rows)
    dev_policy.to_csv(OUT / "v1_workflow_policy_metrics_dev.csv", index=False, encoding="utf-8-sig")
    ext_policy.to_csv(OUT / "v1_workflow_policy_metrics_external.csv", index=False, encoding="utf-8-sig")
    return thresholds, dev_policy, ext_policy


def make_risk_coverage_curve(df: pd.DataFrame, score: str, name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scores = df[score].astype(float).to_numpy()
    for q in np.linspace(0, 0.95, 96):
        thr = float(np.quantile(scores, q))
        row = policy_metrics_for_threshold(df, score, thr)
        row["score"] = score
        row["quantile_cut"] = float(q)
        row["threshold"] = thr
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"v1_risk_coverage_{name}_{score}.csv", index=False, encoding="utf-8-sig")
    return out


def plot_policies(dev_policy: pd.DataFrame, ext_policy: pd.DataFrame) -> None:
    plot_one(
        dev_policy,
        "dev_all_old_plus_third",
        OUT / "v1_policy_dev_all.png",
        [
            "forced_main_all",
            "consensus_all3",
            "v1_guard_plain_acc90_miss10",
            "v1_guard_plain_acc92_miss10",
            "v1_guard_balanced_acc90_miss10",
        ],
    )
    plot_one(
        dev_policy,
        "third_holdout234",
        OUT / "v1_policy_third_holdout234.png",
        [
            "forced_main_all",
            "consensus_all3",
            "v1_guard_plain_acc90_miss10",
            "v1_guard_plain_acc92_miss10",
            "v1_guard_balanced_acc90_miss10",
        ],
    )
    plot_one(
        ext_policy,
        "external_strict",
        OUT / "v1_policy_external_strict.png",
        [
            "forced_main_all",
            "consensus_all3",
            "v1_guard_plain_acc90_miss10",
            "v1_guard_plain_acc92_miss10",
            "quality_v1_guard_plain_acc90_miss10",
            "quality_v1_guard_balanced_acc90_miss10",
        ],
    )


def plot_one(df: pd.DataFrame, group: str, out_path: Path, keep: list[str]) -> None:
    sub = df[df["group"].eq(group) & df["policy"].isin(keep)].copy()
    if sub.empty:
        return
    sub["order"] = sub["policy"].map({p: i for i, p in enumerate(keep)})
    sub = sub.sort_values("order")
    labels = sub["policy"].str.replace("_", "\n", regex=False).tolist()
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    width = 0.36
    ax.bar(x - width / 2, sub["auto_coverage"], width=width, label="auto coverage", color="#5b8ea8")
    ax.bar(x + width / 2, sub["auto_accuracy"], width=width, label="auto accuracy", color="#c7795c")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"GrossPath-RC v1 policy comparison: {group}")
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")
    for i, row in sub.reset_index(drop=True).iterrows():
        ax.text(i - width / 2, row["auto_coverage"] + 0.015, f"{row['auto_coverage']:.2f}", ha="center", fontsize=8)
        if pd.notna(row["auto_accuracy"]):
            ax.text(i + width / 2, row["auto_accuracy"] + 0.015, f"{row['auto_accuracy']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def format_float(x: object) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        return f"{x:.4f}"
    return str(x)


def df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_float(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    forced: pd.DataFrame,
    thresholds: pd.DataFrame,
    dev_policy: pd.DataFrame,
    ext_policy: pd.DataFrame,
    main_model: str,
    guard_meta: dict[str, object],
) -> None:
    selected_cols_forced = [
        "group",
        "model",
        "threshold_from_dev",
        "n",
        "acc",
        "bacc",
        "auc",
        "sens_high",
        "spec_low",
        "fn",
        "fp",
    ]
    selected_cols_policy = [
        "group",
        "policy",
        "total_n",
        "auto_n",
        "review_n",
        "retake_n",
        "auto_coverage",
        "auto_accuracy",
        "auto_balanced_accuracy",
        "auto_sensitivity_high",
        "auto_specificity_low",
        "auto_low_high_miss_rate",
        "auto_high_ppv",
        "review_error_rate_main",
        "high_auto_low_missed",
        "high_review_or_retake",
    ]
    dev_core = dev_policy[
        dev_policy["group"].isin(["dev_all_old_plus_third", "third_holdout234"])
        & dev_policy["policy"].isin(
            [
                "forced_main_all",
                "consensus_all3",
                "v1_guard_plain_acc90_miss10",
                "v1_guard_plain_acc92_miss10",
                "v1_guard_balanced_acc90_miss10",
                "v1_guard_margin_agree_acc90_miss10",
            ]
        )
    ][selected_cols_policy].copy()
    ext_core = ext_policy[
        ext_policy["group"].isin(["external_strict", "external_readable_auto"])
        & ext_policy["policy"].isin(
            [
                "forced_main_all",
                "consensus_all3",
                "v1_guard_plain_acc90_miss10",
                "v1_guard_plain_acc92_miss10",
                "v1_guard_balanced_acc90_miss10",
                "quality_v1_guard_plain_acc90_miss10",
                "quality_v1_guard_balanced_acc90_miss10",
            ]
        )
    ][selected_cols_policy].copy()
    forced_core = forced[
        forced["group"].isin(["dev_all_old_plus_third", "old", "third_holdout234", "external_strict", "external_readable_auto"])
    ][selected_cols_forced].copy()
    thr_core = thresholds[
        [
            "spec_name",
            "score_name",
            "threshold",
            "target_acc",
            "max_auto_low_high_miss",
            "selection_reason",
            "auto_n",
            "auto_coverage",
            "auto_accuracy",
            "auto_balanced_accuracy",
            "auto_low_high_miss_rate",
        ]
    ].copy()

    report = f"""# GrossPath-RC v1 冻结式工作流实验

日期：2026-05-26

## 本轮目的

v1 的目标是把 v0 的分散结果收敛成更正式的冻结式 workflow。我们只用旧数据+第三批开发集训练或选择策略，然后把同一套策略直接套到第三批 holdout 和严格外部集，不根据外部集分数反向调参。

本轮新增两件事：

1. 三路模型概率 stacking：用 `prob162_blend`、`DINOv3/VitL`、`QKVB` 的 OOF 概率训练轻量 stacking 分类头。
2. 可靠性 guard：用开发集 OOF 训练一个“这次主模型是否可靠”的评分器，再用开发集选择自动放行阈值。

当前 v1 自动选择的主分类器是 `{main_model}`。可靠性 guard 的 OOF 正确性识别 AUC：`guard_plain={guard_meta['guard_plain']['dev_auc_correct']:.4f}`，`guard_balanced={guard_meta['guard_balanced']['dev_auc_correct']:.4f}`。

## 强制分类模型对照

{df_to_md(forced_core)}

## 开发集选择出的 guard 阈值

{df_to_md(thr_core)}

## 开发集和第三批 holdout workflow

{df_to_md(dev_core)}

## 外部集冻结评估 workflow

{df_to_md(ext_core)}

## 阶段判断

1. stacking 分类头没有超过 base162。它在开发集和第三批 holdout 上与 base162 基本同分，在外部严格集反而略差，所以这一版不能作为主结果。
2. mean_core 在外部严格集的强制分类比 base162 更平衡，但它在开发集和第三批 holdout 明显下降。这个现象提示 DINOv3/QKVB 分支对外部高危更敏感，但不能直接替代主模型。
3. v1 guard 在开发集内能筛出 90% 以上准确率的自动子集，但冻结到外部集后明显失效。也就是说，v1 guard 是一次有价值的否证实验，不应包装成提分结果。
4. 当前正式可用的外部风险控制信号仍然是多模型一致性，而不是域内训练出来的可靠性评分器。
5. 外部集的质量门控 policy 只使用图像质量判断，不使用真值；它适合写成“安全工作流”，不适合被包装成全量诊断准确率。

## 下一步

下一步不应该继续盲目调阈值，而应该做两个更硬的模块：一是图像质量/视角/尺度的域泛化训练，二是少数核心概念的图像蒸馏，特别是边界、包膜、结节/分叶、囊变坏死和主体尺度。v1 的结果说明，外部泛化问题不是靠域内置信度校准就能解决的。
"""
    report_path = REPORT_DIR / "2026-05-26_GrossPath-RC_v1冻结式工作流实验报告.md"
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dir()
    dev, ext = read_tables()
    stack_meta = train_stackers(dev, ext)
    forced = eval_forced_models(dev, ext, stack_meta)
    main_model = choose_main_model(forced)
    guard_meta = train_guard(dev, ext, main_model)
    thresholds, dev_policy, ext_policy = evaluate_v1_policies(dev, ext)
    make_risk_coverage_curve(dev, "guard_plain", "dev")
    make_risk_coverage_curve(ext[ext["strict_task7_eval"].astype(int).eq(1)].copy(), "guard_plain", "external_strict")
    plot_policies(dev_policy, ext_policy)

    dev.to_csv(OUT / "v1_dev_scores.csv", index=False, encoding="utf-8-sig")
    ext.to_csv(OUT / "v1_external_scores.csv", index=False, encoding="utf-8-sig")
    summary = {
        "main_model": main_model,
        "stack_meta": stack_meta,
        "guard_meta": guard_meta,
        "outputs": {
            "forced_metrics": str(OUT / "v1_forced_model_metrics.csv"),
            "thresholds": str(OUT / "v1_guard_thresholds_from_dev.csv"),
            "dev_policy": str(OUT / "v1_workflow_policy_metrics_dev.csv"),
            "external_policy": str(OUT / "v1_workflow_policy_metrics_external.csv"),
        },
    }
    (OUT / "v1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(forced, thresholds, dev_policy, ext_policy, main_model, guard_meta)


if __name__ == "__main__":
    main()
