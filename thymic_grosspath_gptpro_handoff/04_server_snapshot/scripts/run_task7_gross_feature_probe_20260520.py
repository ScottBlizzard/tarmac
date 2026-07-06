from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


TASK7_CLASS_NAMES = ("low_risk_group", "high_risk_group")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 structured gross-finding feature probe.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--split-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="balanced_accuracy", choices=("accuracy", "balanced_accuracy", "f1"))
    return parser.parse_args()


def safe_float(text: object) -> float:
    try:
        return float(str(text).strip())
    except Exception:
        return float("nan")


def has_any(text: str, words: list[str]) -> int:
    return int(any(word in text for word in words))


def count_any(text: str, words: list[str]) -> int:
    return int(sum(text.count(word) for word in words))


def unit_factor(unit: str | None, values: list[float]) -> float:
    if unit and unit.lower() == "cm":
        return 10.0
    if unit and unit.lower() == "mm":
        return 1.0
    # Most reports use mm when omitted; keep conservative to avoid exploding sizes.
    return 1.0


def parse_size_candidates(text: str) -> list[tuple[list[float], int]]:
    candidates: list[tuple[list[float], int]] = []
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*[*×xX]\s*(\d+(?:\.\d+)?)"
        r"(?:\s*[*×xX]\s*(\d+(?:\.\d+)?))?\s*(cm|mm|CM|MM)?"
    )
    for m in pattern.finditer(text):
        values = [float(m.group(1)), float(m.group(2))]
        if m.group(3) is not None:
            values.append(float(m.group(3)))
        factor = unit_factor(m.group(4), values)
        candidates.append(([v * factor for v in values], m.start()))
    for m in re.finditer(r"直径\s*(\d+(?:\.\d+)?)\s*(cm|mm|CM|MM)?", text):
        value = float(m.group(1))
        factor = unit_factor(m.group(2), [value])
        candidates.append(([value * factor], m.start()))
    return candidates


def size_stats(text: str) -> dict[str, float]:
    candidates = parse_size_candidates(text)
    all_dims = [dims for dims, _ in candidates]
    tumor_dims = []
    for dims, start in candidates:
        window = text[max(0, start - 24) : min(len(text), start + 12)]
        if any(key in window for key in ["肿物", "肿块", "肿瘤", "结节", "病灶", "包块"]):
            if not any(key in window for key in ["脂肪", "肺", "心包", "胸腺组织", "组织大小"]):
                tumor_dims.append(dims)

    def summarize(prefix: str, dims_list: list[list[float]]) -> dict[str, float]:
        out = {
            f"{prefix}_n_size_mentions": float(len(dims_list)),
            f"{prefix}_max_dim_mm": 0.0,
            f"{prefix}_max_area_mm2": 0.0,
            f"{prefix}_max_volume_mm3": 0.0,
        }
        for dims in dims_list:
            if not dims:
                continue
            out[f"{prefix}_max_dim_mm"] = max(out[f"{prefix}_max_dim_mm"], max(dims))
            if len(dims) >= 2:
                out[f"{prefix}_max_area_mm2"] = max(out[f"{prefix}_max_area_mm2"], dims[0] * dims[1])
            if len(dims) >= 3:
                out[f"{prefix}_max_volume_mm3"] = max(out[f"{prefix}_max_volume_mm3"], dims[0] * dims[1] * dims[2])
        return out

    out = summarize("all", all_dims)
    out.update(summarize("tumor", tumor_dims))
    return out


def extract_gross_features(df: pd.DataFrame) -> pd.DataFrame:
    text_col = "肉眼所见"
    feat = pd.DataFrame(index=df.index)
    text = df[text_col].fillna("").astype(str)
    age = pd.to_numeric(df.get("年龄", pd.Series([""] * len(df))).astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    feat["age"] = age.fillna(age.median()).astype(float)
    sex = df.get("性别", pd.Series([""] * len(df))).fillna("").astype(str)
    feat["sex_male"] = sex.str.contains("男", regex=False).astype(float)
    feat["sex_female"] = sex.str.contains("女", regex=False).astype(float)
    feat["gross_text_len"] = text.str.len().astype(float)
    feat["gross_has_text"] = (text.str.len() > 0).astype(float)

    size_rows = [size_stats(t) for t in text]
    feat = pd.concat([feat, pd.DataFrame(size_rows, index=df.index)], axis=1)

    keyword_groups = {
        "boundary_clear": ["界清", "界尚清", "边界清", "边界尚清"],
        "boundary_unclear": ["界不清", "边界不清", "界欠清"],
        "capsule_any": ["包膜"],
        "capsule_complete": ["包膜完整", "有包膜", "似有包膜", "可见包膜"],
        "capsule_absent": ["未见明显包膜", "无包膜", "未见包膜"],
        "capsule_involved": ["包膜侵犯", "侵犯包膜", "突破包膜", "累及包膜"],
        "hemorrhage": ["出血", "暗红", "血性"],
        "necrosis": ["坏死", "坏死样"],
        "cystic": ["囊性", "囊变", "囊腔", "囊实"],
        "calcification": ["钙化"],
        "lobulated": ["分叶", "结节状", "多结节"],
        "septum": ["分隔", "纤维分隔"],
        "fat_attached": ["脂肪"],
        "lung_attached": ["肺"],
        "pericardium_attached": ["心包"],
        "pleura_attached": ["胸膜"],
        "gray_white": ["灰白"],
        "gray_yellow": ["灰黄"],
        "gray_red": ["灰红", "红色"],
        "gray_brown": ["灰褐", "褐色"],
        "gray_black": ["灰黑", "黑色"],
        "texture_tender": ["质嫩", "质软"],
        "texture_medium": ["质中"],
        "texture_tough": ["质韧", "质硬"],
        "texture_fragile": ["质脆", "易碎"],
    }
    for name, words in keyword_groups.items():
        feat[f"kw_{name}"] = text.map(lambda s, ws=words: has_any(s, ws)).astype(float)
        feat[f"cnt_{name}"] = text.map(lambda s, ws=words: count_any(s, ws)).astype(float)

    # Log-transform heavily skewed size/count fields.
    for col in list(feat.columns):
        if col.endswith("_mm") or col.endswith("_mm2") or col.endswith("_mm3") or col.startswith("cnt_") or col == "gross_text_len":
            feat[f"log1p_{col}"] = np.log1p(feat[col].astype(float))
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_image_source(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str})
    required = {"case_id", "fold_id", "label_idx", "prob_high_risk_group"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return df[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": f"prob_{name}"})


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def score(y_true: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y_true, pred))
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if objective == "f1":
        return float(f1_score(y_true, pred, zero_division=0))
    raise ValueError(objective)


def best_threshold(y_true: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        s = score(y_true, pred, objective)
        key = (s, -abs(float(t) - 0.5))
        if key > (best_s, -abs(best_t - 0.5)):
            best_s = s
            best_t = float(t)
    return best_t, best_s


def metric_row(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    y = df["label_idx"].to_numpy(dtype=int)
    prob = df["prob_high_risk_group"].to_numpy(dtype=float)
    pred = df["pred_idx"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "group": name,
        "n": int(len(df)),
        "n_low": int((y == 0).sum()),
        "n_high": int((y == 1).sum()),
        "auc": safe_auc(y, prob),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def build_feature_frame(base_df: pd.DataFrame, gross_feat: pd.DataFrame, image_source_names: list[str], use_gross: bool) -> pd.DataFrame:
    pieces = []
    if image_source_names:
        img = pd.DataFrame(index=base_df.index)
        for name in image_source_names:
            p = base_df[f"prob_{name}"].to_numpy(dtype=float)
            img[f"p_{name}"] = p
            img[f"logit_{name}"] = logit(p)
            img[f"margin_{name}"] = np.abs(p - 0.5)
        for a, b in combinations(image_source_names, 2):
            img[f"diff_{a}_{b}"] = base_df[f"prob_{a}"].to_numpy(dtype=float) - base_df[f"prob_{b}"].to_numpy(dtype=float)
        pieces.append(img)
    if use_gross:
        pieces.append(gross_feat)
    if not pieces:
        raise ValueError("No features selected.")
    return pd.concat(pieces, axis=1)


def inner_select(
    x_all: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    train_mask: np.ndarray,
    objective: str,
) -> tuple[float, float]:
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    fold_ids = sorted(set(folds[train_mask]))
    best_c = 1.0
    best_t = 0.5
    best_s = -1.0
    for c in c_grid:
        inner_prob = np.full(len(y), np.nan)
        for val_fold in fold_ids:
            tr = train_mask & (folds != val_fold)
            va = train_mask & (folds == val_fold)
            if tr.sum() == 0 or va.sum() == 0:
                continue
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_all.loc[tr])
            x_va = scaler.transform(x_all.loc[va])
            clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260520)
            clf.fit(x_tr, y[tr])
            inner_prob[va] = clf.predict_proba(x_va)[:, 1]
        valid = train_mask & ~np.isnan(inner_prob)
        if valid.sum() == 0:
            continue
        threshold, s = best_threshold(y[valid], inner_prob[valid], objective)
        key = (s, -abs(threshold - 0.5), -abs(math.log10(c)))
        if key > (best_s, -abs(best_t - 0.5), -abs(math.log10(best_c))):
            best_s = s
            best_c = c
            best_t = threshold
    return best_c, best_t


def run_probe(
    name: str,
    base_df: pd.DataFrame,
    gross_feat: pd.DataFrame,
    image_source_names: list[str],
    use_gross: bool,
    output_dir: Path,
    objective: str,
) -> dict[str, object]:
    x_all = build_feature_frame(base_df, gross_feat, image_source_names, use_gross)
    y = base_df["label_idx"].to_numpy(dtype=int)
    folds = base_df["fold_id"].to_numpy(dtype=int)
    outputs = []
    choices = []
    for fold in sorted(set(folds)):
        train_mask = folds != fold
        test_mask = folds == fold
        c, threshold = inner_select(x_all, y, folds, train_mask, objective)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all.loc[train_mask])
        x_test = scaler.transform(x_all.loc[test_mask])
        clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260520 + int(fold))
        clf.fit(x_train, y[train_mask])
        prob = clf.predict_proba(x_test)[:, 1]
        out = base_df.loc[test_mask, ["case_id", "fold_id", "label_idx", "task_l6_label", "task_l7_label"]].copy()
        out["prob_high_risk_group"] = prob
        out["prob_low_risk_group"] = 1.0 - prob
        out["pred_idx"] = (prob >= threshold).astype(int)
        outputs.append(out)
        choices.append({"fold_id": int(fold), "c": c, "threshold": threshold, "n_features": int(x_all.shape[1])})

    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(outputs, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False)
    pd.DataFrame(choices).to_csv(run_dir / "fold_choices.csv", index=False)
    metrics = [metric_row("overall", oof)]
    for group, sub in oof.groupby("task_l6_label", sort=True):
        metrics.append(metric_row(f"task6={group}", sub))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(run_dir / "oof_metrics_by_group.csv", index=False)
    overall = metrics[0] | {
        "image_sources": image_source_names,
        "use_gross": bool(use_gross),
        "objective": objective,
        "n_features": int(x_all.shape[1]),
    }
    (run_dir / "overall_metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return overall | {"config": name}


def write_feature_audit(base_df: pd.DataFrame, gross_feat: pd.DataFrame, output_dir: Path) -> None:
    audit = base_df[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "肉眼所见"]].copy()
    audit = pd.concat([audit, gross_feat], axis=1)
    audit.to_csv(output_dir / "gross_structured_features_case_table.csv", index=False, encoding="utf-8-sig")

    rows = []
    for col in gross_feat.columns:
        if not np.issubdtype(gross_feat[col].dtype, np.number):
            continue
        low = gross_feat.loc[base_df["label_idx"] == 0, col]
        high = gross_feat.loc[base_df["label_idx"] == 1, col]
        rows.append(
            {
                "feature": col,
                "low_mean": float(low.mean()),
                "high_mean": float(high.mean()),
                "diff_high_minus_low": float(high.mean() - low.mean()),
                "low_positive_rate": float((low > 0).mean()),
                "high_positive_rate": float((high > 0).mean()),
            }
        )
    pd.DataFrame(rows).sort_values("diff_high_minus_low", key=lambda s: s.abs(), ascending=False).to_csv(
        output_dir / "gross_feature_true_label_differences.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(project_root / args.registry_csv, dtype={"case_id": str, "original_case_id": str})
    split = pd.read_csv(project_root / args.split_csv, dtype={"case_id": str})
    split = split[["case_id", "master_fold_id"]].rename(columns={"master_fold_id": "fold_id"})
    base_df = registry.merge(split, on="case_id", how="inner")
    base_df = base_df[base_df["task_l7_label"].isin(TASK7_CLASS_NAMES)].copy()
    base_df["label_idx"] = (base_df["task_l7_label"] == "high_risk_group").astype(int)
    base_df["fold_id"] = base_df["fold_id"].astype(int)
    base_df["肉眼所见"] = base_df["肉眼所见"].fillna("")

    root = project_root / "outputs" / "batch1_batch2_task567_20260514"
    image_sources = {
        "stage2": root / "task7_curriculum_runs/07_case_mlp_schemeB_m060_stage2only_full5fold/oof_case_predictions_mean.csv",
        "stage3": root / "task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/oof_case_predictions_mean.csv",
        "main": root / "task7_curriculum_runs/12_stage2_salvage_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "upper": root / "task7_curriculum_runs/36_stage3_balcore_foldwise_blend_noncore/oof_case_predictions_mean.csv",
    }
    for name, path in image_sources.items():
        base_df = base_df.merge(load_image_source(path, name), on="case_id", how="left")

    gross_feat = extract_gross_features(base_df)
    write_feature_audit(base_df, gross_feat, output_dir)

    configs = {
        "gross_only": ([], True),
        "main_plus_gross": (["main"], True),
        "upper_plus_gross": (["upper"], True),
        "stage2_stage3_upper_image_only": (["stage2", "stage3", "upper"], False),
        "stage2_stage3_upper_plus_gross": (["stage2", "stage3", "upper"], True),
        "stage2_stage3_main_upper_plus_gross": (["stage2", "stage3", "main", "upper"], True),
    }
    summary = []
    for name, (sources, use_gross) in configs.items():
        summary.append(run_probe(name, base_df, gross_feat, sources, use_gross, output_dir, args.objective))
    summary_df = pd.DataFrame(summary).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(output_dir / "gross_probe_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
