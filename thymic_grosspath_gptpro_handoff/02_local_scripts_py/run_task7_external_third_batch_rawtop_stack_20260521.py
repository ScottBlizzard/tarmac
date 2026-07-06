from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_raw_candidate_stacking_20260521 import build_raw_candidate_features, make_models  # noqa: E402


RAW_TOP_NAMES = [
    "threeway",
    "main",
    "anti34",
    "anti30",
    "anti22",
    "stage3",
    "anti25",
    "stage2",
    "case_selacc",
    "dino_seed888",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 third-batch external raw-top source stacking.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--old-case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--old-curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--old-feature-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_feature_table.csv",
    )
    parser.add_argument(
        "--old-feature-npy",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_features.npy",
    )
    parser.add_argument(
        "--third-registry-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/01_third_batch_64style_image_only_20260521/third_batch_task7_registry.csv",
    )
    parser.add_argument(
        "--third-feature-table",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/01_third_batch_64style_image_only_20260521/third_batch_dino_concat_feature_table.csv",
    )
    parser.add_argument(
        "--third-feature-npy",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/01_third_batch_64style_image_only_20260521/third_batch_dino_concat_features.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/02_third_batch_rawtop_stack_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=14)
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
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
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, dict[str, object]]:
    best_key = None
    best_t = 0.5
    best_row: dict[str, object] = {}
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        row = metric_row(y, pred, prob)
        if objective == "sens90":
            penalty = min(0.0, float(row["sensitivity"]) - 0.90)
            key = (float(row["balanced_accuracy"]) + penalty, float(row["accuracy"]), float(row["f1"]))
        elif objective == "acc":
            key = (float(row["accuracy"]), float(row["balanced_accuracy"]), float(row["f1"]))
        else:
            key = (float(row["balanced_accuracy"]), float(row["accuracy"]), float(row["f1"]))
        if best_key is None or key > best_key:
            best_key = key
            best_t = float(t)
            best_row = row
    return best_t, best_row


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    mode: str
    seed: int
    hidden_dim: int = 512
    dropout: float = 0.2
    stage3_scope: str = "all"
    hard_weight: float = 0.5
    selection_metric: str = "balanced_accuracy"


def source_specs() -> list[SourceSpec]:
    return [
        SourceSpec("threeway", "curriculum_stage3", 2039, hard_weight=0.5, stage3_scope="all"),
        SourceSpec("main", "curriculum_stage3", 2027, hard_weight=0.6, stage3_scope="salvage"),
        SourceSpec("anti34", "curriculum_stage3", 2034, hard_weight=0.2, stage3_scope="all"),
        SourceSpec("anti30", "direct", 2030, hidden_dim=768, dropout=0.25),
        SourceSpec("anti22", "direct", 2022, hidden_dim=512, dropout=0.25),
        SourceSpec("stage3", "curriculum_stage3", 2026, hard_weight=0.5, stage3_scope="salvage"),
        SourceSpec("anti25", "direct", 2025, hidden_dim=768, dropout=0.30),
        SourceSpec("stage2", "curriculum_stage2", 2026, hard_weight=0.5),
        SourceSpec("case_selacc", "direct", 2027, hidden_dim=512, dropout=0.2, selection_metric="accuracy"),
        SourceSpec("dino_seed888", "direct", 888, hidden_dim=512, dropout=0.2),
    ]


def load_aligned_features(frame: pd.DataFrame, table_path: Path, npy_path: Path) -> np.ndarray:
    table = pd.read_csv(table_path, dtype={"case_id": str})
    arr = np.load(npy_path).astype(np.float32)
    order = frame[["case_id"]].merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        missing = order.loc[order["feature_idx"].isna(), "case_id"].head().tolist()
        raise KeyError(f"Missing features: {missing}")
    return arr[order["feature_idx"].astype(int).to_numpy()]


def standardize(train_x: np.ndarray, val_x: np.ndarray, external_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((val_x - mean) / std).astype(np.float32),
        ((external_x - mean) / std).astype(np.float32),
    )


def eval_metric(y: np.ndarray, prob: np.ndarray, metric: str) -> float:
    pred = (prob >= 0.5).astype(int)
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    return float(balanced_accuracy_score(y, pred))


def train_stage(
    model: MLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    seed: int,
    selection_metric: str,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_train = torch.from_numpy(train_x.astype(np.float32))
    y_train = torch.from_numpy(train_y.astype(np.int64))
    w_train = torch.from_numpy(train_weights.astype(np.float32))
    x_val = torch.from_numpy(val_x.astype(np.float32)).to(device)
    counts = np.bincount(train_y, minlength=2).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.from_numpy(class_weights.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_score = -1e9
    stale = 0
    batch_size = min(64, len(train_y))
    for _epoch in range(1, epochs + 1):
        model.train()
        indices = rng.permutation(len(train_y))
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            wb = w_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_vec = F.cross_entropy(model(xb), yb, weight=class_weights_t, reduction="none")
            loss = (loss_vec * wb).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(x_val), dim=1).cpu().numpy()[:, 1]
        score = eval_metric(val_y, prob, selection_metric)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP stage did not produce a best state.")
    return best_state


def fit_source_fold(
    spec: SourceSpec,
    old_x: np.ndarray,
    y: np.ndarray,
    difficulty: pd.Series,
    difficulty_fine: pd.Series,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    external_x: np.ndarray,
    args: argparse.Namespace,
    fold_seed: int,
) -> np.ndarray:
    train_x_raw = old_x[train_mask]
    val_x_raw = old_x[val_mask]
    train_x, val_x, ext_x = standardize(train_x_raw, val_x_raw, external_x)
    train_y_all = y[train_mask]
    val_y_all = y[val_mask]
    train_diff = difficulty[train_mask].reset_index(drop=True)
    train_fine = difficulty_fine[train_mask].reset_index(drop=True)
    val_diff = difficulty[val_mask].reset_index(drop=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = MLP(train_x.shape[1], hidden_dim=spec.hidden_dim, dropout=spec.dropout).to(device)

    def run(mask_train: np.ndarray, mask_val: np.ndarray, weights: np.ndarray, seed_offset: int) -> None:
        nonlocal model
        state = train_stage(
            model=model,
            train_x=train_x[mask_train],
            train_y=train_y_all[mask_train],
            train_weights=weights,
            val_x=val_x[mask_val],
            val_y=val_y_all[mask_val],
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            seed=fold_seed + seed_offset,
            selection_metric=spec.selection_metric,
        )
        model.load_state_dict(state)

    if spec.mode == "direct":
        run(np.ones(len(train_y_all), dtype=bool), np.ones(len(val_y_all), dtype=bool), np.ones(len(train_y_all), dtype=np.float32), 11)
    else:
        easy_tr = train_diff.eq("easy").to_numpy()
        easy_va = val_diff.eq("easy").to_numpy()
        if easy_tr.sum() > 8 and easy_va.sum() > 2:
            run(easy_tr, easy_va, np.ones(int(easy_tr.sum()), dtype=np.float32), 11)
        em_tr = train_diff.isin(["easy", "medium"]).to_numpy()
        em_va = val_diff.isin(["easy", "medium"]).to_numpy()
        run(em_tr, em_va, np.ones(int(em_tr.sum()), dtype=np.float32), 22)
        if spec.mode == "curriculum_stage3":
            if spec.stage3_scope == "salvage":
                all_tr = train_fine.isin(["easy", "medium", "hard_salvage_teacher"]).to_numpy()
                all_va = val_diff.isin(["easy", "medium", "hard"]).to_numpy()
            else:
                all_tr = train_diff.isin(["easy", "medium", "hard"]).to_numpy()
                all_va = val_diff.isin(["easy", "medium", "hard"]).to_numpy()
            weights = np.where(train_diff[all_tr].eq("hard").to_numpy(), float(spec.hard_weight), 1.0).astype(np.float32)
            run(all_tr, all_va, weights, 33)

    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(torch.from_numpy(ext_x).to(device)), dim=1).cpu().numpy()[:, 1]
    return prob.astype(float)


def build_external_sources(
    old_frame: pd.DataFrame,
    old_x: np.ndarray,
    third: pd.DataFrame,
    third_x: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    y = old_frame["label_idx"].astype(int).to_numpy()
    folds = old_frame["fold_id"].astype(int).to_numpy()
    out = third[["case_id", "original_case_id", "label_idx", "task_l6_label", "task_l7_label", "source_folder"]].copy()
    for spec in source_specs():
        fold_probs = []
        print(f"[source] {spec.name}", flush=True)
        for fold in sorted(np.unique(folds)):
            train = folds != fold
            val = folds == fold
            prob = fit_source_fold(
                spec=spec,
                old_x=old_x,
                y=y,
                difficulty=old_frame["difficulty"],
                difficulty_fine=old_frame["difficulty_fine"],
                train_mask=train,
                val_mask=val,
                external_x=third_x,
                args=args,
                fold_seed=args.seed + spec.seed * 10 + int(fold),
            )
            fold_probs.append(prob)
        p = np.mean(np.stack(fold_probs, axis=0), axis=0)
        out[f"p_{spec.name}"] = p
        out[f"pred_{spec.name}"] = (p >= 0.5).astype(int)
    return out


def fit_full_and_predict(model: object, old_x: pd.DataFrame, y: np.ndarray, external_x: pd.DataFrame) -> np.ndarray:
    old_aligned, external_aligned = old_x.align(external_x, join="outer", axis=1, fill_value=0.0)
    clf = clone(model)
    clf.fit(old_aligned.to_numpy(dtype=float), y)
    return clf.predict_proba(external_aligned.to_numpy(dtype=float))[:, 1]


def oof_stack_prob(model: object, x_df: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    prob = np.zeros(len(y), dtype=float)
    x = x_df.to_numpy(dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        clf = clone(model)
        clf.fit(x[train], y[train])
        prob[test] = clf.predict_proba(x[test])[:, 1]
    return prob


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    old_scores = pd.read_csv(project_root / args.old_case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    old_curr = pd.read_csv(project_root / args.old_curriculum_csv, dtype={"case_id": str})
    old_frame = old_scores.merge(
        old_curr[["case_id", "difficulty", "difficulty_fine"]],
        on="case_id",
        how="left",
        suffixes=("", "_cur"),
    )
    old_frame["difficulty"] = old_frame["difficulty"].fillna(old_frame.get("difficulty_cur", "hard")).fillna("hard")
    old_frame["difficulty_fine"] = old_frame["difficulty_fine"].fillna(old_frame.get("difficulty_fine_cur", "hard_core")).fillna("hard_core")
    old_x = load_aligned_features(old_frame, project_root / args.old_feature_table, project_root / args.old_feature_npy)

    third = pd.read_csv(project_root / args.third_registry_csv, dtype={"case_id": str, "original_case_id": str})
    third_x = load_aligned_features(third, project_root / args.third_feature_table, project_root / args.third_feature_npy)

    external_sources = build_external_sources(old_frame, old_x, third, third_x, args)
    external_sources.to_csv(output_dir / "third_batch_rawtop_source_predictions.csv", index=False, encoding="utf-8-sig")

    old_raw_top = build_raw_candidate_features(old_scores, top_only=True)
    external_raw_top = build_raw_candidate_features(external_sources, top_only=True)
    y_old = old_scores["label_idx"].astype(int).to_numpy()
    y_external = third["label_idx"].astype(int).to_numpy()
    folds = old_scores["fold_id"].astype(int).to_numpy()

    models = make_models(args.seed)
    selected_model_names = ["extra_d3_l8", "extra_d2_l10", "logreg_c003", "gb_d1_lr05", "rf_d3_l8"]
    rows: list[dict[str, object]] = []
    case_outputs: list[pd.DataFrame] = []
    for model_name in selected_model_names:
        model = models[model_name]
        old_oof_prob = oof_stack_prob(model, old_raw_top, y_old, folds)
        for objective in ["bacc", "acc", "sens90"]:
            threshold, old_row = choose_threshold(y_old, old_oof_prob, objective)
            external_prob = fit_full_and_predict(model, old_raw_top, y_old, external_raw_top)
            external_pred = (external_prob >= threshold).astype(int)
            row = metric_row(y_external, external_pred, external_prob)
            row.update(
                {
                    "model": model_name,
                    "threshold_objective": objective,
                    "threshold": float(threshold),
                    "old_oof_accuracy": old_row["accuracy"],
                    "old_oof_balanced_accuracy": old_row["balanced_accuracy"],
                    "old_oof_f1": old_row["f1"],
                    "old_oof_sensitivity": old_row["sensitivity"],
                    "old_oof_specificity": old_row["specificity"],
                }
            )
            rows.append(row)

            case_df = third[
                ["case_id", "original_case_id", "source_folder", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]
            ].copy()
            case_df["model"] = model_name
            case_df["threshold_objective"] = objective
            case_df["prob_high_risk_group"] = external_prob
            case_df["pred_idx"] = external_pred
            case_df["correct"] = (external_pred == y_external).astype(int)
            case_outputs.append(case_df)

    summary = pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy", "f1"], ascending=False)
    summary.to_csv(output_dir / "third_batch_rawtop_stack_external_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(case_outputs, ignore_index=True).to_csv(
        output_dir / "third_batch_rawtop_stack_case_predictions_all.csv", index=False, encoding="utf-8-sig"
    )

    best = summary.iloc[0].to_dict()
    best_cases = pd.concat(case_outputs, ignore_index=True)
    best_cases = best_cases[
        (best_cases["model"] == best["model"]) & (best_cases["threshold_objective"] == best["threshold_objective"])
    ].copy()
    best_cases.to_csv(output_dir / "third_batch_rawtop_stack_best_case_predictions.csv", index=False, encoding="utf-8-sig")
    subtype_rows = []
    for subtype, group in best_cases.groupby("task_l6_label"):
        idx = group.index.to_numpy()
        subtype_y = group["label_idx"].astype(int).to_numpy()
        subtype_pred = group["pred_idx"].astype(int).to_numpy()
        subtype_prob = group["prob_high_risk_group"].astype(float).to_numpy()
        row = metric_row(subtype_y, subtype_pred, subtype_prob)
        row.update({"task_l6_label": subtype, "n": int(len(group))})
        subtype_rows.append(row)
    pd.DataFrame(subtype_rows).sort_values("task_l6_label").to_csv(
        output_dir / "third_batch_rawtop_stack_best_by_subtype.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "boundary": {
            "third_batch_is_external_only": True,
            "no_doctor_gross_text": True,
            "no_case_id_lookup": True,
            "note": "Raw-top source models are re-trained as old-fold ensembles from DINO image features; thresholds and stackers are selected on old 285 only.",
        },
        "old_n": int(len(old_frame)),
        "third_n": int(len(third)),
        "best_external": best,
    }
    (output_dir / "external_rawtop_stack_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
