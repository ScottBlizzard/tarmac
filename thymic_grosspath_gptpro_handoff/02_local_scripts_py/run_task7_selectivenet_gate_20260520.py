from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 SelectiveNet-like direct-pass gate on frozen DINO case features.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--feature-table-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_feature_table.csv",
    )
    parser.add_argument(
        "--feature-npy",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_features.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/18_selectivenet_gate_20260520",
    )
    parser.add_argument("--feature-sets", default="dino,dino_oof")
    parser.add_argument("--coverage-targets", default="0.50,0.70")
    parser.add_argument("--accuracy-targets", default="0.90,0.95")
    parser.add_argument("--prediction-sources", default="selnet,upper")
    parser.add_argument("--selector-supervision", default="none", choices=("none", "nonhardcore", "upper_correct"))
    parser.add_argument("--selector-supervision-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=280)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--coverage-lambda", type=float, default=8.0)
    parser.add_argument("--aux-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class SelectiveMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        out_dim = hidden_dim // 2
        self.cls_head = nn.Linear(out_dim, 1)
        self.sel_head = nn.Linear(out_dim, 1)
        self.aux_head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.cls_head(h).squeeze(1), self.sel_head(h).squeeze(1), self.aux_head(h).squeeze(1)


@dataclass
class TrainConfig:
    coverage_target: float
    hidden_dim: int
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    coverage_lambda: float
    aux_weight: float
    selector_supervision_weight: float
    seed: int
    device: torch.device


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    selector_target: np.ndarray | None = None,
) -> SelectiveMLP:
    set_seed(cfg.seed)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if selector_target is None:
        s_tensor = torch.full_like(y_tensor, -1.0)
    else:
        s_tensor = torch.tensor(selector_target, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor, s_tensor), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model = SelectiveMLP(x.shape[1], cfg.hidden_dim, cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb, sb in loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            sb = sb.to(cfg.device)
            cls_logit, sel_logit, aux_logit = model(xb)
            cls_loss_each = bce(cls_logit, yb)
            sel = torch.sigmoid(sel_logit)
            selective_loss = (cls_loss_each * sel).sum() / (sel.sum() + 1e-6)
            coverage = sel.mean()
            coverage_loss = torch.relu(torch.tensor(cfg.coverage_target, device=cfg.device) - coverage).pow(2)
            aux_loss = bce(aux_logit, yb).mean()
            valid_sel = sb >= 0
            sel_supervised_loss = torch.tensor(0.0, device=cfg.device)
            if valid_sel.any() and cfg.selector_supervision_weight > 0:
                sel_supervised_loss = bce(sel_logit[valid_sel], sb[valid_sel]).mean()
            loss = (
                selective_loss
                + cfg.coverage_lambda * coverage_loss
                + cfg.aux_weight * aux_loss
                + cfg.selector_supervision_weight * sel_supervised_loss
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


@torch.no_grad()
def predict_model(model: SelectiveMLP, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    xs = torch.tensor(x, dtype=torch.float32, device=device)
    cls_logit, sel_logit, aux_logit = model(xs)
    cls_prob = torch.sigmoid(cls_logit).cpu().numpy()
    sel_prob = torch.sigmoid(sel_logit).cpu().numpy()
    aux_prob = torch.sigmoid(aux_logit).cpu().numpy()
    return cls_prob, sel_prob, aux_prob


def build_oof_features(df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    out = pd.DataFrame(index=df.index)
    for col in prob_cols:
        p = pd.to_numeric(df[col], errors="coerce").fillna(0.5).astype(float)
        out[col] = p
        out[f"{col}_margin"] = (p - 0.5).abs()
    for col in pred_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    probs = df[prob_cols].astype(float).fillna(0.5).to_numpy()
    votes = (probs >= 0.5).astype(float)
    out["prob_mean"] = probs.mean(axis=1)
    out["prob_median"] = np.median(probs, axis=1)
    out["prob_std"] = probs.std(axis=1)
    out["prob_range"] = probs.max(axis=1) - probs.min(axis=1)
    out["vote_frac"] = votes.mean(axis=1)
    out["vote_disagree"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
    out["upper_conf"] = df["upper_conf"].astype(float)
    out["image_count"] = pd.to_numeric(df.get("image_count", 1), errors="coerce").fillna(1.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def align_features(case_df: pd.DataFrame, table_path: Path, npy_path: Path) -> tuple[np.ndarray, list[str]]:
    table = pd.read_csv(table_path, dtype={"case_id": str})
    features = np.load(npy_path).astype(np.float32)
    lookup = table.reset_index().rename(columns={"index": "row_idx"})[["case_id", "row_idx"]]
    order = case_df[["case_id"]].merge(lookup, on="case_id", how="left")
    if order["row_idx"].isna().any():
        missing = order.loc[order["row_idx"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing DINO features for {missing}")
    x = features[order["row_idx"].astype(int).to_numpy()]
    return x, [f"dino_{i}" for i in range(x.shape[1])]


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(
    sel_score: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    target_acc: float,
    min_accept: int = 8,
) -> dict[str, float | int]:
    order = np.argsort(-sel_score, kind="mergesort")
    correct = (y[order] == pred[order]).astype(float)
    cumsum = np.cumsum(correct)
    ks = np.arange(1, len(order) + 1)
    acc = cumsum / ks
    ok = (acc >= target_acc) & (ks >= min_accept)
    if not ok.any():
        return {"threshold": np.inf, "train_accept_n": 0, "train_accept_acc": np.nan}
    best_idx = np.where(ok)[0][-1]
    return {
        "threshold": float(sel_score[order[best_idx]]),
        "train_accept_n": int(best_idx + 1),
        "train_accept_acc": float(acc[best_idx]),
    }


def make_inner_oof(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    cfg: TrainConfig,
    selector_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cls = np.full(len(y), np.nan, dtype=float)
    sel = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(np.unique(folds)):
        tr = folds != fold
        va = folds == fold
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_va = scaler.transform(x[va])
        st = selector_target[tr] if selector_target is not None else None
        model = train_model(x_tr, y[tr], cfg, st)
        cls_prob, sel_prob, _ = predict_model(model, x_va, cfg.device)
        cls[va] = cls_prob
        sel[va] = sel_prob
    return cls, sel


def get_selector_target(df: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "nonhardcore":
        return 1.0 - df["hard_core"].astype(float).to_numpy()
    if mode == "upper_correct":
        return (
            df["pred_upper"].astype(int).to_numpy()
            == df["label_idx"].astype(int).to_numpy()
        ).astype(float)
    raise ValueError(mode)


def run_combo(
    df: pd.DataFrame,
    x_all: np.ndarray,
    feature_set: str,
    coverage_target: float,
    accuracy_target: float,
    pred_source: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    upper_prob = df["p_upper"].astype(float).to_numpy()
    upper_pred = df["pred_upper"].astype(int).to_numpy()
    hard = df["hard_core"].astype(int).to_numpy().astype(bool)
    selector_target_all = get_selector_target(df, args.selector_supervision)
    final_sel = np.full(len(df), np.nan, dtype=float)
    final_prob = np.full(len(df), np.nan, dtype=float)
    final_pred = np.full(len(df), -1, dtype=int)
    accept = np.zeros(len(df), dtype=bool)
    fold_rows = []

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all[train])
        x_test = scaler.transform(x_all[test])
        cfg = TrainConfig(
            coverage_target=coverage_target,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            coverage_lambda=args.coverage_lambda,
            aux_weight=args.aux_weight,
            selector_supervision_weight=args.selector_supervision_weight,
            seed=args.seed + int(fold) * 97,
            device=device,
        )
        selector_train = selector_target_all[train] if selector_target_all is not None else None
        inner_cls, inner_sel = make_inner_oof(x_train, y[train], folds[train], cfg, selector_train)
        if pred_source == "selnet":
            inner_pred = (inner_cls >= 0.5).astype(int)
        elif pred_source == "upper":
            inner_pred = upper_pred[train]
        else:
            raise ValueError(pred_source)
        chosen = select_threshold(inner_sel, y[train], inner_pred, accuracy_target)

        model = train_model(x_train, y[train], cfg, selector_train)
        cls_prob, sel_prob, _ = predict_model(model, x_test, device)
        pred = (cls_prob >= 0.5).astype(int) if pred_source == "selnet" else upper_pred[test]
        final_sel[test] = sel_prob
        final_prob[test] = cls_prob
        final_pred[test] = pred
        accept[test] = sel_prob >= float(chosen["threshold"])

        fold_rows.append(
            {
                "feature_set": feature_set,
                "coverage_target": coverage_target,
                "accuracy_target": accuracy_target,
                "prediction_source": pred_source,
                "fold_id": int(fold),
                **chosen,
                "test_accept_n": int(accept[test].sum()),
                "test_n": int(test.sum()),
            }
        )

    accepted = accept
    reviewed = ~accept
    accept_metrics = metric_row(y[accepted], final_pred[accepted]) if accepted.any() else {}
    all_metrics = metric_row(y, final_pred)
    base_metrics = metric_row(y, upper_pred)
    summary = {
        "feature_set": feature_set,
        "coverage_target": coverage_target,
        "accuracy_target": accuracy_target,
        "prediction_source": pred_source,
        "selector_supervision": args.selector_supervision,
        "selector_supervision_weight": args.selector_supervision_weight,
        "accept_n": int(accepted.sum()),
        "accept_frac": float(accepted.mean()),
        "review_n": int(reviewed.sum()),
        "review_frac": float(reviewed.mean()),
        "accept_accuracy": accept_metrics.get("accuracy", np.nan),
        "accept_balanced_accuracy": accept_metrics.get("balanced_accuracy", np.nan),
        "accept_sensitivity_high": accept_metrics.get("sensitivity_high", np.nan),
        "accept_specificity_low": accept_metrics.get("specificity_low", np.nan),
        "hardcore_recall_in_review": float((hard & reviewed).sum() / max(1, int(hard.sum()))),
        "hardcore_precision_in_review": float((hard & reviewed).sum() / max(1, int(reviewed.sum()))),
        "base_accuracy": base_metrics["accuracy"],
        "base_balanced_accuracy": base_metrics["balanced_accuracy"],
        "selnet_all_accuracy": all_metrics["accuracy"],
        "selnet_all_balanced_accuracy": all_metrics["balanced_accuracy"],
        "selnet_all_sensitivity_high": all_metrics["sensitivity_high"],
        "selnet_all_specificity_low": all_metrics["specificity_low"],
    }
    case_df = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "pred_upper",
            "p_upper",
            "difficulty_fine",
            "hard_core",
            "upper_wrong",
            "upper_fn",
            "upper_fp",
        ]
    ].copy()
    case_df["feature_set"] = feature_set
    case_df["coverage_target"] = coverage_target
    case_df["accuracy_target"] = accuracy_target
    case_df["prediction_source"] = pred_source
    case_df["selector_supervision"] = args.selector_supervision
    case_df["selnet_prob_high"] = final_prob
    case_df["select_score"] = final_sel
    case_df["pred_used_for_accept"] = final_pred
    case_df["accepted"] = accepted.astype(int)
    return summary, case_df, pd.DataFrame(fold_rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    dino_x, _ = align_features(df, Path(args.feature_table_csv), Path(args.feature_npy))
    oof_x = build_oof_features(df).to_numpy(dtype=np.float32)
    feature_sets = {
        "dino": dino_x,
        "dino_oof": np.concatenate([dino_x, oof_x], axis=1),
    }
    selected_feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]
    coverages = [float(x.strip()) for x in args.coverage_targets.split(",") if x.strip()]
    acc_targets = [float(x.strip()) for x in args.accuracy_targets.split(",") if x.strip()]
    pred_sources = [x.strip() for x in args.prediction_sources.split(",") if x.strip()]

    summaries = []
    cases = []
    folds = []
    for feature_set in selected_feature_sets:
        if feature_set not in feature_sets:
            raise KeyError(f"Unknown feature set: {feature_set}")
        x_all = feature_sets[feature_set]
        for coverage in coverages:
            for acc_target in acc_targets:
                for pred_source in pred_sources:
                    summary, case_df, fold_df = run_combo(
                        df,
                        x_all,
                        feature_set,
                        coverage,
                        acc_target,
                        pred_source,
                        args,
                        device,
                    )
                    summaries.append(summary)
                    cases.append(case_df)
                    folds.append(fold_df)
                    print(json.dumps(summary, ensure_ascii=False))

    summary_df = pd.DataFrame(summaries).sort_values(
        ["accuracy_target", "accept_accuracy", "accept_frac"], ascending=[True, False, False]
    )
    case_out = pd.concat(cases, ignore_index=True)
    fold_out = pd.concat(folds, ignore_index=True)
    summary_df.to_csv(out_dir / "selectivenet_gate_summary.csv", index=False)
    case_out.to_csv(out_dir / "selectivenet_gate_case_outputs.csv", index=False)
    fold_out.to_csv(out_dir / "selectivenet_gate_fold_thresholds.csv", index=False)
    print("\nTop summary:")
    print(summary_df.head(30).to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
