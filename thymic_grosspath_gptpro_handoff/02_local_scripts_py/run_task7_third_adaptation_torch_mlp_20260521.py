from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


PROFILE = {"AB": 24, "B1": 4, "B2": 22, "TC": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch MLP adaptation on frozen whole+crop DINO features.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521")
    parser.add_argument("--third-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521")
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/08_adapt72_highfocus_torch_mlp_20260521")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metric_dict(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y, prob, objective="balanced_accuracy"):
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        score = balanced_accuracy_score(y, pred) if objective == "balanced_accuracy" else accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def read_old(root: Path, rel: str):
    d = root / rel
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    x = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table.get("feature_idx", pd.Series(np.arange(len(table)))).astype(int)
    table["label_idx"] = table["label_idx"].astype(int)
    return table.reset_index(drop=True), x[table["feature_idx"].to_numpy()]


def read_third(root: Path, rel: str):
    d = root / rel
    ft = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    reg = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    x = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    ft["feature_idx"] = ft.get("feature_idx", pd.Series(np.arange(len(ft)))).astype(int)
    frame = reg.merge(ft[["case_id", "feature_idx"]], on="case_id", how="left")
    frame["label_idx"] = frame["label_idx"].astype(int)
    return frame.reset_index(drop=True), x[frame["feature_idx"].astype(int).to_numpy()]


def stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def split_adapt_holdout(third: pd.DataFrame, seed: int):
    chosen = []
    for subtype, n in PROFILE.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        chosen.extend(g.sort_values("_key").head(n).index.tolist())
    mask = np.zeros(len(third), dtype=bool)
    mask[np.array(chosen, dtype=int)] = True
    return np.where(mask)[0], np.where(~mask)[0]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def repeat_indices(indices: np.ndarray, is_adapt: np.ndarray, repeat_adapt: int) -> np.ndarray:
    old = indices[~is_adapt[indices]]
    adapt = indices[is_adapt[indices]]
    if repeat_adapt <= 1:
        return indices
    return np.concatenate([old, np.tile(adapt, repeat_adapt)])


def predict_prob(model, x, device, batch=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.tensor(x[i : i + batch], dtype=torch.float32, device=device)
            prob = torch.softmax(model(xb), dim=1)[:, 1].detach().cpu().numpy()
            outs.append(prob)
    return np.concatenate(outs)


def train_one(x_all, y_all, is_adapt, train_idx, val_idx, x_hold, cfg, seed, device):
    set_seed(seed)
    scaler = StandardScaler()
    scaler.fit(x_all[train_idx])
    x_scaled = scaler.transform(x_all).astype(np.float32)
    x_hold_scaled = scaler.transform(x_hold).astype(np.float32)
    tr_rep = repeat_indices(train_idx, is_adapt, cfg["repeat_adapt"])
    y_tr = y_all[tr_rep]
    counts = np.bincount(y_tr, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    model = MLP(x_scaled.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=cfg["smooth"])
    best = {"score": -1.0, "state": None, "epoch": 0}
    stale = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        order = np.random.permutation(tr_rep)
        for i in range(0, len(order), cfg["batch"]):
            idx = order[i : i + cfg["batch"]]
            xb = torch.tensor(x_scaled[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_all[idx], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        p_val = predict_prob(model, x_scaled[val_idx], device)
        t, _ = best_threshold(y_all[val_idx], p_val, "balanced_accuracy")
        pred_val = (p_val >= t).astype(int)
        score = balanced_accuracy_score(y_all[val_idx], pred_val)
        if score > best["score"]:
            best = {"score": float(score), "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch, "threshold": t}
            stale = 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best["state"])
    p_val = predict_prob(model, x_scaled[val_idx], device)
    p_hold = predict_prob(model, x_hold_scaled, device)
    return best, p_val, p_hold


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)
    adapt_idx, hold_idx = split_adapt_holdout(third, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    x_all = np.concatenate([x_old, x_third[adapt_idx]], axis=0)
    y_all = np.concatenate([old["label_idx"].to_numpy(int), adapt["label_idx"].to_numpy(int)], axis=0)
    is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])
    x_hold = x_third[hold_idx]
    y_hold = hold["label_idx"].to_numpy(int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.18, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x_all, y_all))
    configs = []
    for repeat_adapt in [1, 2, 4, 8]:
        for hidden in [128, 256, 512]:
            for dropout in [0.15, 0.3]:
                configs.append(
                    {
                        "repeat_adapt": repeat_adapt,
                        "hidden": hidden,
                        "dropout": dropout,
                        "lr": 8e-4 if hidden <= 256 else 5e-4,
                        "wd": 1e-4,
                        "smooth": 0.02,
                        "epochs": 160,
                        "patience": 16,
                        "batch": 64,
                    }
                )
    rows = []
    hold_cases = []
    for ci, cfg in enumerate(configs):
        best, p_val, p_hold = train_one(x_all, y_all, is_adapt, train_idx, val_idx, x_hold, cfg, args.seed + ci * 17, device)
        for objective in ["balanced_accuracy", "accuracy"]:
            t, _ = best_threshold(y_all[val_idx], p_val, objective)
            pred_val = (p_val >= t).astype(int)
            pred_hold = (p_hold >= t).astype(int)
            row = {"config_id": ci, "threshold_objective": objective, "threshold": t, "best_epoch": best["epoch"], **cfg}
            row.update({f"val_{k}": v for k, v in metric_dict(y_all[val_idx], pred_val, p_val).items()})
            row.update({f"holdout_{k}": v for k, v in metric_dict(y_hold, pred_hold, p_hold).items()})
            rows.append(row)
            if objective == "balanced_accuracy":
                tmp = hold[["case_id", "original_case_id", "task_l6_label", "label_idx"]].copy()
                tmp["config_id"] = ci
                tmp["prob_high"] = p_hold
                tmp["pred_idx"] = pred_hold
                tmp["correct"] = (pred_hold == y_hold).astype(int)
                hold_cases.append(tmp)
    summary = pd.DataFrame(rows).sort_values(["holdout_balanced_accuracy", "holdout_accuracy", "holdout_f1"], ascending=False)
    summary.to_csv(out / "torch_mlp_summary.csv", index=False, encoding="utf-8-sig")
    if hold_cases:
        pd.concat(hold_cases, ignore_index=True).to_csv(out / "torch_mlp_all_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    top = summary.iloc[0].to_dict()
    best_cases = pd.concat(hold_cases, ignore_index=True)
    best_cases = best_cases[best_cases["config_id"].eq(int(top["config_id"]))].copy()
    best_cases.to_csv(out / "best_torch_mlp_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    subtype = best_cases.groupby("task_l6_label").agg(n=("case_id", "size"), correct=("correct", "sum"), accuracy=("correct", "mean")).reset_index()
    subtype.to_csv(out / "best_torch_mlp_subtype_metrics.csv", index=False, encoding="utf-8-sig")
    (out / "torch_mlp_report.json").write_text(json.dumps({"top": top, "subtype": subtype.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
