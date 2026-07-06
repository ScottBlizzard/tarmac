from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_finetune_old_third_20260523 import DINOv3FineTuneModel  # noqa: E402
from thymic_baseline.config import get_task  # noqa: E402
from thymic_baseline.train import build_dataloader, resolve_device  # noqa: E402


DEFAULT_EXT_REGISTRY = (
    PROJECT_ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_external_runs"
    / "20_external_thymoma_carcinoma_64style_wpc_20260522"
    / "external_folder_task7_registry.csv"
)
DEFAULT_ADAPT_ROOT = PROJECT_ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_adaptation_runs"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_external_runs"
    / "68_dinov3_locked_external_eval_20260523"
)
DEFAULT_RUNS = [
    DEFAULT_ADAPT_ROOT / "64_dinov3_vitb16_task7_whole352_lastblock_lowlr_5fold_20260523",
    DEFAULT_ADAPT_ROOT / "66_dinov3_vitl16_task7_whole352_lastblock_lowlr_5fold_20260523",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked external evaluation for already-trained Task7 DINOv3 fold models.")
    parser.add_argument("--external-registry-csv", default=str(DEFAULT_EXT_REGISTRY))
    parser.add_argument("--run-dirs", default=",".join(str(p) for p in DEFAULT_RUNS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task", default="task7_lowrisk_vs_highrisk_tc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold-grid-step", type=float, default=0.01)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_dict(y_true: np.ndarray, prob_high: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob_high >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=labels).ravel()
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob_high))
    except ValueError:
        out["auc"] = float("nan")
    return out


def select_dev_threshold(run_dir: Path, step: float) -> float:
    pred_path = run_dir / "oof_case_predictions_mean.csv"
    if not pred_path.exists():
        return 0.5
    df = pd.read_csv(pred_path)
    if "prob_high_risk_group" not in df.columns:
        return 0.5
    y = df["label_idx"].astype(int).to_numpy()
    p = df["prob_high_risk_group"].astype(float).to_numpy()
    thresholds = np.arange(0.05, 0.951, step)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        score = float(balanced_accuracy_score(y, pred))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def load_external_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_id", "image_path", "image_name", "label_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"External registry missing columns: {sorted(missing)}")
    missing_paths = [str(p) for p in df["image_path"].tolist() if not Path(str(p)).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing external image files, first 5: {missing_paths[:5]}")
    return df.reset_index(drop=True)


def load_model(run_args: dict[str, Any], checkpoint_path: Path, device: torch.device) -> DINOv3FineTuneModel:
    task = get_task(str(run_args.get("task", "task7_lowrisk_vs_highrisk_tc")))
    model = DINOv3FineTuneModel(
        model_name=str(run_args["model_name"]),
        global_pool=str(run_args.get("global_pool", "token")),
        input_variant=str(run_args.get("input_variant", "whole")),
        num_classes=task.num_classes,
        dropout=float(run_args.get("dropout", 0.2)),
        head_type=str(run_args.get("head_type", "mlp")),
        hidden_dim=int(run_args.get("hidden_dim", 512)),
    )
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, tuple):
        return tuple(move_to_device(item, device) for item in inputs)
    if isinstance(inputs, list):
        return [move_to_device(item, device) for item in inputs]
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def predict_fold(
    model: DINOv3FineTuneModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    progress_desc: str,
) -> tuple[list[str], list[str], list[int], np.ndarray]:
    case_ids: list[str] = []
    image_names: list[str] = []
    labels: list[int] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, batch_labels, batch_case_ids, batch_image_names in tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True):
            inputs = move_to_device(inputs, device)
            logits = model(inputs)
            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(batch_probs)
            case_ids.extend(str(x) for x in batch_case_ids)
            image_names.extend(str(x) for x in batch_image_names)
            labels.extend(int(x) for x in batch_labels.cpu().numpy().tolist())
    return case_ids, image_names, labels, np.concatenate(probs, axis=0)


def summarize_subtypes(pred_df: pd.DataFrame, threshold: float, model_tag: str, subset_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "task_l6_label" not in pred_df.columns:
        return pd.DataFrame(rows)
    for subtype, group in pred_df.groupby("task_l6_label", dropna=False):
        y = group["label_idx"].astype(int).to_numpy()
        p = group["prob_high_risk_group"].astype(float).to_numpy()
        row = metric_dict(y, p, threshold)
        row.update({"model_tag": model_tag, "subset": subset_name, "task_l6_label": subtype})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_run(run_dir: Path, external_df: pd.DataFrame, args: argparse.Namespace, device: torch.device, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_args = read_json(run_dir / "args.json")
    model_tag = run_dir.name
    image_size = int(run_args.get("image_size", 352))
    input_variant = str(run_args.get("input_variant", "whole"))
    batch_size = int(args.batch_size or run_args.get("batch_size", 4))
    loader = build_dataloader(
        external_df,
        input_variant=input_variant,
        image_size=image_size,
        is_train=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    fold_probs: list[np.ndarray] = []
    base_case_ids: list[str] | None = None
    base_image_names: list[str] | None = None
    base_labels: list[int] | None = None
    checkpoints = sorted(run_dir.glob("fold_*/best_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found under {run_dir}")
    for checkpoint in checkpoints:
        fold_id = checkpoint.parent.name
        print(f"[{model_tag}] predicting {fold_id}", flush=True)
        model = load_model(run_args, checkpoint, device)
        case_ids, image_names, labels, probs = predict_fold(model, loader, device, f"{model_tag} {fold_id}")
        if base_case_ids is None:
            base_case_ids = case_ids
            base_image_names = image_names
            base_labels = labels
        elif case_ids != base_case_ids:
            raise RuntimeError(f"Case order mismatch in {model_tag} {fold_id}")
        fold_probs.append(probs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
    pred_df = pd.DataFrame(
        {
            "case_id": base_case_ids,
            "image_name": base_image_names,
            "label_idx": base_labels,
            "prob_low_risk_group": avg_probs[:, 0],
            "prob_high_risk_group": avg_probs[:, 1],
            "pred_idx_050": (avg_probs[:, 1] >= 0.5).astype(int),
        }
    )
    metadata_cols = [
        "case_id",
        "original_case_id",
        "source_folder",
        "selected_original_image_relpath",
        "image_path",
        "task_l6_label",
        "task_l7_label",
        "label_rule",
        "strict_task7_eval",
    ]
    keep = [c for c in metadata_cols if c in external_df.columns]
    pred_df = pred_df.merge(external_df[keep].drop_duplicates("case_id"), on="case_id", how="left")

    dev_threshold = select_dev_threshold(run_dir, float(args.threshold_grid_step))
    pred_df["pred_idx_dev_bacc_threshold"] = (pred_df["prob_high_risk_group"] >= dev_threshold).astype(int)
    pred_path = output_dir / f"{model_tag}_external_locked_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    subtype_frames: list[pd.DataFrame] = []
    for threshold_name, threshold in [("fixed_0.50", 0.5), ("dev_oof_bacc", dev_threshold)]:
        for subset_name, mask in [
            ("all", np.ones(len(pred_df), dtype=bool)),
            ("strict", pred_df.get("strict_task7_eval", pd.Series([1] * len(pred_df))).astype(int).to_numpy() == 1),
        ]:
            group = pred_df.loc[mask].copy()
            metrics = metric_dict(
                group["label_idx"].astype(int).to_numpy(),
                group["prob_high_risk_group"].astype(float).to_numpy(),
                threshold,
            )
            metrics.update(
                {
                    "model_tag": model_tag,
                    "threshold_name": threshold_name,
                    "subset": subset_name,
                    "run_dir": str(run_dir),
                    "external_predictions_csv": str(pred_path),
                    "input_variant": input_variant,
                    "image_size": image_size,
                    "folds": len(checkpoints),
                }
            )
            summary_rows.append(metrics)
            subtype_frames.append(summarize_subtypes(group, threshold, model_tag, f"{subset_name}:{threshold_name}"))

    return pd.DataFrame(summary_rows), pd.concat(subtype_frames, ignore_index=True) if subtype_frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))

    external_df = load_external_df(Path(args.external_registry_csv))
    device = resolve_device(args.device)
    print(f"[data] external={len(external_df)} device={device}", flush=True)

    summaries: list[pd.DataFrame] = []
    subtype_summaries: list[pd.DataFrame] = []
    for item in [x.strip() for x in args.run_dirs.split(",") if x.strip()]:
        run_dir = Path(item)
        summary, subtype_summary = evaluate_run(run_dir, external_df, args, device, output_dir)
        summaries.append(summary)
        subtype_summaries.append(subtype_summary)

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(output_dir / "dinov3_locked_external_summary.csv", index=False, encoding="utf-8-sig")
    if subtype_summaries:
        pd.concat(subtype_summaries, ignore_index=True).to_csv(
            output_dir / "dinov3_locked_external_subtype_summary.csv", index=False, encoding="utf-8-sig"
        )
    write_json(output_dir / "dinov3_locked_external_report.json", summary_df.to_dict(orient="records"))
    print(summary_df.sort_values(["subset", "balanced_accuracy", "accuracy"], ascending=[True, False, False]).to_string(index=False), flush=True)
    print(f"[done] outputs saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
