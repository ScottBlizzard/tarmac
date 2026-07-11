from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_task7_dense_external_20260711 import summarize
from run_task7_lora_dense_finetune_20260711 import LoRADenseTask7Model, load_trainable_state, move_to_device
from thymic_baseline.train import build_dataloader

DEFAULT_REGISTRY = (
    "/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/"
    "task7_four_domain_master_registry.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked image-only external evaluation for LoRA dense CV models.")
    parser.add_argument("--internal-run-dir", required=True)
    parser.add_argument("--registry-csv", default=DEFAULT_REGISTRY)
    parser.add_argument("--domains", default="strict_external,new_external_160")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def external_frame(registry_csv: str, domains: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        registry_csv, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig"
    )
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    frame = frame[frame["domain"].astype(str).isin(domains)].copy()
    frame = frame.sort_values(["domain", "case_id"]).drop_duplicates("case_id").reset_index(drop=True)
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["image_name"] = frame["image_name"].fillna(frame["image_path"].map(lambda value: Path(str(value)).name))
    return frame


def infer(model, loader, device: torch.device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for inputs, labels, case_ids, image_names in loader:
            inputs = move_to_device(inputs, device)
            probability = torch.softmax(model(inputs), dim=1)[:, 1].cpu().numpy()
            for index, case_id in enumerate(case_ids):
                rows.append(
                    {
                        "case_id": str(case_id),
                        "image_name": str(image_names[index]),
                        "label_idx": int(labels[index].item()),
                        "prob_high": float(probability[index]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.internal_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    frame = external_frame(args.registry_csv, domains)
    fold_dirs = sorted(run_dir.glob("fold_*"))
    if len(fold_dirs) != 5:
        raise ValueError(f"Expected five LoRA fold directories, found {len(fold_dirs)}")
    fold_predictions = []
    device = torch.device(args.device)
    for fold_dir in fold_dirs:
        config = json.loads((fold_dir / "run_config.json").read_text(encoding="utf-8"))
        model_args = argparse.Namespace(**config)
        fold_id = int(config["fold_id"])
        loader = build_dataloader(
            frame,
            input_variant=config["input_variant"],
            image_size=int(config["image_size"]),
            is_train=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = LoRADenseTask7Model(model_args, num_classes=2, fold_id=fold_id).to(device)
        load_trainable_state(model, fold_dir / "best_trainable_state.pt", device)
        prediction = infer(model, loader, device).rename(columns={"prob_high": f"prob_high_fold_{fold_id}"})
        fold_predictions.append(prediction)
        del model
        torch.cuda.empty_cache()

    prediction = fold_predictions[0]
    for fold_prediction in fold_predictions[1:]:
        prediction = prediction.merge(
            fold_prediction,
            on=["case_id", "image_name", "label_idx"],
            how="inner",
            validate="one_to_one",
        )
    fold_columns = sorted(column for column in prediction.columns if column.startswith("prob_high_fold_"))
    prediction["prob_high"] = prediction[fold_columns].mean(axis=1)
    prediction["pred_idx"] = (prediction["prob_high"] >= 0.5).astype(int)
    metadata_columns = [
        "case_id",
        "original_case_id",
        "domain",
        "source_dataset",
        "task_l6_label",
        "task_l7_label",
        "image_path",
    ]
    prediction = prediction.merge(frame[metadata_columns], on="case_id", how="left", validate="one_to_one")
    prediction.to_csv(output_dir / "external_lora_dense_predictions.csv", index=False, encoding="utf-8-sig")

    rows = [{"group_type": "overall", "group": "all", **summarize(prediction, "prob_high")}]
    for group_column in ["domain", "source_dataset"]:
        for group_name, group in prediction.groupby(group_column, dropna=False):
            rows.append(
                {"group_type": group_column, "group": str(group_name), **summarize(group, "prob_high")}
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "external_lora_dense_metrics.csv", index=False, encoding="utf-8-sig")
    subtype_rows = []
    for subtype, group in prediction.groupby("task_l6_label"):
        subtype_rows.append(
            {
                "subtype": subtype,
                "n": len(group),
                "risk_accuracy": float((group["pred_idx"] == group["label_idx"]).mean()),
                "mean_prob_high": float(group["prob_high"].mean()),
            }
        )
    pd.DataFrame(subtype_rows).to_csv(
        output_dir / "external_lora_dense_subtype_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "internal_run_dir": str(run_dir),
                "registry_csv": args.registry_csv,
                "domains": domains,
                "fold_columns": fold_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
