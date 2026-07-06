from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_oof_tta_20260524 import load_model, predict_view, read_json  # noqa: E402


DEFAULT_EXTERNAL_REGISTRY = (
    PROJECT_ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_external_runs"
    / "20_external_thymoma_carcinoma_64style_wpc_20260522"
    / "external_folder_task7_registry.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked external TTA evaluation for Task7 DINOv3 fold models.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--external-registry-csv", default=str(DEFAULT_EXTERNAL_REGISTRY))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", default="orig,hflip,vflip,hvflip")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = read_json(run_dir / "args.json")
    external = pd.read_csv(args.external_registry_csv, dtype={"case_id": str}).reset_index(drop=True)
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    device = resolve_device(args.device)

    fold_probs: list[np.ndarray] = []
    base_case_ids: list[str] | None = None
    base_image_names: list[str] | None = None
    base_labels: list[int] | None = None
    checkpoints = sorted(run_dir.glob("fold_*/best_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found under {run_dir}")
    for checkpoint in checkpoints:
        fold_id = checkpoint.parent.name
        print(f"[{fold_id}] loading model", flush=True)
        model = load_model(run_args, checkpoint, device)
        view_probs: list[np.ndarray] = []
        for view in views:
            case_ids, image_names, labels, probs = predict_view(
                model=model,
                frame=external,
                input_variant=str(run_args["input_variant"]),
                image_size=int(run_args["image_size"]),
                view=view,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            if base_case_ids is None:
                base_case_ids = case_ids
                base_image_names = image_names
                base_labels = labels
            elif base_case_ids != case_ids:
                raise RuntimeError(f"Case order mismatch for {fold_id} view {view}")
            view_probs.append(probs)
        fold_probs.append(np.mean(np.stack(view_probs, axis=0), axis=0))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert base_case_ids is not None and base_image_names is not None and base_labels is not None
    mean_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
    pred_idx = mean_probs.argmax(axis=1)
    out = external.copy()
    out["prob_low_risk_group"] = mean_probs[:, 0]
    out["prob_high_risk_group"] = mean_probs[:, 1]
    out["pred_idx"] = pred_idx.astype(int)
    out["tta_views"] = ",".join(views)
    out["n_folds"] = len(checkpoints)
    out["correct"] = (out["pred_idx"].astype(int) == out["label_idx"].astype(int)).astype(int)
    out.to_csv(out_dir / "external_tta_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "args.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "external_registry_csv": str(args.external_registry_csv),
                "views": views,
                "source_args": run_args,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
