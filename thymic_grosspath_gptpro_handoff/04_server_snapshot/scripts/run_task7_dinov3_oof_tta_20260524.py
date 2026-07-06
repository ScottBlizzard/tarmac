from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_finetune_old_third_20260523 import (  # noqa: E402
    DINOv3FineTuneModel,
    load_available_folds,
    make_image_df,
    prepare_fold_data,
)
from thymic_baseline.config import get_task  # noqa: E402
from thymic_baseline.cropping import extract_specimen_crop  # noqa: E402
from thymic_baseline.metrics import summarize_prediction_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic TTA OOF evaluation for Task7 DINOv3 fold models.")
    parser.add_argument("--run-dir", required=True)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def apply_view(image: Image.Image, view: str) -> Image.Image:
    if view == "orig":
        return image
    if view == "hflip":
        return ImageOps.mirror(image)
    if view == "vflip":
        return ImageOps.flip(image)
    if view == "hvflip":
        return ImageOps.flip(ImageOps.mirror(image))
    if view == "rot90":
        return image.transpose(Image.Transpose.ROTATE_90)
    if view == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)
    if view == "rot270":
        return image.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"Unsupported TTA view: {view}")


class TTADataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, input_variant: str, image_size: int, view: str) -> None:
        self.frame = frame.reset_index(drop=True)
        self.input_variant = input_variant
        self.view = view
        self.transform = normalize_transform(image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = apply_view(self._load_image(str(row["image_path"])), self.view)
        if self.input_variant == "whole":
            inputs = self.transform(image)
        elif self.input_variant == "crop":
            inputs = self.transform(extract_specimen_crop(image))
        elif self.input_variant == "whole_plus_crop":
            inputs = (self.transform(image), self.transform(extract_specimen_crop(image)))
        else:
            raise ValueError(f"Unsupported input variant: {self.input_variant}")
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return inputs, label, str(row["case_id"]), str(row["image_name"])


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, tuple):
        return tuple(move_to_device(item, device) for item in inputs)
    if isinstance(inputs, list):
        return [move_to_device(item, device) for item in inputs]
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def load_model(run_args: dict[str, Any], checkpoint: Path, device: torch.device) -> DINOv3FineTuneModel:
    task = get_task(str(run_args["task"]))
    model = DINOv3FineTuneModel(
        model_name=str(run_args["model_name"]),
        global_pool=str(run_args.get("global_pool", "token")),
        input_variant=str(run_args["input_variant"]),
        num_classes=task.num_classes,
        dropout=float(run_args.get("dropout", 0.2)),
        head_type=str(run_args.get("head_type", "mlp")),
        hidden_dim=int(run_args.get("hidden_dim", 512)),
    )
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_view(
    model: DINOv3FineTuneModel,
    frame: pd.DataFrame,
    input_variant: str,
    image_size: int,
    view: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], list[str], list[int], np.ndarray]:
    dataset = TTADataset(frame, input_variant=input_variant, image_size=image_size, view=view)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    case_ids: list[str] = []
    image_names: list[str] = []
    labels: list[int] = []
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, batch_labels, batch_case_ids, batch_image_names in tqdm(loader, desc=f"tta {view}", leave=False, dynamic_ncols=True):
            logits = model(move_to_device(inputs, device))
            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(batch_probs)
            labels.extend(int(x) for x in batch_labels.cpu().numpy().tolist())
            case_ids.extend(str(x) for x in batch_case_ids)
            image_names.extend(str(x) for x in batch_image_names)
    return case_ids, image_names, labels, np.concatenate(probs, axis=0)


def add_metadata(pred: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "master_fold_id",
    ]
    return pred.merge(metadata[[c for c in cols if c in metadata.columns]], on="case_id", how="left")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = read_json(run_dir / "args.json")
    task = get_task(str(run_args["task"]))
    device = resolve_device(args.device)
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    metadata = make_image_df(str(run_args["registry_csv"]), str(run_args["split_csv"]), task)
    fold_ids = load_available_folds(str(run_args["split_csv"]))

    all_rows: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        print(f"[fold {fold_id}] loading model", flush=True)
        _, _, test_df = prepare_fold_data(str(run_args["registry_csv"]), str(run_args["split_csv"]), task, int(fold_id), len(fold_ids))
        model = load_model(run_args, run_dir / f"fold_{fold_id}" / "best_model.pt", device)
        view_probs: list[np.ndarray] = []
        base_case_ids: list[str] | None = None
        base_image_names: list[str] | None = None
        base_labels: list[int] | None = None
        for view in views:
            case_ids, image_names, labels, probs = predict_view(
                model=model,
                frame=test_df,
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
                raise RuntimeError(f"Case order mismatch for fold {fold_id} view {view}")
            view_probs.append(probs)
        assert base_case_ids is not None and base_image_names is not None and base_labels is not None
        mean_probs = np.mean(np.stack(view_probs, axis=0), axis=0)
        pred_idx = mean_probs.argmax(axis=1)
        fold_df = pd.DataFrame(
            {
                "case_id": base_case_ids,
                "label_idx": base_labels,
                "pred_idx": pred_idx.astype(int),
                "n_images": 1,
                "prob_low_risk_group": mean_probs[:, 0],
                "prob_high_risk_group": mean_probs[:, 1],
                "fold_id": int(fold_id),
                "image_name": base_image_names,
                "tta_views": ",".join(views),
            }
        )
        all_rows.append(fold_df)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred_df = pd.concat(all_rows, ignore_index=True)
    pred_df = add_metadata(pred_df, metadata)
    pred_df.to_csv(out_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
    metrics = summarize_prediction_frame(pred_df, task.class_names)
    pd.DataFrame(
        [
            {
                "split": "test_oof",
                "level": "case",
                "aggregation": "tta_mean",
                **metrics,
            }
        ]
    ).to_csv(out_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "args.json").write_text(
        json.dumps({"source_run_dir": str(run_dir), "views": views, "source_args": run_args}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
