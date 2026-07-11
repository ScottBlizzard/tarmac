from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import timm
import torch


DEFAULT_MODELS = (
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_large_patch16_dinov3_qkvb.lvd1689m",
    "vit_base_patch16_siglip_384.v2_webli",
    "vit_large_patch16_siglip_384.v2_webli",
    "vit_large_patch16_siglip_512.v2_webli",
    "vit_so400m_patch14_siglip_378.v2_webli",
    "aimv2_large_patch14_336.apple_pt",
    "local_vitamin_large_384",
)

VITAMIN_CHECKPOINT = Path(
    "/root/.cache/huggingface/hub/models--jienengchen--ViTamin-L-384px/"
    "snapshots/a8bd536320237a2a0fc65480bcdfc4a67c00133d/pytorch_model.bin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline audit of already-cached timm image backbones.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def tensor_shapes(value: Any) -> list[list[int]]:
    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, dict):
        shapes = []
        for item in value.values():
            shapes.extend(tensor_shapes(item))
        return shapes
    if isinstance(value, (tuple, list)):
        shapes = []
        for item in value:
            shapes.extend(tensor_shapes(item))
        return shapes
    return []


def configured_input_size(model: torch.nn.Module) -> int:
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    input_size = cfg.get("input_size", (3, 224, 224))
    if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
        return int(input_size[-1])
    return 224


def probe_model(model_name: str, device: torch.device) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {"model_name": model_name, "status": "failed", "error": ""}
    model = None
    try:
        if model_name == "local_vitamin_large_384":
            model = timm.create_model(
                "vitamin_large_384.datacomp1b_clip",
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
            state = torch.load(VITAMIN_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
            trunk_state = {
                key.removeprefix("visual.trunk."): value
                for key, value in state.items()
                if key.startswith("visual.trunk.")
            }
            for suffix in ("weight", "bias"):
                pooled_key = f"fc_norm.{suffix}"
                dense_key = f"norm.{suffix}"
                if pooled_key in trunk_state and dense_key not in trunk_state:
                    trunk_state[dense_key] = trunk_state.pop(pooled_key)
            missing, unexpected = model.load_state_dict(trunk_state, strict=False)
            row["checkpoint_tensor_count"] = len(trunk_state)
            row["missing_key_count"] = len(missing)
            row["unexpected_key_count"] = len(unexpected)
            row["missing_keys"] = json.dumps(missing, separators=(",", ":"))
            row["unexpected_keys"] = json.dumps(unexpected, separators=(",", ":"))
            del state, trunk_state
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")
        image_size = configured_input_size(model)
        row.update(
            {
                "image_size": image_size,
                "num_features": int(getattr(model, "num_features", 0)),
                "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", 0)),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            }
        )
        model = model.eval().to(device)
        inputs = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model.forward_features(inputs)
        row["forward_feature_shapes"] = json.dumps(tensor_shapes(output), separators=(",", ":"))
        row["status"] = "ok"
    except Exception as exc:  # Keep the audit running after an unsupported cached architecture.
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["elapsed_seconds"] = round(time.time() - started, 3)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = ("*dinov3*", "*siglip*", "*aimv2*", "*vitamin*")
    listing = {pattern: timm.list_models(pattern, pretrained=True) for pattern in patterns}
    (output_dir / "timm_pretrained_model_listing.json").write_text(
        json.dumps(listing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for pattern, names in listing.items():
        print(f"{pattern}: {len(names)}")
        print("\n".join(names))
    if args.list_only:
        return

    device = torch.device(args.device)
    rows = []
    for model_name in args.models:
        print(f"PROBE {model_name}", flush=True)
        row = probe_model(model_name, device)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        pd.DataFrame(rows).to_csv(output_dir / "cached_backbone_probe.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
