from __future__ import annotations

import argparse
import gc
import json

import timm
import torch


DEFAULT_MODELS = [
    "vit_large_patch16_dinov3_qkvb.lvd1689m",
    "vit_large_patch16_siglip_384.v2_webli",
    "eva02_large_patch14_clip_336.merged2b",
    "aimv2_large_patch14_336.apple_pt",
    "convnext_base.dinov3_lvd1689m",
]


def shape_of(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape_of(item) for item in value]
    if isinstance(value, dict):
        return {str(key): shape_of(item) for key, item in value.items()}
    return str(type(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--image-size", type=int, default=0, help="0 uses the model's configured input size")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    for model_name in args.models:
        record = {"model_name": model_name}
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")
            model.eval().to(device)
            configured_size = getattr(getattr(model, "patch_embed", None), "img_size", None)
            if isinstance(configured_size, tuple):
                configured_size = configured_size[0]
            if not configured_size:
                configured_size = getattr(model, "default_cfg", {}).get("input_size", (3, 224, 224))[-1]
            image_size = int(args.image_size or configured_size)
            sample = torch.zeros(1, 3, image_size, image_size, device=device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                dense = model.forward_features(sample)
                pooled = model.forward_head(dense, pre_logits=True)
            patch_embed = getattr(model, "patch_embed", None)
            patch_size = getattr(patch_embed, "patch_size", None)
            record.update(
                {
                    "class": type(model).__name__,
                    "num_features": int(getattr(model, "num_features", 0)),
                    "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", 0)),
                    "patch_size": list(patch_size) if isinstance(patch_size, tuple) else patch_size,
                    "image_size": image_size,
                    "forward_features_shape": shape_of(dense),
                    "forward_head_shape": shape_of(pooled),
                    "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1)
                    if device.type == "cuda"
                    else 0.0,
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic script
            record["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if "model" in locals():
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
