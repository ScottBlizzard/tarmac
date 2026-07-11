from __future__ import annotations

import json
from pathlib import Path

import torch


CHECKPOINT = Path(
    "/root/.cache/huggingface/hub/models--jienengchen--ViTamin-L-384px/"
    "snapshots/a8bd536320237a2a0fc65480bcdfc4a67c00133d/pytorch_model.bin"
)


def main() -> None:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    print(f"container_type={type(state).__name__}")
    if not isinstance(state, dict):
        return
    for container_key in ("state_dict", "model", "module"):
        nested = state.get(container_key)
        if isinstance(nested, dict):
            print(f"using_nested_container={container_key}")
            state = nested
            break
    rows = []
    for key, value in list(state.items())[:120]:
        rows.append(
            {
                "key": str(key),
                "shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
                "type": type(value).__name__,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"tensor_keys={sum(isinstance(value, torch.Tensor) for value in state.values())}")


if __name__ == "__main__":
    main()
