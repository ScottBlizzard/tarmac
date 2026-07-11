from __future__ import annotations

import argparse

import timm
import torch.nn as nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="vit_large_patch16_dinov3_qkvb.lvd1689m")
    parser.add_argument("--last-blocks", type=int, default=2)
    args = parser.parse_args()
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, global_pool="")
    block_ids = []
    for name, _ in model.named_modules():
        pieces = name.split(".")
        if len(pieces) >= 2 and pieces[0] == "blocks" and pieces[1].isdigit():
            block_ids.append(int(pieces[1]))
    selected = set(sorted(set(block_ids))[-args.last_blocks :])
    print(f"model={args.model_name} blocks={sorted(set(block_ids))} selected={sorted(selected)}")
    for name, module in model.named_modules():
        pieces = name.split(".")
        if len(pieces) >= 2 and pieces[0] == "blocks" and pieces[1].isdigit() and int(pieces[1]) in selected:
            if isinstance(module, nn.Linear):
                print(f"{name}\tin={module.in_features}\tout={module.out_features}\tbias={module.bias is not None}")


if __name__ == "__main__":
    main()
