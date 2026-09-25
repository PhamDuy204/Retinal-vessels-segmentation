"""Export a validated training state dict as inference-only safetensors weights."""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(source, dict) or "model_state_dict" not in source:
        parser.error("Expected a training checkpoint with model_state_dict")
    if source.get("model", "our_net") != "our_net":
        parser.error("Only SGMA-Net our_net checkpoints are supported")
    output = args.output or args.checkpoint.with_suffix(".safetensors")
    weights = {key: value.detach().cpu().contiguous()
               for key, value in source["model_state_dict"].items()}
    save_file(weights, str(output), metadata={"model": "our_net"})
    print(output)


if __name__ == "__main__":
    main()
