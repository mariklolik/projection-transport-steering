import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from projection_transport_steering.cacheback_runner import run_sentinel_shard


def horizon_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--horizons", type=horizon_tuple, default=(0, 2, 4, 8))
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--target", type=float, default=0.05)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    data = json.loads(args.data.read_text())
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).cuda().eval()
    model.requires_grad_(False)
    run_sentinel_shard(
        model,
        data,
        args.data,
        args.output,
        args.shard_index,
        args.shard_count,
        args.layer,
        args.horizons,
        args.trajectories,
        args.target,
    )


if __name__ == "__main__":
    main()
