import argparse
import copy
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM

from materialize_outcome_score_data import file_sha256
from projection_transport_steering.branch_runtime import advance_branches, start_branches
from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.prompt_tokens < args.batch or args.steps < 4 or args.steps % 4:
        raise ValueError("invalid synthetic workload dimensions")
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    tokens = torch.arange(args.prompt_tokens, device=args.device).expand(args.batch, -1).clone()
    tokens = (tokens + torch.arange(args.batch, device=args.device)[:, None] * 31) % (
        model.config.vocab_size - 1
    ) + 1
    mask = torch.ones_like(tokens)
    for index in range(args.batch):
        mask[index, :index] = 0
        tokens[index, :index] = 0
    state = start_branches(tokens, mask, [20260907 + index for index in range(args.batch)])
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise ValueError("model generation configuration must declare EOS")
    settings = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "eos_token_ids": tuple(eos) if isinstance(eos, list) else (int(eos),),
        "pad_token_id": model.generation_config.pad_token_id or 0,
    }
    advance_branches(model, state, steps=2, **settings)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    reference = advance_branches(model, state, steps=args.steps, **settings)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - tick
    site = model.config.num_hidden_layers // 2 - 1
    with TorchLayerAction(model, site, nn.Identity()).installed():
        zero = advance_branches(model, state, steps=args.steps, **settings)
    zero_equal = torch.equal(reference.input_ids, zero.input_ids)
    prefix = advance_branches(model, state, steps=args.steps // 2, **settings)
    saved_cache = copy.deepcopy(prefix.cache)
    split = advance_branches(model, prefix, steps=args.steps // 2, **settings)
    split_equal = torch.equal(reference.input_ids, split.input_ids)
    vector = torch.linspace(-0.01, 0.01, model.config.hidden_size, device=args.device, dtype=dtype)
    with TorchLayerAction(model, site, AdditiveAction(vector)).installed():
        branch = advance_branches(model, prefix, steps=args.steps // 4, **settings)
    advance_branches(model, branch, steps=args.steps // 4, **settings)
    cache_equal = all(
        torch.equal(before.keys, after.keys) and torch.equal(before.values, after.values)
        for before, after in zip(saved_cache.layers, prefix.cache.layers, strict=True)
    )
    passed = zero_equal and split_equal and cache_equal
    generated = int(reference.generated_lengths.sum())
    root = Path(__file__).resolve().parents[1]
    result = {
        "status": "pass" if passed else "fail",
        "synthetic_token_inputs": True,
        "task_scoring_performed": False,
        "scope": "Engineering probe only; this does not replace the 16-question runtime packet or measure research accuracy.",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "dtype": str(dtype),
        "attention": "sdpa",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "batch": args.batch,
        "prompt_tokens": args.prompt_tokens,
        "steps": args.steps,
        "settings": settings,
        "site": site,
        "zero_replay_exact": zero_equal,
        "split_replay_exact": split_equal,
        "parent_cache_unchanged": cache_equal,
        "prefix_terminated_rows": int(prefix.terminated.sum()),
        "generated_lengths": reference.generated_lengths.tolist(),
        "terminated": reference.terminated.tolist(),
        "generated_tokens": generated,
        "reference_elapsed_seconds": elapsed,
        "synthetic_tokens_per_second": generated / elapsed,
        "total_elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "token_output_sha256": hashlib.sha256(
            reference.input_ids.cpu().numpy().tobytes()
        ).hexdigest(),
        "model_snapshot": str(args.model),
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "source_sha256": {
            path: file_sha256(root / path)
            for path in (
                "scripts/run_irc_branch_probe.py",
                "src/projection_transport_steering/branch_runtime.py",
                "src/projection_transport_steering/torch_runtime.py",
            )
        },
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise RuntimeError("branch conformance failed; receipt retained")


if __name__ == "__main__":
    main()
