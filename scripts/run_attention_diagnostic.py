import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import runpy
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch

from materialize_outcome_score_data import file_sha256


LABELS = ("grad_output", "no_grad_output", "q_grad", "k_grad", "v_grad")
TENSORS = (
    "query",
    "key",
    "value",
    "q64",
    "k64",
    "v64",
    "gradient",
    "mask",
    "keep",
    "valid",
    "predicate",
    "output",
    "oracle",
    "inference_output",
)


def checked_config(path: Path, expected: str, root: Path) -> dict:
    if file_sha256(path) != expected:
        raise ValueError("config hash mismatch")
    config = json.loads(path.read_text())
    for base, hashes in (
        (root, config["source_sha256"]),
        (Path("/"), config["runtime_files_sha256"]),
    ):
        for name, digest in hashes.items():
            if file_sha256(base / name) != digest:
                raise ValueError(f"source hash mismatch: {name}")
    for name, version in config["expected_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"runtime version mismatch: {name}")
    return config


def diagnose_cell(helper, kwargs: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    original = torch.testing.assert_close
    comparisons, state = [], {}
    row = {"status": "pass", "error": None, "traceback": None, "capture_error": None}

    def observed(*args, **options):
        caller = inspect.currentframe().f_back
        if caller.f_code is not helper.__code__:
            return original(*args, **options)
        item = {"label": LABELS[len(comparisons)], "status": "running"}
        comparisons.append(item)
        try:
            original(*args, **options)
        except AssertionError:
            item["status"] = "fail"
            raise
        except Exception:
            item["status"] = "error"
            raise
        item["status"] = "pass"
        if len(comparisons) == len(LABELS):
            state.update(caller.f_locals)

    started = time.perf_counter()
    try:
        with patch.object(torch.testing, "assert_close", observed):
            helper(**kwargs)
    except Exception as error:
        row.update(
            status="fail",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        for frame, _ in traceback.walk_tb(error.__traceback__):
            if frame.f_code is helper.__code__:
                state.update(frame.f_locals)
        del frame
    if row["status"] == "pass" and len(comparisons) != len(LABELS):
        row.update(status="incomplete", error="helper returned without all five comparisons")
    row["helper_detection_seconds"] = time.perf_counter() - started
    row["comparisons"] = comparisons
    row["numerical_status"] = row["status"]
    row["tensor_sha256"], row["tensor_metadata"] = {}, {}
    try:
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in state.values()):
            torch.cuda.synchronize()
        packet = {"tensors": {}, "gradients": {}, "block": None, "block_shape": None}
        for name in TENSORS:
            tensor = state.get(name)
            if not isinstance(tensor, torch.Tensor):
                continue
            cpu = tensor.detach().cpu()
            packet["tensors"][name] = cpu
            row["tensor_sha256"][name] = hashlib.sha256(
                cpu.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            row["tensor_metadata"][name] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
            }
            if name in ("query", "key", "value", "q64", "k64", "v64"):
                packet["gradients"][name] = (
                    None if tensor.grad is None else tensor.grad.detach().cpu()
                )
        block = state.get("block")
        if block is not None:
            packet["block"] = {"BLOCK_SIZE": list(block.BLOCK_SIZE)}
            packet["block_shape"] = list(block.shape) if hasattr(block, "shape") else None
            for prefix in ("kv", "q", "full_kv", "full_q"):
                for suffix in ("num_blocks", "indices"):
                    name = f"{prefix}_{suffix}"
                    tensor = getattr(block, name)
                    packet["block"][name] = None if tensor is None else tensor.detach().cpu()
        with (output / "tensors.pt").open("xb") as handle:
            torch.save(packet, handle)
        row["packet_sha256"] = file_sha256(output / "tensors.pt")
    except Exception as error:
        row.update(status="capture_error", capture_error=f"{type(error).__name__}: {error}")
    row["total_cell_seconds"] = time.perf_counter() - started
    with (output / "receipt.json").open("x") as handle:
        json.dump(row, handle, indent=2)
    state.clear()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = checked_config(args.config, args.config_sha256, root)
    lengths = config["arms"][args.arm]
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA device unavailable")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    helper = runpy.run_path(str(root / "tests/test_attention_backend.py"))[config["helper_name"]]
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "arm": args.arm,
        "planned_lengths": lengths,
        "config_sha256": args.config_sha256,
        "cells": [],
        "status": "pass",
        "engine_admitted": False,
    }
    for index, length in enumerate(lengths):
        row = diagnose_cell(
            helper,
            {**config["helper_kwargs"], "length": length},
            args.output / f"cell-{index:02d}-q{length}",
        )
        receipt["cells"].append({"length": length, **row})
        print(
            json.dumps(
                {
                    "event": "cell",
                    "length": length,
                    "status": row["status"],
                    "comparisons": row["comparisons"],
                }
            ),
            flush=True,
        )
        gc.collect()
        if row["status"] != "pass":
            receipt["status"] = row["status"]
            break
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    raise SystemExit(0 if receipt["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
