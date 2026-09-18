import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.outcome_score_runtime import TorchLayerTrace, generate_rollout
from projection_transport_steering.outcome_score_sentinel import (
    select_sentinel_rows,
    sentinel_protocol,
)
from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction

BASIS_SHA256 = "8094ab483c43bb9c2e12024c6f2f4cae59077d47d385e2a9757b77586d4a1db7"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(path: Path) -> list[dict[str, object]]:
    return [
        {"path": str(file.relative_to(path)), "size": file.resolve().stat().st_size}
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-ids", nargs="+", required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--protocol-version", type=int, required=True)
    parser.add_argument(
        "--stage",
        choices=("basis_completion", "class_mix", "fit", "termination"),
        required=True,
    )
    return parser.parse_args()


def replay_rollout_index(protocol: dict[str, object]) -> int:
    return int(protocol["rollout_indices"][0])


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    protocol = sentinel_protocol(args.protocol_version, args.stage)
    expected_source_sha256 = str(protocol.get("source_sha256", BASIS_SHA256))
    allocation = str(protocol.get("allocation", "basis"))
    if file_sha256(args.data) != expected_source_sha256:
        raise RuntimeError("source packet hash mismatch")
    if args.layer != protocol["layer"] or len(args.group_ids) != protocol["group_count"]:
        raise ValueError("sentinel request differs from the frozen protocol")
    expected_model_sha256 = protocol.get("model_config_sha256")
    if (
        expected_model_sha256 is not None
        and file_sha256(args.model / "config.json") != expected_model_sha256
    ):
        raise ValueError("model config differs from the frozen protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.reset_peak_memory_stats()
    packet = json.loads(args.data.read_text())
    sources = select_sentinel_rows(packet, args.group_ids, allocation=allocation)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    trace = TorchLayerTrace(model, args.layer)
    rows = []
    traces = {}
    forward_calls = 0

    def count_forward(module, inputs):
        nonlocal forward_calls
        forward_calls += 1

    counter = model.register_forward_pre_hook(count_forward)
    try:
        for source in sources:
            for rollout_index in protocol["rollout_indices"]:
                row, states = generate_rollout(
                    model,
                    tokenizer,
                    trace,
                    source,
                    rollout_index,
                    protocol["max_new_tokens"],
                    device,
                    enable_thinking=protocol["enable_thinking"],
                    answer_format=protocol["answer_format"],
                )
                row.update(
                    {
                        "answer": int(source["answer"]),
                        "index": int(source["index"]),
                        "problem": str(source["problem"]),
                        "year": int(source["year"]),
                    }
                )
                rows.append(row)
                traces[str(row["generation_id"])] = states.to(torch.bfloat16)
        first = sources[0]
        replay_index = replay_rollout_index(protocol)
        original = next(
            row
            for row in rows
            if row["generation_id"] == f"{first['group_id']}:{replay_index}"
        )
        width = next(iter(traces.values())).shape[1]
        zero = TorchLayerAction(model, args.layer, AdditiveAction(torch.zeros(width)))
        with zero.installed():
            replay, _ = generate_rollout(
                model,
                tokenizer,
                trace,
                first,
                replay_index,
                protocol["max_new_tokens"],
                device,
                enable_thinking=protocol["enable_thinking"],
                answer_format=protocol["answer_format"],
            )
    finally:
        counter.remove()
    if replay["completion"] != original["completion"]:
        raise RuntimeError("zero-action replay changed generation")
    identifiers = [str(row["generation_id"]) for row in rows]
    expected = len(sources) * len(protocol["rollout_indices"])
    if (
        len(rows) != expected
        or len(set(identifiers)) != expected
        or set(identifiers) != set(traces)
    ):
        raise RuntimeError("sentinel shard is incomplete")
    torch.save(traces, args.output / "traces.pt")
    (args.output / "rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    loaded = torch.load(args.output / "traces.pt", weights_only=True)
    if set(loaded) != set(traces) or any(
        not torch.equal(loaded[key], traces[key]) for key in traces
    ):
        raise RuntimeError("trace serialization mismatch")
    torch.cuda.synchronize()
    return {
        "allocation": allocation,
        "answer_format": protocol["answer_format"],
        "basis_sha256": BASIS_SHA256,
        "cuda": torch.version.cuda,
        "elapsed_seconds": time.perf_counter() - started,
        "enable_thinking": protocol["enable_thinking"],
        "evaluator": protocol.get("evaluator", "custom"),
        "forward_calls": forward_calls,
        "gpu": torch.cuda.get_device_name(0),
        "group_ids": args.group_ids,
        "host": socket.gethostname(),
        "layer": args.layer,
        "max_new_tokens": protocol["max_new_tokens"],
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "model_manifest": model_manifest(args.model),
        "model_snapshot": str(args.model.resolve()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "platform": platform.platform(),
        "protocol_version": args.protocol_version,
        "replay_exact": True,
        "rollout_indices": protocol["rollout_indices"],
        "rows": len(rows),
        "rows_sha256": file_sha256(args.output / "rows.json"),
        "source_sha256": expected_source_sha256,
        "status": "pass",
        "stage": args.stage,
        "torch": torch.__version__,
        "traces_sha256": file_sha256(args.output / "traces.pt"),
        "transformers": transformers.__version__,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = run(args)
        (args.output / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
        )
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise


if __name__ == "__main__":
    main()
