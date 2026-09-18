import argparse
import importlib.metadata
import json
import platform
import runpy
import time
from pathlib import Path

import torch
from math_verify.errors import TimeoutException
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_irc_sources import load_math_author
from materialize_outcome_score_data import file_sha256, payload_sha256
from projection_transport_steering.outcome_score import score_math_outputs as score_output
from projection_transport_steering.outcome_score_runtime import run_math_batch as run_batch


def load_packet(config: dict, root: Path) -> list[dict]:
    path = root / config["packet"]
    if file_sha256(path) != config["packet_sha256"]:
        raise ValueError("runtime packet hash mismatch")
    packet = json.loads(path.read_text())
    expected = packet.pop("content_sha256")
    if payload_sha256(packet) != expected:
        raise ValueError("runtime packet content hash mismatch")
    rows = packet["rows"]
    ids = [row["cluster_id"] for row in rows]
    if (
        ids != config["cluster_ids"]
        or len(set(ids)) != len(ids)
        or len(ids) != config["num_questions"]
    ):
        raise ValueError("runtime packet identities differ")
    if (
        packet["allocation"] != "runtime"
        or packet["source_allocation"] != "fit"
        or not all(
            row["allocation"] == "fit" and row["split"] == "train" and row["reference_eligible"]
            for row in rows
        )
    ):
        raise ValueError("runtime packet violates the training admission boundary")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    rows = load_packet(config, root)
    condition = config["conditions"][args.condition]
    if condition["batch"] < 1 or condition["steps"] < 4 or condition["steps"] % 2:
        raise ValueError("invalid workload dimensions")
    names = {"torch", "transformers"} | config["expected_versions"].keys()
    versions = {name: importlib.metadata.version(name) for name in names}
    if any(versions.get(name) != value for name, value in config["expected_versions"].items()):
        raise ValueError("runtime package version mismatch")
    for name, expected in config["model_files_sha256"].items():
        if file_sha256(args.model / name) != expected:
            raise ValueError(f"model file hash mismatch: {name}")
    source_hashes = {name: file_sha256(root / name) for name in config["source_sha256"]}
    if source_hashes != config["source_sha256"]:
        raise ValueError("result-generating source hash mismatch")
    extract, equivalent = load_math_author(root)
    boxed = runpy.run_path(str(root / ".external/math-author-source/modeling/dataset/util.py"))[
        "last_boxed_only_string"
    ]
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if config.get("attention") in ("fp32_prefill_flex", "fp32_prefill_flex_eager"):
        from projection_transport_steering.attention_backend import register_fp32_prefill_flex

        register_fp32_prefill_flex()
    receipt = {
        "condition": args.condition,
        "status": "running",
        "expected_ids": config["cluster_ids"],
        "completed_ids": [],
        "chunk_sha256": {},
        "config_sha256": file_sha256(args.config),
        "packet_sha256": config["packet_sha256"],
        "source_sha256": source_hashes,
        "versions": versions,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scientific_method_comparison": False,
        "device": args.device,
    }
    failure = None
    try:
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                attn_implementation=config.get("attention", "sdpa"),
                local_files_only=True,
            )
            .to(args.device)
            .eval()
            .requires_grad_(False)
        )
        receipt["attention"] = model.config._attn_implementation
        receipt["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, padding_side="left"
        )
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(rows), condition["batch"]):
            chunk = run_batch(
                model,
                tokenizer,
                rows[start : start + condition["batch"]],
                config,
                condition,
                (extract, equivalent, boxed),
                reference_path=args.output / f"reference-{start:04d}.json",
                hash_payload=payload_sha256,
            )
            path = args.output / f"batch-{start:04d}.json"
            with path.open("x") as handle:
                json.dump(chunk, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
            receipt["chunk_sha256"][path.name] = file_sha256(path)
            receipt["completed_ids"].extend(row["cluster_id"] for row in chunk["rows"])
            print(
                json.dumps(
                    {
                        "condition": args.condition,
                        "completed": len(receipt["completed_ids"]),
                        "reference_seconds": chunk["reference_seconds"],
                        "generated_tokens": sum(row["generated_tokens"] for row in chunk["rows"]),
                    }
                ),
                flush=True,
            )
            if not all(chunk["checks"].values()) or any(
                row["evaluator_errors"] for row in chunk["rows"]
            ):
                raise RuntimeError("technical runtime conformance failed; chunk retained")
        if receipt["completed_ids"] != receipt["expected_ids"] or len(
            set(receipt["completed_ids"])
        ) != len(rows):
            raise RuntimeError("completed identities do not match expected packet")
        receipt["status"] = "pass"
        receipt["peak_memory_bytes"] = (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        )
    except (Exception, TimeoutException) as error:
        failure = error
        receipt.update(status="fail", failure_type=type(error).__name__, failure_message=str(error))
    receipt["reference_sha256"] = {
        path.name: file_sha256(path) for path in sorted(args.output.glob("reference-*.json"))
    }
    receipt["total_elapsed_seconds"] = time.perf_counter() - started
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "condition": args.condition,
                "status": receipt["status"],
                "completed": len(receipt["completed_ids"]),
            }
        ),
        flush=True,
    )
    if failure is not None:
        raise RuntimeError("runtime failed; immutable receipt retained") from failure


if __name__ == "__main__":
    main()
