import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch
import transformers
from analyze_flow_step_support import load_packet
from flas.generate import load_generator
from run_residual_flow_design_shard import (
    file_sha256,
    load_stage_a,
    model_manifest,
    verify_sources,
)

from projection_transport_steering.flow_step_generation import generate_condition
from projection_transport_steering.flow_step_support import condition_specifications

SEED = 20260907


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--flas-source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept-ids", type=int, nargs="+", required=True)
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--layer", type=int, default=20)
    return parser.parse_args()


def design_concepts(directory: Path) -> tuple[dict[int, dict[str, object]], dict[str, str]]:
    receipts, resources = load_packet(directory)
    concepts = {
        int(concept["concept_id"]): concept
        for receipt in receipts
        for concept in receipt["concepts"]
    }
    hashes = {shard: str(resource["receipt_sha256"]) for shard, resource in resources.items()}
    return concepts, hashes


def run(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    verify_sources(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if len(set(args.concept_ids)) != len(args.concept_ids):
        raise ValueError("concept IDs must be unique")
    if args.prompt_limit < 1 or args.prompt_limit > 8:
        raise ValueError("prompt limit must be between one and eight")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    data = json.loads(args.data.read_text())
    matched, _, stage_a_hashes = load_stage_a(args.stage_a)
    design, design_hashes = design_concepts(args.design)
    specifications = condition_specifications()
    expected_names = [str(specification["name"]) for specification in specifications]
    for concept_id in args.concept_ids:
        concept = design.get(concept_id)
        if (
            concept is None
            or concept["status"] != "pass"
            or concept["condition_order"] != expected_names
            or set(concept["conditions"]) != set(expected_names)
        ):
            raise RuntimeError(f"invalid design concept {concept_id}")
    generator = load_generator(
        args.flow_checkpoint,
        model_id=args.model,
        layer=args.layer,
        num_blocks=1,
    )
    generator.llm.requires_grad_(False)
    generator.flow_fn.requires_grad_(False)
    generator.concept_enc.requires_grad_(False)
    forward_calls = 0

    def count_forward(module, inputs):
        nonlocal forward_calls
        forward_calls += 1

    counter = generator.llm.register_forward_pre_hook(count_forward)
    rows = []
    for concept_id in args.concept_ids:
        key = str(concept_id)
        concept = str(data["concepts"][key]["concept"])
        prompts = [str(row["input"]) for row in matched[key]["evaluation"][: args.prompt_limit]]
        if len(prompts) != args.prompt_limit or len(set(prompts)) != len(prompts):
            raise RuntimeError(f"invalid prompts for concept {concept_id}")
        for specification in specifications:
            generated = generate_condition(
                generator,
                prompts,
                concept,
                specification,
                args.max_new_tokens,
            )
            for row in generated:
                prompt_index = int(row["prompt_idx"])
                condition = str(specification["name"])
                rows.append(
                    {
                        "concept": concept,
                        "concept_id": concept_id,
                        "condition": condition,
                        "generation": row["generation"],
                        "generation_id": f"{concept_id}:{condition}:{prompt_index}",
                        "prompt": row["prompt"],
                        "prompt_index": prompt_index,
                    }
                )
    counter.remove()
    torch.cuda.synchronize()
    expected_rows = len(args.concept_ids) * len(specifications) * args.prompt_limit
    identifiers = [row["generation_id"] for row in rows]
    if len(rows) != expected_rows or len(set(identifiers)) != expected_rows:
        raise RuntimeError("generation packet is incomplete")
    return rows, {
        "condition_specifications": specifications,
        "cuda": torch.version.cuda,
        "data_sha256": file_sha256(args.data),
        "design_receipt_hashes": design_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "flas_config_sha256": file_sha256(args.flow_checkpoint.parent / "config.json"),
        "flas_generate_sha256": file_sha256(args.flas_source / "flas" / "generate.py"),
        "flas_model_sha256": file_sha256(args.flas_source / "flas" / "model.py"),
        "flas_weights_sha256": file_sha256(args.flow_checkpoint),
        "generation_calls": len(args.concept_ids) * len(specifications),
        "generations": len(rows),
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "layer": args.layer,
        "llm_forward_calls": forward_calls,
        "max_new_tokens": args.max_new_tokens,
        "model_manifest": model_manifest(args.model),
        "model_snapshot": str(args.model.resolve()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "planned_concept_ids": args.concept_ids,
        "platform": platform.platform(),
        "prompt_limit": args.prompt_limit,
        "stage_a_input_hashes": stage_a_hashes,
        "status": "pass",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        rows, receipt = run(args)
        generations = args.output / "generations.json"
        generations.write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))
        receipt["generation_serialization_exact"] = json.loads(generations.read_text()) == rows
        receipt["generations_sha256"] = file_sha256(generations)
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
