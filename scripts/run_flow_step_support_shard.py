import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch
import transformers
from flas.generate import load_generator

from projection_transport_steering.flow_step_support import (
    condition_specifications,
    evaluate_flow_step_support,
)
from projection_transport_steering.residual_flow import flas_model_logits
from projection_transport_steering.semantic_outcome_runtime import encode_candidates
from run_residual_flow_design_shard import (
    file_sha256,
    load_stage_a,
    model_manifest,
    verify_sources,
)

SEED = 20260907


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--flas-source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept-ids", type=int, nargs="+", required=True)
    parser.add_argument("--layer", type=int, default=20)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    verify_sources(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if len(set(args.concept_ids)) != len(args.concept_ids):
        raise ValueError("concept IDs must be unique")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    data = json.loads(args.data.read_text())
    matched, _, stage_a_hashes = load_stage_a(args.stage_a)
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
    first_id = str(args.concept_ids[0])
    first = matched[first_id]["construction"][0]
    replay_batch, _ = encode_candidates(
        generator.tokenizer,
        first["input"],
        [first["output"]],
        device,
    )
    first_hidden, first_mask = generator.encode_concept(data["concepts"][first_id]["concept"])
    with torch.inference_mode():
        reference = flas_model_logits(
            generator,
            replay_batch,
            first_hidden,
            first_mask,
        )
        replay = flas_model_logits(
            generator,
            replay_batch,
            first_hidden,
            first_mask,
            intervention={"steps": [0, 1, 2]},
        )
    replay_error = float((reference - replay).abs().max())
    if replay_error != 0.0:
        raise RuntimeError("explicit full-step replay failed")
    concepts = []
    for concept_id in args.concept_ids:
        concept_started = time.perf_counter()
        key = str(concept_id)
        try:
            source = data["concepts"][key]
            rows = matched[key]
            concept_hidden, concept_mask = generator.encode_concept(source["concept"])
            result = evaluate_flow_step_support(
                generator,
                generator.tokenizer,
                rows["construction"],
                rows["evaluation"],
                data["neutral"],
                concept_hidden,
                concept_mask,
                device,
            )
            concepts.append(
                {
                    **result,
                    "concept": source["concept"],
                    "concept_id": concept_id,
                    "elapsed_seconds": time.perf_counter() - concept_started,
                    "status": "pass",
                }
            )
        except Exception as error:
            concepts.append(
                {
                    "concept_id": concept_id,
                    "elapsed_seconds": time.perf_counter() - concept_started,
                    "error": str(error),
                    "status": "fail",
                }
            )
    counter.remove()
    torch.cuda.synchronize()
    return {
        "concepts": concepts,
        "condition_specifications": condition_specifications(),
        "cuda": torch.version.cuda,
        "data_sha256": file_sha256(args.data),
        "elapsed_seconds": time.perf_counter() - started,
        "flas_config_sha256": file_sha256(args.flow_checkpoint.parent / "config.json"),
        "flas_generate_sha256": file_sha256(args.flas_source / "flas" / "generate.py"),
        "flas_model_sha256": file_sha256(args.flas_source / "flas" / "model.py"),
        "flas_weights_sha256": file_sha256(args.flow_checkpoint),
        "flowtime": 2.0,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "layer": args.layer,
        "model_manifest": model_manifest(args.model),
        "model_snapshot": str(args.model.resolve()),
        "n_steps": 3,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "planned_concepts": len(args.concept_ids),
        "platform": platform.platform(),
        "replay_max_abs_logit_error": replay_error,
        "stage_a_input_hashes": stage_a_hashes,
        "status": "pass",
        "teacher_forced_forward_calls": forward_calls,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = run(args)
        serialized = json.dumps(receipt, sort_keys=True, allow_nan=False)
        receipt["condition_serialization_exact"] = json.loads(serialized) == receipt
        path = args.output / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        if json.loads(path.read_text()) != receipt:
            raise RuntimeError("receipt serialization failed")
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise


if __name__ == "__main__":
    main()
