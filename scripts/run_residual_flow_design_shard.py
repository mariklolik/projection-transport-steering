import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from flas.generate import load_generator

from projection_transport_steering.residual_flow import flas_model_logits
from projection_transport_steering.residual_flow_sentinel import (
    construct_geometry,
    evaluate_quantiles,
)
from projection_transport_steering.semantic_outcome_runtime import encode_candidates

DATA_SHA256 = "59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68"
FLAS_CONFIG_SHA256 = "d5414215889017d76686bda35e00e4399ea7efa66815eceb322e18e3e7c7a46a"
FLAS_GENERATE_SHA256 = "14e1bde99c119e6b970cf7601569b6ab05bedfbd44d65286044af52936f529f7"
FLAS_MODEL_SHA256 = "9058008c85835fadeb0ac9737ebd711520fecbf09b92ab8ba3e6ba1bc775dcd3"
FLAS_WEIGHTS_SHA256 = "bca45f7fa5abe11d607407b11ba0f00bdbf7936fa0a104b988cfac765446148e"
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


def load_stage_a(directory: Path) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    matched = {}
    actions = {}
    hashes = {}
    for path in sorted(directory.glob("shard_*/matched_continuations.json")):
        rows = json.loads(path.read_text())
        if set(rows) & set(matched):
            raise RuntimeError("duplicate matched-continuation concept")
        matched.update(rows)
        hashes[str(path.relative_to(directory))] = file_sha256(path)
    for path in sorted(directory.glob("shard_*/actions.pt")):
        rows = torch.load(path, weights_only=True)
        if set(rows) & set(actions):
            raise RuntimeError("duplicate retained-action concept")
        actions.update(rows)
        hashes[str(path.relative_to(directory))] = file_sha256(path)
    return matched, actions, hashes


def verify_sources(args: argparse.Namespace) -> None:
    expected = {
        args.data: DATA_SHA256,
        args.flow_checkpoint: FLAS_WEIGHTS_SHA256,
        args.flow_checkpoint.parent / "config.json": FLAS_CONFIG_SHA256,
        args.flas_source / "flas" / "generate.py": FLAS_GENERATE_SHA256,
        args.flas_source / "flas" / "model.py": FLAS_MODEL_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"source hash mismatch: {path}")


def serialization_error(original: dict[str, object], loaded: dict[str, object]) -> float:
    errors = []
    for concept_id, quantiles in original.items():
        for quantile, actions in quantiles.items():
            for name, action in actions.items():
                errors.append(
                    float((action - loaded[concept_id][quantile][name]).abs().max())
                )
    return max(errors, default=0.0)


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
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
    matched, retained, stage_a_hashes = load_stage_a(args.stage_a)
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
    zero = torch.zeros(generator.llm.config.hidden_size, device=device)
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
            vector=zero,
        )
    replay_error = float((reference - replay).abs().max())
    if replay_error != 0.0:
        raise RuntimeError("zero residual replay failed")
    concepts = []
    saved_actions = {}
    for concept_id in args.concept_ids:
        concept_started = time.perf_counter()
        key = str(concept_id)
        try:
            source = data["concepts"][key]
            rows = matched[key]
            concept_hidden, concept_mask = generator.encode_concept(source["concept"])
            construction, covectors, metric = construct_geometry(
                generator,
                generator.tokenizer,
                rows["construction"],
                data["neutral"],
                concept_hidden,
                concept_mask,
                device,
            )
            quantiles, actions, baseline = evaluate_quantiles(
                generator,
                generator.tokenizer,
                rows["evaluation"],
                data["neutral"],
                concept_hidden,
                concept_mask,
                covectors,
                metric,
                np.asarray(construction["changes"]),
                retained[key]["pooled_sequence_metric"].numpy(),
                concept_id,
                device,
            )
            saved_actions[key] = actions
            concepts.append(
                {
                    "baseline": baseline,
                    "concept": source["concept"],
                    "concept_id": concept_id,
                    "construction": construction,
                    "elapsed_seconds": time.perf_counter() - concept_started,
                    "quantiles": quantiles,
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
    return saved_actions, {
        "backward_products": 24 * len(args.concept_ids),
        "concepts": concepts,
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
        actions, receipt = run(args)
        torch.save(actions, args.output / "actions.pt")
        loaded = torch.load(args.output / "actions.pt", weights_only=True)
        receipt["action_serialization_max_abs_error"] = serialization_error(actions, loaded)
        receipt["actions_sha256"] = file_sha256(args.output / "actions.pt")
        (args.output / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True)
        )
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise


if __name__ == "__main__":
    main()
