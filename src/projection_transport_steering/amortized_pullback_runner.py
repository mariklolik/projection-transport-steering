import hashlib
import json
import platform
import socket
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import Tensor

from projection_transport_steering.amortized_pullback import GPT2Downstream
from projection_transport_steering.amortized_pullback_experiment import (
    concept_probability,
    localize_targets,
    off_target_kl,
    run_trajectory,
    select_step_size,
)
from projection_transport_steering.amortized_pullback_methods import (
    LAYERS,
    base_geometry,
    method_direction,
    method_specs,
)


CONCEPTS = ("third", "ing", "past")
STEP_SIZES = tuple(2.0**exponent for exponent in range(-5, 4))
TARGETS = (0.3, 0.5, 0.7, 0.9)
STEPS = 30
EFFICIENCY = 0.3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for concept in CONCEPTS:
        rows = [row for row in data["contexts"][concept] if row["stage"] == "development"]
        if len(rows) != 12:
            raise ValueError(f"invalid development allocation for {concept}")
        for row in sorted(rows, key=lambda value: value["context_sha256"]):
            for layer in LAYERS:
                cases.append({"concept": concept, "layer": layer, **row})
    return cases


def concept_beta(unembedding: Tensor, mapping: dict[str, Any]) -> Tensor:
    differences = [
        unembedding[pair["target_id"]] - unembedding[pair["base_id"]]
        for pair in mapping["pairs"]
    ]
    return torch.stack(differences).mean(0)


def prepare_case(model: Any, case: dict[str, Any]) -> tuple[GPT2Downstream, Tensor, Tensor]:
    device = model.lm_head.weight.device
    input_ids = torch.tensor(case["token_ids"], device=device).unsqueeze(0)
    captured: list[Tensor] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output if isinstance(output, Tensor) else output[0])

    handle = model.transformer.h[case["layer"]].register_forward_hook(capture)
    try:
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False, output_hidden_states=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"captured {len(captured)} layer outputs")
    hidden = captured[0].detach()
    downstream = GPT2Downstream(model, case["layer"], hidden)
    state = hidden[0, -1]
    with torch.no_grad():
        replayed = model.lm_head(downstream(state))
    replay_error = (replayed - outputs.logits[0, -1]).abs().max()
    replay_tolerance = 1e-4 + 1e-5 * outputs.logits[0, -1].abs().max()
    if replay_error > replay_tolerance:
        raise RuntimeError(f"downstream replay error is {float(replay_error)}")
    return downstream, state, replayed


def evaluate_state(
    downstream: GPT2Downstream,
    unembedding: Tensor,
    state: Tensor,
    base_logits: Tensor,
    mapping: dict[str, Any],
) -> dict[str, float]:
    with torch.no_grad():
        logits = unembedding @ downstream(state)
    concept_ids = mapping["base_ids"] + mapping["target_ids"]
    return {
        "concept_probability": concept_probability(
            logits,
            mapping["base_ids"],
            mapping["target_ids"],
        ),
        "off_target_kl": off_target_kl(base_logits, logits, concept_ids),
    }


def run_method(
    spec: dict[str, Any],
    downstream: GPT2Downstream,
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    base_logits: Tensor,
    mapping: dict[str, Any],
    regularization: float,
) -> dict[str, Any]:
    evaluate = lambda value: evaluate_state(downstream, unembedding, value, base_logits, mapping)
    direction = lambda value: method_direction(
        spec,
        downstream,
        value,
        beta,
        unembedding,
        regularization,
    )
    calibrations: dict[float, list[dict[str, Any]]] = {}
    states: dict[float, list[Tensor]] = {}
    torch.cuda.reset_peak_memory_stats()
    for step_size in STEP_SIZES:
        calibrations[step_size], states[step_size] = run_trajectory(
            state,
            step_size,
            direction,
            evaluate,
            STEPS,
            EFFICIENCY,
        )
    selected = select_step_size(calibrations)
    targets = localize_targets(evaluate, calibrations[selected], states[selected], TARGETS, 24)
    selected_rows = calibrations[selected][1:]
    return {
        "calibrations": [
            {"step_size": step_size, "trajectory": calibrations[step_size]}
            for step_size in STEP_SIZES
        ],
        "calibration_direction_seconds": sum(
            row.get("direction_seconds", 0.0)
            for trajectory in calibrations.values()
            for row in trajectory
        ),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "selected_direction_seconds": sum(row["direction_seconds"] for row in selected_rows),
        "selected_backward_calls": sum(row["backward_calls"] for row in selected_rows),
        "selected_backward_equivalents": sum(
            row["backward_equivalents"] for row in selected_rows
        ),
        "selected_metric_vector_products": sum(
            row["metric_vector_products"] for row in selected_rows
        ),
        "selected_partial_forward_calls": sum(
            row["partial_forward_calls"] for row in selected_rows
        ),
        "selected_partial_forward_equivalents": sum(
            row["partial_forward_equivalents"] for row in selected_rows
        ),
        "selected_step_size": selected,
        "targets": targets,
    }


def append_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def run_development(
    model: Any,
    data: dict[str, Any],
    data_path: Path,
    basis_path: Path,
    output: Path,
    shard_index: int,
    shard_count: int,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index and count are invalid")
    output.mkdir(parents=True, exist_ok=False)
    geometry_root = output / "geometry"
    geometry_root.mkdir()
    started = time.perf_counter()
    basis_packet = torch.load(basis_path, map_location="cpu", weights_only=True)["layers"]
    unembedding = model.lm_head.weight.detach()
    cases = development_cases(data)
    selected_cases = cases[shard_index::shard_count]
    source_manifest = {
        "basis_sha256": file_sha256(basis_path),
        "case_count": len(selected_cases),
        "data_sha256": file_sha256(data_path),
        "development_cases_total": len(cases),
        "model_revision": data["model_revision"],
        "runner_module_sha256": file_sha256(Path(__file__)),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "validation_cases_evaluated": 0,
    }
    (output / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True))
    result_path = output / "results.jsonl"
    geometry_path = output / "geometry.jsonl"
    for case in selected_cases:
        layer = case["layer"]
        mapping = data["mappings"][case["concept"]]
        downstream, state, base_logits = prepare_case(model, case)
        beta = concept_beta(unembedding, mapping)
        layer_packet = basis_packet[layer]
        regularization = float(layer_packet["regularization"])
        specs = method_specs(layer_packet, layer, state.device)
        case_id = f"{case['concept']}-{case['context_sha256'][:16]}-l{layer}"
        tensors, geometry_summary = base_geometry(
            downstream,
            state,
            beta,
            unembedding,
            specs,
            regularization,
        )
        tensor_path = geometry_root / f"{case_id}.pt"
        torch.save(tensors, tensor_path)
        append_json(
            geometry_path,
            {
                "case_id": case_id,
                "concept": case["concept"],
                "context_sha256": case["context_sha256"],
                "geometry_sha256": file_sha256(tensor_path),
                "layer": layer,
                **geometry_summary,
            },
        )
        for spec in specs:
            method_result = run_method(
                spec,
                downstream,
                state,
                beta,
                unembedding,
                base_logits,
                mapping,
                regularization,
            )
            append_json(
                result_path,
                {
                    "case_id": case_id,
                    "concept": case["concept"],
                    "context_sha256": case["context_sha256"],
                    "layer": layer,
                    "method": spec["name"],
                    **method_result,
                },
            )
            print(json.dumps({"case_id": case_id, "method": spec["name"], "status": "pass"}))
    geometry_manifest = {
        path.name: file_sha256(path) for path in sorted(geometry_root.glob("*.pt"))
    }
    geometry_manifest_path = output / "geometry_manifest.json"
    geometry_manifest_path.write_text(json.dumps(geometry_manifest, indent=2, sort_keys=True))
    receipt = {
        "cuda": torch.version.cuda,
        "elapsed_seconds": time.perf_counter() - started,
        "geometry_jsonl_sha256": file_sha256(geometry_path),
        "geometry_manifest_sha256": file_sha256(geometry_manifest_path),
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "result_rows": sum(1 for _ in result_path.open()),
        "results_sha256": file_sha256(result_path),
        "status": "pass",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "validation_cases_evaluated": 0,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
