import hashlib
import json
import platform
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from projection_transport_steering.amortized_pullback_runner import concept_beta, file_sha256
from projection_transport_steering.cacheback_baselines import matched_linear_action
from projection_transport_steering.claim_relative_geometry import (
    minimum_sequence_metric_action,
    nested_sequence_metrics,
)
from projection_transport_steering.future_pullback import (
    future_pullback_geometry,
    prepare_gpt2_teacher_forced,
)


def rollout_seed(state_id: str, rollout_index: int) -> int:
    if not state_id or rollout_index < 0:
        raise ValueError("state id or rollout index is invalid")
    payload = f"{state_id}\0{rollout_index}\0{20260906}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def sentinel_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {"third": 3, "ing": 3, "past": 2}
    cases: list[dict[str, Any]] = []
    for concept, count in expected.items():
        rows = [row for row in data["contexts"][concept] if row["stage"] == "sentinel"]
        if len(rows) != count:
            raise ValueError(f"invalid sentinel allocation for {concept}")
        cases.extend(dict(row, concept=concept) for row in rows)
    return sorted(cases, key=lambda row: row["context_sha256"])


def development_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for concept in ("third", "ing", "past"):
        rows = [row for row in data["contexts"][concept] if row["stage"] == "development"]
        if len(rows) != 8:
            raise ValueError(f"invalid development allocation for {concept}")
        cases.extend(dict(row, concept=concept) for row in rows)
    return sorted(cases, key=lambda row: row["context_sha256"])


def smoke_case(data: dict[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row, concept=concept)
        for concept, concepts in data["contexts"].items()
        for row in concepts
        if row["stage"] == "smoke"
    ]
    if len(rows) != 1:
        raise ValueError("cacheback smoke allocation is invalid")
    return rows[0]


def _sample_continuations(
    model: Any,
    input_ids: Tensor,
    state_id: str,
    trajectories: int,
    tokens: int,
) -> list[Tensor]:
    if trajectories < 1 or tokens < 0:
        raise ValueError("trajectory or token count is invalid")
    sequences: list[Tensor] = []
    attention_mask = torch.ones_like(input_ids)
    for rollout_index in range(trajectories):
        generator = torch.Generator(device=input_ids.device).manual_seed(
            rollout_seed(state_id, rollout_index)
        )
        sequence = input_ids
        mask = attention_mask
        for _ in range(tokens):
            with torch.no_grad():
                logits = model(sequence, attention_mask=mask, use_cache=False).logits[0, -1]
            next_id = torch.multinomial(
                torch.softmax(logits.float(), dim=-1),
                1,
                generator=generator,
            ).reshape(1, 1)
            sequence = torch.cat((sequence, next_id), dim=1)
            mask = torch.ones_like(sequence)
        sequences.append(sequence)
    return sequences


def _forward_kl(base_logits: Tensor, steered_logits: Tensor) -> Tensor:
    base_log = torch.log_softmax(base_logits.double(), dim=-1)
    steered_log = torch.log_softmax(steered_logits.double(), dim=-1)
    return (base_log.exp() * (base_log - steered_log)).sum(-1).clamp_min(0)


def _evaluate_action(
    model: Any,
    prepared: list[dict[str, Any]],
    base_representations: list[Tensor],
    beta: Tensor,
    action: np.ndarray,
    horizon_metric: np.ndarray,
    cumulative_metric: np.ndarray,
) -> dict[str, Any]:
    offset_rows = []
    immediate_effects = []
    for packet, base in zip(prepared, base_representations, strict=True):
        steered_state = packet["state"] + torch.from_numpy(action).to(packet["state"])
        with torch.no_grad():
            steered = packet["downstream"](steered_state)
            offset_rows.append(_forward_kl(model.lm_head(base), model.lm_head(steered)).cpu())
            immediate_effects.append(float(beta @ (steered[0] - base[0])))
    offset = torch.stack(offset_rows).mean(0).numpy()
    return {
        "action_norm": float(np.linalg.norm(action)),
        "cache_only_forward_kl_mean": float(offset[1:].sum()),
        "cache_reset_forward_kl_mean": float(offset[0]),
        "cumulative_forward_kl_mean": float(offset.sum()),
        "immediate_semantic_effect_mean": float(np.mean(immediate_effects)),
        "offset_forward_kl_mean": offset.tolist(),
        "predicted_cumulative_quadratic_cost": float(
            0.5 * action @ cumulative_metric @ action
        ),
        "predicted_horizon_quadratic_cost": float(0.5 * action @ horizon_metric @ action),
    }


def match_actions_to_effect(
    actions: dict[str, np.ndarray],
    evaluate: Any,
    target: float,
    tolerance: float = 1e-4,
    max_evaluations: int = 24,
    require_all: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not actions or target <= 0 or tolerance <= 0 or max_evaluations < 2:
        raise ValueError("actions or effect-matching configuration is invalid")
    zero = np.zeros_like(next(iter(actions.values())))
    zero_evaluation = evaluate(zero)
    zero_effect = float(zero_evaluation["immediate_semantic_effect_mean"])
    matched = {}
    evaluations = {}
    metadata = {}
    for name, action in actions.items():
        if action.shape != zero.shape or not np.all(np.isfinite(action)):
            raise ValueError("action is invalid")
        upper_evaluation = evaluate(action)
        upper_effect = float(upper_evaluation["immediate_semantic_effect_mean"])
        endpoint_effect = upper_effect
        if target < zero_effect - tolerance or target > upper_effect + tolerance:
            if require_all:
                raise RuntimeError(f"realized effect is unreachable for {name}")
            matched[name] = action
            evaluations[name] = upper_evaluation
            metadata[str(name)] = {
                "absolute_effect_error": abs(endpoint_effect - target),
                "endpoint_effect": endpoint_effect,
                "evaluation_count": 1,
                "reached": False,
                "scale": 1.0,
                "scale_bracket": [0.0, 1.0],
            }
            continue
        lower_scale = 0.0
        upper_scale = 1.0
        lower_evaluation = zero_evaluation
        evaluation_count = 1
        for _ in range(max_evaluations - 1):
            if abs(upper_effect - target) <= tolerance:
                break
            middle_scale = 0.5 * (lower_scale + upper_scale)
            middle_evaluation = evaluate(middle_scale * action)
            evaluation_count += 1
            middle_effect = float(middle_evaluation["immediate_semantic_effect_mean"])
            if middle_effect >= target:
                upper_scale = middle_scale
                upper_evaluation = middle_evaluation
                upper_effect = middle_effect
            else:
                lower_scale = middle_scale
                lower_evaluation = middle_evaluation
        lower_error = abs(
            float(lower_evaluation["immediate_semantic_effect_mean"]) - target
        )
        upper_error = abs(upper_effect - target)
        if lower_error < upper_error:
            scale = lower_scale
            evaluation = lower_evaluation
            error = lower_error
        else:
            scale = upper_scale
            evaluation = upper_evaluation
            error = upper_error
        if error > tolerance:
            raise RuntimeError(f"realized effect matching failed for {name}")
        matched[name] = scale * action
        evaluations[name] = evaluation
        metadata[str(name)] = {
            "absolute_effect_error": error,
            "endpoint_effect": endpoint_effect,
            "evaluation_count": evaluation_count,
            "reached": True,
            "scale": scale,
            "scale_bracket": [0.0, 1.0],
        }
    return matched, evaluations, metadata


def run_case(
    model: Any,
    case: dict[str, Any],
    mapping: dict[str, Any],
    layer: int,
    horizons: tuple[int, ...] = (0, 2, 4, 8),
    trajectories: int = 8,
    target: float = 0.05,
    top_k: int | None = None,
    directions: dict[str, Tensor] | None = None,
    realized_effect_target: float | None = None,
    include_actions: bool = False,
    configurations: tuple[tuple[float, float, float | None], ...] | None = None,
    require_effect_reachability: bool = True,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    if tuple(sorted(horizons)) != horizons or horizons[0] != 0:
        raise ValueError("horizons must be sorted and start at zero")
    device = next(model.parameters()).device
    input_ids = torch.tensor(case["token_ids"], device=device).reshape(1, -1)
    position = input_ids.shape[1] - 1
    effective_trajectories = 1 if horizons[-1] == 0 else trajectories
    continuations = _sample_continuations(
        model,
        input_ids,
        case["context_sha256"],
        effective_trajectories,
        horizons[-1],
    )
    beta = concept_beta(model.lm_head.weight.detach(), mapping)
    prepared = []
    base_representations = []
    terms = []
    covectors = []
    automatic_differentiation = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for continuation in continuations:
        packet = prepare_gpt2_teacher_forced(model, continuation, layer, position)
        geometry = future_pullback_geometry(
            packet["downstream"],
            packet["state"],
            beta,
            model.lm_head.weight.detach(),
            top_k,
        )
        prepared.append(packet)
        automatic_differentiation.append(geometry["automatic_differentiation"])
        base_representations.append(geometry["representations"].detach())
        terms.append(geometry["terms"].detach().cpu().numpy())
        covectors.append(geometry["covector"].detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    geometry_seconds = time.perf_counter() - started
    covector = np.mean(covectors, axis=0)
    if max(np.linalg.norm(value - covector) for value in covectors) > 1e-6:
        raise RuntimeError("immediate covector changed across common-random trajectories")
    mean_terms = np.mean(terms, axis=0)
    differentiation_modes = set(automatic_differentiation)
    if len(differentiation_modes) != 1:
        raise RuntimeError("automatic differentiation mode changed across trajectories")
    differentiation_mode = differentiation_modes.pop()
    metrics = nested_sequence_metrics(mean_terms, horizons, tolerance=1e-8)
    eigenvalues = np.linalg.eigvalsh(metrics[0])
    floor = max(1e-12, float(eigenvalues[-1]) * 1e-12)
    positive = eigenvalues[eigenvalues > floor]
    if not positive.size:
        raise RuntimeError("immediate metric has no positive spectrum")
    directions = {} if directions is None else directions
    required_names = {f"cacheback_h{horizon}" for horizon in horizons} | {"euclidean"}
    if set(directions) & required_names:
        raise ValueError("direction name collides with a required method")
    covector_tensor = torch.from_numpy(covector)
    increments = [metrics[right] - metrics[left] for left, right in zip(horizons, horizons[1:])]
    minimum_increment = min(
        (float(np.linalg.eigvalsh(value)[0]) for value in increments),
        default=0.0,
    )

    def evaluate_configuration(
        local_target: float,
        regularization_multiplier: float,
        effect_target: float | None,
    ) -> dict[str, Any]:
        regularization = regularization_multiplier * float(np.median(positive))
        actions = {
            f"cacheback_h{horizon}": minimum_sequence_metric_action(
                metric,
                covector,
                local_target,
                regularization,
            )["action"]
            for horizon, metric in metrics.items()
        }
        actions["euclidean"] = local_target * covector / (covector @ covector)
        for name, direction in directions.items():
            actions[name] = (
                matched_linear_action(
                    direction.detach().cpu().double(),
                    covector_tensor,
                    local_target,
                )
                .numpy()
            )
        effect_matching = None
        if effect_target is not None:
            actions, _, matching_methods = match_actions_to_effect(
                actions,
                lambda action: _evaluate_action(
                    model,
                    prepared,
                    base_representations,
                    beta,
                    action,
                    metrics[horizons[-1]],
                    metrics[horizons[-1]],
                ),
                effect_target,
                require_all=require_effect_reachability,
            )
            effect_matching = {
                "max_evaluations": 24,
                "methods": matching_methods,
                "scale_bracket": [0.0, 1.0],
                "target": effect_target,
                "tolerance": 1e-4,
            }
        methods = {}
        for horizon in horizons:
            name = f"cacheback_h{horizon}"
            methods[name] = _evaluate_action(
                model,
                prepared,
                base_representations,
                beta,
                actions[name],
                metrics[horizon],
                metrics[horizons[-1]],
            )
        methods["euclidean"] = _evaluate_action(
            model,
            prepared,
            base_representations,
            beta,
            actions["euclidean"],
            metrics[horizons[-1]],
            metrics[horizons[-1]],
        )
        for name in directions:
            methods[name] = _evaluate_action(
                model,
                prepared,
                base_representations,
                beta,
                actions[name],
                metrics[horizons[-1]],
                metrics[horizons[-1]],
            )
        action_evaluation_count = len(actions)
        if effect_matching is not None:
            action_evaluation_count += 1 + sum(
                row["evaluation_count"] for row in effect_matching["methods"].values()
            )
        reference = actions["cacheback_h0"]
        action_cosines = {
            str(horizon): float(
                reference
                @ actions[f"cacheback_h{horizon}"]
                / np.linalg.norm(reference)
                / np.linalg.norm(actions[f"cacheback_h{horizon}"])
            )
            for horizon in horizons[1:]
        }
        row = {
            "action_cosine_vs_h0": action_cosines,
            "action_evaluation_count": action_evaluation_count,
            "effect_matching": effect_matching,
            "methods": methods,
            "regularization": regularization,
            "regularization_multiplier": regularization_multiplier,
            "target": local_target,
        }
        if include_actions:
            row["actions"] = {name: action.tolist() for name, action in actions.items()}
        return row

    specifications = configurations or ((target, 0.1, realized_effect_target),)
    if not specifications:
        raise ValueError("configurations are empty")
    configuration_rows = [evaluate_configuration(*values) for values in specifications]
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "automatic_differentiation": differentiation_mode,
        "case_id": case["context_sha256"],
        "concept": case["concept"],
        "derivative_rows": effective_trajectories * (horizons[-1] + 1) * len(covector),
        "fisher_top_k": top_k,
        "full_case_seconds": time.perf_counter() - case_started,
        "full_model_forward_count": effective_trajectories * (horizons[-1] + 1),
        "geometry_seconds": geometry_seconds,
        "horizons": list(horizons),
        "layer": layer,
        "jacobian_calls": effective_trajectories,
        "jvp_count": effective_trajectories * len(covector)
        if differentiation_mode == "forward_mode"
        else 0,
        "minimum_increment_eigenvalue": minimum_increment,
        "peak_memory_allocated_bytes": int(peak_memory),
        "status": "pass",
        "requested_trajectory_count": trajectories,
        "trajectory_count": effective_trajectories,
        "vjp_count": effective_trajectories * (horizons[-1] + 1) * len(covector)
        if differentiation_mode == "reverse_mode_fallback"
        else 0,
    }
    if configurations is None:
        result.update(configuration_rows[0])
    else:
        result["action_evaluation_count"] = sum(
            row["action_evaluation_count"] for row in configuration_rows
        )
        result["configurations"] = configuration_rows
    return result


def run_sentinel_shard(
    model: Any,
    data: dict[str, Any],
    data_path: Path,
    output: Path,
    shard_index: int,
    shard_count: int,
    layer: int = 6,
    horizons: tuple[int, ...] = (0, 2, 4, 8),
    trajectories: int = 8,
    target: float = 0.05,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index or count is invalid")
    cases = sentinel_cases(data)[shard_index::shard_count]
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = {
        "case_count": len(cases),
        "data_sha256": file_sha256(data_path),
        "horizons": list(horizons),
        "layer": layer,
        "model_revision": data.get("model_revision", "test"),
        "pilot_states_observed": 0,
        "runner_sha256": file_sha256(Path(__file__)),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "target": target,
        "trajectories": trajectories,
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True)
    )
    result_path = output / "results.jsonl"
    passed = 0
    for case in cases:
        try:
            row = run_case(
                model,
                case,
                data["mappings"][case["concept"]],
                layer,
                horizons,
                trajectories,
                target,
            )
            passed += 1
        except Exception as error:
            row = {
                "case_id": case["context_sha256"],
                "concept": case["concept"],
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "fail",
            }
        with result_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    receipt = {
        **source_manifest,
        "failed_cases": len(cases) - passed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": socket.gethostname(),
        "passed_cases": passed,
        "platform": platform.platform(),
        "results_sha256": file_sha256(result_path),
        "torch": torch.__version__,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
