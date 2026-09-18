import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from projection_transport_steering.amortized_pullback_runner import concept_beta, file_sha256
from projection_transport_steering.cacheback_runner import (
    _forward_kl,
    _sample_continuations,
    run_case,
    smoke_case,
)
from projection_transport_steering.future_pullback import prepare_gpt2_teacher_forced


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--target", type=float, default=0.05)
    return parser.parse_args(argv)


def main() -> None:
    runner_started = time.perf_counter()
    args = parse_args()
    data = json.loads(args.data.read_text())
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).cuda().eval()
    model.requires_grad_(False)
    case = smoke_case(data)
    construction = run_case(
        model,
        case,
        data["mappings"][case["concept"]],
        args.layer,
        (0, args.tokens),
        args.trajectories,
        args.target,
        None,
        None,
        args.target,
        True,
    )
    method = f"cacheback_h{args.tokens}"
    input_ids = torch.tensor(case["token_ids"], device="cuda").reshape(1, -1)
    sequence = _sample_continuations(
        model,
        input_ids,
        case["context_sha256"],
        1,
        args.tokens,
    )[0]
    packet = prepare_gpt2_teacher_forced(model, sequence, args.layer, input_ids.shape[1] - 1)
    beta = concept_beta(model.lm_head.weight.detach(), data["mappings"][case["concept"]])
    action_values = construction["actions"][method]
    action = torch.tensor(action_values, device="cuda", dtype=packet["state"].dtype)
    with torch.no_grad():
        full_base = packet["downstream"](packet["state"])
        incremental_base = packet["incremental_downstream"](packet["state"])
        full_steered = packet["downstream"](packet["state"] + action)
        incremental_steered = packet["incremental_downstream"](packet["state"] + action)
    base_logits = model.lm_head(full_base)
    full_logits = model.lm_head(full_steered)
    reset_logits = model.lm_head(torch.cat((incremental_steered[:1], incremental_base[1:])))
    cache_only_logits = model.lm_head(torch.cat((incremental_base[:1], incremental_steered[1:])))
    full_kl = _forward_kl(base_logits, full_logits)
    reset_kl = _forward_kl(base_logits, reset_logits)
    cache_only_kl = _forward_kl(base_logits, cache_only_logits)
    result = {
        "action_norm": float(action.norm()),
        "base_replay_max_error": float((incremental_base - full_base).abs().max()),
        "cache_only_immediate_kl": float(cache_only_kl[0]),
        "cache_only_future_kl": float(cache_only_kl[1:].sum()),
        "concept": case["concept"],
        "full_future_kl": float(full_kl[1:].sum()),
        "full_immediate_kl": float(full_kl[0]),
        "immediate_semantic_effect": float(beta @ (full_steered[0] - full_base[0])),
        "method": method,
        "reset_future_kl": float(reset_kl[1:].sum()),
        "reset_immediate_kl": float(reset_kl[0]),
        "state_id": case["context_sha256"],
        "steered_replay_max_error": float((incremental_steered - full_steered).abs().max()),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    action_path = args.output / "action.json"
    action_path.write_text(json.dumps(action_values))
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    receipt = {
        "action_sha256": file_sha256(action_path),
        "automatic_differentiation": construction["automatic_differentiation"],
        "data_sha256": file_sha256(args.data),
        "derivative_rows": construction["derivative_rows"],
        "full_runner_seconds": time.perf_counter() - runner_started,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "jacobian_calls": construction["jacobian_calls"],
        "jvp_count": construction["jvp_count"],
        "matching": construction["effect_matching"]["methods"][method],
        "platform": platform.platform(),
        "pilot_states_observed": 0,
        "result_sha256": file_sha256(result_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "torch": torch.__version__,
        "vjp_count": construction["vjp_count"],
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
