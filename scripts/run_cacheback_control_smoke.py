import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from projection_transport_steering.amortized_pullback_runner import file_sha256
from projection_transport_steering.cacheback_baselines import caa_direction
from projection_transport_steering.cacheback_runner import run_case, smoke_case


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("h0", "exact", "top5000", "caa"), required=True)
    parser.add_argument("--layer", type=int, default=6)
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
    horizons = (0,) if args.mode == "h0" else (0, 2, 4, 8)
    top_k = 5000 if args.mode == "top5000" else None
    directions = None
    caa_fit_seconds = None
    if args.mode == "caa":
        started = time.perf_counter()
        directions = {
            "caa": caa_direction(
                model,
                data["mappings"][case["concept"]],
                args.layer,
                64,
            )
        }
        caa_fit_seconds = time.perf_counter() - started
    result = run_case(
        model,
        case,
        data["mappings"][case["concept"]],
        args.layer,
        horizons,
        args.trajectories,
        args.target,
        top_k,
        directions,
        args.target,
    )
    result["caa_fit_pairs"] = (
        len(data["mappings"][case["concept"]]["pairs"]) if args.mode == "caa" else None
    )
    result["caa_fit_seconds"] = caa_fit_seconds
    result["full_runner_seconds"] = time.perf_counter() - runner_started
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    receipt = {
        "action_evaluation_count": result["action_evaluation_count"],
        "automatic_differentiation": result["automatic_differentiation"],
        "data_sha256": file_sha256(args.data),
        "derivative_rows": result["derivative_rows"],
        "full_case_seconds": result["full_case_seconds"],
        "full_model_forward_count": result["full_model_forward_count"],
        "full_runner_seconds": result["full_runner_seconds"],
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "jacobian_calls": result["jacobian_calls"],
        "jvp_count": result["jvp_count"],
        "mode": args.mode,
        "model_revision": data["model_revision"],
        "pilot_states_observed": 0,
        "platform": platform.platform(),
        "result_sha256": file_sha256(result_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vjp_count": result["vjp_count"],
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
