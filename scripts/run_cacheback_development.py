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
from projection_transport_steering.cacheback_runner import development_cases, run_case

LAYERS = (3, 6, 9)
REGULARIZATION_MULTIPLIERS = (0.01, 0.1, 1.0)
DOSES = (0.125, 0.25, 0.5)
HORIZONS = (0, 2, 4, 8)
CPU_THREADS = 2
INTEROP_THREADS = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args(argv)


def main() -> None:
    runner_started = time.perf_counter()
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index or count is invalid")
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(INTEROP_THREADS)
    data = json.loads(args.data.read_text())
    cases = development_cases(data)[args.shard_index :: args.shard_count]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).cuda().eval()
    model.requires_grad_(False)
    fit_started = time.perf_counter()
    caa_directions = {
        (concept, layer): caa_direction(
            model,
            data["mappings"][concept],
            layer=layer,
            batch_size=64,
        )
        for concept in sorted({case["concept"] for case in cases})
        for layer in LAYERS
    }
    fit_seconds = time.perf_counter() - fit_started
    configurations = tuple(
        (dose, multiplier, dose)
        for multiplier in REGULARIZATION_MULTIPLIERS
        for dose in DOSES
    )
    args.output.mkdir(parents=True, exist_ok=False)
    source_manifest = {
        "case_count": len(cases),
        "cpu_threads": CPU_THREADS,
        "data_sha256": file_sha256(args.data),
        "doses": list(DOSES),
        "horizons": list(HORIZONS),
        "interop_threads": INTEROP_THREADS,
        "layers": list(LAYERS),
        "model_revision": data["model_revision"],
        "pilot_states_observed": 0,
        "regularization_multipliers": list(REGULARIZATION_MULTIPLIERS),
        "runner_sha256": file_sha256(Path(__file__)),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "trajectories": 8,
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True)
    )
    result_path = args.output / "results.jsonl"
    passed = 0
    result_rows = 0
    maximum_memory = 0
    total_derivative_rows = 0
    total_jvp_count = 0
    for case in cases:
        for layer in LAYERS:
            try:
                row = run_case(
                    model,
                    case,
                    data["mappings"][case["concept"]],
                    layer,
                    HORIZONS,
                    8,
                    directions={"caa": caa_directions[(case["concept"], layer)]},
                    configurations=configurations,
                    require_effect_reachability=False,
                )
                passed += 1
                maximum_memory = max(maximum_memory, row["peak_memory_allocated_bytes"])
                total_derivative_rows += row["derivative_rows"]
                total_jvp_count += row["jvp_count"]
            except Exception as error:
                row = {
                    "case_id": case["context_sha256"],
                    "concept": case["concept"],
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "layer": layer,
                    "status": "fail",
                }
            with result_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            result_rows += 1
    receipt = {
        **source_manifest,
        "automatic_differentiation": "forward_mode",
        "caa_fit_seconds": fit_seconds,
        "failed_rows": result_rows - passed,
        "full_runner_seconds": time.perf_counter() - runner_started,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "maximum_peak_memory_allocated_bytes": maximum_memory,
        "passed_rows": passed,
        "platform": platform.platform(),
        "result_rows": result_rows,
        "results_sha256": file_sha256(result_path),
        "torch": torch.__version__,
        "total_derivative_rows": total_derivative_rows,
        "total_jvp_count": total_jvp_count,
        "transformers": transformers.__version__,
        "vjp_count": 0,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
