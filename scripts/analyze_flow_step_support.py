import argparse
import hashlib
import json
from pathlib import Path

from projection_transport_steering.flow_step_support import condition_specifications
from projection_transport_steering.flow_step_support_analysis import analyze_design

EXPECTED_SHARDS = {
    "shard_0": [1, 5],
    "shard_1": [16, 24],
    "shard_2": [29, 33],
    "shard_3": [34, 48],
    "shard_4": [58, 59],
    "shard_5": [60, 73],
}
DATA_SHA256 = "59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68"
FLAS_CONFIG_SHA256 = "d5414215889017d76686bda35e00e4399ea7efa66815eceb322e18e3e7c7a46a"
FLAS_GENERATE_SHA256 = "14e1bde99c119e6b970cf7601569b6ab05bedfbd44d65286044af52936f529f7"
FLAS_MODEL_SHA256 = "9058008c85835fadeb0ac9737ebd711520fecbf09b92ab8ba3e6ba1bc775dcd3"
FLAS_WEIGHTS_SHA256 = "bca45f7fa5abe11d607407b11ba0f00bdbf7936fa0a104b988cfac765446148e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packet(directory: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    receipts = []
    resources = {}
    expected_specifications = condition_specifications()
    for shard, expected_ids in EXPECTED_SHARDS.items():
        receipt_path = directory / shard / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"missing receipt {shard}")
        receipt = json.loads(receipt_path.read_text())
        observed_ids = [concept["concept_id"] for concept in receipt["concepts"]]
        expected_forwards = 2 + 45 * len(expected_ids)
        if (
            observed_ids != expected_ids
            or receipt["planned_concepts"] != len(expected_ids)
            or receipt["condition_specifications"] != expected_specifications
            or receipt["data_sha256"] != DATA_SHA256
            or receipt["flas_config_sha256"] != FLAS_CONFIG_SHA256
            or receipt["flas_generate_sha256"] != FLAS_GENERATE_SHA256
            or receipt["flas_model_sha256"] != FLAS_MODEL_SHA256
            or receipt["flas_weights_sha256"] != FLAS_WEIGHTS_SHA256
            or receipt["flowtime"] != 2.0
            or receipt["n_steps"] != 3
            or receipt["replay_max_abs_logit_error"] != 0.0
            or not receipt["condition_serialization_exact"]
            or receipt["teacher_forced_forward_calls"] != expected_forwards
        ):
            raise RuntimeError(f"invalid receipt {shard}")
        receipts.append(receipt)
        resources[shard] = {
            "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_memory_bytes": receipt["peak_memory_bytes"],
            "receipt_sha256": file_sha256(receipt_path),
            "status": receipt["status"],
        }
    return receipts, resources


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        receipts, resources = load_packet(args.input)
        result = analyze_design(
            receipts,
            expected_concepts=12,
            bootstrap_draws=args.bootstrap_samples,
        )
        result.update(
            {
                "input": str(args.input.resolve()),
                "resources": {
                    "parallel_wall_upper_seconds": max(
                        receipt["elapsed_seconds"] for receipt in receipts
                    ),
                    "peak_memory_bytes_max": max(
                        receipt["peak_memory_bytes"] for receipt in receipts
                    ),
                    "receipts": resources,
                    "teacher_forced_forward_calls": sum(
                        receipt["teacher_forced_forward_calls"] for receipt in receipts
                    ),
                },
                "selection_status": "post_hoc_design_only",
                "status": "valid",
            }
        )
        (args.output / "analysis.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        )
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise


if __name__ == "__main__":
    main()
