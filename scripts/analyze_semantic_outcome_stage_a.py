import argparse
import hashlib
import json
from pathlib import Path

from projection_transport_steering.semantic_outcome_stage_a_analysis import (
    analyze_concepts,
)

EXPECTED_SHARDS = {
    "shard_0": [1, 5],
    "shard_1": [16, 24],
    "shard_2": [29, 33],
    "shard_3": [34, 48],
    "shard_4": [58, 59],
    "shard_5": [60, 73],
}
DATA_SHA256 = "59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68"
METRIC_ACTIONS_SHA256 = "fb6236abebc91010e57355e7f9ea93079863d8f21108f3eba374f15eca448c20"


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
    concepts = []
    receipts = {}
    hosts = set()
    gpus = set()
    for shard, expected_ids in EXPECTED_SHARDS.items():
        shard_path = directory / shard
        receipt_path = shard_path / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"missing receipt {shard}")
        receipt = json.loads(receipt_path.read_text())
        observed_ids = [row["concept_id"] for row in receipt["concepts"]]
        if (
            receipt["data_sha256"] != DATA_SHA256
            or receipt["metric_actions_sha256"] != METRIC_ACTIONS_SHA256
            or receipt["planned_concepts"] != len(expected_ids)
            or receipt["replay_max_abs_logit_error"] != 0.0
            or observed_ids != expected_ids
            or file_sha256(shard_path / "actions.pt") != receipt["actions_sha256"]
            or file_sha256(shard_path / "matched_continuations.json")
            != receipt["matched_continuations_sha256"]
        ):
            raise RuntimeError(f"invalid receipt {shard}")
        concepts.extend(receipt["concepts"])
        hosts.add(receipt["host"])
        gpus.add(receipt["gpu"])
        receipts[shard] = {
            "actions_sha256": receipt["actions_sha256"],
            "elapsed_seconds": receipt["elapsed_seconds"],
            "matched_continuations_sha256": receipt["matched_continuations_sha256"],
            "peak_memory_bytes": receipt["peak_memory_bytes"],
            "receipt_sha256": file_sha256(receipt_path),
            "status": receipt["status"],
        }
    return concepts, {
        "backward_products": sum(16 * len(ids) for ids in EXPECTED_SHARDS.values()),
        "gpu_models": sorted(gpus),
        "hosts": sorted(hosts),
        "parallel_wall_upper_seconds": max(
            value["elapsed_seconds"] for value in receipts.values()
        ),
        "peak_memory_bytes_max": max(
            value["peak_memory_bytes"] for value in receipts.values()
        ),
        "receipts": receipts,
        "teacher_forced_forward_calls": sum(
            2 + 59 * len(ids) for ids in EXPECTED_SHARDS.values()
        ),
    }


def analyze(directory: Path, bootstrap_samples: int) -> dict[str, object]:
    concepts, resources = load_packet(directory)
    result = analyze_concepts(concepts, resamples=bootstrap_samples)
    return {
        **result,
        "decision": (
            "open_stage_b_tuning_generation"
            if result["stage_b_authorized"]
            else "close_semantic_outcome_v1"
        ),
        "input": str(directory.resolve()),
        "resources": resources,
        "selection_status": "exploratory_premise_sentinel",
        "status": "valid",
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = analyze(args.input, args.bootstrap_samples)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
