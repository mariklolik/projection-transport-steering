import argparse
import hashlib
import json
from pathlib import Path

from projection_transport_steering.flow_step_support import condition_specifications
from projection_transport_steering.proxy_validity_analysis import analyze_proxy_validity

EXPECTED_SHARDS = {
    "shard_0": [1, 5],
    "shard_1": [16, 24],
    "shard_2": [29, 33],
    "shard_3": [34, 48],
    "shard_4": [58, 59],
    "shard_5": [60, 73],
}
DATA_SHA256 = "59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68"
FLOW_ANALYSIS_SHA256 = "4e0b7297271834cbb38fd69ce2acfe69116e0538e6c01ee49d86959449821520"
FLAS_CONFIG_SHA256 = "d5414215889017d76686bda35e00e4399ea7efa66815eceb322e18e3e7c7a46a"
FLAS_GENERATE_SHA256 = "14e1bde99c119e6b970cf7601569b6ab05bedfbd44d65286044af52936f529f7"
FLAS_MODEL_SHA256 = "9058008c85835fadeb0ac9737ebd711520fecbf09b92ab8ba3e6ba1bc775dcd3"
FLAS_WEIGHTS_SHA256 = "bca45f7fa5abe11d607407b11ba0f00bdbf7936fa0a104b988cfac765446148e"
MODEL_CONFIG_SHA256 = "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a"
MODEL_MANIFEST_SHA256 = "78a536738d6391c113dc4a5c239644b4336c9db223bba7a576d07c7cb6a1055b"
UPSTREAM_JUDGE_SHA256 = "2a8997810b1930d817262e9c3c249299cdda8804ddab67d72057d1ebe7b62755"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--teacher-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shards(
    generations: Path,
    judgments: Path,
) -> tuple[list[dict[str, object]], dict[str, object], bool]:
    condition_names = [str(specification["name"]) for specification in condition_specifications()]
    rows = []
    resources = {}
    invariants_pass = True
    generation_manifest = None
    stage_a_hashes = None
    for shard, concept_ids in EXPECTED_SHARDS.items():
        generation_path = generations / shard / "generations.json"
        generation_receipt_path = generations / shard / "receipt.json"
        judgment_path = judgments / shard / "judgments.json"
        judgment_receipt_path = judgments / shard / "receipt.json"
        generation_rows = json.loads(generation_path.read_text())
        generation_receipt = json.loads(generation_receipt_path.read_text())
        judgment_rows = json.loads(judgment_path.read_text())
        judgment_receipt = json.loads(judgment_receipt_path.read_text())
        expected_parse_counts = {name: 16 for name in condition_names}
        expected_ids = {
            f"{concept_id}:{condition}:{prompt_index}"
            for concept_id in concept_ids
            for condition in condition_names
            for prompt_index in range(8)
        }
        generation_ids = {row["generation_id"] for row in generation_rows}
        judgment_ids = {row["generation_id"] for row in judgment_rows}
        hard_valid = bool(
            len(generation_rows) == 224
            and len(judgment_rows) == 224
            and generation_ids == expected_ids
            and judgment_ids == expected_ids
            and generation_receipt["planned_concept_ids"] == concept_ids
            and generation_receipt["prompt_limit"] == 8
            and generation_receipt["max_new_tokens"] == 128
            and generation_receipt["generation_calls"] == 28
            and generation_receipt["generations"] == 224
            and generation_receipt["status"] == "pass"
            and generation_receipt["generation_serialization_exact"]
            and generation_receipt["generations_sha256"] == file_sha256(generation_path)
            and generation_receipt["data_sha256"] == DATA_SHA256
            and generation_receipt["flas_config_sha256"] == FLAS_CONFIG_SHA256
            and generation_receipt["flas_generate_sha256"] == FLAS_GENERATE_SHA256
            and generation_receipt["flas_model_sha256"] == FLAS_MODEL_SHA256
            and generation_receipt["flas_weights_sha256"] == FLAS_WEIGHTS_SHA256
            and judgment_receipt["batch_size"] == 8
            and judgment_receipt["max_new_tokens"] == 160
            and judgment_receipt["rows"] == 224
            and judgment_receipt["completions"] == 672
            and judgment_receipt["concept_ids"] == concept_ids
            and judgment_receipt["prompt_limit"] == 8
            and judgment_receipt["generations_sha256"] == file_sha256(generation_path)
            and judgment_receipt["generation_receipt_sha256"]
            == file_sha256(generation_receipt_path)
            and judgment_receipt["judgments_sha256"] == file_sha256(judgment_path)
            and judgment_receipt["judgment_serialization_exact"]
            and judgment_receipt["model_config_sha256"] == MODEL_CONFIG_SHA256
            and judgment_receipt["model_manifest_sha256"] == MODEL_MANIFEST_SHA256
            and judgment_receipt["upstream_judge_sha256"] == UPSTREAM_JUDGE_SHA256
            and all(len(row["judge_completions"]) == 3 for row in judgment_rows)
        )
        if not hard_valid:
            raise RuntimeError(f"invalid shard packet {shard}")
        parse_complete = bool(
            judgment_receipt["status"] == "pass"
            and judgment_receipt["parsed"] == 224
            and judgment_receipt["parse_counts_by_condition"] == expected_parse_counts
        )
        invariants_pass = bool(invariants_pass and parse_complete)
        current_manifest = generation_receipt["model_manifest"]
        current_stage_a = generation_receipt["stage_a_input_hashes"]
        generation_manifest = (
            current_manifest if generation_manifest is None else generation_manifest
        )
        stage_a_hashes = current_stage_a if stage_a_hashes is None else stage_a_hashes
        if current_manifest != generation_manifest or current_stage_a != stage_a_hashes:
            raise RuntimeError(f"source manifest mismatch {shard}")
        rows.extend(judgment_rows)
        resources[shard] = {
            "generation_elapsed_seconds": generation_receipt["elapsed_seconds"],
            "generation_peak_memory_bytes": generation_receipt["peak_memory_bytes"],
            "generation_receipt_sha256": file_sha256(generation_receipt_path),
            "judge_elapsed_seconds": judgment_receipt["elapsed_seconds"],
            "judge_peak_memory_bytes": judgment_receipt["peak_memory_bytes"],
            "judge_receipt_sha256": file_sha256(judgment_receipt_path),
            "parsed": judgment_receipt["parsed"],
        }
    return rows, resources, invariants_pass


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if file_sha256(args.teacher_analysis) != FLOW_ANALYSIS_SHA256:
            raise RuntimeError("teacher analysis hash mismatch")
        teacher = json.loads(args.teacher_analysis.read_text())
        if teacher["status"] != "valid" or teacher["complete_concepts"] != 12:
            raise RuntimeError("teacher analysis is invalid")
        rows, resources, invariants_pass = load_shards(args.generations, args.judgments)
        result = analyze_proxy_validity(
            teacher,
            rows,
            [concept_id for concept_ids in EXPECTED_SHARDS.values() for concept_id in concept_ids],
            invariants_pass,
            args.bootstrap_samples,
        )
        result.update(
            {
                "generations": len(rows),
                "judge_completions": 3 * len(rows),
                "resources": resources,
                "selection_status": "instrument_validation_only",
                "status": "valid",
                "teacher_analysis_sha256": file_sha256(args.teacher_analysis),
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
