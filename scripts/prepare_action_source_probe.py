import argparse
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer

from irc_baseline_support import checked_packet
from materialize_outcome_score_data import file_sha256
from projection_transport_steering.outcome_score_runtime import rollout_seed


def prepare_packet(root: Path, tokenizer_path: Path) -> dict:
    baseline = root / "research/irc-baseline-config-v3.json"
    config = json.loads(baseline.read_text())
    rows = checked_packet(root, config, "fit")[:8]
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if file_sha256(tokenizer_path / name) != config["model_files_sha256"][name]:
            raise ValueError("source tokenizer hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    requests = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": config["instruction"] + row["problem"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if not tokens or len(tokens) + config["generation_steps"] > 12288:
            raise ValueError("empty or overlong source prompt")
        for rollout in range(4):
            requests.append({
                "request_id": f"{row['cluster_id']}:{rollout}",
                "cluster_id": row["cluster_id"], "row_id": row["row_id"],
                "allocation": "fit", "rollout_index": rollout,
                "seed": rollout_seed(row["cluster_id"], rollout), "input_ids": tokens,
            })
    files = [
        "research/action-learning-v1/source-probe-protocol-v2.md",
        "scripts/run_action_source_probe.py", "scripts/prepare_action_source_probe.py",
        "scripts/irc_baseline_support.py", "scripts/materialize_outcome_score_data.py",
        "src/projection_transport_steering/outcome_score_runtime.py",
    ]
    return {
        "schema": "action-source-probe-v2", "requests": requests,
        "vocab_size": config["vocab_size"],
        "source_packet": config["packet"], "source_packet_sha256": config["packet_sha256"],
        "baseline_config_sha256": file_sha256(baseline),
        "source_sha256": {f: file_sha256(root / f) for f in files},
        "preparation_versions": {
            n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers")
        },
        "expected_versions": {
            "sglang": "0.0.0.dev1+g20621aa14.d20260827",
            "torch": "2.13.0+cu129", "transformers": "5.12.1",
        },
        "model_files_sha256": {
            **config["model_files_sha256"],
            "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
            "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
            "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
            "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
            "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
            "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
        },
        "engine": {
            "model_path": config["model_path"], "skip_tokenizer_init": True,
            "dtype": "bfloat16", "tp_size": 1, "context_length": 12288,
            "max_running_requests": 32, "mem_fraction_static": 0.8,
            "attention_backend": "fa3", "sampling_backend": "pytorch",
            "enable_deterministic_inference": True, "disable_radix_cache": True,
            "cuda_graph_max_bs_decode": 32, "random_seed": 20260908,
        },
        "sampling": {
            **config["sampling"], "max_new_tokens": config["generation_steps"],
            "stop_token_ids": config["eos_token_ids"], "ignore_eos": False,
            "no_stop_trim": True,
        },
        "stages": [
            {"name": "warmup", "indices": [0], "cap": 16},
            {"name": "main", "indices": list(range(32)), "cap": 8192},
            {"name": "reversed_four", "indices": [3, 2, 1, 0], "cap": 8192},
            {"name": "singleton_zero", "indices": [0], "cap": 8192},
            {"name": "singleton_one", "indices": [1], "cap": 8192},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = prepare_packet(Path(__file__).resolve().parents[1], args.tokenizer)
    with args.output.open("x") as handle:
        json.dump(packet, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"path": str(args.output), "sha256": file_sha256(args.output)}))


if __name__ == "__main__":
    main()
