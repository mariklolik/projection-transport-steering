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
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.semantic_outcome_runtime import encode_candidates
from projection_transport_steering.semantic_outcome_sentinel import (
    construct_actions,
    evaluate_actions,
    matched_rows,
    model_logits,
)

DATA_SHA256 = "59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68"
METRIC_ACTIONS_SHA256 = "fb6236abebc91010e57355e7f9ea93079863d8f21108f3eba374f15eca448c20"
SEED = 20260907


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metric-actions", type=Path, required=True)
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
        {
            "path": str(file.relative_to(path)),
            "size": file.resolve().stat().st_size,
        }
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    if file_sha256(args.data) != DATA_SHA256:
        raise RuntimeError("sentinel source hash mismatch")
    if file_sha256(args.metric_actions) != METRIC_ACTIONS_SHA256:
        raise RuntimeError("neutral metric source hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    data = json.loads(args.data.read_text())
    if len(set(args.concept_ids)) != len(args.concept_ids):
        raise ValueError("concept IDs must be unique")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    model.requires_grad_(False)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    first = data["concepts"][str(args.concept_ids[0])]["construction"][0]
    replay_batch, _ = encode_candidates(tokenizer, first["input"], [first["output"]], device)
    base = model_logits(model, replay_batch, args.layer)
    zero = torch.zeros(model.config.hidden_size, device=device)
    replay = model_logits(model, replay_batch, args.layer, zero)
    replay_error = float((base - replay).abs().max())
    if replay_error != 0.0:
        raise RuntimeError("zero-action replay failed")
    metric = torch.load(args.metric_actions, weights_only=True)["metric_diagonal"].numpy()
    all_actions = {}
    all_matched = {}
    concepts = []
    for concept_id in args.concept_ids:
        concept_started = time.perf_counter()
        try:
            source = data["concepts"][str(concept_id)]
            matched = matched_rows(model, tokenizer, args.layer, source, device)
            actions, geometry = construct_actions(
                model, tokenizer, args.layer, matched["construction"], metric, device
            )
            conditions = evaluate_actions(
                model,
                tokenizer,
                args.layer,
                actions,
                matched["evaluation"],
                data["neutral"],
                device,
            )
            all_actions[str(concept_id)] = {
                name: torch.tensor(action) for name, action in actions.items()
            }
            all_matched[str(concept_id)] = matched
            concepts.append(
                {
                    "action_metric_costs": {
                        name: float(action @ (metric * action))
                        for name, action in actions.items()
                    },
                    "action_norms": {
                        name: float(np.linalg.norm(action)) for name, action in actions.items()
                    },
                    "conditions": conditions,
                    "concept": source["concept"],
                    "concept_id": concept_id,
                    "elapsed_seconds": time.perf_counter() - concept_started,
                    "geometry": geometry,
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
    torch.save(all_actions, args.output / "actions.pt")
    (args.output / "matched_continuations.json").write_text(
        json.dumps(all_matched, indent=2, sort_keys=True, ensure_ascii=False)
    )
    torch.cuda.synchronize()
    passed = sum(concept["status"] == "pass" for concept in concepts)
    return {
        "actions_sha256": file_sha256(args.output / "actions.pt"),
        "backward_products": 16 * len(args.concept_ids),
        "concepts": concepts,
        "cuda": torch.version.cuda,
        "data_sha256": file_sha256(args.data),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "layer": args.layer,
        "matched_continuations_sha256": file_sha256(
            args.output / "matched_continuations.json"
        ),
        "metric_actions_sha256": file_sha256(args.metric_actions),
        "model_manifest": model_manifest(args.model),
        "model_snapshot": str(args.model.resolve()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "planned_concepts": len(args.concept_ids),
        "platform": platform.platform(),
        "replay_max_abs_logit_error": replay_error,
        "status": "pass" if passed == len(args.concept_ids) else "fail",
        "teacher_forced_forward_calls": 2 + 59 * len(args.concept_ids),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = run(args)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "pass":
        raise RuntimeError("semantic-outcome sentinel shard failed")


if __name__ == "__main__":
    main()
