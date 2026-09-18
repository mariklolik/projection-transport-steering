import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer

from irc_baseline_support import checked_packet
from materialize_outcome_score_data import file_sha256, payload_sha256
from projection_transport_steering import attention_backend
from projection_transport_steering.branch_runtime import start_branches
from projection_transport_steering.outcome_score_runtime import rollout_seed
from run_attention_diagnostic import checked_config


@torch.no_grad()
def prefill_batch(model, tokenizer, rows: list[dict], config: dict, expected: dict) -> dict:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("prefill requires a frozen evaluation model")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": config["instruction"] + row["problem"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for row in rows
    ]
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
    identity = {
        "start": expected["start"], "width": inputs.input_ids.shape[1],
        "cluster_ids": [row["cluster_id"] for row in rows],
        "prompt_sha256": [payload_sha256(prompt) for prompt in prompts],
        "prompt_tokens": inputs.attention_mask.sum(-1).tolist(),
        **{
            name + "_sha256": payload_sha256(inputs[name].tolist())
            for name in ("input_ids", "attention_mask")
        },
    }
    if identity != expected:
        raise ValueError("frozen prefill input mismatch")
    inputs = inputs.to(model.device)
    state = start_branches(
        inputs.input_ids, inputs.attention_mask,
        [rollout_seed(row["cluster_id"], 0) for row in rows],
    )
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(
        input_ids=state.input_ids, attention_mask=state.attention_mask,
        position_ids=(state.attention_mask.cumsum(-1) - 1).clamp_min(0),
        past_key_values=state.cache, use_cache=True, logits_to_keep=1,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        **identity,
        "seconds": time.perf_counter() - started,
        "logits_shape": list(output.logits.shape),
        "logits_dtype": str(output.logits.dtype),
        "finite_logits": bool(torch.isfinite(output.logits).all()),
        "cache_length": (
            None if output.past_key_values is None else output.past_key_values.get_seq_length()
        ),
        "base_parameter_versions_unchanged": versions == {
            name: parameter._version for name, parameter in model.named_parameters()
        },
    }
    if (
        not result["finite_logits"]
        or result["logits_shape"] != [len(rows), 1, model.config.vocab_size]
        or result["cache_length"] != identity["width"]
        or not result["base_parameter_versions_unchanged"]
    ):
        raise RuntimeError("invalid prefill output or base mutation")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    config = checked_config(args.config, args.config_sha256, root)
    for name, expected in config["model_files_sha256"].items():
        if file_sha256(args.model / name) != expected:
            raise ValueError(f"model file hash mismatch: {name}")
    rows = checked_packet(root, config["selection"], "selection")
    inventory = root / config["input_inventory"]
    if file_sha256(inventory) != config["input_inventory_sha256"]:
        raise ValueError("input inventory hash mismatch")
    batches = json.loads(inventory.read_text())["selection_batches"]
    if (
        [batch["start"] for batch in batches] != list(range(0, len(rows), config["batch_size"]))
        or [identity for batch in batches for identity in batch["cluster_ids"]]
        != [row["cluster_id"] for row in rows]
    ):
        raise ValueError("prefill source order mismatch")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    attention_backend.register_fp32_prefill_flex()
    alias = config["attention"]
    if (
        AttentionInterface()[alias] is not attention_backend.fp32_prefill_flex_v4
        or AttentionMaskInterface()[alias] is not attention_backend.eager_flex_attention_mask
    ):
        raise ValueError("unexpected prefill registry binding")
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running", "cells": [], "planned_batches": len(batches),
        "config_sha256": args.config_sha256, "scientific_result": False, "sampling": False,
    }
    failure = None
    started = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=alias, local_files_only=True
        ).to("cuda").eval().requires_grad_(False)
        if model.config._attn_implementation != alias:
            raise ValueError("loaded model attention differs")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, padding_side="left"
        )
        torch.cuda.reset_peak_memory_stats()
        with (args.output / "cells.jsonl").open("x") as log:
            for batch in batches:
                start = batch["start"]
                print(json.dumps({"event": "prefill_started", "start": start}), flush=True)
                result = prefill_batch(
                    model, tokenizer, rows[start:start + config["batch_size"]], config, batch
                )
                receipt["cells"].append(result)
                log.write(json.dumps(result, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps({"event": "prefill_pass", "start": start}), flush=True)
        receipt.update(status="pass", peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        failure = error
        receipt.update(status="fail", failure_type=type(error).__name__, failure_message=str(error))
    receipt["total_elapsed_seconds"] = time.perf_counter() - started
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    if failure is not None:
        raise RuntimeError("prefill failed; immutable receipt retained") from failure


if __name__ == "__main__":
    main()
