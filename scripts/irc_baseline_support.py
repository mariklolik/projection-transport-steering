import importlib.metadata
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup

from materialize_outcome_score_data import file_sha256, payload_sha256
from run_irc_fit_probe import load_examples
from projection_transport_steering.outcome_score_runtime import run_math_batch
from projection_transport_steering.reft_training import (
    epoch_batches,
    fit_step,
    generation_intervention,
)


def checked_packet(root: Path, spec: dict, allocation: str) -> list[dict]:
    path = root / spec["packet"]
    if file_sha256(path) != spec["packet_sha256"]:
        raise ValueError("allocation file hash mismatch")
    packet = json.loads(path.read_text())
    digest = packet.pop("content_sha256")
    rows = packet["rows"]
    ids = [row["cluster_id"] for row in rows]
    if (
        payload_sha256(packet) != digest
        or packet["allocation"] != allocation
        or len(ids) != spec["num_questions"]
        or len(set(ids)) != len(ids)
        or not all(
            row["allocation"] == allocation
            and row["split"] == "train"
            and row["reference_eligible"]
            for row in rows
        )
    ):
        raise ValueError("allocation content or eligibility mismatch")
    return rows


def prepare_inputs(config: dict, root: Path, model_path: Path):
    if any(
        importlib.metadata.version(name) != version
        for name, version in config["expected_versions"].items()
    ):
        raise ValueError("baseline runtime version mismatch")
    for directory, hashes in (
        (root, config["source_sha256"]),
        (model_path, config["model_files_sha256"]),
        (Path("/"), config.get("runtime_files_sha256", {})),
    ):
        for name, digest in hashes.items():
            if file_sha256(directory / name) != digest:
                raise ValueError(f"baseline source hash mismatch: {name}")
    fit_rows = checked_packet(root, config, "fit")
    selection = checked_packet(root, config["selection"], "selection")
    shares = config["zero_shares"]
    if (
        len(shares) != len(config["candidates"])
        or not all(shares)
        or sorted(i for share in shares for i in share) != list(range(len(selection)))
    ):
        raise ValueError("zero shares must partition the complete selection allocation")
    fit_ids = [row["cluster_id"] for row in fit_rows]
    if config.get("groups", {"fit": fit_ids}) != {"fit": fit_ids} or set(fit_ids) & {r["cluster_id"] for r in selection}:
        raise ValueError("fit membership mismatch or selection overlap")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side="right"
    )
    examples = load_examples({**config, "groups": {"fit": fit_ids}}, root, tokenizer)["fit"]
    if payload_sha256(examples) != config["target_examples_sha256"]:
        raise ValueError("complete native targets differ")
    recipe = config["recipe"]
    buckets = recipe["padding_buckets"]
    if (
        buckets != sorted(set(buckets))
        or not buckets
        or buckets[0] < 1
        or max(len(x["input_ids"]) for x in examples) > buckets[-1]
        or min(
            recipe["epochs"],
            recipe["batch_size"],
            recipe["microbatch_size"],
            config["generation_batch"],
            config["generation_steps"],
        )
        < 1
    ):
        raise ValueError("invalid complete training or generation schedule")
    return tokenizer, examples, selection


def train_adapter(model, action, examples, tokenizer, recipe, candidate, output: Path) -> dict:
    started = time.perf_counter()
    count = math.ceil(len(examples) / recipe["batch_size"]) * recipe["epochs"]
    optimizer = torch.optim.AdamW(
        action.parameters(),
        lr=recipe["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(recipe["warmup_ratio"] * count), count
    )
    total = 0
    updates = 0
    order_hashes = []
    with (output / "fit.jsonl").open("x") as handle:
        for epoch in range(recipe["epochs"]):
            groups = epoch_batches(examples, recipe["seed"], epoch, recipe["batch_size"])
            order_hashes.append(payload_sha256(groups))
            if (
                "epoch_order_sha256" in recipe
                and order_hashes[-1] != recipe["epoch_order_sha256"][epoch]
            ):
                raise ValueError("frozen epoch order mismatch")
            for indices in groups:
                batches = []
                for start in range(0, len(indices), recipe["microbatch_size"]):
                    selected = [
                        examples[i] for i in indices[start : start + recipe["microbatch_size"]]
                    ]
                    width = max(len(row["input_ids"]) for row in selected)
                    bucket = next(size for size in recipe["padding_buckets"] if size >= width)
                    collator = DataCollatorForSeq2Seq(
                        tokenizer, padding="max_length", max_length=bucket
                    )
                    batches.append(
                        {key: value.to(model.device) for key, value in collator(selected).items()}
                    )
                step_started = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"]
                step = fit_step(
                    model,
                    action,
                    batches,
                    candidate["site"],
                    candidate["application"],
                    optimizer,
                    max_grad_norm=recipe["max_grad_norm"],
                )
                scheduler.step()
                updates += 1
                total += step["target_tokens"]
                step.update(
                    epoch=epoch,
                    update=updates,
                    examples=len(indices),
                    indices_sha256=payload_sha256(indices),
                    learning_rate=lr,
                    seconds=time.perf_counter() - step_started,
                )
                handle.write(json.dumps(step, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps({"event": "fit_update", **step}), flush=True)
    expected = recipe["epochs"] * sum(
        sum(label != -100 for label in x["labels"][1:]) for x in examples
    )
    if (
        total != expected
        or updates != count
        or any(int(s["step"]) != count for s in optimizer.state.values())
    ):
        raise RuntimeError("incomplete target or optimizer coverage")
    checkpoint = output / "adapter.pt"
    with checkpoint.open("xb") as handle:
        torch.save(torch.nn.Module.state_dict(action), handle)
    return {
        "optimizer_updates": updates,
        "target_tokens": total,
        "epoch_order_sha256": order_hashes,
        "checkpoint_sha256": file_sha256(checkpoint),
        "seconds": time.perf_counter() - started,
        "training_resume_supported": False,
    }


def evaluate(model, tokenizer, rows, config, candidate, action, graders, output: Path) -> dict:
    output.mkdir()
    tokenizer.padding_side = "left"
    chunks = {}
    ids = []
    for start in range(0, len(rows), config["generation_batch"]):
        context = (
            generation_intervention(model, action, candidate["site"], candidate["application"])
            if action is not None
            else nullcontext()
        )
        with context:
            chunk = run_math_batch(
                model,
                tokenizer,
                rows[start : start + config["generation_batch"]],
                config,
                {"steps": config["generation_steps"], "checks": False},
                graders,
                reference_path=output / f"reference-{start:04d}.json",
                hash_payload=payload_sha256,
            )
        path = output / f"batch-{start:04d}.json"
        with path.open("x") as handle:
            json.dump(chunk, handle, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        chunks[path.name] = file_sha256(path)
        ids.extend(row["cluster_id"] for row in chunk["rows"])
        print(
            json.dumps(
                {
                    "event": "selection_chunk",
                    "condition": output.name,
                    "completed": len(ids),
                    "seconds": chunk["reference_seconds"],
                }
            ),
            flush=True,
        )
        if any(row["evaluator_errors"] for row in chunk["rows"]):
            raise RuntimeError("selection evaluator error; raw and scored chunks retained")
    if ids != [row["cluster_id"] for row in rows]:
        raise RuntimeError("incomplete selection identities")
    return {"completed_ids": ids, "chunk_sha256": chunks}
