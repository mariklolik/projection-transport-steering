import json
import math
from pathlib import Path

from materialize_outcome_score_data import file_sha256, payload_sha256
from projection_transport_steering.outcome_score import score_math_outputs
from projection_transport_steering.outcome_score_runtime import rollout_seed
from projection_transport_steering.reft_training import epoch_batches


def audit_condition(
    directory: Path, sources: list[dict], config: dict, receipt: dict, tokenizer, graders: tuple
) -> tuple[list[dict], dict]:
    starts = list(range(0, len(sources), config["generation_batch"]))
    names = {f"{kind}-{start:04d}.json" for kind in ("reference", "batch") for start in starts}
    ids = [row["cluster_id"] for row in sources]
    if (
        not sources
        or len(ids) != len(set(ids))
        or {path.name for path in directory.iterdir()} != names
        or receipt["completed_ids"] != ids
        or set(receipt["chunk_sha256"]) != {f"batch-{start:04d}.json" for start in starts}
    ):
        raise ValueError("incomplete or unexpected condition files or identities")
    hashes = {name: file_sha256(directory / name) for name in sorted(names)}
    if any(hashes[name] != digest for name, digest in receipt["chunk_sha256"].items()):
        raise ValueError("scored chunk hash mismatch")
    all_rows = []
    costs = {"generation_seconds": 0.0, "grading_seconds": 0.0, "check_seconds": 0.0}
    raw_keys = {
        "cluster_id",
        "row_id",
        "seed",
        "prompt_sha256",
        "prompt_tokens",
        "prompt_token_sha256",
        "token_ids",
        "completion",
        "generated_tokens",
        "terminated",
        "sensitivity_supported",
    }
    for start in starts:
        raw = json.loads((directory / f"reference-{start:04d}.json").read_text())
        scored = json.loads((directory / f"batch-{start:04d}.json").read_text())
        expected = sources[start : start + config["generation_batch"]]
        if (
            len(raw["rows"]) != len(expected)
            or len(scored["rows"]) != len(expected)
            or raw["reference_seconds"] != scored["reference_seconds"]
            or scored["checks"]
            or scored["check_tokens"]
            or scored["check_coverage"] != {"intervention_path": "not_requested"}
        ):
            raise ValueError("incomplete chunk or mismatched generation contract")
        for key, source_key in (
            ("generation_seconds", "reference_seconds"),
            ("grading_seconds", "grading_seconds"),
            ("check_seconds", "check_seconds"),
        ):
            seconds = scored[source_key]
            if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("invalid component timing")
            costs[key] += seconds
        for source, before, row in zip(expected, raw["rows"], scored["rows"], strict=True):
            if set(before) != raw_keys or any(row.get(key) != before[key] for key in raw_keys):
                raise ValueError("raw and scored output mismatch")
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": config["instruction"] + source["problem"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            identity = {
                **{key: source[key] for key in ("cluster_id", "row_id", "sensitivity_supported")},
                "seed": rollout_seed(source["cluster_id"], 0),
                "prompt_sha256": payload_sha256(prompt),
                "prompt_tokens": len(prompt_ids),
                "prompt_token_sha256": payload_sha256(prompt_ids),
            }
            tokens = row["token_ids"]
            endings = [i for i, token in enumerate(tokens) if token in config["eos_token_ids"]]
            if (
                any(row[key] != value for key, value in identity.items())
                or type(row["terminated"]) is not bool
                or not all(
                    type(token) is int and 0 <= token < config["vocab_size"] for token in tokens
                )
                or not 0 < len(tokens) <= config["generation_steps"]
                or row["generated_tokens"] != len(tokens)
                or row["completion"] != tokenizer.decode(tokens, skip_special_tokens=True)
                or (
                    endings != [len(tokens) - 1]
                    if row["terminated"]
                    else bool(endings) or len(tokens) != config["generation_steps"]
                )
            ):
                raise ValueError("source, token, prompt, seed or EOS/cap mismatch")
            if row["evaluator_errors"]:
                raise ValueError("evaluator error is not an admissible exclusion")
            grades = score_math_outputs(
                row["completion"],
                source["solution"],
                row["terminated"],
                source["sensitivity_supported"],
                *graders,
            )
            locator = f"path={directory} chunk={start:04d} cluster={row['cluster_id']}"
            digests = f"raw_sha256={hashes[f'reference-{start:04d}.json']} scored_sha256={hashes[f'batch-{start:04d}.json']}"
            if grades["evaluator_errors"]:
                raise ValueError(f"replay_error {locator} {grades['evaluator_errors']} {digests}")
            different = [key for key, value in grades.items() if row[key] != value]
            if different:
                raise ValueError(f"replay_mismatch {locator} fields={different} {digests}")
            all_rows.append(row)
    return all_rows, {"file_sha256": hashes, **costs, "replay_checks_requested": False}


def audit_fit(
    directory: Path, receipt: dict, recipe: dict, examples: list[dict], candidate: dict
) -> dict:
    paths = [directory / "fit.jsonl", directory / "adapter.pt"]
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    orders = [
        epoch_batches(examples, recipe["seed"], e, recipe["batch_size"])
        for e in range(recipe["epochs"])
    ]
    hashes = [payload_sha256(groups) for groups in orders]
    count = sum(len(groups) for groups in orders)
    target_total = recipe["epochs"] * sum(
        sum(label != -100 for label in example["labels"][1:]) for example in examples
    )
    if (
        len(rows) != count
        or receipt["optimizer_updates"] != count
        or receipt["target_tokens"] != target_total
        or receipt["epoch_order_sha256"] != hashes
        or recipe.get("epoch_order_sha256", hashes) != hashes
        or receipt["checkpoint_sha256"] != file_sha256(paths[1])
        or not paths[1].stat().st_size
        or receipt["training_resume_supported"] is not False
    ):
        raise ValueError("incomplete fit or checkpoint/order mismatch")
    warmup = math.ceil(recipe["warmup_ratio"] * count)
    epochs = []
    cursor = 0
    for epoch, groups in enumerate(orders):
        current = []
        for indices in groups:
            row = rows[cursor]
            scale = (
                cursor / max(1, warmup)
                if cursor < warmup
                else (count - cursor) / max(1, count - warmup)
            )
            expected_lr = recipe["learning_rate"] * scale
            selected = [examples[i] for i in indices]
            expected = {
                "epoch": epoch,
                "update": cursor + 1,
                "examples": len(indices),
                "indices_sha256": payload_sha256(indices),
                "target_tokens": sum(sum(y != -100 for y in x["labels"][1:]) for x in selected),
                "microbatches": math.ceil(len(indices) / recipe["microbatch_size"]),
                "active_positions": sum(
                    2 * min(7, x["prompt_length"] // 2)
                    + (
                        len(x["input_ids"]) - x["prompt_length"]
                        if candidate["application"] == "full"
                        else 0
                    )
                    for x in selected
                ),
            }
            if (
                any(row[key] != value for key, value in expected.items())
                or not math.isclose(row["learning_rate"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
                or not all(
                    math.isfinite(row[key]) and row[key] >= 0
                    for key in ("loss", "seconds", "gradient_norm")
                )
                or row["gradient_norm"] <= 0
                or type(row["changed_parameters"]) is not int
                or (
                    row["changed_parameters"] != 0
                    if expected_lr == 0
                    else not 1 <= row["changed_parameters"] <= 3
                )
            ):
                raise ValueError(
                    "fit update differs from frozen schedule or finite optimizer contract"
                )
            current.append(row)
            cursor += 1
        tokens = sum(row["target_tokens"] for row in current)
        epochs.append(
            {
                "epoch": epoch,
                "updates": len(current),
                "examples": sum(row["examples"] for row in current),
                "target_tokens": tokens,
                "token_weighted_loss": sum(row["loss"] * row["target_tokens"] for row in current)
                / tokens,
                "update_seconds": sum(row["seconds"] for row in current),
            }
        )
    seconds = receipt["seconds"]
    if not math.isfinite(seconds) or seconds < sum(row["seconds"] for row in rows):
        raise ValueError("invalid complete fit duration")
    return {
        "optimizer_updates": count,
        "target_tokens": target_total,
        "epochs": epochs,
        "microbatches": sum(row["microbatches"] for row in rows),
        "fit_seconds": seconds,
        "gradient_norm_is_pre_clip": True,
        "file_sha256": {path.name: file_sha256(path) for path in paths},
    }


def audit_component_cost(fit: dict, zero: dict, selection: dict, runner_seconds: float) -> dict:
    seconds = fit["fit_seconds"] + sum(
        part[key]
        for part in (zero, selection)
        for key in ("generation_seconds", "grading_seconds", "check_seconds")
    )
    if (
        not math.isfinite(seconds)
        or not math.isfinite(runner_seconds)
        or seconds > runner_seconds + 0.001
    ):
        raise ValueError("sequential component time exceeds complete runner time")
    return {
        "accounted_seconds": seconds,
        "unattributed_runner_seconds": runner_seconds - seconds,
        "clock_tolerance_seconds": 0.001,
        "nested_update_times_added": False,
    }
