import importlib
import json
import math
import runpy
import sys
from pathlib import Path

import pytest

from projection_transport_steering.outcome_score import score_math_outputs
from projection_transport_steering.outcome_score_runtime import rollout_seed
from projection_transport_steering.reft_training import epoch_batches

pytest_plugins = ("test_reft_training",)


def helpers():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("materialize_outcome_score_data")


def audit():
    helpers()
    return importlib.import_module("irc_baseline_audit")


@pytest.fixture
def chunks(tmp_path, native_tokenizer):
    helper = helpers()
    root = Path(__file__).parents[1]
    extract, equivalent = importlib.import_module("audit_irc_sources").load_math_author(root)
    boxed = runpy.run_path(str(root / ".external/math-author-source/modeling/dataset/util.py"))[
        "last_boxed_only_string"
    ]
    graders = (extract, equivalent, boxed)
    sources = [
        {
            "cluster_id": str(i),
            "row_id": f"r{i}",
            "sensitivity_supported": i == 0,
            "problem": f"Compute {i}.",
            "solution": r"\boxed{2}",
        }
        for i in range(2)
    ]
    config = {
        "instruction": "Solve: ",
        "generation_batch": 2,
        "generation_steps": 32,
        "eos_token_ids": [native_tokenizer.eos_token_id],
        "vocab_size": len(native_tokenizer),
    }
    raw_rows = []
    for source in sources:
        prompt = native_tokenizer.apply_chat_template(
            [{"role": "user", "content": config["instruction"] + source["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_ids = native_tokenizer.encode(prompt, add_special_tokens=False)
        tokens = native_tokenizer.encode(r"\boxed{2}", add_special_tokens=False)
        tokens.append(native_tokenizer.eos_token_id)
        raw_rows.append(
            {
                **{key: source[key] for key in ("cluster_id", "row_id", "sensitivity_supported")},
                "seed": rollout_seed(source["cluster_id"], 0),
                "prompt_sha256": helper.payload_sha256(prompt),
                "prompt_tokens": len(prompt_ids),
                "prompt_token_sha256": helper.payload_sha256(prompt_ids),
                "token_ids": tokens,
                "completion": native_tokenizer.decode(tokens, skip_special_tokens=True),
                "generated_tokens": len(tokens),
                "terminated": True,
            }
        )
    raw = {"rows": raw_rows, "reference_seconds": 1.5}
    scored = {
        **raw,
        "checks": {},
        "check_tokens": {},
        "check_coverage": {"intervention_path": "not_requested"},
        "check_seconds": 0.01,
        "grading_seconds": 0.1,
        "rows": [
            {
                **row,
                **score_math_outputs(
                    row["completion"],
                    source["solution"],
                    True,
                    source["sensitivity_supported"],
                    *graders,
                ),
            }
            for row, source in zip(raw_rows, sources, strict=True)
        ],
    }
    (tmp_path / "reference-0000.json").write_text(json.dumps(raw))
    (tmp_path / "batch-0000.json").write_text(json.dumps(scored))
    receipt = {
        "completed_ids": [r["cluster_id"] for r in sources],
        "chunk_sha256": {"batch-0000.json": helper.file_sha256(tmp_path / "batch-0000.json")},
    }
    return tmp_path, sources, config, receipt, native_tokenizer, graders


def test_raw_scored_source_and_regrading_conformance(chunks):
    rows, evidence = audit().audit_condition(*chunks)
    assert len(rows) == 2 and all(row["primary"]["correct"] for row in rows)
    assert len(evidence["file_sha256"]) == 2
    assert evidence["generation_seconds"] == 1.5


@pytest.mark.parametrize("terminated", [True, False])
def test_eos_exactly_at_cap_is_distinct_from_exhaustion(chunks, terminated):
    directory, sources, config, receipt, _, graders = chunks
    raw_path = directory / "reference-0000.json"
    scored_path = directory / "batch-0000.json"
    raw, scored = json.loads(raw_path.read_text()), json.loads(scored_path.read_text())
    for before, row, source in zip(raw["rows"], scored["rows"], sources, strict=True):
        if not terminated:
            before["token_ids"].pop()
            before["generated_tokens"] -= 1
        before["terminated"] = terminated
        row.update(before)
        row.update(
            score_math_outputs(
                row["completion"],
                source["solution"],
                terminated,
                source["sensitivity_supported"],
                *graders,
            )
        )
    config["generation_steps"] = raw["rows"][0]["generated_tokens"]
    raw_path.write_text(json.dumps(raw))
    scored_path.write_text(json.dumps(scored))
    receipt["chunk_sha256"][scored_path.name] = helpers().file_sha256(scored_path)
    rows, _ = audit().audit_condition(*chunks)
    assert all(row["primary"]["correct"] is terminated for row in rows)


@pytest.mark.parametrize(
    "fault",
    [
        "tokens",
        "score",
        "seed",
        "support",
        "hash",
        "missing",
        "extra",
        "cap",
        "eos",
        "error",
        "prompt",
        "order",
    ],
)
def test_condition_audit_rejects_corruption_even_when_chunk_is_rehashed(chunks, fault):
    directory, _, _, receipt, _, _ = chunks
    raw_path, scored_path = directory / "reference-0000.json", directory / "batch-0000.json"
    raw, scored = json.loads(raw_path.read_text()), json.loads(scored_path.read_text())
    if fault == "tokens":
        raw["rows"][0]["token_ids"][0] += 1
    elif fault == "score":
        scored["rows"][0]["primary"]["correct"] = False
    elif fault in {"seed", "prompt"}:
        key = "seed" if fault == "seed" else "prompt_tokens"
        for data in (raw, scored):
            data["rows"][0][key] += 1
    elif fault == "support":
        for data in (raw, scored):
            data["rows"][0]["sensitivity_supported"] = False
    elif fault == "hash":
        receipt["chunk_sha256"]["batch-0000.json"] = "0" * 64
    elif fault == "missing":
        raw["rows"].pop()
        scored["rows"].pop()
    elif fault == "extra":
        (directory / "reference-0002.json").write_text(json.dumps(raw))
    elif fault == "cap":
        for data in (raw, scored):
            data["rows"][0]["terminated"] = False
    elif fault == "eos":
        for data in (raw, scored):
            data["rows"][0]["token_ids"][0] = chunks[4].eos_token_id
    elif fault == "error":
        scored["rows"][0]["evaluator_errors"] = {"primary": "TimeoutException"}
    else:
        raw["rows"].reverse()
        scored["rows"].reverse()
    raw_path.write_text(json.dumps(raw))
    scored_path.write_text(json.dumps(scored))
    if fault != "hash":
        receipt["chunk_sha256"]["batch-0000.json"] = helpers().file_sha256(scored_path)
    with pytest.raises(ValueError, match="cluster=0.*primary" if fault == "score" else None):
        audit().audit_condition(*chunks)


@pytest.fixture
def fit_packet(tmp_path):
    helper = helpers()
    examples = [
        {"input_ids": [1] * (i + 3), "labels": [-100, -100] + [1] * (i + 1), "prompt_length": 2}
        for i in range(5)
    ]
    recipe = {
        "epochs": 2,
        "batch_size": 4,
        "microbatch_size": 2,
        "seed": 20260907,
        "learning_rate": 0.0009,
        "warmup_ratio": 0.1,
    }
    rows, hashes = [], []
    for epoch in range(2):
        groups = epoch_batches(examples, recipe["seed"], epoch, 4)
        hashes.append(helper.payload_sha256(groups))
        for indices in groups:
            rows.append(
                {
                    "epoch": epoch,
                    "update": len(rows) + 1,
                    "examples": len(indices),
                    "indices_sha256": helper.payload_sha256(indices),
                    "target_tokens": sum(i + 1 for i in indices),
                    "active_positions": sum(i + 3 for i in indices),
                    "microbatches": math.ceil(len(indices) / 2),
                    "loss": 1.0,
                    "gradient_norm": 2.0,
                    "changed_parameters": 0 if not rows else 3,
                    "learning_rate": [0, 0.0009, 0.0006, 0.0003][len(rows)],
                    "seconds": 0.1,
                }
            )
    (tmp_path / "fit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "adapter.pt").write_bytes(b"fixture checkpoint")
    recipe["epoch_order_sha256"] = hashes
    receipt = {
        "optimizer_updates": 4,
        "target_tokens": 30,
        "epoch_order_sha256": hashes,
        "checkpoint_sha256": helper.file_sha256(tmp_path / "adapter.pt"),
        "seconds": 1.0,
        "training_resume_supported": False,
    }
    return tmp_path, receipt, recipe, examples, {"application": "full"}


def test_fit_audit_reconstructs_partial_batches_and_zero_initial_lr(fit_packet):
    result = audit().audit_fit(*fit_packet)
    assert result["optimizer_updates"] == 4
    assert result["target_tokens"] == 30
    assert result["epochs"][0]["examples"] == 5
    assert result["epochs"][0]["token_weighted_loss"] == 1
    assert result["gradient_norm_is_pre_clip"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "learning_rate",
        "indices_sha256",
        "target_tokens",
        "active_positions",
        "loss",
        "changed_parameters",
        "checkpoint",
    ],
)
def test_fit_audit_rejects_false_complete_receipt(fit_packet, fault):
    path = fit_packet[0] / "fit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "missing":
        rows.pop()
    elif fault == "checkpoint":
        fit_packet[1]["checkpoint_sha256"] = "0" * 64
    else:
        rows[1][fault] = "wrong" if fault == "indices_sha256" else 0
        if fault == "loss":
            rows[1][fault] = float("nan")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        audit().audit_fit(*fit_packet)
