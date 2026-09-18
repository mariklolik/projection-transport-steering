import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from math_verify.errors import TimeoutException
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from test_reft_conformance import model


def test_real_runtime_executable_exists():
    assert (Path(__file__).parents[1] / "scripts/run_irc_real_runtime.py").exists()


@pytest.fixture
def runtime(monkeypatch):
    root = Path(__file__).parents[1]
    source = root / "scripts/run_irc_real_runtime.py"
    assert source.exists(), "the real-question runtime executable is missing"
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return runpy.run_path(str(source))


@pytest.fixture
def packet_config(runtime, tmp_path):
    rows = [
        {
            "row_id": f"row-{index}",
            "cluster_id": f"group-{index}",
            "split": "train",
            "allocation": "fit",
            "reference_eligible": True,
            "sensitivity_supported": True,
            "problem": "Compute " + "one " * (index + 1),
            "solution": r"The answer is \boxed{1}.",
        }
        for index in range(2)
    ]
    packet = {"allocation": "runtime", "source_allocation": "fit", "rows": rows}
    packet["content_sha256"] = runtime["payload_sha256"](packet)
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet))
    return {
        "packet": str(path),
        "packet_sha256": runtime["file_sha256"](path),
        "cluster_ids": [row["cluster_id"] for row in rows],
        "num_questions": 2,
        "instruction": "Solve the problem.\n\n",
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "conditions": {"profile": {"batch": 2, "steps": 8, "checks": True}},
        "expected_versions": {},
        "model_files_sha256": {},
        "source_sha256": {
            "scripts/run_irc_real_runtime.py": runtime["file_sha256"](
                Path(__file__).parents[1] / "scripts/run_irc_real_runtime.py"
            )
        },
    }


def test_runtime_packet_verifies_hash_order_and_training_boundary(runtime, packet_config):
    rows = runtime["load_packet"](packet_config, Path("/"))
    assert [row["cluster_id"] for row in rows] == packet_config["cluster_ids"]
    assert all(row["allocation"] == "fit" for row in rows)
    changed = {**packet_config, "packet_sha256": "0" * 64}
    with pytest.raises(ValueError, match="hash"):
        runtime["load_packet"](changed, Path("/"))
    with pytest.raises(ValueError, match="identit"):
        runtime["load_packet"]({**packet_config, "cluster_ids": ["group-1", "group-0"]}, Path("/"))


@pytest.mark.parametrize("change", ["duplicate", "test", "reference", "selection"])
def test_runtime_packet_rejects_rehashed_invalid_rows(runtime, packet_config, change):
    path = Path(packet_config["packet"])
    packet = json.loads(path.read_text())
    packet.pop("content_sha256")
    if change == "duplicate":
        packet["rows"][1]["cluster_id"] = packet["rows"][0]["cluster_id"]
    elif change == "reference":
        packet["rows"][0]["reference_eligible"] = False
    elif change == "test":
        packet["rows"][0]["split"] = "test"
    else:
        packet["rows"][0]["allocation"] = "selection"
    packet["content_sha256"] = runtime["payload_sha256"](packet)
    path.write_text(json.dumps(packet))
    packet_config["packet_sha256"] = runtime["file_sha256"](path)
    with pytest.raises(ValueError):
        runtime["load_packet"](packet_config, Path("/"))


def test_runtime_grades_primary_and_supported_sensitivity_once(runtime):
    root = Path(__file__).parents[1]
    extract, equivalent = runtime["load_math_author"](root)
    boxed = runpy.run_path(str(root / ".external/math-author-source/modeling/dataset/util.py"))[
        "last_boxed_only_string"
    ]
    result = runtime["score_output"](
        r"\boxed{1}", r"\boxed{1}", True, True, extract, equivalent, boxed
    )
    assert result["primary"]["correct"] is True
    assert result["sensitivity"]["correct"] is True
    assert result["evaluator_errors"] == {}
    truncated = runtime["score_output"](
        r"\boxed{1}", r"\boxed{1}", False, False, extract, equivalent, boxed
    )
    assert truncated["primary"]["correct"] is False
    assert truncated["sensitivity"] is None
    assert truncated["evaluator_errors"] == {}


@pytest.mark.parametrize("error_class", [TimeoutException, ValueError])
def test_runtime_retains_sensitivity_exception_without_losing_primary(
    runtime, monkeypatch, error_class
):
    root = Path(__file__).parents[1]
    extract, equivalent = runtime["load_math_author"](root)
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise error_class("retained diagnostic")

    monkeypatch.setitem(runtime["score_output"].__globals__, "score_math_completion", fail)
    result = runtime["score_output"](
        r"\boxed{1}", r"\boxed{1}", True, True, extract, equivalent, extract
    )
    assert result["primary"]["correct"] is True
    assert result["sensitivity"] is None
    assert result["evaluator_errors"] == {"sensitivity": error_class.__name__}
    assert len(calls) == 1


@pytest.fixture
def tokenizer():
    vocabulary = {f"token-{index}": index for index in range(3, 100)}
    vocabulary.update({"[PAD]": 0, "[UNK]": 1, "[EOS]": 100, "one": 2})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}{% if add_generation_prompt %} answer {% endif %}"
    tokenizer.padding_side = "left"
    return tokenizer


def test_finished_reference_survives_a_later_check_failure(
    runtime, packet_config, model, tokenizer, monkeypatch, tmp_path
):
    root = Path(__file__).parents[1]
    rows = runtime["load_packet"](packet_config, root)
    extract, equivalent = runtime["load_math_author"](root)
    model.generation_config.eos_token_id = 100
    model.generation_config.pad_token_id = 0
    namespace = runtime["run_batch"].__globals__
    advance = namespace["advance_branches"]
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("check failed after completed reference")
        return advance(*args, **kwargs)

    monkeypatch.setitem(namespace, "advance_branches", fail_second)
    checkpoint = tmp_path / "reference.json"
    with pytest.raises(RuntimeError, match="check failed"):
        runtime["run_batch"](
            model,
            tokenizer,
            rows,
            packet_config,
            packet_config["conditions"]["profile"],
            (extract, equivalent, extract),
            reference_path=checkpoint,
            hash_payload=runtime["payload_sha256"],
        )
    reference = json.loads(checkpoint.read_text())
    assert [row["cluster_id"] for row in reference["rows"]] == packet_config["cluster_ids"]
    assert all(row["token_ids"] for row in reference["rows"])
    assert all("primary" not in row for row in reference["rows"])
    assert len(calls) == 2


@pytest.mark.parametrize("attention", ["sdpa", "eager"])
def test_real_tiny_model_runtime_persists_all_rows_and_refuses_overwrite(
    runtime, packet_config, model, tokenizer, tmp_path, attention
):
    root = Path(__file__).parents[1]
    snapshot = tmp_path / "model"
    model.generation_config.eos_token_id = 100
    model.generation_config.pad_token_id = 0
    model.save_pretrained(snapshot)
    tokenizer.save_pretrained(snapshot)
    packet_config["model_files_sha256"] = {
        "config.json": runtime["file_sha256"](snapshot / "config.json")
    }
    config_path = tmp_path / "config.json"
    packet_config["attention"] = attention
    config_path.write_text(json.dumps(packet_config))
    output = tmp_path / "output"
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        str(root / "scripts/run_irc_real_runtime.py"),
        "--config",
        str(config_path),
        "--condition",
        "profile",
        "--model",
        str(snapshot),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "pass"
    assert receipt["attention"] == attention
    assert receipt["expected_ids"] == receipt["completed_ids"] == packet_config["cluster_ids"]
    assert receipt["scientific_method_comparison"] is False
    chunks = list(output.glob("batch-*.json"))
    assert len(chunks) == 1
    chunk = json.loads(chunks[0].read_text())
    assert len(chunk["rows"]) == 2
    assert all(not row["evaluator_errors"] for row in chunk["rows"])
    assert chunk["checks"]["zero_replay_exact"] is True
    assert chunk["checks"]["split_replay_exact"] is True
    assert chunk["checks"]["parent_cache_unchanged"] is True
    assert receipt["chunk_sha256"][chunks[0].name] == runtime["file_sha256"](chunks[0])
    before = hashlib.sha256((output / "receipt.json").read_bytes()).hexdigest()
    repeated = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert hashlib.sha256((output / "receipt.json").read_bytes()).hexdigest() == before


def test_absorbing_rows_do_not_claim_an_executed_action(
    runtime, packet_config, model, tokenizer, tmp_path
):
    root = Path(__file__).parents[1]
    rows = runtime["load_packet"](packet_config, root)
    extract, equivalent = runtime["load_math_author"](root)
    model.generation_config.eos_token_id = list(range(model.config.vocab_size))
    model.generation_config.pad_token_id = 0
    chunk = runtime["run_batch"](
        model,
        tokenizer,
        rows,
        packet_config,
        packet_config["conditions"]["profile"],
        (extract, equivalent, extract),
        reference_path=tmp_path / "reference.json",
        hash_payload=runtime["payload_sha256"],
    )
    assert all(row["generated_tokens"] == 1 for row in chunk["rows"])
    assert chunk["check_coverage"]["at_risk_at_cut"] == 0
    assert chunk["check_coverage"]["action_forward_calls"] == 0
    assert chunk["check_coverage"]["intervention_path"] == "not_exercised_absorbing"
