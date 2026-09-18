import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from test_irc_baseline_cost import packet as container_fixture
from test_irc_baseline_audit import audit, helpers

pytest_plugins = ("test_irc_baseline", "test_irc_baseline_audit")


def test_regrading_timeout_reports_locator_and_exception_without_answers(chunks, monkeypatch):
    monkeypatch.setattr(
        audit(),
        "score_math_outputs",
        lambda *args: {
            "primary": None,
            "sensitivity": None,
            "evaluator_errors": {"primary": "TimeoutException"},
        },
    )
    with pytest.raises(ValueError, match="replay_error.*cluster=0.*TimeoutException") as failure:
        audit().audit_condition(*chunks)
    assert chunks[1][0]["solution"] not in str(failure.value)


@pytest.mark.parametrize("valid", [True, False])
def test_generation_token_bound_uses_model_vocabulary_not_tokenizer_length(chunks, valid):
    directory, _, config, receipt, tokenizer, _ = chunks
    config["vocab_size"] = len(tokenizer) + 2
    for name in ("reference-0000.json", "batch-0000.json"):
        path = directory / name
        packet = json.loads(path.read_text())
        for row in packet["rows"]:
            row["token_ids"].insert(-1, len(tokenizer) + (1 if valid else 2))
            row["generated_tokens"] += 1
        path.write_text(json.dumps(packet))
    receipt["chunk_sha256"]["batch-0000.json"] = helpers().file_sha256(
        directory / "batch-0000.json"
    )
    if valid:
        rows, _ = audit().audit_condition(*chunks)
        assert len(rows) == 2
    else:
        with pytest.raises(ValueError):
            audit().audit_condition(*chunks)


@pytest.mark.parametrize("model", ["qwen3"], indirect=True)
def test_eight_real_tiny_workers_to_complete_audit_with_synthetic_container_receipt(
    baseline, runtime, packet_config, model, tokenizer, tmp_path
):
    root = Path(__file__).parents[1]
    fit_path = Path(packet_config["packet"])
    fit = json.loads(fit_path.read_text())
    fit.pop("content_sha256")
    fit["allocation"] = "fit"
    fit["content_sha256"] = baseline.payload_sha256(fit)
    fit_path.write_text(json.dumps(fit))
    selection = {
        "allocation": "selection",
        "rows": [
            {
                **fit["rows"][i % 2],
                "cluster_id": f"selection-{i}",
                "row_id": f"s{i}",
                "allocation": "selection",
                "type": "Algebra",
                "level": "Level 1",
            }
            for i in range(8)
        ],
    }
    selection["content_sha256"] = baseline.payload_sha256(selection)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }} answer "
        "{% else %}{{ m.reasoning_content }} {{ m.content }}{{ eos_token }}\n"
        "{% endif %}{% endfor %}"
    )
    model_config = copy.deepcopy(model.config)
    model_config.num_hidden_layers = 24
    model_config.layer_types = [model_config.layer_types[0]] * 24
    expanded = type(model)(model_config)
    expanded.generation_config.eos_token_id = tokenizer.eos_token_id
    expanded.generation_config.pad_token_id = tokenizer.pad_token_id
    model_path = tmp_path / "model"
    expanded.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config = {
        **packet_config,
        "packet_sha256": baseline.file_sha256(fit_path),
        "groups": {"fit": packet_config["cluster_ids"]},
        "maximum_length": 128,
        "attention": "eager",
        "generation_batch": 2,
        "generation_steps": 4,
        "eos_token_ids": [tokenizer.eos_token_id],
        "selection": {
            "packet": str(selection_path),
            "num_questions": 8,
            "packet_sha256": baseline.file_sha256(selection_path),
        },
        "recipe": {
            "epochs": 2,
            "batch_size": 2,
            "microbatch_size": 1,
            "padding_buckets": [64],
            "learning_rate": 0.0009,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1,
            "seed": 20260907,
        },
        "candidates": [
            {"rank": rank, "site": site, "application": application}
            for application in ("prompt", "full")
            for rank in (4, 8)
            for site in (17, 23)
        ],
        "zero_shares": [[i] for i in range(8)],
    }
    config["target_examples_sha256"] = baseline.payload_sha256(
        baseline.load_examples(config, root, tokenizer)["fit"]
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    stage = tmp_path / "stage"

    def run_worker(index):
        command = [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            str(root / "scripts/run_irc_baseline.py"),
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--output",
            str(stage / f"candidate_{index:02d}" / "run"),
            "--candidate",
            str(index),
            "--device",
            "cpu",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_worker, range(8)))
    containers, dispatch = container_fixture()
    dispatch["config_sha256"] = baseline.file_sha256(config_path)
    for name, value in (("terminal", {"containers": containers}), ("dispatch", dispatch)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    output = tmp_path / "audit.json"
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        str(root / "scripts/audit_irc_baseline.py"),
        "--root",
        str(root),
        "--config",
        str(config_path),
        "--input",
        str(stage),
        "--model",
        str(model_path),
        "--dispatch",
        str(tmp_path / "dispatch.json"),
        "--terminal",
        str(tmp_path / "terminal.json"),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (
        result.stdout + result.stderr + (output.read_text() if output.exists() else "")
    )
    report = json.loads(output.read_text())
    assert report["status"] == "pass"
    assert len(report["analysis"]["conditions"]) == 9
    assert all(row["questions"] == 8 for row in report["analysis"]["conditions"])
    assert len(report["workers"]) == 8
    assert all(row["fit"]["optimizer_updates"] == 2 for row in report["workers"])
    assert not report["sota_achieved"]
    assert not report["analysis"]["response_launch_authorized"]
    candidate = stage / "candidate_00" / "run"
    chunk_path = candidate / "selection" / "batch-0000.json"
    chunk = json.loads(chunk_path.read_text())
    chunk["rows"][0]["primary"]["correct"] = not chunk["rows"][0]["primary"]["correct"]
    chunk_path.write_text(json.dumps(chunk))
    receipt_path = candidate / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["selection"]["chunk_sha256"][chunk_path.name] = baseline.file_sha256(chunk_path)
    receipt_path.write_text(json.dumps(receipt))
    command[-1] = str(tmp_path / "corrupted-audit.json")
    assert subprocess.run(command, capture_output=True).returncode != 0
    failed = json.loads(Path(command[-1]).read_text())
    assert "selection/batch-0000.json" in failed["workers"][0]["input_file_sha256"]
    assert "cluster=selection-0" in failed["failure"]["message"]
