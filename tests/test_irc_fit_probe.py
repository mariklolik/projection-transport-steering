import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ("test_irc_real_runtime",)


def test_fit_probe_cli_records_updates_without_exporting_a_baseline(
    runtime, packet_config, model, tokenizer, tmp_path
):
    root = Path(__file__).parents[1]
    script = root / "scripts/run_irc_fit_probe.py"
    assert script.exists(), "the bounded actual-weight fit probe is missing"
    packet_path = Path(packet_config["packet"])
    packet = json.loads(packet_path.read_text())
    packet.pop("content_sha256")
    packet["allocation"] = "fit"
    packet["content_sha256"] = runtime["payload_sha256"](packet)
    packet_path.write_text(json.dumps(packet))
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }} answer "
        "{% else %}{{ m.reasoning_content }} {{ m.content }}{{ eos_token }}\n"
        "{% endif %}{% endfor %}"
    )
    model_path = tmp_path / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config = {
        **packet_config,
        "packet_sha256": runtime["file_sha256"](packet_path),
        "groups": {"fixture": packet_config["cluster_ids"]},
        "applications": ["prompt", "full"],
        "repeats": 1,
        "rank": 4,
        "site": 1,
        "seed": 20260907,
        "maximum_length": 128,
        "pad_multiple": 8,
        "attention": "eager",
        "learning_rate": 0.0009,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "fit-probe"
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        str(script),
        "--config",
        str(config_path),
        "--model",
        str(model_path),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "pass"
    assert receipt["base_parameter_versions_unchanged"] is True
    assert receipt["baseline_eligible"] is False
    assert len(receipt["step_sha256"]) == 2
    for name, digest in receipt["step_sha256"].items():
        assert runtime["file_sha256"](output / name) == digest
        step = json.loads((output / name).read_text())
        assert step["gradient_norm"] > 0 and step["changed_parameters"] > 0
    before = runtime["file_sha256"](output / "receipt.json")
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert runtime["file_sha256"](output / "receipt.json") == before
