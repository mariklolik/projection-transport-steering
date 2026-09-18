import json
import subprocess
import sys
from pathlib import Path

from test_reft_conformance import model


def test_branch_probe_records_real_model_checks_without_task_scores(model, tmp_path):
    root = Path(__file__).parents[1]
    snapshot = tmp_path / "model"
    model.generation_config.eos_token_id = model.config.vocab_size - 1
    model.generation_config.pad_token_id = 0
    model.save_pretrained(snapshot)
    output = tmp_path / "probe.json"
    command = [
        sys.executable,
        str(root / "scripts/run_irc_branch_probe.py"),
        "--model",
        str(snapshot),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--batch",
        "2",
        "--prompt-tokens",
        "8",
        "--steps",
        "8",
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "pass"
    assert result["task_scoring_performed"] is False
    assert result["synthetic_token_inputs"] is True
    assert result["zero_replay_exact"] is True
    assert result["split_replay_exact"] is True
    assert result["parent_cache_unchanged"] is True
    assert result["batch"] == 2
    assert result["steps"] == 8
