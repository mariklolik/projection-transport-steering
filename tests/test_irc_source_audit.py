import json
import subprocess
import sys
from pathlib import Path


def test_irc_source_audit_reconciles_complete_sources_without_model_outcomes(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "source-audit.json"
    command = [sys.executable, str(root / "scripts/audit_irc_sources.py"), "--output", str(output)]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    audit = json.loads(output.read_text())
    assert audit["math"]["split_rows"] == {"train": 7500, "test": 5000}
    assert audit["math"]["author_linked_rows"] == 12500
    assert audit["math"]["union_missing_rows"] == 0
    assert audit["math"]["union_extra_rows"] == 0
    assert audit["math"]["full_record_train_test_overlap"] == 0
    gold = audit["training_gold_self_check"]
    assert gold["self_correct"] + len(gold["failures"]) == 7500
    assert gold["failures"]
    assert audit["benchmark_ready"] is False
    assert audit["model_outcomes_inspected"] == 0
    assert {value["config"]["model_type"] for value in audit["models"].values()} == {
        "qwen3",
        "llama",
    }
    before = output.read_bytes()
    retry = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert retry.returncode != 0
    assert output.read_bytes() == before
