import json
import sys
from types import SimpleNamespace

import pytest
import torch

from test_irc_baseline import baseline


def test_evaluator_failure_retains_raw_and_scored_chunks(baseline, monkeypatch, tmp_path):
    import irc_baseline_support as support

    calls = []

    def failed_batch(*args, reference_path, **kwargs):
        calls.append(True)
        reference_path.write_text(json.dumps({"raw": "preserved"}))
        return {
            "rows": [{"cluster_id": "q", "evaluator_errors": {"primary": "timeout"}}],
            "reference_seconds": 1.0,
        }

    monkeypatch.setattr(support, "run_math_batch", failed_batch)
    output = tmp_path / "selection"
    with pytest.raises(RuntimeError, match="evaluator error"):
        baseline.evaluate(
            None,
            SimpleNamespace(),
            [{"cluster_id": "q"}],
            {"generation_batch": 8, "generation_steps": 8192},
            {},
            None,
            (),
            output,
        )
    assert calls == [True]
    assert json.loads((output / "reference-0000.json").read_text()) == {"raw": "preserved"}
    assert json.loads((output / "batch-0000.json").read_text())["rows"][0]["evaluator_errors"]


@pytest.mark.parametrize("attention", ["fp32_prefill_flex", "fp32_prefill_flex_eager", "sdpa"])
def test_baseline_registers_attention_before_model_load(baseline, monkeypatch, tmp_path, attention):
    events = []
    config = {
        "attention": attention,
        "source_sha256": {},
        "candidates": [{"rank": 4, "site": 1, "application": "full"}],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "run_irc_baseline.py", "--config", str(path), "--candidate", "0",
        "--model", str(tmp_path), "--output", str(output), "--device", "cpu",
    ])
    monkeypatch.setattr(baseline, "prepare_inputs", lambda *args: (None, [], []))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    original = baseline.register_fp32_prefill_flex

    def register():
        events.append("register")
        original()

    def loader(*args, **kwargs):
        events.append(("load", kwargs["attn_implementation"]))
        raise ValueError("model loader boundary")

    monkeypatch.setattr(baseline, "register_fp32_prefill_flex", register)
    monkeypatch.setattr(baseline.AutoModelForCausalLM, "from_pretrained", loader)
    with pytest.raises(RuntimeError, match="baseline incomplete"):
        baseline.main()
    expected = ([] if attention == "sdpa" else ["register"]) + [("load", attention)]
    assert events == expected
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["failure_message"] == "model loader boundary"
