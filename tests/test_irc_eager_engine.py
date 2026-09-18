import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

from projection_transport_steering import attention_backend
from test_irc_real_runtime import packet_config, runtime, tokenizer
from test_reft_conformance import model


@pytest.mark.parametrize("attention", ["fp32_prefill_flex", "fp32_prefill_flex_eager", "sdpa"])
def test_runtime_registers_custom_backend_before_model_load(
    runtime, packet_config, monkeypatch, tmp_path, attention
):
    events = []
    packet_config["attention"] = attention
    path = tmp_path / "config.json"
    path.write_text(json.dumps(packet_config))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "run_irc_real_runtime.py", "--config", str(path), "--condition", "profile",
        "--model", str(tmp_path), "--output", str(output), "--device", "cpu",
    ])
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(
        attention_backend, "register_fp32_prefill_flex", lambda: events.append("register")
    )

    def loader(*args, **kwargs):
        events.append(("load", kwargs["attn_implementation"]))
        raise ValueError("model loader boundary")

    monkeypatch.setattr(runtime["AutoModelForCausalLM"], "from_pretrained", loader)
    with pytest.raises(RuntimeError, match="immutable receipt"):
        runtime["main"]()
    expected = ([] if attention == "sdpa" else ["register"]) + [("load", attention)]
    assert events == expected
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["failure_message"] == "model loader boundary"


def test_prefill_probe_exists():
    assert (Path(__file__).parents[1] / "scripts/run_irc_prefill_probe.py").is_file()


@pytest.fixture
def probe(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return runpy.run_path(str(root / "scripts/run_irc_prefill_probe.py"))


@pytest.fixture
def prefill_inputs(runtime, packet_config, tokenizer):
    rows = runtime["load_packet"](packet_config, Path("/"))
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": packet_config["instruction"] + row["problem"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for row in rows
    ]
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
    expected = {
        "start": 0, "width": inputs.input_ids.shape[1],
        "cluster_ids": [row["cluster_id"] for row in rows],
        "prompt_sha256": [runtime["payload_sha256"](prompt) for prompt in prompts],
        "prompt_tokens": inputs.attention_mask.sum(-1).tolist(),
        **{
            name + "_sha256": runtime["payload_sha256"](inputs[name].tolist())
            for name in ("input_ids", "attention_mask")
        },
    }
    return rows, inputs, expected


def test_prefill_matches_actual_branch_first_forward_without_sampling(
    probe, runtime, packet_config, prefill_inputs, model, tokenizer, monkeypatch, tmp_path
):
    rows, inputs, expected = prefill_inputs
    calls = []
    model.generation_config.eos_token_id = 100
    model.generation_config.pad_token_id = 0

    def capture(module, args, kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("original first-forward boundary")

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    with pytest.raises(RuntimeError, match="original first-forward"):
        runtime["run_batch"](
            model, tokenizer, rows, packet_config, packet_config["conditions"]["profile"],
            (None, None, None), reference_path=tmp_path / "unused",
            hash_payload=runtime["payload_sha256"],
        )

    def forbidden(*args, **kwargs):
        pytest.fail("prefill probe attempted sampling")

    monkeypatch.setattr(torch, "multinomial", forbidden)
    try:
        result = probe["prefill_batch"](model, tokenizer, rows, packet_config, expected)
    finally:
        handle.remove()
    assert len(calls) == 2
    assert set(calls[0]) == set(calls[1])
    for name in calls[0]:
        if isinstance(calls[0][name], torch.Tensor):
            assert torch.equal(calls[0][name], calls[1][name])
        else:
            assert calls[0][name] == calls[1][name]
    assert {name: result[name] for name in expected} == expected
    assert result["logits_shape"] == [2, 1, model.config.vocab_size]
    assert result["cache_length"] == inputs.input_ids.shape[1]
    assert result["finite_logits"] is True
    assert result["base_parameter_versions_unchanged"] is True
    assert not any(name in result for name in ("token_ids", "primary", "scores", "completion"))


@pytest.mark.parametrize("field", [
    "width", "cluster_ids", "prompt_sha256", "prompt_tokens",
    "input_ids_sha256", "attention_mask_sha256",
])
def test_prefill_rejects_input_drift_before_model_forward(
    probe, packet_config, prefill_inputs, model, tokenizer, field
):
    rows, _, expected = prefill_inputs
    expected[field] = "wrong"
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(ValueError, match="prefill input"):
            probe["prefill_batch"](model, tokenizer, rows, packet_config, expected)
    finally:
        handle.remove()
    assert calls == []

@pytest.mark.parametrize("failure", ["nonfinite", "cache", "versions", "shape"])
def test_prefill_rejects_invalid_model_output(
    probe, packet_config, prefill_inputs, model, tokenizer, failure
):
    rows, _, expected = prefill_inputs

    def corrupt(module, args, output):
        if failure == "nonfinite":
            output.logits.fill_(float("nan"))
        elif failure == "cache":
            output.past_key_values = None
        elif failure == "shape":
            output.logits = output.logits[:, :, :1]
        else:
            next(module.parameters()).add_(0)

    handle = model.register_forward_hook(corrupt)
    try:
        with pytest.raises(RuntimeError, match="prefill output"):
            probe["prefill_batch"](model, tokenizer, rows, packet_config, expected)
    finally:
        handle.remove()


def test_prefill_requires_frozen_eval_model(probe, packet_config, prefill_inputs, model, tokenizer):
    rows, _, expected = prefill_inputs
    model.train()
    with pytest.raises(ValueError, match="frozen evaluation"):
        probe["prefill_batch"](model, tokenizer, rows, packet_config, expected)
