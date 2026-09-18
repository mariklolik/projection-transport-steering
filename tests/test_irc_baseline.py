import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from projection_transport_steering.branch_runtime import advance_branches, start_branches
from projection_transport_steering import reft_training as training

pytest_plugins = ("test_reft_training", "test_irc_real_runtime")


@pytest.fixture
def baseline():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_irc_baseline")


def test_epoch_groups_preserve_random_logical_membership_and_all_rows():
    examples = [{"input_ids": [1] * (i % 19 + 1)} for i in range(1500)]
    seed = 20260907
    for epoch in (0, 1, 11):
        actual = training.epoch_batches(examples, seed, epoch, 32)
        order = torch.randperm(1500, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        assert len(actual) == 47 and len(actual[-1]) == 28
        assert sorted(i for batch in actual for i in batch) == list(range(1500))
        for start, batch in zip(range(0, 1500, 32), actual, strict=True):
            assert set(batch) == set(order[start : start + 32])
            assert batch == sorted(batch, key=lambda i: (-len(examples[i]["input_ids"]), i))
        assert actual == training.epoch_batches(examples, seed, epoch, 32)


@pytest.mark.parametrize("application", ["prompt", "full"])
def test_generation_context_matches_explicit_masks_and_raw_checkpoint(
    model, upstream_reft, tmp_path, application
):
    tokens = torch.tensor([[1, 5, 8, 3], [0, 0, 1, 2]])
    valid = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    positions = (valid.cumsum(-1) - 1).clamp_min(0)
    action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    checkpoint = tmp_path / "adapter.pt"
    torch.save(torch.nn.Module.state_dict(action), checkpoint)
    restored = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    torch.nn.Module.load_state_dict(restored, torch.load(checkpoint, weights_only=True))
    state = start_branches(tokens, valid, [42, 43])
    settings = dict(
        steps=3, temperature=0.6, top_p=0.95, top_k=20, eos_token_ids=(34,), pad_token_id=0
    )
    outputs = []
    for current in (action, restored):
        with training.generation_intervention(model, current, 1, application):
            outputs.append(advance_branches(model, state, **settings))
    assert torch.equal(outputs[0].input_ids, outputs[1].input_ids)
    assert not model._forward_pre_hooks and not model.model.layers[1]._forward_hooks
    mask = training.position_mask(valid, valid.sum(-1), application)
    with (
        torch.no_grad(),
        training.TorchLayerAction(model, 1, training.MaskedAction(action, mask)).installed(),
    ):
        expected = model(tokens, attention_mask=valid, position_ids=positions, use_cache=True)
    with torch.no_grad(), training.generation_intervention(model, action, 1, application):
        actual = model(
            input_ids=tokens, attention_mask=valid, position_ids=positions, use_cache=True
        )
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        extended = torch.cat((valid, torch.ones(2, 1, dtype=torch.long)), -1)
        next_ids = torch.tensor([[5], [7]])
        next_positions = (extended.cumsum(-1) - 1)[:, -1:]
        actual_next = model(
            input_ids=next_ids,
            attention_mask=extended,
            position_ids=next_positions,
            past_key_values=actual.past_key_values,
        )
    decode = action if application == "full" else torch.nn.Identity()
    with torch.no_grad(), training.TorchLayerAction(model, 1, decode).installed():
        expected_next = model(
            next_ids,
            attention_mask=extended,
            position_ids=next_positions,
            past_key_values=expected.past_key_values,
        )
    torch.testing.assert_close(actual_next.logits, expected_next.logits, rtol=0, atol=0)
    with (
        pytest.raises(RuntimeError),
        training.generation_intervention(model, action, 1, application),
    ):
        raise RuntimeError("intentional")
    assert not model._forward_pre_hooks and not model.model.layers[1]._forward_hooks


def test_complete_epoch_loop_handles_partial_batch_scheduler_and_final_checkpoint(
    baseline, model, upstream_reft, native_tokenizer, tmp_path
):
    examples = [
        {
            "input_ids": [1, 2, 3, 4 + i, 7][: 4 + i % 2],
            "attention_mask": [1] * (4 + i % 2),
            "labels": [-100, -100, 3, 4 + i, 7][: 4 + i % 2],
            "prompt_length": 2,
        }
        for i in range(5)
    ]
    native_tokenizer.pad_token_id = 0
    recipe = dict(
        epochs=2,
        batch_size=4,
        microbatch_size=2,
        padding_buckets=[8, 16],
        learning_rate=0.0009,
        warmup_ratio=0.1,
        max_grad_norm=1,
        seed=20260907,
    )
    candidate = dict(rank=4, site=1, application="full")
    action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = baseline.train_adapter(
        model, action, examples, native_tokenizer, recipe, candidate, tmp_path
    )
    rows = [json.loads(line) for line in (tmp_path / "fit.jsonl").read_text().splitlines()]
    assert result["optimizer_updates"] == len(rows) == 4
    assert result["target_tokens"] == 2 * sum(len(x["input_ids"]) - 2 for x in examples)
    assert [row["learning_rate"] for row in rows] == pytest.approx([0, 0.0009, 0.0006, 0.0003])
    assert [row["examples"] for row in rows] == [4, 1, 4, 1]
    assert rows[0]["changed_parameters"] == 0
    assert all(row["changed_parameters"] > 0 for row in rows[1:])
    assert result["checkpoint_sha256"] == baseline.file_sha256(tmp_path / "adapter.pt")
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    raw = torch.load(tmp_path / "adapter.pt", weights_only=True)
    assert set(raw) == set(torch.nn.Module.state_dict(action))
    assert all(
        torch.equal(value.cpu(), raw[key])
        for key, value in torch.nn.Module.state_dict(action).items()
    )
    assert math.isfinite(result["seconds"])
    with pytest.raises(FileExistsError):
        baseline.train_adapter(
            model, action, examples, native_tokenizer, recipe, candidate, tmp_path
        )


@pytest.mark.parametrize("application", ["prompt", "full"])
def test_full_cli_keeps_complete_zero_fit_and_selection(
    baseline, runtime, packet_config, model, tokenizer, tmp_path, application
):
    root = Path(__file__).parents[1]
    path = Path(packet_config["packet"])
    packet = json.loads(path.read_text())
    packet.pop("content_sha256")
    packet["allocation"] = "fit"
    packet["content_sha256"] = baseline.payload_sha256(packet)
    path.write_text(json.dumps(packet))
    selected = {
        "allocation": "selection",
        "rows": [
            {**row, "cluster_id": f"selection-{i}", "allocation": "selection"}
            for i, row in enumerate(packet["rows"])
        ],
    }
    selected["content_sha256"] = baseline.payload_sha256(selected)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selected))
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }} answer "
        "{% else %}{{ m.reasoning_content }} {{ m.content }}{{ eos_token }}\n"
        "{% endif %}{% endfor %}"
    )
    model_path = tmp_path / "model"
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config = {
        **packet_config,
        "packet_sha256": baseline.file_sha256(path),
        "groups": {"fit": packet_config["cluster_ids"]},
        "maximum_length": 128,
        "attention": "eager",
        "generation_batch": 2,
        "generation_steps": 4,
        "selection": {
            "packet": str(selection_path),
            "num_questions": 2,
            "packet_sha256": baseline.file_sha256(selection_path),
        },
        "recipe": dict(
            epochs=2,
            batch_size=2,
            microbatch_size=1,
            padding_buckets=[64],
            learning_rate=0.0009,
            warmup_ratio=0.1,
            max_grad_norm=1,
            seed=20260907,
        ),
        "candidates": [dict(rank=4, site=1, application=application)],
        "zero_shares": [[0, 1]],
    }
    config["target_examples_sha256"] = baseline.payload_sha256(
        baseline.load_examples(config, root, tokenizer)["fit"]
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "full-run"
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        str(root / "scripts/run_irc_baseline.py"),
        "--config",
        str(config_path),
        "--model",
        str(model_path),
        "--output",
        str(output),
        "--candidate",
        "0",
        "--device",
        "cpu",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "pass" and not receipt["positive_action_admitted"]
    assert receipt["base_parameter_versions_unchanged"]
    assert receipt["fit"]["optimizer_updates"] == 2
    for condition in ("zero", "selection"):
        assert receipt[condition]["completed_ids"] == ["selection-0", "selection-1"]
        for name, digest in receipt[condition]["chunk_sha256"].items():
            assert baseline.file_sha256(output / condition / name) == digest
    original = baseline.file_sha256(output / "receipt.json")
    again = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert again.returncode != 0 and baseline.file_sha256(output / "receipt.json") == original
    with pytest.raises(ValueError, match="zero"):
        baseline.prepare_inputs({**config, "zero_shares": [[0, 0]]}, root, model_path)
