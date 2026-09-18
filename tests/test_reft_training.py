import importlib
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction

pytest_plugins = ("test_reft_conformance",)


@pytest.fixture
def training():
    path = Path(__file__).parents[1] / "src/projection_transport_steering/reft_training.py"
    assert path.exists(), "the explicit target and position integration is missing"
    return importlib.import_module("projection_transport_steering.reft_training")


@pytest.fixture
def native_tokenizer():
    return AutoTokenizer.from_pretrained(
        Path(__file__).parents[1] / "tmp/irc-qwen3-tokenizer-v1", local_files_only=True
    )


def test_native_target_preserves_prefix_source_and_one_terminal_eos(training, native_tokenizer):
    messages = [{"role": "user", "content": "Compute one plus one."}]
    solution = r"Adding one to one gives \boxed{2}."
    example = training.supervised_example(native_tokenizer, messages, solution, "2", 4096)
    prefix = native_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )["input_ids"]
    assert example["input_ids"][: example["prompt_length"]] == prefix
    assert example["labels"][: len(prefix)] == [-100] * len(prefix)
    assert example["labels"][len(prefix) :] == example["input_ids"][len(prefix) :]
    assert example["input_ids"][-1] == native_tokenizer.eos_token_id
    assert example["input_ids"][len(prefix) :].count(native_tokenizer.eos_token_id) == 1
    assert solution in native_tokenizer.decode(example["input_ids"])
    assert example["attention_mask"] == [1] * len(example["input_ids"])
    with pytest.raises(ValueError, match="length"):
        training.supervised_example(native_tokenizer, messages, solution, "2", 10)


def test_native_target_rejects_an_embedded_eos(training, native_tokenizer):
    messages = [{"role": "user", "content": "Compute one plus one."}]
    with pytest.raises(ValueError, match="EOS"):
        training.supervised_example(
            native_tokenizer, messages, "Interrupted <|im_end|> solution", "2", 4096
        )


def test_native_target_rejects_alternative_stop_and_template_source_loss(
    training, native_tokenizer
):
    messages = [{"role": "user", "content": "Compute one plus one."}]
    with pytest.raises(ValueError, match="EOS"):
        training.supervised_example(
            native_tokenizer,
            messages,
            "Interrupted <|endoftext|> solution",
            "2",
            4096,
            eos_token_ids=(151643, 151645),
        )
    with pytest.raises(ValueError, match="source"):
        training.supervised_example(
            native_tokenizer, messages, "\n\nExact solution.\n\n", "2", 4096
        )


@pytest.mark.parametrize("lengths", [torch.tensor([6.5]), torch.tensor([float("nan")])])
def test_position_mask_rejects_noninteger_lengths(training, lengths):
    with pytest.raises(ValueError):
        training.position_mask(torch.ones(1, 8, dtype=torch.long), lengths, "prompt")


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("application", ["prompt", "full"])
def test_positions_follow_unpadded_prompt_and_response(training, side, application):
    valid = torch.tensor([[1] * 20 + [0] * 4, [1] * 24])
    if side == "left":
        valid = valid.flip(-1)
    lengths = torch.tensor([16, 6])
    mask = training.position_mask(valid, lengths, application)
    expected = [list(range(7)) + list(range(9, 16)), list(range(6))]
    if application == "full":
        expected[0] += list(range(16, 20))
        expected[1] += list(range(6, 24))
    for row, positions in enumerate(expected):
        assert torch.where(mask[row][valid[row].bool()])[0].tolist() == positions
    assert not torch.any(mask[~valid.bool()])
    with pytest.raises(ValueError):
        training.position_mask(valid, torch.tensor([25, 6]), application)
    with pytest.raises(ValueError):
        training.position_mask(valid, lengths, "unknown")


def test_masked_action_changes_only_selected_states(training):
    hidden = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False, False, True, True]])
    action = training.MaskedAction(AdditiveAction(torch.ones(3)), mask)
    output = action(hidden)
    assert torch.equal(output[~mask], hidden[~mask])
    assert torch.equal(output[mask], hidden[mask] + 1)
    output.sum().backward()
    assert torch.equal(hidden.grad, torch.ones_like(hidden))
    with pytest.raises(ValueError, match="shape"):
        action(torch.ones(2, 3, 3))


@pytest.mark.parametrize("application", ["prompt", "full"])
def test_accumulated_target_token_loss_matches_combined_gradients(
    training, model, upstream_reft, application
):
    tokens = torch.tensor([[1, 5, 8, 3, 7, 9], [1, 7, 4, 2, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    lengths = torch.tensor([2, 2])
    labels = tokens.clone()
    labels[:, :2] = -100
    labels[~valid.bool()] = -100
    total = int((labels[:, 1:] != -100).sum())
    mask = training.position_mask(valid, lengths, application)
    action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    with TorchLayerAction(model, 1, training.MaskedAction(action, mask)).installed():
        loss = model(
            tokens, attention_mask=valid, labels=labels, use_cache=False, num_items_in_batch=total
        ).loss
        loss.backward()
    expected = [value.grad.clone() for value in action.parameters()]
    action.zero_grad(set_to_none=True)
    for index in range(2):
        with TorchLayerAction(
            model, 1, training.MaskedAction(action, mask[index : index + 1])
        ).installed():
            model(
                tokens[index : index + 1],
                attention_mask=valid[index : index + 1],
                labels=labels[index : index + 1],
                use_cache=False,
                num_items_in_batch=total,
            ).loss.backward()
    for parameter, gradient in zip(action.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-5, atol=1e-6)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_collator_preserves_target_mask_and_prompt_length(training, native_tokenizer):
    examples = [
        training.supervised_example(
            native_tokenizer,
            [{"role": "user", "content": "one " * count}],
            r"The result is \boxed{2}.",
            "2",
            4096,
        )
        for count in (2, 17)
    ]
    batch = DataCollatorForSeq2Seq(native_tokenizer, pad_to_multiple_of=128)(examples)
    assert batch["input_ids"].shape[1] % 128 == 0
    assert batch["prompt_length"].tolist() == [x["prompt_length"] for x in examples]
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert int((batch["labels"] != -100).sum()) == sum(
        len(x["input_ids"]) - x["prompt_length"] for x in examples
    )


@pytest.mark.parametrize("application", ["prompt", "full"])
def test_cached_and_teacher_forced_position_policies_agree(
    training, model, upstream_reft, application
):
    tokens = torch.tensor([[1, 5, 8, 3, 7, 9], [0, 0, 1, 2, 3, 4]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    lengths = torch.tensor([4, 2])
    positions = (valid.cumsum(-1) - 1).clamp_min(0)
    action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    mask = training.position_mask(valid, lengths, application)
    with (
        torch.no_grad(),
        TorchLayerAction(model, 1, training.MaskedAction(action, mask)).installed(),
    ):
        whole = model(tokens, attention_mask=valid, position_ids=positions, use_cache=False)
    with (
        torch.no_grad(),
        TorchLayerAction(model, 1, training.MaskedAction(action, mask[:, :4])).installed(),
    ):
        prefix = model(
            tokens[:, :4],
            attention_mask=valid[:, :4],
            position_ids=positions[:, :4],
            use_cache=True,
        )
    generation_action = action if application == "full" else torch.nn.Identity()
    cached_logits = []
    cache = prefix.past_key_values
    with torch.no_grad(), TorchLayerAction(model, 1, generation_action).installed():
        for index in range(4, 6):
            cached = model(
                tokens[:, index : index + 1],
                attention_mask=valid[:, : index + 1],
                position_ids=positions[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = cached.past_key_values
            cached_logits.append(cached.logits)
    torch.testing.assert_close(
        torch.cat(cached_logits, dim=1), whole.logits[:, 4:], rtol=1e-5, atol=1e-6
    )


def test_fit_step_updates_only_owned_adapter_and_reports_real_gradients(
    training, model, upstream_reft
):
    assert hasattr(training, "fit_step"), "the bounded fit step is missing"
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
        "prompt_length": torch.tensor([2]),
    }
    action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(action.parameters(), lr=0.0009, weight_decay=0)
    result = training.fit_step(model, action, batch, 1, "full", optimizer)
    assert result["target_tokens"] == 2
    assert result["active_positions"] == 4
    assert result["gradient_norm"] > 0
    assert result["changed_parameters"] > 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    bad = torch.optim.AdamW(model.parameters(), lr=0.0009)
    with pytest.raises(ValueError, match="ownership"):
        training.fit_step(model, action, batch, 1, "full", bad)
    for limit in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            training.fit_step(model, action, batch, 1, "full", optimizer, max_grad_norm=limit)


@pytest.mark.parametrize("application", ["prompt", "full"])
@pytest.mark.parametrize("max_grad_norm", [None, 0.05])
@pytest.mark.parametrize("learning_rate", [0.0, 0.0009])
def test_fit_step_accumulates_one_token_normalized_optimizer_update(
    training, model, upstream_reft, application, max_grad_norm, learning_rate
):
    batch = {
        "input_ids": torch.tensor([[1, 5, 8, 3, 7, 9], [1, 7, 4, 2, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, 8, 3, 7, 9], [-100, -100, 4, 2, -100, -100]]),
        "prompt_length": torch.tensor([2, 2]),
    }
    action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    reference = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
    torch.nn.Module.load_state_dict(reference, torch.nn.Module.state_dict(action))
    assert all(
        torch.equal(value, torch.nn.Module.state_dict(reference)[key])
        for key, value in torch.nn.Module.state_dict(action).items()
    )
    optimizer = torch.optim.AdamW(action.parameters(), lr=learning_rate, weight_decay=0)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=learning_rate, weight_decay=0)
    expected = training.fit_step(
        model, reference, batch, 1, application, reference_optimizer, max_grad_norm=max_grad_norm
    )
    microbatches = [{key: value[i : i + 1] for key, value in batch.items()} for i in range(2)]
    result = training.fit_step(
        model, action, microbatches, 1, application, optimizer, max_grad_norm=max_grad_norm
    )
    assert result["target_tokens"] == expected["target_tokens"] == 6
    assert result["active_positions"] == expected["active_positions"]
    assert result["loss"] == pytest.approx(expected["loss"], rel=1e-5, abs=1e-6)
    assert result["gradient_norm"] == pytest.approx(expected["gradient_norm"], rel=1e-5, abs=1e-6)
    assert all(float(state["step"]) == 1 for state in optimizer.state.values())
    assert (result["changed_parameters"] == 0) == (learning_rate == 0)
    for parameter, other in zip(action.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter, other, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg"],
            reference_optimizer.state[other]["exp_avg"],
            rtol=1e-5,
            atol=1e-6,
        )
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="no supervised"):
        training.fit_step(model, action, [], 1, application, optimizer)
