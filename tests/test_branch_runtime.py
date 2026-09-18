import copy

import pytest
import torch
from torch import nn

from test_reft_conformance import model

from projection_transport_steering.branch_runtime import advance_branches, start_branches
from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction


SETTINGS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "eos_token_ids": (100,),
    "pad_token_id": 0,
}


@pytest.fixture
def state():
    tokens = torch.tensor([[0, 0, 1, 4], [1, 3, 5, 7]])
    return start_branches(tokens, tokens.ne(0).long(), [7, 13])


def test_branch_parent_and_global_rng_are_unchanged(model, state):
    primed = advance_branches(model, state, steps=3, **SETTINGS)
    snapshot = copy.deepcopy(primed)
    global_rng = torch.get_rng_state().clone()
    first = advance_branches(model, primed, steps=5, **SETTINGS)
    with TorchLayerAction(model, 1, AdditiveAction(torch.arange(32) / 8)).installed():
        advance_branches(model, primed, steps=5, **SETTINGS)
    second = advance_branches(model, primed, steps=5, **SETTINGS)
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(primed.input_ids, snapshot.input_ids)
    for before, after in zip(snapshot.cache.layers, primed.cache.layers, strict=True):
        assert torch.equal(before.keys, after.keys)
        assert torch.equal(before.values, after.values)
    for before, after in zip(snapshot.rng_states, primed.rng_states, strict=True):
        assert torch.equal(before, after)


def test_identity_hook_replays_the_same_engine_exactly(model, state):
    reference = advance_branches(model, state, steps=8, **SETTINGS)
    with TorchLayerAction(model, 1, nn.Identity()).installed():
        identity = advance_branches(model, state, steps=8, **SETTINGS)
    assert torch.equal(reference.input_ids, identity.input_ids)
    assert torch.equal(reference.generated_lengths, identity.generated_lengths)
    assert torch.equal(reference.terminated, identity.terminated)


def test_row_reordering_and_partitioning_keep_each_rows_random_stream(model, state):
    batched = advance_branches(model, state, steps=8, **SETTINGS)
    reversed_state = start_branches(state.input_ids.flip(0), state.attention_mask.flip(0), [13, 7])
    reversed_output = advance_branches(model, reversed_state, steps=8, **SETTINGS)
    assert torch.equal(batched.input_ids, reversed_output.input_ids.flip(0))
    for index, seed in enumerate([7, 13]):
        valid = state.attention_mask[index].bool()
        tokens = state.input_ids[index, valid].unsqueeze(0)
        separate = start_branches(tokens, torch.ones_like(tokens), [seed])
        output = advance_branches(model, separate, steps=8, **SETTINGS)
        count = int(output.generated_lengths[0])
        assert torch.equal(
            output.input_ids[0, tokens.shape[1] :], batched.input_ids[index, 4 : 4 + count]
        )


def test_cached_state_keeps_the_last_sampled_token_pending(model, state):
    output = advance_branches(model, state, steps=3, **SETTINGS)
    assert output.cache.get_seq_length() == output.input_ids.shape[1] - 1
    positions = (output.attention_mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        cached = model(
            output.input_ids[:, -1:],
            attention_mask=output.attention_mask,
            position_ids=positions[:, -1:],
            past_key_values=copy.deepcopy(output.cache),
            use_cache=True,
            logits_to_keep=1,
        ).logits
        uncached = model(
            output.input_ids,
            attention_mask=output.attention_mask,
            position_ids=positions,
            use_cache=False,
            logits_to_keep=1,
        ).logits
    torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-6)


def test_eos_is_absorbing_and_does_not_advance_randomness(model, state):
    settings = {**SETTINGS, "eos_token_ids": tuple(range(101))}
    stopped = advance_branches(model, state, steps=8, **settings)
    assert stopped.generated_lengths.tolist() == [1, 1]
    assert stopped.terminated.tolist() == [True, True]
    again = advance_branches(model, stopped, steps=8, **settings)
    assert torch.equal(again.input_ids, stopped.input_ids)
    for before, after in zip(stopped.rng_states, again.rng_states, strict=True):
        assert torch.equal(before, after)


def test_resuming_preserves_exact_tokens_and_mixed_eos_rows(model, state):
    first = advance_branches(model, state, steps=1, **SETTINGS)
    row_zero_eos = int(first.input_ids[0, -1])
    assert row_zero_eos != int(first.input_ids[1, -1])
    settings = {**SETTINGS, "eos_token_ids": (row_zero_eos,)}
    prefix = advance_branches(model, state, steps=1, **settings)
    resumed = advance_branches(model, prefix, steps=7, **settings)
    uninterrupted = advance_branches(model, state, steps=8, **settings)
    assert torch.equal(resumed.input_ids, uninterrupted.input_ids)
    assert resumed.generated_lengths[0] == 1
    assert resumed.generated_lengths[1] > 1
    assert torch.equal(prefix.rng_states[0], resumed.rng_states[0])


def test_finite_action_window_does_not_extend_into_reference_continuation(model, state):
    prefix = advance_branches(model, state, steps=3, **SETTINGS)
    action = AdditiveAction(torch.arange(32) / 8)
    calls = []
    handle = action.register_forward_hook(
        lambda module, inputs, output: calls.append(inputs[0].shape)
    )
    try:
        with TorchLayerAction(model, 1, action).installed():
            window = advance_branches(model, prefix, steps=3, **SETTINGS)
        advance_branches(model, window, steps=4, **SETTINGS)
    finally:
        handle.remove()
    assert calls == [torch.Size([2, 1, 32])] * 3


@pytest.mark.parametrize(
    ("tokens", "mask", "seeds"),
    [
        ([[1, 2]], [[1, 0]], [1]),
        ([[1, 2]], [[0, 0]], [1]),
        ([[1, 2]], [[1, 1]], []),
        ([[1, 2]], [[1, 2]], [1]),
        ([[1, 2]], [[1]], [1]),
    ],
)
def test_start_rejects_invalid_batch_contract(tokens, mask, seeds):
    with pytest.raises(ValueError):
        start_branches(torch.tensor(tokens), torch.tensor(mask), seeds)


def test_continuation_rejects_training_mode_and_negative_budget(model, state):
    with pytest.raises(ValueError):
        advance_branches(model, state, steps=-1, **SETTINGS)
    with pytest.raises(ValueError):
        advance_branches(model.train(), state, steps=1, **SETTINGS)
