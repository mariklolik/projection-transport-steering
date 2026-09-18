import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from projection_transport_steering.future_pullback import (
    CachePathDownstream,
    GPT2TeacherForcedDownstream,
    full_vocabulary_fisher,
    future_pullback_geometry,
    prepare_gpt2_teacher_forced,
    topk_vocabulary_fisher,
)


def test_full_vocabulary_fisher_matches_explicit_softmax_metric():
    representation = torch.tensor([0.3, -0.2], dtype=torch.float64)
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        dtype=torch.float64,
    )
    logits = unembedding @ representation
    probabilities = torch.softmax(logits, dim=0)
    explicit = unembedding.T @ (
        torch.diag(probabilities) - torch.outer(probabilities, probabilities)
    ) @ unembedding

    result = full_vocabulary_fisher(representation, unembedding)

    assert torch.allclose(result["metric"], explicit)
    assert torch.allclose(result["probabilities"], probabilities)


def test_full_vocabulary_fisher_accumulates_float32_inputs_in_float64():
    representation = torch.tensor([0.3, -0.2], dtype=torch.float32)
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        dtype=torch.float32,
    )

    result = full_vocabulary_fisher(representation, unembedding)

    assert result["metric"].dtype == torch.float64
    assert result["probabilities"].dtype == torch.float64


def test_topk_vocabulary_fisher_matches_the_renormalized_selected_distribution():
    representation = torch.tensor([0.3, -0.2], dtype=torch.float64)
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )
    logits = unembedding @ representation
    selected_logits, selected_ids = logits.topk(2)
    probabilities = torch.softmax(selected_logits, dim=0)
    selected = unembedding[selected_ids]
    expected = selected.T @ (
        torch.diag(probabilities) - torch.outer(probabilities, probabilities)
    ) @ selected

    result = topk_vocabulary_fisher(representation, unembedding, 2)

    assert torch.equal(result["selected_ids"], selected_ids)
    assert torch.allclose(result["metric"], expected)
    assert torch.allclose(result["probabilities"], probabilities)


def test_future_pullback_geometry_uses_every_nested_offset():
    matrices = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[0.5, 0.0], [1.0, 1.0]],
            [[1.0, -1.0], [0.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    bias = torch.tensor([[0.1, 0.2], [0.0, -0.1], [0.3, 0.4]], dtype=torch.float64)
    state = torch.tensor([0.2, -0.3], dtype=torch.float64)
    beta = torch.tensor([0.4, -0.7], dtype=torch.float64)
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        dtype=torch.float64,
    )

    def downstream(value):
        return torch.einsum("toi,i->to", matrices, value) + bias

    result = future_pullback_geometry(downstream, state, beta, unembedding)

    assert torch.allclose(result["covector"], matrices[0].T @ beta)
    assert torch.allclose(result["jacobians"], matrices)
    assert result["terms"].shape == (3, 2, 2)
    for offset, representation in enumerate(downstream(state)):
        fisher = full_vocabulary_fisher(representation, unembedding)["metric"]
        expected = matrices[offset].T @ fisher @ matrices[offset]
        assert torch.allclose(result["terms"][offset], expected)
        assert torch.linalg.eigvalsh(result["terms"][offset]).min() >= -1e-12


def test_future_pullback_geometry_replays_the_downstream_once_for_all_offset_jacobians():
    calls = 0
    matrices = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    state = torch.tensor([0.2, -0.3], dtype=torch.float64)
    beta = torch.tensor([0.4, -0.7], dtype=torch.float64)
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        dtype=torch.float64,
    )

    def downstream(value):
        nonlocal calls
        calls += 1
        return torch.einsum("toi,i->to", matrices, value)

    future_pullback_geometry(downstream, state, beta, unembedding)

    assert calls == 2


def test_future_pullback_geometry_accumulates_float32_jacobians_in_float64():
    matrices = torch.eye(2, dtype=torch.float32).repeat(2, 1, 1)
    state = torch.tensor([0.2, -0.3], dtype=torch.float32)
    beta = torch.tensor([0.4, -0.7], dtype=torch.float32)
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        dtype=torch.float32,
    )

    result = future_pullback_geometry(
        lambda value: torch.einsum("toi,i->to", matrices, value),
        state,
        beta,
        unembedding,
    )

    assert result["terms"].dtype == torch.float64
    assert result["covector"].dtype == torch.float64


def test_future_pullback_geometry_supports_a_named_topk_approximation():
    matrices = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    state = torch.tensor([0.2, -0.3], dtype=torch.float64)
    beta = torch.tensor([0.4, -0.7], dtype=torch.float64)
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )

    result = future_pullback_geometry(
        lambda value: torch.einsum("toi,i->to", matrices, value),
        state,
        beta,
        unembedding,
        top_k=2,
    )

    expected = topk_vocabulary_fisher(state, unembedding, 2)["metric"]
    assert torch.allclose(result["terms"][0], expected)
    assert result["top_k"] == 2


def test_cache_path_downstream_partitions_immediate_and_future_jacobians():
    matrices = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[0.5, 0.0], [1.0, 1.0]],
            [[1.0, -1.0], [0.0, 0.5]],
        ]
    )
    base_state = torch.tensor([0.2, -0.3])

    def downstream(value):
        return torch.einsum("toi,i->to", matrices, value)

    reset = CachePathDownstream(downstream, base_state, "reset")
    cache_only = CachePathDownstream(downstream, base_state, "cache_only")
    reset_jacobian = torch.autograd.functional.jacobian(reset, base_state)
    cache_only_jacobian = torch.autograd.functional.jacobian(cache_only, base_state)

    assert torch.allclose(reset_jacobian[0], matrices[0])
    assert torch.count_nonzero(reset_jacobian[1:]) == 0
    assert torch.count_nonzero(cache_only_jacobian[0]) == 0
    assert torch.allclose(cache_only_jacobian[1:], matrices[1:])
    assert torch.allclose(reset_jacobian + cache_only_jacobian, matrices)


def test_gpt2_teacher_forced_downstream_replays_all_future_positions():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=3,
            n_positions=16,
            vocab_size=31,
        )
    ).eval()
    input_ids = torch.tensor([[3, 5, 7, 11, 13]])

    prepared = prepare_gpt2_teacher_forced(model, input_ids, layer=0, position=2)
    replayed = model.lm_head(prepared["downstream"](prepared["state"]))

    assert torch.allclose(replayed, prepared["base_logits"], atol=1e-6, rtol=1e-5)
    jacobian = torch.autograd.functional.jacobian(
        prepared["downstream"],
        prepared["state"],
        vectorize=True,
    )
    assert jacobian.shape == (3, 12, 12)


def test_gpt2_incremental_cache_replay_matches_full_downstream_after_intervention():
    torch.manual_seed(19)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=3,
            n_positions=16,
            vocab_size=31,
            bos_token_id=30,
            eos_token_id=30,
            _attn_implementation="eager",
        )
    ).eval()
    prepared = prepare_gpt2_teacher_forced(
        model,
        torch.tensor([[3, 5, 7, 11, 13]]),
        layer=0,
        position=2,
    )
    steered_state = prepared["state"] + torch.linspace(-0.01, 0.01, 12)

    full = prepared["downstream"](steered_state)
    incremental = prepared["incremental_downstream"](steered_state)

    assert torch.allclose(incremental, full, atol=1e-6, rtol=1e-5)


def test_gpt2_teacher_forced_downstream_batches_common_prefix_trajectories():
    torch.manual_seed(17)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=3,
            n_positions=16,
            vocab_size=31,
            bos_token_id=30,
            eos_token_id=30,
        )
    ).eval()
    input_ids = torch.tensor([[3, 5, 7, 11, 13], [3, 5, 7, 17, 19]])

    prepared = prepare_gpt2_teacher_forced(model, input_ids, layer=0, position=2)
    replayed = model.lm_head(prepared["downstream"](prepared["state"]))

    assert prepared["trajectory_count"] == 2
    assert prepared["offset_count"] == 3
    assert replayed.shape == (6, 31)
    assert torch.allclose(replayed, prepared["base_logits"], atol=1e-6, rtol=1e-5)


def test_gpt2_teacher_forced_downstream_preserves_the_causal_mask():
    torch.manual_seed(11)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=3,
            n_positions=16,
            vocab_size=31,
            _attn_implementation="eager",
        )
    ).eval()
    observed = []

    def record(_module, inputs, kwargs):
        mask = inputs[2] if len(inputs) > 2 else kwargs.get("attention_mask")
        observed.append(mask is not None)

    handle = model.transformer.h[2].register_forward_pre_hook(record, with_kwargs=True)
    try:
        prepare_gpt2_teacher_forced(
            model,
            torch.tensor([[3, 5, 7, 11, 13]]),
            layer=0,
            position=2,
        )
    finally:
        handle.remove()

    assert len(observed) == 2
    assert observed[0] == observed[1]


def test_gpt2_teacher_forced_downstream_forwards_an_explicit_causal_mask():
    class RecordingBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.observed = None

        def forward(
            self,
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            use_cache=False,
        ):
            self.observed = attention_mask
            return hidden_states

    class Transformer(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.h = nn.ModuleList([nn.Identity(), block])
            self.ln_f = nn.Identity()

    class Model(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.transformer = Transformer(block)

    block = RecordingBlock()
    model = Model(block)
    hidden_states = torch.randn(1, 3, 2)
    causal_mask = torch.triu(torch.full((1, 1, 3, 3), float("-inf")), diagonal=1)
    downstream = GPT2TeacherForcedDownstream(model, 0, hidden_states, 1, causal_mask)

    downstream(hidden_states[0, 1])

    assert torch.equal(block.observed, causal_mask)
