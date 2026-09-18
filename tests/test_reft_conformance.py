import copy
import hashlib
import io
import runpy
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from projection_transport_steering.torch_runtime import TorchLayerAction


@pytest.fixture(scope="module")
def upstream_reft():
    source = Path(__file__).parents[1] / ".external/pyreft/pyreft/interventions.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "ecd9f0cdea62bafea68086598fd4797730a4e86794c82a9aeb0be96479b50de5"
    )
    return runpy.run_path(str(source))["LoreftIntervention"]


@pytest.fixture(params=["qwen3", "llama"])
def model(request):
    torch.manual_seed(20260907)
    config_class, model_class = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
    }[request.param]
    config = config_class(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    return model_class(config).eval().requires_grad_(False)


@pytest.mark.parametrize("rank", [4, 8])
def test_upstream_reft_trains_only_adapter_and_roundtrips_logits(model, upstream_reft, rank):
    tokens = torch.tensor([[1, 5, 8, 3], [1, 7, 4, 2]])
    reference = copy.deepcopy(model.state_dict())
    action = upstream_reft(embed_dim=32, low_rank_dimension=rank, dtype=torch.float32)
    assert sum(parameter.numel() for parameter in action.parameters()) == 65 * rank
    optimizer = torch.optim.AdamW(action.parameters(), lr=0.005)
    with TorchLayerAction(model, 1, action).installed():
        before = model(tokens, labels=tokens, use_cache=False)
        before.loss.backward()
        gradients = [parameter.grad for parameter in action.parameters()]
        assert gradients and all(value is not None for value in gradients)
        assert all(torch.isfinite(value).all() and value.norm() > 0 for value in gradients)
        optimizer.step()
        with torch.no_grad():
            trained_logits = model(tokens, use_cache=False).logits
    assert not torch.equal(before.logits, trained_logits)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in reference.items())
    state = io.BytesIO()
    torch.save(action.state_dict(), state)
    state.seek(0)
    restored = upstream_reft(embed_dim=32, low_rank_dimension=rank, dtype=torch.float32)
    restored.load_state_dict(torch.load(state, weights_only=True))
    with torch.no_grad(), TorchLayerAction(model, 1, restored).installed():
        restored_logits = model(tokens, use_cache=False).logits
    torch.testing.assert_close(trained_logits, restored_logits, rtol=1e-6, atol=1e-7)


def test_zero_action_and_removed_hook_preserve_logits_exactly(model, upstream_reft):
    tokens = torch.tensor([[1, 5, 8, 3], [1, 7, 4, 2]])
    with torch.no_grad():
        reference = model(tokens, use_cache=False).logits
        with TorchLayerAction(model, 1, nn.Identity()).installed():
            zero = model(tokens, use_cache=False).logits
        action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.float32)
        with TorchLayerAction(model, 1, action).installed():
            steered = model(tokens, use_cache=False).logits
        after = model(tokens, use_cache=False).logits
    assert torch.equal(reference, zero)
    assert torch.equal(reference, after)
    assert not torch.equal(reference, steered)


def test_upstream_reft_preserves_bfloat16_output(upstream_reft):
    torch.manual_seed(20260907)
    action = upstream_reft(embed_dim=32, low_rank_dimension=4, dtype=torch.bfloat16)
    hidden = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    result = action(hidden)
    assert result.dtype == hidden.dtype
    assert result.shape == hidden.shape
    assert torch.isfinite(result).all()
