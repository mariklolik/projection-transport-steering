import copy

import pytest
import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import flex_attention_mask

from projection_transport_steering import attention_backend


@pytest.mark.parametrize("length", [1, 2, 113, 127, 128, 129])
@pytest.mark.parametrize("options", [None, {"num_stages": 3, "FORCE_USE_FLEX_ATTENTION": False}])
def test_short_prefill_dispatch_preserves_native_contract(monkeypatch, length, options):
    backend = getattr(attention_backend, "fp32_prefill_flex_v4", None)
    assert callable(backend), "the separately named short-prefill backend is missing"
    original = copy.deepcopy(options)
    tensors = [
        torch.randn(2, heads, length, 4, dtype=torch.bfloat16, requires_grad=True)
        for heads in (4, 1, 1)
    ]
    mask = torch.ones(2, 1, length, length, dtype=torch.bool).tril()
    observed = []
    owner = object()

    def native(module, query, key, value, attention_mask, **kwargs):
        observed.append((module, query.dtype, key.dtype, value.dtype, attention_mask, kwargs))
        output = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), attention_mask, enable_gqa=True
        ).transpose(1, 2)
        return output, output.sum(-1)

    monkeypatch.setattr(attention_backend, "flex_attention_forward", native)
    output, lse = backend(owner, *tensors, mask, kernel_options=options, scaling=0.5)
    expected_options = original
    if 1 < length < 128:
        expected_options = {**(original or {}), "FORCE_USE_FLEX_ATTENTION": True}
    assert options == original
    assert observed[0][0] is owner and observed[0][4] is mask
    dtype = torch.float32 if length > 1 else torch.bfloat16
    assert observed[0][1:4] == (dtype, dtype, dtype)
    assert observed[0][5] == {"kernel_options": expected_options, "scaling": 0.5}
    assert output.dtype == lse.dtype == torch.bfloat16
    assert output.shape == (2, length, 4, 4)
    output.float().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)


def test_short_prefill_backend_uses_existing_registration():
    previous = torch.get_float32_matmul_precision()
    try:
        attention_backend.register_fp32_prefill_flex()
        assert "fp32_prefill_flex_v4" in AttentionInterface()
        assert (
            AttentionInterface()["fp32_prefill_flex_v4"] is attention_backend.fp32_prefill_flex_v4
        )
        assert AttentionMaskInterface()["fp32_prefill_flex_v4"] is flex_attention_mask
        assert AttentionInterface()["fp32_prefill_flex"] is attention_backend.fp32_prefill_flex
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous)
