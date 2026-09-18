from types import SimpleNamespace

import pytest
import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import flex_attention_mask

from projection_transport_steering import attention_backend


@pytest.mark.parametrize("length", [1, 129])
@pytest.mark.parametrize("with_lse", [False, True])
def test_precision_adapter_preserves_mask_dtype_and_gradients(monkeypatch, length, with_lse):
    tensors = [
        torch.randn(2, heads, length, 4, dtype=torch.bfloat16, requires_grad=True)
        for heads in (2, 1, 1)
    ]
    mask = torch.ones(2, 1, length, length, dtype=torch.bool).tril()
    observed = []
    module = object()

    def native(owner, query, key, value, attention_mask, **kwargs):
        observed.append((owner, query.dtype, key.dtype, value.dtype, attention_mask, kwargs))
        output = (
            torch.nn.functional.scaled_dot_product_attention(
                query.float(), key.float(), value.float(), attention_mask, enable_gqa=True
            )
            .transpose(1, 2)
            .to(query.dtype)
        )
        return output, output.sum(-1) if with_lse else None

    monkeypatch.setattr(attention_backend, "flex_attention_forward", native)
    output, lse = attention_backend.fp32_prefill_flex(
        module, *tensors, mask, scaling=0.5, dropout=0.0
    )
    dtype = torch.float32 if length > 1 else torch.bfloat16
    assert observed[0][:4] == (module, dtype, dtype, dtype)
    assert observed[0][4] is mask
    assert observed[0][5] == {"scaling": 0.5, "dropout": 0.0}
    assert output.shape == (2, length, 2, 4)
    assert output.dtype == torch.bfloat16
    assert (lse.dtype == torch.bfloat16) if with_lse else lse is None
    output.float().sum().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in tensors)


def test_registration_reuses_native_mask_and_requires_ieee():
    previous = torch.get_float32_matmul_precision()
    try:
        attention_backend.register_fp32_prefill_flex()
        assert AttentionInterface()["fp32_prefill_flex"] is attention_backend.fp32_prefill_flex
        assert AttentionMaskInterface()["fp32_prefill_flex"] is flex_attention_mask
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the qualified CUDA runtime")
def test_native_fp32_prefill_backward_matches_grouped_fp64(
    length=129,
    query_heads=4,
    kv_heads=2,
    training=True,
    batch_size=2,
    backend_name="fp32_prefill_flex",
    check_no_grad=False,
):
    previous = torch.get_float32_matmul_precision()
    try:
        attention_backend.register_fp32_prefill_flex()
        torch.manual_seed(20260908)
        groups = query_heads // kv_heads
        query, key, value = [
            torch.randn(
                batch_size, heads, length, 128, device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            for heads in (query_heads, kv_heads, kv_heads)
        ]
        q64, k64, v64 = [x.detach().double().requires_grad_() for x in (query, key, value)]
        mask = torch.ones(batch_size, length, dtype=torch.bool, device="cuda")
        mask[1, 0] = False
        mask[0, -3:] = False
        ids = torch.arange(length, device="cuda")
        keep = (ids[None, :, None] >= ids[None, None, :]) & mask[:, None, :]
        valid = keep.any(-1)
        scores = torch.einsum(
            "bghqd,bgkd->bghqk", q64.reshape(batch_size, kv_heads, groups, length, 128), k64
        )
        scores = (scores * 128**-0.5).masked_fill(~keep[:, None, None], -torch.inf)
        scores = torch.where(valid[:, None, None, :, None], scores, torch.zeros_like(scores))
        weights = torch.where(
            valid[:, None, None, :, None], scores.softmax(-1), torch.zeros_like(scores)
        )
        oracle = torch.einsum("bghqk,bgkd->bghqd", weights, v64)
        oracle = oracle.reshape(batch_size, query_heads, length, 128).transpose(1, 2)
        block = flex_attention_mask(batch_size, length, length, attention_mask=mask, device="cuda")
        predicate = block.mask_mod(
            torch.arange(batch_size, device="cuda")[:, None, None],
            0,
            ids[None, :, None],
            ids[None, None, :],
        )
        assert torch.equal(predicate.expand_as(keep), keep)
        module = SimpleNamespace(num_key_value_groups=groups, is_causal=True, training=training)
        output, _ = getattr(attention_backend, backend_name)(
            module, query, key, value, block, scaling=128**-0.5, dropout=0.0
        )
        torch.testing.assert_close(output.double(), oracle, rtol=0.005, atol=0.0005)
        assert torch.all(output[~valid] == 0)
        if check_no_grad:
            with torch.no_grad():
                inference_output, _ = getattr(attention_backend, backend_name)(
                    module,
                    query.detach(),
                    key.detach(),
                    value.detach(),
                    block,
                    scaling=128**-0.5,
                    dropout=0.0,
                )
            torch.testing.assert_close(inference_output.double(), oracle, rtol=0.005, atol=0.0005)
            assert torch.all(inference_output[~valid] == 0)
        gradient = torch.randn_like(output)
        output.backward(gradient)
        oracle.backward(gradient.double())
        for tensor, reference in zip((query, key, value), (q64, k64, v64), strict=True):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
            torch.testing.assert_close(
                tensor.grad.double(), reference.grad, rtol=0.005, atol=0.0005
            )
    finally:
        torch.set_float32_matmul_precision(previous)
