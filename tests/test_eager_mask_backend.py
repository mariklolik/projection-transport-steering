import inspect
from pathlib import Path

import pytest
import torch
from transformers import AttentionInterface, AttentionMaskInterface, Qwen3Config, masking_utils
from transformers.cache_utils import DynamicCache

from projection_transport_steering import attention_backend


def test_eager_mask_binding_exists():
    assert callable(getattr(attention_backend, "eager_flex_attention_mask", None))


@pytest.fixture
def eager_mask():
    factory = getattr(attention_backend, "eager_flex_attention_mask", None)
    assert callable(factory), "isolated native eager mask binding is missing"
    return factory


def test_isolated_mask_preserves_native_code_and_all_other_globals(eager_mask):
    native = masking_utils.flex_attention_mask
    assert eager_mask.__code__ is native.__code__
    assert eager_mask.__defaults__ == native.__defaults__
    assert eager_mask.__kwdefaults__ == native.__kwdefaults__
    assert eager_mask.__closure__ is native.__closure__
    assert inspect.signature(eager_mask) == inspect.signature(native)
    assert not hasattr(eager_mask, "__wrapped__")
    assert eager_mask.__globals__ is not native.__globals__
    assert set(eager_mask.__globals__) == set(native.__globals__)
    changed = [
        name
        for name in native.__globals__
        if eager_mask.__globals__[name] is not native.__globals__[name]
    ]
    assert changed == ["create_block_mask"]


@pytest.mark.parametrize("failing", [False, True])
def test_native_builder_delegation_changes_only_compile_and_never_shared_state(
    eager_mask, monkeypatch, failing
):
    before = dict(masking_utils.flex_attention_mask.__globals__)
    calls = []
    returned = object()

    def builder(**kwargs):
        calls.append(kwargs)
        if failing:
            raise RuntimeError("native builder failure")
        return returned

    monkeypatch.setattr(attention_backend, "create_block_mask", builder)
    options = dict(batch_size=2, q_length=3, kv_length=7, q_offset=4, device="cpu")
    if failing:
        with pytest.raises(RuntimeError, match="native builder failure"):
            eager_mask(**options)
    else:
        assert eager_mask(**options) is returned
    assert len(calls) == 1
    kwargs = calls[0]
    assert {k: v for k, v in kwargs.items() if k != "mask_mod"} == {
        "B": 2, "H": None, "Q_LEN": 3, "KV_LEN": 7, "device": "cpu", "_compile": False
    }
    q = torch.arange(3)[:, None]
    k = torch.arange(7)[None, :]
    assert torch.equal(kwargs["mask_mod"](0, 0, q, k), q + 4 >= k)
    assert all(masking_utils.flex_attention_mask.__globals__[k] is v for k, v in before.items())


@pytest.mark.parametrize("compile_value", [False, None])
def test_isolated_builder_rejects_unexpected_compile_request(eager_mask, compile_value):
    with pytest.raises(ValueError, match="compile"):
        eager_mask.__globals__["create_block_mask"](_compile=compile_value)


@pytest.mark.parametrize("shape", [(3, 3, 0, 0), (1, 7, 6, 0), (3, 7, 4, 0), (2, 4, 6, 3)])
@pytest.mark.parametrize("padding", [False, True])
def test_real_eager_mask_equals_native_eager_padding_offsets(
    eager_mask, monkeypatch, shape, padding
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_mask_diagnostic import block_metadata, mask_construction

    q, kv, q_offset, kv_offset = shape
    options = dict(
        batch_size=2, q_length=q, kv_length=kv, q_offset=q_offset,
        kv_offset=kv_offset, device="cpu",
    )
    if padding:
        mask = torch.ones(2, kv + kv_offset, dtype=torch.bool)
        mask[0] = False
        mask[1, :2] = False
        options["attention_mask"] = mask
    actual = eager_mask(**options)
    with mask_construction("eager"):
        expected = masking_utils.flex_attention_mask(**options)
    assert block_metadata(actual) == block_metadata(expected)
    b = torch.arange(2)[:, None, None]
    i = torch.arange(q)[None, :, None]
    j = torch.arange(kv)[None, None, :]
    assert torch.equal(actual.mask_mod(b, 0, i, j), expected.mask_mod(b, 0, i, j))


def test_new_registration_is_idempotent_and_old_aliases_stay_native(eager_mask):
    before = torch.get_float32_matmul_precision()
    try:
        attention_backend.register_fp32_prefill_flex()
        attention_backend.register_fp32_prefill_flex()
        assert AttentionInterface()["fp32_prefill_flex_eager"] is attention_backend.fp32_prefill_flex_v4
        assert AttentionMaskInterface()["fp32_prefill_flex_eager"] is eager_mask
        assert AttentionMaskInterface()["fp32_prefill_flex"] is masking_utils.flex_attention_mask
        assert AttentionMaskInterface()["fp32_prefill_flex_v4"] is masking_utils.flex_attention_mask
    finally:
        torch.set_float32_matmul_precision(before)


@pytest.mark.parametrize("past", [0, 6])
def test_actual_causal_mask_registry_path_and_cache_offset(eager_mask, past):
    before = torch.get_float32_matmul_precision()
    try:
        attention_backend.register_fp32_prefill_flex()
        config = Qwen3Config(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        )
        config._attn_implementation = "fp32_prefill_flex_eager"
        cache = DynamicCache(config=config)
        if past:
            cache.update(torch.zeros(2, 2, past, 4), torch.zeros(2, 2, past, 4), 0)
        q = 1 if past else 3
        padding = torch.ones(2, past + q, dtype=torch.long)
        padding[0, :2] = 0
        block = masking_utils.create_causal_mask(
            config, torch.zeros(2, q, 16), padding, cache if past else None
        )
        b = torch.arange(2)[:, None, None]
        i = torch.arange(q)[None, :, None]
        j = torch.arange(past + q)[None, None, :]
        expected = (i + past >= j) & padding[:, None].bool()
        assert torch.equal(block.mask_mod(b, 0, i, j), expected)
        for supplied in (block, torch.ones(2, 1, q, past + q, dtype=torch.bool)):
            assert masking_utils.create_causal_mask(
                config, torch.zeros(2, q, 16), supplied, cache if past else None
            ) is supplied
    finally:
        torch.set_float32_matmul_precision(before)
